//! The language prompt projector of a multilingual streaming checkpoint, such as
//! Nemotron 3.5 ASR Streaming
//!
//! NeMo joins each encoder frame with a one-hot prompt of the stream's language and
//! projects it back to `d_model` through `Linear, ReLU, Linear` before the joint.
//! The one-hot only selects one column of the first weight, so that column folds
//! into a per-prompt bias, and each stream in a batch can have its own language

use cudakit::cudarc::driver::CudaSlice;
use cudakit::{DeviceTensor, Math};

use crate::cuda::kernels::{CudaRuntime, EncoderKernels, StreamKernels};
use crate::cuda::{linear, upload, zeros};
use crate::error::TranscriptionError;
use crate::native::spec::PromptSpec;
use crate::native::weights::WeightFile;

pub(super) struct PromptProjector {
    /// `[hidden, d_model]`, the first layer's columns for the encoder output
    weight_in: DeviceTensor,
    /// `[prompts, hidden]`, each prompt's column of the first layer plus its bias
    table: DeviceTensor,
    weight_out: DeviceTensor,
    bias_out: DeviceTensor,
    d_model: usize,
    hidden: usize,
    prompts: usize,
    encoder_kernels: EncoderKernels,
    stream_kernels: StreamKernels,
}

/// Device buffers for one batch shape
pub(super) struct PromptBuffers {
    /// `[batch]` each row's prompt
    prompts: CudaSlice<u32>,
    hidden: CudaSlice<f32>,
    /// `[batch * frames, d_model]`, the projected encoder output
    out: CudaSlice<f32>,
    frames: usize,
}

impl PromptBuffers {
    /// The projected encoder output, `[batch * frames, d_model]`
    pub(super) fn output(&self) -> &CudaSlice<f32> {
        &self.out
    }
}

impl PromptProjector {
    pub(super) fn load(
        runtime: &CudaRuntime,
        file: &WeightFile,
        spec: &PromptSpec,
        d_model: usize,
    ) -> Result<Self, TranscriptionError> {
        let name_in = "prompt_kernel.0.weight";
        let inputs = d_model + spec.prompts;
        let hidden = match file.shape(name_in) {
            Some([hidden, width]) if *width == inputs => *hidden,
            shape => {
                return Err(TranscriptionError::InvalidWeights(format!(
                    "`{name_in}` has shape {shape:?}, not [hidden, {inputs}]"
                )));
            }
        };
        let weight = file.tensor(name_in, &[hidden, inputs])?;
        let bias = file.tensor("prompt_kernel.0.bias", &[hidden])?;

        let mut weight_in = Vec::with_capacity(hidden * d_model);
        for row in weight.chunks_exact(inputs) {
            weight_in.extend_from_slice(&row[..d_model]);
        }
        let mut table = Vec::with_capacity(spec.prompts * hidden);
        for prompt in 0..spec.prompts {
            table.extend(
                (0..hidden).map(|unit| bias[unit] + weight[unit * inputs + d_model + prompt]),
            );
        }

        Ok(Self {
            weight_in: DeviceTensor::upload(runtime.stream(), &weight_in, &[hidden, d_model])?,
            table: DeviceTensor::upload(runtime.stream(), &table, &[spec.prompts, hidden])?,
            weight_out: upload(runtime, file, "prompt_kernel.2.weight", &[d_model, hidden])?,
            bias_out: upload(runtime, file, "prompt_kernel.2.bias", &[d_model])?,
            d_model,
            hidden,
            prompts: spec.prompts,
            encoder_kernels: EncoderKernels::load(runtime)?,
            stream_kernels: StreamKernels::load(runtime)?,
        })
    }

    pub(super) fn buffers(
        &self,
        runtime: &CudaRuntime,
        batch: usize,
        frames: usize,
    ) -> Result<PromptBuffers, TranscriptionError> {
        let stream = runtime.stream();
        Ok(PromptBuffers {
            prompts: zeros(stream, batch)?,
            hidden: zeros(stream, batch * frames * self.hidden)?,
            out: zeros(stream, batch * frames * self.d_model)?,
            frames,
        })
    }

    /// Uploads each row's prompt for the next [`Self::enqueue`]
    pub(super) fn set_prompts(
        &self,
        runtime: &CudaRuntime,
        buffers: &mut PromptBuffers,
        prompts: &[u32],
    ) -> Result<(), TranscriptionError> {
        if prompts.len() != buffers.prompts.len()
            || prompts
                .iter()
                .any(|prompt| *prompt as usize >= self.prompts)
        {
            return Err(TranscriptionError::InvalidConfiguration(format!(
                "prompts {prompts:?} for {} rows of {} prompts",
                buffers.prompts.len(),
                self.prompts
            )));
        }
        runtime
            .stream()
            .memcpy_htod(prompts, &mut buffers.prompts)?;
        Ok(())
    }

    /// Queues the projection of `encoder_out` (`[batch * frames, d_model]`) into
    /// the buffers' output; capturable in a CUDA graph
    pub(super) fn enqueue(
        &self,
        runtime: &CudaRuntime,
        buffers: &mut PromptBuffers,
        encoder_out: &CudaSlice<f32>,
    ) -> Result<(), TranscriptionError> {
        let stream = runtime.stream();
        let rows = buffers.prompts.len() * buffers.frames;
        runtime.sgemm(
            linear(rows, self.hidden, self.d_model, 1.0, 0.0, Math::Fp32),
            encoder_out,
            self.weight_in.data(),
            &mut buffers.hidden,
        )?;
        self.stream_kernels.prompt(
            stream,
            &mut buffers.hidden.slice_mut(..),
            &self.table.data().slice(..),
            &buffers.prompts.slice(..),
            buffers.frames,
        )?;
        self.encoder_kernels.fill_rows(
            stream,
            &self.bias_out.data().slice(..),
            &mut buffers.out.slice_mut(..),
        )?;
        runtime.sgemm(
            linear(rows, self.d_model, self.hidden, 1.0, 1.0, Math::Fp32),
            &buffers.hidden,
            self.weight_out.data(),
            &mut buffers.out,
        )?;
        Ok(())
    }
}
