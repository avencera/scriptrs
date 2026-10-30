//! FastConformer encoder on cuBLAS plus the `encoder` kernel area
//!
//! The forward pass follows NeMo's `ConformerEncoder` operation by operation in FP32,
//! so each stage can be held against the reference activations. Everything a pass
//! touches lives in an [`EncoderWorkspace`] allocated once per input shape, which
//! keeps [`CudaEncoder::enqueue`] free of allocations and host syncs and therefore
//! capturable in a CUDA graph.
//!
//! [`EncoderMath::Fp16`] stores the Conformer layers' linear weights in FP16 and has
//! the kernels before each of those GEMMs write an FP16 copy of their output, so the
//! GEMMs read half the bytes and run at the FP16 tensor-core rate. The residual
//! stream, the GEMM outputs and their accumulation stay FP32
//!
//! With the `low-precision` feature, `EncoderMath::Quantized` keeps the FP16
//! activations and quantizes the same linear layers further (see the `quantized`
//! module)

#[cfg(feature = "low-precision")]
mod quantized;

use cudakit::cudarc::driver::{CudaSlice, CudaView, DevicePtrMut};
use cudakit::{DeviceTensor, Math, Sgemm, SgemmBatched};

use crate::cuda::fp16::linear_fp16;
use crate::cuda::kernels::{CudaRuntime, EncoderKernels, FeatureMap, GemmOperand, Heads};
#[cfg(feature = "low-precision")]
use crate::cuda::quant::Scheme;
use crate::cuda::{Direction, HostStaging, linear, upload, upload_host, zeros};
use crate::error::TranscriptionError;
use crate::native::spec::EncoderSpec;
use crate::native::weights::WeightFile;

/// PyTorch's default `eps` for `LayerNorm` and `BatchNorm1d`, which NeMo keeps
const NORM_EPS: f32 = 1e-5;

/// Every `dw_striding` convolution is 3x3 with stride 2 and padding 1
fn subsampled(len: usize) -> usize {
    // NeMo's `calculate_conv_output_size`: (len + 1 + 1 - 3) // 2 + 1
    (len + 2).saturating_sub(3) / 2 + 1
}

/// Precision of the encoder's GEMMs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EncoderMath {
    /// Every GEMM on FP32 operands in this cuBLAS math
    Single(Math),
    /// The Conformer layers' linear layers on FP16 operands with FP32 output and
    /// accumulation. The attention score and mix products and the subsampling module
    /// keep FP32 operands in TF32: they are a few percent of the work, and their
    /// operands would need conversions of their own
    Fp16,
    /// [`Self::Fp16`] with the linear layers' weights, and for every scheme but INT8
    /// their activations, quantized in the scheme
    #[cfg(feature = "low-precision")]
    Quantized(Scheme),
}

impl EncoderMath {
    /// The math of the GEMMs that keep FP32 operands
    fn single(self) -> Math {
        match self {
            Self::Single(math) => math,
            Self::Fp16 => Math::Tf32,
            #[cfg(feature = "low-precision")]
            Self::Quantized(_) => Math::Tf32,
        }
    }

    /// Whether the kernels write FP16 copies of the linear layers' inputs
    fn fp16(self) -> bool {
        !matches!(self, Self::Single(_))
    }
}

/// A linear layer's `[out, in]` weight, in the precision its GEMM reads
enum Linear {
    Single(DeviceTensor),
    /// FP16 bits
    Fp16(CudaSlice<u16>),
    #[cfg(feature = "low-precision")]
    Quantized(quantized::QuantLinear),
}

/// The activation a linear layer reads, matching its [`Linear`] weight
enum Operand<'a> {
    Single(CudaView<'a, f32>),
    Fp16(CudaView<'a, u16>),
}

struct Norm {
    weight: DeviceTensor,
    bias: DeviceTensor,
}

struct FeedForward {
    linear1: Linear,
    linear2: Linear,
}

struct Attention {
    /// `linear_q`, `linear_k` and `linear_v` stacked to `[3 * d_model, d_model]`
    qkv: Linear,
    pos: Linear,
    out: Linear,
    bias_u: DeviceTensor,
    bias_v: DeviceTensor,
}

struct ConvModule {
    pointwise1: Linear,
    depthwise: DeviceTensor,
    /// Eval-mode BatchNorm folded to `x * scale + shift`
    scale: DeviceTensor,
    shift: DeviceTensor,
    pointwise2: Linear,
}

struct Layer {
    norm_ff1: Norm,
    ff1: FeedForward,
    norm_att: Norm,
    attention: Attention,
    norm_conv: Norm,
    conv: ConvModule,
    norm_ff2: Norm,
    ff2: FeedForward,
    norm_out: Norm,
}

struct Subsampling {
    conv_in: (DeviceTensor, DeviceTensor),
    depthwise1: (DeviceTensor, DeviceTensor),
    pointwise1: (DeviceTensor, DeviceTensor),
    depthwise2: (DeviceTensor, DeviceTensor),
    pointwise2: (DeviceTensor, DeviceTensor),
    out: (DeviceTensor, DeviceTensor),
}

/// Encoder weights on the GPU and the kernels that run them
pub(crate) struct CudaEncoder {
    spec: EncoderSpec,
    math: EncoderMath,
    subsampling: Subsampling,
    layers: Vec<Layer>,
    kernels: EncoderKernels,
    #[cfg(feature = "low-precision")]
    quant: Option<quantized::QuantState>,
}

/// Device buffers for one `(batch, mel frames)` input shape
pub(crate) struct EncoderWorkspace {
    batch: usize,
    mel_frames: usize,
    /// Encoder frames after 8x subsampling
    frames: usize,
    /// `[batch, mel_frames, features]`
    mel: CudaSlice<f32>,
    /// Pinned host copy of `mel` that [`CudaEncoder::set_input`] fills
    mel_host: HostStaging<f32>,
    /// Valid frames per batch row before and after each subsampling stage
    lens: [CudaSlice<u32>; 4],
    host_lens: Vec<usize>,
    /// Subsampling feature maps, ping-ponged: `[0]` fits the first stage, `[1]` the
    /// second
    maps: [CudaSlice<f32>; 2],
    flat: CudaSlice<f32>,
    /// The residual stream, `[batch * frames, d_model]`
    x: CudaSlice<f32>,
    normed: CudaSlice<f32>,
    hidden: CudaSlice<f32>,
    qkv: CudaSlice<f32>,
    q_u: CudaSlice<f32>,
    q_v: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    pos_emb: CudaSlice<f32>,
    pos: CudaSlice<f32>,
    pos_heads: CudaSlice<f32>,
    scores: CudaSlice<f32>,
    bd: CudaSlice<f32>,
    context: CudaSlice<f32>,
    merged: CudaSlice<f32>,
    /// FP16 GEMM inputs under [`EncoderMath::Fp16`]
    fp16: Option<Fp16Inputs>,
    quant: QuantScratch,
}

/// The buffers the quantized linear layers quantize their activations into, empty
/// without the `low-precision` feature
struct QuantScratch {
    #[cfg(feature = "low-precision")]
    inputs: Option<quantized::QuantInputs>,
}

impl QuantScratch {
    fn bytes(&self) -> usize {
        #[cfg(feature = "low-precision")]
        if let Some(inputs) = &self.inputs {
            return inputs.bytes();
        }
        0
    }
}

/// FP16 copies of the activations the FP16 linear layers read
struct Fp16Inputs {
    /// `[batch * frames, d_model]` LayerNorm output
    normed: CudaSlice<u16>,
    /// `[batch * frames, ff_hidden]` SiLU output
    hidden: CudaSlice<u16>,
    /// `[batch * frames, d_model]` attention context or conv module output
    merged: CudaSlice<u16>,
    /// `[2 * frames - 1, d_model]` relative position table
    pos_emb: CudaSlice<u16>,
}

impl EncoderWorkspace {
    /// Mel frames per batch row of the input
    pub(crate) fn mel_frames(&self) -> usize {
        self.mel_frames
    }

    pub(crate) fn frames(&self) -> usize {
        self.frames
    }

    /// Valid encoder frames per batch row of the current input
    pub(crate) fn output_lens(&self) -> &[usize] {
        &self.host_lens
    }

    /// The encoder output, `[batch * frames, d_model]`
    pub(crate) fn output(&self) -> &CudaSlice<f32> {
        &self.x
    }

    /// Device bytes held by the workspace's buffers
    pub(crate) fn bytes(&self) -> usize {
        let lens: usize = self.lens.iter().map(CudaSlice::num_bytes).sum();
        let buffers = [
            &self.mel,
            &self.maps[0],
            &self.maps[1],
            &self.flat,
            &self.x,
            &self.normed,
            &self.hidden,
            &self.qkv,
            &self.q_u,
            &self.q_v,
            &self.k,
            &self.v,
            &self.pos_emb,
            &self.pos,
            &self.pos_heads,
            &self.scores,
            &self.bd,
            &self.context,
            &self.merged,
        ];
        let fp16 = self.fp16.as_ref().map_or(0, |fp16| {
            [&fp16.normed, &fp16.hidden, &fp16.merged, &fp16.pos_emb]
                .iter()
                .map(|buffer| buffer.num_bytes())
                .sum()
        });
        lens + fp16
            + self.quant.bytes()
            + buffers
                .iter()
                .map(|buffer| buffer.num_bytes())
                .sum::<usize>()
    }
}

impl CudaEncoder {
    /// Uploads the encoder weights and loads the kernels; the GEMMs run at `math`
    pub(crate) fn load(
        runtime: &CudaRuntime,
        file: &WeightFile,
        spec: EncoderSpec,
        math: EncoderMath,
    ) -> Result<Self, TranscriptionError> {
        let kernels = EncoderKernels::load(runtime)?;
        let d = spec.d_model;
        let channels = spec.subsampling_channels;
        let flat = channels * spec.features / spec.subsampling_factor;
        let pair = |name: &str, shape: &[usize]| -> Result<_, TranscriptionError> {
            Ok((
                upload(
                    runtime,
                    file,
                    &format!("encoder.pre_encode.{name}.weight"),
                    shape,
                )?,
                upload(
                    runtime,
                    file,
                    &format!("encoder.pre_encode.{name}.bias"),
                    &shape[..1],
                )?,
            ))
        };
        let subsampling = Subsampling {
            conv_in: pair("conv.0", &[channels, 1, 3, 3])?,
            depthwise1: pair("conv.2", &[channels, 1, 3, 3])?,
            pointwise1: pair("conv.3", &[channels, channels, 1, 1])?,
            depthwise2: pair("conv.5", &[channels, 1, 3, 3])?,
            pointwise2: pair("conv.6", &[channels, channels, 1, 1])?,
            out: pair("out", &[d, flat])?,
        };

        #[cfg(feature = "low-precision")]
        let quant = quantized::QuantState::load(runtime, file, spec, math)?;
        let mut layers = Vec::with_capacity(spec.layers);
        for index in 0..spec.layers {
            layers.push(Self::load_layer(
                runtime, &kernels, file, spec, math, index,
            )?);
        }

        Ok(Self {
            spec,
            math,
            subsampling,
            layers,
            kernels,
            #[cfg(feature = "low-precision")]
            quant,
        })
    }

    fn load_layer(
        runtime: &CudaRuntime,
        kernels: &EncoderKernels,
        file: &WeightFile,
        spec: EncoderSpec,
        math: EncoderMath,
        index: usize,
    ) -> Result<Layer, TranscriptionError> {
        let d = spec.d_model;
        let prefix = format!("encoder.layers.{index}");
        let tensor = |name: &str, shape: &[usize]| file.tensor(&format!("{prefix}.{name}"), shape);
        let load =
            |name: &str, shape: &[usize]| upload(runtime, file, &format!("{prefix}.{name}"), shape);
        #[cfg(not(feature = "low-precision"))]
        let to_linear = |weight: DeviceTensor| Linear::new(runtime, kernels, weight, math);
        #[cfg(not(feature = "low-precision"))]
        let load_linear = |name: &str, shape: &[usize]| to_linear(load(name, shape)?);
        // the linear weights may be stored FP16 or quantized
        #[cfg(feature = "low-precision")]
        let load_linear = |name: &str, shape: &[usize]| {
            let name = format!("{prefix}.{name}");
            quantized::load_linear(runtime, kernels, file, &name, shape, math)
        };
        let norm = |name: &str| -> Result<Norm, TranscriptionError> {
            Ok(Norm {
                weight: load(&format!("{name}.weight"), &[d])?,
                bias: load(&format!("{name}.bias"), &[d])?,
            })
        };
        let feed_forward = |name: &str| -> Result<FeedForward, TranscriptionError> {
            Ok(FeedForward {
                linear1: load_linear(&format!("{name}.linear1.weight"), &[spec.ff_hidden, d])?,
                linear2: load_linear(&format!("{name}.linear2.weight"), &[d, spec.ff_hidden])?,
            })
        };

        #[cfg(not(feature = "low-precision"))]
        let mut qkv = Vec::with_capacity(3 * d * d);
        #[cfg(not(feature = "low-precision"))]
        for name in ["linear_q", "linear_k", "linear_v"] {
            qkv.extend_from_slice(tensor(&format!("self_attn.{name}.weight"), &[d, d])?);
        }
        let head_dim = d / spec.heads;
        let attention = Attention {
            #[cfg(not(feature = "low-precision"))]
            qkv: to_linear(DeviceTensor::upload(runtime.stream(), &qkv, &[3 * d, d])?)?,
            // the reader stacks the three weights of a file that keeps them apart
            #[cfg(feature = "low-precision")]
            qkv: load_linear("self_attn.linear_qkv.weight", &[3 * d, d])?,
            pos: load_linear("self_attn.linear_pos.weight", &[d, d])?,
            out: load_linear("self_attn.linear_out.weight", &[d, d])?,
            bias_u: load("self_attn.pos_bias_u", &[spec.heads, head_dim])?,
            bias_v: load("self_attn.pos_bias_v", &[spec.heads, head_dim])?,
        };

        // fold in f64 so the folded constants round once
        let bn = |name: &str| tensor(&format!("conv.batch_norm.{name}"), &[d]);
        let (gamma, beta, mean, var) = (
            bn("weight")?,
            bn("bias")?,
            bn("running_mean")?,
            bn("running_var")?,
        );
        let mut scale = Vec::with_capacity(d);
        let mut shift = Vec::with_capacity(d);
        for c in 0..d {
            let s = f64::from(gamma[c]) / (f64::from(var[c]) + f64::from(NORM_EPS)).sqrt();
            scale.push(s as f32);
            shift.push((f64::from(beta[c]) - f64::from(mean[c]) * s) as f32);
        }
        let conv = ConvModule {
            pointwise1: load_linear("conv.pointwise_conv1.weight", &[2 * d, d, 1])?,
            depthwise: load("conv.depthwise_conv.weight", &[d, 1, spec.conv_kernel])?,
            scale: upload_host(runtime, &scale)?,
            shift: upload_host(runtime, &shift)?,
            pointwise2: load_linear("conv.pointwise_conv2.weight", &[d, d, 1])?,
        };

        Ok(Layer {
            norm_ff1: norm("norm_feed_forward1")?,
            ff1: feed_forward("feed_forward1")?,
            norm_att: norm("norm_self_att")?,
            attention,
            norm_conv: norm("norm_conv")?,
            conv,
            norm_ff2: norm("norm_feed_forward2")?,
            ff2: feed_forward("feed_forward2")?,
            norm_out: norm("norm_out")?,
        })
    }

    /// Allocates the buffers for `batch` inputs of up to `mel_frames` frames, and the
    /// relative position table for that length
    pub(crate) fn workspace(
        &self,
        runtime: &CudaRuntime,
        batch: usize,
        mel_frames: usize,
    ) -> Result<EncoderWorkspace, TranscriptionError> {
        let stream = runtime.stream();
        let spec = self.spec;
        let (d, heads) = (spec.d_model, spec.heads);
        let channels = spec.subsampling_channels;
        let maps = self.feature_maps(batch, mel_frames);
        let frames = maps[3].frames;
        let rows = batch * frames;
        let positions = 2 * frames - 1;
        let pos_emb = stream.clone_htod(&relative_positions(frames, d))?;
        let fp16 = match self.math {
            EncoderMath::Single(_) => None,
            // the quantized maths also read FP16 activations
            _ => {
                let mut table = zeros(stream, pos_emb.len())?;
                self.kernels
                    .to_f16(stream, &pos_emb.slice(..), &mut table.slice_mut(..))?;
                Some(Fp16Inputs {
                    normed: zeros(stream, rows * d)?,
                    hidden: zeros(stream, rows * spec.ff_hidden)?,
                    merged: zeros(stream, rows * d)?,
                    pos_emb: table,
                })
            }
        };

        Ok(EncoderWorkspace {
            batch,
            mel_frames,
            frames,
            mel: zeros(stream, batch * mel_frames * spec.features)?,
            mel_host: HostStaging::new(
                stream,
                batch * mel_frames * spec.features,
                Direction::Upload,
            )?,
            lens: [
                zeros(stream, batch)?,
                zeros(stream, batch)?,
                zeros(stream, batch)?,
                zeros(stream, batch)?,
            ],
            host_lens: vec![0; batch],
            maps: [zeros(stream, maps[1].len())?, zeros(stream, maps[2].len())?],
            flat: zeros(stream, rows * channels * maps[3].features)?,
            x: zeros(stream, rows * d)?,
            normed: zeros(stream, rows * d)?,
            hidden: zeros(stream, rows * spec.ff_hidden.max(2 * d))?,
            qkv: zeros(stream, rows * 3 * d)?,
            q_u: zeros(stream, rows * d)?,
            q_v: zeros(stream, rows * d)?,
            k: zeros(stream, rows * d)?,
            v: zeros(stream, rows * d)?,
            pos_emb,
            pos: zeros(stream, positions * d)?,
            pos_heads: zeros(stream, positions * d)?,
            scores: zeros(stream, batch * heads * frames * frames)?,
            bd: zeros(stream, rows * heads * positions)?,
            context: zeros(stream, rows * d)?,
            merged: zeros(stream, rows * d)?,
            fp16,
            quant: QuantScratch {
                #[cfg(feature = "low-precision")]
                inputs: match &self.quant {
                    Some(quant) => quant.inputs(runtime, spec, rows, positions)?,
                    None => None,
                },
            },
        })
    }

    /// Input and subsampled feature maps: `[0]` is the mel, `[3]` the last stage
    fn feature_maps(&self, batch: usize, mel_frames: usize) -> [FeatureMap; 4] {
        let channels = self.spec.subsampling_channels;
        let mel = FeatureMap {
            batch,
            channels: 1,
            frames: mel_frames,
            features: self.spec.features,
        };
        let next = |map: FeatureMap| FeatureMap {
            batch,
            channels,
            frames: subsampled(map.frames),
            features: subsampled(map.features),
        };
        let first = next(mel);
        let second = next(first);
        [mel, first, second, next(second)]
    }

    /// Queues the upload of a padded batch: `fill` writes every value of the
    /// `[batch, mel_frames, features]` (time-major) input, and `lens` holds the valid
    /// mel frames of each row
    ///
    /// `fill` writes straight into pinned memory, so the upload queues behind earlier
    /// work instead of blocking the host
    pub(crate) fn set_input(
        &self,
        runtime: &CudaRuntime,
        workspace: &mut EncoderWorkspace,
        lens: &[usize],
        fill: impl FnOnce(&mut [f32]),
    ) -> Result<(), TranscriptionError> {
        if lens.len() != workspace.batch {
            return Err(TranscriptionError::InvalidConfiguration(format!(
                "encoder input has {} rows; the workspace takes {}",
                lens.len(),
                workspace.batch
            )));
        }
        if let Some(len) = lens
            .iter()
            .find(|len| **len == 0 || **len > workspace.mel_frames)
        {
            return Err(TranscriptionError::InvalidConfiguration(format!(
                "encoder input row has {len} valid frames of {}",
                workspace.mel_frames
            )));
        }

        let stream = runtime.stream();
        fill(workspace.mel_host.write()?);
        workspace.mel_host.upload(stream, &mut workspace.mel)?;
        let mut stage: Vec<usize> = lens.to_vec();
        for buffer in &mut workspace.lens {
            let values: Vec<u32> = stage.iter().map(|len| *len as u32).collect();
            stream.memcpy_htod(&values, buffer)?;
            stage = stage.into_iter().map(subsampled).collect();
        }
        // the loop subsampled once more than there are stages
        workspace.host_lens = lens
            .iter()
            .map(|len| subsampled(subsampled(subsampled(*len))))
            .collect();
        Ok(())
    }

    /// Queues the whole encoder on the runtime's stream
    pub(crate) fn enqueue(
        &self,
        runtime: &CudaRuntime,
        workspace: &mut EncoderWorkspace,
    ) -> Result<(), TranscriptionError> {
        self.enqueue_subsampling(runtime, workspace)?;
        for index in 0..self.layers.len() {
            self.enqueue_layer(runtime, workspace, index)?;
        }
        Ok(())
    }

    /// `pre_encode`: three stride-2 convolutions, then the projection to `d_model`
    pub(crate) fn enqueue_subsampling(
        &self,
        runtime: &CudaRuntime,
        ws: &mut EncoderWorkspace,
    ) -> Result<(), TranscriptionError> {
        let stream = runtime.stream();
        let kernels = &self.kernels;
        let sub = &self.subsampling;
        let maps = self.feature_maps(ws.batch, ws.mel_frames);
        let [mel_map, first, second, third] = maps;
        let channels = self.spec.subsampling_channels;
        let [large, small] = &mut ws.maps;

        kernels.subsample_conv_in(
            stream,
            &ws.mel.slice(..),
            &ws.lens[0].slice(..),
            &ws.lens[1].slice(..),
            &sub.conv_in.0.data().slice(..),
            &sub.conv_in.1.data().slice(..),
            &mut large.slice_mut(..),
            mel_map,
            first,
        )?;
        kernels.subsample_depthwise(
            stream,
            &large.slice(..),
            &ws.lens[2].slice(..),
            &sub.depthwise1.0.data().slice(..),
            &sub.depthwise1.1.data().slice(..),
            &mut small.slice_mut(..),
            first,
            second,
        )?;
        self.pointwise(runtime, &sub.pointwise1, small, large, second, &ws.lens[2])?;
        kernels.subsample_depthwise(
            stream,
            &large.slice(..second.len()),
            &ws.lens[3].slice(..),
            &sub.depthwise2.0.data().slice(..),
            &sub.depthwise2.1.data().slice(..),
            &mut small.slice_mut(..third.len()),
            second,
            third,
        )?;
        self.pointwise(runtime, &sub.pointwise2, small, large, third, &ws.lens[3])?;
        kernels.subsample_flatten(
            stream,
            &large.slice(..third.len()),
            &mut ws.flat.slice_mut(..),
            third,
        )?;

        let rows = ws.batch * ws.frames;
        let (weight, bias) = &sub.out;
        kernels.fill_rows(stream, &bias.data().slice(..), &mut ws.x.slice_mut(..))?;
        runtime.sgemm(
            linear(
                rows,
                self.spec.d_model,
                channels * third.features,
                1.0,
                1.0,
                self.math.single(),
            ),
            &ws.flat,
            weight.data(),
            &mut ws.x,
        )?;
        Ok(())
    }

    /// A pointwise subsampling convolution, `[channels, channels]` over each row's
    /// `[channels, frames * features]`, then bias, ReLU and the time mask
    fn pointwise(
        &self,
        runtime: &CudaRuntime,
        (weight, bias): &(DeviceTensor, DeviceTensor),
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        map: FeatureMap,
        lens: &CudaSlice<u32>,
    ) -> Result<(), TranscriptionError> {
        let plane = map.frames * map.features;
        let per_row = map.channels * plane;
        let gemm = Sgemm {
            math: self.math.single(),
            ..Sgemm::new(map.channels, plane, map.channels)
        };
        runtime.sgemm_batched(
            SgemmBatched {
                gemm,
                batch: map.batch,
                stride_a: 0,
                stride_b: per_row,
                stride_c: per_row,
            },
            weight.data(),
            &input.slice(..map.len()),
            &mut output.slice_mut(..map.len()),
        )?;
        self.kernels.subsample_bias_relu(
            runtime.stream(),
            &mut output.slice_mut(..map.len()),
            &bias.data().slice(..),
            &lens.slice(..),
            map,
        )?;
        Ok(())
    }

    /// One Conformer layer on the residual stream
    ///
    /// Layers run in order after [`Self::enqueue_subsampling`]: a layer's closing
    /// norm also writes the next layer's first norm to `normed` (or its FP16 copy),
    /// so every layer but the first starts from it
    pub(crate) fn enqueue_layer(
        &self,
        runtime: &CudaRuntime,
        ws: &mut EncoderWorkspace,
        index: usize,
    ) -> Result<(), TranscriptionError> {
        let layer = &self.layers[index];
        if index == 0 {
            self.norm(runtime, ws, &layer.norm_ff1)?;
        }
        self.feed_forward(runtime, ws, &layer.ff1)?;
        self.norm(runtime, ws, &layer.norm_att)?;
        self.attention(runtime, ws, &layer.attention)?;
        self.norm(runtime, ws, &layer.norm_conv)?;
        self.conv_module(runtime, ws, &layer.conv)?;
        self.norm(runtime, ws, &layer.norm_ff2)?;
        self.feed_forward(runtime, ws, &layer.ff2)?;

        // the final norm is not residual, so it runs in place, and the next layer's
        // first norm reads its result before it leaves the registers
        let next = self.layers.get(index + 1).map(|next| {
            (
                next.norm_ff1.weight.data().slice(..),
                next.norm_ff1.bias.data().slice(..),
            )
        });
        self.kernels.layer_norm_pair(
            runtime.stream(),
            &mut ws.x.slice_mut(..),
            (
                &layer.norm_out.weight.data().slice(..),
                &layer.norm_out.bias.data().slice(..),
            ),
            next.as_ref().map(|(gamma, beta)| (gamma, beta)),
            self.gemm_output(
                &mut ws.normed,
                ws.fp16.as_mut().map(|fp16| &mut fp16.normed),
            )?,
            self.spec.d_model,
            NORM_EPS,
        )?;
        Ok(())
    }

    /// LayerNorm of the residual stream for the next linear layer: into `ws.normed`,
    /// or its FP16 copy under [`EncoderMath::Fp16`]
    fn norm(
        &self,
        runtime: &CudaRuntime,
        ws: &mut EncoderWorkspace,
        norm: &Norm,
    ) -> Result<(), TranscriptionError> {
        let out = self.gemm_output(
            &mut ws.normed,
            ws.fp16.as_mut().map(|fp16| &mut fp16.normed),
        )?;
        self.kernels.layer_norm(
            runtime.stream(),
            &ws.x.slice(..),
            &norm.weight.data().slice(..),
            &norm.bias.data().slice(..),
            out,
            self.spec.d_model,
            NORM_EPS,
        )?;
        Ok(())
    }

    /// Where a kernel writes the activation a linear layer reads next: `single`, or
    /// `fp16`, its FP16 copy, under [`EncoderMath::Fp16`]
    fn gemm_output<'a>(
        &self,
        single: &'a mut CudaSlice<f32>,
        fp16: Option<&'a mut CudaSlice<u16>>,
    ) -> Result<GemmOperand<'a>, TranscriptionError> {
        match (self.math.fp16(), fp16) {
            (false, _) => Ok(GemmOperand::Single(single.slice_mut(..))),
            (true, Some(fp16)) => Ok(GemmOperand::Fp16(fp16.slice_mut(..))),
            (true, None) => Err(mixed_precision()),
        }
    }

    /// The activation a linear layer reads: `single`, or `fp16`, its FP16 copy, under
    /// [`EncoderMath::Fp16`]
    fn gemm_input<'a>(
        &self,
        single: CudaView<'a, f32>,
        fp16: Option<CudaView<'a, u16>>,
    ) -> Result<Operand<'a>, TranscriptionError> {
        match (self.math.fp16(), fp16) {
            (false, _) => Ok(Operand::Single(single)),
            (true, Some(fp16)) => Ok(Operand::Fp16(fp16)),
            (true, None) => Err(mixed_precision()),
        }
    }

    /// Runs `spec`, a [`linear`] product, on `a` and `weight` in their shared precision;
    /// a quantized product quantizes `a` into `quant` first
    fn linear<C: DevicePtrMut<f32>>(
        &self,
        runtime: &CudaRuntime,
        spec: Sgemm,
        a: Operand<'_>,
        weight: &Linear,
        c: &mut C,
        #[cfg_attr(not(feature = "low-precision"), allow(unused_variables))]
        quant: &mut QuantScratch,
    ) -> Result<(), TranscriptionError> {
        match (a, weight, self.math) {
            (Operand::Single(a), Linear::Single(w), EncoderMath::Single(_)) => {
                runtime.sgemm(spec, &a, w.data(), c)?;
            }
            (Operand::Fp16(a), Linear::Fp16(w), EncoderMath::Fp16) => {
                linear_fp16(runtime, spec, &a, w, c)?;
            }
            #[cfg(feature = "low-precision")]
            (Operand::Fp16(a), Linear::Quantized(w), EncoderMath::Quantized(_)) => {
                let state = self.quant.as_ref().ok_or_else(mixed_precision)?;
                state.linear(runtime, spec, &a, w, c, quant.inputs.as_mut())?;
            }
            _ => return Err(mixed_precision()),
        }
        Ok(())
    }

    /// `x += 0.5 * linear2(silu(linear1(normed)))`, with `normed` the sublayer's
    /// norm of `x`
    fn feed_forward(
        &self,
        runtime: &CudaRuntime,
        ws: &mut EncoderWorkspace,
        weights: &FeedForward,
    ) -> Result<(), TranscriptionError> {
        let (d, hidden) = (self.spec.d_model, self.spec.ff_hidden);
        let math = self.math.single();
        let rows = ws.batch * ws.frames;
        let normed = self.gemm_input(
            ws.normed.slice(..),
            ws.fp16.as_ref().map(|fp16| fp16.normed.slice(..)),
        )?;
        self.linear(
            runtime,
            linear(rows, hidden, d, 1.0, 0.0, math),
            normed,
            &weights.linear1,
            &mut ws.hidden.slice_mut(..rows * hidden),
            &mut ws.quant,
        )?;
        let stream = runtime.stream();
        match ws.fp16.as_mut() {
            Some(fp16) => self.kernels.silu_f16(
                stream,
                &ws.hidden.slice(..rows * hidden),
                &mut fp16.hidden.slice_mut(..),
            )?,
            None => self
                .kernels
                .silu(stream, &mut ws.hidden.slice_mut(..rows * hidden))?,
        }
        let activated = self.gemm_input(
            ws.hidden.slice(..rows * hidden),
            ws.fp16.as_ref().map(|fp16| fp16.hidden.slice(..)),
        )?;
        // NeMo scales the feed-forward residual by 0.5 (`fc_factor`)
        self.linear(
            runtime,
            linear(rows, d, hidden, 0.5, 1.0, math),
            activated,
            &weights.linear2,
            &mut ws.x,
            &mut ws.quant,
        )?;
        Ok(())
    }

    /// `x += linear_out(rel_pos_attention(normed))`, with `normed` the sublayer's
    /// norm of `x`
    fn attention(
        &self,
        runtime: &CudaRuntime,
        ws: &mut EncoderWorkspace,
        weights: &Attention,
    ) -> Result<(), TranscriptionError> {
        let stream = runtime.stream();
        let spec = self.spec;
        let (d, heads, head_dim) = (spec.d_model, spec.heads, spec.d_model / spec.heads);
        let (batch, frames) = (ws.batch, ws.frames);
        let rows = batch * frames;
        let positions = 2 * frames - 1;
        let shape = Heads {
            batch,
            frames,
            heads,
            head_dim,
        };

        let math = self.math.single();
        let normed = self.gemm_input(
            ws.normed.slice(..),
            ws.fp16.as_ref().map(|fp16| fp16.normed.slice(..)),
        )?;
        self.linear(
            runtime,
            linear(rows, 3 * d, d, 1.0, 0.0, math),
            normed,
            &weights.qkv,
            &mut ws.qkv,
            &mut ws.quant,
        )?;
        self.kernels.split_qkv(
            stream,
            &ws.qkv.slice(..),
            &weights.bias_u.data().slice(..),
            &weights.bias_v.data().slice(..),
            &mut ws.q_u.slice_mut(..),
            &mut ws.q_v.slice_mut(..),
            &mut ws.k.slice_mut(..),
            &mut ws.v.slice_mut(..),
            shape,
        )?;
        let table = self.gemm_input(
            ws.pos_emb.slice(..),
            ws.fp16.as_ref().map(|fp16| fp16.pos_emb.slice(..)),
        )?;
        self.linear(
            runtime,
            linear(positions, d, d, 1.0, 0.0, math),
            table,
            &weights.pos,
            &mut ws.pos,
            &mut ws.quant,
        )?;
        self.kernels.split_heads(
            stream,
            &ws.pos.slice(..),
            &mut ws.pos_heads.slice_mut(..),
            positions,
            heads,
            head_dim,
        )?;

        // content scores: (q + u) · kᵀ per (batch, head)
        let content = Sgemm {
            b_transposed: true,
            math,
            ..Sgemm::new(frames, frames, head_dim)
        };
        runtime.sgemm_batched(
            SgemmBatched::packed(content, batch * heads),
            &ws.q_u,
            &ws.k,
            &mut ws.scores,
        )?;
        // position scores: (q + v) · pᵀ per head, every utterance against one table
        let position = Sgemm {
            b_transposed: true,
            math,
            ..Sgemm::new(rows, positions, head_dim)
        };
        runtime.sgemm_batched(
            SgemmBatched::packed(position, heads),
            &ws.q_v,
            &ws.pos_heads,
            &mut ws.bd,
        )?;
        self.kernels.rel_pos_softmax(
            stream,
            &mut ws.scores.slice_mut(..),
            &ws.bd.slice(..),
            &ws.lens[3].slice(..),
            shape,
        )?;

        let mix = Sgemm {
            math,
            ..Sgemm::new(frames, head_dim, frames)
        };
        runtime.sgemm_batched(
            SgemmBatched::packed(mix, batch * heads),
            &ws.scores,
            &ws.v,
            &mut ws.context,
        )?;
        let merged = self.gemm_output(
            &mut ws.merged,
            ws.fp16.as_mut().map(|fp16| &mut fp16.merged),
        )?;
        self.kernels
            .merge_heads(stream, &ws.context.slice(..), merged, shape)?;
        let merged = self.gemm_input(
            ws.merged.slice(..),
            ws.fp16.as_ref().map(|fp16| fp16.merged.slice(..)),
        )?;
        self.linear(
            runtime,
            linear(rows, d, d, 1.0, 1.0, math),
            merged,
            &weights.out,
            &mut ws.x,
            &mut ws.quant,
        )?;
        Ok(())
    }

    /// `x += pointwise2(silu(bn(depthwise(mask(glu(pointwise1(normed)))))))`, with
    /// `normed` the sublayer's norm of `x`
    fn conv_module(
        &self,
        runtime: &CudaRuntime,
        ws: &mut EncoderWorkspace,
        weights: &ConvModule,
    ) -> Result<(), TranscriptionError> {
        let d = self.spec.d_model;
        let math = self.math.single();
        let rows = ws.batch * ws.frames;
        let pw1_len = rows * 2 * d;
        let normed = self.gemm_input(
            ws.normed.slice(..),
            ws.fp16.as_ref().map(|fp16| fp16.normed.slice(..)),
        )?;
        self.linear(
            runtime,
            linear(rows, 2 * d, d, 1.0, 0.0, math),
            normed,
            &weights.pointwise1,
            &mut ws.hidden.slice_mut(..pw1_len),
            &mut ws.quant,
        )?;
        let glued = self.gemm_output(
            &mut ws.merged,
            ws.fp16.as_mut().map(|fp16| &mut fp16.merged),
        )?;
        self.kernels.conv_glue(
            runtime.stream(),
            &ws.hidden.slice(..pw1_len),
            &ws.lens[3].slice(..),
            &weights.depthwise.data().slice(..),
            &weights.scale.data().slice(..),
            &weights.shift.data().slice(..),
            glued,
            ws.batch,
            ws.frames,
            d,
            self.spec.conv_kernel,
        )?;
        let glued = self.gemm_input(
            ws.merged.slice(..),
            ws.fp16.as_ref().map(|fp16| fp16.merged.slice(..)),
        )?;
        self.linear(
            runtime,
            linear(rows, d, d, 1.0, 1.0, math),
            glued,
            &weights.pointwise2,
            &mut ws.x,
            &mut ws.quant,
        )?;
        Ok(())
    }
}

impl Linear {
    /// Keeps `weight` for an FP32 or TF32 GEMM, or rounds it to FP16 on the device for
    /// [`EncoderMath::Fp16`] and frees the FP32 copy
    fn new(
        runtime: &CudaRuntime,
        kernels: &EncoderKernels,
        weight: DeviceTensor,
        math: EncoderMath,
    ) -> Result<Self, TranscriptionError> {
        if !math.fp16() {
            return Ok(Self::Single(weight));
        }
        let stream = runtime.stream();
        let mut bits = zeros(stream, weight.data().len())?;
        kernels.to_f16(stream, &weight.data().slice(..), &mut bits.slice_mut(..))?;
        Ok(Self::Fp16(bits))
    }
}

/// A workspace or weight in the wrong precision for the encoder's math, which
/// [`CudaEncoder::load`] and [`CudaEncoder::workspace`] never build
fn mixed_precision() -> TranscriptionError {
    TranscriptionError::InvalidModelOutput(
        "an encoder GEMM's operands differ in precision".to_owned(),
    )
}

/// NeMo's `RelPositionalEncoding` rows for positions `frames - 1` down to
/// `-(frames - 1)`, `[2 * frames - 1, d_model]`, computed in FP32 as NeMo does
fn relative_positions(frames: usize, d_model: usize) -> Vec<f32> {
    let scale = -(10000f64.ln() / d_model as f64) as f32;
    let div_term: Vec<f32> = (0..d_model)
        .step_by(2)
        .map(|i| (i as f32 * scale).exp())
        .collect();
    let mut table = Vec::with_capacity((2 * frames - 1) * d_model);
    for row in 0..2 * frames - 1 {
        let position = (frames - 1) as f32 - row as f32;
        for term in &div_term {
            let angle = position * term;
            table.push(angle.sin());
            table.push(angle.cos());
        }
    }
    table
}

#[cfg(test)]
mod tests {
    use cudakit::{Library, Math, Options};
    use ndarray::{ArrayD, Axis, Ix3};

    use super::{CudaEncoder, EncoderMath, relative_positions};
    use crate::cuda::kernels::CudaRuntime;
    use crate::cuda::test_support::weights;
    use crate::deployment::{ParakeetDeployment, ParakeetVersion};
    use crate::test_support::NemoReference;

    /// Relative L2 error and max abs difference over the valid frames of `[batch,
    /// frames, d_model]` outputs
    ///
    /// Panics on a non-finite value in either side, which `f64::max` would skip
    fn compare(ours: &[f32], reference: &ArrayD<f32>, lens: &[usize]) -> (f64, f64) {
        let reference = reference
            .view()
            .into_dimensionality::<Ix3>()
            .expect("[batch, frames, d]");
        assert_eq!(ours.len(), reference.len(), "output size");
        let (frames, width) = (reference.shape()[1], reference.shape()[2]);
        let (mut error, mut norm, mut max_abs) = (0f64, 0f64, 0f64);
        for (b, row) in reference.axis_iter(Axis(0)).enumerate() {
            for t in 0..lens[b] {
                for c in 0..width {
                    let expected = f64::from(row[[t, c]]);
                    let value = f64::from(ours[(b * frames + t) * width + c]);
                    assert!(
                        value.is_finite() && expected.is_finite(),
                        "row {b} frame {t} channel {c}: {value} against {expected}"
                    );
                    let diff = value - expected;
                    error += diff * diff;
                    norm += expected * expected;
                    max_abs = max_abs.max(diff.abs());
                }
            }
        }
        ((error / norm).sqrt(), max_abs)
    }

    #[test]
    #[should_panic(expected = "row 0 frame 1")]
    fn compare_rejects_nan_output() {
        let reference = ArrayD::from_elem(vec![1, 2, 2], 1.0f32);
        compare(&[1.0, 1.0, f32::NAN, 1.0], &reference, &[2]);
    }

    #[test]
    #[ignore = "reads NeMo references from /shared/datasets/parakeet-ref"]
    fn nemo_parity_relative_positions() {
        let reference = NemoReference::open(ParakeetVersion::V2, "b8");
        let table = reference.f32("encoder.layers.0.self_attn.linear_pos.in");
        let positions = table.shape()[1];
        let ours = relative_positions(positions.div_ceil(2), table.shape()[2]);
        assert_eq!(ours.len(), table.len(), "table size");
        assert!(
            ours.iter()
                .chain(table.iter())
                .all(|value| value.is_finite()),
            "non-finite relative position"
        );
        let max_abs = ours
            .iter()
            .zip(table.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        // sin and cos of arguments up to a few thousand radians differ by an ulp or so
        assert!(
            max_abs < 1e-3,
            "relative position table differs by {max_abs}"
        );
    }

    /// Runs the encoder stage by stage and holds every stage against NeMo FP32
    ///
    /// Needs a CUDA GPU: run it through the shared queue with `just gpu-parity`
    #[test]
    #[ignore = "needs a CUDA GPU and the NeMo references; `just gpu-parity` runs it on gpu0"]
    fn gpu_encoder_matches_nemo_layer_by_layer() {
        let runtime = CudaRuntime::open(0, Options::default()).expect("CUDA runtime");
        runtime.prepare(Library::Cublas).expect("cuBLAS");
        let mut worst = 0f64;
        for version in [ParakeetVersion::V2, ParakeetVersion::V3] {
            let deployment = match version {
                ParakeetVersion::V2 => ParakeetDeployment::v2(),
                ParakeetVersion::V3 => ParakeetDeployment::v3(),
            };
            let file = weights(version);
            let spec = file.parakeet_spec(&deployment).expect("spec");
            let encoder = CudaEncoder::load(
                &runtime,
                &file,
                spec.encoder,
                EncoderMath::Single(Math::Fp32),
            )
            .expect("encoder");

            for set in ["b1", "b8", "long-full"] {
                let reference = NemoReference::open(version, set);
                let mel = reference
                    .f32("mel")
                    .into_dimensionality::<Ix3>()
                    .expect("[batch, features, frames]");
                let (batch, mel_frames) = (mel.shape()[0], mel.shape()[2]);
                let lens: Vec<usize> = reference
                    .i64("mel_len")
                    .iter()
                    .map(|len| *len as usize)
                    .collect();
                // NeMo's featurizer is feature-major; the encoder takes time-major rows
                let time_major: Vec<f32> = mel.permuted_axes([0, 2, 1]).iter().copied().collect();

                let mut ws = encoder
                    .workspace(&runtime, batch, mel_frames)
                    .expect("workspace");
                encoder
                    .set_input(&runtime, &mut ws, &lens, |mel| {
                        mel.copy_from_slice(&time_major)
                    })
                    .expect("input");
                let out_lens: Vec<usize> = reference
                    .i64("encoder.out_len")
                    .iter()
                    .map(|len| *len as usize)
                    .collect();
                assert_eq!(
                    ws.output_lens(),
                    out_lens.as_slice(),
                    "{version:?} {set} encoder lengths"
                );

                let download = |ws: &super::EncoderWorkspace| {
                    runtime.stream().clone_dtoh(ws.output()).expect("download")
                };
                encoder
                    .enqueue_subsampling(&runtime, &mut ws)
                    .expect("subsampling");
                let (rel, max_abs) = compare(
                    &download(&ws),
                    &reference.f32("encoder.pre_encode.out"),
                    &out_lens,
                );
                println!("{version:?} {set} pre_encode rel {rel:.2e} max {max_abs:.2e}");
                for index in 0..spec.encoder.layers {
                    encoder
                        .enqueue_layer(&runtime, &mut ws, index)
                        .expect("layer");
                    let (rel, max_abs) = compare(
                        &download(&ws),
                        &reference.f32(&format!("encoder.layers.{index}.out")),
                        &out_lens,
                    );
                    println!("{version:?} {set} layer {index:2} rel {rel:.2e} max {max_abs:.2e}");
                }

                let expected = reference.f32("encoder.out").permuted_axes(vec![0, 2, 1]);
                let (rel, max_abs) = compare(
                    &download(&ws),
                    &expected.as_standard_layout().into_owned(),
                    &out_lens,
                );
                println!("{version:?} {set} encoder.out rel {rel:.2e} max {max_abs:.2e}");
                worst = worst.max(rel);
            }
        }

        // NeMo FP32 with TF32 off against cuBLAS FP32: summation order differs only
        assert!(worst < 1e-4, "encoder output relative L2 {worst:.2e}");
    }

    /// Runs the TF32 and FP16 encoders end to end and holds each near NeMo FP32 and
    /// finite, which an FP16 overflow would break. Each output must differ from
    /// the FP32 encoder's, or its GEMMs never left the FP32 path
    #[test]
    #[ignore = "needs a CUDA GPU and the NeMo references; `just gpu-parity` runs it on gpu0"]
    fn gpu_fast_encoders_stay_near_nemo() {
        // a 10-bit mantissa per product through 24 layers lands at 6e-4 to 1.3e-2 in
        // TF32 and 7e-4 to 6e-3 in FP16 on an RTX 5090; a broken GEMM is off by order
        // 1. WER is the accuracy gate
        let fast = [
            ("TF32", EncoderMath::Single(Math::Tf32)),
            ("FP16", EncoderMath::Fp16),
        ];
        let runtime = CudaRuntime::open(0, Options::default()).expect("CUDA runtime");
        runtime.prepare(Library::Cublas).expect("cuBLAS");
        let mut worst = [0f64; 2];
        for version in [ParakeetVersion::V2, ParakeetVersion::V3] {
            let deployment = match version {
                ParakeetVersion::V2 => ParakeetDeployment::v2(),
                ParakeetVersion::V3 => ParakeetDeployment::v3(),
            };
            let file = weights(version);
            let spec = file.parakeet_spec(&deployment).expect("spec");
            let load =
                |math| CudaEncoder::load(&runtime, &file, spec.encoder, math).expect("encoder");
            let fp32 = load(EncoderMath::Single(Math::Fp32));

            for set in ["b1", "b8", "long-full"] {
                let reference = NemoReference::open(version, set);
                let mel = reference
                    .f32("mel")
                    .into_dimensionality::<Ix3>()
                    .expect("[batch, features, frames]");
                let (batch, mel_frames) = (mel.shape()[0], mel.shape()[2]);
                let lens: Vec<usize> = reference
                    .i64("mel_len")
                    .iter()
                    .map(|len| *len as usize)
                    .collect();
                let time_major: Vec<f32> = mel.permuted_axes([0, 2, 1]).iter().copied().collect();
                let encode = |encoder: &CudaEncoder| {
                    let mut ws = encoder
                        .workspace(&runtime, batch, mel_frames)
                        .expect("workspace");
                    encoder
                        .set_input(&runtime, &mut ws, &lens, |mel| {
                            mel.copy_from_slice(&time_major)
                        })
                        .expect("input");
                    encoder.enqueue(&runtime, &mut ws).expect("encode");
                    let output = runtime.stream().clone_dtoh(ws.output()).expect("download");
                    (output, ws.output_lens().to_vec())
                };
                let (full, out_lens) = encode(&fp32);
                let expected = reference.f32("encoder.out").permuted_axes(vec![0, 2, 1]);
                let expected = expected.as_standard_layout().into_owned();
                for (index, (name, math)) in fast.iter().enumerate() {
                    // loaded per set so only one extra copy of the weights is resident
                    let (output, _) = encode(&load(*math));
                    assert!(
                        full != output,
                        "{version:?} {set}: {name} matches FP32 bit for bit"
                    );
                    let (rel, max_abs) = compare(&output, &expected, &out_lens);
                    println!(
                        "{version:?} {set} {name} encoder.out rel {rel:.2e} max {max_abs:.2e}"
                    );
                    worst[index] = worst[index].max(rel);
                }
            }
        }

        for ((name, _), worst) in fast.iter().zip(worst) {
            assert!(
                worst < 5e-2,
                "{name} encoder output relative L2 {worst:.2e}"
            );
        }
    }

    /// Captures the encoder in a CUDA graph at each precision, checks the replay
    /// against the eager pass bit for bit, and prints replay times for the v3
    /// reference shapes
    #[test]
    #[ignore = "needs a CUDA GPU and the NeMo references; `just gpu-parity` runs it on gpu0"]
    fn gpu_encoder_graph_replays_the_eager_pass() {
        let runtime = CudaRuntime::open(0, Options::default()).expect("CUDA runtime");
        runtime.prepare(Library::Cublas).expect("cuBLAS");
        let file = weights(ParakeetVersion::V3);
        let spec = file.parakeet_spec(&ParakeetDeployment::v3()).expect("spec");
        let precisions = [
            ("FP32", EncoderMath::Single(Math::Fp32)),
            ("TF32", EncoderMath::Single(Math::Tf32)),
            ("FP16", EncoderMath::Fp16),
        ];

        for (name, math) in precisions {
            let encoder = CudaEncoder::load(&runtime, &file, spec.encoder, math).expect("encoder");
            for set in ["b1", "b8"] {
                time_graph(&runtime, &encoder, name, set);
            }
        }
    }

    /// Checks one graph replay of `encoder` on the v3 reference `set` against its
    /// eager pass and prints the mean replay time
    pub(super) fn time_graph(runtime: &CudaRuntime, encoder: &CudaEncoder, name: &str, set: &str) {
        const REPLAYS: u32 = 100;
        let reference = NemoReference::open(ParakeetVersion::V3, set);
        let mel = reference
            .f32("mel")
            .into_dimensionality::<Ix3>()
            .expect("[batch, features, frames]");
        let (batch, mel_frames) = (mel.shape()[0], mel.shape()[2]);
        let lens: Vec<usize> = reference
            .i64("mel_len")
            .iter()
            .map(|len| *len as usize)
            .collect();
        let time_major: Vec<f32> = mel.permuted_axes([0, 2, 1]).iter().copied().collect();
        let mut ws = encoder
            .workspace(runtime, batch, mel_frames)
            .expect("workspace");
        encoder
            .set_input(runtime, &mut ws, &lens, |mel| {
                mel.copy_from_slice(&time_major)
            })
            .expect("input");

        // the eager pass also finishes cuBLAS's lazy setup before capture
        encoder.enqueue(runtime, &mut ws).expect("eager");
        let eager = runtime.stream().clone_dtoh(ws.output()).expect("download");
        // SAFETY: the graph is declared after `ws` and `encoder` outlives this call, so
        // it drops before the buffers it points at, which stay in place on the
        // runtime's stream
        let graph = unsafe { runtime.capture_nonempty(|| encoder.enqueue(runtime, &mut ws)) }
            .expect("capture");
        graph.launch().expect("replay");
        let replayed = runtime.stream().clone_dtoh(ws.output()).expect("download");
        assert!(
            eager == replayed,
            "v3 {set} {name}: graph replay differs from the eager pass"
        );

        let start = std::time::Instant::now();
        for _ in 0..REPLAYS {
            graph.launch().expect("replay");
        }
        runtime.synchronize().expect("sync");
        let replay_ms = start.elapsed().as_secs_f64() * 1e3 / f64::from(REPLAYS);
        let audio_s: f64 = reference
            .i64("audio_len")
            .iter()
            .map(|len| *len as f64 / 16000.0)
            .sum();
        println!(
            "v3 {set} {name}: batch {batch}, {audio_s:.1} s audio, encoder graph replay {replay_ms:.3} ms"
        );
    }
}
