//! Greedy TDT decode on the device: the joint's GEMMs on cuBLAS, and the LSTM
//! predictor and the token decision in the `decode` kernel area
//!
//! A step decides one token for every unfinished batch row and advances its frame
//! cursor on the device, NeMo's label-looping rule. Steps need no host sync, so a
//! batch runs as a replayed graph of steps, and the host only reads the cursors back
//! between replays to see whether every row has reached its last frame

use cudakit::cudarc::driver::CudaSlice;
use cudakit::{DeviceTensor, Math};

use crate::backend::LanguageMask;
use crate::cuda::kernels::{
    CudaRuntime, DecodeFilter, DecodeKernels, DecodeOutput, DecodeShape, DecodeState,
    EncoderKernels, LstmInput, LstmState, LstmWeights, MAX_FILTER_TOP_K, Projection,
};
use crate::cuda::{linear, upload, upload_host, zeros};
use crate::decode::RawTranscription;
use crate::error::TranscriptionError;
use crate::native::spec::ParakeetSpec;
use crate::native::weights::WeightFile;

/// A PyTorch `Linear` with bias
struct Linear {
    weight: DeviceTensor,
    bias: DeviceTensor,
}

struct LstmLayer {
    /// `[4 * hidden, inputs]`, gates in PyTorch order: input, forget, cell, output
    weight_ih: DeviceTensor,
    weight_hh: DeviceTensor,
    /// `bias_ih + bias_hh`
    bias: DeviceTensor,
}

pub(crate) struct CudaDecoder {
    /// Predictor width, also the LSTM state width
    hidden: usize,
    /// Joint hidden width
    joint_hidden: usize,
    encoder_hidden: usize,
    /// Token classes including the blank, which is the last
    classes: usize,
    /// Joint outputs: classes, then one logit per duration
    outputs: usize,
    max_symbols: usize,
    embed: DeviceTensor,
    lstm: Vec<LstmLayer>,
    joint_enc: Linear,
    joint_pred: Linear,
    joint_out: Linear,
    durations: CudaSlice<u32>,
    kernels: DecodeKernels,
    /// `fill_rows` for the GEMM biases
    encoder_kernels: EncoderKernels,
}

/// Device buffers for one `(batch, encoder frames)` shape
pub(crate) struct DecodeWorkspace {
    batch: usize,
    frames: usize,
    /// Output slots per row: a row emits at most `max_symbols` tokens per frame
    capacity: usize,
    host_lens: Vec<usize>,
    lens: CudaSlice<u32>,
    enc_proj: CudaSlice<f32>,
    pred_proj: CudaSlice<f32>,
    joint_in: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    /// LSTM states, `[layers, batch, hidden]` each: `h` before a step, `h_next` the
    /// layers' new states until `decode_project` copies them back, and `c` in place
    h: CudaSlice<f32>,
    h_next: CudaSlice<f32>,
    c: CudaSlice<f32>,
    frame: CudaSlice<u32>,
    emitted: CudaSlice<u32>,
    label: CudaSlice<u32>,
    emit: CudaSlice<u32>,
    counts: CudaSlice<u32>,
    faults: CudaSlice<u32>,
    tokens: CudaSlice<u32>,
    token_frames: CudaSlice<u32>,
    token_durations: CudaSlice<u32>,
    confidences: CudaSlice<f32>,
    /// `[classes]`, nonzero for the ids the language filter allows
    allowed: CudaSlice<u32>,
    /// `[1]`, the language filter's top-K width, 0 for none; a buffer rather than a
    /// launch argument so captured graphs follow [`CudaDecoder::set_language`]
    filter_top_k: CudaSlice<u32>,
}

impl DecodeWorkspace {
    /// Device bytes held by the workspace's buffers
    pub(crate) fn bytes(&self) -> usize {
        let floats = [
            &self.enc_proj,
            &self.pred_proj,
            &self.joint_in,
            &self.logits,
            &self.h,
            &self.h_next,
            &self.c,
            &self.confidences,
        ];
        let state = [
            &self.lens,
            &self.frame,
            &self.emitted,
            &self.label,
            &self.emit,
            &self.counts,
            &self.faults,
            &self.tokens,
            &self.token_frames,
            &self.token_durations,
            &self.allowed,
            &self.filter_top_k,
        ];
        floats.into_iter().map(CudaSlice::num_bytes).sum::<usize>()
            + state.into_iter().map(CudaSlice::num_bytes).sum::<usize>()
    }

    /// Upper bound on decode steps for the current lengths: each step either emits
    /// one of at most `max_symbols` tokens at a frame or moves to a later frame
    fn max_steps(&self, max_symbols: usize) -> usize {
        let longest = self.host_lens.iter().copied().max().unwrap_or(0);
        longest * (max_symbols + 1)
    }
}

impl CudaDecoder {
    /// Uploads the predictor and joint weights and loads the kernels
    pub(crate) fn load(
        runtime: &CudaRuntime,
        file: &WeightFile,
        spec: &ParakeetSpec,
    ) -> Result<Self, TranscriptionError> {
        let hidden = spec.predictor.hidden;
        let classes = spec.predictor.vocab_size + 1;
        let joint_hidden = spec.joint.hidden;
        let encoder_hidden = spec.encoder.d_model;
        let gates = 4 * hidden;
        if spec.predictor.layers == 0 {
            return Err(TranscriptionError::InvalidWeights(
                "the predictor has no LSTM layers".to_owned(),
            ));
        }

        let mut lstm = Vec::with_capacity(spec.predictor.layers);
        for layer in 0..spec.predictor.layers {
            let name = |part: &str| format!("decoder.prediction.dec_rnn.lstm.{part}_l{layer}");
            let bias_ih = file.tensor(&name("bias_ih"), &[gates])?;
            let bias_hh = file.tensor(&name("bias_hh"), &[gates])?;
            let bias: Vec<f32> = bias_ih.iter().zip(bias_hh).map(|(a, b)| a + b).collect();
            lstm.push(LstmLayer {
                weight_ih: upload(runtime, file, &name("weight_ih"), &[gates, hidden])?,
                weight_hh: upload(runtime, file, &name("weight_hh"), &[gates, hidden])?,
                bias: upload_host(runtime, &bias)?,
            });
        }

        let linear_weights =
            |prefix: &str, out: usize, inputs: usize| -> Result<Linear, TranscriptionError> {
                Ok(Linear {
                    weight: upload(runtime, file, &format!("{prefix}.weight"), &[out, inputs])?,
                    bias: upload(runtime, file, &format!("{prefix}.bias"), &[out])?,
                })
            };
        let durations: Vec<u32> = spec
            .durations
            .iter()
            .map(|duration| {
                u32::try_from(*duration).map_err(|_| {
                    TranscriptionError::InvalidWeights(format!("duration {duration} overflows"))
                })
            })
            .collect::<Result<_, _>>()?;

        Ok(Self {
            hidden,
            joint_hidden,
            encoder_hidden,
            classes,
            outputs: spec.joint.outputs,
            max_symbols: spec.max_symbols,
            embed: upload(
                runtime,
                file,
                "decoder.prediction.embed.weight",
                &[classes, hidden],
            )?,
            lstm,
            joint_enc: linear_weights("joint.enc", joint_hidden, encoder_hidden)?,
            joint_pred: linear_weights("joint.pred", joint_hidden, hidden)?,
            joint_out: linear_weights("joint.joint_net.2", spec.joint.outputs, joint_hidden)?,
            durations: runtime.stream().clone_htod(&durations)?,
            kernels: DecodeKernels::load(runtime)?,
            encoder_kernels: EncoderKernels::load(runtime)?,
        })
    }

    pub(crate) fn workspace(
        &self,
        runtime: &CudaRuntime,
        batch: usize,
        frames: usize,
    ) -> Result<DecodeWorkspace, TranscriptionError> {
        let stream = runtime.stream();
        let capacity = frames.max(1) * self.max_symbols;
        let state = |len: usize| zeros::<u32>(stream, len);
        let lstm_state = batch * self.lstm.len() * self.hidden;
        Ok(DecodeWorkspace {
            batch,
            frames,
            capacity,
            host_lens: vec![0; batch],
            lens: state(batch)?,
            enc_proj: zeros(stream, batch * frames * self.joint_hidden)?,
            pred_proj: zeros(stream, batch * self.joint_hidden)?,
            joint_in: zeros(stream, batch * self.joint_hidden)?,
            logits: zeros(stream, batch * self.outputs)?,
            h: zeros(stream, lstm_state)?,
            h_next: zeros(stream, lstm_state)?,
            c: zeros(stream, lstm_state)?,
            frame: state(batch)?,
            emitted: state(batch)?,
            label: state(batch)?,
            emit: state(batch)?,
            counts: state(batch)?,
            faults: state(batch)?,
            tokens: state(batch * capacity)?,
            token_frames: state(batch * capacity)?,
            token_durations: state(batch * capacity)?,
            confidences: zeros(stream, batch * capacity)?,
            allowed: state(self.classes)?,
            filter_top_k: state(1)?,
        })
    }

    /// Sets the language filter later decode steps apply, or turns it off for `None`
    ///
    /// Steps read the filter from the workspace, so graphs captured with one filter
    /// replay with whichever the last call set
    pub(crate) fn set_language(
        &self,
        runtime: &CudaRuntime,
        ws: &mut DecodeWorkspace,
        mask: Option<&LanguageMask>,
    ) -> Result<(), TranscriptionError> {
        let stream = runtime.stream();
        let top_k = match mask {
            None => 0,
            Some(mask) => {
                let (allowed, top_k) = (mask.allowed(), mask.top_k());
                // 0 would turn the filter off, and the kernel takes at most the
                // classes and its shared-memory bound
                if allowed.len() != self.classes
                    || top_k == 0
                    || top_k > self.classes.min(MAX_FILTER_TOP_K)
                {
                    return Err(TranscriptionError::InvalidConfiguration(format!(
                        "a top-{top_k} language filter over {} tokens does not fit the CUDA \
                         decoder's {} classes (at most {MAX_FILTER_TOP_K} candidates)",
                        allowed.len(),
                        self.classes
                    )));
                }

                let allowed: Vec<u32> = allowed.iter().map(|id| u32::from(*id)).collect();
                stream.memcpy_htod(&allowed, &mut ws.allowed)?;
                top_k as u32
            }
        };
        stream.memcpy_htod(&[top_k], &mut ws.filter_top_k)?;
        Ok(())
    }

    /// Projects `encoder_out` (`[batch * frames, d_model]`) for the joint, resets
    /// every row to the start symbol and runs the predictor on it
    pub(crate) fn start(
        &self,
        runtime: &CudaRuntime,
        ws: &mut DecodeWorkspace,
        encoder_out: &CudaSlice<f32>,
        lens: &[usize],
    ) -> Result<(), TranscriptionError> {
        if lens.len() != ws.batch || lens.iter().any(|len| *len > ws.frames) {
            return Err(TranscriptionError::InvalidConfiguration(format!(
                "decode lengths {lens:?} do not fit a batch of {} with {} frames",
                ws.batch, ws.frames
            )));
        }

        let stream = runtime.stream();
        let rows = ws.batch * ws.frames;
        self.encoder_kernels.fill_rows(
            stream,
            &self.joint_enc.bias.data().slice(..),
            &mut ws.enc_proj.slice_mut(..),
        )?;
        runtime.sgemm(
            linear(
                rows,
                self.joint_hidden,
                self.encoder_hidden,
                1.0,
                1.0,
                Math::Fp32,
            ),
            encoder_out,
            self.joint_enc.weight.data(),
            &mut ws.enc_proj,
        )?;

        ws.host_lens.copy_from_slice(lens);
        let lens: Vec<u32> = lens.iter().map(|len| *len as u32).collect();
        stream.memcpy_htod(&lens, &mut ws.lens)?;
        stream.memset_zeros(&mut ws.h)?;
        stream.memset_zeros(&mut ws.c)?;
        for buffer in [
            &mut ws.frame,
            &mut ws.emitted,
            &mut ws.counts,
            &mut ws.faults,
        ] {
            stream.memset_zeros(buffer)?;
        }
        // the blank is NeMo's start symbol, and every row feeds it once
        let blank = vec![(self.classes - 1) as u32; ws.batch];
        stream.memcpy_htod(&blank, &mut ws.label)?;
        stream.memcpy_htod(&vec![1u32; ws.batch], &mut ws.emit)?;
        self.predict(runtime, ws)
    }

    /// Feeds each emitting row's label through the LSTM and projects the top layer
    /// for the joint; the other rows keep their state and projection. Also prepares
    /// the next step: its joint input at each row's current frame, and the joint
    /// bias in the logits
    fn predict(
        &self,
        runtime: &CudaRuntime,
        ws: &mut DecodeWorkspace,
    ) -> Result<(), TranscriptionError> {
        let stream = runtime.stream();
        let n = ws.batch * self.hidden;
        for (index, layer) in self.lstm.iter().enumerate() {
            let weights = LstmWeights {
                weight_ih: layer.weight_ih.data().slice(..),
                weight_hh: layer.weight_hh.data().slice(..),
                bias: layer.bias.data().slice(..),
            };
            // the layer below's new state sits just before this layer's in `h_next`
            let (below, mut rest) = ws.h_next.split_at_mut(index * n);
            let input = match index {
                0 => LstmInput::Embedded {
                    table: self.embed.data().slice(..),
                    label: ws.label.slice(..),
                },
                _ => LstmInput::Below(below.slice((index - 1) * n..)),
            };
            let mut state = LstmState {
                h: ws.h.slice(index * n..(index + 1) * n),
                h_next: rest.slice_mut(..n),
                c: ws.c.slice_mut(index * n..(index + 1) * n),
            };
            self.kernels.lstm(
                stream,
                &weights,
                &input,
                &ws.emit.slice(..),
                &mut state,
                self.hidden,
            )?;
        }

        let top = (self.lstm.len() - 1) * n;
        let mut buffers = Projection {
            weight: self.joint_pred.weight.data().slice(..),
            bias: self.joint_pred.bias.data().slice(..),
            top: ws.h_next.slice(top..),
            enc_proj: ws.enc_proj.slice(..),
            frame: ws.frame.slice(..),
            lens: ws.lens.slice(..),
            joint_bias: self.joint_out.bias.data().slice(..),
            h_next: ws.h_next.slice(..),
            h: ws.h.slice_mut(..),
            pred_proj: ws.pred_proj.slice_mut(..),
            joint_in: ws.joint_in.slice_mut(..),
            logits: ws.logits.slice_mut(..),
        };
        self.kernels.project(
            stream,
            &mut buffers,
            &ws.emit.slice(..),
            ws.frames,
            self.hidden,
        )?;
        Ok(())
    }

    /// One greedy step for every unfinished row; capturable in a CUDA graph
    ///
    /// The previous step (or [`Self::start`]) left the joint input and the joint
    /// bias in the logits, so the step is the joint GEMM, the decision and the
    /// predictor
    pub(crate) fn step(
        &self,
        runtime: &CudaRuntime,
        ws: &mut DecodeWorkspace,
    ) -> Result<(), TranscriptionError> {
        runtime.sgemm(
            linear(
                ws.batch,
                self.outputs,
                self.joint_hidden,
                1.0,
                1.0,
                Math::Fp32,
            ),
            &ws.joint_in,
            self.joint_out.weight.data(),
            &mut ws.logits,
        )?;
        self.decide(runtime, ws)?;
        self.predict(runtime, ws)
    }

    /// Decides every unfinished row's token from the joint logits in `ws` and
    /// advances its frame cursor
    fn decide(
        &self,
        runtime: &CudaRuntime,
        ws: &mut DecodeWorkspace,
    ) -> Result<(), TranscriptionError> {
        let shape = DecodeShape {
            batch: ws.batch,
            classes: self.classes,
            max_symbols: self.max_symbols,
            capacity: ws.capacity,
        };
        self.kernels.decide(
            runtime.stream(),
            &ws.logits.slice(..),
            &ws.lens.slice(..),
            &self.durations.slice(..),
            &DecodeFilter {
                allowed: ws.allowed.slice(..),
                top_k: ws.filter_top_k.slice(..),
            },
            &mut DecodeState {
                frame: ws.frame.slice_mut(..),
                emitted: ws.emitted.slice_mut(..),
                label: ws.label.slice_mut(..),
                emit: ws.emit.slice_mut(..),
                counts: ws.counts.slice_mut(..),
                faults: ws.faults.slice_mut(..),
            },
            &mut DecodeOutput {
                tokens: ws.tokens.slice_mut(..),
                frames: ws.token_frames.slice_mut(..),
                durations: ws.token_durations.slice_mut(..),
                confidences: ws.confidences.slice_mut(..),
            },
            shape,
        )?;
        Ok(())
    }

    /// Whether every row has reached its last frame; syncs the stream
    pub(crate) fn finished(
        &self,
        runtime: &CudaRuntime,
        ws: &DecodeWorkspace,
    ) -> Result<bool, TranscriptionError> {
        let frames = runtime.stream().clone_dtoh(&ws.frame)?;
        Ok(frames
            .iter()
            .zip(&ws.host_lens)
            .all(|(frame, len)| *frame as usize >= *len))
    }

    /// Runs steps eagerly, `steps_per_check` at a time, until every row finishes
    pub(crate) fn run(
        &self,
        runtime: &CudaRuntime,
        ws: &mut DecodeWorkspace,
        steps_per_check: usize,
    ) -> Result<(), TranscriptionError> {
        self.run_with(runtime, ws, steps_per_check, |ws| {
            (0..steps_per_check).try_for_each(|_| self.step(runtime, ws))
        })
    }

    /// Runs `advance` (which must run `steps` decode steps) until every row
    /// finishes, failing if the decode does not end within its step bound
    pub(crate) fn run_with(
        &self,
        runtime: &CudaRuntime,
        ws: &mut DecodeWorkspace,
        steps: usize,
        mut advance: impl FnMut(&mut DecodeWorkspace) -> Result<(), TranscriptionError>,
    ) -> Result<(), TranscriptionError> {
        let bound = ws.max_steps(self.max_symbols);
        let mut done = 0;
        while !self.finished(runtime, ws)? {
            if done >= bound {
                return Err(TranscriptionError::InvalidModelOutput(format!(
                    "decode did not finish within {bound} steps"
                )));
            }
            advance(ws)?;
            done += steps.max(1);
        }
        Ok(())
    }

    /// Each row's emitted tokens; syncs the stream
    pub(crate) fn output(
        &self,
        runtime: &CudaRuntime,
        ws: &DecodeWorkspace,
    ) -> Result<Vec<RawTranscription>, TranscriptionError> {
        let stream = runtime.stream();
        let counts = stream.clone_dtoh(&ws.counts)?;
        let faults = stream.clone_dtoh(&ws.faults)?;
        if let Some(row) = faults.iter().position(|fault| *fault != 0) {
            return Err(TranscriptionError::InvalidModelOutput(format!(
                "joint logits for batch row {row} were not finite"
            )));
        }
        if let Some(count) = counts.iter().find(|count| **count as usize > ws.capacity) {
            return Err(TranscriptionError::InvalidModelOutput(format!(
                "decode emitted {count} tokens for {} output slots",
                ws.capacity
            )));
        }

        let tokens = stream.clone_dtoh(&ws.tokens)?;
        let frames = stream.clone_dtoh(&ws.token_frames)?;
        let durations = stream.clone_dtoh(&ws.token_durations)?;
        let confidences = stream.clone_dtoh(&ws.confidences)?;
        Ok(counts
            .iter()
            .enumerate()
            .map(|(row, count)| {
                let slots = row * ws.capacity..row * ws.capacity + *count as usize;
                RawTranscription {
                    token_ids: tokens[slots.clone()].to_vec(),
                    frame_indices: frames[slots.clone()].iter().map(|f| *f as usize).collect(),
                    durations: durations[slots.clone()]
                        .iter()
                        .map(|d| *d as usize)
                        .collect(),
                    confidences: confidences[slots].to_vec(),
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use cudakit::cudarc::driver::CudaSlice;
    use cudakit::{Library, Math, Options};
    use ndarray::Ix3;

    use super::{CudaDecoder, DecodeWorkspace};
    use crate::backend::{DecodePolicy, LanguageMask, decide_from_logits};
    use crate::cuda::encoder::CudaEncoder;
    use crate::cuda::kernels::CudaRuntime;
    use crate::cuda::test_support::{native_root, weights};
    use crate::decode::RawTranscription;
    use crate::deployment::{ParakeetDeployment, ParakeetVersion};
    use crate::language::{LanguageSelection, ParakeetLanguage};
    use crate::models::ModelBundle;
    use crate::test_support::NemoReference;

    /// Steps per host check of the frame cursors
    const STEPS_PER_CHECK: usize = 16;

    const SETS: [&str; 3] = ["b1", "b8", "long-full"];

    fn deployment(version: ParakeetVersion) -> ParakeetDeployment {
        match version {
            ParakeetVersion::V2 => ParakeetDeployment::v2(),
            ParakeetVersion::V3 => ParakeetDeployment::v3(),
        }
    }

    fn runtime() -> CudaRuntime {
        let runtime = CudaRuntime::open(0, Options::default()).expect("CUDA runtime");
        runtime.prepare(Library::Cublas).expect("cuBLAS");
        runtime
    }

    /// NeMo's encoder output as `[batch * frames, d_model]` on the device, with its
    /// valid lengths
    fn nemo_encoder_out(
        runtime: &CudaRuntime,
        reference: &NemoReference,
    ) -> (CudaSlice<f32>, usize, usize, Vec<usize>) {
        let out = reference
            .f32("encoder.out")
            .into_dimensionality::<Ix3>()
            .expect("[batch, d_model, frames]");
        let (batch, frames) = (out.shape()[0], out.shape()[2]);
        let time_major: Vec<f32> = out.permuted_axes([0, 2, 1]).iter().copied().collect();
        let lens = reference
            .i64("encoder.out_len")
            .iter()
            .map(|len| *len as usize)
            .collect();
        let device = runtime.stream().clone_htod(&time_major).expect("upload");
        (device, batch, frames, lens)
    }

    /// Holds each row against NeMo's batched greedy decode: tokens, frames and
    /// durations exactly
    fn assert_matches_nemo(
        name: &str,
        reference: &NemoReference,
        decoded: &[RawTranscription],
    ) -> usize {
        let mut tokens = 0;
        for (row, raw) in decoded.iter().enumerate() {
            let as_u32 = |key: &str| -> Vec<u32> {
                reference
                    .i32(&format!("utt{row}.hyp.{key}"))
                    .iter()
                    .map(|value| *value as u32)
                    .collect()
            };
            let frames: Vec<u32> = raw.frame_indices.iter().map(|f| *f as u32).collect();
            let durations: Vec<u32> = raw.durations.iter().map(|d| *d as u32).collect();
            assert_eq!(raw.token_ids, as_u32("tokens"), "{name} row {row} tokens");
            assert_eq!(frames, as_u32("timestamps"), "{name} row {row} frames");
            assert_eq!(durations, as_u32("durations"), "{name} row {row} durations");
            assert!(
                raw.confidences
                    .iter()
                    .all(|confidence| *confidence > 0.0 && *confidence <= 1.0),
                "{name} row {row} confidences"
            );
            tokens += raw.token_ids.len();
        }
        tokens
    }

    fn decode_eager(
        runtime: &CudaRuntime,
        decoder: &CudaDecoder,
        ws: &mut DecodeWorkspace,
        encoder_out: &CudaSlice<f32>,
        lens: &[usize],
    ) -> Vec<RawTranscription> {
        decoder
            .start(runtime, ws, encoder_out, lens)
            .expect("start");
        decoder.run(runtime, ws, STEPS_PER_CHECK).expect("run");
        decoder.output(runtime, ws).expect("output")
    }

    /// Decodes NeMo's own encoder output, then this crate's encoder output from NeMo's
    /// mel, and holds both against NeMo's decode
    #[test]
    #[ignore = "needs a CUDA GPU and the NeMo references; `just gpu-parity` runs it on gpu0"]
    fn gpu_decoder_matches_nemo_hypotheses() {
        let runtime = runtime();
        for version in [ParakeetVersion::V2, ParakeetVersion::V3] {
            let file = weights(version);
            let spec = file.parakeet_spec(&deployment(version)).expect("spec");
            let decoder = CudaDecoder::load(&runtime, &file, &spec).expect("decoder");
            let encoder =
                CudaEncoder::load(&runtime, &file, spec.encoder, Math::Fp32).expect("encoder");

            for set in SETS {
                let reference = NemoReference::open(version, set);
                let (encoder_out, batch, frames, lens) = nemo_encoder_out(&runtime, &reference);
                let mut ws = decoder
                    .workspace(&runtime, batch, frames)
                    .expect("workspace");
                let decoded = decode_eager(&runtime, &decoder, &mut ws, &encoder_out, &lens);
                let tokens = assert_matches_nemo(
                    &format!("{version:?} {set} NeMo encoder"),
                    &reference,
                    &decoded,
                );

                let mel = reference
                    .f32("mel")
                    .into_dimensionality::<Ix3>()
                    .expect("[batch, features, frames]");
                let mel_frames = mel.shape()[2];
                let mel_lens: Vec<usize> = reference
                    .i64("mel_len")
                    .iter()
                    .map(|len| *len as usize)
                    .collect();
                let time_major: Vec<f32> = mel.permuted_axes([0, 2, 1]).iter().copied().collect();
                let mut encoder_ws = encoder
                    .workspace(&runtime, batch, mel_frames)
                    .expect("encoder workspace");
                encoder
                    .set_input(&runtime, &mut encoder_ws, &time_major, &mel_lens)
                    .expect("input");
                encoder.enqueue(&runtime, &mut encoder_ws).expect("encoder");
                assert_eq!(encoder_ws.frames(), frames, "encoder frames");
                let decoded = decode_eager(
                    &runtime,
                    &decoder,
                    &mut ws,
                    encoder_ws.output(),
                    encoder_ws.output_lens(),
                );
                assert_matches_nemo(
                    &format!("{version:?} {set} CUDA encoder"),
                    &reference,
                    &decoded,
                );
                println!("{version:?} {set}: {tokens} tokens match NeMo from both encoders");
            }
        }
    }

    /// Replays a captured graph of decode steps, checks it against the eager decode
    /// and prints the decode time per set
    #[test]
    #[ignore = "needs a CUDA GPU and the NeMo references; `just gpu-parity` runs it on gpu0"]
    fn gpu_decoder_graph_replays_the_eager_decode() {
        const REPEATS: u32 = 20;
        let runtime = runtime();
        let version = ParakeetVersion::V3;
        let file = weights(version);
        let spec = file.parakeet_spec(&deployment(version)).expect("spec");
        let decoder = CudaDecoder::load(&runtime, &file, &spec).expect("decoder");

        for set in SETS {
            let reference = NemoReference::open(version, set);
            let (encoder_out, batch, frames, lens) = nemo_encoder_out(&runtime, &reference);
            let mut ws = decoder
                .workspace(&runtime, batch, frames)
                .expect("workspace");
            // the eager decode also finishes cuBLAS's lazy setup before capture
            let eager = decode_eager(&runtime, &decoder, &mut ws, &encoder_out, &lens);

            decoder
                .start(&runtime, &mut ws, &encoder_out, &lens)
                .expect("start");
            // SAFETY: the graph is declared after `ws` and `decoder`, so it drops
            // before the buffers it points at, which stay in place on the runtime's
            // stream; the closure returns errors instead of panicking
            let graph = unsafe {
                runtime.capture_nonempty(|| {
                    (0..STEPS_PER_CHECK).try_for_each(|_| decoder.step(&runtime, &mut ws))
                })
            }
            .expect("capture");

            let decode = |ws: &mut DecodeWorkspace| -> Vec<RawTranscription> {
                decoder
                    .start(&runtime, ws, &encoder_out, &lens)
                    .expect("start");
                decoder
                    .run_with(&runtime, ws, STEPS_PER_CHECK, |_| Ok(graph.launch()?))
                    .expect("run");
                decoder.output(&runtime, ws).expect("output")
            };
            let replayed = decode(&mut ws);
            for (row, (a, b)) in eager.iter().zip(&replayed).enumerate() {
                assert_eq!(a.token_ids, b.token_ids, "{set} row {row} tokens");
                assert_eq!(a.frame_indices, b.frame_indices, "{set} row {row} frames");
                assert_eq!(a.confidences, b.confidences, "{set} row {row} confidences");
            }

            let start = std::time::Instant::now();
            for _ in 0..REPEATS {
                decode(&mut ws);
            }
            let ms = start.elapsed().as_secs_f64() * 1e3 / f64::from(REPEATS);
            let tokens: usize = replayed.iter().map(|raw| raw.token_ids.len()).sum();
            let longest = lens.iter().copied().max().unwrap_or(0);
            println!(
                "v3 {set}: batch {batch}, {longest} frames, {tokens} tokens, graph decode {ms:.3} ms"
            );
        }
    }

    /// Rows in the language filter fixture, 48 of each of its eight kinds
    const FILTER_ROWS: usize = 384;

    /// Seeded generator, so a failing fixture row reproduces
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 40
        }

        /// Uniform in `[low, high)`
        fn uniform(&mut self, low: f32, high: f32) -> f32 {
            low + (high - low) * (self.next() as f32 / (1u64 << 24) as f32)
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }

        /// `count` distinct entries of `ids`
        fn pick(&mut self, ids: &[usize], count: usize) -> Vec<usize> {
            let mut ids = ids.to_vec();
            for i in 0..count {
                let j = i + self.below(ids.len() - i);
                ids.swap(i, j);
            }
            ids.truncate(count);
            ids
        }
    }

    /// Joint logits, one row per decision, built around `mask` so the filter meets
    /// its edge cases: ties at the max and at the K-th rank, a best allowed id
    /// ranked K-1 and K, nothing allowed in the top K, and a blank argmax
    fn filter_fixture(mask: &LanguageMask, bins: usize, seed: u64) -> Vec<Vec<f32>> {
        let flags = mask.allowed();
        let classes = flags.len();
        let blank = classes - 1;
        let k = mask.top_k();
        let allowed: Vec<usize> = (0..blank).filter(|id| flags[*id]).collect();
        let others: Vec<usize> = (0..blank).filter(|id| !flags[*id]).collect();
        assert!(
            allowed.len() > k + 32 && others.len() > k + 32,
            "{} allowed and {} other ids are too few for the fixture",
            allowed.len(),
            others.len()
        );

        let mut rng = Lcg(seed);
        (0..FILTER_ROWS)
            .map(|row| {
                let mut logits = vec![0.0f32; classes + bins];
                let tokens = &mut logits[..classes];
                let noise = |rng: &mut Lcg, tokens: &mut [f32]| {
                    tokens.iter_mut().for_each(|v| *v = rng.uniform(-2.0, 2.0));
                };
                let floor = |rng: &mut Lcg, tokens: &mut [f32]| {
                    tokens
                        .iter_mut()
                        .for_each(|v| *v = rng.uniform(-10.0, -5.0));
                };
                match row % 8 {
                    // noise with rare spikes, the shape of the host rule's own test
                    0 => tokens.iter_mut().for_each(|v| {
                        let unit = rng.uniform(0.0, 1.0);
                        *v = if unit > 0.97 {
                            unit * 12.0
                        } else {
                            unit * 4.0 - 2.0
                        };
                    }),
                    // half steps, so ties fall everywhere, the max and the K-th included
                    1 => tokens
                        .iter_mut()
                        .for_each(|v| *v = (rng.uniform(-6.0, 6.0) * 2.0).round() / 2.0),
                    // a blank argmax, which the filter leaves alone
                    2 => {
                        noise(&mut rng, tokens);
                        tokens[blank] = 9.0;
                    }
                    // the best allowed id ranks K-1, the last place the filter takes,
                    // or K, the first it does not
                    3 | 4 => {
                        floor(&mut rng, tokens);
                        let ahead = if row % 8 == 3 { k - 1 } else { k };
                        for id in rng.pick(&others, ahead) {
                            tokens[id] = rng.uniform(5.0, 10.0);
                        }
                        tokens[allowed[rng.below(allowed.len())]] = 4.0;
                    }
                    // twelve equal logits straddle the K-th rank, so the top K takes
                    // the four lowest ids, which may or may not hold an allowed one
                    5 => {
                        floor(&mut rng, tokens);
                        let picked = rng.pick(&others, k - 4 + 8);
                        let (high, tied) = picked.split_at(k - 4);
                        for id in high {
                            tokens[*id] = rng.uniform(6.0, 10.0);
                        }
                        for id in tied.iter().chain(&rng.pick(&allowed, 4)) {
                            tokens[*id] = 5.0;
                        }
                    }
                    // no allowed id anywhere near the top K
                    6 => {
                        floor(&mut rng, tokens);
                        for id in rng.pick(&others, k + 20) {
                            tokens[id] = rng.uniform(0.0, 10.0);
                        }
                    }
                    // an allowed and another id tie at the max, the lower id winning
                    // the argmax
                    _ => {
                        noise(&mut rng, tokens);
                        tokens[allowed[rng.below(allowed.len())]] = 9.0;
                        tokens[others[rng.below(others.len())]] = 9.0;
                    }
                }
                // whole numbers, so duration ties come up too
                for v in &mut logits[classes..] {
                    *v = rng.uniform(-3.0, 3.0).round();
                }
                logits
            })
            .collect()
    }

    /// Device state after one decision per row
    struct Decided {
        counts: Vec<u32>,
        faults: Vec<u32>,
        frame: Vec<u32>,
        tokens: Vec<u32>,
        durations: Vec<u32>,
        confidences: Vec<f32>,
    }

    /// One `decide` launch over `rows`, each with a single frame left, under `mask`
    fn decide_once(
        runtime: &CudaRuntime,
        decoder: &CudaDecoder,
        ws: &mut DecodeWorkspace,
        rows: &[Vec<f32>],
        mask: Option<&LanguageMask>,
    ) -> Decided {
        let stream = runtime.stream();
        let flat: Vec<f32> = rows.concat();
        stream.memcpy_htod(&flat, &mut ws.logits).expect("logits");
        stream
            .memcpy_htod(&vec![1u32; ws.batch], &mut ws.lens)
            .expect("lens");
        for buffer in [
            &mut ws.frame,
            &mut ws.emitted,
            &mut ws.counts,
            &mut ws.faults,
        ] {
            stream.memset_zeros(buffer).expect("reset");
        }
        decoder.set_language(runtime, ws, mask).expect("filter");
        decoder.decide(runtime, ws).expect("decide");

        let read = |buffer: &CudaSlice<u32>| stream.clone_dtoh(buffer).expect("read back");
        Decided {
            counts: read(&ws.counts),
            faults: read(&ws.faults),
            frame: read(&ws.frame),
            tokens: read(&ws.tokens),
            durations: read(&ws.token_durations),
            confidences: stream.clone_dtoh(&ws.confidences).expect("read back"),
        }
    }

    /// One decision per row of synthetic v3 logits, with each language's filter and
    /// with none, against the host rule `decide_from_logits`: the same token and
    /// duration, and the same confidence within the sum-order tolerances. Each
    /// fixture also runs under the next language's mask straight after its own, at
    /// the same K, so a mask left over from the previous call picks the wrong ids.
    /// Rows with a non-finite class logit fault under the filter, as the host rule
    /// rejects them
    #[test]
    #[ignore = "needs a CUDA GPU and the native weights; `just gpu-parity` runs it on gpu0"]
    fn gpu_language_filter_matches_the_host_rule() {
        // the device adds the top-K softmax terms in the host's order, so only `exp`
        // rounding (2 ulp on the device) separates the two
        const TOP_K_TOLERANCE: f32 = 1e-6;
        // the full softmax adds 8193 terms one after another on the host but strided
        // and then as a tree on the device; on peaked rows the host's running sum
        // drops terms below half an ulp, about 4e-6 at worst on this fixture when
        // both orders are simulated in f32
        const FULL_SOFTMAX_TOLERANCE: f32 = 2e-5;

        let runtime = runtime();
        let deployment = ParakeetDeployment::v3();
        let file = weights(ParakeetVersion::V3);
        let spec = file.parakeet_spec(&deployment).expect("spec");
        let decoder = CudaDecoder::load(&runtime, &file, &spec).expect("decoder");
        let vocab = ModelBundle::from_native_dir(native_root(), deployment.clone())
            .vocabulary()
            .expect("vocabulary");
        let blank = vocab.blank_id();
        assert_eq!(blank + 1, decoder.classes, "vocabulary and joint classes");
        let bins = deployment.duration_bins().len();
        let auto = DecodePolicy::new(&deployment, &vocab, LanguageSelection::Auto).expect("policy");
        let mut ws = decoder
            .workspace(&runtime, FILTER_ROWS, 1)
            .expect("workspace");

        let (mut retargeted, mut rescored, mut outside) = (0, 0, 0);
        let languages = [
            ParakeetLanguage::Polish,
            ParakeetLanguage::Russian,
            ParakeetLanguage::Greek,
        ];
        let policies: Vec<DecodePolicy<'_>> = languages
            .iter()
            .map(|language| {
                DecodePolicy::new(&deployment, &vocab, LanguageSelection::Language(*language))
                    .expect("policy")
            })
            .collect();
        let masks: Vec<LanguageMask> = policies
            .iter()
            .map(|policy| {
                policy
                    .language_mask()
                    .expect("v3 filters an explicit language")
            })
            .collect();
        for (seed, language) in languages.into_iter().enumerate() {
            let rows = filter_fixture(&masks[seed], bins, seed as u64);
            let next = (seed + 1) % languages.len();

            for active in [Some(seed), Some(next), None] {
                let (policy, filter) = match active {
                    Some(index) => (&policies[index], Some(&masks[index])),
                    None => (&policies[seed], None),
                };
                let decided = decide_once(&runtime, &decoder, &mut ws, &rows, filter);
                for (row, logits) in rows.iter().enumerate() {
                    let context = format!(
                        "{language:?} row {row}, filter {:?}",
                        active.map(|index| languages[index])
                    );
                    let host = decide_from_logits(policy, logits, filter).expect("host rule");
                    let plain = decide_from_logits(&auto, logits, None).expect("host rule");
                    assert_eq!(decided.faults[row], 0, "{context}: fault");
                    if host.token_id == blank {
                        assert_eq!(decided.counts[row], 0, "{context}: emitted for a blank");
                        assert_eq!(
                            decided.frame[row] as usize,
                            host.duration_frames.max(1),
                            "{context}: advance"
                        );
                        continue;
                    }

                    let slot = row * ws.capacity;
                    assert_eq!(decided.counts[row], 1, "{context}: emitted");
                    assert_eq!(
                        decided.tokens[slot] as usize, host.token_id,
                        "{context}: token"
                    );
                    assert_eq!(
                        decided.durations[slot] as usize, host.duration_frames,
                        "{context}: duration"
                    );
                    // an allowed argmax ranks first, so the top-K softmax scores it, as
                    // it does a filtered token; otherwise no allowed id made the top K
                    let top_k = filter.is_some_and(|mask| {
                        host.token_id != plain.token_id || mask.allowed()[plain.token_id]
                    });
                    let tolerance = if top_k {
                        TOP_K_TOLERANCE
                    } else {
                        FULL_SOFTMAX_TOLERANCE
                    };
                    let confidence = decided.confidences[slot];
                    assert!(
                        (confidence - host.confidence).abs() <= tolerance,
                        "{context}: confidence {confidence} against {}",
                        host.confidence
                    );

                    match (filter.is_some(), top_k) {
                        (true, true) if host.token_id != plain.token_id => retargeted += 1,
                        (true, true) => rescored += 1,
                        (true, false) => outside += 1,
                        (false, _) => {}
                    }
                }
            }
        }

        // non-finite class logits: a low -inf under a finite max, the max at +inf, a
        // NaN, a -inf beside a blank max, which the filter itself skips, and fewer
        // finite classes than K
        let mut rows = filter_fixture(&masks[0], bins, 7);
        let finite = blank / 2;
        rows[0][3] = f32::NEG_INFINITY;
        rows[1][finite] = f32::INFINITY;
        rows[2][finite] = f32::NAN;
        rows[3][blank] = 50.0;
        rows[3][3] = f32::NEG_INFINITY;
        rows[4][..blank].fill(f32::NEG_INFINITY);
        rows[4][finite..finite + 10].fill(1.0);
        let decided = decide_once(&runtime, &decoder, &mut ws, &rows, Some(&masks[0]));
        for (row, logits) in rows.iter().enumerate().take(5) {
            assert!(
                decide_from_logits(&policies[0], logits, Some(&masks[0])).is_err(),
                "row {row}: the host rule accepted a non-finite logit"
            );
            assert_ne!(decided.faults[row], 0, "non-finite row {row} did not fault");
        }
        assert!(
            decoder.output(&runtime, &ws).is_err(),
            "faulted rows reached the output"
        );

        // the fixture must reach every branch of the filter, not only the argmax
        println!("{retargeted} retargeted, {rescored} rescored, {outside} outside the top K");
        assert!(
            retargeted > 100 && rescored > 50 && outside > 100,
            "{retargeted} retargeted, {rescored} rescored, {outside} outside the top K"
        );
    }
}
