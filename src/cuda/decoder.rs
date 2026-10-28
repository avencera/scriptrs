//! Greedy TDT decode on the device: LSTM predictor and joint on cuBLAS plus the
//! `decode` kernel area
//!
//! A step decides one token for every unfinished batch row and advances its frame
//! cursor on the device, NeMo's label-looping rule. Steps need no host sync, so a
//! batch runs as a replayed graph of steps, and the host only reads the cursors back
//! between replays to see whether every row has reached its last frame

use cudakit::cudarc::driver::CudaSlice;
use cudakit::{DeviceTensor, Error};

use crate::cuda::kernels::{
    CudaRuntime, DecodeKernels, DecodeOutput, DecodeShape, DecodeState, EncoderKernels,
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
    embedded: CudaSlice<f32>,
    gates: CudaSlice<f32>,
    h: Vec<CudaSlice<f32>>,
    c: Vec<CudaSlice<f32>>,
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
}

impl DecodeWorkspace {
    pub(crate) fn batch(&self) -> usize {
        self.batch
    }

    pub(crate) fn frames(&self) -> usize {
        self.frames
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
        let layers = self.lstm.len();
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
            embedded: zeros(stream, batch * self.hidden)?,
            gates: zeros(stream, batch * 4 * self.hidden)?,
            h: (0..layers)
                .map(|_| zeros(stream, batch * self.hidden))
                .collect::<Result<_, Error>>()?,
            c: (0..layers)
                .map(|_| zeros(stream, batch * self.hidden))
                .collect::<Result<_, Error>>()?,
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
        })
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
            linear(rows, self.joint_hidden, self.encoder_hidden, 1.0, 1.0),
            encoder_out,
            self.joint_enc.weight.data(),
            &mut ws.enc_proj,
        )?;

        ws.host_lens.copy_from_slice(lens);
        let lens: Vec<u32> = lens.iter().map(|len| *len as u32).collect();
        stream.memcpy_htod(&lens, &mut ws.lens)?;
        for buffer in ws.h.iter_mut().chain(ws.c.iter_mut()) {
            stream.memset_zeros(buffer)?;
        }
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
    /// for the joint; the other rows keep their state and projection
    fn predict(
        &self,
        runtime: &CudaRuntime,
        ws: &mut DecodeWorkspace,
    ) -> Result<(), TranscriptionError> {
        let stream = runtime.stream();
        let batch = ws.batch;
        self.kernels.embed(
            stream,
            &self.embed.data().slice(..),
            &ws.label.slice(..),
            &mut ws.embedded.slice_mut(..),
            self.hidden,
        )?;

        let gates = 4 * self.hidden;
        for (index, layer) in self.lstm.iter().enumerate() {
            self.encoder_kernels.fill_rows(
                stream,
                &layer.bias.data().slice(..),
                &mut ws.gates.slice_mut(..),
            )?;
            let input = if index == 0 {
                &ws.embedded
            } else {
                &ws.h[index - 1]
            };
            runtime.sgemm(
                linear(batch, gates, self.hidden, 1.0, 1.0),
                input,
                layer.weight_ih.data(),
                &mut ws.gates,
            )?;
            runtime.sgemm(
                linear(batch, gates, self.hidden, 1.0, 1.0),
                &ws.h[index],
                layer.weight_hh.data(),
                &mut ws.gates,
            )?;
            let (h, c) = (&mut ws.h[index], &mut ws.c[index]);
            self.kernels.lstm_cell(
                stream,
                &ws.gates.slice(..),
                &ws.emit.slice(..),
                &mut h.slice_mut(..),
                &mut c.slice_mut(..),
                self.hidden,
            )?;
        }

        // rows that did not emit have an unchanged top state, so recomputing their
        // projection reproduces it
        self.encoder_kernels.fill_rows(
            stream,
            &self.joint_pred.bias.data().slice(..),
            &mut ws.pred_proj.slice_mut(..),
        )?;
        runtime.sgemm(
            linear(batch, self.joint_hidden, self.hidden, 1.0, 1.0),
            &ws.h[self.lstm.len() - 1],
            self.joint_pred.weight.data(),
            &mut ws.pred_proj,
        )?;
        Ok(())
    }

    /// One greedy step for every unfinished row; capturable in a CUDA graph
    pub(crate) fn step(
        &self,
        runtime: &CudaRuntime,
        ws: &mut DecodeWorkspace,
    ) -> Result<(), TranscriptionError> {
        let stream = runtime.stream();
        self.kernels.joint_input(
            stream,
            &ws.enc_proj.slice(..),
            &ws.pred_proj.slice(..),
            &ws.frame.slice(..),
            &ws.lens.slice(..),
            &mut ws.joint_in.slice_mut(..),
            ws.frames,
            self.joint_hidden,
        )?;
        self.encoder_kernels.fill_rows(
            stream,
            &self.joint_out.bias.data().slice(..),
            &mut ws.logits.slice_mut(..),
        )?;
        runtime.sgemm(
            linear(ws.batch, self.outputs, self.joint_hidden, 1.0, 1.0),
            &ws.joint_in,
            self.joint_out.weight.data(),
            &mut ws.logits,
        )?;

        let shape = DecodeShape {
            batch: ws.batch,
            classes: self.classes,
            max_symbols: self.max_symbols,
            capacity: ws.capacity,
        };
        self.kernels.decide(
            stream,
            &ws.logits.slice(..),
            &ws.lens.slice(..),
            &self.durations.slice(..),
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
        self.predict(runtime, ws)
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
    use cudakit::{Library, Options};
    use ndarray::Ix3;

    use super::{CudaDecoder, DecodeWorkspace};
    use crate::cuda::encoder::CudaEncoder;
    use crate::cuda::kernels::CudaRuntime;
    use crate::cuda::test_support::weights;
    use crate::decode::RawTranscription;
    use crate::deployment::{ParakeetDeployment, ParakeetVersion};
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
            let encoder = CudaEncoder::load(&runtime, &file, spec.encoder).expect("encoder");

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
}
