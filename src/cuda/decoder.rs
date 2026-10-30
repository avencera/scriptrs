//! Greedy TDT decode on the device: the joint's GEMMs on cuBLAS, and the LSTM
//! predictor and the token decision in the `decode` kernel area
//!
//! A step decides one token for every unfinished batch row and advances its frame
//! cursor on the device, NeMo's label-looping rule. Steps need no host sync, so a
//! batch runs as a replayed graph of steps, and the host only reads the cursors back
//! between replays to see whether every row has reached its last frame

use cudakit::cudarc::driver::CudaSlice;
use cudakit::{DeviceTensor, Graph, Math};

use crate::backend::LanguageMask;
use crate::cuda::kernels::{
    CudaRuntime, DecodeFilter, DecodeKernels, DecodeOutput, DecodeShape, DecodeState,
    EncoderKernels, LstmInput, LstmState, LstmWeights, MAX_FILTER_TOP_K, Projection, StreamKernels,
};
use crate::cuda::{Direction, HostStaging, capture, linear, upload, upload_host, zeros};
use crate::decode::{RawToken, RawTranscription};
use crate::error::TranscriptionError;
use crate::native::spec::TransducerSpec;
use crate::native::weights::WeightFile;

/// Decode steps per replay of the decode graph, between host checks of the frame
/// cursors
pub(crate) const STEPS_PER_CHECK: usize = 16;

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
    /// Most frames one step moves a cursor: the longest duration, or 1, which a
    /// blank or a frame's last symbol always moves
    max_advance: usize,
    embed: DeviceTensor,
    lstm: Vec<LstmLayer>,
    joint_enc: Linear,
    joint_pred: Linear,
    joint_out: Linear,
    durations: CudaSlice<u32>,
    kernels: DecodeKernels,
    /// `fill_rows` for the GEMM biases
    encoder_kernels: EncoderKernels,
    /// Slot gathers and scatters for [`DecoderStates`]
    stream_kernels: StreamKernels,
}

/// Each stream's predictor between its chunks, one slot per stream
///
/// A chunk's decode resumes from the state its stream's last chunk left: the LSTM
/// after the last token and the joint's projection of it, as NeMo's label-looping
/// decode carries them between `conformer_stream_step` calls
pub(crate) struct DecoderStates {
    slots: usize,
    /// `[slots, layers, hidden]` each
    h: CudaSlice<f32>,
    c: CudaSlice<f32>,
    /// `[slots, joint_hidden]`
    pred_proj: CudaSlice<f32>,
}

/// One stream's row in a chunk decode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DecodeRow {
    /// The stream's slot in the [`DecoderStates`]
    pub slot: usize,
    /// The stream's first chunk, which starts from the blank instead of a saved
    /// state
    pub fresh: bool,
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
    /// `[batch]` each row's stream slot, for [`CudaDecoder::resume`]
    slots: CudaSlice<u32>,
    host: DecodeReadback,
}

/// Pinned host copies of the cursors and outputs the host reads back
struct DecodeReadback {
    frame: HostStaging<u32>,
    counts: HostStaging<u32>,
    faults: HostStaging<u32>,
    tokens: HostStaging<u32>,
    token_frames: HostStaging<u32>,
    token_durations: HostStaging<u32>,
    confidences: HostStaging<f32>,
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
            &self.slots,
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
    ///
    /// An RNNT joint gets one zero duration logit appended, so it runs on the TDT
    /// decision as a single duration of 0: a blank then moves to the next frame and
    /// a token stays on its frame until `max_symbols`, NeMo's RNNT greedy rule
    pub(crate) fn load(
        runtime: &CudaRuntime,
        file: &WeightFile,
        spec: &TransducerSpec,
    ) -> Result<Self, TranscriptionError> {
        let hidden = spec.predictor.hidden;
        let classes = spec.predictor.vocab_size + 1;
        let joint_hidden = spec.joint.hidden;
        let encoder_hidden = spec.encoder_hidden;
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
        let (joint_out, outputs, durations) = if spec.durations.is_empty() {
            let weight = file.tensor("joint.joint_net.2.weight", &[classes, joint_hidden])?;
            let bias = file.tensor("joint.joint_net.2.bias", &[classes])?;
            let weight: Vec<f32> = weight
                .iter()
                .copied()
                .chain(std::iter::repeat_n(0.0, joint_hidden))
                .collect();
            let bias: Vec<f32> = bias.iter().copied().chain([0.0]).collect();
            let joint_out = Linear {
                weight: upload_host(runtime, &weight)?,
                bias: upload_host(runtime, &bias)?,
            };
            (joint_out, classes + 1, vec![0])
        } else {
            let joint_out = linear_weights("joint.joint_net.2", spec.joint.outputs, joint_hidden)?;
            (joint_out, spec.joint.outputs, spec.durations.clone())
        };
        let max_advance = durations.iter().copied().max().unwrap_or(0).max(1);
        let durations: Vec<u32> = durations
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
            outputs,
            max_symbols: spec.max_symbols,
            max_advance,
            embed: upload(
                runtime,
                file,
                "decoder.prediction.embed.weight",
                &[classes, hidden],
            )?,
            lstm,
            joint_enc: linear_weights("joint.enc", joint_hidden, encoder_hidden)?,
            joint_pred: linear_weights("joint.pred", joint_hidden, hidden)?,
            joint_out,
            durations: runtime.stream().clone_htod(&durations)?,
            kernels: DecodeKernels::load(runtime)?,
            encoder_kernels: EncoderKernels::load(runtime)?,
            stream_kernels: StreamKernels::load(runtime)?,
        })
    }

    /// Zeroed predictor states for `slots` streams
    pub(crate) fn states(
        &self,
        runtime: &CudaRuntime,
        slots: usize,
    ) -> Result<DecoderStates, TranscriptionError> {
        let stream = runtime.stream();
        let lstm = slots * self.lstm.len() * self.hidden;
        Ok(DecoderStates {
            slots,
            h: zeros(stream, lstm)?,
            c: zeros(stream, lstm)?,
            pred_proj: zeros(stream, slots * self.joint_hidden)?,
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
        let readback = |len: usize| HostStaging::<u32>::new(stream, len, Direction::Download);
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
            slots: state(batch)?,
            host: DecodeReadback {
                frame: readback(batch)?,
                counts: readback(batch)?,
                faults: readback(batch)?,
                tokens: readback(batch * capacity)?,
                token_frames: readback(batch * capacity)?,
                token_durations: readback(batch * capacity)?,
                confidences: HostStaging::new(stream, batch * capacity, Direction::Download)?,
            },
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
        self.project_encoder(runtime, ws, encoder_out, lens)?;
        let stream = runtime.stream();
        stream.memset_zeros(&mut ws.h)?;
        stream.memset_zeros(&mut ws.c)?;
        self.reset_cursors(runtime, ws, &vec![1u32; ws.batch])?;
        self.predict(runtime, ws)
    }

    /// [`Self::start`] for a chunk of each row's stream: a fresh row starts from the
    /// blank, and every other row resumes from its slot in `states`
    pub(crate) fn resume(
        &self,
        runtime: &CudaRuntime,
        ws: &mut DecodeWorkspace,
        encoder_out: &CudaSlice<f32>,
        lens: &[usize],
        states: &mut DecoderStates,
        rows: &[DecodeRow],
    ) -> Result<(), TranscriptionError> {
        let mut seen = vec![false; states.slots];
        for row in rows {
            match seen.get_mut(row.slot) {
                Some(seen) if !*seen => *seen = true,
                _ => {
                    return Err(TranscriptionError::InvalidConfiguration(format!(
                        "decoder slot {} is repeated or outside the {} slots",
                        row.slot, states.slots
                    )));
                }
            }
        }
        if rows.len() != ws.batch {
            return Err(TranscriptionError::InvalidConfiguration(format!(
                "{} decode rows for a batch of {}",
                rows.len(),
                ws.batch
            )));
        }
        self.project_encoder(runtime, ws, encoder_out, lens)?;

        let stream = runtime.stream();
        // a fresh stream's slot may hold a finished stream's state
        let lstm = self.lstm.len() * self.hidden;
        for row in rows.iter().filter(|row| row.fresh) {
            let range = row.slot * lstm..(row.slot + 1) * lstm;
            stream.memset_zeros(&mut states.h.slice_mut(range.clone()))?;
            stream.memset_zeros(&mut states.c.slice_mut(range))?;
        }
        let slots: Vec<u32> = rows.iter().map(|row| row.slot as u32).collect();
        stream.memcpy_htod(&slots, &mut ws.slots)?;
        let layers = self.lstm.len();
        // the buffers `save` scatters back, in the same order
        for (slots_state, rows, groups) in [
            (&states.h, &mut ws.h, layers),
            (&states.c, &mut ws.c, layers),
            (&states.pred_proj, &mut ws.pred_proj, 1),
        ] {
            self.stream_kernels.gather(
                stream,
                &slots_state.slice(..),
                &ws.slots.slice(..),
                &mut rows.slice_mut(..),
                groups,
            )?;
        }

        // a resumed row's predictor already holds its last token, so it only
        // computes the first frame's joint input
        let emit: Vec<u32> = rows.iter().map(|row| u32::from(row.fresh)).collect();
        self.reset_cursors(runtime, ws, &emit)?;
        self.predict(runtime, ws)
    }

    /// Saves each row's predictor to its slot after a [`Self::resume`] decode
    pub(crate) fn save(
        &self,
        runtime: &CudaRuntime,
        ws: &DecodeWorkspace,
        states: &mut DecoderStates,
    ) -> Result<(), TranscriptionError> {
        let stream = runtime.stream();
        let layers = self.lstm.len();
        // the buffers `resume` gathers, in the same order
        for (rows, slots_state, groups) in [
            (&ws.h, &mut states.h, layers),
            (&ws.c, &mut states.c, layers),
            (&ws.pred_proj, &mut states.pred_proj, 1),
        ] {
            self.stream_kernels.scatter(
                stream,
                &rows.slice(..),
                &ws.slots.slice(..),
                &mut slots_state.slice_mut(..),
                groups,
            )?;
        }
        Ok(())
    }

    /// Puts every cursor on frame 0 with no tokens and marks the rows in `emit` to
    /// feed the blank
    fn reset_cursors(
        &self,
        runtime: &CudaRuntime,
        ws: &mut DecodeWorkspace,
        emit: &[u32],
    ) -> Result<(), TranscriptionError> {
        let stream = runtime.stream();
        for buffer in [
            &mut ws.frame,
            &mut ws.emitted,
            &mut ws.counts,
            &mut ws.faults,
        ] {
            stream.memset_zeros(buffer)?;
        }
        // the blank is NeMo's start symbol, and every fresh row feeds it once
        let blank = vec![(self.classes - 1) as u32; ws.batch];
        stream.memcpy_htod(&blank, &mut ws.label)?;
        stream.memcpy_htod(emit, &mut ws.emit)?;
        Ok(())
    }

    /// Projects `encoder_out` for the joint and records the rows' valid frames
    fn project_encoder(
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
        Ok(())
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

    /// Steps every row still needs at least, given each row's frame cursor: a step
    /// moves a cursor at most `max_advance` frames
    fn steps_left(&self, lens: &[usize], frames: impl IntoIterator<Item = usize>) -> usize {
        lens.iter()
            .zip(frames)
            .map(|(len, frame)| len.saturating_sub(frame).div_ceil(self.max_advance))
            .max()
            .unwrap_or(0)
    }

    /// [`Self::steps_left`] at the device's current cursors; waits for the stream
    fn steps_left_on_device(
        &self,
        runtime: &CudaRuntime,
        ws: &mut DecodeWorkspace,
    ) -> Result<usize, TranscriptionError> {
        ws.host.frame.download(runtime.stream(), &ws.frame)?;
        let frames = ws.host.frame.read()?;
        Ok(self.steps_left(&ws.host_lens, frames.iter().map(|frame| *frame as usize)))
    }

    /// Runs every row to its end like [`Self::run`], replaying `graph`, a capture of
    /// [`STEPS_PER_CHECK`] steps; the first time, with `graph` empty, it runs the
    /// steps eagerly, which also finishes cuBLAS's lazy setup, then captures them
    ///
    /// # Safety
    ///
    /// `graph` must drop before `ws`, whose buffers it records, and `ws` must stay
    /// on the runtime's stream without its buffers being reallocated while `graph`
    /// holds the graph, as [`capture`] requires
    pub(crate) unsafe fn run_graphed(
        &self,
        runtime: &CudaRuntime,
        ws: &mut DecodeWorkspace,
        graph: &mut Option<Graph>,
    ) -> Result<(), TranscriptionError> {
        match graph {
            Some(captured) => {
                self.run_with(runtime, ws, STEPS_PER_CHECK, |_| Ok(captured.launch()?))
            }
            None => {
                self.run(runtime, ws, STEPS_PER_CHECK)?;
                // SAFETY: the caller upholds the contract for `ws` and `graph`, and the
                // steps launch only on the runtime's stream and return errors instead of
                // panicking
                let captured = unsafe {
                    capture(runtime, || {
                        (0..STEPS_PER_CHECK).try_for_each(|_| self.step(runtime, ws))
                    })
                }?;
                *graph = Some(captured);
                Ok(())
            }
        }
    }

    /// Runs steps eagerly, `steps_per_check` at a time, until every row finishes
    fn run(
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
    ///
    /// Between cursor reads, it queues every `advance` the cursors prove some row
    /// still needs ([`Self::steps_left`]), so the GPU runs them back to back and
    /// only waits on the host when the decode may be done. A row that finishes
    /// early idles through the rest, as it would on any batch
    fn run_with(
        &self,
        runtime: &CudaRuntime,
        ws: &mut DecodeWorkspace,
        steps: usize,
        mut advance: impl FnMut(&mut DecodeWorkspace) -> Result<(), TranscriptionError>,
    ) -> Result<(), TranscriptionError> {
        let steps = steps.max(1);
        let bound = ws.max_steps(self.max_symbols);
        let mut done = 0;
        // `start` put every cursor on frame 0
        let mut needed = self.steps_left(&ws.host_lens, std::iter::repeat(0));
        while needed > 0 {
            for _ in 0..needed.div_ceil(steps) {
                if done >= bound {
                    return Err(TranscriptionError::InvalidModelOutput(format!(
                        "decode did not finish within {bound} steps"
                    )));
                }
                advance(ws)?;
                done += steps;
            }
            needed = self.steps_left_on_device(runtime, ws)?;
        }
        Ok(())
    }

    /// Each row's emitted tokens; waits for the stream
    pub(crate) fn output(
        &self,
        runtime: &CudaRuntime,
        ws: &mut DecodeWorkspace,
    ) -> Result<Vec<RawTranscription>, TranscriptionError> {
        let stream = runtime.stream();
        let host = &mut ws.host;
        // queue every copy, so the host waits for the stream once
        host.counts.download(stream, &ws.counts)?;
        host.faults.download(stream, &ws.faults)?;
        host.tokens.download(stream, &ws.tokens)?;
        host.token_frames.download(stream, &ws.token_frames)?;
        host.token_durations.download(stream, &ws.token_durations)?;
        host.confidences.download(stream, &ws.confidences)?;
        // the copies run in order on one stream, so once the last has landed the
        // other reads below find their copies done and do not wait again
        let confidences = host.confidences.read()?;
        let counts = host.counts.read()?;
        let faults = host.faults.read()?;
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

        let tokens = host.tokens.read()?;
        let frames = host.token_frames.read()?;
        let durations = host.token_durations.read()?;
        Ok(counts
            .iter()
            .enumerate()
            .map(|(row, count)| {
                let slots = row * ws.capacity..row * ws.capacity + *count as usize;
                let tokens = tokens[slots.clone()]
                    .iter()
                    .zip(&frames[slots.clone()])
                    .zip(&durations[slots.clone()])
                    .zip(&confidences[slots])
                    .map(|(((&id, &frame), &duration), &confidence)| RawToken {
                        id,
                        frame: frame as usize,
                        duration: duration as usize,
                        confidence,
                    })
                    .collect();
                RawTranscription { tokens }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests;
