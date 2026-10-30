use std::cell::RefCell;

use ndarray::{Array1, Array2, Array3};

use crate::backend::{ChunkDecode, DecodePolicy, EncoderChunk};
use crate::decode::{RawToken, RawTranscription};
use crate::deployment::DecoderDimensions;
use crate::error::TranscriptionError;
use crate::language::TopKCandidates;
use crate::model_io::{DecoderStepOutputs, JointOutputs};

/// Batch-1 encoder, decoder and joint calls behind the host greedy loop
///
/// The loop owns encoder input padding, decoder-state caching, blank handling,
/// duration advancement and the per-frame token cap; a runtime only moves
/// tensors.
pub(crate) trait StepRuntime {
    /// Encode `[1, features, target_frames]` mel input with `feature_frames`
    /// valid frames into `[1, hidden, time]` and its valid time steps
    fn run_encoder(
        &self,
        input: &[f32],
        feature_size: usize,
        target_frames: usize,
        feature_frames: usize,
    ) -> Result<(Array3<f32>, usize), TranscriptionError>;

    /// Advance the prediction network by one token
    fn run_decoder(
        &self,
        targets: &Array2<i32>,
        target_length: &Array1<i32>,
        hidden_state: &Array3<f32>,
        cell_state: &Array3<f32>,
    ) -> Result<DecoderStepOutputs, TranscriptionError>;

    /// Joint decision for one encoder frame and decoder output
    fn run_joint(
        &self,
        encoder_step: &Array3<f32>,
        decoder_step: &Array3<f32>,
    ) -> Result<JointOutputs, TranscriptionError>;
}

/// Host greedy TDT loop shared by step runtimes
///
/// Chunks run one at a time at batch 1, with one joint call per decode step
/// and one decoder call per emitted token.
#[derive(Debug)]
pub(crate) struct HostLoop {
    dims: DecoderDimensions,
    encoder_input: RefCell<EncoderInputBuffer>,
}

impl Clone for HostLoop {
    fn clone(&self) -> Self {
        let input = self.encoder_input.borrow();
        Self {
            dims: self.dims,
            encoder_input: RefCell::new(EncoderInputBuffer::new(
                input.feature_size,
                input.target_frames,
            )),
        }
    }
}

impl HostLoop {
    /// Loop for a runtime with a fixed `[1, feature_size, target_frames]` encoder input
    pub(crate) fn new(dims: DecoderDimensions, feature_size: usize, target_frames: usize) -> Self {
        Self {
            dims,
            encoder_input: RefCell::new(EncoderInputBuffer::new(feature_size, target_frames)),
        }
    }

    pub(crate) fn transcribe_chunks(
        &self,
        runtime: &impl StepRuntime,
        chunks: &[EncoderChunk<'_>],
        policy: &DecodePolicy<'_>,
    ) -> Result<Vec<ChunkDecode>, TranscriptionError> {
        chunks
            .iter()
            .map(|chunk| self.transcribe_chunk(runtime, chunk, policy))
            .collect()
    }

    fn transcribe_chunk(
        &self,
        runtime: &impl StepRuntime,
        chunk: &EncoderChunk<'_>,
        policy: &DecodePolicy<'_>,
    ) -> Result<ChunkDecode, TranscriptionError> {
        let (encoder_output, time_steps) = {
            let mut input = self.encoder_input.borrow_mut();
            input.copy_from_chunk(chunk)?;
            runtime.run_encoder(
                input.values(),
                input.feature_size,
                input.target_frames,
                chunk.frames(),
            )?
        };
        if time_steps > encoder_output.shape()[2] {
            return Err(TranscriptionError::InvalidModelOutput(format!(
                "encoder reported {time_steps} valid steps for {} output frames",
                encoder_output.shape()[2]
            )));
        }

        let mut state = GreedyDecodeState::new(policy.blank_id(), self.dims, policy.keep_top_k());
        while state.frame_idx < time_steps {
            state.ensure_decoder_step(runtime)?;
            state.copy_encoder_frame(&encoder_output, self.dims.encoder_hidden());
            let cached_decoder = state.cached_decoder()?;
            let joint = runtime.run_joint(&state.encoder_step, &cached_decoder.decoder_step)?;
            let decision = policy.decide(&joint)?;

            if decision.token_id != policy.blank_id() {
                let cached_decoder = state.take_cached_decoder()?;
                state.record_emission(
                    decision.token_id,
                    decision.duration_frames,
                    decision.confidence,
                    joint.top_k,
                    cached_decoder,
                )?;
            }

            state.advance(
                decision.token_id,
                decision.duration_frames,
                policy.blank_id(),
                policy.max_symbols_per_frame(),
            );
        }

        Ok(state.into_decode())
    }
}

#[derive(Debug, Clone)]
struct GreedyDecodeState {
    hidden_state: Array3<f32>,
    cell_state: Array3<f32>,
    decoder_targets: Array2<i32>,
    decoder_target_length: Array1<i32>,
    encoder_step: Array3<f32>,
    cached_decoder: Option<DecoderStepOutputs>,
    raw: RawTranscription,
    top_k: Option<Vec<TopKCandidates>>,
    frame_idx: usize,
    emitted_tokens: usize,
    last_token: i32,
}

impl GreedyDecodeState {
    fn new(blank_id: usize, dims: DecoderDimensions, keep_top_k: bool) -> Self {
        Self {
            hidden_state: Array3::<f32>::zeros((dims.decoder_layers(), 1, dims.decoder_hidden())),
            cell_state: Array3::<f32>::zeros((dims.decoder_layers(), 1, dims.decoder_hidden())),
            decoder_targets: Array2::<i32>::zeros((1, 1)),
            decoder_target_length: Array1::from_elem(1, 1i32),
            encoder_step: Array3::<f32>::zeros((1, dims.encoder_hidden(), 1)),
            cached_decoder: None,
            raw: RawTranscription::empty(),
            top_k: keep_top_k.then(Vec::new),
            frame_idx: 0,
            emitted_tokens: 0,
            last_token: blank_id as i32,
        }
    }

    fn copy_encoder_frame(&mut self, encoder_output: &Array3<f32>, encoder_hidden: usize) {
        for hidden_idx in 0..encoder_hidden {
            self.encoder_step[[0, hidden_idx, 0]] = encoder_output[[0, hidden_idx, self.frame_idx]];
        }
    }

    fn ensure_decoder_step(
        &mut self,
        runtime: &impl StepRuntime,
    ) -> Result<(), TranscriptionError> {
        if self.cached_decoder.is_some() {
            return Ok(());
        }

        self.decoder_targets[[0, 0]] = self.last_token;
        self.cached_decoder = Some(runtime.run_decoder(
            &self.decoder_targets,
            &self.decoder_target_length,
            &self.hidden_state,
            &self.cell_state,
        )?);
        Ok(())
    }

    fn cached_decoder(&self) -> Result<&DecoderStepOutputs, TranscriptionError> {
        self.cached_decoder.as_ref().ok_or_else(|| {
            TranscriptionError::InvalidModelOutput("decoder cache was not primed".to_owned())
        })
    }

    fn take_cached_decoder(&mut self) -> Result<DecoderStepOutputs, TranscriptionError> {
        self.cached_decoder.take().ok_or_else(|| {
            TranscriptionError::InvalidModelOutput("decoder cache was not primed".to_owned())
        })
    }

    fn record_emission(
        &mut self,
        token_id: usize,
        duration_step: usize,
        confidence: f32,
        top_k: Option<TopKCandidates>,
        decoder: DecoderStepOutputs,
    ) -> Result<(), TranscriptionError> {
        if let Some(kept) = &mut self.top_k {
            kept.push(top_k.ok_or_else(|| {
                TranscriptionError::InvalidModelOutput(
                    "top-k candidates were requested but the joint returned none".to_owned(),
                )
            })?);
        }

        self.hidden_state = decoder.hidden_state;
        self.cell_state = decoder.cell_state;
        self.raw.tokens.push(RawToken {
            id: token_id as u32,
            frame: self.frame_idx,
            duration: duration_step,
            confidence,
        });
        self.last_token = token_id as i32;
        self.emitted_tokens += 1;
        Ok(())
    }

    fn advance(
        &mut self,
        token_id: usize,
        duration_step: usize,
        blank_id: usize,
        max_symbols: usize,
    ) {
        if duration_step > 0 {
            self.frame_idx += duration_step;
            self.emitted_tokens = 0;
            return;
        }

        if token_id == blank_id || self.emitted_tokens >= max_symbols {
            self.frame_idx += 1;
            self.emitted_tokens = 0;
        }
    }

    fn into_decode(self) -> ChunkDecode {
        ChunkDecode {
            raw: self.raw,
            top_k: self.top_k,
        }
    }
}

/// Zero-padded `[1, features, target_frames]` encoder input reused across chunks
#[derive(Debug, Clone)]
struct EncoderInputBuffer {
    values: Vec<f32>,
    feature_size: usize,
    target_frames: usize,
    last_feature_frames: usize,
}

impl EncoderInputBuffer {
    fn new(feature_size: usize, target_frames: usize) -> Self {
        Self {
            values: vec![0.0; feature_size * target_frames],
            feature_size,
            target_frames,
            last_feature_frames: 0,
        }
    }

    fn copy_from_chunk(&mut self, chunk: &EncoderChunk<'_>) -> Result<(), TranscriptionError> {
        let features = chunk.features();
        let feature_frames = chunk.frames();
        if features.ncols() != self.feature_size {
            return Err(TranscriptionError::InvalidModelOutput(format!(
                "feature size {} did not match encoder input {}",
                features.ncols(),
                self.feature_size
            )));
        }
        if feature_frames > self.target_frames {
            return Err(TranscriptionError::InvalidModelOutput(format!(
                "feature frame count {feature_frames} exceeded encoder target {}",
                self.target_frames
            )));
        }

        for feature_idx in 0..self.feature_size {
            let base = feature_idx * self.target_frames;
            for frame_idx in 0..feature_frames {
                self.values[base + frame_idx] = features[[frame_idx, feature_idx]];
            }
            if feature_frames < self.last_feature_frames {
                self.values[base + feature_frames..base + self.last_feature_frames].fill(0.0);
            }
        }

        self.last_feature_frames = feature_frames;
        Ok(())
    }

    fn values(&self) -> &[f32] {
        &self.values
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use ndarray::{Array1, Array2, Array3};

    use super::{HostLoop, StepRuntime};
    use crate::backend::{DecodePolicy, EncoderChunk};
    use crate::deployment::ParakeetDeployment;
    use crate::error::TranscriptionError;
    use crate::language::{LanguageSelection, TopKCandidates};
    use crate::model_io::{DecoderStepOutputs, JointOutputs};
    use crate::vocab::Vocabulary;

    /// Scripted runtime: the joint answers from a queue and records the frame
    /// and decoder state it was called with
    struct ScriptedRuntime {
        time_steps: usize,
        joint: RefCell<Vec<(usize, usize)>>,
        joint_calls: RefCell<Vec<(usize, f32)>>,
        decoder_tokens: RefCell<Vec<i32>>,
    }

    impl ScriptedRuntime {
        /// `joint` is consumed front to back as `(token, duration bin)`
        fn new(time_steps: usize, joint: &[(usize, usize)]) -> Self {
            Self {
                time_steps,
                joint: RefCell::new(joint.iter().rev().copied().collect()),
                joint_calls: RefCell::new(Vec::new()),
                decoder_tokens: RefCell::new(Vec::new()),
            }
        }
    }

    impl StepRuntime for ScriptedRuntime {
        fn run_encoder(
            &self,
            _input: &[f32],
            _feature_size: usize,
            _target_frames: usize,
            _feature_frames: usize,
        ) -> Result<(Array3<f32>, usize), TranscriptionError> {
            // frame t carries t in every hidden unit so joint calls reveal the frame
            let hidden = ParakeetDeployment::v3()
                .decoder_dimensions()
                .encoder_hidden();
            let encoder = Array3::from_shape_fn((1, hidden, self.time_steps), |(_, _, t)| t as f32);
            Ok((encoder, self.time_steps))
        }

        fn run_decoder(
            &self,
            targets: &Array2<i32>,
            _target_length: &Array1<i32>,
            hidden_state: &Array3<f32>,
            cell_state: &Array3<f32>,
        ) -> Result<DecoderStepOutputs, TranscriptionError> {
            self.decoder_tokens.borrow_mut().push(targets[[0, 0]]);
            // the decoder output carries the token it was fed
            let dims = ParakeetDeployment::v3().decoder_dimensions();
            Ok(DecoderStepOutputs {
                decoder_step: Array3::from_elem(
                    (1, dims.decoder_hidden(), 1),
                    targets[[0, 0]] as f32,
                ),
                hidden_state: hidden_state + 1.0,
                cell_state: cell_state.clone(),
            })
        }

        fn run_joint(
            &self,
            encoder_step: &Array3<f32>,
            decoder_step: &Array3<f32>,
        ) -> Result<JointOutputs, TranscriptionError> {
            self.joint_calls
                .borrow_mut()
                .push((encoder_step[[0, 0, 0]] as usize, decoder_step[[0, 0, 0]]));
            let (token_id, duration_bin) = self.joint.borrow_mut().pop().unwrap_or((4, 1));
            Ok(JointOutputs {
                token_id,
                token_prob: 0.5,
                duration_bin,
                top_k: Some(TopKCandidates::from_parallel(vec![token_id], vec![1.0]).unwrap()),
            })
        }
    }

    fn run(runtime: &ScriptedRuntime, keep_top_k: bool) -> crate::backend::ChunkDecode {
        let deployment = ParakeetDeployment::v3();
        let vocab = Vocabulary::from_tokens(&["<unk>", "a", "b", "c", "<blk>"]);
        let mut policy = DecodePolicy::new(&deployment, &vocab, LanguageSelection::Auto).unwrap();
        if keep_top_k {
            policy = policy.with_top_k();
        }
        let host = HostLoop::new(deployment.decoder_dimensions(), 128, 16);
        let features = Array2::<f32>::zeros((10, 128));
        let chunks = [EncoderChunk::new(features.view())];
        host.transcribe_chunks(runtime, &chunks, &policy)
            .unwrap()
            .pop()
            .unwrap()
    }

    #[test]
    fn loop_follows_durations_and_only_feeds_emitted_tokens_to_the_decoder() {
        // frame 0: `a` dur 0, `b` dur 2 -> frame 2: blank dur 0 (forced +1)
        // -> frame 3: `c` dur 1 -> frame 4: blank dur 2 -> frame 6 == end
        let runtime = ScriptedRuntime::new(6, &[(1, 0), (2, 2), (4, 0), (3, 1), (4, 2)]);
        let decode = run(&runtime, true);

        assert_eq!(decode.raw.ids(), vec![1, 2, 3]);
        assert_eq!(decode.raw.frames(), vec![0, 0, 3]);
        assert_eq!(decode.raw.durations(), vec![0, 2, 1]);
        assert_eq!(decode.top_k.map(|top_k| top_k.len()), Some(3));

        // decoder runs for SOS (blank) and each emitted token, never for blanks
        assert_eq!(*runtime.decoder_tokens.borrow(), vec![4, 1, 2, 3]);
        let calls = runtime.joint_calls.borrow();
        assert_eq!(
            *calls,
            vec![(0, 4.0), (0, 1.0), (2, 2.0), (3, 2.0), (4, 3.0)]
        );
    }

    #[test]
    fn frame_advances_after_max_symbols_zero_duration_tokens() {
        let runtime = ScriptedRuntime::new(2, &[(1, 0); 12]);
        let decode = run(&runtime, false);

        // ten tokens at frame 0, then the cap forces frame 1, where two more arrive
        let at_frame_zero = decode
            .raw
            .tokens
            .iter()
            .filter(|token| token.frame == 0)
            .count();
        assert_eq!(at_frame_zero, 10);
        assert!(decode.top_k.is_none());
    }
}
