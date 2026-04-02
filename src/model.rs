use ndarray::{Array1, Array2, Array3};

use crate::constants::{
    DECODER_HIDDEN_SIZE, DECODER_LAYERS, ENCODER_HIDDEN_SIZE, MAX_TOKENS_PER_STEP,
    TOKEN_DURATION_CLASSES,
};
use crate::decode::RawTranscription;
use crate::error::TranscriptionError;
use crate::models::ModelBundle;
use crate::vocab::Vocabulary;

#[derive(Debug, Clone)]
pub(crate) struct ParakeetModel {
    inner: ParakeetModelInner,
    blank_id: usize,
}

impl ParakeetModel {
    pub(crate) fn from_bundle(
        bundle: &ModelBundle,
        vocab: &Vocabulary,
    ) -> Result<Self, TranscriptionError> {
        Ok(Self {
            inner: ParakeetModelInner::from_bundle(bundle)?,
            blank_id: vocab.blank_id(),
        })
    }

    pub(crate) fn transcribe(
        &self,
        features: &Array2<f32>,
    ) -> Result<RawTranscription, TranscriptionError> {
        let (encoder_output, time_steps) = self.run_encoder(features)?;
        self.greedy_decode(&encoder_output, time_steps)
    }

    fn run_encoder(
        &self,
        features: &Array2<f32>,
    ) -> Result<(Array3<f32>, usize), TranscriptionError> {
        let feature_size = features.shape()[1];
        let time_steps = features.shape()[0];
        let input = features
            .t()
            .to_shape((1, feature_size, time_steps))
            .map_err(|error| {
                TranscriptionError::InvalidModelOutput(format!(
                    "failed to reshape encoder input: {error}"
                ))
            })?
            .to_owned();
        self.inner.run_encoder(input, vec![time_steps as i32])
    }

    fn greedy_decode(
        &self,
        encoder_output: &Array3<f32>,
        time_steps: usize,
    ) -> Result<RawTranscription, TranscriptionError> {
        let mut state = GreedyDecodeState::new(self.blank_id);

        while state.frame_idx < time_steps {
            let frame = reshape_encoder_frame(encoder_output, state.frame_idx)?;
            let (targets, target_length) = state.decoder_inputs()?;
            let decoder_output = self.inner.run_decoder(
                frame,
                targets,
                target_length,
                state.hidden_state.clone(),
                state.cell_state.clone(),
            )?;

            let token_id = max_logit_index(&decoder_output.vocab_logits).unwrap_or(self.blank_id);
            let duration_step = max_logit_index(&decoder_output.duration_logits).unwrap_or(0);

            if token_id != self.blank_id {
                let confidence = softmax_max(&decoder_output.vocab_logits);
                state.record_emission(token_id, duration_step, confidence, decoder_output);
            }

            state.advance(token_id, duration_step, self.blank_id);
        }

        Ok(state.into_raw())
    }
}

#[derive(Debug, Clone)]
struct GreedyDecodeState {
    hidden_state: Array3<f32>,
    cell_state: Array3<f32>,
    raw: RawTranscription,
    frame_idx: usize,
    emitted_tokens: usize,
    last_token: i32,
}

impl GreedyDecodeState {
    fn new(blank_id: usize) -> Self {
        Self {
            hidden_state: Array3::<f32>::zeros((DECODER_LAYERS, 1, DECODER_HIDDEN_SIZE)),
            cell_state: Array3::<f32>::zeros((DECODER_LAYERS, 1, DECODER_HIDDEN_SIZE)),
            raw: RawTranscription::empty(),
            frame_idx: 0,
            emitted_tokens: 0,
            last_token: blank_id as i32,
        }
    }

    fn decoder_inputs(&self) -> Result<(Array2<i32>, Array1<i32>), TranscriptionError> {
        let targets = Array2::from_shape_vec((1, 1), vec![self.last_token]).map_err(|error| {
            TranscriptionError::InvalidModelOutput(format!("failed to shape targets: {error}"))
        })?;
        Ok((targets, Array1::from_vec(vec![1i32])))
    }

    fn record_emission(
        &mut self,
        token_id: usize,
        duration_step: usize,
        confidence: f32,
        decoder_output: DecoderOutput,
    ) {
        self.hidden_state = decoder_output.hidden_state;
        self.cell_state = decoder_output.cell_state;
        self.raw.token_ids.push(token_id as u32);
        self.raw.frame_indices.push(self.frame_idx);
        self.raw.durations.push(duration_step);
        self.raw.confidences.push(confidence);
        self.last_token = token_id as i32;
        self.emitted_tokens += 1;
    }

    fn advance(&mut self, token_id: usize, duration_step: usize, blank_id: usize) {
        if duration_step > 0 {
            self.frame_idx += duration_step;
            self.emitted_tokens = 0;
            return;
        }

        if token_id == blank_id || self.emitted_tokens >= MAX_TOKENS_PER_STEP {
            self.frame_idx += 1;
            self.emitted_tokens = 0;
        }
    }

    fn into_raw(self) -> RawTranscription {
        self.raw
    }
}

fn reshape_encoder_frame(
    encoder_output: &Array3<f32>,
    frame_idx: usize,
) -> Result<Array3<f32>, TranscriptionError> {
    let frame = encoder_output
        .slice(ndarray::s![0, .., frame_idx])
        .to_owned()
        .to_shape((1, ENCODER_HIDDEN_SIZE, 1))
        .map_err(|error| {
            TranscriptionError::InvalidModelOutput(format!(
                "failed to reshape encoder frame: {error}"
            ))
        })?
        .to_owned();
    Ok(frame)
}

fn max_logit_index(logits: &[f32]) -> Option<usize> {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
}

fn softmax_max(logits: &[f32]) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum = logits
        .iter()
        .map(|logit| (*logit - max_logit).exp())
        .sum::<f32>();
    if exp_sum == 0.0 {
        return 0.0;
    }
    logits
        .iter()
        .map(|logit| (*logit - max_logit).exp() / exp_sum)
        .fold(0.0, f32::max)
}

#[derive(Debug, Clone)]
struct DecoderOutput {
    vocab_logits: Vec<f32>,
    duration_logits: Vec<f32>,
    hidden_state: Array3<f32>,
    cell_state: Array3<f32>,
}

#[derive(Debug, Clone)]
enum ParakeetModelInner {
    #[cfg(target_os = "macos")]
    CoreMl(crate::coreml::ParakeetCoreMlModel),
    #[cfg(not(target_os = "macos"))]
    Unsupported,
}

impl ParakeetModelInner {
    fn from_bundle(bundle: &ModelBundle) -> Result<Self, TranscriptionError> {
        #[cfg(target_os = "macos")]
        {
            Ok(Self::CoreMl(crate::coreml::ParakeetCoreMlModel::new(
                bundle.encoder_dir(),
                bundle.decoder_joint_dir(),
            )?))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = bundle;
            Err(TranscriptionError::UnsupportedPlatform)
        }
    }

    fn run_encoder(
        &self,
        input: Array3<f32>,
        lengths: Vec<i32>,
    ) -> Result<(Array3<f32>, usize), TranscriptionError> {
        match self {
            #[cfg(target_os = "macos")]
            Self::CoreMl(model) => model.run_encoder(input, lengths),
            #[cfg(not(target_os = "macos"))]
            Self::Unsupported => Err(TranscriptionError::UnsupportedPlatform),
        }
    }

    fn run_decoder(
        &self,
        frame: Array3<f32>,
        targets: Array2<i32>,
        target_length: Array1<i32>,
        hidden_state: Array3<f32>,
        cell_state: Array3<f32>,
    ) -> Result<DecoderOutput, TranscriptionError> {
        match self {
            #[cfg(target_os = "macos")]
            Self::CoreMl(model) => {
                let output =
                    model.run_decoder(frame, targets, target_length, hidden_state, cell_state)?;
                let vocab_logits = output
                    .logits
                    .iter()
                    .take(output.vocab_size)
                    .copied()
                    .collect();
                let duration_logits = output
                    .logits
                    .iter()
                    .skip(output.vocab_size)
                    .take(TOKEN_DURATION_CLASSES)
                    .copied()
                    .collect();
                Ok(DecoderOutput {
                    vocab_logits,
                    duration_logits,
                    hidden_state: output.hidden_state,
                    cell_state: output.cell_state,
                })
            }
            #[cfg(not(target_os = "macos"))]
            Self::Unsupported => Err(TranscriptionError::UnsupportedPlatform),
        }
    }
}
