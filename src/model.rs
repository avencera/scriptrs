use std::cell::RefCell;

use ndarray::{Array1, Array2, Array3};

use crate::config::TranscriptionConfig;
use crate::constants::MAX_TOKENS_PER_STEP;
use crate::decode::RawTranscription;
use crate::deployment::{DecoderDimensions, ParakeetDeployment};
use crate::error::TranscriptionError;
use crate::language::LanguageSelection;
use crate::model_io::{DecoderStepOutputs, JointOutputs, decode_joint_outputs};
use crate::models::{ModelBundle, RuntimeBundle};
use crate::vocab::Vocabulary;

#[derive(Debug, Clone)]
pub(crate) struct ParakeetModel {
    inner: ParakeetModelInner,
    blank_id: usize,
    dims: DecoderDimensions,
    deployment: ParakeetDeployment,
    vocabulary: Vocabulary,
    encoder_input: RefCell<EncoderInputBuffer>,
}

impl ParakeetModel {
    pub(crate) fn from_bundle(
        bundle: &ModelBundle,
        vocab: &Vocabulary,
        config: &TranscriptionConfig,
    ) -> Result<Self, TranscriptionError> {
        let deployment = bundle.deployment().clone();
        config.validate_for(&deployment)?;
        Ok(Self {
            inner: ParakeetModelInner::from_bundle(bundle)?,
            blank_id: vocab.blank_id(),
            dims: deployment.decoder_dimensions(),
            vocabulary: vocab.clone(),
            encoder_input: RefCell::new(EncoderInputBuffer::new(
                config.feature_size,
                config.max_feature_frames(),
            )),
            deployment,
        })
    }

    pub(crate) fn transcribe(
        &self,
        features: &Array2<f32>,
        feature_frames: usize,
        target_frames: usize,
        language: LanguageSelection,
    ) -> Result<RawTranscription, TranscriptionError> {
        let language = self.deployment.select_language(language)?;
        let (encoder_output, time_steps) =
            self.run_encoder(features, feature_frames, target_frames)?;
        self.greedy_decode(&encoder_output, time_steps, language)
    }

    fn run_encoder(
        &self,
        features: &Array2<f32>,
        feature_frames: usize,
        target_frames: usize,
    ) -> Result<(Array3<f32>, usize), TranscriptionError> {
        let mut encoder_input = self.encoder_input.borrow_mut();
        encoder_input.copy_from_features(features, feature_frames, target_frames)?;
        self.inner.run_encoder(
            encoder_input.values(),
            encoder_input.feature_size(),
            encoder_input.target_frames(),
            feature_frames,
        )
    }

    fn greedy_decode(
        &self,
        encoder_output: &Array3<f32>,
        time_steps: usize,
        language: LanguageSelection,
    ) -> Result<RawTranscription, TranscriptionError> {
        let mut state = GreedyDecodeState::new(self.blank_id, self.dims);

        while state.frame_idx < time_steps {
            state.ensure_decoder_step(&self.inner)?;
            state.copy_encoder_frame(encoder_output, self.dims.encoder_hidden())?;
            let cached_decoder = state.cached_decoder()?;
            let raw_decision = self
                .inner
                .run_joint(state.encoder_step(), &cached_decoder.decoder_step)?;
            let decision =
                decode_joint_outputs(&raw_decision, &self.deployment, &self.vocabulary, language)?;

            if decision.token_id != self.blank_id {
                let cached_decoder = state.take_cached_decoder()?;
                state.record_emission(
                    decision.token_id,
                    decision.duration_frames,
                    decision.confidence,
                    cached_decoder.hidden_state,
                    cached_decoder.cell_state,
                );
            }

            state.advance(decision.token_id, decision.duration_frames, self.blank_id);
        }

        Ok(state.into_raw())
    }
}

#[derive(Debug, Clone)]
struct GreedyDecodeState {
    hidden_state: Array3<f32>,
    cell_state: Array3<f32>,
    decoder_targets: Array2<i32>,
    decoder_target_length: Array1<i32>,
    encoder_step: Array3<f32>,
    cached_decoder: Option<CachedDecoderStep>,
    raw: RawTranscription,
    frame_idx: usize,
    emitted_tokens: usize,
    last_token: i32,
}

impl GreedyDecodeState {
    fn new(blank_id: usize, dims: DecoderDimensions) -> Self {
        Self {
            hidden_state: Array3::<f32>::zeros((dims.decoder_layers(), 1, dims.decoder_hidden())),
            cell_state: Array3::<f32>::zeros((dims.decoder_layers(), 1, dims.decoder_hidden())),
            decoder_targets: Array2::<i32>::zeros((1, 1)),
            decoder_target_length: Array1::from_elem(1, 1i32),
            encoder_step: Array3::<f32>::zeros((1, dims.encoder_hidden(), 1)),
            cached_decoder: None,
            raw: RawTranscription::empty(),
            frame_idx: 0,
            emitted_tokens: 0,
            last_token: blank_id as i32,
        }
    }

    fn copy_encoder_frame(
        &mut self,
        encoder_output: &Array3<f32>,
        encoder_hidden: usize,
    ) -> Result<(), TranscriptionError> {
        if self.frame_idx >= encoder_output.shape()[2] {
            return Err(TranscriptionError::InvalidModelOutput(format!(
                "encoder frame index {} exceeded time steps {}",
                self.frame_idx,
                encoder_output.shape()[2]
            )));
        }

        for hidden_idx in 0..encoder_hidden {
            self.encoder_step[[0, hidden_idx, 0]] = encoder_output[[0, hidden_idx, self.frame_idx]];
        }

        Ok(())
    }

    fn ensure_decoder_step(
        &mut self,
        model: &ParakeetModelInner,
    ) -> Result<(), TranscriptionError> {
        if self.cached_decoder.is_some() {
            return Ok(());
        }

        self.decoder_targets[[0, 0]] = self.last_token;
        self.cached_decoder = Some(model.run_decoder(
            &self.decoder_targets,
            &self.decoder_target_length,
            &self.hidden_state,
            &self.cell_state,
        )?);
        Ok(())
    }

    fn cached_decoder(&self) -> Result<&CachedDecoderStep, TranscriptionError> {
        self.cached_decoder.as_ref().ok_or_else(|| {
            TranscriptionError::InvalidModelOutput("decoder cache was not primed".to_owned())
        })
    }

    fn take_cached_decoder(&mut self) -> Result<CachedDecoderStep, TranscriptionError> {
        self.cached_decoder.take().ok_or_else(|| {
            TranscriptionError::InvalidModelOutput("decoder cache was not primed".to_owned())
        })
    }

    fn encoder_step(&self) -> &Array3<f32> {
        &self.encoder_step
    }

    fn record_emission(
        &mut self,
        token_id: usize,
        duration_step: usize,
        confidence: f32,
        hidden_state: Array3<f32>,
        cell_state: Array3<f32>,
    ) {
        self.hidden_state = hidden_state;
        self.cell_state = cell_state;
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

#[derive(Debug, Clone)]
struct CachedDecoderStep {
    decoder_step: Array3<f32>,
    hidden_state: Array3<f32>,
    cell_state: Array3<f32>,
}

impl From<DecoderStepOutputs> for CachedDecoderStep {
    fn from(outputs: DecoderStepOutputs) -> Self {
        Self {
            decoder_step: outputs.decoder_step,
            hidden_state: outputs.hidden_state,
            cell_state: outputs.cell_state,
        }
    }
}

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

    fn copy_from_features(
        &mut self,
        features: &Array2<f32>,
        feature_frames: usize,
        target_frames: usize,
    ) -> Result<(), TranscriptionError> {
        if features.shape()[1] != self.feature_size {
            return Err(TranscriptionError::InvalidModelOutput(format!(
                "feature size {} did not match encoder input {}",
                features.shape()[1],
                self.feature_size
            )));
        }
        if self.target_frames != target_frames {
            self.values.resize(self.feature_size * target_frames, 0.0);
            self.target_frames = target_frames;
            self.last_feature_frames = 0;
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

    fn feature_size(&self) -> usize {
        self.feature_size
    }

    fn target_frames(&self) -> usize {
        self.target_frames
    }
}

/// Runtime port behind the shared greedy decoder
///
/// Every variant moves the same tensors; encoder invocation, decoder-state
/// updates, blank handling, duration advancement, and token caps stay in
/// [`ParakeetModel`] and [`GreedyDecodeState`].
#[derive(Debug, Clone)]
enum ParakeetModelInner {
    // boxed: the Core ML model caches input shapes inline and would dwarf the
    // shared ONNX session handle
    #[cfg(target_os = "macos")]
    SplitCoreMl(Box<crate::coreml::ParakeetSplitCoreMlModel>),
    #[cfg(feature = "onnx-cuda")]
    OnnxCuda(crate::onnx::ParakeetOnnxModel),
    #[cfg(not(any(target_os = "macos", feature = "onnx-cuda")))]
    Unsupported,
}

impl ParakeetModelInner {
    fn from_bundle(bundle: &ModelBundle) -> Result<Self, TranscriptionError> {
        match bundle.runtime() {
            RuntimeBundle::CoreMl(layout) => {
                #[cfg(target_os = "macos")]
                {
                    Ok(Self::SplitCoreMl(Box::new(
                        crate::coreml::ParakeetSplitCoreMlModel::new(
                            layout.encoder_dir(),
                            layout.decoder_dir(),
                            layout.joint_decision_dir(),
                            bundle.deployment().model_io(),
                            bundle.deployment().decoder_dimensions(),
                        )?,
                    )))
                }
                #[cfg(not(target_os = "macos"))]
                {
                    let _ = layout;
                    Err(TranscriptionError::UnsupportedPlatform)
                }
            }
            #[cfg(feature = "onnx-cuda")]
            RuntimeBundle::OnnxCuda(onnx) => Ok(Self::OnnxCuda(onnx.parakeet_model())),
        }
    }

    fn run_encoder(
        &self,
        input: &[f32],
        feature_size: usize,
        target_frames: usize,
        feature_frames: usize,
    ) -> Result<(Array3<f32>, usize), TranscriptionError> {
        match self {
            #[cfg(target_os = "macos")]
            Self::SplitCoreMl(model) => {
                model.run_encoder(input, feature_size, target_frames, &[feature_frames as i32])
            }
            #[cfg(feature = "onnx-cuda")]
            Self::OnnxCuda(model) => {
                model.run_encoder(input, feature_size, target_frames, feature_frames)
            }
            #[cfg(not(any(target_os = "macos", feature = "onnx-cuda")))]
            Self::Unsupported => Err(TranscriptionError::UnsupportedPlatform),
        }
    }

    fn run_decoder(
        &self,
        targets: &Array2<i32>,
        target_length: &Array1<i32>,
        hidden_state: &Array3<f32>,
        cell_state: &Array3<f32>,
    ) -> Result<CachedDecoderStep, TranscriptionError> {
        match self {
            #[cfg(target_os = "macos")]
            Self::SplitCoreMl(model) => {
                let output = model.run_decoder(targets, target_length, hidden_state, cell_state)?;
                Ok(CachedDecoderStep {
                    decoder_step: output.decoder_step,
                    hidden_state: output.hidden_state,
                    cell_state: output.cell_state,
                })
            }
            #[cfg(feature = "onnx-cuda")]
            Self::OnnxCuda(model) => model
                .run_decoder(targets, target_length, hidden_state, cell_state)
                .map(CachedDecoderStep::from),
            #[cfg(not(any(target_os = "macos", feature = "onnx-cuda")))]
            Self::Unsupported => Err(TranscriptionError::UnsupportedPlatform),
        }
    }

    fn run_joint(
        &self,
        encoder_step: &Array3<f32>,
        decoder_step: &Array3<f32>,
    ) -> Result<JointOutputs, TranscriptionError> {
        match self {
            #[cfg(target_os = "macos")]
            Self::SplitCoreMl(model) => model.run_joint(encoder_step, decoder_step),
            #[cfg(feature = "onnx-cuda")]
            Self::OnnxCuda(model) => model.run_joint(encoder_step, decoder_step),
            #[cfg(not(any(target_os = "macos", feature = "onnx-cuda")))]
            Self::Unsupported => Err(TranscriptionError::UnsupportedPlatform),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::GreedyDecodeState;
    use crate::constants::MAX_TOKENS_PER_STEP;
    use crate::deployment::ParakeetDeployment;
    use crate::language::{LanguageSelection, ParakeetLanguage};
    use crate::model_io::{JointOutputs, decode_joint_outputs};
    use crate::vocab::Vocabulary;

    #[test]
    fn greedy_state_advances_by_mapped_duration() {
        let dims = ParakeetDeployment::v3().decoder_dimensions();
        let mut state = GreedyDecodeState::new(4, dims);
        state.advance(1, 3, 4);
        assert_eq!(state.frame_idx, 3);
    }

    #[test]
    fn greedy_state_force_advances_after_max_tokens() {
        let dims = ParakeetDeployment::v3().decoder_dimensions();
        let mut state = GreedyDecodeState::new(4, dims);
        state.emitted_tokens = MAX_TOKENS_PER_STEP;
        state.advance(1, 0, 4);
        assert_eq!(state.frame_idx, 1);
    }

    #[test]
    fn fixture_joint_path_does_not_need_coreml() {
        let deployment = ParakeetDeployment::v3();
        let vocab = Vocabulary::from_tokens(&["<unk>", "cat", "кот", "<blk>"]);
        let outputs = JointOutputs {
            token_id: 2,
            token_prob: 0.8,
            duration_bin: 2,
            top_k_ids: Some(vec![2, 1]),
            top_k_logits: Some(vec![5.0, 1.0]),
        };
        let decoded = decode_joint_outputs(
            &outputs,
            &deployment,
            &vocab,
            LanguageSelection::Language(ParakeetLanguage::Polish),
        )
        .unwrap();
        assert_eq!(decoded.token_id, 1);
        assert_eq!(decoded.duration_frames, 2);
    }
}
