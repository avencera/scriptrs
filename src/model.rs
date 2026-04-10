use std::cell::RefCell;

use ndarray::{Array1, Array2, Array3};

use crate::config::TranscriptionConfig;
use crate::constants::{
    DECODER_HIDDEN_SIZE, DECODER_LAYERS, ENCODER_HIDDEN_SIZE, MAX_TOKENS_PER_STEP,
};
use crate::decode::RawTranscription;
use crate::error::TranscriptionError;
use crate::models::ModelBundle;
use crate::vocab::Vocabulary;

#[derive(Debug, Clone)]
pub(crate) struct ParakeetModel {
    inner: ParakeetModelInner,
    blank_id: usize,
    encoder_input: RefCell<EncoderInputBuffer>,
}

impl ParakeetModel {
    pub(crate) fn from_bundle(
        bundle: &ModelBundle,
        vocab: &Vocabulary,
        config: &TranscriptionConfig,
    ) -> Result<Self, TranscriptionError> {
        Ok(Self {
            inner: ParakeetModelInner::from_bundle(bundle)?,
            blank_id: vocab.blank_id(),
            encoder_input: RefCell::new(EncoderInputBuffer::new(
                config.feature_size,
                config.max_feature_frames(),
            )),
        })
    }

    pub(crate) fn transcribe(
        &self,
        features: &Array2<f32>,
        feature_frames: usize,
        target_frames: usize,
    ) -> Result<RawTranscription, TranscriptionError> {
        let (encoder_output, time_steps) =
            self.run_encoder(features, feature_frames, target_frames)?;
        self.greedy_decode(&encoder_output, time_steps)
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
    ) -> Result<RawTranscription, TranscriptionError> {
        let mut state = GreedyDecodeState::new(self.blank_id);

        while state.frame_idx < time_steps {
            state.ensure_decoder_step(&self.inner)?;
            let frame = reshape_encoder_frame(encoder_output, state.frame_idx)?;
            let cached_decoder = state.cached_decoder()?;
            let decision = self.inner.run_joint(&frame, &cached_decoder.decoder_step)?;

            if decision.token_id != self.blank_id {
                let cached_decoder = state.take_cached_decoder()?;
                state.record_emission(
                    decision.token_id,
                    decision.duration_step,
                    decision.confidence,
                    cached_decoder.hidden_state,
                    cached_decoder.cell_state,
                );
            }

            state.advance(decision.token_id, decision.duration_step, self.blank_id);
        }

        Ok(state.into_raw())
    }
}

#[derive(Debug, Clone)]
struct GreedyDecodeState {
    hidden_state: Array3<f32>,
    cell_state: Array3<f32>,
    cached_decoder: Option<CachedDecoderStep>,
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
            cached_decoder: None,
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

    fn ensure_decoder_step(
        &mut self,
        model: &ParakeetModelInner,
    ) -> Result<(), TranscriptionError> {
        if self.cached_decoder.is_some() {
            return Ok(());
        }

        let (targets, target_length) = self.decoder_inputs()?;
        self.cached_decoder = Some(model.run_decoder(
            &targets,
            &target_length,
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

#[derive(Debug, Clone)]
struct JointDecision {
    token_id: usize,
    duration_step: usize,
    confidence: f32,
}

#[derive(Debug, Clone)]
enum ParakeetModelInner {
    #[cfg(target_os = "macos")]
    SplitCoreMl(crate::coreml::ParakeetSplitCoreMlModel),
    #[cfg(not(target_os = "macos"))]
    Unsupported,
}

impl ParakeetModelInner {
    fn from_bundle(bundle: &ModelBundle) -> Result<Self, TranscriptionError> {
        #[cfg(target_os = "macos")]
        {
            Ok(Self::SplitCoreMl(
                crate::coreml::ParakeetSplitCoreMlModel::new(
                    bundle.encoder_dir(),
                    bundle.decoder_dir(),
                    bundle.joint_decision_dir(),
                )?,
            ))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = bundle;
            Err(TranscriptionError::UnsupportedPlatform)
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
            #[cfg(not(target_os = "macos"))]
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
            #[cfg(not(target_os = "macos"))]
            Self::Unsupported => Err(TranscriptionError::UnsupportedPlatform),
        }
    }

    fn run_joint(
        &self,
        encoder_step: &Array3<f32>,
        decoder_step: &Array3<f32>,
    ) -> Result<JointDecision, TranscriptionError> {
        match self {
            #[cfg(target_os = "macos")]
            Self::SplitCoreMl(model) => {
                let output = model.run_joint(encoder_step, decoder_step)?;
                Ok(JointDecision {
                    token_id: output.token_id,
                    duration_step: output.duration_step,
                    confidence: output.token_prob,
                })
            }
            #[cfg(not(target_os = "macos"))]
            Self::Unsupported => Err(TranscriptionError::UnsupportedPlatform),
        }
    }
}
