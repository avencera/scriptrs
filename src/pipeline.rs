use crate::config::TranscriptionConfig;
use crate::constants::SAMPLES_PER_ENCODER_FRAME;
use crate::decode::{ParakeetTdtDecoder, RawTranscription};
use crate::error::TranscriptionError;
use crate::frontend::ParakeetFeatureExtractor;
use crate::model::ParakeetModel;
use crate::models::ModelBundle;
use crate::types::TranscriptionResult;
use crate::vocab::Vocabulary;
use ndarray::{Array2, s};

/// Single-chunk Parakeet v2 transcription pipeline
///
/// This is the base `scriptrs` entry point for audio that already fits inside a
/// single Parakeet window. It expects mono 16kHz audio samples as `&[f32]`.
///
/// If the input is longer than [`TranscriptionConfig::max_audio_samples`], this
/// pipeline returns [`TranscriptionError::AudioTooLong`] instead of splitting it.
/// Use `LongFormTranscriptionPipeline` with the `long-form` feature if you want
/// `scriptrs` to own chunk planning internally.
#[derive(Debug, Clone)]
pub struct TranscriptionPipeline {
    bundle: ModelBundle,
    config: TranscriptionConfig,
    extractor: ParakeetFeatureExtractor,
    decoder: ParakeetTdtDecoder,
    model: ParakeetModel,
}

impl TranscriptionPipeline {
    /// Build a transcription pipeline from a local model directory
    ///
    /// The directory must contain the Parakeet runtime bundle expected by
    /// [`ModelBundle::from_dir`].
    pub fn from_dir(models_dir: impl Into<std::path::PathBuf>) -> Result<Self, TranscriptionError> {
        let bundle = ModelBundle::from_dir(models_dir);
        Self::from_bundle(bundle)
    }

    /// Build a transcription pipeline from a resolved model bundle
    pub fn from_bundle(bundle: ModelBundle) -> Result<Self, TranscriptionError> {
        bundle.validate_base()?;
        let encoder_spec = bundle.encoder_spec()?;
        let config = TranscriptionConfig::from_encoder_max_frames(encoder_spec.max_frames);
        let vocab = Vocabulary::from_file(bundle.vocab_path())?;
        let model = ParakeetModel::from_bundle(&bundle, &vocab, encoder_spec)?;
        Ok(Self {
            extractor: ParakeetFeatureExtractor::new(&config),
            decoder: ParakeetTdtDecoder::new(vocab),
            config,
            model,
            bundle,
        })
    }

    #[cfg(feature = "online")]
    /// Download models and build a transcription pipeline
    ///
    /// With the default configuration this resolves models from
    /// `avencera/scriptrs-models` on Hugging Face. Set `SCRIPTRS_MODELS_DIR` to
    /// force a local bundle or `SCRIPTRS_MODELS_REPO` to override the repo.
    pub fn from_pretrained() -> Result<Self, TranscriptionError> {
        let bundle = ModelBundle::from_pretrained().map_err(|error| {
            TranscriptionError::CoreMl(format!("model download failed: {error}"))
        })?;
        Self::from_bundle(bundle)
    }

    /// Transcribe a single chunk of audio
    ///
    /// `audio` must be mono 16kHz samples. Empty input returns
    /// [`TranscriptionError::EmptyAudio`]. Oversized input returns
    /// [`TranscriptionError::AudioTooLong`].
    pub fn run(&self, audio: &[f32]) -> Result<TranscriptionResult, TranscriptionError> {
        self.run_with_config(audio, &self.config)
    }

    /// Transcribe a single chunk of audio with an explicit config
    ///
    /// This is mainly useful if you want to reuse the same pipeline with a
    /// tweaked [`TranscriptionConfig`] instead of the default frontend settings.
    pub fn run_with_config(
        &self,
        audio: &[f32],
        config: &TranscriptionConfig,
    ) -> Result<TranscriptionResult, TranscriptionError> {
        let raw = self.transcribe_raw(audio, 0, 0, config)?;
        let duration_seconds = audio.len() as f64 / config.sample_rate as f64;
        Ok(self.decoder.decode(&raw, duration_seconds))
    }

    /// Return the default pipeline config
    pub fn config(&self) -> &TranscriptionConfig {
        &self.config
    }

    /// Return the resolved model bundle
    pub fn bundle(&self) -> &ModelBundle {
        &self.bundle
    }

    #[cfg(feature = "long-form")]
    pub(crate) fn decode_raw(
        &self,
        raw: &RawTranscription,
        duration_seconds: f64,
    ) -> TranscriptionResult {
        self.decoder.decode(raw, duration_seconds)
    }

    pub(crate) fn transcribe_raw(
        &self,
        audio: &[f32],
        global_sample_offset: usize,
        context_samples: usize,
        config: &TranscriptionConfig,
    ) -> Result<RawTranscription, TranscriptionError> {
        if audio.is_empty() {
            return Err(TranscriptionError::EmptyAudio);
        }
        if audio.len() > config.max_audio_samples {
            return Err(TranscriptionError::AudioTooLong {
                max_seconds: config.max_duration_seconds(),
                actual_seconds: audio.len() as f64 / config.sample_rate as f64,
            });
        }

        let features = self.extractor.extract(audio)?;
        let feature_frames = features.shape()[0];
        let padded_features = pad_features(features, config.max_feature_frames());
        let mut raw = self.model.transcribe(&padded_features, feature_frames)?;
        apply_time_offsets(
            &mut raw,
            global_sample_offset / SAMPLES_PER_ENCODER_FRAME,
            context_samples / SAMPLES_PER_ENCODER_FRAME,
        );
        Ok(raw)
    }
}

fn pad_features(features: Array2<f32>, target_frames: usize) -> Array2<f32> {
    let current_frames = features.shape()[0];
    if current_frames >= target_frames {
        return features;
    }

    let feature_size = features.shape()[1];
    let mut padded = Array2::<f32>::zeros((target_frames, feature_size));
    padded.slice_mut(s![..current_frames, ..]).assign(&features);
    padded
}

fn apply_time_offsets(raw: &mut RawTranscription, frame_offset: usize, context_frames: usize) {
    if context_frames == 0 {
        for frame_idx in &mut raw.frame_indices {
            *frame_idx += frame_offset;
        }
        return;
    }

    let mut keep_indices = Vec::new();
    for (index, frame_idx) in raw.frame_indices.iter_mut().enumerate() {
        if *frame_idx < context_frames {
            continue;
        }
        *frame_idx = *frame_idx - context_frames + frame_offset;
        keep_indices.push(index);
    }

    raw.token_ids = keep_indices
        .iter()
        .map(|index| raw.token_ids[*index])
        .collect();
    raw.frame_indices = keep_indices
        .iter()
        .map(|index| raw.frame_indices[*index])
        .collect();
    raw.durations = keep_indices
        .iter()
        .map(|index| raw.durations[*index])
        .collect();
    raw.confidences = keep_indices
        .iter()
        .map(|index| raw.confidences[*index])
        .collect();
}
