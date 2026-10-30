use crate::backend::EncoderChunk;
use crate::config::TranscriptionConfig;
use crate::constants::SAMPLES_PER_ENCODER_FRAME;
use crate::decode::{ParakeetTdtDecoder, RawTranscription};
use crate::deployment::ParakeetDeployment;
use crate::error::TranscriptionError;
use crate::frontend::ParakeetFeatureExtractor;
use crate::model::ParakeetModel;
use crate::models::ModelBundle;
use crate::types::TranscriptionResult;
use ndarray::Array2;

/// Single-chunk Parakeet transcription pipeline
///
/// This is the base `scriptrs` entry point for audio that already fits inside a
/// single Parakeet window. It expects mono 16kHz audio samples as `&[f32]`
///
/// If the input is longer than [`TranscriptionConfig::max_audio_samples`], this
/// pipeline returns [`TranscriptionError::AudioTooLong`] instead of splitting it.
/// Use `LongFormTranscriptionPipeline` with the `long-form` feature if you want
/// `scriptrs` to own chunk planning internally
#[derive(Debug, Clone)]
pub struct TranscriptionPipeline {
    bundle: ModelBundle,
    config: TranscriptionConfig,
    extractor: ParakeetFeatureExtractor,
    decoder: ParakeetTdtDecoder,
    model: ParakeetModel,
}

impl TranscriptionPipeline {
    /// Build a v2 transcription pipeline from a local model directory
    ///
    /// The directory must contain the Parakeet runtime bundle expected by
    /// [`ModelBundle::from_dir`]
    pub fn from_dir(models_dir: impl Into<std::path::PathBuf>) -> Result<Self, TranscriptionError> {
        Self::from_dir_with_deployment(models_dir, ParakeetDeployment::v2())
    }

    /// Build a transcription pipeline for an explicit deployment
    pub fn from_dir_with_deployment(
        models_dir: impl Into<std::path::PathBuf>,
        deployment: ParakeetDeployment,
    ) -> Result<Self, TranscriptionError> {
        let bundle = ModelBundle::from_dir_with_deployment(models_dir, deployment);
        Self::from_bundle(bundle)
    }

    /// Build a transcription pipeline from a resolved model bundle
    pub fn from_bundle(bundle: ModelBundle) -> Result<Self, TranscriptionError> {
        bundle.validate_base()?;
        let config = TranscriptionConfig::from_deployment(bundle.deployment());
        let vocab = bundle.vocabulary()?;
        let model = ParakeetModel::from_bundle(&bundle, &vocab, &config)?;
        Ok(Self {
            extractor: ParakeetFeatureExtractor::new(&config),
            decoder: ParakeetTdtDecoder::new(vocab),
            config,
            model,
            bundle,
        })
    }

    /// Download models and build a transcription pipeline
    ///
    /// With the default configuration this resolves models from the pinned
    /// `avencera/scriptrs-models` snapshot on Hugging Face. Set
    /// `SCRIPTRS_MODELS_DIR` to force a local bundle
    #[cfg(feature = "online")]
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
    /// [`TranscriptionError::AudioTooLong`]
    pub fn run(&self, audio: &[f32]) -> Result<TranscriptionResult, TranscriptionError> {
        self.run_with_config(audio, &self.config)
    }

    /// Transcribe a single chunk of audio with an explicit config
    ///
    /// The configuration must use the same deployment-owned frontend settings
    /// as this pipeline. Use its language field to select a deployment-supported
    /// language for this run
    pub fn run_with_config(
        &self,
        audio: &[f32],
        config: &TranscriptionConfig,
    ) -> Result<TranscriptionResult, TranscriptionError> {
        config.validate_for(self.deployment())?;
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

    /// Return the deployment that owns this pipeline
    pub fn deployment(&self) -> &ParakeetDeployment {
        self.bundle.deployment()
    }

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
        let prepared = self.prepare_chunk(audio, config)?;
        self.transcribe_prepared_raw(
            prepared,
            global_sample_offset,
            context_samples,
            config.language,
        )
    }

    pub(crate) fn transcribe_prepared_raw(
        &self,
        prepared: PreparedChunk,
        global_sample_offset: usize,
        context_samples: usize,
        language: crate::language::LanguageSelection,
    ) -> Result<RawTranscription, TranscriptionError> {
        let window = PreparedWindow {
            prepared,
            global_sample_offset,
            context_samples,
        };
        let mut raws = self.transcribe_prepared_batch(std::slice::from_ref(&window), language)?;
        Ok(raws.remove(0))
    }

    /// Whether the model runs a batch of windows together on the device
    #[cfg(feature = "long-form")]
    pub(crate) fn batches_on_device(&self) -> bool {
        self.model.batches_on_device()
    }

    /// Transcribes windows in one model call, each placed on the recording's timeline
    pub(crate) fn transcribe_prepared_batch(
        &self,
        windows: &[PreparedWindow],
        language: crate::language::LanguageSelection,
    ) -> Result<Vec<RawTranscription>, TranscriptionError> {
        let chunks: Vec<EncoderChunk<'_>> = windows
            .iter()
            .map(|window| EncoderChunk::new(window.prepared.features.view()))
            .collect();
        let decoded = self.model.transcribe_batch(&chunks, language)?;
        Ok(decoded
            .into_iter()
            .zip(windows)
            .map(|(decoded, window)| {
                let mut raw = decoded.raw;
                raw.drop_context_frames(
                    window.context_samples / SAMPLES_PER_ENCODER_FRAME,
                    window.global_sample_offset / SAMPLES_PER_ENCODER_FRAME,
                );
                raw
            })
            .collect())
    }

    pub(crate) fn chunk_preparer(config: &TranscriptionConfig) -> ChunkPreparer {
        ChunkPreparer::new(config)
    }

    fn prepare_chunk(
        &self,
        audio: &[f32],
        config: &TranscriptionConfig,
    ) -> Result<PreparedChunk, TranscriptionError> {
        PreparedChunk::from_audio(audio, config, &self.extractor)
    }
}

#[derive(Debug)]
pub(crate) struct PreparedChunk {
    features: Array2<f32>,
}

/// A prepared long-form window and where it sits in the recording
#[derive(Debug)]
pub(crate) struct PreparedWindow {
    pub(crate) prepared: PreparedChunk,
    /// Recording sample the window's own audio starts at
    pub(crate) global_sample_offset: usize,
    /// Left-context samples before that start, whose tokens are dropped
    pub(crate) context_samples: usize,
}

impl PreparedChunk {
    fn from_audio(
        audio: &[f32],
        config: &TranscriptionConfig,
        extractor: &ParakeetFeatureExtractor,
    ) -> Result<Self, TranscriptionError> {
        if audio.is_empty() {
            return Err(TranscriptionError::EmptyAudio);
        }
        if audio.len() > config.max_audio_samples {
            return Err(TranscriptionError::AudioTooLong {
                max_seconds: config.max_duration_seconds(),
                actual_seconds: audio.len() as f64 / config.sample_rate as f64,
            });
        }

        Ok(Self {
            features: extractor.extract(audio)?,
        })
    }
}

#[derive(Debug)]
pub(crate) struct ChunkPreparer {
    config: TranscriptionConfig,
    extractor: ParakeetFeatureExtractor,
}

impl ChunkPreparer {
    fn new(config: &TranscriptionConfig) -> Self {
        Self {
            config: config.clone(),
            extractor: ParakeetFeatureExtractor::new(config),
        }
    }

    pub(crate) fn prepare(&self, audio: &[f32]) -> Result<PreparedChunk, TranscriptionError> {
        PreparedChunk::from_audio(audio, &self.config, &self.extractor)
    }
}
