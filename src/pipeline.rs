use crate::backend::EncoderChunk;
use crate::config::TranscriptionConfig;
use crate::decode::{ParakeetTdtDecoder, RawTranscription};
use crate::deployment::{FrontendSpec, ParakeetDeployment};
use crate::error::TranscriptionError;
use crate::frontend::ParakeetFeatureExtractor;
use crate::model::{BatcherShape, ModelBatcher, ParakeetModel};
use crate::models::ModelBundle;
use crate::types::TranscriptionResult;
use ndarray::Array2;

/// Single-chunk Parakeet transcription pipeline
///
/// This is the base `scriptrs` entry point for audio that already fits inside a
/// single Parakeet window. It expects mono 16kHz audio samples as `&[f32]`
///
/// If the input is longer than the deployment's
/// [`ParakeetDeployment::hard_sample_max`], this pipeline returns
/// [`TranscriptionError::AudioTooLong`] instead of splitting it.
/// Use `LongFormTranscriptionPipeline` with the `long-form` feature if you want
/// `scriptrs` to own chunk planning internally
#[derive(Debug, Clone)]
pub struct TranscriptionPipeline {
    bundle: ModelBundle,
    config: TranscriptionConfig,
    chunk: ChunkSpec,
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
        let vocab = bundle.vocabulary()?;
        let model = ParakeetModel::from_bundle(&bundle, &vocab)?;
        Ok(Self {
            extractor: ParakeetFeatureExtractor::new(bundle.deployment().frontend()),
            chunk: ChunkSpec::for_deployment(bundle.deployment()),
            decoder: ParakeetTdtDecoder::new(vocab),
            config: TranscriptionConfig::default(),
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
    /// The config's language must be in this pipeline's deployment allowlist
    pub fn run_with_config(
        &self,
        audio: &[f32],
        config: &TranscriptionConfig,
    ) -> Result<TranscriptionResult, TranscriptionError> {
        self.deployment().select_language(config.language)?;
        let raw = self.transcribe_raw(audio, 0, 0, config)?;
        let duration_seconds = self.chunk.duration_seconds(audio.len());
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
        let prepared = PreparedChunk::from_audio(audio, self.chunk, &self.extractor)?;
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
                Self::place_window(
                    &mut raw,
                    window.global_sample_offset,
                    window.context_samples,
                );
                raw
            })
            .collect())
    }

    /// A batcher on the pipeline's model; see [`ParakeetModel::batcher`]
    pub(crate) fn batcher(
        &self,
        language: crate::language::LanguageSelection,
        shape: BatcherShape,
    ) -> Result<ModelBatcher, TranscriptionError> {
        self.model.batcher(language, shape)
    }

    /// Places a window's tokens on its recording's timeline, dropping those in its
    /// left context, as [`Self::transcribe_prepared_batch`] does
    pub(crate) fn place_window(
        raw: &mut RawTranscription,
        global_sample_offset: usize,
        context_samples: usize,
    ) {
        raw.place_on_timeline(context_samples, global_sample_offset);
    }

    /// The deployment's frontend and single-chunk limit
    pub(crate) fn chunk_spec(&self) -> ChunkSpec {
        self.chunk
    }
}

/// Frontend and single-chunk sample limit of a deployment
///
/// It is `Copy` and `Sync` where the pipeline is not, so worker threads can build
/// their own [`ChunkPreparer`] from it
#[derive(Debug, Clone, Copy)]
pub(crate) struct ChunkSpec {
    frontend: FrontendSpec,
    max_samples: usize,
}

impl ChunkSpec {
    fn for_deployment(deployment: &ParakeetDeployment) -> Self {
        Self {
            frontend: deployment.frontend(),
            max_samples: deployment.hard_sample_max(),
        }
    }

    /// A preparer with its own feature extractor
    pub(crate) fn preparer(self) -> ChunkPreparer {
        ChunkPreparer {
            chunk: self,
            extractor: ParakeetFeatureExtractor::new(self.frontend),
        }
    }

    /// Mel frames of the longest single input
    pub(crate) fn max_feature_frames(self) -> usize {
        self.max_samples / self.frontend.hop_length() + 1
    }

    /// Duration of `samples` at the deployment's sample rate
    pub(crate) fn duration_seconds(self, samples: usize) -> f64 {
        samples as f64 / self.frontend.sample_rate() as f64
    }

    /// Rejects empty input and input past the limit
    pub(crate) fn check(self, audio: &[f32]) -> Result<(), TranscriptionError> {
        if audio.is_empty() {
            return Err(TranscriptionError::EmptyAudio);
        }
        if audio.len() > self.max_samples {
            return Err(TranscriptionError::AudioTooLong {
                max_seconds: self.duration_seconds(self.max_samples),
                actual_seconds: self.duration_seconds(audio.len()),
            });
        }

        Ok(())
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
    /// Log-mel features, `[frames, features]`
    pub(crate) fn features(&self) -> &Array2<f32> {
        &self.features
    }

    /// Mel frames
    pub(crate) fn frames(&self) -> usize {
        self.features.nrows()
    }

    fn from_audio(
        audio: &[f32],
        chunk: ChunkSpec,
        extractor: &ParakeetFeatureExtractor,
    ) -> Result<Self, TranscriptionError> {
        chunk.check(audio)?;
        Ok(Self {
            features: extractor.extract(audio)?,
        })
    }
}

#[derive(Debug)]
pub(crate) struct ChunkPreparer {
    chunk: ChunkSpec,
    extractor: ParakeetFeatureExtractor,
}

impl ChunkPreparer {
    pub(crate) fn prepare(&self, audio: &[f32]) -> Result<PreparedChunk, TranscriptionError> {
        PreparedChunk::from_audio(audio, self.chunk, &self.extractor)
    }
}

#[cfg(test)]
mod tests {
    use super::ChunkSpec;
    use crate::deployment::ParakeetDeployment;
    use crate::error::TranscriptionError;

    fn assert_limit(deployment: &ParakeetDeployment, max: usize) {
        let chunk = ChunkSpec::for_deployment(deployment);
        assert!(chunk.check(&vec![0.0; max]).is_ok());
        assert!(matches!(
            chunk.check(&vec![0.0; max + 1]),
            Err(TranscriptionError::AudioTooLong { .. })
        ));
    }

    #[test]
    fn chunk_limit_follows_the_deployment() {
        assert_limit(&ParakeetDeployment::v2(), 240_000);
        assert_limit(&ParakeetDeployment::v3(), 240_000);
    }

    // native bundles raise the limit on the deployment itself, which the pipeline
    // has to read instead of a separate config value
    #[cfg(feature = "cuda")]
    #[test]
    fn chunk_limit_follows_a_raised_native_limit() {
        assert_limit(
            &ParakeetDeployment::v3().with_hard_sample_max(640_000),
            640_000,
        );
    }
}
