mod merge;
mod planner;
mod vad;

use crate::config::TranscriptionConfig;
use crate::constants::SAMPLE_RATE;
use crate::error::TranscriptionError;
use crate::models::ModelBundle;
use crate::pipeline::TranscriptionPipeline;
use crate::types::{TimedToken, TranscriptChunk, TranscriptionResult};

pub use planner::{OverlapChunkConfig, VadConfig, VadSegmentationConfig};

use self::merge::merge_overlapping_windows;
use self::planner::{
    SampleRange, detect_speech_regions, plan_region_subsegments, region_probability_slice,
};
use self::vad::SileroVad;

/// Long-form transcription pipeline with VAD and overlap fallback
///
/// This is the opt-in `scriptrs` entry point for long recordings. It wraps the
/// base [`TranscriptionPipeline`] and adds:
///
/// - VAD-based speech region detection
/// - silence-based splitting when possible
/// - overlap-window fallback when speech runs too long without silence
///
/// It expects the same mono 16kHz `&[f32]` input as the base pipeline.
#[derive(Debug, Clone)]
pub struct LongFormTranscriptionPipeline {
    inner: TranscriptionPipeline,
    vad: SileroVad,
    default_config: LongFormConfig,
}

/// Configuration for `LongFormTranscriptionPipeline`
///
/// This groups together the base transcription settings, VAD thresholds, and
/// overlap-fallback settings used for long recordings.
#[derive(Debug, Clone, Default)]
pub struct LongFormConfig {
    /// Single-chunk transcription settings
    pub transcription: TranscriptionConfig,
    /// VAD processing settings
    pub vad: VadConfig,
    /// VAD segmentation settings
    pub segmentation: VadSegmentationConfig,
    /// Overlap fallback settings
    pub overlap: OverlapChunkConfig,
}

impl LongFormTranscriptionPipeline {
    /// Build a long-form pipeline from a local model directory
    ///
    /// The directory must contain the base Parakeet bundle plus the VAD model
    /// expected by [`ModelBundle::validate_long_form`].
    pub fn from_dir(models_dir: impl Into<std::path::PathBuf>) -> Result<Self, TranscriptionError> {
        let bundle = ModelBundle::from_dir(models_dir);
        Self::from_bundle(bundle)
    }

    /// Build a long-form pipeline from a resolved model bundle
    pub fn from_bundle(bundle: ModelBundle) -> Result<Self, TranscriptionError> {
        bundle.validate_long_form()?;
        let inner = TranscriptionPipeline::from_bundle(bundle.clone())?;
        let vad = SileroVad::new(bundle.vad_dir())?;
        Ok(Self {
            inner,
            vad,
            default_config: LongFormConfig::default(),
        })
    }

    #[cfg(feature = "online")]
    /// Download models and build a long-form pipeline
    ///
    /// With the default configuration this resolves models from
    /// `avencera/scriptrs-models` on Hugging Face. Set `SCRIPTRS_MODELS_DIR` to
    /// force a local bundle or `SCRIPTRS_MODELS_REPO` to override the repo.
    pub fn from_pretrained() -> Result<Self, TranscriptionError> {
        let bundle = ModelBundle::from_pretrained_long_form().map_err(|error| {
            TranscriptionError::CoreMl(format!("model download failed: {error}"))
        })?;
        Self::from_bundle(bundle)
    }

    /// Transcribe audio, applying VAD and overlap fallback when needed
    ///
    /// Short clips still go through the base single-chunk path. Longer clips are
    /// regionized with VAD and split automatically.
    pub fn run(&self, audio: &[f32]) -> Result<TranscriptionResult, TranscriptionError> {
        self.run_with_config(audio, &self.default_config)
    }

    /// Transcribe audio with an explicit long-form config
    pub fn run_with_config(
        &self,
        audio: &[f32],
        config: &LongFormConfig,
    ) -> Result<TranscriptionResult, TranscriptionError> {
        if audio.is_empty() {
            return Err(TranscriptionError::EmptyAudio);
        }
        if audio.len() <= config.transcription.max_audio_samples {
            return self.inner.run_with_config(audio, &config.transcription);
        }

        self.run_long_form(audio, config)
    }

    /// Run the inner single-chunk pipeline directly
    ///
    /// This is useful when you want to reuse the long-form model bundle but feed
    /// already-split chunks through the base transcription path yourself.
    pub fn run_chunk(&self, audio: &[f32]) -> Result<TranscriptionResult, TranscriptionError> {
        self.inner.run(audio)
    }

    fn run_long_form(
        &self,
        audio: &[f32],
        config: &LongFormConfig,
    ) -> Result<TranscriptionResult, TranscriptionError> {
        let probabilities = self.vad.process(audio, &config.vad)?;
        let regions = detect_speech_regions(
            &probabilities,
            audio.len(),
            config.segmentation.threshold(config.vad.default_threshold),
            &config.segmentation,
        );
        if regions.is_empty() {
            return Ok(TranscriptionResult::empty(duration_seconds(audio.len())));
        }

        let mut tokens = Vec::new();
        let mut chunks = Vec::new();
        for region in regions {
            let region_tokens = self.transcribe_region(audio, &probabilities, region, config)?;
            if let Some(chunk) = build_chunk(&region_tokens) {
                chunks.push(chunk);
                tokens.extend(region_tokens);
            }
        }

        Ok(build_result(audio.len(), chunks, tokens))
    }

    fn transcribe_region(
        &self,
        audio: &[f32],
        probabilities: &[f32],
        region: SampleRange,
        config: &LongFormConfig,
    ) -> Result<Vec<TimedToken>, TranscriptionError> {
        let region_audio = &audio[region.start..region.end];
        if region_audio.len() <= config.transcription.max_audio_samples {
            return self.transcribe_single_segment(region_audio, region.start, config);
        }

        if let Some(subsegments) = plan_region_subsegments(
            region,
            region_probability_slice(probabilities, region),
            &config.segmentation,
            config.transcription.max_audio_samples,
        ) {
            return self.transcribe_subsegments(audio, subsegments, config);
        }

        self.transcribe_overlap_region(audio, region, config)
    }

    fn transcribe_single_segment(
        &self,
        audio: &[f32],
        sample_offset: usize,
        config: &LongFormConfig,
    ) -> Result<Vec<TimedToken>, TranscriptionError> {
        let mut tokens = self
            .inner
            .run_with_config(audio, &config.transcription)?
            .tokens;
        offset_tokens(&mut tokens, sample_offset);
        Ok(tokens)
    }

    fn transcribe_subsegments(
        &self,
        audio: &[f32],
        subsegments: Vec<SampleRange>,
        config: &LongFormConfig,
    ) -> Result<Vec<TimedToken>, TranscriptionError> {
        let mut tokens = Vec::new();
        for subsegment in subsegments {
            let sub_audio = &audio[subsegment.start..subsegment.end];
            tokens.extend(self.transcribe_single_segment(sub_audio, subsegment.start, config)?);
        }
        Ok(tokens)
    }

    fn transcribe_overlap_region(
        &self,
        audio: &[f32],
        region: SampleRange,
        config: &LongFormConfig,
    ) -> Result<Vec<TimedToken>, TranscriptionError> {
        let raw_windows = self.transcribe_overlap_windows(audio, region, config)?;
        let merged = merge_overlapping_windows(raw_windows);
        Ok(self
            .inner
            .decode_raw(&merged, duration_seconds(audio.len()))
            .tokens)
    }

    fn transcribe_overlap_windows(
        &self,
        audio: &[f32],
        region: SampleRange,
        config: &LongFormConfig,
    ) -> Result<Vec<crate::decode::RawTranscription>, TranscriptionError> {
        let mut raw_windows = Vec::new();
        for chunk in config.overlap.plan(region) {
            let context_start = chunk.start.saturating_sub(config.overlap.context_samples);
            let chunk_audio = &audio[context_start..chunk.end];
            raw_windows.push(self.inner.transcribe_raw(
                chunk_audio,
                chunk.start,
                chunk.start - context_start,
                &config.transcription,
            )?);
        }
        Ok(raw_windows)
    }
}

fn join_token_text(tokens: &[TimedToken]) -> String {
    tokens
        .iter()
        .map(|token| token.text.as_str())
        .collect::<String>()
        .trim()
        .to_owned()
}

fn build_chunk(tokens: &[TimedToken]) -> Option<TranscriptChunk> {
    Some(TranscriptChunk {
        start: tokens.first()?.start,
        end: tokens.last()?.end,
        text: join_token_text(tokens),
    })
}

fn build_result(
    audio_len: usize,
    chunks: Vec<TranscriptChunk>,
    tokens: Vec<TimedToken>,
) -> TranscriptionResult {
    let text = chunks
        .iter()
        .map(|chunk| chunk.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    TranscriptionResult {
        text,
        chunks,
        tokens,
        duration_seconds: duration_seconds(audio_len),
    }
}

fn offset_tokens(tokens: &mut [TimedToken], sample_offset: usize) {
    for token in tokens {
        offset_token(token, sample_offset);
    }
}

fn offset_token(token: &mut TimedToken, sample_offset: usize) {
    let seconds = sample_offset as f64 / SAMPLE_RATE as f64;
    token.start += seconds;
    token.end += seconds;
}

fn duration_seconds(sample_count: usize) -> f64 {
    sample_count as f64 / SAMPLE_RATE as f64
}
