mod execute;
mod merge;
mod planner;
#[cfg(feature = "vad")]
pub(crate) mod vad;

use crate::config::TranscriptionConfig;
use crate::constants::SAMPLE_RATE;
use crate::decode::RawTranscription;
use crate::deployment::ParakeetDeployment;
use crate::error::TranscriptionError;
use crate::models::ModelBundle;
use crate::pipeline::TranscriptionPipeline;
use crate::preprocessing::{
    IdentityPreprocessor, PreprocessedTranscriptionError, TranscriptionControl, WindowIndex,
    WindowPreprocessor,
};
use crate::types::{TimedToken, TranscriptChunk, TranscriptionResult};

pub use planner::OverlapChunkConfig;

/// Overlap window for native bundles, 30 s at 16 kHz
#[cfg(feature = "cuda")]
const NATIVE_WINDOW_SAMPLES: usize = 480_000;
#[cfg(feature = "cuda")]
const NATIVE_OVERLAP_SECONDS: f64 = 8.0;
/// Windows per CUDA call in native long-form runs
#[cfg(feature = "cuda")]
const NATIVE_BATCH_SIZE: usize = 16;
#[cfg(feature = "vad")]
pub use planner::{VadConfig, VadSegmentationConfig};

use self::execute::{RunRequest, execute_plan, transcribe_task};
use self::merge::merge_overlapping_windows;
use self::planner::{ChunkTask, LongFormExecutionPlan, RegionTaskKind, build_fast_execution_plan};
#[cfg(feature = "vad")]
use self::planner::{build_vad_execution_plan, detect_speech_regions};
#[cfg(feature = "vad")]
use self::vad::SileroVad;

/// Long-form transcription pipeline with overlap chunking and optional VAD
///
/// This is the opt-in `scriptrs` entry point for long recordings. It wraps the
/// base [`TranscriptionPipeline`] and adds internal chunk planning for audio
/// that exceeds a single Parakeet window
///
/// With `long-form`, the default path chunks the full recording with overlap
/// windows and runs up to 4 workers in parallel, or, on a native CUDA bundle,
/// runs batches of windows through one model
///
/// With `vad`, you can switch to VAD-based speech region detection
/// for recordings that contain long silences or a lot of non-speech content
///
/// It expects the same mono 16kHz `&[f32]` input as the base pipeline
#[derive(Debug, Clone)]
pub struct LongFormTranscriptionPipeline {
    inner: TranscriptionPipeline,
    #[cfg(feature = "vad")]
    vad: Option<SileroVad>,
    default_config: LongFormConfig,
}

/// Configuration for `LongFormTranscriptionPipeline`
///
/// This groups together the base transcription settings, execution mode, and
/// overlap-window settings used for long recordings
#[derive(Debug, Clone)]
pub struct LongFormConfig {
    /// Long-form planning mode
    pub mode: LongFormMode,
    /// Parallel chunk workers used for long-form transcription, each with its own
    /// model; when windows run in batches, the threads that prepare them for the
    /// pipeline's one model instead
    pub worker_count: usize,
    /// Windows per model call on backends that batch on the device (native CUDA);
    /// above 1, windows run in batches through the pipeline's one model instead of
    /// on parallel workers. Core ML runs one window per call and ignores it
    pub batch_size: usize,
    /// Single-chunk transcription settings
    pub transcription: TranscriptionConfig,
    /// VAD processing settings
    #[cfg(feature = "vad")]
    pub vad: VadConfig,
    /// VAD segmentation settings
    #[cfg(feature = "vad")]
    pub segmentation: VadSegmentationConfig,
    /// Overlap fallback settings
    pub overlap: OverlapChunkConfig,
}

impl Default for LongFormConfig {
    fn default() -> Self {
        Self::from_deployment(&ParakeetDeployment::v2())
    }
}

impl LongFormConfig {
    /// Build long-form settings from a deployment hard maximum
    pub fn from_deployment(deployment: &ParakeetDeployment) -> Self {
        Self {
            mode: LongFormMode::default(),
            worker_count: 4,
            batch_size: 1,
            transcription: TranscriptionConfig::from_deployment(deployment),
            #[cfg(feature = "vad")]
            vad: VadConfig::default(),
            #[cfg(feature = "vad")]
            segmentation: VadSegmentationConfig::default(),
            overlap: OverlapChunkConfig::from_deployment(deployment),
        }
    }

    /// Long-form settings for a bundle: the deployment's, adjusted to its runtime
    ///
    /// A native CUDA bundle gets 30 s windows with 8 s of overlap, run 16 at a time
    /// through one model. Longer windows lose accuracy: on Earnings-21 they did best
    /// at 20-40 s with 4-8 s of overlap, and from 60 s up WER rose by 1-4 points
    pub fn for_bundle(bundle: &ModelBundle) -> Self {
        #[cfg_attr(
            not(feature = "cuda"),
            expect(unused_mut, reason = "only native bundles adjust the defaults")
        )]
        let mut config = Self::from_deployment(bundle.deployment());
        #[cfg(feature = "cuda")]
        if matches!(bundle.runtime(), crate::models::RuntimeBundle::Native(_)) {
            config.overlap.max_model_samples = NATIVE_WINDOW_SAMPLES;
            config.overlap.overlap_seconds = NATIVE_OVERLAP_SECONDS;
            config.batch_size = NATIVE_BATCH_SIZE;
        }
        config
    }

    /// Validate all settings against the deployment that owns the pipeline
    pub fn validate_for(&self, deployment: &ParakeetDeployment) -> Result<(), TranscriptionError> {
        if self.batch_size == 0 {
            return Err(TranscriptionError::InvalidConfiguration(
                "long-form batch_size must be positive".to_owned(),
            ));
        }
        self.transcription.validate_for(deployment)?;
        self.overlap.validate_for(deployment)?;

        #[cfg(feature = "vad")]
        {
            self.vad.validate()?;
            self.segmentation.validate_for(self.vad.default_threshold)?;
        }

        Ok(())
    }
}

/// Long-form planning mode
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LongFormMode {
    /// Skip VAD and chunk the full recording with overlap windows
    #[default]
    Fast,
    /// Detect speech with VAD before chunk planning
    #[cfg(feature = "vad")]
    Vad,
}

impl LongFormTranscriptionPipeline {
    /// Build a long-form pipeline from a local model directory
    ///
    /// The directory must contain the base Parakeet bundle expected by
    /// [`ModelBundle::validate_base`]
    ///
    /// With `vad`, the VAD model is optional at construction time and
    /// only required when the VAD mode is used
    pub fn from_dir(models_dir: impl Into<std::path::PathBuf>) -> Result<Self, TranscriptionError> {
        Self::from_dir_with_deployment(models_dir, ParakeetDeployment::v2())
    }

    /// Build a long-form pipeline for an explicit deployment
    pub fn from_dir_with_deployment(
        models_dir: impl Into<std::path::PathBuf>,
        deployment: ParakeetDeployment,
    ) -> Result<Self, TranscriptionError> {
        let bundle = ModelBundle::from_dir_with_deployment(models_dir, deployment);
        Self::from_bundle(bundle)
    }

    /// Build a long-form pipeline from a resolved model bundle
    pub fn from_bundle(bundle: ModelBundle) -> Result<Self, TranscriptionError> {
        bundle.validate_base()?;
        let default_config = LongFormConfig::for_bundle(&bundle);
        #[cfg(feature = "vad")]
        let vad = SileroVad::from_bundle(&bundle)?;
        let inner = TranscriptionPipeline::from_bundle(bundle)?;
        Ok(Self {
            inner,
            #[cfg(feature = "vad")]
            vad,
            default_config,
        })
    }

    /// Download models and build a long-form pipeline
    ///
    /// With the default configuration this resolves models from the pinned
    /// `avencera/scriptrs-models` snapshot on Hugging Face. Set
    /// `SCRIPTRS_MODELS_DIR` to force a local bundle
    #[cfg(feature = "online")]
    pub fn from_pretrained() -> Result<Self, TranscriptionError> {
        #[cfg(feature = "vad")]
        let bundle = ModelBundle::from_pretrained_long_form().map_err(|error| {
            TranscriptionError::CoreMl(format!("model download failed: {error}"))
        })?;
        #[cfg(not(feature = "vad"))]
        let bundle = ModelBundle::from_pretrained().map_err(|error| {
            TranscriptionError::CoreMl(format!("model download failed: {error}"))
        })?;
        Self::from_bundle(bundle)
    }

    /// Transcribe audio with the default long-form config
    ///
    /// Short clips still go through the base single-chunk path. Longer clips use
    /// overlap chunking by default, with optional VAD planning when
    /// `vad` is enabled
    pub fn run(&self, audio: &[f32]) -> Result<TranscriptionResult, TranscriptionError> {
        self.run_with_config(audio, &self.default_config)
    }

    /// Transcribe audio with an explicit long-form config
    pub fn run_with_config(
        &self,
        audio: &[f32],
        config: &LongFormConfig,
    ) -> Result<TranscriptionResult, TranscriptionError> {
        self.run_with_preprocessor(
            audio,
            config,
            &TranscriptionControl::internal_unbounded(),
            &IdentityPreprocessor,
        )
        .map_err(identity_error)
    }

    /// Transcribe audio while preprocessing each planned ASR window once
    pub fn run_with_preprocessor<P: WindowPreprocessor + ?Sized>(
        &self,
        audio: &[f32],
        config: &LongFormConfig,
        control: &TranscriptionControl,
        preprocessor: &P,
    ) -> Result<TranscriptionResult, PreprocessedTranscriptionError<P::Error>> {
        control.check()?;
        config
            .validate_for(self.inner.deployment())
            .map_err(PreprocessedTranscriptionError::from)?;
        if audio.is_empty() {
            return Err(TranscriptionError::EmptyAudio.into());
        }
        validate_finite_samples(audio, 0)?;

        let request = RunRequest {
            audio,
            config,
            control,
            preprocessor,
        };
        if audio.len() <= config.transcription.max_audio_samples {
            control.check_plan(1)?;
            return self.run_single_task(request);
        }

        self.run_long_form(request)
    }

    /// Run the inner single-chunk pipeline directly
    ///
    /// This is useful when you want to reuse the long-form model bundle but feed
    /// already-split chunks through the base transcription path yourself
    pub fn run_chunk(&self, audio: &[f32]) -> Result<TranscriptionResult, TranscriptionError> {
        let control = TranscriptionControl::internal_unbounded();
        let config = &self.default_config;
        config.validate_for(self.inner.deployment())?;
        if audio.is_empty() {
            return Err(TranscriptionError::EmptyAudio);
        }
        if audio.len() > config.transcription.max_audio_samples {
            return Err(TranscriptionError::AudioTooLong {
                max_seconds: config.transcription.max_duration_seconds(),
                actual_seconds: audio.len() as f64 / config.transcription.sample_rate as f64,
            });
        }
        validate_finite_samples(audio, 0)?;
        control.check_plan(1).map_err(TranscriptionError::from)?;
        self.run_single_task(RunRequest {
            audio,
            config,
            control: &control,
            preprocessor: &IdentityPreprocessor,
        })
        .map_err(identity_error)
    }

    /// Return the single-chunk pipeline this pipeline wraps
    ///
    /// Clones share the loaded model, so a clone can move onto a background queue
    /// with [`TranscriptionPipeline::into_queued`] without loading the model again
    pub fn base_pipeline(&self) -> &TranscriptionPipeline {
        &self.inner
    }

    /// Return the default long-form config
    pub fn config(&self) -> &LongFormConfig {
        &self.default_config
    }

    fn run_long_form<P: WindowPreprocessor + ?Sized>(
        &self,
        request: RunRequest<'_, P>,
    ) -> Result<TranscriptionResult, PreprocessedTranscriptionError<P::Error>> {
        match request.config.mode {
            LongFormMode::Fast => self.run_fast_long_form(request),
            #[cfg(feature = "vad")]
            LongFormMode::Vad => self.run_vad_long_form(request),
        }
    }

    #[cfg(feature = "vad")]
    fn run_vad_long_form<P: WindowPreprocessor + ?Sized>(
        &self,
        request: RunRequest<'_, P>,
    ) -> Result<TranscriptionResult, PreprocessedTranscriptionError<P::Error>> {
        let RunRequest {
            audio,
            config,
            control,
            ..
        } = request;
        control.check()?;
        let vad = self
            .vad
            .as_ref()
            .ok_or_else(|| self.inner.bundle().missing_vad_error())
            .map_err(PreprocessedTranscriptionError::from)?;
        let probabilities = vad
            .process(audio, &config.vad, control)
            .map_err(PreprocessedTranscriptionError::from)?;
        control.check()?;
        let regions = detect_speech_regions(
            &probabilities,
            audio.len(),
            config.segmentation.threshold(config.vad.default_threshold),
            &config.segmentation,
        );
        let execution_plan = build_vad_execution_plan(&regions, &probabilities, config);
        control.check_plan(execution_plan.tasks.len())?;
        self.run_plan(request, execution_plan)
    }

    fn run_fast_long_form<P: WindowPreprocessor + ?Sized>(
        &self,
        request: RunRequest<'_, P>,
    ) -> Result<TranscriptionResult, PreprocessedTranscriptionError<P::Error>> {
        request.control.check()?;
        let execution_plan = build_fast_execution_plan(request.audio.len(), request.config);
        request.control.check_plan(execution_plan.tasks.len())?;
        self.run_plan(request, execution_plan)
    }

    fn run_single_task<P: WindowPreprocessor + ?Sized>(
        &self,
        request: RunRequest<'_, P>,
    ) -> Result<TranscriptionResult, PreprocessedTranscriptionError<P::Error>> {
        let audio_len = request.audio.len();
        let task = ChunkTask {
            audio_start: 0,
            audio_end: audio_len,
            global_sample_offset: 0,
            context_samples: 0,
        };
        let raw = transcribe_task(
            &request.task_context(&self.inner),
            &task,
            WindowIndex::new(0),
        )?;
        request.control.check()?;

        Ok(self.inner.decode_raw(&raw, duration_seconds(audio_len)))
    }

    /// Runs every window of the plan, then decodes and merges them in plan order
    fn run_plan<P: WindowPreprocessor + ?Sized>(
        &self,
        request: RunRequest<'_, P>,
        execution_plan: LongFormExecutionPlan,
    ) -> Result<TranscriptionResult, PreprocessedTranscriptionError<P::Error>> {
        request.control.check()?;
        let audio_len = request.audio.len();
        if execution_plan.tasks.is_empty() {
            return Ok(TranscriptionResult::empty(duration_seconds(audio_len)));
        }

        let raw_tasks = execute_plan(&self.inner, request, &execution_plan.tasks)?;
        Ok(self.build_plan_result(audio_len, execution_plan, raw_tasks)?)
    }

    fn build_plan_result(
        &self,
        audio_len: usize,
        execution_plan: LongFormExecutionPlan,
        mut raw_tasks: Vec<Option<RawTranscription>>,
    ) -> Result<TranscriptionResult, TranscriptionError> {
        let mut tokens = Vec::new();
        let mut chunks = Vec::new();

        for region in execution_plan.regions {
            let region_tokens = match region.kind {
                #[cfg(feature = "vad")]
                RegionTaskKind::Segments => self.decode_segment_tasks(
                    &execution_plan.tasks,
                    &region.task_indices,
                    &mut raw_tasks,
                )?,
                RegionTaskKind::Overlap => {
                    self.decode_overlap_tasks(audio_len, &region.task_indices, &mut raw_tasks)?
                }
            };

            if let Some(chunk) = build_chunk(&region_tokens) {
                chunks.push(chunk);
                tokens.extend(region_tokens);
            }
        }

        Ok(build_result(audio_len, chunks, tokens))
    }

    /// Decodes each VAD segment on its own and moves it to where it starts in the
    /// recording
    #[cfg(feature = "vad")]
    fn decode_segment_tasks(
        &self,
        tasks: &[ChunkTask],
        task_indices: &[usize],
        raw_tasks: &mut [Option<RawTranscription>],
    ) -> Result<Vec<TimedToken>, TranscriptionError> {
        let mut tokens = Vec::new();
        for task_idx in task_indices {
            let task = &tasks[*task_idx];
            let raw = take_raw_task(raw_tasks, *task_idx)?;
            let duration_samples = task.audio_end.saturating_sub(task.audio_start);
            let mut chunk_tokens = self
                .inner
                .decode_raw(&raw, duration_seconds(duration_samples))
                .tokens;
            offset_tokens(&mut chunk_tokens, task.audio_start);
            tokens.extend(chunk_tokens);
        }
        Ok(tokens)
    }

    fn decode_overlap_tasks(
        &self,
        audio_len: usize,
        task_indices: &[usize],
        raw_tasks: &mut [Option<RawTranscription>],
    ) -> Result<Vec<TimedToken>, TranscriptionError> {
        let mut windows = Vec::with_capacity(task_indices.len());
        for task_idx in task_indices {
            windows.push(take_raw_task(raw_tasks, *task_idx)?);
        }
        let merged = merge_overlapping_windows(windows);
        Ok(self
            .inner
            .decode_raw(&merged, duration_seconds(audio_len))
            .tokens)
    }
}

fn validate_finite_samples(
    samples: &[f32],
    source_offset: usize,
) -> Result<(), TranscriptionError> {
    let Some((index, _)) = samples
        .iter()
        .enumerate()
        .find(|(_, sample)| !sample.is_finite())
    else {
        return Ok(());
    };

    Err(TranscriptionError::InvalidAudioSample {
        index: source_offset + index,
    })
}

fn identity_error(
    error: PreprocessedTranscriptionError<std::convert::Infallible>,
) -> TranscriptionError {
    match error {
        PreprocessedTranscriptionError::Transcription(error) => error,
        PreprocessedTranscriptionError::Preprocessing(error) => match error {},
    }
}

fn take_raw_task(
    raw_tasks: &mut [Option<RawTranscription>],
    task_idx: usize,
) -> Result<RawTranscription, TranscriptionError> {
    raw_tasks[task_idx].take().ok_or_else(|| {
        TranscriptionError::CoreMl(format!(
            "missing prepared transcription output for task {task_idx}"
        ))
    })
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

#[cfg(feature = "vad")]
fn offset_tokens(tokens: &mut [TimedToken], sample_offset: usize) {
    for token in tokens {
        offset_token(token, sample_offset);
    }
}

#[cfg(feature = "vad")]
fn offset_token(token: &mut TimedToken, sample_offset: usize) {
    let seconds = sample_offset as f64 / SAMPLE_RATE as f64;
    token.start += seconds;
    token.end += seconds;
}

fn duration_seconds(sample_count: usize) -> f64 {
    sample_count as f64 / SAMPLE_RATE as f64
}

#[cfg(test)]
mod tests {
    use std::fmt::{Display, Formatter};
    use std::sync::{Arc, Mutex};

    #[cfg(all(feature = "cuda", feature = "vad"))]
    use super::planner::{SampleRange, build_region_plan};
    use super::{LongFormConfig, LongFormMode, WindowPreprocessor};
    #[cfg(feature = "cuda")]
    use super::{LongFormTranscriptionPipeline, build_fast_execution_plan};
    #[cfg(all(feature = "cuda", feature = "vad"))]
    use super::{SileroVad, detect_speech_regions};
    #[cfg(all(feature = "cuda", feature = "vad"))]
    use crate::constants::SAMPLE_RATE;
    #[cfg(feature = "cuda")]
    use crate::cuda::test_support::nemo_parity_bundle;
    use crate::deployment::ParakeetDeployment;
    #[cfg(feature = "cuda")]
    use crate::error::TranscriptionError;
    #[cfg(feature = "cuda")]
    use crate::preprocessing::PreprocessedTranscriptionError;
    use crate::preprocessing::{AudioWindowMut, TranscriptionControl};
    #[cfg(feature = "cuda")]
    use crate::preprocessing::{TranscriptionDeadline, WindowLimit};
    #[cfg(all(feature = "cuda", feature = "vad"))]
    use crate::test_support::NemoReference;

    #[derive(Debug)]
    pub(super) struct HookError;

    impl Display for HookError {
        fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("test hook failed")
        }
    }

    impl std::error::Error for HookError {}

    #[derive(Debug, Clone)]
    pub(super) struct RecordingHook(pub(super) Arc<Mutex<Vec<(usize, usize, usize)>>>);

    impl WindowPreprocessor for RecordingHook {
        type Error = HookError;

        fn process(
            &self,
            window: &mut AudioWindowMut<'_>,
            _control: &TranscriptionControl,
        ) -> Result<(), Self::Error> {
            let range = window.source_range();
            self.0
                .lock()
                .expect("test observer should not be poisoned")
                .push((window.index().get(), range.start(), range.end()));
            Ok(())
        }
    }

    #[test]
    fn long_form_mode_defaults_to_fast() {
        assert_eq!(LongFormConfig::default().mode, LongFormMode::Fast);
    }

    #[test]
    fn long_form_worker_count_defaults_to_four() {
        assert_eq!(LongFormConfig::default().worker_count, 4);
    }

    #[test]
    fn finite_sample_validation_rejects_non_finite_audio_before_frontend() {
        let error = super::validate_finite_samples(&[0.0, f32::NAN], 12)
            .expect_err("non-finite input must fail at the shared boundary");

        assert!(matches!(
            error,
            crate::error::TranscriptionError::InvalidAudioSample { index: 13 }
        ));
    }

    #[test]
    fn long_form_config_rejects_overlap_above_deployment_limit() {
        let deployment = ParakeetDeployment::v3();
        let mut config = LongFormConfig::from_deployment(&deployment);
        config.overlap.max_model_samples = deployment.hard_sample_max() + 1;

        assert!(config.validate_for(&deployment).is_err());
    }

    #[cfg(feature = "vad")]
    #[test]
    fn long_form_config_rejects_invalid_vad_threshold() {
        let deployment = ParakeetDeployment::v3();
        let mut config = LongFormConfig::from_deployment(&deployment);
        config.vad.default_threshold = f32::NAN;

        assert!(config.validate_for(&deployment).is_err());
    }

    #[cfg(feature = "vad")]
    #[test]
    fn long_form_config_rejects_non_positive_vad_duration() {
        let deployment = ParakeetDeployment::v3();
        let mut config = LongFormConfig::from_deployment(&deployment);
        config.segmentation.min_speech_duration = 0.0;

        assert!(config.validate_for(&deployment).is_err());
    }

    #[test]
    fn long_form_config_rejects_zero_model_maximum() {
        let deployment = ParakeetDeployment::v3();
        let mut config = LongFormConfig::from_deployment(&deployment);
        config.overlap.max_model_samples = 0;

        assert!(config.validate_for(&deployment).is_err());
    }

    /// Silence past the native single-input limit, so it takes the long-form path
    #[cfg(feature = "cuda")]
    const LONG_AUDIO_SAMPLES: usize = 800_000;

    /// A long-form execution path, as `execute_plan` dispatches it
    #[cfg(feature = "cuda")]
    #[derive(Debug, Clone, Copy)]
    enum ExecutionPath {
        Serial,
        Parallel,
        Batched,
    }

    #[cfg(feature = "cuda")]
    impl ExecutionPath {
        /// Config that sends a multi-window plan down this path on a native bundle
        fn config(self, base: &LongFormConfig) -> LongFormConfig {
            match self {
                Self::Serial => LongFormConfig {
                    worker_count: 1,
                    batch_size: 1,
                    ..base.clone()
                },
                // each parallel worker loads its own model onto the GPU, so two keep
                // the load short while still running windows concurrently
                Self::Parallel => LongFormConfig {
                    worker_count: 2,
                    batch_size: 1,
                    ..base.clone()
                },
                Self::Batched => base.clone(),
            }
        }
    }

    /// Native v3 pipeline and a Fast config whose plan for [`LONG_AUDIO_SAMPLES`]
    /// spans several batches
    ///
    /// The 3 s windows keep the plan long while the GPU work stays small; the
    /// native 8 s overlap is capped at half a window, which `validate_for` accepts
    #[cfg(feature = "cuda")]
    fn native_pipeline() -> (LongFormTranscriptionPipeline, LongFormConfig) {
        let bundle = nemo_parity_bundle(ParakeetDeployment::v3());
        let mut config = LongFormConfig {
            mode: LongFormMode::Fast,
            ..LongFormConfig::for_bundle(&bundle)
        };
        config.overlap.max_model_samples = 48_000;
        assert!(
            config.batch_size > 1 && config.worker_count > 1,
            "the native defaults must select the batched path"
        );
        assert!(
            LONG_AUDIO_SAMPLES > config.transcription.max_audio_samples,
            "the audio must exceed the single-input limit"
        );

        let plan = build_fast_execution_plan(LONG_AUDIO_SAMPLES, &config);
        assert!(
            plan.tasks.len() > config.batch_size,
            "the plan must span more than one batch"
        );

        let pipeline = LongFormTranscriptionPipeline::from_bundle(bundle).expect("pipeline");
        (pipeline, config)
    }

    #[cfg(feature = "cuda")]
    fn recording_control(max_windows: usize) -> TranscriptionControl {
        TranscriptionControl::new(
            TranscriptionDeadline::from_now(std::time::Duration::from_secs(60)),
            WindowLimit::new(max_windows).expect("the window limit is nonzero"),
        )
    }

    #[cfg(feature = "cuda")]
    fn observed_hooks(hook: &RecordingHook) -> Vec<(usize, usize, usize)> {
        hook.0
            .lock()
            .expect("test observer should not be poisoned")
            .clone()
    }

    #[cfg(feature = "cuda")]
    fn planned_windows(audio_len: usize, config: &LongFormConfig) -> Vec<(usize, usize, usize)> {
        build_fast_execution_plan(audio_len, config)
            .tasks
            .iter()
            .enumerate()
            .map(|(index, task)| (index, task.audio_start, task.audio_end))
            .collect()
    }

    /// Audio within the single-input limit is preprocessed once, as one window
    /// covering all of it
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs a CUDA GPU and the native weights; `just gpu-parity` runs it on gpu0"]
    fn gpu_short_path_runs_the_preprocessor_once() {
        let (pipeline, config) = native_pipeline();
        let audio = vec![0.0; 16_000];
        let hook = RecordingHook(Arc::new(Mutex::new(Vec::new())));

        pipeline
            .run_with_preprocessor(&audio, &config, &recording_control(1), &hook)
            .expect("short audio should transcribe");

        assert_eq!(observed_hooks(&hook), vec![(0, 0, audio.len())]);
    }

    /// Every path preprocesses each planned window exactly once, with its index and
    /// source range
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs a CUDA GPU and the native weights; `just gpu-parity` runs it on gpu0"]
    fn gpu_long_form_runs_the_preprocessor_once_per_planned_window() {
        let (pipeline, base) = native_pipeline();
        let audio = vec![0.0; LONG_AUDIO_SAMPLES];

        for path in [
            ExecutionPath::Serial,
            ExecutionPath::Parallel,
            ExecutionPath::Batched,
        ] {
            let config = path.config(&base);
            let planned = planned_windows(audio.len(), &config);
            let hook = RecordingHook(Arc::new(Mutex::new(Vec::new())));

            pipeline
                .run_with_preprocessor(&audio, &config, &recording_control(planned.len()), &hook)
                .unwrap_or_else(|error| panic!("{path:?} run failed: {error}"));

            let mut observed = observed_hooks(&hook);
            // only the serial path runs the windows in plan order
            if !matches!(path, ExecutionPath::Serial) {
                observed.sort_unstable();
            }
            assert_eq!(observed, planned, "{path:?} preprocessed windows");
        }
    }

    /// Once the deadline passes, every long-form path stops before the remaining
    /// windows and reports the deadline
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs a CUDA GPU and the native weights; `just gpu-parity` runs it on gpu0"]
    fn gpu_deadline_stops_later_windows() {
        struct OutlastingHook(RecordingHook);

        impl WindowPreprocessor for OutlastingHook {
            type Error = HookError;

            fn process(
                &self,
                window: &mut AudioWindowMut<'_>,
                control: &TranscriptionControl,
            ) -> Result<(), Self::Error> {
                self.0.process(window, control)?;
                // waiting out the deadline makes the check after this window the one
                // that fails, however long the run took to reach it
                while !control.deadline().is_expired() {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                Ok(())
            }
        }

        let (pipeline, base) = native_pipeline();
        let audio = vec![0.0; LONG_AUDIO_SAMPLES];

        for path in [
            ExecutionPath::Serial,
            ExecutionPath::Parallel,
            ExecutionPath::Batched,
        ] {
            let config = path.config(&base);
            let planned = planned_windows(audio.len(), &config).len();
            let hook = OutlastingHook(RecordingHook(Arc::new(Mutex::new(Vec::new()))));
            // usually long enough for the parallel workers to load their models
            // first
            let control = TranscriptionControl::new(
                TranscriptionDeadline::from_now(std::time::Duration::from_secs(10)),
                WindowLimit::new(planned).expect("the plan has windows"),
            );

            let error = pipeline
                .run_with_preprocessor(&audio, &config, &control, &hook)
                .expect_err("the deadline should expire before later windows");

            assert!(
                matches!(
                    error,
                    PreprocessedTranscriptionError::Transcription(
                        TranscriptionError::DeadlineExpired
                    )
                ),
                "{path:?} returned {error}"
            );
            let observed = observed_hooks(&hook.0).len();
            // parallel workers load their models inside the run, and a slow load
            // may correctly use up the whole deadline before any window
            if !matches!(path, ExecutionPath::Parallel) {
                assert!(
                    observed > 0,
                    "{path:?} started no window before the deadline"
                );
            }
            assert!(
                observed < planned,
                "{path:?} preprocessed all {planned} windows past the deadline"
            );
        }
    }

    /// Cancelling from inside a window stops the parallel and batched paths before
    /// the remaining windows and reports the cancellation
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs a CUDA GPU and the native weights; `just gpu-parity` runs it on gpu0"]
    fn gpu_cancellation_stops_workers() {
        /// Cancels on window 1 only, so the other workers must notice the shared
        /// cancellation rather than stop on a failure of their own
        struct CancellingHook(RecordingHook);

        impl WindowPreprocessor for CancellingHook {
            type Error = HookError;

            fn process(
                &self,
                window: &mut AudioWindowMut<'_>,
                control: &TranscriptionControl,
            ) -> Result<(), Self::Error> {
                self.0.process(window, control)?;
                if window.index().get() == 1 {
                    control.cancel();
                }
                Ok(())
            }
        }

        let (pipeline, base) = native_pipeline();
        let audio = vec![0.0; LONG_AUDIO_SAMPLES];

        for path in [ExecutionPath::Parallel, ExecutionPath::Batched] {
            let config = path.config(&base);
            let planned = planned_windows(audio.len(), &config).len();
            let hook = CancellingHook(RecordingHook(Arc::new(Mutex::new(Vec::new()))));

            let error = pipeline
                .run_with_preprocessor(&audio, &config, &recording_control(planned), &hook)
                .expect_err("cancellation should stop remaining windows");

            assert!(
                matches!(
                    error,
                    PreprocessedTranscriptionError::Transcription(TranscriptionError::Cancelled)
                ),
                "{path:?} returned {error}"
            );
            assert!(
                observed_hooks(&hook.0).len() < planned,
                "{path:?} preprocessed all {planned} windows after cancelling"
            );
        }
    }

    /// A preprocessor error stops the parallel and batched workers, which join and
    /// return that error rather than finishing the plan
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs a CUDA GPU and the native weights; `just gpu-parity` runs it on gpu0"]
    fn gpu_workers_join_after_the_first_preprocessor_error() {
        /// Fails window 1 only, so the other workers keep succeeding and must be
        /// stopped by the run; on the batched path they would otherwise park at the
        /// look-ahead gate and hang the join
        struct FailingHook(RecordingHook);

        impl WindowPreprocessor for FailingHook {
            type Error = HookError;

            fn process(
                &self,
                window: &mut AudioWindowMut<'_>,
                control: &TranscriptionControl,
            ) -> Result<(), Self::Error> {
                self.0.process(window, control)?;
                if window.index().get() == 1 {
                    return Err(HookError);
                }
                Ok(())
            }
        }

        let (pipeline, base) = native_pipeline();
        let audio = vec![0.0; LONG_AUDIO_SAMPLES];

        for path in [ExecutionPath::Parallel, ExecutionPath::Batched] {
            let config = path.config(&base);
            let planned = planned_windows(audio.len(), &config).len();
            let hook = FailingHook(RecordingHook(Arc::new(Mutex::new(Vec::new()))));

            let error = pipeline
                .run_with_preprocessor(&audio, &config, &recording_control(planned), &hook)
                .expect_err("a preprocessor failure should join every worker");

            assert!(
                matches!(
                    error,
                    PreprocessedTranscriptionError::Preprocessing(HookError)
                ),
                "{path:?} returned {error}"
            );
            let observed = observed_hooks(&hook.0).len();
            assert!(observed > 0, "{path:?} never called the preprocessor");
            assert!(
                observed < planned,
                "{path:?} preprocessed all {planned} windows after the first error"
            );
        }
    }

    /// A worker that panics on the batched path makes the run panic instead of
    /// leaving the model side waiting on workers parked at the look-ahead gate
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs a CUDA GPU and the native weights; `just gpu-parity` runs it on gpu0"]
    fn gpu_batched_worker_panic_does_not_hang() {
        struct PanickingHook;

        impl WindowPreprocessor for PanickingHook {
            type Error = HookError;

            fn process(
                &self,
                window: &mut AudioWindowMut<'_>,
                _control: &TranscriptionControl,
            ) -> Result<(), Self::Error> {
                // the model side waits on window 0, so only its preparer unwinds
                assert_ne!(window.index().get(), 0, "test hook panics on window 0");
                Ok(())
            }
        }

        let (pipeline, base) = native_pipeline();
        let config = ExecutionPath::Batched.config(&base);
        let audio = vec![0.0; LONG_AUDIO_SAMPLES];
        let planned = planned_windows(audio.len(), &config).len();
        // native_pipeline guarantees more windows than one batch, so the other
        // workers reach the gate and would park there without the panic closing it
        assert!(
            config.worker_count >= 2,
            "the batched path needs other workers"
        );

        // a regression parks the run forever, so it runs on its own thread and the
        // test fails after a bound instead of blocking the rest of the suite
        let (done, outcome) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pipeline.run_with_preprocessor(
                    &audio,
                    &config,
                    &recording_control(planned),
                    &PanickingHook,
                )
            }));
            let _ = done.send(result.is_err());
        });

        let panicked = outcome
            .recv_timeout(std::time::Duration::from_secs(120))
            .expect("the batched run should end after a worker panics, not hang");
        assert!(panicked, "the batched run should raise the worker's panic");
    }

    /// Speech probabilities of the reference recording, from the native VAD and
    /// from Silero itself
    #[cfg(all(feature = "cuda", feature = "vad"))]
    fn reference_probabilities() -> (Vec<f32>, Vec<f32>, usize) {
        let reference = NemoReference::silero();
        let audio = reference.f32("audio");
        let audio = audio.as_slice().expect("contiguous audio");
        let vad = SileroVad::from_bundle(&nemo_parity_bundle(ParakeetDeployment::v3()))
            .expect("native VAD should load")
            .expect("the native bundle should have a VAD model");
        let ours = vad
            .process(
                audio,
                &LongFormConfig::default().vad,
                &TranscriptionControl::internal_unbounded(),
            )
            .expect("VAD scan");
        (
            ours,
            reference.f32("probs").into_iter().collect(),
            audio.len(),
        )
    }

    /// Regions and planned ASR windows `(start, end)` for one set of probabilities
    #[cfg(all(feature = "cuda", feature = "vad"))]
    fn speech_plan(
        probabilities: &[f32],
        audio_len: usize,
        config: &LongFormConfig,
    ) -> (Vec<SampleRange>, Vec<(usize, usize)>) {
        let regions = detect_speech_regions(
            probabilities,
            audio_len,
            config.segmentation.threshold(config.vad.default_threshold),
            &config.segmentation,
        );
        let mut tasks = Vec::new();
        for region in &regions {
            build_region_plan(*region, probabilities, config, &mut tasks);
        }
        let windows = tasks
            .iter()
            .map(|task| (task.audio_start, task.audio_end))
            .collect();
        (regions, windows)
    }

    /// The native VAD's probabilities plan the same speech regions and ASR windows
    /// as Silero's own on the reference recording, with the native defaults and with
    /// 3 s windows that make the planner split the region on its probabilities
    #[cfg(all(feature = "cuda", feature = "vad"))]
    #[test]
    #[ignore = "reads Silero references from /shared/datasets/parakeet-ref"]
    fn nemo_parity_native_vad_plans_the_reference_windows() {
        let (ours, expected, audio_len) = reference_probabilities();
        let defaults = LongFormConfig::for_bundle(&nemo_parity_bundle(ParakeetDeployment::v3()));
        let mut short_windows = defaults.clone();
        short_windows.overlap.max_model_samples = 48_000;

        for config in [defaults, short_windows] {
            let plan = speech_plan(&ours, audio_len, &config);
            assert!(!plan.0.is_empty(), "the reference recording is speech");
            assert_eq!(plan, speech_plan(&expected, audio_len, &config));
        }
    }

    /// Reference speech, 20 s of silence, speech, silence, speech: past the native
    /// single-input limit, with gaps the VAD must leave out
    #[cfg(all(feature = "cuda", feature = "vad"))]
    fn sparse_speech() -> (Vec<f32>, Vec<usize>) {
        let speech = NemoReference::silero().f32("audio");
        let speech = speech.as_slice().expect("contiguous audio");
        let gap = vec![0.0; 20 * SAMPLE_RATE];
        let audio = [speech, &gap, speech, &gap, speech].concat();
        let gap_midpoints = (1..=2)
            .map(|gap_index| gap_index * speech.len() + (2 * gap_index - 1) * gap.len() / 2)
            .collect();
        (audio, gap_midpoints)
    }

    /// The native pipeline in VAD mode, otherwise configured as [`native_pipeline`]
    #[cfg(all(feature = "cuda", feature = "vad"))]
    fn native_vad_pipeline() -> (LongFormTranscriptionPipeline, LongFormConfig) {
        let (pipeline, config) = native_pipeline();
        let config = LongFormConfig {
            mode: LongFormMode::Vad,
            ..config
        };
        (pipeline, config)
    }

    /// Windows the VAD plan for `audio` sends to the model, as `(index, start, end)`
    #[cfg(all(feature = "cuda", feature = "vad"))]
    fn planned_vad_windows(
        pipeline: &LongFormTranscriptionPipeline,
        audio: &[f32],
        config: &LongFormConfig,
    ) -> Vec<(usize, usize, usize)> {
        let vad = pipeline
            .vad
            .as_ref()
            .expect("the native bundle has a VAD model");
        let probabilities = vad
            .process(
                audio,
                &config.vad,
                &TranscriptionControl::internal_unbounded(),
            )
            .expect("VAD scan");
        let (_, windows) = speech_plan(&probabilities, audio.len(), config);
        windows
            .into_iter()
            .enumerate()
            .map(|(index, (start, end))| (index, start, end))
            .collect()
    }

    /// In VAD mode every path preprocesses exactly the planned speech windows, and
    /// none of them reaches into the middle of a silent gap
    #[cfg(all(feature = "cuda", feature = "vad"))]
    #[test]
    #[ignore = "needs a CUDA GPU and the native weights; `just gpu-parity` runs it on gpu0"]
    fn gpu_vad_runs_the_preprocessor_once_per_planned_speech_window() {
        let (pipeline, base) = native_vad_pipeline();
        let (audio, gap_midpoints) = sparse_speech();
        assert!(audio.len() > base.transcription.max_audio_samples);

        for path in [
            ExecutionPath::Serial,
            ExecutionPath::Parallel,
            ExecutionPath::Batched,
        ] {
            let config = path.config(&base);
            let planned = planned_vad_windows(&pipeline, &audio, &config);
            assert!(planned.len() >= 3, "{path:?} planned {planned:?}");
            for midpoint in &gap_midpoints {
                assert!(
                    planned
                        .iter()
                        .all(|(_, start, end)| end <= midpoint || start >= midpoint),
                    "{path:?} planned a window across the gap at {midpoint}: {planned:?}"
                );
            }
            let hook = RecordingHook(Arc::new(Mutex::new(Vec::new())));

            let result = pipeline
                .run_with_preprocessor(&audio, &config, &recording_control(planned.len()), &hook)
                .unwrap_or_else(|error| panic!("{path:?} run failed: {error}"));

            let mut observed = observed_hooks(&hook);
            if !matches!(path, ExecutionPath::Serial) {
                observed.sort_unstable();
            }
            assert_eq!(observed, planned, "{path:?} preprocessed windows");
            assert!(!result.text.is_empty(), "{path:?} transcribed no speech");
        }
    }

    /// Silence plans no speech, so no path calls the preprocessor or the model
    #[cfg(all(feature = "cuda", feature = "vad"))]
    #[test]
    #[ignore = "needs a CUDA GPU and the native weights; `just gpu-parity` runs it on gpu0"]
    fn gpu_vad_skips_the_model_on_silence() {
        let (pipeline, base) = native_vad_pipeline();
        let audio = vec![0.0; LONG_AUDIO_SAMPLES];

        for path in [
            ExecutionPath::Serial,
            ExecutionPath::Parallel,
            ExecutionPath::Batched,
        ] {
            let config = path.config(&base);
            let hook = RecordingHook(Arc::new(Mutex::new(Vec::new())));

            let result = pipeline
                .run_with_preprocessor(&audio, &config, &recording_control(1), &hook)
                .unwrap_or_else(|error| panic!("{path:?} run failed: {error}"));

            assert!(observed_hooks(&hook).is_empty(), "{path:?} ran windows");
            assert!(result.text.is_empty(), "{path:?} transcribed silence");
        }
    }
}
