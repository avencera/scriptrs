mod merge;
mod planner;
#[cfg(feature = "vad")]
pub(crate) mod vad;

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, sync_channel};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread;

use crate::config::TranscriptionConfig;
use crate::constants::SAMPLE_RATE;
use crate::decode::RawTranscription;
use crate::deployment::ParakeetDeployment;
use crate::error::TranscriptionError;
use crate::models::ModelBundle;
use crate::pipeline::{ChunkPreparer, PreparedWindow, TranscriptionPipeline};
use crate::preprocessing::{
    AudioWindowMut, IdentityPreprocessor, PreprocessedTranscriptionError, SourceSampleRange,
    TranscriptionControl, WindowIndex, WindowPreprocessor,
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

use self::merge::merge_overlapping_windows;
use self::planner::SampleRange;
#[cfg(feature = "vad")]
use self::planner::{detect_speech_regions, plan_region_subsegments, region_probability_slice};
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
/// It expects the same mono 16kHz `&[f32]` input as the base pipeline.
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
    #[cfg(feature = "vad")]
    /// VAD processing settings
    pub vad: VadConfig,
    #[cfg(feature = "vad")]
    /// VAD segmentation settings
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
            allow(unused_mut, reason = "only native bundles adjust the defaults")
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
    #[cfg(feature = "vad")]
    /// Detect speech with VAD before chunk planning
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

    #[cfg(feature = "online")]
    /// Download models and build a long-form pipeline
    ///
    /// With the default configuration this resolves models from the pinned
    /// `avencera/scriptrs-models` snapshot on Hugging Face. Set
    /// `SCRIPTRS_MODELS_DIR` to force a local bundle.
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

    /// Transcribe audio while preprocessing each planned ASR window once.
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

        if audio.len() <= config.transcription.max_audio_samples {
            control.check_plan(1)?;
            return self.run_single_task(audio, config, control, preprocessor);
        }

        self.run_long_form(audio, config, control, preprocessor)
    }

    /// Run the inner single-chunk pipeline directly
    ///
    /// This is useful when you want to reuse the long-form model bundle but feed
    /// already-split chunks through the base transcription path yourself.
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
        self.run_single_task(audio, config, &control, &IdentityPreprocessor)
            .map_err(identity_error)
    }

    fn run_long_form<P: WindowPreprocessor + ?Sized>(
        &self,
        audio: &[f32],
        config: &LongFormConfig,
        control: &TranscriptionControl,
        preprocessor: &P,
    ) -> Result<TranscriptionResult, PreprocessedTranscriptionError<P::Error>> {
        match config.mode {
            LongFormMode::Fast => self.run_fast_long_form(audio, config, control, preprocessor),
            #[cfg(feature = "vad")]
            LongFormMode::Vad => self.run_vad_long_form(audio, config, control, preprocessor),
        }
    }

    #[cfg(feature = "vad")]
    fn run_vad_long_form<P: WindowPreprocessor + ?Sized>(
        &self,
        audio: &[f32],
        config: &LongFormConfig,
        control: &TranscriptionControl,
        preprocessor: &P,
    ) -> Result<TranscriptionResult, PreprocessedTranscriptionError<P::Error>> {
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
        let execution_plan = self.build_execution_plan(&regions, &probabilities, config);
        control.check_plan(execution_plan.tasks.len())?;
        self.execute_plan(audio, execution_plan, config, control, preprocessor)
    }

    fn run_fast_long_form<P: WindowPreprocessor + ?Sized>(
        &self,
        audio: &[f32],
        config: &LongFormConfig,
        control: &TranscriptionControl,
        preprocessor: &P,
    ) -> Result<TranscriptionResult, PreprocessedTranscriptionError<P::Error>> {
        control.check()?;
        let execution_plan = Self::build_fast_execution_plan(audio.len(), config);
        control.check_plan(execution_plan.tasks.len())?;
        self.execute_plan(audio, execution_plan, config, control, preprocessor)
    }

    fn run_single_task<P: WindowPreprocessor + ?Sized>(
        &self,
        audio: &[f32],
        config: &LongFormConfig,
        control: &TranscriptionControl,
        preprocessor: &P,
    ) -> Result<TranscriptionResult, PreprocessedTranscriptionError<P::Error>> {
        let task = ChunkTask {
            audio_start: 0,
            audio_end: audio.len(),
            global_sample_offset: 0,
            context_samples: 0,
            #[cfg(feature = "vad")]
            kind: ChunkTaskKind::Segment {
                sample_offset: 0,
                duration_samples: audio.len(),
            },
        };
        let preparer = TranscriptionPipeline::chunk_preparer(&config.transcription);
        let context = TaskContext {
            pipeline: &self.inner,
            preparer: &preparer,
            preparation: PreparationContext {
                audio,
                control,
                preprocessor,
            },
            language: config.transcription.language,
        };
        let raw = transcribe_task(&context, &task, WindowIndex::new(0))?;
        control.check()?;

        Ok(self.inner.decode_raw(&raw, duration_seconds(audio.len())))
    }

    fn execute_plan<P: WindowPreprocessor + ?Sized>(
        &self,
        audio: &[f32],
        execution_plan: LongFormExecutionPlan,
        config: &LongFormConfig,
        control: &TranscriptionControl,
        preprocessor: &P,
    ) -> Result<TranscriptionResult, PreprocessedTranscriptionError<P::Error>> {
        control.check()?;
        if execution_plan.tasks.is_empty() {
            return Ok(TranscriptionResult::empty(duration_seconds(audio.len())));
        }
        // a host-loop backend runs a batch one window at a time with no control
        // checks between them, so it stays on the per-window paths
        if config.batch_size > 1 && self.inner.batches_on_device() {
            return self.run_batched_plan(audio, execution_plan, config, control, preprocessor);
        }
        let worker_count = parallel_worker_count(execution_plan.tasks.len(), config.worker_count);
        if worker_count <= 1 {
            return self.run_serial_plan(audio, execution_plan, config, control, preprocessor);
        }

        self.run_parallel_regions(
            audio,
            execution_plan,
            config,
            worker_count,
            control,
            preprocessor,
        )
    }

    fn run_serial_plan<P: WindowPreprocessor + ?Sized>(
        &self,
        audio: &[f32],
        execution_plan: LongFormExecutionPlan,
        config: &LongFormConfig,
        control: &TranscriptionControl,
        preprocessor: &P,
    ) -> Result<TranscriptionResult, PreprocessedTranscriptionError<P::Error>> {
        let LongFormExecutionPlan { tasks, regions } = execution_plan;
        let preparer = TranscriptionPipeline::chunk_preparer(&config.transcription);
        let context = TaskContext {
            pipeline: &self.inner,
            preparer: &preparer,
            preparation: PreparationContext {
                audio,
                control,
                preprocessor,
            },
            language: config.transcription.language,
        };
        let mut raw_tasks = vec![None; tasks.len()];
        for (task_idx, task) in tasks.iter().enumerate() {
            context.preparation.control.check()?;
            raw_tasks[task_idx] =
                Some(transcribe_task(&context, task, WindowIndex::new(task_idx))?);
        }
        control.check()?;
        Ok(self.build_parallel_result(audio.len(), &tasks, regions, raw_tasks)?)
    }

    /// Runs the plan's windows `batch_size` at a time through the pipeline's model
    ///
    /// `worker_count` threads prepare windows on the CPU ahead of the model, so the
    /// next batch is ready while the model runs the current one; on a 1 h recording
    /// preparation was 30 % of the wall time when it ran in between, and the first
    /// batch still waits on it. Workers stay at most one batch ahead of the model,
    /// which bounds the prepared windows held in memory
    fn run_batched_plan<P: WindowPreprocessor + ?Sized>(
        &self,
        audio: &[f32],
        execution_plan: LongFormExecutionPlan,
        config: &LongFormConfig,
        control: &TranscriptionControl,
        preprocessor: &P,
    ) -> Result<TranscriptionResult, PreprocessedTranscriptionError<P::Error>> {
        let LongFormExecutionPlan { tasks, regions } = execution_plan;
        let batch_size = config.batch_size;
        let worker_count = parallel_worker_count(tasks.len(), config.worker_count);
        let next_task = AtomicUsize::new(0);
        let lookahead = Lookahead::new(batch_size);

        let raw_tasks = thread::scope(
            |scope| -> Result<Vec<_>, PreprocessedTranscriptionError<P::Error>> {
                let (sender, receiver) = sync_channel(batch_size);
                for _ in 0..worker_count {
                    let sender = sender.clone();
                    let (next_task, lookahead, tasks) = (&next_task, &lookahead, &tasks);
                    scope.spawn(move || {
                        // a panicking worker closes the gate, so the others stop and
                        // drop their senders, the model side sees the channel close,
                        // and the scope's join raises the panic instead of hanging
                        let _stop_on_panic = StopOnPanic(lookahead);
                        let preparer = TranscriptionPipeline::chunk_preparer(&config.transcription);
                        let preparation = PreparationContext {
                            audio,
                            control,
                            preprocessor,
                        };
                        loop {
                            let task_idx = next_task.fetch_add(1, Ordering::Relaxed);
                            let Some(task) = tasks.get(task_idx) else {
                                break;
                            };
                            if !lookahead.wait_for(task_idx) {
                                break;
                            }
                            let window = prepare_window(&preparation, &preparer, task, task_idx);
                            let failed = window.is_err();
                            // a closed channel means the model side has stopped
                            if sender.send((task_idx, window)).is_err() || failed {
                                break;
                            }
                        }
                    });
                }
                drop(sender);
                // a panic in the model call unwinds past the cleanup below, and the
                // scope would then wait on workers parked at the gate
                let _stop_on_panic = StopOnPanic(&lookahead);

                // windows arrive in any order; batches are cut in plan order, and an
                // error surfaces when its window is reached, as on the serial path
                let consume = || -> Result<Vec<_>, PreprocessedTranscriptionError<P::Error>> {
                    let mut ready = BTreeMap::new();
                    let mut raw_tasks = Vec::with_capacity(tasks.len());
                    for batch_start in (0..tasks.len()).step_by(batch_size) {
                        let batch_end = (batch_start + batch_size).min(tasks.len());
                        let mut windows = Vec::with_capacity(batch_end - batch_start);
                        for task_idx in batch_start..batch_end {
                            while !ready.contains_key(&task_idx) {
                                let (done_idx, window) = receiver.recv().map_err(|_| {
                                    TranscriptionError::InvalidModelOutput(format!(
                                        "long-form window {task_idx} was never prepared"
                                    ))
                                })?;
                                ready.insert(done_idx, window);
                            }
                            if let Some(window) = ready.remove(&task_idx) {
                                windows.push(window?);
                            }
                        }
                        // the next batch prepares while this one runs
                        lookahead.allow(batch_end + batch_size);

                        control.check()?;
                        let raws = self
                            .inner
                            .transcribe_prepared_batch(windows, config.transcription.language)?;
                        raw_tasks.extend(raws.into_iter().map(Some));
                        control.check()?;
                    }
                    Ok(raw_tasks)
                };
                let result = consume();

                // the scope joins the workers next: stop them rather than wait for
                // windows nobody will read, as the parallel path does after an error
                if let Err(error) = &result
                    && should_cancel_after_error(error)
                {
                    control.cancel();
                }
                lookahead.stop();
                drop(receiver);
                result
            },
        )?;

        Ok(self.build_parallel_result(audio.len(), &tasks, regions, raw_tasks)?)
    }

    fn run_parallel_regions<P: WindowPreprocessor + ?Sized>(
        &self,
        audio: &[f32],
        execution_plan: LongFormExecutionPlan,
        config: &LongFormConfig,
        worker_count: usize,
        control: &TranscriptionControl,
        preprocessor: &P,
    ) -> Result<TranscriptionResult, PreprocessedTranscriptionError<P::Error>> {
        let LongFormExecutionPlan { tasks, regions } = execution_plan;
        let queue_capacity = (worker_count * 2).max(1);

        thread::scope(
            |scope| -> Result<TranscriptionResult, PreprocessedTranscriptionError<P::Error>> {
                let next_task = Arc::new(AtomicUsize::new(0));
                let (sender, receiver) = sync_channel::<
                    Result<CompletedTask, PreprocessedTranscriptionError<P::Error>>,
                >(queue_capacity);

                for _ in 0..worker_count {
                    let sender = sender.clone();
                    let next_task = Arc::clone(&next_task);
                    let tasks = &tasks;
                    let bundle = self.inner.bundle().clone();
                    let transcription = &config.transcription;
                    let control = control.clone();
                    scope.spawn(move || {
                        let pipeline = match TranscriptionPipeline::from_bundle(bundle.clone()) {
                            Ok(pipeline) => pipeline,
                            Err(error) => {
                                control.cancel();
                                let _ =
                                    sender.send(Err(PreprocessedTranscriptionError::from(error)));
                                return;
                            }
                        };
                        let preparer = TranscriptionPipeline::chunk_preparer(transcription);
                        let context = TaskContext {
                            pipeline: &pipeline,
                            preparer: &preparer,
                            preparation: PreparationContext {
                                audio,
                                control: &control,
                                preprocessor,
                            },
                            language: transcription.language,
                        };
                        loop {
                            let task_idx = next_task.fetch_add(1, Ordering::Relaxed);
                            if task_idx >= tasks.len() {
                                break;
                            }

                            if control.check().is_err() {
                                break;
                            }
                            let result = transcribe_task(
                                &context,
                                &tasks[task_idx],
                                WindowIndex::new(task_idx),
                            )
                            .map(|raw| CompletedTask { task_idx, raw });
                            if let Err(error) = &result
                                && should_cancel_after_error(error)
                            {
                                control.cancel();
                            }
                            if sender.send(result).is_err() {
                                break;
                            }
                        }
                    });
                }
                drop(sender);

                self.consume_prepared_tasks(audio.len(), &tasks, regions, receiver, control)
            },
        )
    }

    fn consume_prepared_tasks<E>(
        &self,
        audio_len: usize,
        tasks: &[ChunkTask],
        regions: Vec<RegionTask>,
        receiver: Receiver<Result<CompletedTask, PreprocessedTranscriptionError<E>>>,
        control: &TranscriptionControl,
    ) -> Result<TranscriptionResult, PreprocessedTranscriptionError<E>>
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        let mut raw_tasks = vec![None; tasks.len()];
        let mut pending = BTreeMap::new();
        let mut first_error = None;

        while let Ok(message) = receiver.recv() {
            match message {
                Ok(completed) => {
                    pending.insert(completed.task_idx, completed.raw);
                }
                Err(error) => remember_parallel_error(&mut first_error, error),
            }
        }

        if let Some(error) = first_error {
            return Err(error);
        }

        for (task_idx, raw) in pending {
            raw_tasks[task_idx] = Some(raw);
        }
        if raw_tasks.iter().any(Option::is_none) {
            control.check()?;
            return Err(TranscriptionError::CoreMl(
                "parallel chunk preparation stopped before all tasks completed".to_owned(),
            )
            .into());
        }

        control.check()?;
        Ok(self.build_parallel_result(audio_len, tasks, regions, raw_tasks)?)
    }

    fn build_parallel_result(
        &self,
        audio_len: usize,
        tasks: &[ChunkTask],
        regions: Vec<RegionTask>,
        mut raw_tasks: Vec<Option<RawTranscription>>,
    ) -> Result<TranscriptionResult, TranscriptionError> {
        #[cfg(not(feature = "vad"))]
        let _ = tasks;
        let mut tokens = Vec::new();
        let mut chunks = Vec::new();

        for region in regions {
            let region_tokens = match region.kind {
                #[cfg(feature = "vad")]
                RegionTaskKind::Segments => {
                    self.decode_segment_tasks(tasks, &region.task_indices, &mut raw_tasks)?
                }
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
            let ChunkTaskKind::Segment {
                sample_offset,
                duration_samples,
            } = task.kind
            else {
                unreachable!("segment region should only reference segment tasks");
            };
            let raw = take_raw_task(raw_tasks, *task_idx)?;
            let mut chunk_tokens = self
                .inner
                .decode_raw(&raw, duration_seconds(duration_samples))
                .tokens;
            offset_tokens(&mut chunk_tokens, sample_offset);
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

    #[cfg(feature = "vad")]
    fn build_execution_plan(
        &self,
        regions: &[SampleRange],
        probabilities: &[f32],
        config: &LongFormConfig,
    ) -> LongFormExecutionPlan {
        let mut tasks = Vec::new();
        let mut region_tasks = Vec::with_capacity(regions.len());
        for region in regions.iter().copied() {
            region_tasks.push(build_region_plan(region, probabilities, config, &mut tasks));
        }
        LongFormExecutionPlan {
            tasks,
            regions: region_tasks,
        }
    }

    fn build_fast_execution_plan(
        audio_len: usize,
        config: &LongFormConfig,
    ) -> LongFormExecutionPlan {
        let mut tasks = Vec::new();
        let full_audio = SampleRange {
            start: 0,
            end: audio_len,
        };
        let task_indices = config
            .overlap
            .plan(full_audio)
            .into_iter()
            .map(|chunk| push_overlap_task(&mut tasks, chunk, config))
            .collect();
        let regions = vec![RegionTask {
            task_indices,
            kind: RegionTaskKind::Overlap,
        }];
        LongFormExecutionPlan { tasks, regions }
    }
}

#[derive(Debug)]
struct LongFormExecutionPlan {
    tasks: Vec<ChunkTask>,
    regions: Vec<RegionTask>,
}

#[derive(Debug, Clone, Copy)]
struct ChunkTask {
    audio_start: usize,
    audio_end: usize,
    global_sample_offset: usize,
    context_samples: usize,
    #[cfg(feature = "vad")]
    kind: ChunkTaskKind,
}

#[cfg(feature = "vad")]
#[derive(Debug, Clone, Copy)]
enum ChunkTaskKind {
    #[cfg(feature = "vad")]
    Segment {
        sample_offset: usize,
        duration_samples: usize,
    },
    Overlap,
}

#[derive(Debug)]
struct RegionTask {
    task_indices: Vec<usize>,
    kind: RegionTaskKind,
}

#[derive(Debug, Clone, Copy)]
enum RegionTaskKind {
    #[cfg(feature = "vad")]
    Segments,
    Overlap,
}

#[derive(Debug)]
struct CompletedTask {
    task_idx: usize,
    raw: RawTranscription,
}

#[cfg(feature = "vad")]
fn build_region_plan(
    region: SampleRange,
    probabilities: &[f32],
    config: &LongFormConfig,
    tasks: &mut Vec<ChunkTask>,
) -> RegionTask {
    let region_len = region.end.saturating_sub(region.start);
    let max_chunk_samples = config.overlap.max_model_samples;
    if region_len <= max_chunk_samples {
        return RegionTask {
            task_indices: vec![push_segment_task(tasks, region)],
            kind: RegionTaskKind::Segments,
        };
    }

    if let Some(subsegments) = plan_region_subsegments(
        region,
        region_probability_slice(probabilities, region),
        &config.segmentation,
        max_chunk_samples,
    ) {
        return RegionTask {
            task_indices: subsegments
                .into_iter()
                .map(|subsegment| push_segment_task(tasks, subsegment))
                .collect(),
            kind: RegionTaskKind::Segments,
        };
    }

    RegionTask {
        task_indices: config
            .overlap
            .plan(region)
            .into_iter()
            .map(|chunk| push_overlap_task(tasks, chunk, config))
            .collect(),
        kind: RegionTaskKind::Overlap,
    }
}

#[cfg(feature = "vad")]
fn push_segment_task(tasks: &mut Vec<ChunkTask>, segment: SampleRange) -> usize {
    let task_idx = tasks.len();
    tasks.push(ChunkTask {
        audio_start: segment.start,
        audio_end: segment.end,
        global_sample_offset: 0,
        context_samples: 0,
        #[cfg(feature = "vad")]
        kind: ChunkTaskKind::Segment {
            sample_offset: segment.start,
            duration_samples: segment.end.saturating_sub(segment.start),
        },
    });
    task_idx
}

fn push_overlap_task(
    tasks: &mut Vec<ChunkTask>,
    chunk: SampleRange,
    config: &LongFormConfig,
) -> usize {
    let task_idx = tasks.len();
    let context_start = chunk.start.saturating_sub(config.overlap.context_samples);
    tasks.push(ChunkTask {
        audio_start: context_start,
        audio_end: chunk.end,
        global_sample_offset: chunk.start,
        context_samples: chunk.start - context_start,
        #[cfg(feature = "vad")]
        kind: ChunkTaskKind::Overlap,
    });
    task_idx
}

/// Lets batched-plan workers prepare a window only while it is less than a bound
/// ahead of the model, so prepared windows waiting in memory stay bounded however
/// unevenly the workers finish
struct Lookahead {
    /// Windows below this index may be prepared, and whether the model side stopped
    state: Mutex<(usize, bool)>,
    changed: Condvar,
}

impl Lookahead {
    fn new(allowed: usize) -> Self {
        Self {
            state: Mutex::new((allowed, false)),
            changed: Condvar::new(),
        }
    }

    /// Blocks until window `task_idx` may be prepared; false once the model side
    /// has stopped
    fn wait_for(&self, task_idx: usize) -> bool {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        loop {
            let (allowed, stopped) = *state;
            if stopped {
                return false;
            }
            if task_idx < allowed {
                return true;
            }
            state = self
                .changed
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    fn allow(&self, allowed: usize) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.0 = state.0.max(allowed);
        self.changed.notify_all();
    }

    fn stop(&self) {
        self.state.lock().unwrap_or_else(PoisonError::into_inner).1 = true;
        self.changed.notify_all();
    }
}

/// Closes a [`Lookahead`] when the code holding it unwinds
struct StopOnPanic<'a>(&'a Lookahead);

impl Drop for StopOnPanic<'_> {
    fn drop(&mut self) {
        if thread::panicking() {
            self.0.stop();
        }
    }
}

struct TaskContext<'a, P: ?Sized> {
    pipeline: &'a TranscriptionPipeline,
    preparer: &'a ChunkPreparer,
    preparation: PreparationContext<'a, P>,
    language: crate::language::LanguageSelection,
}

struct PreparationContext<'a, P: ?Sized> {
    audio: &'a [f32],
    control: &'a TranscriptionControl,
    preprocessor: &'a P,
}

fn transcribe_task<P: WindowPreprocessor + ?Sized>(
    context: &TaskContext<'_, P>,
    task: &ChunkTask,
    index: WindowIndex,
) -> Result<RawTranscription, PreprocessedTranscriptionError<P::Error>> {
    context.preparation.control.check()?;
    let samples = prepare_window_samples(&context.preparation, task, index)?;

    let prepared = context.preparer.prepare(&samples)?;
    context.preparation.control.check()?;
    let raw = context.pipeline.transcribe_prepared_raw(
        prepared,
        task.global_sample_offset,
        task.context_samples,
        context.language,
    )?;
    context.preparation.control.check()?;
    Ok(raw)
}

/// Prepares window `task_idx` of the plan for the model
fn prepare_window<P: WindowPreprocessor + ?Sized>(
    context: &PreparationContext<'_, P>,
    preparer: &ChunkPreparer,
    task: &ChunkTask,
    task_idx: usize,
) -> Result<PreparedWindow, PreprocessedTranscriptionError<P::Error>> {
    let samples = prepare_window_samples(context, task, WindowIndex::new(task_idx))?;
    Ok(PreparedWindow {
        prepared: preparer.prepare(&samples)?,
        global_sample_offset: task.global_sample_offset,
        context_samples: task.context_samples,
    })
}

fn prepare_window_samples<P: WindowPreprocessor + ?Sized>(
    context: &PreparationContext<'_, P>,
    task: &ChunkTask,
    index: WindowIndex,
) -> Result<Vec<f32>, PreprocessedTranscriptionError<P::Error>> {
    context.control.check()?;
    let source_range = SourceSampleRange::new(task.audio_start, task.audio_end);
    let mut samples = context.audio[source_range.start()..source_range.end()].to_vec();
    validate_finite_samples(&samples, source_range.start())?;

    {
        let mut window = AudioWindowMut::new(&mut samples, source_range, index);
        context
            .preprocessor
            .process(&mut window, context.control)
            .map_err(PreprocessedTranscriptionError::Preprocessing)?;
    }

    context.control.check()?;
    validate_finite_samples(&samples, source_range.start())?;
    Ok(samples)
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

fn should_cancel_after_error<E>(error: &PreprocessedTranscriptionError<E>) -> bool
where
    E: std::error::Error + Send + Sync + 'static,
{
    match error {
        PreprocessedTranscriptionError::Preprocessing(_) => true,
        PreprocessedTranscriptionError::Transcription(error) => !matches!(
            error,
            TranscriptionError::Cancelled | TranscriptionError::DeadlineExpired
        ),
    }
}

fn remember_parallel_error<E>(
    first_error: &mut Option<PreprocessedTranscriptionError<E>>,
    error: PreprocessedTranscriptionError<E>,
) where
    E: std::error::Error + Send + Sync + 'static,
{
    match first_error {
        None => *first_error = Some(error),
        Some(PreprocessedTranscriptionError::Transcription(TranscriptionError::Cancelled))
        | Some(PreprocessedTranscriptionError::Transcription(
            TranscriptionError::DeadlineExpired,
        )) => {
            if !matches!(
                error,
                PreprocessedTranscriptionError::Transcription(TranscriptionError::Cancelled)
                    | PreprocessedTranscriptionError::Transcription(
                        TranscriptionError::DeadlineExpired
                    )
            ) {
                *first_error = Some(error);
            }
        }
        Some(_) => {}
    }
}

fn parallel_worker_count(task_count: usize, requested_workers: usize) -> usize {
    if task_count <= 1 {
        return 1;
    }

    requested_workers.max(1).min(task_count)
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

    #[cfg(feature = "vad")]
    use super::{ChunkTaskKind, SampleRange, build_region_plan};
    use super::{
        LongFormConfig, LongFormMode, PreparationContext, RegionTaskKind, WindowPreprocessor,
        parallel_worker_count, prepare_window_samples,
    };
    #[cfg(feature = "cuda")]
    use super::LongFormTranscriptionPipeline;
    #[cfg(feature = "cuda")]
    use crate::cuda::test_support::nemo_parity_bundle;
    use crate::deployment::ParakeetDeployment;
    #[cfg(feature = "cuda")]
    use crate::error::TranscriptionError;
    #[cfg(feature = "cuda")]
    use crate::preprocessing::PreprocessedTranscriptionError;
    use crate::preprocessing::{
        AudioWindowMut, TranscriptionControl, TranscriptionDeadline, WindowLimit,
    };

    #[derive(Debug)]
    struct HookError;

    impl Display for HookError {
        fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("test hook failed")
        }
    }

    impl std::error::Error for HookError {}

    #[derive(Debug, Clone)]
    struct RecordingHook(Arc<Mutex<Vec<(usize, usize, usize)>>>);

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

    #[cfg(feature = "vad")]
    #[test]
    fn short_region_stays_as_a_single_segment_task() {
        let config = LongFormConfig::default();
        let mut tasks = Vec::new();
        let region = SampleRange {
            start: 16_000,
            end: 32_000,
        };

        let plan = build_region_plan(region, &[], &config, &mut tasks);

        assert!(matches!(plan.kind, RegionTaskKind::Segments));
        assert_eq!(plan.task_indices, vec![0]);
        assert_eq!(tasks.len(), 1);
        assert!(matches!(
            tasks[0].kind,
            ChunkTaskKind::Segment {
                sample_offset: 16_000,
                duration_samples: 16_000,
            }
        ));
    }

    #[cfg(feature = "vad")]
    #[test]
    fn overlap_plan_keeps_window_order_and_context_offsets() {
        let mut config = LongFormConfig::default();
        config.overlap.max_model_samples = 48_000;
        config.segmentation.min_silence_at_max_speech = 10.0;

        let mut tasks = Vec::new();
        let region = SampleRange {
            start: 0,
            end: 96_000,
        };
        let probabilities = vec![1.0; region.end.div_ceil(512)];

        let plan = build_region_plan(region, &probabilities, &config, &mut tasks);

        assert!(matches!(plan.kind, RegionTaskKind::Overlap));
        assert_eq!(plan.task_indices.len(), tasks.len());
        assert!(
            tasks
                .windows(2)
                .all(|window| window[0].global_sample_offset <= window[1].global_sample_offset)
        );
        assert!(
            tasks
                .iter()
                .all(|task| task.context_samples <= task.global_sample_offset)
        );
    }

    #[test]
    fn parallel_worker_count_respects_requested_workers() {
        assert_eq!(parallel_worker_count(0, 4), 1);
        assert_eq!(parallel_worker_count(1, 4), 1);
        assert_eq!(parallel_worker_count(3, 0), 1);
        assert_eq!(parallel_worker_count(3, 2), 2);
        assert_eq!(parallel_worker_count(3, 8), 3);
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
    fn fast_execution_plan_uses_one_overlap_region_for_full_audio() {
        let config = LongFormConfig {
            mode: LongFormMode::Fast,
            ..LongFormConfig::default()
        };
        let plan = super::LongFormTranscriptionPipeline::build_fast_execution_plan(96_000, &config);

        assert_eq!(plan.regions.len(), 1);
        assert!(matches!(plan.regions[0].kind, RegionTaskKind::Overlap));
        assert_eq!(plan.regions[0].task_indices.len(), plan.tasks.len());
        assert!(!plan.tasks.is_empty());
    }

    #[test]
    fn preparation_hook_runs_once_for_each_planned_window_with_source_coordinates() {
        let mut config = LongFormConfig::default();
        config.overlap.max_model_samples = 48_000;
        let audio = vec![0.0; 96_000];
        let plan =
            super::LongFormTranscriptionPipeline::build_fast_execution_plan(audio.len(), &config);
        let control = TranscriptionControl::new(
            TranscriptionDeadline::from_now(std::time::Duration::from_secs(1)),
            WindowLimit::new(plan.tasks.len()).expect("the plan has windows"),
        );
        let observed = Arc::new(Mutex::new(Vec::new()));
        let hook = RecordingHook(Arc::clone(&observed));
        let context = PreparationContext {
            audio: &audio,
            control: &control,
            preprocessor: &hook,
        };

        for (index, task) in plan.tasks.iter().enumerate() {
            prepare_window_samples(&context, task, super::WindowIndex::new(index))
                .expect("the recording hook should succeed");
        }

        let observed = observed
            .lock()
            .expect("test observer should not be poisoned")
            .clone();
        assert_eq!(observed.len(), plan.tasks.len());
        assert_eq!(
            observed,
            plan.tasks
                .iter()
                .enumerate()
                .map(|(index, task)| (index, task.audio_start, task.audio_end))
                .collect::<Vec<_>>()
        );
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

    #[cfg(feature = "vad")]
    #[test]
    fn hook_source_range_keeps_vad_segment_position_separate_from_decoder_offset() {
        let config = LongFormConfig::default();
        let mut tasks = Vec::new();
        let plan = build_region_plan(
            SampleRange {
                start: 16_000,
                end: 32_000,
            },
            &[],
            &config,
            &mut tasks,
        );
        let task = tasks[plan.task_indices[0]];

        assert_eq!(task.audio_start, 16_000);
        assert_eq!(task.audio_end, 32_000);
        assert_eq!(task.global_sample_offset, 0);
        assert_eq!(task.context_samples, 0);
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

        let plan =
            LongFormTranscriptionPipeline::build_fast_execution_plan(LONG_AUDIO_SAMPLES, &config);
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
        LongFormTranscriptionPipeline::build_fast_execution_plan(audio_len, config)
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
}
