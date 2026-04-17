mod merge;
mod planner;
#[cfg(feature = "vad")]
mod vad;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, sync_channel};
use std::thread;

use crate::config::TranscriptionConfig;
use crate::constants::SAMPLE_RATE;
use crate::decode::RawTranscription;
use crate::error::TranscriptionError;
use crate::models::ModelBundle;
use crate::pipeline::{ChunkPreparer, TranscriptionPipeline};
use crate::types::{TimedToken, TranscriptChunk, TranscriptionResult};

pub use planner::OverlapChunkConfig;
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
/// windows and runs up to 4 workers in parallel
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
    /// Parallel chunk workers used for long-form transcription
    pub worker_count: usize,
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
        Self {
            mode: LongFormMode::default(),
            worker_count: 4,
            transcription: TranscriptionConfig::default(),
            #[cfg(feature = "vad")]
            vad: VadConfig::default(),
            #[cfg(feature = "vad")]
            segmentation: VadSegmentationConfig::default(),
            overlap: OverlapChunkConfig::default(),
        }
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
        let bundle = ModelBundle::from_dir(models_dir);
        Self::from_bundle(bundle)
    }

    /// Build a long-form pipeline from a resolved model bundle
    pub fn from_bundle(bundle: ModelBundle) -> Result<Self, TranscriptionError> {
        bundle.validate_base()?;
        let inner = TranscriptionPipeline::from_bundle(bundle.clone())?;
        Ok(Self {
            inner,
            #[cfg(feature = "vad")]
            vad: load_vad(bundle.vad_dir())?,
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
        match config.mode {
            LongFormMode::Fast => self.run_fast_long_form(audio, config),
            #[cfg(feature = "vad")]
            LongFormMode::Vad => self.run_vad_long_form(audio, config),
        }
    }

    #[cfg(feature = "vad")]
    fn run_vad_long_form(
        &self,
        audio: &[f32],
        config: &LongFormConfig,
    ) -> Result<TranscriptionResult, TranscriptionError> {
        let vad = self
            .vad
            .as_ref()
            .ok_or_else(|| TranscriptionError::MissingModelAsset {
                path: self.inner.bundle().vad_dir().to_path_buf(),
            })?;
        let probabilities = vad.process(audio, &config.vad)?;
        let regions = detect_speech_regions(
            &probabilities,
            audio.len(),
            config.segmentation.threshold(config.vad.default_threshold),
            &config.segmentation,
        );
        if regions.is_empty() {
            return Ok(TranscriptionResult::empty(duration_seconds(audio.len())));
        }

        self.execute_plan(
            audio,
            self.build_execution_plan(&regions, &probabilities, config),
            config,
        )
    }

    fn run_fast_long_form(
        &self,
        audio: &[f32],
        config: &LongFormConfig,
    ) -> Result<TranscriptionResult, TranscriptionError> {
        self.execute_plan(
            audio,
            Self::build_fast_execution_plan(audio.len(), config),
            config,
        )
    }

    fn execute_plan(
        &self,
        audio: &[f32],
        execution_plan: LongFormExecutionPlan,
        config: &LongFormConfig,
    ) -> Result<TranscriptionResult, TranscriptionError> {
        if execution_plan.tasks.is_empty() {
            return Ok(TranscriptionResult::empty(duration_seconds(audio.len())));
        }
        let worker_count = parallel_worker_count(execution_plan.tasks.len(), config.worker_count);
        if worker_count <= 1 {
            return self.run_serial_plan(audio, execution_plan, config);
        }

        self.run_parallel_regions(audio, execution_plan, config, worker_count)
    }

    fn run_serial_plan(
        &self,
        audio: &[f32],
        execution_plan: LongFormExecutionPlan,
        config: &LongFormConfig,
    ) -> Result<TranscriptionResult, TranscriptionError> {
        let LongFormExecutionPlan { tasks, regions } = execution_plan;
        let mut raw_tasks = vec![None; tasks.len()];
        for (task_idx, task) in tasks.iter().enumerate() {
            raw_tasks[task_idx] = Some(self.inner.transcribe_raw(
                &audio[task.audio_start..task.audio_end],
                task.global_sample_offset,
                task.context_samples,
                &config.transcription,
            )?);
        }
        self.build_parallel_result(audio.len(), &tasks, regions, raw_tasks)
    }

    fn run_parallel_regions(
        &self,
        audio: &[f32],
        execution_plan: LongFormExecutionPlan,
        config: &LongFormConfig,
        worker_count: usize,
    ) -> Result<TranscriptionResult, TranscriptionError> {
        let LongFormExecutionPlan { tasks, regions } = execution_plan;
        let queue_capacity = (worker_count * 2).max(1);

        thread::scope(|scope| -> Result<TranscriptionResult, TranscriptionError> {
            let next_task = Arc::new(AtomicUsize::new(0));
            let (sender, receiver) =
                sync_channel::<Result<CompletedTask, TranscriptionError>>(queue_capacity);

            for _ in 0..worker_count {
                let sender = sender.clone();
                let next_task = Arc::clone(&next_task);
                let tasks = &tasks;
                let bundle = self.inner.bundle().clone();
                let transcription = &config.transcription;
                scope.spawn(move || {
                    let pipeline = match TranscriptionPipeline::from_bundle(bundle.clone()) {
                        Ok(pipeline) => pipeline,
                        Err(error) => {
                            let _ = sender.send(Err(error));
                            return;
                        }
                    };
                    let preparer = TranscriptionPipeline::chunk_preparer(transcription);
                    loop {
                        let task_idx = next_task.fetch_add(1, Ordering::Relaxed);
                        if task_idx >= tasks.len() {
                            break;
                        }

                        let result = transcribe_task(&pipeline, &preparer, audio, &tasks[task_idx])
                            .map(|raw| CompletedTask { task_idx, raw });
                        if sender.send(result).is_err() {
                            break;
                        }
                    }
                });
            }
            drop(sender);

            self.consume_prepared_tasks(audio.len(), &tasks, regions, receiver)
        })
    }

    fn consume_prepared_tasks(
        &self,
        audio_len: usize,
        tasks: &[ChunkTask],
        regions: Vec<RegionTask>,
        receiver: Receiver<Result<CompletedTask, TranscriptionError>>,
    ) -> Result<TranscriptionResult, TranscriptionError> {
        let mut raw_tasks = vec![None; tasks.len()];
        let mut pending = BTreeMap::new();
        let mut next_task = 0usize;

        while next_task < tasks.len() {
            let completed = receiver.recv().map_err(|_| {
                TranscriptionError::CoreMl(
                    "parallel chunk preparation stopped before all tasks completed".to_owned(),
                )
            })??;
            pending.insert(completed.task_idx, completed.raw);

            while let Some(raw) = pending.remove(&next_task) {
                raw_tasks[next_task] = Some(raw);
                next_task += 1;
            }
        }

        self.build_parallel_result(audio_len, tasks, regions, raw_tasks)
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
    if region_len <= config.transcription.max_audio_samples {
        return RegionTask {
            task_indices: vec![push_segment_task(tasks, region)],
            kind: RegionTaskKind::Segments,
        };
    }

    if let Some(subsegments) = plan_region_subsegments(
        region,
        region_probability_slice(probabilities, region),
        &config.segmentation,
        config.transcription.max_audio_samples,
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

fn transcribe_task(
    pipeline: &TranscriptionPipeline,
    preparer: &ChunkPreparer,
    audio: &[f32],
    task: &ChunkTask,
) -> Result<RawTranscription, TranscriptionError> {
    let prepared = preparer.prepare(&audio[task.audio_start..task.audio_end])?;
    pipeline.transcribe_prepared_raw(prepared, task.global_sample_offset, task.context_samples)
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

#[cfg(feature = "vad")]
fn load_vad(model_path: &std::path::Path) -> Result<Option<SileroVad>, TranscriptionError> {
    if !model_path.exists() {
        return Ok(None);
    }

    SileroVad::new(model_path).map(Some)
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "vad")]
    use super::{ChunkTaskKind, SampleRange, build_region_plan};
    use super::{LongFormConfig, LongFormMode, RegionTaskKind, parallel_worker_count};

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
        config.transcription.max_audio_samples = 48_000;
        config.segmentation.max_speech_duration = 1.0;
        config.segmentation.min_silence_at_max_speech = 10.0;
        config.segmentation.use_max_possible_silence_at_max_speech = false;

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
}
