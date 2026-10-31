//! Runs a long-form plan's windows serially, on parallel workers, or in batches

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, sync_channel};
use std::sync::{Condvar, Mutex, PoisonError};
use std::thread;

use super::planner::ChunkTask;
use super::{LongFormConfig, validate_finite_samples};
use crate::decode::RawTranscription;
use crate::error::TranscriptionError;
use crate::pipeline::{ChunkPreparer, PreparedWindow, TranscriptionPipeline};
use crate::preprocessing::{
    AudioWindowMut, PreprocessedTranscriptionError, SourceSampleRange, TranscriptionControl,
    TranscriptionControlError, WindowIndex, WindowPreprocessor,
};

/// One long-form run's input, shared by every execution path
pub(super) struct RunRequest<'a, P: ?Sized> {
    pub(super) audio: &'a [f32],
    pub(super) config: &'a LongFormConfig,
    pub(super) control: &'a TranscriptionControl,
    pub(super) preprocessor: &'a P,
}

// derived impls would require `P: Clone`, but only references are copied
impl<P: ?Sized> Clone for RunRequest<'_, P> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<P: ?Sized> Copy for RunRequest<'_, P> {}

impl<'a, P: ?Sized> RunRequest<'a, P> {
    /// Context for running this request's windows through `pipeline`
    pub(super) fn task_context<'p>(self, pipeline: &'p TranscriptionPipeline) -> TaskContext<'p, P>
    where
        'a: 'p,
    {
        TaskContext {
            pipeline,
            preparer: pipeline.chunk_spec().preparer(),
            request: self,
        }
    }
}

/// Runs the plan's windows on the path the config and backend select
///
/// Returns each window's raw transcription in plan order
pub(super) fn execute_plan<P: WindowPreprocessor + ?Sized>(
    pipeline: &TranscriptionPipeline,
    request: RunRequest<'_, P>,
    tasks: &[ChunkTask],
) -> Result<Vec<Option<RawTranscription>>, PreprocessedTranscriptionError<P::Error>> {
    // a host-loop backend runs a batch one window at a time with no control
    // checks between them, so it stays on the per-window paths
    if request.config.batch_size > 1 && pipeline.batches_on_device() {
        return run_batched_plan(pipeline, request, tasks);
    }
    let worker_count = parallel_worker_count(tasks.len(), request.config.worker_count);
    if worker_count <= 1 {
        return run_serial_plan(pipeline, request, tasks);
    }

    run_parallel_plan(pipeline, request, tasks, worker_count)
}

fn run_serial_plan<P: WindowPreprocessor + ?Sized>(
    pipeline: &TranscriptionPipeline,
    request: RunRequest<'_, P>,
    tasks: &[ChunkTask],
) -> Result<Vec<Option<RawTranscription>>, PreprocessedTranscriptionError<P::Error>> {
    let context = request.task_context(pipeline);
    let mut raw_tasks = vec![None; tasks.len()];
    for (task_idx, task) in tasks.iter().enumerate() {
        request.control.check()?;
        raw_tasks[task_idx] = Some(transcribe_task(&context, task, WindowIndex::new(task_idx))?);
    }
    request.control.check()?;
    Ok(raw_tasks)
}

/// Runs the plan's windows `batch_size` at a time through the pipeline's model
///
/// `worker_count` threads prepare windows on the CPU ahead of the model, so the
/// next batch is ready while the model runs the current one; on a 1 h recording
/// preparation was 30 % of the wall time when it ran in between, and the first
/// batch still waits on it. Workers stay at most one batch ahead of the model,
/// which bounds the prepared windows held in memory
fn run_batched_plan<P: WindowPreprocessor + ?Sized>(
    pipeline: &TranscriptionPipeline,
    request: RunRequest<'_, P>,
    tasks: &[ChunkTask],
) -> Result<Vec<Option<RawTranscription>>, PreprocessedTranscriptionError<P::Error>> {
    let RunRequest {
        config, control, ..
    } = request;
    let batch_size = config.batch_size;
    let worker_count = parallel_worker_count(tasks.len(), config.worker_count);
    let next_task = AtomicUsize::new(0);
    let lookahead = Lookahead::new(batch_size);
    // the pipeline is not `Sync`, so each worker builds its preparer from this
    let chunk = pipeline.chunk_spec();

    thread::scope(
        |scope| -> Result<Vec<_>, PreprocessedTranscriptionError<P::Error>> {
            let (sender, receiver) = sync_channel(batch_size);
            for _ in 0..worker_count {
                let sender = sender.clone();
                let (next_task, lookahead) = (&next_task, &lookahead);
                scope.spawn(move || {
                    // a panicking worker closes the gate, so the others stop and
                    // drop their senders, the model side sees the channel close,
                    // and the scope's join raises the panic instead of hanging
                    let _stop_on_panic = StopOnPanic(lookahead);
                    let preparer = chunk.preparer();
                    loop {
                        let task_idx = next_task.fetch_add(1, Ordering::Relaxed);
                        let Some(task) = tasks.get(task_idx) else {
                            break;
                        };
                        if !lookahead.wait_for(task_idx) {
                            break;
                        }
                        let window = prepare_window(&request, &preparer, task, task_idx);
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
                        let window = ready.remove(&task_idx).ok_or_else(|| {
                            TranscriptionError::InvalidModelOutput(format!(
                                "long-form window {task_idx} was lost before its batch"
                            ))
                        })?;
                        windows.push(window?);
                    }
                    // the next batch prepares while this one runs
                    lookahead.allow(batch_end + batch_size);

                    control.check()?;
                    let raws = pipeline
                        .transcribe_prepared_batch(&windows, config.transcription.language)?;
                    raw_tasks.extend(raws.into_iter().map(Some));
                    control.check()?;
                }
                Ok(raw_tasks)
            };
            let result = consume();

            // the scope joins the workers next: stop them rather than wait for
            // windows nobody will read, as the parallel path does after an error
            if let Err(error) = &result
                && !is_interruption(error)
            {
                control.cancel();
            }
            lookahead.stop();
            drop(receiver);
            result
        },
    )
}

/// Runs the plan's windows on `worker_count` threads, each with its own model
fn run_parallel_plan<P: WindowPreprocessor + ?Sized>(
    pipeline: &TranscriptionPipeline,
    request: RunRequest<'_, P>,
    tasks: &[ChunkTask],
    worker_count: usize,
) -> Result<Vec<Option<RawTranscription>>, PreprocessedTranscriptionError<P::Error>> {
    let control = request.control;
    let queue_capacity = (worker_count * 2).max(1);
    let next_task = AtomicUsize::new(0);

    thread::scope(|scope| {
        let (sender, receiver) = sync_channel::<
            Result<CompletedTask, PreprocessedTranscriptionError<P::Error>>,
        >(queue_capacity);

        for _ in 0..worker_count {
            let sender = sender.clone();
            let next_task = &next_task;
            // the pipeline is not shared across threads, so each worker builds its
            // own from the bundle
            let bundle = pipeline.bundle().clone();
            scope.spawn(move || {
                let pipeline = match TranscriptionPipeline::from_bundle(bundle) {
                    Ok(pipeline) => pipeline,
                    Err(error) => {
                        control.cancel();
                        let _ = sender.send(Err(PreprocessedTranscriptionError::from(error)));
                        return;
                    }
                };
                let context = request.task_context(&pipeline);
                loop {
                    let task_idx = next_task.fetch_add(1, Ordering::Relaxed);
                    if task_idx >= tasks.len() {
                        break;
                    }

                    if control.check().is_err() {
                        break;
                    }
                    let result =
                        transcribe_task(&context, &tasks[task_idx], WindowIndex::new(task_idx))
                            .map(|raw| CompletedTask { task_idx, raw });
                    if let Err(error) = &result
                        && !is_interruption(error)
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

        collect_completed_tasks(tasks.len(), receiver, control)
    })
}

/// Gathers the parallel workers' windows into plan order, or the error that
/// stopped them
fn collect_completed_tasks<E>(
    task_count: usize,
    receiver: Receiver<Result<CompletedTask, PreprocessedTranscriptionError<E>>>,
    control: &TranscriptionControl,
) -> Result<Vec<Option<RawTranscription>>, PreprocessedTranscriptionError<E>>
where
    E: std::error::Error + Send + Sync + 'static,
{
    let mut raw_tasks = vec![None; task_count];
    let mut first_error = None;

    while let Ok(message) = receiver.recv() {
        match message {
            Ok(completed) => raw_tasks[completed.task_idx] = Some(completed.raw),
            Err(error) => remember_parallel_error(&mut first_error, error),
        }
    }

    if let Some(error) = first_error {
        return Err(error);
    }

    if raw_tasks.iter().any(Option::is_none) {
        control.check()?;
        return Err(TranscriptionError::CoreMl(
            "parallel chunk preparation stopped before all tasks completed".to_owned(),
        )
        .into());
    }

    control.check()?;
    Ok(raw_tasks)
}

#[derive(Debug)]
struct CompletedTask {
    task_idx: usize,
    raw: RawTranscription,
}

/// Lets batched-plan workers prepare a window only while it is less than a bound
/// ahead of the model, so prepared windows waiting in memory stay bounded however
/// unevenly the workers finish
struct Lookahead {
    state: Mutex<LookaheadState>,
    changed: Condvar,
}

struct LookaheadState {
    /// Windows below this index may be prepared
    allowed: usize,
    /// Whether the model side has stopped
    stopped: bool,
}

impl Lookahead {
    fn new(allowed: usize) -> Self {
        Self {
            state: Mutex::new(LookaheadState {
                allowed,
                stopped: false,
            }),
            changed: Condvar::new(),
        }
    }

    /// Blocks until window `task_idx` may be prepared; false once the model side
    /// has stopped
    fn wait_for(&self, task_idx: usize) -> bool {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        loop {
            if state.stopped {
                return false;
            }
            if task_idx < state.allowed {
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
        state.allowed = state.allowed.max(allowed);
        self.changed.notify_all();
    }

    fn stop(&self) {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .stopped = true;
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

pub(super) struct TaskContext<'a, P: ?Sized> {
    pipeline: &'a TranscriptionPipeline,
    preparer: ChunkPreparer,
    request: RunRequest<'a, P>,
}

pub(super) fn transcribe_task<P: WindowPreprocessor + ?Sized>(
    context: &TaskContext<'_, P>,
    task: &ChunkTask,
    index: WindowIndex,
) -> Result<RawTranscription, PreprocessedTranscriptionError<P::Error>> {
    let control = context.request.control;
    control.check()?;
    let samples = prepare_window_samples(&context.request, task, index)?;

    let prepared = context.preparer.prepare(&samples)?;
    control.check()?;
    let raw = context.pipeline.transcribe_prepared_raw(
        prepared,
        task.global_sample_offset,
        task.context_samples,
        context.request.config.transcription.language,
    )?;
    control.check()?;
    Ok(raw)
}

/// Prepares window `task_idx` of the plan for the model
fn prepare_window<P: WindowPreprocessor + ?Sized>(
    request: &RunRequest<'_, P>,
    preparer: &ChunkPreparer,
    task: &ChunkTask,
    task_idx: usize,
) -> Result<PreparedWindow, PreprocessedTranscriptionError<P::Error>> {
    let samples = prepare_window_samples(request, task, WindowIndex::new(task_idx))?;
    Ok(PreparedWindow {
        prepared: preparer.prepare(&samples)?,
        global_sample_offset: task.global_sample_offset,
        context_samples: task.context_samples,
    })
}

fn prepare_window_samples<P: WindowPreprocessor + ?Sized>(
    request: &RunRequest<'_, P>,
    task: &ChunkTask,
    index: WindowIndex,
) -> Result<Vec<f32>, PreprocessedTranscriptionError<P::Error>> {
    request.control.check()?;
    let source_range = SourceSampleRange::new(task.audio_start, task.audio_end);
    let mut samples = request.audio[source_range.start()..source_range.end()].to_vec();
    validate_finite_samples(&samples, source_range.start())?;

    {
        let mut window = AudioWindowMut::new(&mut samples, source_range, index);
        request
            .preprocessor
            .process(&mut window, request.control)
            .map_err(PreprocessedTranscriptionError::Preprocessing)?;
    }

    request.control.check()?;
    validate_finite_samples(&samples, source_range.start())?;
    Ok(samples)
}

/// Whether the error only reports that the run was cancelled or ran out of time,
/// rather than a failure that should stop the other workers
fn is_interruption<E>(error: &PreprocessedTranscriptionError<E>) -> bool
where
    E: std::error::Error + Send + Sync + 'static,
{
    matches!(
        error,
        PreprocessedTranscriptionError::Transcription(TranscriptionError::Control(
            TranscriptionControlError::Cancelled | TranscriptionControlError::DeadlineExpired
        ))
    )
}

/// Keeps the first error, unless it was only an interruption and a real failure
/// arrives after it
fn remember_parallel_error<E>(
    first_error: &mut Option<PreprocessedTranscriptionError<E>>,
    error: PreprocessedTranscriptionError<E>,
) where
    E: std::error::Error + Send + Sync + 'static,
{
    match first_error {
        None => *first_error = Some(error),
        Some(existing) if is_interruption(existing) && !is_interruption(&error) => {
            *first_error = Some(error);
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

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::{RunRequest, parallel_worker_count, prepare_window_samples};
    use crate::long_form::LongFormConfig;
    use crate::long_form::planner::build_fast_execution_plan;
    use crate::long_form::tests::RecordingHook;
    use crate::preprocessing::{
        TranscriptionControl, TranscriptionDeadline, WindowIndex, WindowLimit,
    };

    #[test]
    fn parallel_worker_count_respects_requested_workers() {
        assert_eq!(parallel_worker_count(0, 4), 1);
        assert_eq!(parallel_worker_count(1, 4), 1);
        assert_eq!(parallel_worker_count(3, 0), 1);
        assert_eq!(parallel_worker_count(3, 2), 2);
        assert_eq!(parallel_worker_count(3, 8), 3);
    }

    #[test]
    fn preparation_hook_runs_once_for_each_planned_window_with_source_coordinates() {
        let mut config = LongFormConfig::default();
        config.overlap.max_model_samples = 48_000;
        let audio = vec![0.0; 96_000];
        let plan = build_fast_execution_plan(audio.len(), &config);
        let control = TranscriptionControl::new(
            TranscriptionDeadline::from_now(std::time::Duration::from_secs(1)),
            WindowLimit::new(plan.tasks.len()).expect("the plan has windows"),
        );
        let observed = Arc::new(Mutex::new(Vec::new()));
        let hook = RecordingHook(Arc::clone(&observed));
        let request = RunRequest {
            audio: &audio,
            config: &config,
            control: &control,
            preprocessor: &hook,
        };

        for (index, task) in plan.tasks.iter().enumerate() {
            prepare_window_samples(&request, task, WindowIndex::new(index))
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
}
