//! Shared batching for independent concurrent requests
//!
//! [`TranscriptionPipeline::into_scheduler`] (and, with `long-form`,
//! `LongFormTranscriptionPipeline::into_scheduler`) moves a pipeline behind a
//! [`TranscriptionScheduler`] that any number of threads submit to. Each request is
//! planned into windows on the submitting thread. Worker threads compute the
//! windows' features, oldest request first, and sort them into length buckets. One
//! model thread encodes a bucket's windows together, whichever requests they came
//! from, and decodes every window in flight in one batch whose rows take new
//! windows as soon as earlier ones finish. When a request's last window finishes,
//! the thread waiting on it merges the windows as the pipeline's own `run` does
//!
//! On CUDA, an encoder batch pads the windows it holds only to the next power of
//! two, and a decode step runs only the rows in use, so a lone request runs about
//! as fast as the pipeline's `run` and a busy scheduler fills its batches. Core ML
//! bundles run each window alone as the model thread reaches it

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use crate::backend::EncoderChunk;
use crate::config::TranscriptionConfig;
use crate::decode::RawTranscription;
use crate::error::TranscriptionError;
use crate::model::{BatcherShape, Bucket, ModelBatcher};
use crate::pipeline::{ChunkSpec, PreparedChunk, TranscriptionPipeline};
use crate::types::TranscriptionResult;

/// Options for a [`TranscriptionScheduler`]
///
/// Batch shapes follow the windows waiting, so a request's transcript can differ
/// from [`TranscriptionPipeline::run`] in a rare token, as any change of batch
/// shape can
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerConfig {
    /// Unfinished requests the scheduler holds at once
    ///
    /// Past this, [`TranscriptionScheduler::submit`] waits for a request to finish
    /// or be dropped, and [`TranscriptionScheduler::try_submit`] returns
    /// [`TranscriptionError::SchedulerFull`]. Finished results waiting for
    /// [`PendingTranscription::wait`] do not count. The default keeps a few seconds
    /// of short clips queued on an RTX 5090, far more than its batches need
    pub max_requests: usize,
    /// Windows the decoder works on at once; a finished window's row takes the
    /// next window between decode steps
    ///
    /// A decode step runs the rows up to the last busy one, rounded up to a power
    /// of two, so rows cost nothing until windows fill them
    pub decoder_rows: usize,
    /// Mel frames (10 ms each) one full encoder batch covers
    ///
    /// Windows are padded to the nearest of a ladder of length buckets, and each
    /// bucket encodes up to `frames_per_batch / bucket length` windows at a time
    /// (at least [`Self::min_batch`], at most [`Self::max_batch`]); fewer run when
    /// fewer wait. Larger batches run more windows per second but take longer, so
    /// the windows waiting behind them start later. The defaults were tuned on an
    /// RTX 5090
    pub frames_per_batch: usize,
    /// Fewest windows a bucket's encoder batch may grow to, however long the
    /// bucket
    ///
    /// Long windows mostly come many at a time from one long request, so their
    /// batches fill even without other requests
    pub min_batch: usize,
    /// Most windows one encoder batch of the shortest buckets holds
    pub max_batch: usize,
    /// Prepared windows each length bucket holds while they wait for the model;
    /// workers pause once a bucket is full, which bounds the features in memory
    pub bucket_capacity: usize,
    /// Threads that compute window features ahead of the model
    pub workers: usize,
    /// Longest a part-full encoder batch waits for more windows while the decoder
    /// has other work; a full batch, or any batch when the decoder is idle, starts
    /// at once
    pub batch_wait: Duration,
}

impl SchedulerConfig {
    /// Default for [`Self::max_requests`]
    pub const DEFAULT_MAX_REQUESTS: usize = 1024;
    /// Default for [`Self::decoder_rows`]
    pub const DEFAULT_DECODER_ROWS: usize = 64;
    /// Default for [`Self::frames_per_batch`]
    pub const DEFAULT_FRAMES_PER_BATCH: usize = 8192;
    /// Default for [`Self::min_batch`]
    pub const DEFAULT_MIN_BATCH: usize = 6;
    /// Default for [`Self::max_batch`]
    pub const DEFAULT_MAX_BATCH: usize = 32;
    /// Default for [`Self::bucket_capacity`]
    pub const DEFAULT_BUCKET_CAPACITY: usize = 64;
    /// Default for [`Self::workers`]
    pub const DEFAULT_WORKERS: usize = 4;
    /// Default for [`Self::batch_wait`]
    pub const DEFAULT_BATCH_WAIT: Duration = Duration::from_millis(5);

    fn validate(&self) -> Result<(), TranscriptionError> {
        let invalid = |message: &str| Err(TranscriptionError::InvalidConfiguration(message.into()));
        if self.max_requests == 0
            || self.decoder_rows == 0
            || self.frames_per_batch == 0
            || self.min_batch == 0
            || self.max_batch == 0
            || self.bucket_capacity == 0
            || self.workers == 0
        {
            return invalid("scheduler sizes must be at least 1");
        }
        if self.min_batch > self.max_batch {
            return invalid("scheduler min_batch must not exceed max_batch");
        }
        if self.decoder_rows < self.max_batch {
            return invalid("scheduler decoder_rows must hold at least max_batch windows");
        }
        Ok(())
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_requests: Self::DEFAULT_MAX_REQUESTS,
            decoder_rows: Self::DEFAULT_DECODER_ROWS,
            frames_per_batch: Self::DEFAULT_FRAMES_PER_BATCH,
            min_batch: Self::DEFAULT_MIN_BATCH,
            max_batch: Self::DEFAULT_MAX_BATCH,
            bucket_capacity: Self::DEFAULT_BUCKET_CAPACITY,
            workers: Self::DEFAULT_WORKERS,
            batch_wait: Self::DEFAULT_BATCH_WAIT,
        }
    }
}

/// A span of a request's audio that runs as one model window, and where its
/// tokens land in the request
#[derive(Debug, Clone, Copy)]
pub(crate) struct WindowSpan {
    pub start: usize,
    pub end: usize,
    /// Request sample the window's own audio starts at
    pub global_sample_offset: usize,
    /// Left-context samples before that start, whose tokens are dropped
    pub context_samples: usize,
}

/// How requests become windows and windows become results
#[derive(Clone)]
enum Planner {
    Single {
        pipeline: Box<TranscriptionPipeline>,
        config: TranscriptionConfig,
    },
    #[cfg(feature = "long-form")]
    LongForm {
        pipeline: Box<crate::long_form::LongFormTranscriptionPipeline>,
        config: crate::long_form::LongFormConfig,
    },
}

/// A request's plan, kept until its windows merge
enum Plan {
    Single,
    #[cfg(feature = "long-form")]
    LongForm(crate::long_form::ScheduledPlan),
}

impl Planner {
    fn pipeline(&self) -> &TranscriptionPipeline {
        match self {
            Self::Single { pipeline, .. } => pipeline,
            #[cfg(feature = "long-form")]
            Self::LongForm { pipeline, .. } => pipeline.inner(),
        }
    }

    fn transcription(&self) -> &TranscriptionConfig {
        match self {
            Self::Single { config, .. } => config,
            #[cfg(feature = "long-form")]
            Self::LongForm { config, .. } => &config.transcription,
        }
    }

    fn plan(&self, audio: &[f32]) -> Result<(Plan, Vec<WindowSpan>), TranscriptionError> {
        match self {
            Self::Single { pipeline, .. } => {
                // the same checks `run` makes, before any work is queued
                pipeline.chunk_spec().check(audio)?;
                let window = WindowSpan {
                    start: 0,
                    end: audio.len(),
                    global_sample_offset: 0,
                    context_samples: 0,
                };
                Ok((Plan::Single, vec![window]))
            }
            #[cfg(feature = "long-form")]
            Self::LongForm { pipeline, config } => {
                let plan = pipeline.plan_scheduled(audio, config)?;
                let windows = plan.windows(audio.len());
                Ok((Plan::LongForm(plan), windows))
            }
        }
    }

    fn merge(
        &self,
        audio_len: usize,
        plan: Plan,
        raws: Vec<RawTranscription>,
    ) -> Result<TranscriptionResult, TranscriptionError> {
        let duration_seconds = self.pipeline().chunk_spec().duration_seconds(audio_len);
        match (self, plan) {
            (Self::Single { pipeline, .. }, Plan::Single) => {
                let raw = raws.into_iter().next().ok_or_else(|| {
                    TranscriptionError::InvalidModelOutput("a scheduled window went missing".into())
                })?;
                Ok(pipeline.decode_raw(&raw, duration_seconds))
            }
            #[cfg(feature = "long-form")]
            (Self::LongForm { pipeline, .. }, Plan::LongForm(plan)) => {
                pipeline.merge_scheduled(audio_len, plan, raws)
            }
            #[cfg(feature = "long-form")]
            _ => Err(TranscriptionError::InvalidModelOutput(
                "a scheduled plan does not match its planner".into(),
            )),
        }
    }
}

/// Planners for the threads that plan and merge requests at once
///
/// A pipeline's feature extractor keeps a scratch workspace, so a pipeline serves
/// one thread at a time; each caller takes an idle clone, and the first stays as
/// the template for more
struct Planners {
    idle: Mutex<Vec<Planner>>,
}

impl Planners {
    fn new(planner: Planner) -> Self {
        Self {
            idle: Mutex::new(vec![planner]),
        }
    }

    fn with<R>(&self, use_planner: impl FnOnce(&Planner) -> R) -> R {
        let planner = {
            let mut idle = self.idle.lock().unwrap_or_else(PoisonError::into_inner);
            match idle.len() {
                0 | 1 => idle.first().cloned(),
                _ => idle.pop(),
            }
        };
        // the template is never taken, so the pool is never empty
        let planner = planner.expect("the scheduler's planner pool keeps its template");
        let result = use_planner(&planner);
        self.idle
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(planner);
        result
    }
}

impl TranscriptionPipeline {
    /// Moves the pipeline behind a scheduler that batches windows across
    /// concurrent requests, transcribing each with the pipeline's default config
    ///
    /// Requests must fit one model input, as for [`Self::run`]; with `long-form`,
    /// `LongFormTranscriptionPipeline::into_scheduler` takes any length
    pub fn into_scheduler(
        self,
        scheduler: SchedulerConfig,
    ) -> Result<TranscriptionScheduler, TranscriptionError> {
        let config = *self.config();
        self.into_scheduler_with_config(config, scheduler)
    }

    /// Moves the pipeline behind a scheduler that transcribes every request with
    /// `config`
    pub fn into_scheduler_with_config(
        self,
        config: TranscriptionConfig,
        scheduler: SchedulerConfig,
    ) -> Result<TranscriptionScheduler, TranscriptionError> {
        self.deployment().select_language(config.language)?;
        TranscriptionScheduler::start(
            Planner::Single {
                pipeline: Box::new(self),
                config,
            },
            scheduler,
        )
    }
}

#[cfg(feature = "long-form")]
impl crate::long_form::LongFormTranscriptionPipeline {
    /// Moves the pipeline behind a scheduler that batches windows across
    /// concurrent requests of any length, with the pipeline's default config
    ///
    /// Each request is planned and merged as [`Self::run`] plans and merges it;
    /// its windows share encoder and decoder batches with other requests'. The
    /// config's `worker_count` and `batch_size` do not apply: `scheduler` sets the
    /// threads and batches
    pub fn into_scheduler(
        self,
        scheduler: SchedulerConfig,
    ) -> Result<TranscriptionScheduler, TranscriptionError> {
        let config = crate::long_form::LongFormConfig::for_bundle(self.inner().bundle());
        self.into_scheduler_with_config(config, scheduler)
    }

    /// Moves the pipeline behind a scheduler that transcribes every request with
    /// `config`
    pub fn into_scheduler_with_config(
        self,
        config: crate::long_form::LongFormConfig,
        scheduler: SchedulerConfig,
    ) -> Result<TranscriptionScheduler, TranscriptionError> {
        config.validate_for(self.inner().deployment())?;
        TranscriptionScheduler::start(
            Planner::LongForm {
                pipeline: Box::new(self),
                config,
            },
            scheduler,
        )
    }
}

/// Transcribes requests from any number of threads, batching their windows together
///
/// Clones submit to the same scheduler. Once every clone is dropped, the scheduler
/// finishes the requests already submitted and its threads exit
///
/// ```no_run
/// use scriptrs::{SchedulerConfig, TranscriptionPipeline};
///
/// # fn clips() -> Vec<Vec<f32>> { Vec::new() }
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let pipeline = TranscriptionPipeline::from_dir("models")?;
/// let scheduler = pipeline.into_scheduler(SchedulerConfig::default())?;
///
/// let callers: Vec<_> = clips()
///     .into_iter()
///     .map(|audio| {
///         let scheduler = scheduler.clone();
///         std::thread::spawn(move || scheduler.transcribe(audio))
///     })
///     .collect();
/// for caller in callers {
///     println!("{}", caller.join().expect("caller panicked")?.text);
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct TranscriptionScheduler {
    handle: Arc<Handle>,
}

impl fmt::Debug for TranscriptionScheduler {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TranscriptionScheduler")
            .field("config", &self.handle.shared.config)
            .finish_non_exhaustive()
    }
}

/// Closes the scheduler when the last [`TranscriptionScheduler`] clone drops
struct Handle {
    shared: Arc<Shared>,
}

impl Drop for Handle {
    fn drop(&mut self) {
        let mut state = self.shared.lock();
        state.closed = true;
        drop(state);
        self.shared.wake_all();
    }
}

impl TranscriptionScheduler {
    fn start(planner: Planner, config: SchedulerConfig) -> Result<Self, TranscriptionError> {
        config.validate()?;
        let transcription = planner.transcription();
        let shape = BatcherShape {
            rows: config.decoder_rows,
            frames_per_batch: config.frames_per_batch,
            min_batch: config.min_batch,
            max_batch: config.max_batch,
            max_mel_frames: planner.pipeline().chunk_spec().max_feature_frames(),
        };
        let batcher = planner.pipeline().batcher(transcription.language, shape)?;
        let buckets = BucketOf(batcher.bucket_fn());

        let chunk = planner.pipeline().chunk_spec();
        let shared = Arc::new(Shared {
            planners: Planners::new(planner),
            chunk,
            config,
            buckets,
            state: Mutex::new(State::default()),
            prepare: Condvar::new(),
            model: Condvar::new(),
            room: Condvar::new(),
            next_request: AtomicU64::new(0),
        });

        let model_shared = Arc::clone(&shared);
        thread::Builder::new()
            .name("scriptrs-scheduler".to_owned())
            .spawn(move || {
                let _stop = StopOnExit(&model_shared);
                ModelLoop {
                    shared: &model_shared,
                    batcher,
                    rows: HashMap::new(),
                }
                .run();
            })
            .map_err(TranscriptionError::Io)?;
        for worker in 0..config.workers {
            let worker_shared = Arc::clone(&shared);
            let spawned = thread::Builder::new()
                .name(format!("scriptrs-scheduler-prep-{worker}"))
                .spawn(move || {
                    let _stop = StopOnExit(&worker_shared);
                    prepare_windows(&worker_shared);
                });
            if let Err(error) = spawned {
                // the threads already started see the close and exit
                shared.lock().closed = true;
                shared.wake_all();
                return Err(TranscriptionError::Io(error));
            }
        }

        Ok(Self {
            handle: Arc::new(Handle { shared }),
        })
    }

    /// The scheduler's options
    pub fn config(&self) -> SchedulerConfig {
        self.handle.shared.config
    }

    /// Queues `audio` (mono 16 kHz samples), waiting first while the scheduler
    /// holds [`SchedulerConfig::max_requests`] unfinished requests
    ///
    /// Planning errors, such as empty or overlong audio, return here; errors from
    /// the model return from [`PendingTranscription::wait`]. With VAD long-form
    /// planning, the VAD runs on this thread
    pub fn submit(
        &self,
        audio: impl Into<Arc<[f32]>>,
    ) -> Result<PendingTranscription, TranscriptionError> {
        self.queue(audio.into(), true)
    }

    /// Queues `audio` as [`Self::submit`] does, but returns
    /// [`TranscriptionError::SchedulerFull`] instead of waiting for room
    ///
    /// A server can turn that error into a busy reply. Passing an `Arc` clone of
    /// the audio keeps it for a later retry without a copy
    pub fn try_submit(
        &self,
        audio: impl Into<Arc<[f32]>>,
    ) -> Result<PendingTranscription, TranscriptionError> {
        // a cheap early refusal, so a full scheduler does not plan (or run the VAD
        // on) audio it then rejects; the check after planning is the one that holds
        if self.handle.shared.full(&self.handle.shared.lock()) {
            return Err(TranscriptionError::SchedulerFull);
        }
        self.queue(audio.into(), false)
    }

    fn queue(
        &self,
        audio: Arc<[f32]>,
        wait_for_room: bool,
    ) -> Result<PendingTranscription, TranscriptionError> {
        let shared = &self.handle.shared;
        let (plan, windows) = shared.planners.with(|planner| planner.plan(&audio))?;
        let request = Arc::new(Request {
            id: shared.next_request.fetch_add(1, Ordering::Relaxed),
            progress: Mutex::new(Progress {
                raws: (0..windows.len()).map(|_| None).collect(),
                remaining: windows.len(),
                plan: Some(plan),
                outcome: None,
            }),
            windows,
            next_window: AtomicUsize::new(0),
            audio,
            done: Condvar::new(),
        });
        let pending = PendingTranscription {
            request: Arc::clone(&request),
            shared: Arc::clone(shared),
            waited: false,
        };
        let mut state = shared.lock();
        loop {
            if let Some(reason) = &state.stopped {
                return Err(TranscriptionError::SchedulerStopped(reason.clone()));
            }
            // a VAD plan that found no speech has no windows to hold room for
            if request.windows.is_empty() || !shared.full(&state) {
                break;
            }
            if !wait_for_room {
                return Err(TranscriptionError::SchedulerFull);
            }
            state = shared
                .room
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
        if request.windows.is_empty() {
            drop(state);
            request.finish(Outcome::Done);
            return Ok(pending);
        }

        state.live.insert(request.id, Arc::clone(&request));
        state.pending.push_back(request);
        drop(state);
        // workers waiting for room in a bucket share this condvar, so one wakeup
        // could land on a worker that cannot take the request
        shared.prepare.notify_all();
        Ok(pending)
    }

    /// Transcribes `audio` (mono 16 kHz samples), blocking until it is done
    pub fn transcribe(
        &self,
        audio: impl Into<Arc<[f32]>>,
    ) -> Result<TranscriptionResult, TranscriptionError> {
        self.submit(audio)?.wait()
    }
}

/// A submitted request; [`Self::wait`] blocks for its result
///
/// Dropping it without waiting cancels the request's windows that have not
/// started on the model
pub struct PendingTranscription {
    request: Arc<Request>,
    shared: Arc<Shared>,
    waited: bool,
}

impl fmt::Debug for PendingTranscription {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PendingTranscription")
            .field("windows", &self.request.windows.len())
            .finish_non_exhaustive()
    }
}

impl PendingTranscription {
    /// Whether the result is ready, so [`Self::wait`] will not block on the model
    pub fn is_finished(&self) -> bool {
        self.request.progress().outcome.is_some()
    }

    /// Blocks until every window has run, then merges them into the result on
    /// this thread
    pub fn wait(mut self) -> Result<TranscriptionResult, TranscriptionError> {
        self.waited = true;
        let (plan, raws) = self.request.wait_windows()?;
        let audio_len = self.request.audio.len();
        self.shared
            .planners
            .with(|planner| planner.merge(audio_len, plan, raws))
    }
}

impl Drop for PendingTranscription {
    fn drop(&mut self) {
        if !self.waited && self.request.progress().outcome.is_none() {
            self.request.finish(Outcome::Abandoned);
            self.shared.forget(self.request.id);
        }
    }
}

/// One submitted request and its windows' progress
struct Request {
    id: u64,
    audio: Arc<[f32]>,
    windows: Vec<WindowSpan>,
    /// The next window to prepare; advanced only under the scheduler's state lock
    next_window: AtomicUsize,
    progress: Mutex<Progress>,
    done: Condvar,
}

struct Progress {
    raws: Vec<Option<RawTranscription>>,
    remaining: usize,
    plan: Option<Plan>,
    /// Set once every window is done, or at the first failure
    outcome: Option<Outcome>,
}

/// How a request ended
enum Outcome {
    /// Every window delivered its output
    Done,
    /// A window or the plan failed, or the scheduler stopped
    Failed(TranscriptionError),
    /// Nothing will read the outcome: the caller dropped its handle, or took the
    /// error already
    Abandoned,
}

impl Request {
    fn progress(&self) -> MutexGuard<'_, Progress> {
        self.progress.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn finished(&self) -> bool {
        self.progress().outcome.is_some()
    }

    /// Records the end of the request, unless it already ended
    fn finish(&self, outcome: Outcome) {
        let mut progress = self.progress();
        if progress.outcome.is_none() {
            progress.outcome = Some(outcome);
            self.done.notify_all();
        }
    }

    /// Blocks until the request ends, then takes its plan and window outputs, or
    /// the error that ended it
    fn wait_windows(&self) -> Result<(Plan, Vec<RawTranscription>), TranscriptionError> {
        let mut progress = self.progress();
        while progress.outcome.is_none() {
            progress = self
                .done
                .wait(progress)
                .unwrap_or_else(PoisonError::into_inner);
        }
        // the outcome stays set, so the request's windows still waiting see it has
        // ended and never start
        if let Some(Outcome::Failed(error)) = progress.outcome.replace(Outcome::Abandoned) {
            return Err(error);
        }

        let plan = progress.plan.take().ok_or_else(|| {
            TranscriptionError::InvalidModelOutput("a request's plan went missing".into())
        })?;
        let raws = progress
            .raws
            .iter_mut()
            .map(|raw| {
                raw.take().ok_or_else(|| {
                    TranscriptionError::InvalidModelOutput("a scheduled window went missing".into())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((plan, raws))
    }

    /// Stores window `index`'s output; true once that was the last window
    fn deliver(&self, index: usize, raw: RawTranscription) -> bool {
        let mut progress = self.progress();
        if progress.outcome.is_some() || progress.raws[index].is_some() {
            return false;
        }
        progress.raws[index] = Some(raw);
        progress.remaining -= 1;
        if progress.remaining > 0 {
            return false;
        }
        progress.outcome = Some(Outcome::Done);
        self.done.notify_all();
        true
    }
}

/// A window's features, waiting in its bucket for the model
struct Ready {
    request: Arc<Request>,
    index: usize,
    chunk: PreparedChunk,
    since: Instant,
}

/// The bucket of a window's mel frames, shared by the workers and the model thread
struct BucketOf(Box<dyn Fn(usize) -> Bucket + Send + Sync>);

struct Shared {
    planners: Planners,
    /// The deployment's frontend and input limit, which workers prepare windows with
    chunk: ChunkSpec,
    config: SchedulerConfig,
    buckets: BucketOf,
    state: Mutex<State>,
    /// Workers wait here for windows to prepare and for room in a bucket
    prepare: Condvar,
    /// The model thread waits here for prepared windows
    model: Condvar,
    /// Submits wait here for a request to finish once the scheduler is full
    room: Condvar,
    next_request: AtomicU64,
}

#[derive(Default)]
struct State {
    /// Every scheduler handle is gone, so no request will arrive
    closed: bool,
    /// Why a scheduler thread stopped; every request fails with it
    stopped: Option<String>,
    /// Requests with windows left to prepare, in arrival order
    ///
    /// Taking windows oldest request first lets requests finish in about the
    /// order they came, as a server running one at a time would; taking one
    /// window from each in turn would hold every long request back until
    /// nearly all of them were done
    pending: VecDeque<Arc<Request>>,
    /// Requests not yet finished, by id
    live: HashMap<u64, Arc<Request>>,
    /// Prepared windows by bucket, oldest first
    buckets: BTreeMap<Bucket, VecDeque<Ready>>,
}

impl State {
    /// Whether the scheduler's threads have nothing left to do, ever
    fn drained(&self) -> bool {
        self.stopped.is_some() || (self.closed && self.live.is_empty())
    }
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn wake_all(&self) {
        self.prepare.notify_all();
        self.model.notify_all();
        self.room.notify_all();
    }

    /// Whether a new request must wait for room
    fn full(&self, state: &State) -> bool {
        state.live.len() >= self.config.max_requests
    }

    /// Drops a finished request from the live set, waking a submit that waits for
    /// room and the threads that wait for the scheduler to drain
    fn forget(&self, id: u64) {
        let mut state = self.lock();
        let removed = state.live.remove(&id).is_some();
        let drained = state.drained();
        drop(state);
        if removed {
            self.room.notify_one();
        }
        if drained {
            self.wake_all();
        }
    }

    /// Fails `request` with `error` and forgets it
    fn fail(&self, request: &Request, error: TranscriptionError) {
        request.finish(Outcome::Failed(error));
        self.forget(request.id);
    }
}

/// Stops the scheduler when a thread of it exits, so a panic fails every request
/// instead of leaving callers waiting
struct StopOnExit<'a>(&'a Shared);

impl Drop for StopOnExit<'_> {
    fn drop(&mut self) {
        let mut state = self.0.lock();
        if state.drained() && !thread::panicking() {
            return;
        }
        let reason = if thread::panicking() {
            format!(
                "the {} thread panicked",
                thread::current().name().unwrap_or("scheduler")
            )
        } else {
            "a scheduler thread exited".to_owned()
        };
        state.stopped.get_or_insert(reason.clone());
        let live: Vec<Arc<Request>> = state.live.drain().map(|(_, request)| request).collect();
        state.pending.clear();
        state.buckets.clear();
        drop(state);
        for request in live {
            request.finish(Outcome::Failed(TranscriptionError::SchedulerStopped(
                reason.clone(),
            )));
        }
        self.0.wake_all();
    }
}

/// A worker: prepares one window at a time, oldest request first
fn prepare_windows(shared: &Shared) {
    let preparer = shared.chunk.preparer();
    while let Some((request, index)) = next_window(shared) {
        let span = request.windows[index];
        let prepared = preparer.prepare(&request.audio[span.start..span.end]);
        match prepared {
            Ok(chunk) => queue_ready(shared, request, index, chunk),
            Err(error) => shared.fail(&request, error),
        }
    }
}

/// The next window to prepare, or `None` once the scheduler drains
fn next_window(shared: &Shared) -> Option<(Arc<Request>, usize)> {
    let mut state = shared.lock();
    loop {
        if state.drained() {
            return None;
        }
        while let Some(request) = state.pending.pop_front() {
            if request.finished() {
                continue;
            }
            let index = request.next_window.fetch_add(1, Ordering::Relaxed);
            if index + 1 < request.windows.len() {
                state.pending.push_front(Arc::clone(&request));
            }
            return Some((request, index));
        }
        state = shared
            .prepare
            .wait(state)
            .unwrap_or_else(PoisonError::into_inner);
    }
}

/// Puts a prepared window in its bucket, waiting while the bucket is full
fn queue_ready(shared: &Shared, request: Arc<Request>, index: usize, chunk: PreparedChunk) {
    let bucket = (shared.buckets.0)(chunk.frames());
    let mut state = shared.lock();
    loop {
        if state.stopped.is_some() || request.finished() {
            return;
        }
        let waiting = state.buckets.get(&bucket).map_or(0, VecDeque::len);
        if waiting < shared.config.bucket_capacity {
            break;
        }
        state = shared
            .prepare
            .wait(state)
            .unwrap_or_else(PoisonError::into_inner);
    }
    state.buckets.entry(bucket).or_default().push_back(Ready {
        request,
        index,
        chunk,
        since: Instant::now(),
    });
    drop(state);
    shared.model.notify_one();
}

/// The one thread that runs the model
struct ModelLoop<'a> {
    shared: &'a Shared,
    batcher: ModelBatcher,
    /// The window each batcher row is decoding
    rows: HashMap<usize, (Arc<Request>, usize)>,
}

impl ModelLoop<'_> {
    fn run(mut self) {
        loop {
            let Some(batch) = self.next_batch() else {
                return;
            };
            if let Some((bucket, windows)) = batch {
                self.admit(bucket, windows);
            } else {
                self.advance();
            }
        }
    }

    /// The next batch to start, `Some(None)` to decode what is in flight, or `None`
    /// once the scheduler drains
    #[allow(clippy::type_complexity, reason = "a private loop step")]
    fn next_batch(&mut self) -> Option<Option<(Bucket, Vec<Ready>)>> {
        let shared = self.shared;
        let mut state = shared.lock();
        loop {
            if state.drained() {
                return None;
            }
            let active = self.batcher.active_rows();
            if let Some(bucket) = self.choose(&mut state, active) {
                let Some(queue) = state.buckets.get_mut(&bucket) else {
                    continue;
                };
                let take = queue.len().min(bucket.batch);
                let windows: Vec<Ready> = queue.drain(..take).collect();
                if queue.is_empty() {
                    state.buckets.remove(&bucket);
                }
                drop(state);
                // room in the bucket for workers waiting on it
                shared.prepare.notify_all();
                return Some(Some((bucket, windows)));
            }
            if active > 0 {
                return Some(None);
            }
            state = shared
                .model
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// The bucket to start next: of those whose next batch fits the free rows and
    /// is full, or may run part-full because the decoder is idle or its oldest
    /// window has waited [`SchedulerConfig::batch_wait`], the one whose oldest
    /// window has waited longest
    fn choose(&self, state: &mut State, active: usize) -> Option<Bucket> {
        let free = self.batcher.free_rows();
        let now = Instant::now();
        let mut dropped = false;
        for queue in state.buckets.values_mut() {
            // windows of requests that already ended never start
            let before = queue.len();
            queue.retain(|ready| !ready.request.finished());
            dropped |= queue.len() < before;
        }
        state.buckets.retain(|_, queue| !queue.is_empty());
        if dropped {
            // the dropped windows made room for workers waiting on a full bucket
            self.shared.prepare.notify_all();
        }
        state
            .buckets
            .iter()
            .filter_map(|(bucket, queue)| {
                let oldest = queue.front()?.since;
                let take = queue.len().min(bucket.batch);
                let ready = take == bucket.batch
                    || active == 0
                    || now.duration_since(oldest) >= self.shared.config.batch_wait;
                (take <= free && ready).then_some((oldest, *bucket))
            })
            .min()
            .map(|(_, bucket)| bucket)
    }

    fn admit(&mut self, bucket: Bucket, windows: Vec<Ready>) {
        let chunks: Vec<EncoderChunk<'_>> = windows
            .iter()
            .map(|ready| EncoderChunk::new(ready.chunk.features().view()))
            .collect();
        match self.batcher.admit(bucket, &chunks) {
            Ok(rows) => {
                for (row, ready) in rows.into_iter().zip(windows) {
                    self.rows.insert(row, (ready.request, ready.index));
                }
            }
            // the windows already decoding go on. A failure while the decoder was
            // starting the batch's rows can leave some of them running with no
            // window mapped to them; they retire within their frames' steps, and
            // `advance` drops their output. An error that broke the device fails
            // every window at the next advance
            Err(error) => {
                tracing::warn!(%error, "scheduler encoder batch failed");
                let requests = windows.into_iter().map(|ready| ready.request);
                fail_each(self.shared, requests, error);
            }
        }
    }

    fn advance(&mut self) {
        match self.batcher.advance() {
            Ok(finished) => {
                for (row, outcome) in finished {
                    let Some((request, index)) = self.rows.remove(&row) else {
                        continue;
                    };
                    self.finish_window(&request, index, outcome);
                }
            }
            Err(error) => self.fail_in_flight(error),
        }
    }

    fn finish_window(
        &self,
        request: &Request,
        index: usize,
        outcome: Result<RawTranscription, TranscriptionError>,
    ) {
        let mut raw = match outcome {
            Ok(raw) => raw,
            Err(error) => return self.shared.fail(request, error),
        };
        let span = request.windows[index];
        TranscriptionPipeline::place_window(
            &mut raw,
            span.global_sample_offset,
            span.context_samples,
        );
        if request.deliver(index, raw) {
            self.shared.forget(request.id);
        }
    }

    /// Fails every window in flight with `error`, after which the batcher starts
    /// empty
    fn fail_in_flight(&mut self, error: TranscriptionError) {
        tracing::warn!(%error, "scheduler decode failed");
        self.batcher.reset();
        let requests = self.rows.drain().map(|(_, (request, _))| request);
        fail_each(self.shared, requests, error);
    }
}

/// Fails each of `requests` with the one `error` they shared
fn fail_each(
    shared: &Shared,
    requests: impl IntoIterator<Item = Arc<Request>>,
    error: TranscriptionError,
) {
    let error = Arc::new(error);
    for request in requests {
        shared.fail(
            &request,
            TranscriptionError::BatchFailed(Arc::clone(&error)),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(windows: usize) -> Request {
        let span = WindowSpan {
            start: 0,
            end: 1,
            global_sample_offset: 0,
            context_samples: 0,
        };
        Request {
            id: 0,
            audio: Arc::from(vec![0.0f32]),
            windows: vec![span; windows],
            next_window: AtomicUsize::new(0),
            progress: Mutex::new(Progress {
                raws: (0..windows).map(|_| None).collect(),
                remaining: windows,
                plan: Some(Plan::Single),
                outcome: None,
            }),
            done: Condvar::new(),
        }
    }

    #[test]
    fn a_waited_error_leaves_the_request_ended() {
        let request = request(3);
        request.finish(Outcome::Failed(TranscriptionError::EmptyAudio));

        let waited = request.wait_windows();
        assert!(matches!(waited, Err(TranscriptionError::EmptyAudio)));
        // workers and the model loop skip the windows of an ended request
        assert!(request.finished());
        assert!(!request.deliver(1, RawTranscription::empty()));
    }
}
