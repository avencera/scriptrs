//! Background queue for many independent short inputs
//!
//! [`TranscriptionPipeline::into_queued`] moves a pipeline onto a worker thread and
//! returns a [`QueueSender`] to submit inputs and a [`QueueReceiver`] to collect
//! results. Inside, a preparation thread computes each input's features while the
//! model thread transcribes earlier inputs, and the model thread runs every
//! prepared input that is waiting, up to [`QueueConfig::max_batch`], in one model
//! call. A blocking [`TranscriptionPipeline::run`] cannot overlap the two: the
//! device sits idle while the host prepares the next input

use std::any::Any;
use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{
    Receiver, RecvTimeoutError, SyncSender, TryRecvError, TrySendError, sync_channel,
};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::config::TranscriptionConfig;
use crate::decode::RawTranscription;
use crate::error::TranscriptionError;
use crate::pipeline::{PreparedChunk, PreparedWindow, TranscriptionPipeline};
use crate::types::TranscriptionResult;

// the worker thread owns the pipeline, so every backend must be movable to it
const _: () = {
    const fn movable<T: Send>() {}
    movable::<TranscriptionPipeline>();
};

/// Identifier the queue assigns to each accepted request
///
/// Ids are unique across every clone of one [`QueueSender`] and local to that
/// queue. A push the queue rejects still uses one up, and with several threads
/// pushing at once, ids need not follow the order the queue accepted requests in
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QueuedTranscriptionJobId(u64);

impl QueuedTranscriptionJobId {
    /// The numeric id
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

/// Options for a background transcription queue
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueConfig {
    /// Requests the queue holds before [`QueueSender::try_push`] reports
    /// [`QueueError::Full`]; results are held up to the same bound
    pub capacity: usize,
    /// Most inputs one model call runs together
    ///
    /// The model thread batches whichever prepared inputs are waiting, so a batch
    /// only fills while requests arrive faster than the model runs them. Inputs in
    /// one batch are padded to the longest, so submitting inputs of similar length
    /// together wastes the least work. At 1 every input runs alone, exactly as
    /// [`TranscriptionPipeline::run_with_config`] runs it
    ///
    /// Only CUDA bundles run a batch in one pass; a batch's shape can change how
    /// cuBLAS orders its sums, so a token can differ from the batch-1 result.
    /// Core ML bundles run a batch's inputs one after another
    pub max_batch: usize,
}

impl QueueConfig {
    /// Default for [`Self::capacity`]
    pub const DEFAULT_CAPACITY: usize = 64;
    /// Default for [`Self::max_batch`]
    pub const DEFAULT_MAX_BATCH: usize = 16;

    fn validate(self) -> Result<(), QueueError> {
        if self.capacity == 0 {
            return Err(QueueError::InvalidCapacity);
        }
        if self.max_batch == 0 {
            return Err(QueueError::InvalidMaxBatch);
        }
        Ok(())
    }
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            capacity: Self::DEFAULT_CAPACITY,
            max_batch: Self::DEFAULT_MAX_BATCH,
        }
    }
}

/// One input for the queue: mono 16 kHz samples and a caller-chosen id
pub struct QueuedTranscriptionRequest {
    id: String,
    audio: Vec<f32>,
}

impl QueuedTranscriptionRequest {
    /// A request for `audio` (mono 16 kHz samples), labeled `id` in its result
    pub fn new(id: impl Into<String>, audio: Vec<f32>) -> Self {
        Self {
            id: id.into(),
            audio,
        }
    }

    /// The caller's id for this request
    pub fn id(&self) -> &str {
        &self.id
    }
}

impl fmt::Debug for QueuedTranscriptionRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("QueuedTranscriptionRequest")
            .field("id", &self.id)
            .field("samples", &self.audio.len())
            .finish()
    }
}

/// The outcome of one queued request
///
/// A request that fails, such as empty or overlong audio, reports its error here
/// and the queue keeps running
#[derive(Debug)]
pub struct QueuedTranscriptionResult {
    /// The id [`QueueSender::try_push`] returned for the request
    pub job_id: QueuedTranscriptionJobId,
    /// The caller's id from the request
    pub id: String,
    /// The transcription, or why this request failed
    pub result: Result<TranscriptionResult, TranscriptionError>,
}

/// Errors from the transcription queue itself, as opposed to one request's
/// transcription
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum QueueError {
    /// Every sender was dropped and every result has been received
    #[error("the queue has finished every submitted request")]
    Closed,
    /// The worker has stopped, so the queue takes no more requests
    #[error("the queue worker has stopped")]
    WorkerGone,
    /// The queue holds [`QueueConfig::capacity`] requests already; the rejected
    /// request comes back for a retry
    #[error("the queue is full")]
    Full(QueuedTranscriptionRequest),
    /// [`QueueConfig::capacity`] was 0
    #[error("queue capacity must be at least 1")]
    InvalidCapacity,
    /// [`QueueConfig::max_batch`] was 0
    #[error("queue max_batch must be at least 1")]
    InvalidMaxBatch,
    /// The transcription config does not fit the pipeline
    #[error(transparent)]
    InvalidConfig(#[from] TranscriptionError),
    /// The worker thread could not start
    #[error("failed to start the queue worker: {0}")]
    WorkerStart(#[source] std::io::Error),
    /// The worker thread panicked; results sent before the panic were delivered
    #[error("the queue worker panicked: {0}")]
    WorkerPanicked(String),
}

impl TranscriptionPipeline {
    /// Moves the pipeline onto a background queue with its default config and
    /// [`QueueConfig::default`]
    ///
    /// Drop every [`QueueSender`] once all requests are pushed; the receiver then
    /// reports [`QueueError::Closed`] after the last result
    ///
    /// ```no_run
    /// use std::time::Duration;
    ///
    /// use scriptrs::{QueueError, QueuedTranscriptionRequest, TranscriptionPipeline};
    ///
    /// # fn inputs() -> Vec<(String, Vec<f32>)> { Vec::new() }
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let pipeline = TranscriptionPipeline::from_dir("models")?;
    /// let (sender, receiver) = pipeline.into_queued()?;
    ///
    /// let producer = std::thread::spawn(move || -> Result<(), QueueError> {
    ///     for (id, audio) in inputs() {
    ///         let mut request = QueuedTranscriptionRequest::new(id, audio);
    ///         loop {
    ///             match sender.try_push(request) {
    ///                 Ok(_) => break,
    ///                 Err(QueueError::Full(rejected)) => {
    ///                     request = rejected;
    ///                     std::thread::sleep(Duration::from_millis(1));
    ///                 }
    ///                 Err(error) => return Err(error),
    ///             }
    ///         }
    ///     }
    ///     Ok(())
    /// });
    ///
    /// for queued in receiver {
    ///     let queued = queued?;
    ///     println!("{}: {}", queued.id, queued.result?.text);
    /// }
    /// producer.join().expect("producer panicked")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn into_queued(self) -> Result<(QueueSender, QueueReceiver), QueueError> {
        let config = self.config().clone();
        self.into_queued_with_config(config, QueueConfig::default())
    }

    /// Moves the pipeline onto a background queue that transcribes every request
    /// with `config`
    pub fn into_queued_with_config(
        self,
        config: TranscriptionConfig,
        queue: QueueConfig,
    ) -> Result<(QueueSender, QueueReceiver), QueueError> {
        queue.validate()?;
        config.validate_for(self.deployment())?;

        let (request_tx, request_rx) = sync_channel(queue.capacity);
        let (result_tx, result_rx) = sync_channel(queue.capacity);
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let worker = thread::Builder::new()
            .name("scriptrs-queue".to_owned())
            .spawn(move || {
                run_worker(&self, &config, queue, request_rx, &result_tx, &worker_stop);
            })
            .map_err(QueueError::WorkerStart)?;
        let sender = QueueSender {
            request_tx,
            next_job_id: Arc::new(AtomicU64::new(0)),
            capacity: queue.capacity,
        };
        let receiver = QueueReceiver {
            result_rx,
            worker: Some(worker),
            state: ReceiverState::Open,
            stop,
        };
        Ok((sender, receiver))
    }
}

/// Submits requests to a background transcription queue
///
/// Clones submit to the same queue. Drop every clone to close the queue
#[derive(Clone)]
pub struct QueueSender {
    request_tx: SyncSender<(QueuedTranscriptionJobId, QueuedTranscriptionRequest)>,
    next_job_id: Arc<AtomicU64>,
    capacity: usize,
}

impl fmt::Debug for QueueSender {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("QueueSender")
            .field("capacity", &self.capacity)
            .finish_non_exhaustive()
    }
}

impl QueueSender {
    /// Requests the queue holds before it reports [`QueueError::Full`]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Submits a request without blocking
    ///
    /// Returns [`QueueError::Full`] with the request when the queue is at
    /// capacity, and [`QueueError::WorkerGone`] once the worker has stopped
    pub fn try_push(
        &self,
        request: QueuedTranscriptionRequest,
    ) -> Result<QueuedTranscriptionJobId, QueueError> {
        let job_id = QueuedTranscriptionJobId(self.next_job_id.fetch_add(1, Ordering::Relaxed));
        match self.request_tx.try_send((job_id, request)) {
            Ok(()) => Ok(job_id),
            Err(TrySendError::Full((_, request))) => Err(QueueError::Full(request)),
            Err(TrySendError::Disconnected(_)) => Err(QueueError::WorkerGone),
        }
    }
}

enum ReceiverState {
    Open,
    Closed,
    Panicked(String),
}

/// Receives results from a background transcription queue, in the order the
/// queue accepted their requests
///
/// The receive calls take `&mut self` so the receiver can join the worker once
/// and then report the same end state on every later call
///
/// Dropping the receiver stops the worker once its current batch is done, or
/// within about 50 ms when it is idle, dropping the requests still queued; later
/// pushes report [`QueueError::WorkerGone`]
pub struct QueueReceiver {
    result_rx: Receiver<QueuedTranscriptionResult>,
    worker: Option<JoinHandle<()>>,
    state: ReceiverState,
    /// Set when the receiver drops; shared with the worker
    stop: Arc<AtomicBool>,
}

impl Drop for QueueReceiver {
    fn drop(&mut self) {
        // the worker is not joined, so dropping the receiver never blocks
        self.stop.store(true, Ordering::Release);
    }
}

impl fmt::Debug for QueueReceiver {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("QueueReceiver")
            .finish_non_exhaustive()
    }
}

impl QueueReceiver {
    /// Blocks until the next result
    ///
    /// Returns [`QueueError::Closed`] once every sender is dropped and every
    /// result received, or [`QueueError::WorkerPanicked`] if the worker panicked
    pub fn recv(&mut self) -> Result<QueuedTranscriptionResult, QueueError> {
        self.end_state()?;
        match self.result_rx.recv() {
            Ok(result) => Ok(result),
            Err(_) => Err(self.finish()),
        }
    }

    /// The next result if one is ready, or `None` while the queue is still working
    pub fn try_recv(&mut self) -> Result<Option<QueuedTranscriptionResult>, QueueError> {
        self.end_state()?;
        match self.result_rx.try_recv() {
            Ok(result) => Ok(Some(result)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(self.finish()),
        }
    }

    /// Waits up to `timeout` for the next result, returning `None` if none came
    ///
    /// Lets a caller that must stay responsive wait in bounded slices
    pub fn recv_timeout(
        &mut self,
        timeout: Duration,
    ) -> Result<Option<QueuedTranscriptionResult>, QueueError> {
        self.end_state()?;
        match self.result_rx.recv_timeout(timeout) {
            Ok(result) => Ok(Some(result)),
            Err(RecvTimeoutError::Timeout) => Ok(None),
            Err(RecvTimeoutError::Disconnected) => Err(self.finish()),
        }
    }

    fn end_state(&self) -> Result<(), QueueError> {
        match &self.state {
            ReceiverState::Open => Ok(()),
            ReceiverState::Closed => Err(QueueError::Closed),
            ReceiverState::Panicked(message) => Err(QueueError::WorkerPanicked(message.clone())),
        }
    }

    /// Joins the worker once the result channel has closed and records how it ended
    fn finish(&mut self) -> QueueError {
        if let Some(worker) = self.worker.take() {
            self.state = match worker.join() {
                Ok(()) => ReceiverState::Closed,
                Err(payload) => ReceiverState::Panicked(panic_message(payload.as_ref())),
            };
        }
        match self.end_state() {
            Err(error) => error,
            // `worker` is only taken here, and the state set with it
            Ok(()) => QueueError::Closed,
        }
    }
}

impl IntoIterator for QueueReceiver {
    type Item = Result<QueuedTranscriptionResult, QueueError>;
    type IntoIter = QueueReceiverIter;

    fn into_iter(self) -> Self::IntoIter {
        QueueReceiverIter {
            receiver: self,
            ended: false,
        }
    }
}

/// Yields a [`QueueReceiver`]'s results until the queue closes
///
/// A worker panic is yielded once, as an error, before the iterator ends
#[derive(Debug)]
pub struct QueueReceiverIter {
    receiver: QueueReceiver,
    ended: bool,
}

impl Iterator for QueueReceiverIter {
    type Item = Result<QueuedTranscriptionResult, QueueError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.ended {
            return None;
        }
        match self.receiver.recv() {
            Ok(result) => Some(Ok(result)),
            Err(QueueError::Closed) => {
                self.ended = true;
                None
            }
            Err(error) => {
                self.ended = true;
                Some(Err(error))
            }
        }
    }
}

fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".to_owned()
    }
}

/// A request after feature extraction, or the error that stopped it there
struct Prepared {
    job_id: QueuedTranscriptionJobId,
    id: String,
    samples: usize,
    chunk: Result<PreparedChunk, TranscriptionError>,
}

/// Longest the preparation thread waits for a request before it checks whether
/// the worker should stop
///
/// A blocked receive cannot be woken while a sender lives, so preparation waits
/// in slices to notice a dropped receiver or a stopped model loop
const STOP_POLL: Duration = Duration::from_millis(50);

/// Sets the stop flag when dropped, so the model loop signals preparation however
/// it ends, including by panic
struct StopOnDrop<'a>(&'a AtomicBool);

impl Drop for StopOnDrop<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

/// Prepares requests on one thread and transcribes them on this one until the
/// senders close, the receiver goes away, or either thread panics
///
/// The scope joins both threads before it returns or rethrows a panic, so each
/// thread's exit must wake the other: preparation dropping `prepared_tx` ends the
/// model loop's receive, and the model loop ending sets `stop` and drops
/// `prepared_rx`, which ends preparation within [`STOP_POLL`] or at its next send
fn run_worker(
    pipeline: &TranscriptionPipeline,
    config: &TranscriptionConfig,
    queue: QueueConfig,
    request_rx: Receiver<(QueuedTranscriptionJobId, QueuedTranscriptionRequest)>,
    result_tx: &SyncSender<QueuedTranscriptionResult>,
    stop: &AtomicBool,
) {
    // the model drains up to a batch of prepared inputs while the next batch is
    // prepared, which bounds the features held in memory
    let (prepared_tx, prepared_rx) = sync_channel::<Prepared>(queue.max_batch);
    thread::scope(|scope| {
        thread::Builder::new()
            .name("scriptrs-queue-prep".to_owned())
            .spawn_scoped(scope, move || {
                let preparer = TranscriptionPipeline::chunk_preparer(config);
                while !stop.load(Ordering::Acquire) {
                    let (job_id, request) = match request_rx.recv_timeout(STOP_POLL) {
                        Ok(next) => next,
                        Err(RecvTimeoutError::Timeout) => continue,
                        Err(RecvTimeoutError::Disconnected) => break,
                    };
                    let prepared = Prepared {
                        job_id,
                        samples: request.audio.len(),
                        chunk: preparer.prepare(&request.audio),
                        id: request.id,
                    };
                    if prepared_tx.send(prepared).is_err() {
                        break;
                    }
                }
            })
            .expect("spawning the queue's preparation thread");

        let _stop_preparation = StopOnDrop(stop);
        transcribe_prepared(pipeline, config, queue.max_batch, prepared_rx, result_tx);
    });
}

/// Transcribes prepared inputs in batches of up to `max_batch` until preparation
/// ends or the receiver goes away
///
/// `prepared_rx` is taken by value so it drops as soon as this loop ends, by
/// return or by panic, which wakes preparation if it is blocked on a full channel
fn transcribe_prepared(
    pipeline: &TranscriptionPipeline,
    config: &TranscriptionConfig,
    max_batch: usize,
    prepared_rx: Receiver<Prepared>,
    result_tx: &SyncSender<QueuedTranscriptionResult>,
) {
    let mut batch = VecDeque::with_capacity(max_batch);
    while let Ok(first) = prepared_rx.recv() {
        batch.push_back(first);
        while batch.len() < max_batch {
            match prepared_rx.try_recv() {
                Ok(next) => batch.push_back(next),
                Err(_) => break,
            }
        }

        if !transcribe_batch(pipeline, config, &mut batch, result_tx) {
            break;
        }
    }
}

/// Transcribes the prepared inputs in `batch` and sends their results in order;
/// false once the receiver is gone
fn transcribe_batch(
    pipeline: &TranscriptionPipeline,
    config: &TranscriptionConfig,
    batch: &mut VecDeque<Prepared>,
    result_tx: &SyncSender<QueuedTranscriptionResult>,
) -> bool {
    // each request's window index, or the error that stopped its preparation
    let mut requests = Vec::with_capacity(batch.len());
    let mut windows = Vec::with_capacity(batch.len());
    for prepared in batch.drain(..) {
        let outcome = prepared.chunk.map(|chunk| {
            windows.push(PreparedWindow {
                prepared: chunk,
                global_sample_offset: 0,
                context_samples: 0,
            });
            windows.len() - 1
        });
        requests.push((prepared.job_id, prepared.id, prepared.samples, outcome));
    }

    let mut raws: Vec<Option<Result<RawTranscription, TranscriptionError>>> =
        match pipeline.transcribe_prepared_batch(&windows, config.language) {
            Ok(raws) => raws.into_iter().map(|raw| Some(Ok(raw))).collect(),
            // as speakrs's queue does, a failed batch reruns each input alone, so one
            // bad input fails only its own request
            Err(_) if windows.len() > 1 => windows
                .iter()
                .map(|window| {
                    let alone = std::slice::from_ref(window);
                    Some(
                        pipeline
                            .transcribe_prepared_batch(alone, config.language)
                            .map(|mut raws| raws.remove(0)),
                    )
                })
                .collect(),
            Err(error) => vec![Some(Err(error))],
        };

    let results = requests
        .into_iter()
        .map(|(job_id, id, samples, outcome)| {
            let raw = outcome.and_then(|index| {
                raws.get_mut(index)
                    .and_then(Option::take)
                    .unwrap_or_else(|| {
                        Err(TranscriptionError::InvalidModelOutput(format!(
                            "the queue's model call returned no output for window {index}"
                        )))
                    })
            });
            let duration_seconds = samples as f64 / config.sample_rate as f64;
            QueuedTranscriptionResult {
                job_id,
                id,
                result: raw.map(|raw| pipeline.decode_raw(&raw, duration_seconds)),
            }
        })
        .collect::<Vec<_>>();
    results
        .into_iter()
        .all(|result| result_tx.send(result).is_ok())
}
