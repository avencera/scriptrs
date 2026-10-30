use std::convert::Infallible;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::error::TranscriptionError;

/// An absolute deadline shared by every operation in one transcription
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TranscriptionDeadline(Instant);

impl TranscriptionDeadline {
    /// Create a deadline at an absolute instant
    pub fn at(instant: Instant) -> Self {
        Self(instant)
    }

    /// Create a deadline relative to the current monotonic clock
    pub fn from_now(timeout: Duration) -> Self {
        Self(Instant::now() + timeout)
    }

    /// Return the absolute instant owned by this deadline
    pub fn instant(self) -> Instant {
        self.0
    }

    /// Return whether this deadline has passed
    pub fn is_expired(self) -> bool {
        Instant::now() >= self.0
    }
}

impl From<Instant> for TranscriptionDeadline {
    fn from(value: Instant) -> Self {
        Self::at(value)
    }
}

/// A nonzero upper bound on the number of planned transcription windows
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowLimit(NonZeroUsize);

impl WindowLimit {
    /// Create a window limit, rejecting zero
    pub fn new(value: usize) -> Result<Self, WindowLimitError> {
        NonZeroUsize::new(value)
            .map(Self)
            .ok_or(WindowLimitError::Zero)
    }

    /// Return the maximum number of windows
    pub fn get(self) -> usize {
        self.0.get()
    }
}

/// An invalid window-limit value
#[derive(Debug, thiserror::Error, Clone, Copy, PartialEq, Eq)]
pub enum WindowLimitError {
    /// A transcription must allow at least one window
    #[error("the transcription window limit must be greater than zero")]
    Zero,
}

/// Errors reported by shared transcription execution control
#[derive(Debug, thiserror::Error, Clone, Copy, PartialEq, Eq)]
pub enum TranscriptionControlError {
    /// The caller cancelled the transcription
    #[error("transcription was cancelled")]
    Cancelled,
    /// The shared absolute deadline has passed
    #[error("transcription deadline expired")]
    DeadlineExpired,
    /// The completed plan has more windows than the configured limit
    #[error("transcription planned {planned} windows, exceeding the limit of {limit}")]
    WindowLimitExceeded {
        /// Number of windows in the completed plan
        planned: usize,
        /// Maximum number of windows allowed for the request
        limit: usize,
    },
}

/// Shared cancellation, deadline, and task-limit state for one transcription
///
/// Only [`TranscriptionControl::cancel`] cancels a control. A run that fails
/// stops its own workers without cancelling the control it was given, so the
/// same control can start another run after an error
#[derive(Debug, Clone)]
pub struct TranscriptionControl {
    deadline: TranscriptionDeadline,
    max_windows: WindowLimit,
    cancelled: Arc<AtomicBool>,
    // a run's own stop signal, kept apart from `cancelled` so stopping a run's
    // workers after a failure never cancels the caller's control
    run_stopped: Arc<AtomicBool>,
}

impl TranscriptionControl {
    /// Create control for one transcription request
    pub fn new(deadline: TranscriptionDeadline, max_windows: WindowLimit) -> Self {
        Self {
            deadline,
            max_windows,
            cancelled: Arc::new(AtomicBool::new(false)),
            run_stopped: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Create control for an existing API that does not set a request deadline
    pub(crate) fn internal_unbounded() -> Self {
        Self::new(
            TranscriptionDeadline::from_now(Duration::from_secs(60 * 60 * 24 * 365 * 100)),
            WindowLimit::new(usize::MAX).expect("the internal window limit is nonzero"),
        )
    }

    /// Request cancellation for this transcription and every clone of its control
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Derive control for one run's workers
    ///
    /// The result shares this control's cancellation, deadline and window limit,
    /// and adds a stop signal of its own for [`Self::stop_run`]
    pub(crate) fn run_scope(&self) -> Self {
        Self {
            run_stopped: Arc::new(AtomicBool::new(false)),
            ..self.clone()
        }
    }

    /// Stop every worker of this run without cancelling the caller's control
    ///
    /// Workers see the stop as [`TranscriptionControlError::Cancelled`]; the run
    /// reports the failure that stopped it instead
    pub(crate) fn stop_run(&self) {
        self.run_stopped.store(true, Ordering::Release);
    }

    /// Check cancellation and the absolute deadline
    pub fn check(&self) -> Result<(), TranscriptionControlError> {
        if self.cancelled.load(Ordering::Acquire) || self.run_stopped.load(Ordering::Acquire) {
            return Err(TranscriptionControlError::Cancelled);
        }
        if self.deadline.is_expired() {
            return Err(TranscriptionControlError::DeadlineExpired);
        }

        Ok(())
    }

    /// Return the time left before the absolute deadline
    pub fn remaining(&self) -> Result<Duration, TranscriptionControlError> {
        self.check()?;
        Ok(self
            .deadline
            .instant()
            .saturating_duration_since(Instant::now()))
    }

    /// Return the absolute deadline owned by this control
    pub const fn deadline(&self) -> TranscriptionDeadline {
        self.deadline
    }

    /// Return the configured maximum number of windows
    pub fn max_windows(&self) -> usize {
        self.max_windows.get()
    }

    /// Check a completed plan before any window preprocessing or inference
    pub(crate) fn check_plan(&self, planned: usize) -> Result<(), TranscriptionControlError> {
        self.check()?;
        if planned > self.max_windows() {
            return Err(TranscriptionControlError::WindowLimitExceeded {
                planned,
                limit: self.max_windows(),
            });
        }

        Ok(())
    }
}

/// A validated half-open source range in the original 16 kHz recording
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceSampleRange {
    start: usize,
    end: usize,
}

impl SourceSampleRange {
    /// Create a half-open source range in the original recording
    pub fn new(start: usize, end: usize) -> Self {
        debug_assert!(end >= start);
        Self { start, end }
    }

    /// Return the first sample in the range
    pub fn start(self) -> usize {
        self.start
    }

    /// Return the exclusive end sample in the range
    pub fn end(self) -> usize {
        self.end
    }

    /// Return the number of samples in the range
    pub fn len(self) -> usize {
        self.end - self.start
    }

    /// Return whether the range contains no samples
    pub fn is_empty(self) -> bool {
        self.start == self.end
    }
}

/// The stable index of a task in the completed Scriptrs execution plan
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WindowIndex(usize);

impl WindowIndex {
    /// Create a stable task index in the completed execution plan
    pub const fn new(value: usize) -> Self {
        Self(value)
    }

    /// Return the task index
    pub fn get(self) -> usize {
        self.0
    }
}

/// A fixed-size mutable audio window passed to a preprocessing hook
pub struct AudioWindowMut<'a> {
    samples: &'a mut [f32],
    source_range: SourceSampleRange,
    index: WindowIndex,
}

impl std::fmt::Debug for AudioWindowMut<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AudioWindowMut")
            .field("sample_count", &self.samples.len())
            .field("source_range", &self.source_range)
            .field("index", &self.index)
            .finish()
    }
}

impl AudioWindowMut<'_> {
    /// Create a mutable window over a fixed-size sample slice
    ///
    /// The slice length must equal [`SourceSampleRange::len`]
    pub fn new(
        samples: &mut [f32],
        source_range: SourceSampleRange,
        index: WindowIndex,
    ) -> AudioWindowMut<'_> {
        debug_assert_eq!(samples.len(), source_range.len());
        AudioWindowMut {
            samples,
            source_range,
            index,
        }
    }

    /// Return the current window samples without changing its length
    pub fn samples(&self) -> &[f32] {
        self.samples
    }

    /// Return mutable access to the current window samples
    pub fn samples_mut(&mut self) -> &mut [f32] {
        self.samples
    }

    /// Return the immutable source range in the original recording
    pub fn source_range(&self) -> SourceSampleRange {
        self.source_range
    }

    /// Return the immutable task index in the completed plan
    pub fn index(&self) -> WindowIndex {
        self.index
    }
}

/// A typed preprocessing hook applied once to every planned ASR window
pub trait WindowPreprocessor: Sync {
    /// Error returned by this preprocessing hook
    type Error: std::error::Error + Send + Sync + 'static;

    /// Process one fixed-size window before frontend extraction
    fn process(
        &self,
        window: &mut AudioWindowMut<'_>,
        control: &TranscriptionControl,
    ) -> Result<(), Self::Error>;
}

/// Result error that keeps preprocessing failures typed
#[derive(Debug, thiserror::Error)]
pub enum PreprocessedTranscriptionError<E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    /// A normal Scriptrs or execution-control failure
    #[error(transparent)]
    Transcription(#[from] TranscriptionError),
    /// A failure returned by the window preprocessor
    #[error("window preprocessing failed: {0}")]
    Preprocessing(#[source] E),
}

impl<E> From<TranscriptionControlError> for PreprocessedTranscriptionError<E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn from(error: TranscriptionControlError) -> Self {
        Self::Transcription(TranscriptionError::Control(error))
    }
}

/// A no-op preprocessor used by the existing non-hooked APIs
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct IdentityPreprocessor;

impl WindowPreprocessor for IdentityPreprocessor {
    type Error = Infallible;

    fn process(
        &self,
        _window: &mut AudioWindowMut<'_>,
        _control: &TranscriptionControl,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AudioWindowMut, SourceSampleRange, TranscriptionControl, TranscriptionControlError,
        TranscriptionDeadline, WindowIndex, WindowLimit, WindowPreprocessor,
    };
    use std::fmt::{Display, Formatter};
    use std::time::{Duration, Instant};

    #[derive(Debug)]
    struct TestError;

    impl Display for TestError {
        fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("test preprocessing error")
        }
    }

    impl std::error::Error for TestError {}

    #[derive(Debug, Clone, Copy)]
    struct OffsetPreprocessor;

    impl WindowPreprocessor for OffsetPreprocessor {
        type Error = TestError;

        fn process(
            &self,
            window: &mut AudioWindowMut<'_>,
            _control: &TranscriptionControl,
        ) -> Result<(), Self::Error> {
            let offset = window.index().get() as f32;
            for sample in window.samples_mut() {
                *sample += offset;
            }
            Ok(())
        }
    }

    #[test]
    fn zero_window_limit_is_rejected() {
        assert!(WindowLimit::new(0).is_err());
    }

    #[test]
    fn past_deadline_expires_immediately() {
        let control = TranscriptionControl::new(
            TranscriptionDeadline::at(Instant::now() - Duration::from_secs(1)),
            WindowLimit::new(1).expect("one is valid"),
        );

        assert_eq!(
            control.check(),
            Err(TranscriptionControlError::DeadlineExpired)
        );
    }

    #[test]
    fn cancellation_is_shared_by_clones() {
        let control = TranscriptionControl::new(
            TranscriptionDeadline::from_now(Duration::from_secs(1)),
            WindowLimit::new(1).expect("one is valid"),
        );
        let clone = control.clone();
        clone.cancel();

        assert_eq!(control.check(), Err(TranscriptionControlError::Cancelled));
    }

    /// A run that stops its own workers after a failure leaves the caller's
    /// control usable for the next run
    #[test]
    fn stopping_a_run_leaves_the_caller_control_usable() {
        let caller = TranscriptionControl::new(
            TranscriptionDeadline::from_now(Duration::from_secs(1)),
            WindowLimit::new(1).expect("one is valid"),
        );
        let run = caller.run_scope();
        run.stop_run();

        assert_eq!(run.check(), Err(TranscriptionControlError::Cancelled));
        assert_eq!(caller.check(), Ok(()));
        assert_eq!(caller.run_scope().check(), Ok(()));
    }

    /// A run still stops when its caller cancels or the shared deadline passes
    #[test]
    fn run_scope_observes_caller_cancellation_and_deadline() {
        let caller = TranscriptionControl::new(
            TranscriptionDeadline::from_now(Duration::from_secs(1)),
            WindowLimit::new(1).expect("one is valid"),
        );
        let run = caller.run_scope();
        caller.cancel();
        let expired = TranscriptionControl::new(
            TranscriptionDeadline::at(Instant::now() - Duration::from_secs(1)),
            WindowLimit::new(1).expect("one is valid"),
        );

        assert_eq!(run.check(), Err(TranscriptionControlError::Cancelled));
        assert_eq!(
            expired.run_scope().check(),
            Err(TranscriptionControlError::DeadlineExpired)
        );
    }

    #[test]
    fn plan_limit_is_checked_before_window_work() {
        let control = TranscriptionControl::new(
            TranscriptionDeadline::from_now(Duration::from_secs(1)),
            WindowLimit::new(2).expect("two is valid"),
        );

        assert_eq!(
            control.check_plan(3),
            Err(TranscriptionControlError::WindowLimitExceeded {
                planned: 3,
                limit: 2,
            })
        );
    }

    #[test]
    fn hook_receives_fixed_source_coordinates_and_can_change_samples() {
        let mut samples = [1.0, 2.0, 3.0];
        let range = SourceSampleRange::new(8, 11);
        let control = TranscriptionControl::new(
            TranscriptionDeadline::from_now(Duration::from_secs(1)),
            WindowLimit::new(1).expect("one is valid"),
        );
        let processed_len = {
            let mut window = AudioWindowMut::new(&mut samples, range, WindowIndex::new(4));

            assert_eq!(window.samples().len(), range.len());
            assert_eq!(window.source_range(), range);
            assert_eq!(window.index().get(), 4);
            OffsetPreprocessor
                .process(&mut window, &control)
                .expect("the test preprocessor should succeed");
            window.samples().len()
        };

        assert_eq!(samples, [5.0, 6.0, 7.0]);
        assert_eq!(processed_len, 3);
    }
}
