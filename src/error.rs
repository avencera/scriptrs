use std::path::PathBuf;
use std::sync::Arc;

use crate::deployment::ParakeetVersion;

/// Errors returned by `scriptrs`
#[derive(Debug, thiserror::Error)]
pub enum TranscriptionError {
    /// Input audio was empty
    #[error("audio input was empty")]
    EmptyAudio,
    /// Input audio exceeded the base pipeline limit
    #[error(
        "audio exceeds the single-chunk limit: actual={actual_seconds:.2}s max={max_seconds:.2}s"
    )]
    AudioTooLong {
        /// The single-chunk limit in seconds
        max_seconds: f64,
        /// The provided audio length in seconds
        actual_seconds: f64,
    },
    /// A sample was not finite at the transcription preparation boundary
    #[error("audio sample at index {index} was not finite")]
    InvalidAudioSample {
        /// Index of the invalid sample in the validated input or window
        index: usize,
    },
    /// The transcription was cancelled through its shared control state
    #[error("transcription was cancelled")]
    Cancelled,
    /// The transcription's absolute deadline has passed
    #[error("transcription deadline expired")]
    DeadlineExpired,
    /// The planned task count exceeded the request limit
    #[error("transcription planned {planned} windows, exceeding the limit of {limit}")]
    WindowLimitExceeded {
        /// Number of windows in the completed plan
        planned: usize,
        /// Maximum number of windows allowed for the request
        limit: usize,
    },
    /// Required model assets were missing
    #[error("missing model asset: {path}")]
    MissingModelAsset {
        /// Missing path
        path: PathBuf,
    },
    /// The directory layout did not match the requested Parakeet version
    #[error("model layout did not match {expected:?}: found {found:?}")]
    LayoutMismatch {
        /// Requested deployment version
        expected: ParakeetVersion,
        /// Layout that was present on disk
        found: ParakeetVersion,
    },
    /// A language is not in the deployment allowlist
    #[error("language `{language}` is not supported by {version:?}")]
    UnsupportedLanguage {
        /// Requested language code
        language: String,
        /// Deployment that rejected the language
        version: ParakeetVersion,
    },
    /// A duration bin was outside the deployment mapping
    #[error("duration bin {bin} is outside the deployment mapping of length {len}")]
    InvalidDurationBin {
        /// Duration bin index from the joint head
        bin: usize,
        /// Number of bins owned by the deployment
        len: usize,
    },
    /// A transcription or long-form setting did not match its deployment
    #[error("invalid transcription configuration: {0}")]
    InvalidConfiguration(String),
    /// A language candidate logit was not finite
    #[error("language candidate logit at index {index} was not finite: {value}")]
    InvalidLanguageLogit {
        /// Candidate index
        index: usize,
        /// Invalid candidate value
        value: f32,
    },
    /// File system I/O failed
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Vocabulary data was malformed
    #[error("{0}")]
    InvalidVocabulary(String),
    /// Model output shapes or dtypes were unexpected
    #[error("{0}")]
    InvalidModelOutput(String),
    /// A native weight file was malformed or does not match its deployment
    #[error("invalid model weights: {0}")]
    InvalidWeights(String),
    /// The CUDA driver, a CUDA library or a kernel launch failed
    #[cfg(feature = "cuda")]
    #[error("CUDA: {0}")]
    Cuda(#[from] cudakit::Error),
    /// CoreML model loading or prediction failed
    #[error("{0}")]
    CoreMl(String),
    /// A `TranscriptionScheduler` thread stopped, so the scheduler takes no more
    /// requests and fails those it held
    #[error("the transcription scheduler stopped: {0}")]
    SchedulerStopped(String),
    /// `TranscriptionScheduler::try_submit` found the scheduler holding
    /// `SchedulerConfig::max_requests` unfinished requests already
    #[error("the transcription scheduler is full")]
    SchedulerFull,
    /// A batch failed, failing every request with a window in it; each of them
    /// holds the same error
    #[error("a batch shared with other requests failed: {0}")]
    BatchFailed(Arc<TranscriptionError>),
    /// The current platform does not support native CoreML inference
    #[error("native CoreML transcription is only supported on macOS")]
    UnsupportedPlatform,
}

#[cfg(feature = "cuda")]
impl From<cudakit::cudarc::driver::DriverError> for TranscriptionError {
    fn from(error: cudakit::cudarc::driver::DriverError) -> Self {
        Self::Cuda(error.into())
    }
}
