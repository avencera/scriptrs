use std::path::PathBuf;

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
    /// CoreML model loading or prediction failed
    #[error("{0}")]
    CoreMl(String),
    /// The current platform does not support native CoreML inference
    #[error("native CoreML transcription is only supported on macOS")]
    UnsupportedPlatform,
}
