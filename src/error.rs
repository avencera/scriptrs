use std::path::PathBuf;

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
