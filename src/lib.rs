#![warn(missing_docs)]
#![warn(clippy::undocumented_unsafe_blocks)]

//! Rust transcription with native CoreML Parakeet v2 inference
//!
//! The base crate exposes a single-chunk [`TranscriptionPipeline`]. Long-audio
//! chunking, VAD, and overlap merging are available behind the `long-form`
//! feature via [`LongFormTranscriptionPipeline`].

mod config;
mod constants;
mod decode;
mod error;
mod frontend;
mod model;
mod models;
mod pipeline;
mod types;
mod vocab;

#[cfg(target_os = "macos")]
mod coreml;

#[cfg(feature = "long-form")]
mod long_form;

pub use config::TranscriptionConfig;
pub use error::TranscriptionError;
#[cfg(feature = "long-form")]
pub use long_form::{
    LongFormConfig, LongFormTranscriptionPipeline, OverlapChunkConfig, VadConfig,
    VadSegmentationConfig,
};
pub use models::ModelBundle;
#[cfg(feature = "online")]
pub use models::ModelManager;
pub use pipeline::TranscriptionPipeline;
pub use types::{TimedToken, TranscriptChunk, TranscriptionResult};
