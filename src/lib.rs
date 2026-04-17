#![warn(missing_docs)]
#![warn(clippy::undocumented_unsafe_blocks)]

//! Rust transcription with native CoreML Parakeet v2 inference
//!
//! `scriptrs` is currently a narrow macOS-first transcription crate:
//!
//! - macOS only
//! - CoreML only
//! - Parakeet TDT v2 only
//! - mono 16kHz `&[f32]` input
//!
//! The base crate exposes a single-chunk [`TranscriptionPipeline`]. Fast
//! long-audio chunking and overlap merging are available behind the
//! `long-form` feature via [`LongFormTranscriptionPipeline`]. VAD-backed speech
//! region planning is available behind the `vad` feature, which also enables
//! `long-form`.
//!
//! # Choosing a pipeline
//!
//! Use [`TranscriptionPipeline`] when your audio already fits in one Parakeet
//! window. If the input is too long, it returns [`TranscriptionError::AudioTooLong`].
//!
//! Use [`LongFormTranscriptionPipeline`] with the `long-form` feature when you
//! want `scriptrs` to own long-audio chunking internally. This default path is
//! tuned for speed and works well on dense, mostly continuous speech.
//!
//! Add `vad` when you need VAD-backed speech region planning for
//! sparse speech, long silences, or recordings with a lot of non-speech audio.
//!
//! # Model loading
//!
//! With the default `online` feature, [`TranscriptionPipeline::from_pretrained`]
//! and [`LongFormTranscriptionPipeline::from_pretrained`] download the runtime
//! bundle from `avencera/scriptrs-models` on Hugging Face.
//!
//! If you already manage models yourself, use [`TranscriptionPipeline::from_dir`]
//! or [`LongFormTranscriptionPipeline::from_dir`] with a local bundle layout:
//!
//! ```text
//! models/
//!   parakeet-v2/
//!     encoder.mlmodelc/
//!     decoder.mlmodelc/
//!     joint-decision.mlmodelc/
//!     vocab.txt
//! ```
//!
//! With `vad`, add:
//!
//! ```text
//! models/
//!   vad/
//!     silero-vad.mlmodelc/
//! ```
//!
//! You can also override model resolution with environment variables:
//!
//! - `SCRIPTRS_MODELS_DIR=/path/to/models`
//! - `SCRIPTRS_MODELS_REPO=owner/repo`
//!
//! # Input requirements
//!
//! `scriptrs` expects mono 16kHz audio samples as `&[f32]`. File decoding stays
//! outside the core library on purpose.
//!
//! # Examples
//!
//! Single-chunk transcription:
//!
//! ```no_run
//! use scriptrs::TranscriptionPipeline;
//!
//! # fn load_audio() -> Vec<f32> { Vec::new() }
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let audio = load_audio();
//! let pipeline = TranscriptionPipeline::from_pretrained()?;
//! let result = pipeline.run(&audio)?;
//!
//! println!("{}", result.text);
//! # Ok(())
//! # }
//! ```
//!
//! Long-form transcription:
//!
//! ```no_run
//! # #[cfg(feature = "long-form")]
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use scriptrs::LongFormTranscriptionPipeline;
//!
//! # fn load_audio() -> Vec<f32> { Vec::new() }
//! let audio = load_audio();
//! let pipeline = LongFormTranscriptionPipeline::from_pretrained()?;
//! let result = pipeline.run(&audio)?;
//!
//! println!("{}", result.text);
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "long-form"))]
//! # fn main() {}
//! ```
//!
//! VAD-backed long-form transcription:
//!
//! ```no_run
//! # #[cfg(feature = "vad")]
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use scriptrs::{LongFormConfig, LongFormMode, LongFormTranscriptionPipeline};
//!
//! # fn load_audio() -> Vec<f32> { Vec::new() }
//! let audio = load_audio();
//! let pipeline = LongFormTranscriptionPipeline::from_pretrained()?;
//! let config = LongFormConfig {
//!     mode: LongFormMode::Vad,
//!     ..LongFormConfig::default()
//! };
//! let result = pipeline.run_with_config(&audio, &config)?;
//!
//! println!("{}", result.text);
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "vad"))]
//! # fn main() {}
//! ```

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
    LongFormConfig, LongFormMode, LongFormTranscriptionPipeline, OverlapChunkConfig,
};
#[cfg(feature = "vad")]
pub use long_form::{VadConfig, VadSegmentationConfig};
pub use models::ModelBundle;
#[cfg(feature = "online")]
pub use models::ModelManager;
pub use pipeline::TranscriptionPipeline;
pub use types::{TimedToken, TranscriptChunk, TranscriptionResult};
