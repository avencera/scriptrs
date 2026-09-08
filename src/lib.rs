#![warn(missing_docs)]
#![warn(clippy::undocumented_unsafe_blocks)]

//! Rust transcription with native CoreML Parakeet v2 and v3 inference
//!
//! `scriptrs` is currently a narrow transcription crate:
//!
//! - CoreML on macOS by default
//! - ONNX Runtime with the CUDA execution provider behind the `onnx-cuda`
//!   feature, using a local manifest-described bundle
//! - Parakeet TDT v2 and v3
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
//! bundle from a pinned snapshot of `avencera/scriptrs-models` on Hugging Face.
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
//!   parakeet-v3/
//!     encoder.mlmodelc/
//!     decoder.mlmodelc/
//!     joint-decision.mlmodelc/
//!     parakeet_vocab.json
//! ```
//!
//! Select a version with [`ParakeetDeployment::v2`] or [`ParakeetDeployment::v3`].
//!
//! With `vad`, add:
//!
//! ```text
//! models/
//!   vad/
//!     silero-vad.mlmodelc/
//! ```
//!
//! You can force local model resolution with an environment variable:
//!
//! - `SCRIPTRS_MODELS_DIR=/path/to/models`
//!
//! # ONNX/CUDA bundles
//!
//! With the `onnx-bundle` feature, validate a local bundle with
//! [`validate_onnx_bundle`]. With the `onnx-cuda` feature, open it with
//! [`OnnxCudaBundle::open`] and pass it to [`ModelBundle::from_onnx_cuda`].
//! The bundle directory holds `manifest.json` plus exactly the graphs and
//! tokenizer the manifest declares; see `docs/onnx-cuda-bundle.md` for the
//! manifest schema and graph tensor contract. Opening never downloads
//! anything and never falls back to the CPU provider:
//!
//! ```no_run
//! # #[cfg(feature = "onnx-cuda")]
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use scriptrs::{CudaDevice, ModelBundle, OnnxCudaBundle, ParakeetDeployment, TranscriptionPipeline};
//!
//! let bundle = OnnxCudaBundle::open("bundles/parakeet-v3", ParakeetDeployment::v3(), CudaDevice::new(0))?;
//! let pipeline = TranscriptionPipeline::from_bundle(ModelBundle::from_onnx_cuda(bundle))?;
//! # let _ = pipeline;
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "onnx-cuda"))]
//! # fn main() {}
//! ```
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
mod deployment;
mod error;
mod frontend;
mod language;
mod model;
mod model_io;
mod models;
mod pipeline;
#[cfg(feature = "long-form")]
mod preprocessing;
mod types;
mod vocab;

#[cfg(target_os = "macos")]
mod coreml;

#[cfg(feature = "long-form")]
mod long_form;

#[cfg(feature = "onnx-bundle")]
mod onnx;

pub use config::TranscriptionConfig;
pub use deployment::{
    DecoderDimensions, DecoderIo, EncoderIo, FrontendSpec, JointIo, ModelIo, ModelLayout,
    ParakeetDeployment, ParakeetVersion, VocabKind,
};
pub use error::TranscriptionError;
pub use language::{
    FilteredToken, LanguageSelection, ParakeetLanguage, Script, V3_LANGUAGES,
    detect_language_from_tokens, filter_top_k_language, select_language_from_logits,
};
#[cfg(feature = "long-form")]
pub use long_form::{
    LongFormConfig, LongFormMode, LongFormTranscriptionPipeline, OverlapChunkConfig,
};
#[cfg(feature = "vad")]
pub use long_form::{VadConfig, VadSegmentationConfig};
pub use model_io::{DecodedJointStep, EncoderOutputs, JointOutputs, decode_joint_outputs};
pub use models::ModelBundle;
#[cfg(feature = "online")]
pub use models::ModelManager;
#[cfg(feature = "onnx-bundle")]
pub use onnx::{
    BundlePath, BundlePathError, DigestParseError, ExportIdentity, FileAsset, GitRevision,
    GitRevisionParseError, GraphAsset, GraphKind, MANIFEST_FILE_NAME, ManifestDeployment,
    ManifestSchema, OnnxBundleError, OnnxBundleManifest, OnnxBundleValidation,
    PARAKEET_BUNDLE_SCHEMA, ParakeetGraphContract, ParakeetGraphs, Sha256Digest,
    SileroGraphContract, SourceCheckpoint, TokenizerAsset, TokenizerFormat, VadAsset,
    validate_onnx_bundle,
};
#[cfg(feature = "onnx-cuda")]
pub use onnx::{CudaDevice, OnnxCudaBundle};
pub use pipeline::TranscriptionPipeline;
#[cfg(feature = "long-form")]
pub use preprocessing::{
    AudioWindowMut, PreprocessedTranscriptionError, SourceSampleRange, TranscriptionControl,
    TranscriptionControlError, TranscriptionDeadline, WindowIndex, WindowLimit, WindowLimitError,
    WindowPreprocessor,
};
pub use types::{TimedToken, TranscriptChunk, TranscriptionResult};
pub use vocab::Vocabulary;
