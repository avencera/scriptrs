//! The `scriptrs._native` extension module
//!
//! The Python package in `python/scriptrs` wraps the pipeline class and
//! re-exports the result types from here unchanged

use std::sync::{Mutex, MutexGuard, PoisonError};

use pyo3::prelude::*;
use scriptrs::{ParakeetDeployment, TranscriptionError, V3_LANGUAGES};

pyo3::import_exception!(scriptrs._errors, ScriptrsError);

mod audio;
mod pipeline;
mod results;
mod samples;

use pipeline::{_from_dir, _from_pretrained, PyPipeline};
use results::{PyTimedToken, PyTranscriptChunk, PyTranscriptionResult};

/// Backend compiled into this build, or `None` where scriptrs has no backend
const BACKEND: Option<&str> = if cfg!(target_os = "macos") {
    Some("coreml")
} else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
    Some("cuda")
} else {
    None
};

pub(crate) fn scriptrs_err(code: &str, message: impl ToString) -> PyErr {
    ScriptrsError::new_err((code.to_owned(), message.to_string()))
}

pub(crate) fn transcription_err(error: TranscriptionError) -> PyErr {
    scriptrs_err(transcription_code(&error), error)
}

/// Stable code for a transcription failure, so Python callers can branch on it
/// without parsing messages
pub(crate) fn transcription_code(error: &TranscriptionError) -> &'static str {
    match error {
        TranscriptionError::EmptyAudio => "empty_audio",
        TranscriptionError::AudioTooLong { .. } => "audio_too_long",
        TranscriptionError::InvalidAudioSample { .. } => "invalid_audio",
        TranscriptionError::UnsupportedLanguage { .. } => "unsupported_language",
        TranscriptionError::InvalidConfiguration(_) => "configuration",
        TranscriptionError::MissingModelAsset { .. }
        | TranscriptionError::LayoutMismatch { .. } => "missing_model",
        TranscriptionError::InvalidVocabulary(_) | TranscriptionError::InvalidWeights(_) => {
            "invalid_model"
        }
        TranscriptionError::UnsupportedPlatform => "unsupported_platform",
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        TranscriptionError::Cuda(_) => "cuda",
        TranscriptionError::CoreMl(_) => "coreml",
        TranscriptionError::Io(_) => "io",
        TranscriptionError::SchedulerStopped(_) | TranscriptionError::SchedulerFull => "scheduler",
        _ => "transcription",
    }
}

pub(crate) fn deployment(model: &str) -> PyResult<ParakeetDeployment> {
    match model {
        "parakeet/v2" => Ok(ParakeetDeployment::v2()),
        "parakeet/v3" => Ok(ParakeetDeployment::v3()),
        other => Err(scriptrs_err(
            "configuration",
            format!("unknown model `{other}`; expected `parakeet/v2` or `parakeet/v3`"),
        )),
    }
}

// a panic while holding the pipeline lock leaves the pipeline itself unchanged
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Language codes a model transcribes, such as `["en"]` for `parakeet/v2`
#[pyfunction]
fn languages(model: &str) -> PyResult<Vec<&'static str>> {
    Ok(deployment(model)?
        .languages()
        .iter()
        .map(|language| language.code())
        .collect())
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("BACKEND", BACKEND)?;
    m.add(
        "ALL_LANGUAGES",
        V3_LANGUAGES
            .iter()
            .map(|language| language.code())
            .collect::<Vec<_>>(),
    )?;
    m.add_class::<PyTimedToken>()?;
    m.add_class::<PyTranscriptChunk>()?;
    m.add_class::<PyTranscriptionResult>()?;
    m.add_class::<PyPipeline>()?;
    m.add_function(wrap_pyfunction!(_from_pretrained, m)?)?;
    m.add_function(wrap_pyfunction!(_from_dir, m)?)?;
    m.add_function(wrap_pyfunction!(languages, m)?)?;
    Ok(())
}
