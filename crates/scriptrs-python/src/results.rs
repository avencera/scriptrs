//! Result types returned to Python

use pyo3::prelude::*;
use scriptrs::{TimedToken, TranscriptChunk, TranscriptionResult};

/// One token with its timing and confidence
#[pyclass(name = "TimedToken", module = "scriptrs", frozen, get_all, eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyTimedToken {
    /// Token id in the model vocabulary
    token_id: u32,
    /// Token text, with a leading space where a word starts
    text: String,
    /// Start time in seconds
    start: f64,
    /// End time in seconds
    end: f64,
    /// Model confidence between 0 and 1
    confidence: f32,
}

#[pymethods]
impl PyTimedToken {
    fn __repr__(&self) -> String {
        format!(
            "TimedToken(text={:?}, start={:.3}, end={:.3}, confidence={:.3})",
            self.text, self.start, self.end, self.confidence
        )
    }
}

impl From<TimedToken> for PyTimedToken {
    fn from(token: TimedToken) -> Self {
        Self {
            token_id: token.token_id,
            text: token.text,
            start: token.start,
            end: token.end,
            confidence: token.confidence,
        }
    }
}

/// A timed span of the transcript
#[pyclass(name = "TranscriptChunk", module = "scriptrs", frozen, get_all, eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyTranscriptChunk {
    /// Start time in seconds
    start: f64,
    /// End time in seconds
    end: f64,
    /// Text of this span
    text: String,
}

#[pymethods]
impl PyTranscriptChunk {
    fn __repr__(&self) -> String {
        format!(
            "TranscriptChunk(start={:.3}, end={:.3}, text={:?})",
            self.start, self.end, self.text
        )
    }
}

impl From<TranscriptChunk> for PyTranscriptChunk {
    fn from(chunk: TranscriptChunk) -> Self {
        Self {
            start: chunk.start,
            end: chunk.end,
            text: chunk.text,
        }
    }
}

/// A finished transcription
#[pyclass(name = "TranscriptionResult", module = "scriptrs", frozen, get_all, eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyTranscriptionResult {
    /// Full transcript text
    text: String,
    /// Timed spans of the transcript
    chunks: Vec<PyTranscriptChunk>,
    /// Every token with its timing and confidence
    tokens: Vec<PyTimedToken>,
    /// Length of the input audio in seconds
    duration_seconds: f64,
}

#[pymethods]
impl PyTranscriptionResult {
    fn __str__(&self) -> &str {
        &self.text
    }

    fn __repr__(&self) -> String {
        format!(
            "TranscriptionResult(text={:?}, duration_seconds={:.3}, chunks={}, tokens={})",
            self.text,
            self.duration_seconds,
            self.chunks.len(),
            self.tokens.len()
        )
    }
}

impl From<TranscriptionResult> for PyTranscriptionResult {
    fn from(result: TranscriptionResult) -> Self {
        Self {
            text: result.text,
            chunks: result.chunks.into_iter().map(Into::into).collect(),
            tokens: result.tokens.into_iter().map(Into::into).collect(),
            duration_seconds: result.duration_seconds,
        }
    }
}
