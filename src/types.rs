/// A single token with timing metadata
#[derive(Debug, Clone, PartialEq)]
pub struct TimedToken {
    /// Token identifier from the model vocabulary
    pub token_id: u32,
    /// Human-readable token text after SentencePiece cleanup
    pub text: String,
    /// Start time in seconds
    pub start: f64,
    /// End time in seconds
    pub end: f64,
    /// Confidence score for the token
    pub confidence: f32,
}

/// A text chunk in the final transcript
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptChunk {
    /// Chunk start time in seconds
    pub start: f64,
    /// Chunk end time in seconds
    pub end: f64,
    /// Decoded text for this chunk
    pub text: String,
}

/// Final transcription output
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionResult {
    /// Full transcript text
    pub text: String,
    /// Chunked transcript regions
    pub chunks: Vec<TranscriptChunk>,
    /// Token-level timestamps
    pub tokens: Vec<TimedToken>,
    /// Input audio duration in seconds
    pub duration_seconds: f64,
}

impl TranscriptionResult {
    pub(crate) fn empty(duration_seconds: f64) -> Self {
        Self {
            text: String::new(),
            chunks: Vec::new(),
            tokens: Vec::new(),
            duration_seconds,
        }
    }
}
