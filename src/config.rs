use crate::constants::{
    MAX_MODEL_SAMPLES, MEL_HOP_SAMPLES, SAMPLE_RATE, SAMPLES_PER_ENCODER_FRAME,
};

/// Configuration for single-chunk transcription
///
/// The defaults match the current Parakeet TDT v2 frontend. Most callers should
/// start with `TranscriptionConfig::default()` and only override fields when
/// they intentionally want to change frontend behavior.
#[derive(Debug, Clone)]
pub struct TranscriptionConfig {
    /// Sample rate expected by the pipeline
    pub sample_rate: usize,
    /// Feature count for the log-mel frontend
    pub feature_size: usize,
    /// FFT window size
    pub n_fft: usize,
    /// STFT window length
    pub win_length: usize,
    /// STFT hop length
    pub hop_length: usize,
    /// Preemphasis coefficient
    pub preemphasis: f32,
    /// Maximum supported single-chunk sample count
    pub max_audio_samples: usize,
    /// Encoder frame size in samples
    pub samples_per_encoder_frame: usize,
}

impl Default for TranscriptionConfig {
    fn default() -> Self {
        Self {
            sample_rate: SAMPLE_RATE,
            feature_size: 128,
            n_fft: 512,
            win_length: 400,
            hop_length: MEL_HOP_SAMPLES,
            preemphasis: 0.97,
            max_audio_samples: MAX_MODEL_SAMPLES,
            samples_per_encoder_frame: SAMPLES_PER_ENCODER_FRAME,
        }
    }
}

impl TranscriptionConfig {
    pub(crate) fn max_duration_seconds(&self) -> f64 {
        self.max_audio_samples as f64 / self.sample_rate as f64
    }

    pub(crate) fn max_feature_frames(&self) -> usize {
        self.max_audio_samples / self.hop_length + 1
    }
}
