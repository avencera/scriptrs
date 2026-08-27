use crate::deployment::ParakeetDeployment;
use crate::language::LanguageSelection;

/// Configuration for single-chunk transcription
///
/// Defaults come from [`ParakeetDeployment::v2`]. Most callers should start
/// with `TranscriptionConfig::from_deployment` so the hard sample maximum and
/// frontend numbers stay owned by the deployment.
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
    /// Language selection for this run
    pub language: LanguageSelection,
}

impl Default for TranscriptionConfig {
    fn default() -> Self {
        Self::from_deployment(&ParakeetDeployment::v2())
    }
}

impl TranscriptionConfig {
    /// Build frontend and limit settings from a deployment
    pub fn from_deployment(deployment: &ParakeetDeployment) -> Self {
        let frontend = deployment.frontend();
        Self {
            sample_rate: frontend.sample_rate(),
            feature_size: frontend.feature_size(),
            n_fft: frontend.n_fft(),
            win_length: frontend.win_length(),
            hop_length: frontend.hop_length(),
            preemphasis: frontend.preemphasis(),
            max_audio_samples: deployment.hard_sample_max(),
            samples_per_encoder_frame: frontend.samples_per_encoder_frame(),
            language: LanguageSelection::Auto,
        }
    }

    pub(crate) fn max_duration_seconds(&self) -> f64 {
        self.max_audio_samples as f64 / self.sample_rate as f64
    }

    pub(crate) fn max_feature_frames(&self) -> usize {
        self.max_audio_samples / self.hop_length + 1
    }
}

#[cfg(test)]
mod tests {
    use super::TranscriptionConfig;
    use crate::deployment::ParakeetDeployment;

    #[test]
    fn default_config_matches_v2_deployment() {
        let from_v2 = TranscriptionConfig::from_deployment(&ParakeetDeployment::v2());
        let default = TranscriptionConfig::default();
        assert_eq!(default.sample_rate, from_v2.sample_rate);
        assert_eq!(default.max_audio_samples, from_v2.max_audio_samples);
        assert_eq!(default.max_audio_samples, 240_000);
    }

    #[test]
    fn v3_config_uses_deployment_hard_max() {
        let config = TranscriptionConfig::from_deployment(&ParakeetDeployment::v3());
        assert_eq!(
            config.max_audio_samples,
            ParakeetDeployment::v3().hard_sample_max()
        );
    }
}
