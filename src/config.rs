use crate::deployment::ParakeetDeployment;
use crate::error::TranscriptionError;
use crate::language::LanguageSelection;

/// Configuration for single-chunk transcription
///
/// Defaults come from [`ParakeetDeployment::v2`]. Build this with
/// [`Self::from_deployment`] so the hard sample maximum and frontend numbers
/// stay owned by the deployment.
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

    /// Validate that this configuration belongs to `deployment`
    ///
    /// Frontend dimensions and the single-chunk limit are deployment-owned.
    /// Callers may still use a configuration value as an input boundary, but
    /// a value that was edited or built for another deployment is rejected
    /// before it reaches the extractor or model.
    pub fn validate_for(&self, deployment: &ParakeetDeployment) -> Result<(), TranscriptionError> {
        let frontend = deployment.frontend();
        ensure_equal("sample_rate", self.sample_rate, frontend.sample_rate())?;
        ensure_equal("feature_size", self.feature_size, frontend.feature_size())?;
        ensure_equal("n_fft", self.n_fft, frontend.n_fft())?;
        ensure_equal("win_length", self.win_length, frontend.win_length())?;
        ensure_equal("hop_length", self.hop_length, frontend.hop_length())?;
        ensure_equal("preemphasis", self.preemphasis, frontend.preemphasis())?;
        ensure_equal(
            "max_audio_samples",
            self.max_audio_samples,
            deployment.hard_sample_max(),
        )?;
        ensure_equal(
            "samples_per_encoder_frame",
            self.samples_per_encoder_frame,
            frontend.samples_per_encoder_frame(),
        )?;
        deployment.select_language(self.language)?;
        Ok(())
    }

    pub(crate) fn max_duration_seconds(&self) -> f64 {
        self.max_audio_samples as f64 / self.sample_rate as f64
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn max_feature_frames(&self) -> usize {
        self.max_audio_samples / self.hop_length + 1
    }
}

fn ensure_equal<T: PartialEq + std::fmt::Display>(
    field: &str,
    actual: T,
    expected: T,
) -> Result<(), TranscriptionError> {
    if actual == expected {
        return Ok(());
    }

    Err(TranscriptionError::InvalidConfiguration(format!(
        "{field}={actual} does not match deployment value {expected}"
    )))
}

#[cfg(test)]
mod tests {
    use super::TranscriptionConfig;
    use crate::deployment::ParakeetDeployment;
    use crate::error::TranscriptionError;
    use crate::language::{LanguageSelection, ParakeetLanguage};

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
        let deployment = ParakeetDeployment::v3();
        let config = TranscriptionConfig::from_deployment(&deployment);
        assert_eq!(config.max_audio_samples, deployment.hard_sample_max());
        config
            .validate_for(&deployment)
            .expect("deployment-owned config should validate");
    }

    #[test]
    fn config_rejects_a_limit_that_disagrees_with_its_deployment() {
        let deployment = ParakeetDeployment::v3();
        let mut config = TranscriptionConfig::from_deployment(&deployment);
        config.max_audio_samples += 1;

        let error = config
            .validate_for(&deployment)
            .expect_err("edited deployment-owned limit should fail");
        assert!(matches!(error, TranscriptionError::InvalidConfiguration(_)));
    }

    #[test]
    fn config_validates_language_at_the_deployment_boundary() {
        let deployment = ParakeetDeployment::v2();
        let mut config = TranscriptionConfig::from_deployment(&deployment);
        config.language = LanguageSelection::Language(ParakeetLanguage::Polish);

        let error = config
            .validate_for(&deployment)
            .expect_err("v2 must reject a non-English language");
        assert!(matches!(
            error,
            TranscriptionError::UnsupportedLanguage { .. }
        ));
    }
}
