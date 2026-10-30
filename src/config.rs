use crate::language::LanguageSelection;

/// Per-run settings a caller chooses for single-chunk transcription
///
/// Frontend numbers and the single-chunk sample limit belong to the model's
/// deployment, so they are read from [`crate::ParakeetDeployment::frontend`]
/// and [`crate::ParakeetDeployment::hard_sample_max`] instead of being set here
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TranscriptionConfig {
    /// Language selection for this run, checked against the deployment's
    /// allowlist when a run starts
    pub language: LanguageSelection,
}
