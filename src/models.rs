use std::path::{Path, PathBuf};

use crate::error::TranscriptionError;

const ENCODER_DIR: &str = "parakeet-v2/encoder.mlmodelc";
const DECODER_JOINT_DIR: &str = "parakeet-v2/decoder-joint.mlmodelc";
const VOCAB_FILE: &str = "parakeet-v2/vocab.txt";
#[cfg(feature = "long-form")]
const VAD_DIR: &str = "vad/silero-vad.mlmodelc";

/// Resolved model paths for `scriptrs`
#[derive(Debug, Clone)]
pub struct ModelBundle {
    root: PathBuf,
    encoder_dir: PathBuf,
    decoder_joint_dir: PathBuf,
    vocab_path: PathBuf,
    #[cfg(feature = "long-form")]
    vad_dir: PathBuf,
}

impl ModelBundle {
    /// Resolve the expected model layout from a local directory
    pub fn from_dir(models_dir: impl Into<PathBuf>) -> Self {
        let root = models_dir.into();
        Self {
            encoder_dir: root.join(ENCODER_DIR),
            decoder_joint_dir: root.join(DECODER_JOINT_DIR),
            vocab_path: root.join(VOCAB_FILE),
            #[cfg(feature = "long-form")]
            vad_dir: root.join(VAD_DIR),
            root,
        }
    }

    /// Validate that the base Parakeet model assets are present
    pub fn validate_base(&self) -> Result<(), TranscriptionError> {
        for path in [
            self.encoder_dir.as_path(),
            self.decoder_joint_dir.as_path(),
            self.vocab_path.as_path(),
        ] {
            if !path.exists() {
                return Err(TranscriptionError::MissingModelAsset {
                    path: path.to_path_buf(),
                });
            }
        }
        Ok(())
    }

    #[cfg(feature = "long-form")]
    /// Validate that the base Parakeet and VAD model assets are present
    pub fn validate_long_form(&self) -> Result<(), TranscriptionError> {
        self.validate_base()?;
        if !self.vad_dir.exists() {
            return Err(TranscriptionError::MissingModelAsset {
                path: self.vad_dir.clone(),
            });
        }
        Ok(())
    }

    pub(crate) fn encoder_dir(&self) -> &Path {
        &self.encoder_dir
    }

    pub(crate) fn decoder_joint_dir(&self) -> &Path {
        &self.decoder_joint_dir
    }

    pub(crate) fn vocab_path(&self) -> &Path {
        &self.vocab_path
    }

    #[cfg(feature = "long-form")]
    pub(crate) fn vad_dir(&self) -> &Path {
        &self.vad_dir
    }

    /// Return the root directory that contains the model bundle
    pub fn root(&self) -> &Path {
        &self.root
    }

    #[cfg(feature = "online")]
    /// Download the base Parakeet model bundle from Hugging Face
    pub fn from_pretrained() -> Result<Self, hf_hub::api::sync::ApiError> {
        ModelManager::new()?.ensure_base()
    }

    #[cfg(all(feature = "online", feature = "long-form"))]
    /// Download the Parakeet and VAD model bundle from Hugging Face
    pub fn from_pretrained_long_form() -> Result<Self, hf_hub::api::sync::ApiError> {
        ModelManager::new()?.ensure_long_form()
    }
}

#[cfg(feature = "online")]
const HF_REPO: &str = "avencera/scriptrs-models";

/// Downloads and caches model bundles from Hugging Face
#[cfg(feature = "online")]
pub struct ModelManager {
    repo: hf_hub::api::sync::ApiRepo,
}

#[cfg(feature = "online")]
impl ModelManager {
    /// Create a model manager using the default Hugging Face cache
    pub fn new() -> Result<Self, hf_hub::api::sync::ApiError> {
        let api = hf_hub::api::sync::Api::new()?;
        Ok(Self {
            repo: api.model(HF_REPO.to_owned()),
        })
    }

    /// Create a model manager using a custom cache directory
    pub fn with_cache_dir(cache_dir: PathBuf) -> Result<Self, hf_hub::api::sync::ApiError> {
        let api =
            hf_hub::api::sync::ApiBuilder::from_cache(hf_hub::Cache::new(cache_dir)).build()?;
        Ok(Self {
            repo: api.model(HF_REPO.to_owned()),
        })
    }

    /// Ensure the base Parakeet model bundle is cached locally
    pub fn ensure_base(&self) -> Result<ModelBundle, hf_hub::api::sync::ApiError> {
        self.ensure_bundle(required_base_files())
    }

    #[cfg(feature = "long-form")]
    /// Ensure the Parakeet and VAD model bundle is cached locally
    pub fn ensure_long_form(&self) -> Result<ModelBundle, hf_hub::api::sync::ApiError> {
        self.ensure_bundle(required_long_form_files())
    }

    fn ensure_bundle(
        &self,
        files: Vec<String>,
    ) -> Result<ModelBundle, hf_hub::api::sync::ApiError> {
        for file in &files {
            self.repo.get(file)?;
        }
        let first = self.repo.get(&files[0])?;
        let models_root = first
            .ancestors()
            .nth(3)
            .map(Path::to_path_buf)
            .unwrap_or(first);
        Ok(ModelBundle::from_dir(models_root))
    }
}

#[cfg(feature = "online")]
fn required_base_files() -> Vec<String> {
    let mut files = vec!["parakeet-v2/vocab.txt".to_owned()];
    files.extend(mlmodelc_files("parakeet-v2/encoder.mlmodelc"));
    files.extend(mlmodelc_files("parakeet-v2/decoder-joint.mlmodelc"));
    files
}

#[cfg(all(feature = "online", feature = "long-form"))]
fn required_long_form_files() -> Vec<String> {
    let mut files = required_base_files();
    files.extend(mlmodelc_files("vad/silero-vad.mlmodelc"));
    files
}

#[cfg(feature = "online")]
fn mlmodelc_files(prefix: &str) -> Vec<String> {
    vec![
        format!("{prefix}/model.mil"),
        format!("{prefix}/coremldata.bin"),
        format!("{prefix}/weights/weight.bin"),
        format!("{prefix}/analytics/coremldata.bin"),
    ]
}
