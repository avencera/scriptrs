use std::path::{Path, PathBuf};

use crate::error::TranscriptionError;

const ENCODER_DIR: &str = "parakeet-v2/encoder.mlmodelc";
const DECODER_JOINT_DIR: &str = "parakeet-v2/decoder-joint.mlmodelc";
const DECODER_DIR: &str = "parakeet-v2/decoder.mlmodelc";
const JOINT_DECISION_DIR: &str = "parakeet-v2/joint-decision.mlmodelc";
const VOCAB_FILE: &str = "parakeet-v2/vocab.txt";
#[cfg(feature = "long-form")]
const VAD_DIR: &str = "vad/silero-vad.mlmodelc";
#[cfg(feature = "online")]
const SCRIPTRS_MODELS_DIR_ENV: &str = "SCRIPTRS_MODELS_DIR";
#[cfg(feature = "online")]
const SCRIPTRS_MODELS_REPO_ENV: &str = "SCRIPTRS_MODELS_REPO";

/// Resolved model paths for `scriptrs`
#[derive(Debug, Clone)]
pub struct ModelBundle {
    root: PathBuf,
    encoder_dir: PathBuf,
    decoder_joint_dir: Option<PathBuf>,
    decoder_dir: Option<PathBuf>,
    joint_decision_dir: Option<PathBuf>,
    vocab_path: PathBuf,
    #[cfg(feature = "long-form")]
    vad_dir: PathBuf,
}

impl ModelBundle {
    /// Resolve the expected model layout from a local directory
    pub fn from_dir(models_dir: impl Into<PathBuf>) -> Self {
        let root = models_dir.into();
        let decoder_joint_dir = root.join(DECODER_JOINT_DIR);
        let decoder_dir = root.join(DECODER_DIR);
        let joint_decision_dir = root.join(JOINT_DECISION_DIR);
        Self {
            encoder_dir: root.join(ENCODER_DIR),
            decoder_joint_dir: decoder_joint_dir.exists().then_some(decoder_joint_dir),
            decoder_dir: decoder_dir.exists().then_some(decoder_dir),
            joint_decision_dir: joint_decision_dir.exists().then_some(joint_decision_dir),
            vocab_path: root.join(VOCAB_FILE),
            #[cfg(feature = "long-form")]
            vad_dir: root.join(VAD_DIR),
            root,
        }
    }

    /// Validate that the base Parakeet model assets are present
    pub fn validate_base(&self) -> Result<(), TranscriptionError> {
        for path in [self.encoder_dir.as_path(), self.vocab_path.as_path()] {
            if path.exists() {
                continue;
            }
            return Err(TranscriptionError::MissingModelAsset {
                path: path.to_path_buf(),
            });
        }

        if let Some(path) = self.decoder_joint_dir()
            && path.exists()
        {
            return Ok(());
        }

        for path in [self.decoder_dir(), self.joint_decision_dir()] {
            let Some(path) = path else {
                return Err(TranscriptionError::MissingModelAsset {
                    path: self.root.join(DECODER_DIR),
                });
            };
            if path.exists() {
                continue;
            }
            return Err(TranscriptionError::MissingModelAsset {
                path: path.to_path_buf(),
            });
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

    pub(crate) fn decoder_joint_dir(&self) -> Option<&Path> {
        self.decoder_joint_dir.as_deref()
    }

    pub(crate) fn decoder_dir(&self) -> Option<&Path> {
        self.decoder_dir.as_deref()
    }

    pub(crate) fn joint_decision_dir(&self) -> Option<&Path> {
        self.joint_decision_dir.as_deref()
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
        if let Ok(models_dir) = std::env::var(SCRIPTRS_MODELS_DIR_ENV) {
            return Ok(Self::from_dir(models_dir));
        }
        ModelManager::new()?.ensure_base()
    }

    #[cfg(all(feature = "online", feature = "long-form"))]
    /// Download the Parakeet and VAD model bundle from Hugging Face
    pub fn from_pretrained_long_form() -> Result<Self, hf_hub::api::sync::ApiError> {
        if let Ok(models_dir) = std::env::var(SCRIPTRS_MODELS_DIR_ENV) {
            return Ok(Self::from_dir(models_dir));
        }
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
        Ok(Self::from_api(api))
    }

    /// Create a model manager using a custom cache directory
    pub fn with_cache_dir(cache_dir: PathBuf) -> Result<Self, hf_hub::api::sync::ApiError> {
        let api =
            hf_hub::api::sync::ApiBuilder::from_cache(hf_hub::Cache::new(cache_dir)).build()?;
        Ok(Self::from_api(api))
    }

    fn from_api(api: hf_hub::api::sync::Api) -> Self {
        let repo_id = std::env::var(SCRIPTRS_MODELS_REPO_ENV).unwrap_or_else(|_| HF_REPO.into());
        Self {
            repo: api.model(repo_id),
        }
    }

    /// Ensure the base Parakeet model bundle is cached locally
    pub fn ensure_base(&self) -> Result<ModelBundle, hf_hub::api::sync::ApiError> {
        self.ensure_unified(false)
    }

    #[cfg(feature = "long-form")]
    /// Ensure the Parakeet and VAD model bundle is cached locally
    pub fn ensure_long_form(&self) -> Result<ModelBundle, hf_hub::api::sync::ApiError> {
        self.ensure_unified(true)
    }

    fn ensure_unified(
        &self,
        include_vad: bool,
    ) -> Result<ModelBundle, hf_hub::api::sync::ApiError> {
        #[cfg(not(feature = "long-form"))]
        let files = {
            let _ = include_vad;
            base_repo_files()
        };
        #[cfg(feature = "long-form")]
        let files = {
            let mut files = base_repo_files();
            if include_vad {
                files.extend(mlmodelc_files(VAD_DIR));
            }
            files
        };
        let root = self.download_repo_layout(&self.repo, &files)?;
        Ok(ModelBundle::from_dir(root))
    }

    fn download_repo_layout(
        &self,
        repo: &hf_hub::api::sync::ApiRepo,
        files: &[String],
    ) -> Result<PathBuf, hf_hub::api::sync::ApiError> {
        let mut first = None;
        for file in files {
            let cached_path = repo.get(file)?;
            if first.is_none() {
                first = Some((cached_path, file.as_str()));
            }
        }

        let Some((cached_path, relative_path)) = first else {
            unreachable!("model file list should never be empty");
        };
        Ok(snapshot_root_from_cached_file(&cached_path, relative_path))
    }
}

#[cfg(feature = "online")]
fn mlmodelc_files(prefix: &str) -> Vec<String> {
    vec![
        format!("{prefix}/model.mil"),
        format!("{prefix}/coremldata.bin"),
        format!("{prefix}/weights/weight.bin"),
        format!("{prefix}/analytics/coremldata.bin"),
        format!("{prefix}/metadata.json"),
    ]
}

#[cfg(feature = "online")]
fn base_repo_files() -> Vec<String> {
    let mut files = vec![VOCAB_FILE.to_owned()];
    files.extend(mlmodelc_files(ENCODER_DIR));
    files.extend(mlmodelc_files(DECODER_DIR));
    files.extend(mlmodelc_files(JOINT_DECISION_DIR));
    files
}

#[cfg(feature = "online")]
fn snapshot_root_from_cached_file(cached_path: &Path, relative_path: &str) -> PathBuf {
    let mut root = cached_path.to_path_buf();
    for _ in Path::new(relative_path).components() {
        root.pop();
    }
    root
}

#[cfg(all(test, feature = "online"))]
mod tests {
    use super::{ENCODER_DIR, VOCAB_FILE, base_repo_files, snapshot_root_from_cached_file};
    use std::path::Path;

    #[test]
    fn base_repo_files_include_required_assets() {
        let files = base_repo_files();
        assert!(files.iter().any(|file| file == VOCAB_FILE));
        assert!(
            files
                .iter()
                .any(|file| file == &format!("{ENCODER_DIR}/model.mil"))
        );
    }

    #[test]
    fn snapshot_root_strips_relative_layout() {
        let cached = Path::new(
            "/tmp/hf/models--avencera--scriptrs-models/snapshots/abc/parakeet-v2/encoder.mlmodelc/model.mil",
        );
        let root = snapshot_root_from_cached_file(cached, "parakeet-v2/encoder.mlmodelc/model.mil");
        assert_eq!(
            root,
            Path::new("/tmp/hf/models--avencera--scriptrs-models/snapshots/abc")
        );
    }
}
