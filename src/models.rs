use std::path::{Path, PathBuf};

use crate::error::TranscriptionError;

const ENCODER_DIR: &str = "parakeet-v2/encoder.mlmodelc";
const DECODER_JOINT_DIR: &str = "parakeet-v2/decoder-joint.mlmodelc";
const DECODER_DIR: &str = "parakeet-v2/decoder.mlmodelc";
const JOINT_DECISION_DIR: &str = "parakeet-v2/joint-decision.mlmodelc";
const VOCAB_FILE: &str = "parakeet-v2/vocab.txt";
#[cfg(feature = "long-form")]
const VAD_DIR: &str = "vad/silero-vad.mlmodelc";

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

    #[cfg(feature = "online")]
    fn from_pretrained_paths(
        root: PathBuf,
        encoder_dir: PathBuf,
        decoder_dir: PathBuf,
        joint_decision_dir: PathBuf,
        vocab_path: PathBuf,
        #[cfg(feature = "long-form")] vad_dir: Option<PathBuf>,
    ) -> Self {
        Self {
            root,
            encoder_dir,
            decoder_joint_dir: None,
            decoder_dir: Some(decoder_dir),
            joint_decision_dir: Some(joint_decision_dir),
            vocab_path,
            #[cfg(feature = "long-form")]
            vad_dir: vad_dir.unwrap_or_default(),
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
        ModelManager::new()?.ensure_base()
    }

    #[cfg(all(feature = "online", feature = "long-form"))]
    /// Download the Parakeet and VAD model bundle from Hugging Face
    pub fn from_pretrained_long_form() -> Result<Self, hf_hub::api::sync::ApiError> {
        ModelManager::new()?.ensure_long_form()
    }
}

#[cfg(feature = "online")]
const PARAKEET_COREML_REPO: &str = "FluidInference/parakeet-tdt-0.6b-v2-coreml";
#[cfg(feature = "online")]
const PARAKEET_VOCAB_REPO: &str = "istupakov/parakeet-tdt-0.6b-v2-onnx";
#[cfg(all(feature = "online", feature = "long-form"))]
const SILERO_VAD_REPO: &str = "aufklarer/Silero-VAD-v5-CoreML";

/// Downloads and caches model bundles from Hugging Face
#[cfg(feature = "online")]
pub struct ModelManager {
    coreml_repo: hf_hub::api::sync::ApiRepo,
    vocab_repo: hf_hub::api::sync::ApiRepo,
    #[cfg(feature = "long-form")]
    vad_repo: hf_hub::api::sync::ApiRepo,
}

#[cfg(feature = "online")]
impl ModelManager {
    /// Create a model manager using the default Hugging Face cache
    pub fn new() -> Result<Self, hf_hub::api::sync::ApiError> {
        let api = hf_hub::api::sync::Api::new()?;
        Ok(Self {
            coreml_repo: api.model(PARAKEET_COREML_REPO.to_owned()),
            vocab_repo: api.model(PARAKEET_VOCAB_REPO.to_owned()),
            #[cfg(feature = "long-form")]
            vad_repo: api.model(SILERO_VAD_REPO.to_owned()),
        })
    }

    /// Create a model manager using a custom cache directory
    pub fn with_cache_dir(cache_dir: PathBuf) -> Result<Self, hf_hub::api::sync::ApiError> {
        let api =
            hf_hub::api::sync::ApiBuilder::from_cache(hf_hub::Cache::new(cache_dir)).build()?;
        Ok(Self {
            coreml_repo: api.model(PARAKEET_COREML_REPO.to_owned()),
            vocab_repo: api.model(PARAKEET_VOCAB_REPO.to_owned()),
            #[cfg(feature = "long-form")]
            vad_repo: api.model(SILERO_VAD_REPO.to_owned()),
        })
    }

    /// Ensure the base Parakeet model bundle is cached locally
    pub fn ensure_base(&self) -> Result<ModelBundle, hf_hub::api::sync::ApiError> {
        let encoder_dir = self.download_mlmodelc_bundle(&self.coreml_repo, "Encoder.mlmodelc")?;
        let decoder_dir = self.download_mlmodelc_bundle(&self.coreml_repo, "Decoder.mlmodelc")?;
        let joint_decision_dir =
            self.download_mlmodelc_bundle(&self.coreml_repo, "JointDecision.mlmodelc")?;
        let vocab_path = self.vocab_repo.get("vocab.txt")?;
        let root = encoder_dir
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| encoder_dir.clone());
        Ok(ModelBundle::from_pretrained_paths(
            root,
            encoder_dir,
            decoder_dir,
            joint_decision_dir,
            vocab_path,
            #[cfg(feature = "long-form")]
            None,
        ))
    }

    #[cfg(feature = "long-form")]
    /// Ensure the Parakeet and VAD model bundle is cached locally
    pub fn ensure_long_form(&self) -> Result<ModelBundle, hf_hub::api::sync::ApiError> {
        let mut bundle = self.ensure_base()?;
        bundle.vad_dir = self.download_mlmodelc_bundle(&self.vad_repo, "silero_vad.mlmodelc")?;
        Ok(bundle)
    }

    fn download_mlmodelc_bundle(
        &self,
        repo: &hf_hub::api::sync::ApiRepo,
        prefix: &str,
    ) -> Result<PathBuf, hf_hub::api::sync::ApiError> {
        for file in mlmodelc_files(prefix) {
            repo.get(&file)?;
        }
        let model = repo.get(&format!("{prefix}/model.mil"))?;
        Ok(model.parent().map(Path::to_path_buf).unwrap_or(model))
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
