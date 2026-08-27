use std::path::{Path, PathBuf};

use crate::deployment::ParakeetDeployment;
use crate::error::TranscriptionError;

#[cfg(feature = "vad")]
const VAD_DIR: &str = "vad/silero-vad.mlmodelc";
#[cfg(feature = "online")]
const SCRIPTRS_MODELS_DIR_ENV: &str = "SCRIPTRS_MODELS_DIR";
#[cfg(feature = "online")]
const SCRIPTRS_MODELS_REPO_ENV: &str = "SCRIPTRS_MODELS_REPO";

/// Resolved model paths for `scriptrs`
///
/// Most callers will use `TranscriptionPipeline::from_pretrained` or
/// `TranscriptionPipeline::from_dir` instead of constructing this directly.
#[derive(Debug, Clone)]
pub struct ModelBundle {
    root: PathBuf,
    encoder_dir: PathBuf,
    decoder_dir: PathBuf,
    joint_decision_dir: PathBuf,
    vocab_path: PathBuf,
    deployment: ParakeetDeployment,
    #[cfg(feature = "vad")]
    vad_dir: PathBuf,
}

impl ModelBundle {
    /// Resolve the v2 model layout from a local directory
    ///
    /// This only resolves paths. File existence is checked later by
    /// [`Self::validate_base`].
    pub fn from_dir(models_dir: impl Into<PathBuf>) -> Self {
        Self::from_dir_with_deployment(models_dir, ParakeetDeployment::v2())
    }

    /// Resolve model paths using an explicit deployment
    pub fn from_dir_with_deployment(
        models_dir: impl Into<PathBuf>,
        deployment: ParakeetDeployment,
    ) -> Self {
        let root = models_dir.into();
        let layout = deployment.layout();
        Self {
            encoder_dir: root.join(layout.encoder()),
            decoder_dir: root.join(layout.decoder()),
            joint_decision_dir: root.join(layout.joint_decision()),
            vocab_path: root.join(layout.vocab()),
            #[cfg(feature = "vad")]
            vad_dir: root.join(VAD_DIR),
            deployment,
            root,
        }
    }

    /// Deployment that owns this bundle's layout and limits
    pub fn deployment(&self) -> &ParakeetDeployment {
        &self.deployment
    }

    /// Validate that the deployment's Parakeet assets are present
    pub fn validate_base(&self) -> Result<(), TranscriptionError> {
        let required = [
            self.encoder_dir.as_path(),
            self.decoder_dir.as_path(),
            self.joint_decision_dir.as_path(),
            self.vocab_path.as_path(),
        ];
        let Some(missing) = required.iter().copied().find(|path| !path.exists()) else {
            return Ok(());
        };

        let sibling = self.root.join(self.deployment.sibling_vocab_path());
        if sibling.exists() {
            return Err(TranscriptionError::LayoutMismatch {
                expected: self.deployment.version(),
                found: self.deployment.sibling_version(),
            });
        }

        Err(TranscriptionError::MissingModelAsset {
            path: missing.to_path_buf(),
        })
    }

    #[cfg(feature = "vad")]
    /// Validate that the Parakeet and VAD model assets are present
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

    pub(crate) fn decoder_dir(&self) -> &Path {
        &self.decoder_dir
    }

    pub(crate) fn joint_decision_dir(&self) -> &Path {
        &self.joint_decision_dir
    }

    pub(crate) fn vocab_path(&self) -> &Path {
        &self.vocab_path
    }

    #[cfg(feature = "vad")]
    pub(crate) fn vad_dir(&self) -> &Path {
        &self.vad_dir
    }

    /// Return the root directory that contains the model bundle
    pub fn root(&self) -> &Path {
        &self.root
    }

    #[cfg(feature = "online")]
    /// Download the v2 Parakeet model bundle from Hugging Face
    ///
    /// By default this resolves models from `avencera/scriptrs-models`. Set
    /// `SCRIPTRS_MODELS_DIR` to force a local bundle or `SCRIPTRS_MODELS_REPO`
    /// to override the Hugging Face repo.
    pub fn from_pretrained() -> Result<Self, hf_hub::api::sync::ApiError> {
        Self::from_pretrained_with_deployment(ParakeetDeployment::v2())
    }

    #[cfg(feature = "online")]
    /// Download a Parakeet bundle for an explicit deployment
    pub fn from_pretrained_with_deployment(
        deployment: ParakeetDeployment,
    ) -> Result<Self, hf_hub::api::sync::ApiError> {
        if let Ok(models_dir) = std::env::var(SCRIPTRS_MODELS_DIR_ENV) {
            return Ok(Self::from_dir_with_deployment(models_dir, deployment));
        }
        ModelManager::new()?.ensure_with_deployment(deployment, false)
    }

    #[cfg(all(feature = "online", feature = "vad"))]
    /// Download the Parakeet and VAD model bundle from Hugging Face
    ///
    /// By default this resolves models from `avencera/scriptrs-models`. Set
    /// `SCRIPTRS_MODELS_DIR` to force a local bundle or `SCRIPTRS_MODELS_REPO`
    /// to override the Hugging Face repo.
    pub fn from_pretrained_long_form() -> Result<Self, hf_hub::api::sync::ApiError> {
        Self::from_pretrained_long_form_with_deployment(ParakeetDeployment::v2())
    }

    #[cfg(all(feature = "online", feature = "vad"))]
    /// Download Parakeet and VAD assets for an explicit deployment
    pub fn from_pretrained_long_form_with_deployment(
        deployment: ParakeetDeployment,
    ) -> Result<Self, hf_hub::api::sync::ApiError> {
        if let Ok(models_dir) = std::env::var(SCRIPTRS_MODELS_DIR_ENV) {
            return Ok(Self::from_dir_with_deployment(models_dir, deployment));
        }
        ModelManager::new()?.ensure_with_deployment(deployment, true)
    }
}

#[cfg(feature = "online")]
const HF_REPO: &str = "avencera/scriptrs-models";

/// Downloads and caches model bundles from Hugging Face
///
/// This is mainly useful if you want direct control over the Hugging Face cache
/// directory or want to prefetch assets before building a pipeline.
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

    /// Ensure the v2 Parakeet model bundle is cached locally
    ///
    /// The returned [`ModelBundle`] points at the resolved snapshot directory
    /// inside the Hugging Face cache.
    pub fn ensure_base(&self) -> Result<ModelBundle, hf_hub::api::sync::ApiError> {
        self.ensure_with_deployment(ParakeetDeployment::v2(), false)
    }

    #[cfg(feature = "vad")]
    /// Ensure the Parakeet and VAD model bundle is cached locally
    ///
    /// The returned [`ModelBundle`] points at the resolved snapshot directory
    /// inside the Hugging Face cache.
    pub fn ensure_long_form(&self) -> Result<ModelBundle, hf_hub::api::sync::ApiError> {
        self.ensure_with_deployment(ParakeetDeployment::v2(), true)
    }

    /// Ensure a deployment's model files are cached locally
    pub fn ensure_with_deployment(
        &self,
        deployment: ParakeetDeployment,
        include_vad: bool,
    ) -> Result<ModelBundle, hf_hub::api::sync::ApiError> {
        #[cfg(feature = "vad")]
        let mut files = repo_files(&deployment);
        #[cfg(not(feature = "vad"))]
        let files = repo_files(&deployment);
        #[cfg(feature = "vad")]
        if include_vad {
            files.extend(mlmodelc_files(VAD_DIR));
        }
        #[cfg(not(feature = "vad"))]
        let _ = include_vad;
        let root = self.download_repo_layout(&self.repo, &files)?;
        Ok(ModelBundle::from_dir_with_deployment(root, deployment))
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
fn repo_files(deployment: &ParakeetDeployment) -> Vec<String> {
    let layout = deployment.layout();
    let mut files = vec![layout.vocab().to_owned()];
    files.extend(mlmodelc_files(layout.encoder()));
    files.extend(mlmodelc_files(layout.decoder()));
    files.extend(mlmodelc_files(layout.joint_decision()));
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

#[cfg(test)]
mod tests {
    use super::ModelBundle;
    use crate::deployment::{ParakeetDeployment, ParakeetVersion};
    use crate::error::TranscriptionError;
    use std::fs;
    use std::path::PathBuf;
    use tempfile::tempdir;

    fn fixture_models() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/models")
    }

    #[test]
    fn v2_bundle_resolves_existing_fixture_paths() {
        let bundle = ModelBundle::from_dir(fixture_models());
        bundle
            .validate_base()
            .expect("v2 fixture layout should validate");
        assert_eq!(bundle.deployment().version(), ParakeetVersion::V2);
        assert!(
            bundle
                .encoder_dir()
                .ends_with("parakeet-v2/encoder.mlmodelc")
        );
        assert!(bundle.vocab_path().ends_with("parakeet-v2/vocab.txt"));
    }

    #[test]
    fn v3_deployment_rejects_a_v2_layout() {
        let bundle =
            ModelBundle::from_dir_with_deployment(fixture_models(), ParakeetDeployment::v3());
        let error = bundle
            .validate_base()
            .expect_err("v3 must reject the v2 fixture layout");
        match error {
            TranscriptionError::LayoutMismatch { expected, found } => {
                assert_eq!(expected, ParakeetVersion::V3);
                assert_eq!(found, ParakeetVersion::V2);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn v2_deployment_rejects_a_v3_layout() {
        let temp_dir = tempdir().expect("temp dir should be created");
        let root = temp_dir.path();
        fs::create_dir_all(root.join("parakeet-v3/encoder.mlmodelc"))
            .expect("v3 encoder dir should be created");
        fs::create_dir_all(root.join("parakeet-v3/decoder.mlmodelc"))
            .expect("v3 decoder dir should be created");
        fs::create_dir_all(root.join("parakeet-v3/joint-decision.mlmodelc"))
            .expect("v3 joint dir should be created");
        fs::write(root.join("parakeet-v3/parakeet_vocab.json"), "{}")
            .expect("v3 vocab should be created");

        let bundle = ModelBundle::from_dir_with_deployment(root, ParakeetDeployment::v2());
        let error = bundle
            .validate_base()
            .expect_err("v2 must reject a v3-only layout");
        match error {
            TranscriptionError::LayoutMismatch { expected, found } => {
                assert_eq!(expected, ParakeetVersion::V2);
                assert_eq!(found, ParakeetVersion::V3);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn validate_base_requires_split_bundles() {
        let temp_dir = tempdir().expect("temp dir should be created");
        let root = temp_dir.path();
        let deployment = ParakeetDeployment::v2();
        fs::create_dir_all(root.join(deployment.layout().encoder()))
            .expect("encoder dir should be created");
        fs::create_dir_all(root.join(deployment.layout().decoder()))
            .expect("decoder dir should be created");
        fs::write(root.join(deployment.layout().vocab()), [])
            .expect("vocab file should be created");

        let bundle = ModelBundle::from_dir(root);
        let error = bundle
            .validate_base()
            .expect_err("missing joint bundle should fail");
        match error {
            TranscriptionError::MissingModelAsset { path } => {
                assert_eq!(path, root.join(deployment.layout().joint_decision()));
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn validate_base_rejects_fused_only_bundle() {
        let temp_dir = tempdir().expect("temp dir should be created");
        let root = temp_dir.path();
        let deployment = ParakeetDeployment::v2();
        fs::create_dir_all(root.join(deployment.layout().encoder()))
            .expect("encoder dir should be created");
        fs::create_dir_all(root.join("parakeet-v2/decoder-joint.mlmodelc"))
            .expect("fused decoder dir should be created");
        fs::write(root.join(deployment.layout().vocab()), [])
            .expect("vocab file should be created");

        let bundle = ModelBundle::from_dir(root);
        let error = bundle
            .validate_base()
            .expect_err("fused-only bundle should fail");
        match error {
            TranscriptionError::MissingModelAsset { path } => {
                assert_eq!(path, root.join(deployment.layout().decoder()));
            }
            other => panic!("unexpected error: {other}"),
        }
    }
}

#[cfg(all(test, feature = "online"))]
mod online_tests {
    use super::{ModelBundle, repo_files, snapshot_root_from_cached_file};
    use crate::deployment::ParakeetDeployment;
    use std::path::Path;

    #[test]
    fn base_repo_files_include_required_assets() {
        let deployment = ParakeetDeployment::v2();
        let files = repo_files(&deployment);
        let layout = deployment.layout();
        assert!(files.iter().any(|file| file == layout.vocab()));
        assert!(
            files
                .iter()
                .any(|file| file == &format!("{}/model.mil", layout.encoder()))
        );
        assert!(
            files
                .iter()
                .any(|file| file == &format!("{}/model.mil", layout.decoder()))
        );
        assert!(
            files
                .iter()
                .any(|file| file == &format!("{}/model.mil", layout.joint_decision()))
        );
        assert!(
            !files
                .iter()
                .any(|file| file.contains("decoder-joint.mlmodelc"))
        );
    }

    #[test]
    fn v3_repo_files_use_the_json_tokenizer() {
        let files = repo_files(&ParakeetDeployment::v3());
        assert!(
            files
                .iter()
                .any(|file| file == "parakeet-v3/parakeet_vocab.json")
        );
        assert!(
            files
                .iter()
                .any(|file| file == "parakeet-v3/encoder.mlmodelc/model.mil")
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

    #[test]
    fn pretrained_default_still_uses_v2() {
        assert_eq!(
            ModelBundle::from_dir("/tmp").deployment().version(),
            crate::deployment::ParakeetVersion::V2
        );
    }
}
