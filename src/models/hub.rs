//! Pinned Hugging Face sources for the published bundles, and their downloads

use std::path::{Path, PathBuf};

use super::ModelBundle;
#[cfg(feature = "vad")]
use super::VAD_DIR;
use crate::deployment::ParakeetDeployment;
#[cfg(feature = "cuda")]
use crate::deployment::ParakeetVersion;
use crate::hf::{Hub, HubError, HubRepo};

const SCRIPTRS_MODELS_DIR_ENV: &str = "SCRIPTRS_MODELS_DIR";

impl ModelBundle {
    /// Download the v2 Parakeet model bundle from Hugging Face
    ///
    /// By default this resolves models from the pinned `avencera/scriptrs-models`
    /// snapshot. Set `SCRIPTRS_MODELS_DIR` to force a local bundle
    pub fn from_pretrained() -> Result<Self, HubError> {
        Self::from_pretrained_with_deployment(ParakeetDeployment::v2())
    }

    /// Download a Parakeet bundle for an explicit deployment
    pub fn from_pretrained_with_deployment(
        deployment: ParakeetDeployment,
    ) -> Result<Self, HubError> {
        if let Some(models_dir) = models_dir_override() {
            return Ok(Self::from_dir_with_deployment(models_dir, deployment));
        }
        ModelManager::new()?.ensure_with_deployment(deployment, false)
    }

    /// Download the Parakeet and VAD model bundle from Hugging Face
    ///
    /// By default this resolves models from the pinned `avencera/scriptrs-models`
    /// snapshot. Set `SCRIPTRS_MODELS_DIR` to force a local bundle
    #[cfg(feature = "vad")]
    pub fn from_pretrained_long_form() -> Result<Self, HubError> {
        Self::from_pretrained_long_form_with_deployment(ParakeetDeployment::v2())
    }

    /// Download Parakeet and VAD assets for an explicit deployment
    #[cfg(feature = "vad")]
    pub fn from_pretrained_long_form_with_deployment(
        deployment: ParakeetDeployment,
    ) -> Result<Self, HubError> {
        if let Some(models_dir) = models_dir_override() {
            return Ok(Self::from_dir_with_deployment(models_dir, deployment));
        }
        ModelManager::new()?.ensure_with_deployment(deployment, true)
    }

    /// Download native CUDA weights for a deployment from Hugging Face
    ///
    /// This resolves `model.safetensors` and the vocabulary from a pinned
    /// snapshot of the deployment's model repository,
    /// `avencera/parakeet-tdt-0.6b-v2-cuda` or `avencera/parakeet-tdt-0.6b-v3-cuda`,
    /// then behaves like [`Self::from_native_dir`]. Set `SCRIPTRS_MODELS_DIR` to
    /// force a local native bundle
    #[cfg(feature = "cuda")]
    pub fn from_pretrained_native(deployment: ParakeetDeployment) -> Result<Self, HubError> {
        if let Some(models_dir) = models_dir_override() {
            return Ok(Self::from_native_dir(models_dir, deployment));
        }
        ModelManager::new()?.ensure_native(deployment, false)
    }

    /// Download native CUDA weights and the native Silero VAD weights
    ///
    /// The VAD weights come from a pinned snapshot of
    /// `avencera/silero-vad-v5-safetensors`. Set `SCRIPTRS_MODELS_DIR` to force a
    /// local native bundle
    #[cfg(all(feature = "cuda", feature = "vad"))]
    pub fn from_pretrained_native_long_form(
        deployment: ParakeetDeployment,
    ) -> Result<Self, HubError> {
        if let Some(models_dir) = models_dir_override() {
            return Ok(Self::from_native_dir(models_dir, deployment));
        }
        ModelManager::new()?.ensure_native(deployment, true)
    }
}

const HF_REPO: HuggingFaceRepository = HuggingFaceRepository("avencera/scriptrs-models");
const HF_REVISION: HuggingFaceRevision =
    HuggingFaceRevision("63f6f9cbce43e5561c219229896f86f20c0eb660");

// native CUDA weights live in one repository per model, named after the model, so
// each is listed under its base model on Hugging Face and found by searching for it
#[cfg(feature = "cuda")]
const NATIVE_V2_SOURCE: HuggingFaceModelSource = HuggingFaceModelSource {
    repository: HuggingFaceRepository("avencera/parakeet-tdt-0.6b-v2-cuda"),
    revision: HuggingFaceRevision("ff259a9006fcd3fe19077ec5b76165c7703a7245"),
};
#[cfg(feature = "cuda")]
const NATIVE_V3_SOURCE: HuggingFaceModelSource = HuggingFaceModelSource {
    repository: HuggingFaceRepository("avencera/parakeet-tdt-0.6b-v3-cuda"),
    revision: HuggingFaceRevision("01ffcdb65d1e0087f4f12bbe87f5413f9b429626"),
};
#[cfg(all(feature = "cuda", feature = "vad"))]
const NATIVE_VAD_SOURCE: HuggingFaceModelSource = HuggingFaceModelSource {
    repository: HuggingFaceRepository("avencera/silero-vad-v5-safetensors"),
    revision: HuggingFaceRevision("bc58b9e061a5cb07bc3ec839df301647753dcc59"),
};
/// Weight file name in each native model repository
#[cfg(feature = "cuda")]
const NATIVE_HUB_WEIGHTS: &str = "model.safetensors";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HuggingFaceRepository(&'static str);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HuggingFaceRevision(&'static str);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HuggingFaceModelSource {
    repository: HuggingFaceRepository,
    revision: HuggingFaceRevision,
}

impl HuggingFaceModelSource {
    const fn runtime() -> Self {
        Self {
            repository: HF_REPO,
            revision: HF_REVISION,
        }
    }

    #[cfg(feature = "cuda")]
    fn native(deployment: &ParakeetDeployment) -> Self {
        match deployment.version() {
            ParakeetVersion::V2 => NATIVE_V2_SOURCE,
            ParakeetVersion::V3 => NATIVE_V3_SOURCE,
        }
    }

    fn pinned_repo(self, hub: &Hub) -> HubRepo {
        hub.model(self.repository.0, self.revision.0)
    }
}

/// Downloads and caches model bundles from Hugging Face
///
/// This is mainly useful if you want direct control over the Hugging Face cache
/// directory or want to prefetch assets before building a pipeline
pub struct ModelManager {
    // only native weights live outside the Core ML repository, in per-model ones
    #[cfg(feature = "cuda")]
    hub: Hub,
    repo: HubRepo,
}

impl ModelManager {
    /// Create a model manager using the default Hugging Face cache
    ///
    /// That is `HF_HUB_CACHE` when set, else `$HF_HOME/hub`, else
    /// `~/.cache/huggingface/hub`, the same cache the Python `huggingface_hub` uses.
    /// Downloads read `HF_ENDPOINT` and the token in `HF_TOKEN` or saved by
    /// `hf auth login`
    pub fn new() -> Result<Self, HubError> {
        Ok(Self::from_hub(Hub::from_env()?))
    }

    /// Create a model manager using a custom cache directory
    pub fn with_cache_dir(cache_dir: PathBuf) -> Result<Self, HubError> {
        Ok(Self::from_hub(Hub::with_cache_dir(cache_dir)?))
    }

    fn from_hub(hub: Hub) -> Self {
        Self {
            repo: HuggingFaceModelSource::runtime().pinned_repo(&hub),
            #[cfg(feature = "cuda")]
            hub,
        }
    }

    /// Ensure the v2 Parakeet model bundle is cached locally
    ///
    /// The returned [`ModelBundle`] points at the resolved snapshot directory
    /// inside the Hugging Face cache
    pub fn ensure_base(&self) -> Result<ModelBundle, HubError> {
        self.ensure_with_deployment(ParakeetDeployment::v2(), false)
    }

    /// Ensure the Parakeet and VAD model bundle is cached locally
    ///
    /// The returned [`ModelBundle`] points at the resolved snapshot directory
    /// inside the Hugging Face cache
    #[cfg(feature = "vad")]
    pub fn ensure_long_form(&self) -> Result<ModelBundle, HubError> {
        self.ensure_with_deployment(ParakeetDeployment::v2(), true)
    }

    /// Ensure a deployment's model files are cached locally
    pub fn ensure_with_deployment(
        &self,
        deployment: ParakeetDeployment,
        include_vad: bool,
    ) -> Result<ModelBundle, HubError> {
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

    /// Ensure a deployment's native CUDA weights are cached locally
    ///
    /// The weights come from the deployment's model repository, such as
    /// `avencera/parakeet-tdt-0.6b-v3-cuda`, and with `include_vad` the Silero VAD
    /// weights from `avencera/silero-vad-v5-safetensors`, each at a pinned revision.
    /// The returned [`ModelBundle`] is a native bundle rooted at the Parakeet
    /// snapshot directory
    #[cfg(feature = "cuda")]
    pub fn ensure_native(
        &self,
        deployment: ParakeetDeployment,
        include_vad: bool,
    ) -> Result<ModelBundle, HubError> {
        let repo = HuggingFaceModelSource::native(&deployment).pinned_repo(&self.hub);
        let weights_path = repo.get(NATIVE_HUB_WEIGHTS)?;
        let vocab_path = repo.get(native_hub_vocab(&deployment))?;

        #[cfg(feature = "vad")]
        let vad_path = match include_vad {
            true => Some(
                NATIVE_VAD_SOURCE
                    .pinned_repo(&self.hub)
                    .get(NATIVE_HUB_WEIGHTS)?,
            ),
            false => None,
        };
        #[cfg(not(feature = "vad"))]
        let vad_path = {
            let _ = include_vad;
            None
        };

        Ok(ModelBundle::from_native_files(
            weights_path,
            vocab_path,
            vad_path,
            deployment,
        ))
    }

    fn download_repo_layout(&self, repo: &HubRepo, files: &[String]) -> Result<PathBuf, HubError> {
        let mut first = None;
        for file in files {
            let cached_path = repo.get(file)?;
            if first.is_none() {
                first = Some((cached_path, file.as_str()));
            }
        }

        // every layout lists files; an empty list names no file to root the snapshot at
        let Some((cached_path, relative_path)) = first else {
            return Err(HubError::InvalidFile {
                file: String::new(),
            });
        };
        Ok(snapshot_root_from_cached_file(&cached_path, relative_path))
    }
}

/// Vocabulary file name at the root of a native model repository, the file name of
/// the deployment's vocabulary in the local layout
#[cfg(feature = "cuda")]
fn native_hub_vocab(deployment: &ParakeetDeployment) -> &'static str {
    let vocab = deployment.layout().vocab();
    vocab.rsplit_once('/').map_or(vocab, |(_, name)| name)
}

// metadata.json is left out: Core ML never reads it, and the published v3 encoder
// has none, so requiring it made every v3 download fail with a 404
fn mlmodelc_files(prefix: &str) -> Vec<String> {
    vec![
        format!("{prefix}/model.mil"),
        format!("{prefix}/coremldata.bin"),
        format!("{prefix}/weights/weight.bin"),
        format!("{prefix}/analytics/coremldata.bin"),
    ]
}

fn repo_files(deployment: &ParakeetDeployment) -> Vec<String> {
    let layout = deployment.layout();
    let mut files = vec![layout.vocab().to_owned()];
    files.extend(mlmodelc_files(layout.encoder()));
    files.extend(mlmodelc_files(layout.decoder()));
    files.extend(mlmodelc_files(layout.joint_decision()));
    files
}

/// `SCRIPTRS_MODELS_DIR`, which replaces any download with a local bundle
fn models_dir_override() -> Option<String> {
    std::env::var(SCRIPTRS_MODELS_DIR_ENV).ok()
}

fn snapshot_root_from_cached_file(cached_path: &Path, relative_path: &str) -> PathBuf {
    let mut root = cached_path.to_path_buf();
    for _ in Path::new(relative_path).components() {
        root.pop();
    }
    root
}

#[cfg(test)]
mod tests {
    use super::{
        HF_REPO, HF_REVISION, HuggingFaceModelSource, ModelBundle, ModelManager, repo_files,
        snapshot_root_from_cached_file,
    };
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

    #[test]
    fn runtime_source_uses_the_published_immutable_revision() {
        // a file already in the cache under the pinned commit resolves without a
        // request, so this checks the pin and the cache layout offline
        let cache_dir = tempfile::tempdir().expect("temp dir should be created");
        let file = "parakeet-v3/parakeet_vocab.json";
        let snapshot = cache_dir
            .path()
            .join("models--avencera--scriptrs-models/snapshots")
            .join(HF_REVISION.0);
        let cached = snapshot.join(file);
        std::fs::create_dir_all(cached.parent().expect("file should have a parent"))
            .expect("snapshot dir should be created");
        std::fs::write(&cached, "{}").expect("cached file should be written");

        let manager = ModelManager::with_cache_dir(cache_dir.path().to_path_buf())
            .expect("Hugging Face client should build without credentials");
        let source = HuggingFaceModelSource::runtime();

        assert_eq!(source.repository, HF_REPO);
        assert_eq!(source.revision, HF_REVISION);
        assert_eq!(
            manager
                .repo
                .get(file)
                .expect("pinned file should resolve from the cache"),
            cached
        );
        assert_eq!(snapshot_root_from_cached_file(&cached, file), snapshot);
    }
}
