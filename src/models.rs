use std::path::{Path, PathBuf};

use crate::deployment::ParakeetDeployment;
use crate::error::TranscriptionError;
use crate::vocab::Vocabulary;

#[cfg(all(feature = "online", feature = "cuda"))]
use crate::deployment::ParakeetVersion;
#[cfg(feature = "online")]
use hf_hub::{Repo, RepoType};

#[cfg(feature = "vad")]
const VAD_DIR: &str = "vad/silero-vad.mlmodelc";
/// Silero v5 weights in a native bundle, as `dev-tools/nemo/export_native.py`
/// writes them next to the Parakeet directories
#[cfg(all(feature = "cuda", feature = "vad"))]
const NATIVE_VAD_PATH: &str = "vad/silero-vad-v5.safetensors";
/// Longest single input for native bundles, 40 s at 16 kHz; the CUDA encoder has no
/// fixed input width, and full attention over 40 s still fits easily in memory
#[cfg(feature = "cuda")]
const NATIVE_HARD_SAMPLE_MAX: usize = 640_000;
#[cfg(feature = "online")]
const SCRIPTRS_MODELS_DIR_ENV: &str = "SCRIPTRS_MODELS_DIR";

/// Resolved model bundle for `scriptrs`
///
/// A bundle pairs one [`ParakeetDeployment`] with one runtime layout. Core ML
/// bundles are resolved from a directory with [`Self::from_dir`], and native
/// weight bundles for the CUDA backend with `Self::from_native_dir` (the `cuda`
/// feature). The runtime is never inferred from file extensions or the host
/// platform.
///
/// Most callers will use `TranscriptionPipeline::from_pretrained` or
/// `TranscriptionPipeline::from_dir` instead of constructing this directly.
#[derive(Debug, Clone)]
pub struct ModelBundle {
    root: PathBuf,
    deployment: ParakeetDeployment,
    runtime: RuntimeBundle,
}

/// Runtime-specific asset layout behind a [`ModelBundle`]
#[derive(Debug, Clone)]
pub(crate) enum RuntimeBundle {
    CoreMl(CoreMlLayout),
    #[cfg(feature = "cuda")]
    Native(NativeLayout),
}

/// Arithmetic precision of the CUDA backend's encoder matrix multiplies
///
/// The residual stream, the matrix multiplies' outputs and their accumulation stay
/// FP32 at every precision, and decoding always runs in FP32
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Precision {
    /// FP16 weights and operands for the Conformer layers' linear layers, with FP32
    /// accumulation; the rest of the encoder runs as [`Precision::Tf32`]. FP16 keeps
    /// TF32's 10-bit mantissa, so the weights take half the memory and the multiplies
    /// run at the faster FP16 tensor-core rate. Activations past ±65504 would overflow
    #[default]
    Fp16,
    /// TF32 tensor-core multiplies (10-bit mantissa) on Ampere and newer GPUs, with
    /// FP32 weights and operands; cuBLAS runs them as FP32 on older GPUs
    Tf32,
    /// Full FP32 multiplies, matching NeMo's FP32 decode token for token
    Fp32,
    /// FP8 E4M3 weights with one scale per weight tensor, and FP8 activations with one
    /// dynamic scale per product, through cuBLASLt with FP32 accumulation and output.
    /// Needs an Ada (sm_89) or newer GPU; loading fails on GPUs whose cuBLASLt runs
    /// no FP8 products. Needs the `low-precision` feature
    ///
    /// On an RTX 5090 with Parakeet v3, WER stays within 0.03 points of
    /// [`Precision::Fp16`] on every set measured: 1.94 % against 1.93 % on LibriSpeech
    /// test-clean, 8.22 % against 8.24 % on Earnings-21, 4.56 % against 4.57 % on
    /// FLEURS (de, es, fr, it, nl, pt) and 6.28 % against 6.25 % on MLS (es, fr, it,
    /// pt). It runs about 7 % slower than FP16 one utterance at a time and 2 % faster
    /// in batches of 16, and its weight file is 697 MB against FP16's 1300 MB
    ///
    /// cuBLASLt's algorithm for each product shape is chosen by timing when the model
    /// loads, so a few transcripts can differ between processes
    #[cfg(feature = "low-precision")]
    Fp8,
    /// INT8 weights with one scale per 128 inputs, dequantized to FP16 before each
    /// [`Precision::Fp16`] product, so it runs on any GPU FP16 does. Needs the
    /// `low-precision` feature
    ///
    /// On an RTX 5090 with Parakeet v3, WER stays within 0.04 points of FP16 on every
    /// set measured: 1.93 % against 1.93 % on LibriSpeech test-clean, 8.26 % against
    /// 8.24 % on Earnings-21, 4.57 % against 4.57 % on FLEURS and 6.28 % against
    /// 6.25 % on MLS. Dequantizing costs about 25 % of FP16's speed one utterance at a
    /// time and 7 % in batches of 16; its weight file is 715 MB
    #[cfg(feature = "low-precision")]
    Int8,
    /// NVFP4: E2M1 weights and activations in blocks of 16 with E4M3 scales and one
    /// scale per tensor, through cuBLASLt's block-scaled products. Needs a Blackwell
    /// GPU (compute capability 10.0 or newer; measured only on sm_120); loading fails
    /// on GPUs whose cuBLASLt runs no NVFP4 products. Needs the `low-precision` feature
    ///
    /// Faster in batches but lossy. On an RTX 5090 with Parakeet v3 it runs 16 %
    /// faster than [`Precision::Fp16`] in batches of 16 and 26 % faster on long-form
    /// Earnings-21 calls, but 22 % slower one utterance at a time, and WER rises by
    /// about 0.08 points on LibriSpeech test-clean (2.01 % against 1.93 %), 0.25 on
    /// FLEURS (4.82 % against 4.57 %) and 0.4 on MLS (6.64 % against 6.25 %). Its
    /// weight file is 432 MB
    ///
    /// cuBLASLt's algorithm for each product shape is chosen by timing when the model
    /// loads, so a few transcripts can differ between processes
    #[cfg(feature = "low-precision")]
    Nvfp4,
}

#[cfg(feature = "low-precision")]
impl Precision {
    /// The `<name>` of the `model-<name>.safetensors` weight file
    /// [`write_native_weights`](crate::write_native_weights) writes for this precision,
    /// which a native bundle loads instead of `model.safetensors` when it is present;
    /// `None` for [`Precision::Tf32`] and [`Precision::Fp32`], which read the FP32 file
    pub fn weights_name(self) -> Option<&'static str> {
        match self {
            Self::Tf32 | Self::Fp32 => None,
            Self::Fp16 => Some("fp16"),
            Self::Fp8 => Some("fp8"),
            Self::Int8 => Some("int8"),
            Self::Nvfp4 => Some("nvfp4"),
        }
    }
}

/// Native fp32 weights and vocabulary for one deployment, as
/// `dev-tools/nemo/export_native.py` writes them
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub(crate) struct NativeLayout {
    weights_path: PathBuf,
    vocab_path: PathBuf,
    #[cfg(feature = "vad")]
    vad_path: PathBuf,
    precision: Precision,
}

#[cfg(feature = "cuda")]
impl NativeLayout {
    fn resolve(root: &Path, deployment: &ParakeetDeployment) -> Self {
        Self {
            weights_path: root.join(deployment.native_weights_path()),
            vocab_path: root.join(deployment.layout().vocab()),
            #[cfg(feature = "vad")]
            vad_path: root.join(NATIVE_VAD_PATH),
            precision: Precision::default(),
        }
    }

    fn validate_base(&self) -> Result<(), TranscriptionError> {
        // a sibling deployment's weights are caught when the weight file's config is
        // checked against the deployment
        // with `low-precision`, a bundle holding only the precision's own weight file
        // needs no FP32 file
        let missing = [self.weights_path(), self.vocab_path.clone()]
            .into_iter()
            .find(|path| !path.exists());
        match missing {
            Some(path) => Err(TranscriptionError::MissingModelAsset { path }),
            None => Ok(()),
        }
    }

    /// The FP32 weight file, or with `low-precision` the precision's own weight file
    /// beside it when that is present (see `Precision::weights_name`)
    pub(crate) fn weights_path(&self) -> PathBuf {
        #[cfg(feature = "low-precision")]
        if let Some(own) = self.own_weights_path() {
            return own;
        }
        self.weights_path.clone()
    }

    /// `model-<name>.safetensors` beside `model.safetensors`, if it exists
    #[cfg(feature = "low-precision")]
    fn own_weights_path(&self) -> Option<PathBuf> {
        let name = self.precision.weights_name()?;
        let stem = self.weights_path.file_stem()?.to_str()?;
        let own = self
            .weights_path
            .with_file_name(format!("{stem}-{name}.safetensors"));
        own.exists().then_some(own)
    }

    #[cfg(feature = "vad")]
    pub(crate) fn vad_path(&self) -> &Path {
        &self.vad_path
    }

    pub(crate) fn precision(&self) -> Precision {
        self.precision
    }
}

/// Compiled Core ML asset paths for one deployment
#[derive(Debug, Clone)]
pub(crate) struct CoreMlLayout {
    encoder_dir: PathBuf,
    decoder_dir: PathBuf,
    joint_decision_dir: PathBuf,
    vocab_path: PathBuf,
    #[cfg(feature = "vad")]
    vad_dir: PathBuf,
}

impl CoreMlLayout {
    fn resolve(root: &Path, deployment: &ParakeetDeployment) -> Self {
        let layout = deployment.layout();
        Self {
            encoder_dir: root.join(layout.encoder()),
            decoder_dir: root.join(layout.decoder()),
            joint_decision_dir: root.join(layout.joint_decision()),
            vocab_path: root.join(layout.vocab()),
            #[cfg(feature = "vad")]
            vad_dir: root.join(VAD_DIR),
        }
    }

    fn validate_base(
        &self,
        root: &Path,
        deployment: &ParakeetDeployment,
    ) -> Result<(), TranscriptionError> {
        let required = [
            self.encoder_dir.as_path(),
            self.decoder_dir.as_path(),
            self.joint_decision_dir.as_path(),
            self.vocab_path.as_path(),
        ];
        let Some(missing) = required.iter().copied().find(|path| !path.exists()) else {
            return Ok(());
        };

        let sibling = root.join(deployment.sibling_vocab_path());
        if sibling.exists() {
            return Err(TranscriptionError::LayoutMismatch {
                expected: deployment.version(),
                found: deployment.sibling_version(),
            });
        }

        Err(TranscriptionError::MissingModelAsset {
            path: missing.to_path_buf(),
        })
    }

    #[cfg(any(test, target_os = "macos"))]
    pub(crate) fn encoder_dir(&self) -> &Path {
        &self.encoder_dir
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn decoder_dir(&self) -> &Path {
        &self.decoder_dir
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn joint_decision_dir(&self) -> &Path {
        &self.joint_decision_dir
    }

    #[cfg(feature = "vad")]
    pub(crate) fn vad_dir(&self) -> &Path {
        &self.vad_dir
    }
}

impl ModelBundle {
    /// Resolve the v2 Core ML model layout from a local directory
    ///
    /// This only resolves paths. File existence is checked later by
    /// [`Self::validate_base`].
    pub fn from_dir(models_dir: impl Into<PathBuf>) -> Self {
        Self::from_dir_with_deployment(models_dir, ParakeetDeployment::v2())
    }

    /// Resolve Core ML model paths using an explicit deployment
    pub fn from_dir_with_deployment(
        models_dir: impl Into<PathBuf>,
        deployment: ParakeetDeployment,
    ) -> Self {
        let root = models_dir.into();
        let runtime = RuntimeBundle::CoreMl(CoreMlLayout::resolve(&root, &deployment));
        Self {
            root,
            deployment,
            runtime,
        }
    }

    /// Resolve native fp32 weights for the CUDA backend from a local directory
    ///
    /// The directory holds `parakeet-v2/` or `parakeet-v3/` with `model.safetensors`
    /// and the vocabulary, as `dev-tools/nemo/export_native.py` writes them. With
    /// `vad`, the Silero VAD weights are `vad/silero-vad-v5.safetensors` in the same
    /// directory. This only resolves paths; [`Self::validate_base`] checks they
    /// exist, and loading checks the weights against the deployment.
    ///
    /// The CUDA encoder takes inputs of any length, so a native bundle raises the
    /// deployment's single-input limit to 40 s, enough for whole LibriSpeech
    /// utterances. Longer audio goes through the long-form pipeline.
    #[cfg(feature = "cuda")]
    pub fn from_native_dir(models_dir: impl Into<PathBuf>, deployment: ParakeetDeployment) -> Self {
        let root = models_dir.into();
        let deployment = deployment.with_hard_sample_max(NATIVE_HARD_SAMPLE_MAX);
        let runtime = RuntimeBundle::Native(NativeLayout::resolve(&root, &deployment));
        Self {
            root,
            deployment,
            runtime,
        }
    }

    /// Native bundle from files resolved one at a time, such as snapshots of the
    /// separate Hugging Face repositories for Parakeet and Silero VAD
    ///
    /// The root is the weights' directory. Without `vad_path`, the VAD path keeps
    /// the [`Self::from_native_dir`] layout under that root, so a long-form run
    /// reports it missing.
    #[cfg(all(feature = "online", feature = "cuda"))]
    fn from_native_files(
        weights_path: PathBuf,
        vocab_path: PathBuf,
        vad_path: Option<PathBuf>,
        deployment: ParakeetDeployment,
    ) -> Self {
        let root = weights_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_default();
        let mut bundle = Self::from_native_dir(root, deployment);
        if let RuntimeBundle::Native(layout) = &mut bundle.runtime {
            layout.weights_path = weights_path;
            layout.vocab_path = vocab_path;

            #[cfg(feature = "vad")]
            if let Some(vad_path) = vad_path {
                layout.vad_path = vad_path;
            }
        }

        #[cfg(not(feature = "vad"))]
        let _ = vad_path;
        bundle
    }

    /// Set the precision of the CUDA encoder's matrix multiplies, FP16 by default
    ///
    /// [`Precision::Tf32`] keeps the weights and operands in FP32, out of FP16's
    /// overflow range, and [`Precision::Fp32`] gives up the tensor-core speedup for
    /// output that matches NeMo's FP32 decode exactly. Core ML bundles ignore it
    #[cfg_attr(
        feature = "low-precision",
        doc = "",
        doc = "A native bundle reads the precision's own weight file when it is present \
               (see [`Precision::weights_name`])"
    )]
    #[cfg(feature = "cuda")]
    pub fn with_precision(mut self, precision: Precision) -> Self {
        if let RuntimeBundle::Native(layout) = &mut self.runtime {
            layout.precision = precision;
        }
        self
    }

    /// Deployment that owns this bundle's model semantics and limits
    pub fn deployment(&self) -> &ParakeetDeployment {
        &self.deployment
    }

    /// Validate that the deployment's Parakeet assets are present
    pub fn validate_base(&self) -> Result<(), TranscriptionError> {
        match &self.runtime {
            RuntimeBundle::CoreMl(layout) => layout.validate_base(&self.root, &self.deployment),
            #[cfg(feature = "cuda")]
            RuntimeBundle::Native(layout) => layout.validate_base(),
        }
    }

    #[cfg(feature = "vad")]
    /// Validate that the Parakeet and VAD model assets are present
    pub fn validate_long_form(&self) -> Result<(), TranscriptionError> {
        self.validate_base()?;
        match &self.runtime {
            RuntimeBundle::CoreMl(layout) if !layout.vad_dir.exists() => {
                Err(TranscriptionError::MissingModelAsset {
                    path: layout.vad_dir.clone(),
                })
            }
            RuntimeBundle::CoreMl(_) => Ok(()),
            #[cfg(feature = "cuda")]
            RuntimeBundle::Native(layout) if !layout.vad_path.exists() => {
                Err(self.missing_vad_error())
            }
            #[cfg(feature = "cuda")]
            RuntimeBundle::Native(_) => Ok(()),
        }
    }

    pub(crate) fn runtime(&self) -> &RuntimeBundle {
        &self.runtime
    }

    #[cfg(test)]
    pub(crate) fn vocab_path(&self) -> &Path {
        match &self.runtime {
            RuntimeBundle::CoreMl(layout) => &layout.vocab_path,
            #[cfg(feature = "cuda")]
            RuntimeBundle::Native(layout) => &layout.vocab_path,
        }
    }

    pub(crate) fn vocabulary(&self) -> Result<Vocabulary, TranscriptionError> {
        match &self.runtime {
            RuntimeBundle::CoreMl(layout) => {
                Vocabulary::load_for_deployment(&layout.vocab_path, &self.deployment)
            }
            #[cfg(feature = "cuda")]
            RuntimeBundle::Native(layout) => {
                Vocabulary::load_for_deployment(&layout.vocab_path, &self.deployment)
            }
        }
    }

    /// Error to return when a VAD run is requested but the bundle has no VAD model
    #[cfg(feature = "vad")]
    pub(crate) fn missing_vad_error(&self) -> TranscriptionError {
        match &self.runtime {
            RuntimeBundle::CoreMl(layout) => TranscriptionError::MissingModelAsset {
                path: layout.vad_dir.clone(),
            },
            #[cfg(feature = "cuda")]
            RuntimeBundle::Native(layout) => TranscriptionError::MissingModelAsset {
                path: layout.vad_path.clone(),
            },
        }
    }

    /// Return the root directory that contains the model bundle
    pub fn root(&self) -> &Path {
        &self.root
    }

    #[cfg(feature = "online")]
    /// Download the v2 Parakeet model bundle from Hugging Face
    ///
    /// By default this resolves models from the pinned `avencera/scriptrs-models`
    /// snapshot. Set `SCRIPTRS_MODELS_DIR` to force a local bundle.
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
    /// By default this resolves models from the pinned `avencera/scriptrs-models`
    /// snapshot. Set `SCRIPTRS_MODELS_DIR` to force a local bundle.
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

    #[cfg(all(feature = "online", feature = "cuda"))]
    /// Download native CUDA weights for a deployment from Hugging Face
    ///
    /// This resolves `model.safetensors` and the vocabulary from a pinned
    /// snapshot of the deployment's model repository,
    /// `avencera/parakeet-tdt-0.6b-v2-cuda` or `avencera/parakeet-tdt-0.6b-v3-cuda`,
    /// then behaves like [`Self::from_native_dir`]. Set `SCRIPTRS_MODELS_DIR` to
    /// force a local native bundle.
    pub fn from_pretrained_native(
        deployment: ParakeetDeployment,
    ) -> Result<Self, hf_hub::api::sync::ApiError> {
        if let Ok(models_dir) = std::env::var(SCRIPTRS_MODELS_DIR_ENV) {
            return Ok(Self::from_native_dir(models_dir, deployment));
        }
        ModelManager::new()?.ensure_native(deployment, false)
    }

    #[cfg(all(feature = "online", feature = "cuda", feature = "vad"))]
    /// Download native CUDA weights and the native Silero VAD weights
    ///
    /// The VAD weights come from a pinned snapshot of
    /// `avencera/silero-vad-v5-safetensors`. Set `SCRIPTRS_MODELS_DIR` to force a
    /// local native bundle.
    pub fn from_pretrained_native_long_form(
        deployment: ParakeetDeployment,
    ) -> Result<Self, hf_hub::api::sync::ApiError> {
        if let Ok(models_dir) = std::env::var(SCRIPTRS_MODELS_DIR_ENV) {
            return Ok(Self::from_native_dir(models_dir, deployment));
        }
        ModelManager::new()?.ensure_native(deployment, true)
    }
}

#[cfg(feature = "online")]
const HF_REPO: HuggingFaceRepository = HuggingFaceRepository("avencera/scriptrs-models");
#[cfg(feature = "online")]
const HF_REVISION: HuggingFaceRevision =
    HuggingFaceRevision("63f6f9cbce43e5561c219229896f86f20c0eb660");

// native CUDA weights live in one repository per model, named after the model, so
// each is listed under its base model on Hugging Face and found by searching for it
#[cfg(all(feature = "online", feature = "cuda"))]
const NATIVE_V2_SOURCE: HuggingFaceModelSource = HuggingFaceModelSource {
    repository: HuggingFaceRepository("avencera/parakeet-tdt-0.6b-v2-cuda"),
    revision: HuggingFaceRevision("ff259a9006fcd3fe19077ec5b76165c7703a7245"),
};
#[cfg(all(feature = "online", feature = "cuda"))]
const NATIVE_V3_SOURCE: HuggingFaceModelSource = HuggingFaceModelSource {
    repository: HuggingFaceRepository("avencera/parakeet-tdt-0.6b-v3-cuda"),
    revision: HuggingFaceRevision("01ffcdb65d1e0087f4f12bbe87f5413f9b429626"),
};
#[cfg(all(feature = "online", feature = "cuda", feature = "vad"))]
const NATIVE_VAD_SOURCE: HuggingFaceModelSource = HuggingFaceModelSource {
    repository: HuggingFaceRepository("avencera/silero-vad-v5-safetensors"),
    revision: HuggingFaceRevision("bc58b9e061a5cb07bc3ec839df301647753dcc59"),
};
/// Weight file name in each native model repository
#[cfg(all(feature = "online", feature = "cuda"))]
const NATIVE_HUB_WEIGHTS: &str = "model.safetensors";

#[cfg(feature = "online")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HuggingFaceRepository(&'static str);

#[cfg(feature = "online")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HuggingFaceRevision(&'static str);

#[cfg(feature = "online")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HuggingFaceModelSource {
    repository: HuggingFaceRepository,
    revision: HuggingFaceRevision,
}

#[cfg(feature = "online")]
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

    fn api_repo(self, api: &hf_hub::api::sync::Api) -> hf_hub::api::sync::ApiRepo {
        api.repo(Repo::with_revision(
            self.repository.0.to_owned(),
            RepoType::Model,
            self.revision.0.to_owned(),
        ))
    }
}

/// Downloads and caches model bundles from Hugging Face
///
/// This is mainly useful if you want direct control over the Hugging Face cache
/// directory or want to prefetch assets before building a pipeline.
#[cfg(feature = "online")]
pub struct ModelManager {
    api: hf_hub::api::sync::Api,
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
        Self {
            repo: HuggingFaceModelSource::runtime().api_repo(&api),
            api,
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

    /// Ensure a deployment's native CUDA weights are cached locally
    ///
    /// The weights come from the deployment's model repository, such as
    /// `avencera/parakeet-tdt-0.6b-v3-cuda`, and with `include_vad` the Silero VAD
    /// weights from `avencera/silero-vad-v5-safetensors`, each at a pinned revision.
    /// The returned [`ModelBundle`] is a native bundle rooted at the Parakeet
    /// snapshot directory.
    #[cfg(feature = "cuda")]
    pub fn ensure_native(
        &self,
        deployment: ParakeetDeployment,
        include_vad: bool,
    ) -> Result<ModelBundle, hf_hub::api::sync::ApiError> {
        let repo = HuggingFaceModelSource::native(&deployment).api_repo(&self.api);
        let weights_path = repo.get(NATIVE_HUB_WEIGHTS)?;
        let vocab_path = repo.get(native_hub_vocab(&deployment))?;

        #[cfg(feature = "vad")]
        let vad_path = match include_vad {
            true => Some(
                NATIVE_VAD_SOURCE
                    .api_repo(&self.api)
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

/// Vocabulary file name at the root of a native model repository, the file name of
/// the deployment's vocabulary in the local layout
#[cfg(all(feature = "online", feature = "cuda"))]
fn native_hub_vocab(deployment: &ParakeetDeployment) -> &'static str {
    let vocab = deployment.layout().vocab();
    vocab.rsplit_once('/').map_or(vocab, |(_, name)| name)
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
    use super::{CoreMlLayout, ModelBundle, RuntimeBundle};
    use crate::deployment::{ParakeetDeployment, ParakeetVersion};
    use crate::error::TranscriptionError;
    use std::fs;
    use std::path::PathBuf;
    use tempfile::tempdir;

    fn fixture_models() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/models")
    }

    fn coreml_layout(bundle: &ModelBundle) -> &CoreMlLayout {
        match bundle.runtime() {
            RuntimeBundle::CoreMl(layout) => layout,
            #[cfg(feature = "cuda")]
            RuntimeBundle::Native(_) => panic!("expected a Core ML bundle"),
        }
    }

    #[test]
    fn v2_bundle_resolves_existing_fixture_paths() {
        let bundle = ModelBundle::from_dir(fixture_models());
        bundle
            .validate_base()
            .expect("v2 fixture layout should validate");
        assert_eq!(bundle.deployment().version(), ParakeetVersion::V2);
        assert!(
            coreml_layout(&bundle)
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
        let source = HuggingFaceModelSource::runtime();
        let api =
            hf_hub::api::sync::ApiBuilder::from_cache(hf_hub::Cache::new(std::env::temp_dir()))
                .build()
                .expect("Hugging Face API should build without credentials");
        let manager = ModelManager::from_api(api);

        assert_eq!(source.repository, HF_REPO);
        assert_eq!(source.revision, HF_REVISION);
        assert_eq!(
            manager.repo.url("parakeet-v3/parakeet_vocab.json"),
            format!(
                "https://huggingface.co/{}/resolve/{}/parakeet-v3/parakeet_vocab.json",
                HF_REPO.0, HF_REVISION.0
            )
        );
    }
}
