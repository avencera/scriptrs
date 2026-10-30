#[cfg(feature = "online")]
mod hub;

use std::path::{Path, PathBuf};

#[cfg(feature = "low-precision")]
use crate::cuda::encoder::EncoderMath;
use crate::deployment::ParakeetDeployment;
use crate::error::TranscriptionError;
use crate::vocab::Vocabulary;

#[cfg(feature = "online")]
pub use hub::ModelManager;

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

impl RuntimeBundle {
    fn vocab_path(&self) -> &Path {
        match self {
            Self::CoreMl(layout) => &layout.vocab_path,
            #[cfg(feature = "cuda")]
            Self::Native(layout) => &layout.vocab_path,
        }
    }

    /// Where the bundle's VAD model is, whether or not it is there
    #[cfg(feature = "vad")]
    fn vad_path(&self) -> &Path {
        match self {
            Self::CoreMl(layout) => &layout.vad_dir,
            #[cfg(feature = "cuda")]
            Self::Native(layout) => &layout.vad_path,
        }
    }
}

/// Arithmetic precision of the CUDA backend's encoder matrix multiplies
///
/// The residual stream, the matrix multiplies' outputs and their accumulation stay
/// FP32 at every precision, and decoding always runs in FP32
///
/// # Which to use
///
/// Use the default, [`Precision::Fp16`]. It is the fastest one utterance at a time
/// and scores the same WER as FP32
///
/// [`Precision::Fp32`] and [`Precision::Tf32`] are for compatibility: use them only
/// to keep matching transcripts you already produced at that precision, such as
/// NeMo's FP32 output token for token. They are slower and no more accurate
///
/// `Precision::Fp8`, `Precision::Int8` and `Precision::Nvfp4` (`low-precision`
/// feature) are not recommended, and most users should leave the feature off. They
/// are not faster: all three are slower than FP16 one utterance at a time. Nor do
/// they save much GPU memory, since most of a running model's memory is not its
/// weights: on an RTX 5090 with Parakeet v3 they cut FP16's peak of about 7 GB by
/// only 0.4 to 0.7 GB. Reach for one only when one of these outweighs the speed:
///
/// - FP8 and INT8 keep FP16's WER at about half the download (around 700 MB against
///   1300 MB), for deployments where download or disk size is the constraint
/// - NVFP4 is faster than FP16 in large batches and on long-form audio on Blackwell
///   GPUs, at a measurable WER loss; it suits high-volume batch work that can accept
///   that loss
///
/// The three apply to offline transcription only: a streaming model fails to load
/// with them
///
/// Matches on `Precision` need a wildcard arm: the `low-precision` feature adds
/// variants, and any crate in a build can turn it on
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Precision {
    /// The default, and the right choice for almost every use
    ///
    /// FP16 weights and operands for the Conformer layers' linear layers, with FP32
    /// accumulation; the rest of the encoder runs as [`Precision::Tf32`]. FP16 keeps
    /// TF32's 10-bit mantissa, so the weights take half the memory and the multiplies
    /// run at the faster FP16 tensor-core rate. Activations past ±65504 would overflow
    #[default]
    Fp16,
    /// For compatibility with transcripts already produced in TF32; slower than
    /// [`Precision::Fp16`] at the same WER
    ///
    /// TF32 tensor-core multiplies (10-bit mantissa) on Ampere and newer GPUs, with
    /// FP32 weights and operands; cuBLAS runs them as FP32 on older GPUs
    Tf32,
    /// For compatibility: matches NeMo's FP32 decode token for token, so use it only
    /// when you need that exact output, such as to stay consistent with transcripts
    /// you already produced in FP32. About half as fast as [`Precision::Fp16`] at the
    /// same WER
    Fp32,
    /// For limited cases only: half the download of [`Precision::Fp16`] at the same
    /// WER, but slower one utterance at a time
    ///
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
    /// cuBLASLt's algorithm for each product shape is chosen by timing the first time
    /// that shape runs, so the first transcription of each new size waits for the
    /// timing, and a few transcripts can differ between processes
    #[cfg(feature = "low-precision")]
    Fp8,
    /// For limited cases only: half the download of [`Precision::Fp16`] at the same
    /// WER on GPUs without FP8, but slower than FP16
    ///
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
    /// For limited cases only: high-volume batch work on Blackwell GPUs that can
    /// accept a small WER loss
    ///
    /// NVFP4: E2M1 weights and activations in blocks of 16 with E4M3 scales and one
    /// scale per tensor, through cuBLASLt's block-scaled products. Needs a Blackwell
    /// GPU (compute capability 10.0 or newer; measured only on sm_120); loading fails
    /// on GPUs whose cuBLASLt runs no NVFP4 products. Needs the `low-precision` feature
    ///
    /// On an RTX 5090 with Parakeet v3 it runs 16 %
    /// faster than [`Precision::Fp16`] in batches of 16 and 26 % faster on long-form
    /// Earnings-21 calls, but 22 % slower one utterance at a time, and WER rises by
    /// about 0.08 points on LibriSpeech test-clean (2.01 % against 1.93 %), 0.25 on
    /// FLEURS (4.82 % against 4.57 %) and 0.4 on MLS (6.64 % against 6.25 %). Its
    /// weight file is 432 MB
    ///
    /// cuBLASLt's algorithm for each product shape is chosen by timing the first time
    /// that shape runs, so the first transcription of each new size waits for the
    /// timing, and a few transcripts can differ between processes
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
        // the names are the `linear_weights` values the weight files store
        match crate::cuda::math(self) {
            EncoderMath::Single(_) => None,
            EncoderMath::Fp16 => Some(crate::cuda::stored::FP16),
            EncoderMath::Quantized(scheme) => Some(scheme.name()),
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
        #[cfg(feature = "vad")]
        let vad_path = vad_path.unwrap_or_else(|| root.join(NATIVE_VAD_PATH));
        #[cfg(not(feature = "vad"))]
        let _ = vad_path;
        let runtime = RuntimeBundle::Native(NativeLayout {
            weights_path,
            vocab_path,
            #[cfg(feature = "vad")]
            vad_path,
            precision: Precision::default(),
        });
        Self {
            root,
            deployment: deployment.with_hard_sample_max(NATIVE_HARD_SAMPLE_MAX),
            runtime,
        }
    }

    /// Set the precision of the CUDA encoder's matrix multiplies, FP16 by default
    ///
    /// Most users should keep the default; see [`Precision`] for when another fits.
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

    /// Validate that the Parakeet and VAD model assets are present
    #[cfg(feature = "vad")]
    pub fn validate_long_form(&self) -> Result<(), TranscriptionError> {
        self.validate_base()?;
        if !self.runtime.vad_path().exists() {
            return Err(self.missing_vad_error());
        }
        Ok(())
    }

    pub(crate) fn runtime(&self) -> &RuntimeBundle {
        &self.runtime
    }

    pub(crate) fn vocabulary(&self) -> Result<Vocabulary, TranscriptionError> {
        Vocabulary::load_for_deployment(self.runtime.vocab_path(), &self.deployment)
    }

    /// Error to return when a VAD run is requested but the bundle has no VAD model
    #[cfg(feature = "vad")]
    pub(crate) fn missing_vad_error(&self) -> TranscriptionError {
        TranscriptionError::MissingModelAsset {
            path: self.runtime.vad_path().to_path_buf(),
        }
    }

    /// Return the root directory that contains the model bundle
    pub fn root(&self) -> &Path {
        &self.root
    }
}
#[cfg(test)]
mod tests {
    use super::{CoreMlLayout, ModelBundle, RuntimeBundle};
    use crate::deployment::{ParakeetDeployment, ParakeetVersion};
    use crate::error::TranscriptionError;
    use std::fs;
    use tempfile::{TempDir, tempdir};

    /// An empty v2 Core ML layout: what `validate_base` checks is which paths exist
    fn v2_layout() -> TempDir {
        let temp_dir = tempdir().expect("temp dir should be created");
        let layout = ParakeetDeployment::v2().layout();
        for dir in [layout.encoder(), layout.decoder(), layout.joint_decision()] {
            fs::create_dir_all(temp_dir.path().join(dir)).expect("v2 bundle dir should be created");
        }
        fs::write(temp_dir.path().join(layout.vocab()), []).expect("v2 vocab should be created");
        temp_dir
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
        let layout = v2_layout();
        let bundle = ModelBundle::from_dir(layout.path());
        bundle
            .validate_base()
            .expect("v2 fixture layout should validate");
        assert_eq!(bundle.deployment().version(), ParakeetVersion::V2);
        assert!(
            coreml_layout(&bundle)
                .encoder_dir()
                .ends_with("parakeet-v2/encoder.mlmodelc")
        );
        assert!(
            bundle
                .runtime()
                .vocab_path()
                .ends_with("parakeet-v2/vocab.txt")
        );
    }

    #[test]
    fn v3_deployment_rejects_a_v2_layout() {
        let layout = v2_layout();
        let bundle = ModelBundle::from_dir_with_deployment(layout.path(), ParakeetDeployment::v3());
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
