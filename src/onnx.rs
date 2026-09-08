//! Portable Parakeet and Silero inference through ONNX Runtime on CUDA
//!
//! This module owns the canonical ONNX bundle contract and the CUDA runtime.
//! It loads a local, manifest-described bundle of split ONNX graphs and runs
//! them with the ONNX Runtime CUDA execution provider. The frontend, greedy
//! TDT decoder, language filtering, timing, long-form planner, overlap merge,
//! and VAD chunking stay in their existing owners; only tensor movement lives
//! here.
//!
//! A bundle is a directory that contains `manifest.json` plus exactly the
//! files the manifest declares. See `docs/onnx-cuda-bundle.md` in the
//! repository for the manifest schema and the exact graph tensor contract.
//!
//! Opening a bundle validates, in order: the manifest schema and deployment,
//! the tokenizer format and revision, every declared path and digest, the
//! directory closure, CUDA provider availability, and finally each graph's
//! declared inputs and outputs. Any failure refuses to start. There is no
//! fallback to Core ML, to another file, or to the CPU provider.

#[cfg(all(test, feature = "onnx-bundle"))]
mod bundle_tests;
mod contract;
mod digest;
mod error;
mod graph;
mod manifest;
#[cfg(feature = "onnx-cuda")]
mod parakeet;
#[cfg(feature = "onnx-cuda")]
mod session;
#[cfg(all(feature = "onnx-cuda", feature = "vad"))]
mod silero;
#[cfg(all(test, feature = "onnx-bundle"))]
pub(crate) mod test_support;
#[cfg(all(test, feature = "onnx-cuda"))]
mod tests;

use std::path::{Path, PathBuf};

use crate::deployment::ParakeetDeployment;
#[cfg(feature = "onnx-cuda")]
use crate::deployment::ParakeetVersion;
#[cfg(feature = "onnx-cuda")]
use crate::error::TranscriptionError;
#[cfg(feature = "onnx-cuda")]
use crate::onnx::parakeet::ParakeetGraphPaths;
#[cfg(feature = "onnx-cuda")]
use crate::onnx::session::SessionProvider;

pub use digest::{DigestParseError, Sha256Digest};
pub use error::{GraphKind, OnnxBundleError};
pub use manifest::{
    BundlePath, BundlePathError, ExportIdentity, FileAsset, GitRevision, GitRevisionParseError,
    GraphAsset, MANIFEST_FILE_NAME, ManifestDeployment, ManifestSchema, OnnxBundleManifest,
    PARAKEET_BUNDLE_SCHEMA, ParakeetGraphContract, ParakeetGraphs, SileroGraphContract,
    SourceCheckpoint, TokenizerAsset, TokenizerFormat, VadAsset,
};
#[cfg(feature = "onnx-cuda")]
pub use session::CudaDevice;

#[cfg(feature = "onnx-cuda")]
pub(crate) use parakeet::ParakeetOnnxModel;
#[cfg(all(feature = "onnx-cuda", feature = "vad"))]
pub(crate) use silero::SileroOnnxModel;

/// Evidence that one ONNX bundle passed the ORT-free canonical validator
#[derive(Debug, Clone)]
pub struct OnnxBundleValidation {
    root: PathBuf,
    manifest: OnnxBundleManifest,
    vocabulary: crate::vocab::Vocabulary,
}

impl OnnxBundleValidation {
    /// Bundle root that was validated
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Manifest that was validated
    pub fn manifest(&self) -> &OnnxBundleManifest {
        &self.manifest
    }

    /// Parsed tokenizer retained from the validated file bytes
    pub fn vocabulary(&self) -> &crate::vocab::Vocabulary {
        &self.vocabulary
    }

    #[cfg(feature = "onnx-cuda")]
    pub(crate) fn into_parts(self) -> (PathBuf, OnnxBundleManifest, crate::vocab::Vocabulary) {
        (self.root, self.manifest, self.vocabulary)
    }
}

/// Validate a local bundle without loading ONNX Runtime or CUDA
pub fn validate_onnx_bundle(
    root: impl Into<PathBuf>,
    deployment: ParakeetDeployment,
) -> Result<OnnxBundleValidation, OnnxBundleError> {
    let root = root.into();
    let manifest = OnnxBundleManifest::read(&root)?;
    manifest.validate_deployment(deployment.version())?;
    manifest.validate_files(&root)?;
    manifest.validate_graphs(&root, &deployment)?;
    let tokenizer_path = manifest.tokenizer.path.resolve(&root);
    let vocabulary = crate::vocab::Vocabulary::load_for_deployment(&tokenizer_path, &deployment)
        .map_err(|error| OnnxBundleError::InvalidTokenizer {
            path: tokenizer_path,
            message: error.to_string(),
        })?;
    Ok(OnnxBundleValidation {
        root,
        manifest,
        vocabulary,
    })
}

/// A validated local ONNX bundle with its CUDA sessions already loaded
///
/// A value of this type is proof that the bundle directory matched its
/// manifest byte for byte and that every graph satisfied the split-graph
/// contract for the requested [`ParakeetDeployment`]. Clones share the same
/// device sessions.
#[derive(Debug, Clone)]
#[cfg(feature = "onnx-cuda")]
pub struct OnnxCudaBundle {
    root: PathBuf,
    deployment: ParakeetDeployment,
    manifest: OnnxBundleManifest,
    vocabulary: crate::vocab::Vocabulary,
    parakeet: ParakeetOnnxModel,
    #[cfg(feature = "vad")]
    vad: Option<SileroOnnxModel>,
}

#[cfg(feature = "onnx-cuda")]
impl OnnxCudaBundle {
    /// Validate a bundle directory and load its graphs on one CUDA device
    ///
    /// The manifest must describe `deployment`. Every declared file must be
    /// present with the recorded digest and no undeclared regular file may
    /// exist under `root`. Session creation registers the CUDA execution
    /// provider with error-on-failure semantics, so a host without a usable
    /// CUDA runtime fails here instead of running on the CPU.
    pub fn open(
        root: impl Into<PathBuf>,
        deployment: ParakeetDeployment,
        device: CudaDevice,
    ) -> Result<Self, TranscriptionError> {
        Self::open_with_provider(root.into(), deployment, SessionProvider::Cuda(device))
            .map_err(Into::into)
    }

    #[cfg(test)]
    pub(crate) fn open_on_cpu(
        root: impl Into<PathBuf>,
        deployment: ParakeetDeployment,
    ) -> Result<Self, OnnxBundleError> {
        Self::open_with_provider(root.into(), deployment, SessionProvider::Cpu)
    }

    fn open_with_provider(
        root: PathBuf,
        deployment: ParakeetDeployment,
        provider: SessionProvider,
    ) -> Result<Self, OnnxBundleError> {
        let validation = validate_onnx_bundle(&root, deployment.clone())?;
        let (root, manifest, vocabulary) = validation.into_parts();
        provider.ensure_available()?;

        let graphs = &manifest.graphs;
        let parakeet = ParakeetOnnxModel::load(
            ParakeetGraphPaths {
                encoder: &graphs.encoder.path.resolve(&root),
                decoder: &graphs.decoder.path.resolve(&root),
                joint: &graphs.joint.path.resolve(&root),
            },
            provider,
            &deployment,
        )?;
        #[cfg(feature = "vad")]
        let vad = manifest
            .vad
            .as_ref()
            .map(|vad| SileroOnnxModel::load(&vad.graph.path.resolve(&root), provider))
            .transpose()?;

        Ok(Self {
            vocabulary,
            root,
            deployment,
            manifest,
            parakeet,
            #[cfg(feature = "vad")]
            vad,
        })
    }

    /// Bundle root directory
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Manifest the bundle was validated against
    pub fn manifest(&self) -> &OnnxBundleManifest {
        &self.manifest
    }

    /// Deployment the bundle was opened for
    pub fn deployment(&self) -> &ParakeetDeployment {
        &self.deployment
    }

    /// Parakeet version declared by the manifest
    pub fn version(&self) -> ParakeetVersion {
        self.manifest.deployment.version()
    }

    /// Whether the manifest declares a Silero VAD graph
    pub fn has_vad(&self) -> bool {
        self.manifest.vad.is_some()
    }

    #[cfg(feature = "vad")]
    pub(crate) fn manifest_path(&self) -> PathBuf {
        self.root.join(MANIFEST_FILE_NAME)
    }

    pub(crate) fn vocabulary(&self) -> &crate::vocab::Vocabulary {
        &self.vocabulary
    }

    pub(crate) fn parakeet_model(&self) -> ParakeetOnnxModel {
        self.parakeet.clone()
    }

    #[cfg(feature = "vad")]
    pub(crate) fn vad_model(&self) -> Option<SileroOnnxModel> {
        self.vad.clone()
    }
}
