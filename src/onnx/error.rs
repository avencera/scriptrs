use std::fmt;
use std::path::PathBuf;

use crate::deployment::{ParakeetVersion, VocabKind};
use crate::onnx::digest::Sha256Digest;
use crate::onnx::manifest::TokenizerFormat;

/// Graph slot inside an ONNX bundle
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphKind {
    /// Parakeet Conformer encoder
    Encoder,
    /// Parakeet RNNT prediction network with LSTM state
    Decoder,
    /// Parakeet single-step joint decision head
    Joint,
    /// Silero VAD recurrent window classifier
    SileroVad,
}

impl fmt::Display for GraphKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Encoder => "encoder",
            Self::Decoder => "decoder",
            Self::Joint => "joint",
            Self::SileroVad => "silero-vad",
        })
    }
}

/// Startup validation failures for a local ONNX/CUDA bundle
///
/// Every variant is a refusal to start. None of them fall back to another
/// runtime, another file, or the CPU execution provider.
#[derive(Debug, thiserror::Error)]
pub enum OnnxBundleError {
    /// The bundle root or a declared path contains a symlink or special file
    #[error("bundle path {path} is not a regular file or directory")]
    NonRegularFile {
        /// Path that was inspected
        path: PathBuf,
    },
    /// The manifest file could not be read
    #[error("failed to read bundle manifest {path}: {source}")]
    ManifestRead {
        /// Manifest path
        path: PathBuf,
        /// Underlying I/O failure
        #[source]
        source: std::io::Error,
    },
    /// The manifest was not valid JSON for the typed schema
    #[error("bundle manifest {path} is malformed: {message}")]
    ManifestParse {
        /// Manifest path
        path: PathBuf,
        /// Parser message
        message: String,
    },
    /// The manifest declares a schema this crate does not implement
    #[error("bundle manifest {path} uses schema `{found}`; this crate supports `{expected}`")]
    UnsupportedSchema {
        /// Manifest path
        path: PathBuf,
        /// Schema tag found in the manifest
        found: String,
        /// Schema tag this crate implements
        expected: &'static str,
    },
    /// The manifest was exported for another Parakeet version
    #[error("bundle manifest declares {found:?} but {expected:?} was requested")]
    DeploymentMismatch {
        /// Version requested by the caller
        expected: ParakeetVersion,
        /// Version declared by the manifest
        found: ParakeetVersion,
    },
    /// The tokenizer format does not match the deployment tokenizer contract
    #[error(
        "tokenizer format {found:?} does not match the {expected:?} tokenizer required by {version:?}"
    )]
    TokenizerFormatMismatch {
        /// Deployment version
        version: ParakeetVersion,
        /// Tokenizer parser required by the deployment
        expected: VocabKind,
        /// Format declared by the manifest
        found: TokenizerFormat,
    },
    /// The tokenizer was taken from a different checkpoint revision than the graphs
    #[error("tokenizer revision {tokenizer} does not match checkpoint revision {checkpoint}")]
    TokenizerRevisionMismatch {
        /// Revision recorded for the tokenizer file
        tokenizer: String,
        /// Revision recorded for the exported checkpoint
        checkpoint: String,
    },
    /// Two manifest entries point at the same bundle file
    #[error("bundle path `{path}` is declared more than once")]
    DuplicatePath {
        /// Relative path inside the bundle
        path: String,
    },
    /// A declared file does not exist as a regular file
    #[error("bundle file {path} is missing")]
    MissingFile {
        /// Absolute path that was expected
        path: PathBuf,
    },
    /// A declared file could not be read while hashing
    #[error("failed to read bundle file {path}: {source}")]
    FileRead {
        /// Absolute path that failed
        path: PathBuf,
        /// Underlying I/O failure
        #[source]
        source: std::io::Error,
    },
    /// A tokenizer could not be parsed using the deployment contract
    #[error("tokenizer {path} is invalid: {message}")]
    InvalidTokenizer {
        /// Tokenizer path
        path: PathBuf,
        /// Parser message
        message: String,
    },
    /// A declared file has different contents than the manifest records
    #[error("bundle file {path} has digest {found} but the manifest records {expected}")]
    DigestMismatch {
        /// Absolute path that was hashed
        path: PathBuf,
        /// Digest recorded by the manifest
        expected: Sha256Digest,
        /// Digest computed from the file
        found: Sha256Digest,
    },
    /// The bundle directory contains a file the manifest does not declare
    #[error("bundle contains undeclared file {path}")]
    UndeclaredFile {
        /// Absolute path of the extra file
        path: PathBuf,
    },
    /// The bundle directory could not be walked for closure validation
    #[error("failed to inspect bundle directory {path}: {source}")]
    DirectoryRead {
        /// Directory that failed
        path: PathBuf,
        /// Underlying I/O failure
        #[source]
        source: std::io::Error,
    },
    /// A VAD graph was required but the manifest declares none
    #[error("bundle manifest {manifest} declares no Silero VAD graph")]
    MissingVad {
        /// Manifest path
        manifest: PathBuf,
    },
    /// The CUDA execution provider cannot be used by this process
    #[error("CUDA execution provider is unavailable: {reason}")]
    CudaUnavailable {
        /// Why CUDA could not be registered
        reason: String,
    },
    /// ONNX Runtime refused to build a session for a graph
    #[error("{graph} graph {path} failed to load: {message}")]
    SessionLoad {
        /// Graph slot
        graph: GraphKind,
        /// Graph file path
        path: PathBuf,
        /// ONNX Runtime message
        message: String,
    },
    /// The graph's declared inputs or outputs differ from the split-graph contract
    #[error("{graph} graph {path} does not satisfy the {contract} contract: {message}")]
    GraphIo {
        /// Graph slot
        graph: GraphKind,
        /// Graph file path
        path: PathBuf,
        /// Contract identifier the graph was checked against
        contract: &'static str,
        /// Mismatch description
        message: String,
    },
    /// A serialized graph could not be decoded as an ONNX ModelProto
    #[error("{graph} graph {path} is malformed: {message}")]
    GraphParse {
        /// Graph slot
        graph: GraphKind,
        /// Graph file path
        path: PathBuf,
        /// Parser message
        message: String,
    },
    /// A graph has no single valid default-domain opset declaration
    #[error("{graph} graph {path} has invalid default-domain opset: {message}")]
    GraphOpset {
        /// Graph slot
        graph: GraphKind,
        /// Graph file path
        path: PathBuf,
        /// Validation message
        message: String,
    },
    /// Graph external tensor locations do not match its manifest entry
    #[error(
        "{graph} graph {path} external-data closure differs from the manifest: declared={declared:?} referenced={referenced:?}"
    )]
    GraphExternalDataMismatch {
        /// Graph slot
        graph: GraphKind,
        /// Graph file path
        path: PathBuf,
        /// Paths declared for the graph
        declared: Vec<String>,
        /// Paths referenced by all TensorProto owners in the graph
        referenced: Vec<String>,
    },
    /// An external tensor-data location or range is malformed
    #[error("{graph} graph {path} has invalid external tensor data: {message}")]
    GraphExternalData {
        /// Graph slot
        graph: GraphKind,
        /// Graph file path
        path: PathBuf,
        /// Validation message
        message: String,
    },
}
