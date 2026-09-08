use std::collections::BTreeSet;
use std::fmt;
use std::fs;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::deployment::{ParakeetDeployment, ParakeetVersion, VocabKind};
use crate::onnx::digest::Sha256Digest;
use crate::onnx::error::{GraphKind, OnnxBundleError};
use crate::onnx::graph::{inspect_graph, validate_graph_contract};

/// File name of the bundle manifest inside the bundle root
pub const MANIFEST_FILE_NAME: &str = "manifest.json";

/// Schema tag this crate implements for Parakeet ONNX bundles
pub const PARAKEET_BUNDLE_SCHEMA: &str = "avencera.scriptrs.parakeet-onnx-bundle/1";

/// Manifest prefix used to report an unsupported schema before strict parsing
#[derive(Debug, Deserialize)]
struct ManifestHeader {
    schema: String,
}

/// Versioned description of one exported Parakeet ONNX bundle
///
/// The manifest is the only file a bundle may contain that is not itself
/// declared here. Every other regular file must be listed with its digest.
/// Unknown fields are rejected so an exporter cannot ship data this crate
/// silently ignores.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OnnxBundleManifest {
    /// Schema tag, always [`PARAKEET_BUNDLE_SCHEMA`] for this crate
    pub schema: ManifestSchema,
    /// Parakeet version the graphs were exported from
    pub deployment: ManifestDeployment,
    /// Source checkpoint identity, kept separate from exported file digests
    pub source: SourceCheckpoint,
    /// Exporter identity and graph contract
    pub export: ExportIdentity,
    /// Tokenizer file shipped with the graphs
    pub tokenizer: TokenizerAsset,
    /// The three split graphs
    pub graphs: ParakeetGraphs,
    /// Optional Silero VAD graph for long-form region planning
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vad: Option<VadAsset>,
}

/// Closed set of manifest schema tags
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ManifestSchema {
    /// First Parakeet split-graph bundle schema
    #[serde(rename = "avencera.scriptrs.parakeet-onnx-bundle/1")]
    ParakeetOnnxBundleV1,
}

/// Parakeet version tag recorded by the exporter
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ManifestDeployment {
    /// `nvidia/parakeet-tdt-0.6b-v2`
    #[serde(rename = "parakeet-tdt-0.6b-v2")]
    ParakeetTdt06bV2,
    /// `nvidia/parakeet-tdt-0.6b-v3`
    #[serde(rename = "parakeet-tdt-0.6b-v3")]
    ParakeetTdt06bV3,
}

impl ManifestDeployment {
    /// Runtime version described by this tag
    pub const fn version(self) -> ParakeetVersion {
        match self {
            Self::ParakeetTdt06bV2 => ParakeetVersion::V2,
            Self::ParakeetTdt06bV3 => ParakeetVersion::V3,
        }
    }

    /// Tag for a runtime version
    pub const fn from_version(version: ParakeetVersion) -> Self {
        match version {
            ParakeetVersion::V2 => Self::ParakeetTdt06bV2,
            ParakeetVersion::V3 => Self::ParakeetTdt06bV3,
        }
    }
}

/// Immutable source checkpoint identity
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceCheckpoint {
    /// Hugging Face repository such as `nvidia/parakeet-tdt-0.6b-v3`
    pub repository: String,
    /// Git commit the checkpoint was read from
    pub revision: GitRevision,
}

/// Full 40-character lowercase git commit hash
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct GitRevision(String);

impl GitRevision {
    /// Commit hash text
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for GitRevision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Reason a git revision string was rejected
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid git revision `{0}`: expected 40 lowercase hexadecimal characters")]
pub struct GitRevisionParseError(String);

impl TryFrom<String> for GitRevision {
    type Error = GitRevisionParseError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        let canonical = value.len() == 40
            && value
                .bytes()
                .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'));
        if !canonical {
            return Err(GitRevisionParseError(value));
        }

        Ok(Self(value))
    }
}

impl From<GitRevision> for String {
    fn from(revision: GitRevision) -> Self {
        revision.0
    }
}

/// Identity of the tool that produced the graphs
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExportIdentity {
    /// Exporter program name
    pub exporter: String,
    /// Exporter program version or commit
    pub exporter_version: String,
    /// ONNX default-domain opset every graph was exported with
    pub opset: u32,
    /// Split-graph tensor contract the graphs satisfy
    pub graph_contract: ParakeetGraphContract,
}

/// Closed set of Parakeet split-graph contracts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParakeetGraphContract {
    /// Encoder, decoder, and joint graphs with the tensor names and shapes
    /// listed in the crate documentation for the ONNX bundle format
    #[serde(rename = "avencera.parakeet-split/1")]
    SplitV1,
}

impl ParakeetGraphContract {
    /// Contract identifier as written in the manifest
    pub const fn id(self) -> &'static str {
        match self {
            Self::SplitV1 => "avencera.parakeet-split/1",
        }
    }
}

/// Tokenizer file shipped with the graphs
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizerAsset {
    /// Path relative to the bundle root
    pub path: BundlePath,
    /// Digest of the tokenizer file
    pub sha256: Sha256Digest,
    /// Parser the file requires
    pub format: TokenizerFormat,
    /// Checkpoint revision the tokenizer was copied from
    pub source_revision: GitRevision,
}

/// Tokenizer file formats a bundle may declare
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenizerFormat {
    /// SentencePiece `vocab.txt` with `token id` lines and an explicit blank
    #[serde(rename = "sentencepiece-vocab-txt")]
    SentencePieceVocabTxt,
    /// NVIDIA Parakeet JSON table without a blank entry
    #[serde(rename = "parakeet-vocab-json")]
    ParakeetVocabJson,
}

impl TokenizerFormat {
    /// Parser this format maps to
    pub const fn vocab_kind(self) -> VocabKind {
        match self {
            Self::SentencePieceVocabTxt => VocabKind::SentencePieceTxt,
            Self::ParakeetVocabJson => VocabKind::JsonTokenizer,
        }
    }

    /// Format required by a deployment tokenizer contract
    pub const fn from_vocab_kind(kind: VocabKind) -> Self {
        match kind {
            VocabKind::SentencePieceTxt => Self::SentencePieceVocabTxt,
            VocabKind::JsonTokenizer => Self::ParakeetVocabJson,
        }
    }
}

/// The three Parakeet split graphs
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParakeetGraphs {
    /// Mel features to encoded frames
    pub encoder: GraphAsset,
    /// Previous token and LSTM state to projected decoder output
    pub decoder: GraphAsset,
    /// Encoder frame and decoder output to token, probability, and duration
    pub joint: GraphAsset,
}

/// One ONNX graph plus any external tensor data files it references
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphAsset {
    /// Graph path relative to the bundle root
    pub path: BundlePath,
    /// Digest of the graph file
    pub sha256: Sha256Digest,
    /// External data files the graph loads, relative to the bundle root
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub external_data: Vec<FileAsset>,
}

/// A plain data file with its digest
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileAsset {
    /// Path relative to the bundle root
    pub path: BundlePath,
    /// Digest of the file
    pub sha256: Sha256Digest,
}

/// Silero VAD graph with its own source lineage
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VadAsset {
    /// The VAD graph
    pub graph: GraphAsset,
    /// Source repository and revision the graph was exported from
    pub source: SourceCheckpoint,
    /// Tensor contract the graph satisfies
    pub graph_contract: SileroGraphContract,
}

/// Closed set of Silero VAD graph contracts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SileroGraphContract {
    /// Silero VAD v5 16 kHz graph with `input`, `state`, `sr`, `output`, `stateN`
    #[serde(rename = "avencera.silero-vad/1")]
    V1,
}

impl SileroGraphContract {
    /// Contract identifier as written in the manifest
    pub const fn id(self) -> &'static str {
        match self {
            Self::V1 => "avencera.silero-vad/1",
        }
    }
}

/// Relative, normalized, forward-slash path inside a bundle
///
/// Absolute paths, `.` and `..` components, empty components, backslashes,
/// and the manifest file name itself are rejected at parse time.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct BundlePath(String);

impl BundlePath {
    /// Path text with `/` separators
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Resolve this path under a bundle root
    pub fn resolve(&self, root: &Path) -> PathBuf {
        let mut resolved = root.to_path_buf();
        for component in self.0.split('/') {
            resolved.push(component);
        }
        resolved
    }
}

impl fmt::Display for BundlePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Reason a bundle path was rejected
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid bundle path `{path}`: {reason}")]
pub struct BundlePathError {
    path: String,
    reason: &'static str,
}

impl TryFrom<String> for BundlePath {
    type Error = BundlePathError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        let reject = |reason| BundlePathError {
            path: value.clone(),
            reason,
        };
        if value.is_empty() {
            return Err(reject("path is empty"));
        }
        if value.starts_with('/') {
            return Err(reject("path must be relative"));
        }
        if value.contains('\\') {
            return Err(reject("path must use `/` separators"));
        }
        if value == MANIFEST_FILE_NAME {
            return Err(reject("path must not be the manifest"));
        }
        for component in value.split('/') {
            match component {
                "" => return Err(reject("path contains an empty component")),
                "." | ".." => return Err(reject("path must not contain `.` or `..`")),
                _ => {}
            }
        }

        Ok(Self(value))
    }
}

impl From<BundlePath> for String {
    fn from(path: BundlePath) -> Self {
        path.0
    }
}

impl OnnxBundleManifest {
    /// Read and strictly parse the manifest inside `root`
    pub fn read(root: &Path) -> Result<Self, OnnxBundleError> {
        let path = root.join(MANIFEST_FILE_NAME);
        let metadata =
            fs::symlink_metadata(&path).map_err(|source| OnnxBundleError::ManifestRead {
                path: path.clone(),
                source,
            })?;
        if !metadata.file_type().is_file() {
            return Err(OnnxBundleError::NonRegularFile { path });
        }
        let text = fs::read_to_string(&path).map_err(|source| OnnxBundleError::ManifestRead {
            path: path.clone(),
            source,
        })?;
        Self::parse(&text, &path)
    }

    /// Strictly parse manifest text; `path` is only used for error reporting
    pub fn parse(text: &str, path: &Path) -> Result<Self, OnnxBundleError> {
        let header: ManifestHeader =
            serde_json::from_str(text).map_err(|error| OnnxBundleError::ManifestParse {
                path: path.to_path_buf(),
                message: error.to_string(),
            })?;
        if header.schema != PARAKEET_BUNDLE_SCHEMA {
            return Err(OnnxBundleError::UnsupportedSchema {
                path: path.to_path_buf(),
                found: header.schema,
                expected: PARAKEET_BUNDLE_SCHEMA,
            });
        }

        serde_json::from_str(text).map_err(|error| OnnxBundleError::ManifestParse {
            path: path.to_path_buf(),
            message: error.to_string(),
        })
    }

    /// Serialize the manifest as pretty JSON
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("manifest types serialize without failure")
    }

    /// Every file the manifest declares, in manifest order
    pub fn declared_files(&self) -> Vec<FileAsset> {
        let mut files = vec![FileAsset {
            path: self.tokenizer.path.clone(),
            sha256: self.tokenizer.sha256,
        }];
        for graph in [
            &self.graphs.encoder,
            &self.graphs.decoder,
            &self.graphs.joint,
        ] {
            push_graph_files(&mut files, graph);
        }
        if let Some(vad) = &self.vad {
            push_graph_files(&mut files, &vad.graph);
        }
        files
    }

    /// Reject a manifest whose semantics do not match the requested deployment
    pub fn validate_deployment(&self, version: ParakeetVersion) -> Result<(), OnnxBundleError> {
        let found = self.deployment.version();
        if found != version {
            return Err(OnnxBundleError::DeploymentMismatch {
                expected: version,
                found,
            });
        }

        let expected_kind = match version {
            ParakeetVersion::V2 => VocabKind::SentencePieceTxt,
            ParakeetVersion::V3 => VocabKind::JsonTokenizer,
        };
        if self.tokenizer.format.vocab_kind() != expected_kind {
            return Err(OnnxBundleError::TokenizerFormatMismatch {
                version,
                expected: expected_kind,
                found: self.tokenizer.format,
            });
        }
        if self.tokenizer.source_revision != self.source.revision {
            return Err(OnnxBundleError::TokenizerRevisionMismatch {
                tokenizer: self.tokenizer.source_revision.to_string(),
                checkpoint: self.source.revision.to_string(),
            });
        }

        Ok(())
    }

    /// Verify that the bundle directory contains exactly the declared files
    /// with the declared digests
    pub fn validate_files(&self, root: &Path) -> Result<(), OnnxBundleError> {
        let root_metadata =
            fs::symlink_metadata(root).map_err(|source| OnnxBundleError::DirectoryRead {
                path: root.to_path_buf(),
                source,
            })?;
        if !root_metadata.file_type().is_dir() {
            return Err(OnnxBundleError::NonRegularFile {
                path: root.to_path_buf(),
            });
        }
        let declared = self.declared_files();
        let mut seen = BTreeSet::new();
        for file in &declared {
            if !seen.insert(file.path.clone()) {
                return Err(OnnxBundleError::DuplicatePath {
                    path: file.path.to_string(),
                });
            }
        }

        for file in &declared {
            verify_file(root, file)?;
        }

        verify_closure(root, root, &seen)
    }

    /// Validate every graph against the canonical ORT-free bundle contract
    pub(crate) fn validate_graphs(
        &self,
        root: &Path,
        deployment: &ParakeetDeployment,
    ) -> Result<(), OnnxBundleError> {
        let graphs = [
            (GraphKind::Encoder, &self.graphs.encoder),
            (GraphKind::Decoder, &self.graphs.decoder),
            (GraphKind::Joint, &self.graphs.joint),
        ];
        for (kind, graph) in graphs {
            validate_graph(root, self, deployment, kind, graph)?;
        }
        if let Some(vad) = &self.vad {
            validate_graph(root, self, deployment, GraphKind::SileroVad, &vad.graph)?;
        }
        Ok(())
    }
}

fn validate_graph(
    root: &Path,
    manifest: &OnnxBundleManifest,
    deployment: &ParakeetDeployment,
    kind: GraphKind,
    graph: &GraphAsset,
) -> Result<(), OnnxBundleError> {
    let path = graph.path.resolve(root);
    let bytes = fs::read(&path).map_err(|source| OnnxBundleError::FileRead {
        path: path.clone(),
        source,
    })?;
    let inspection = inspect_graph(&graph.path, kind, &bytes)?;
    validate_graph_contract(
        &graph.path,
        kind,
        &inspection,
        deployment,
        manifest.export.opset,
    )?;

    let declared: BTreeSet<&str> = graph
        .external_data
        .iter()
        .map(|file| file.path.as_str())
        .collect();
    let referenced: BTreeSet<&str> = inspection
        .external_data
        .iter()
        .map(|reference| reference.path.as_str())
        .collect();
    if declared != referenced {
        return Err(OnnxBundleError::GraphExternalDataMismatch {
            graph: kind,
            path,
            declared: declared.into_iter().map(str::to_owned).collect(),
            referenced: referenced.into_iter().map(str::to_owned).collect(),
        });
    }

    for reference in &inspection.external_data {
        let data_path = reference.path.resolve(root);
        let metadata =
            fs::symlink_metadata(&data_path).map_err(|source| OnnxBundleError::FileRead {
                path: data_path.clone(),
                source,
            })?;
        let file_len = metadata.len();
        let offset = reference.offset.unwrap_or(0);
        if offset > file_len {
            return Err(OnnxBundleError::GraphExternalData {
                graph: kind,
                path: path.clone(),
                message: format!("external-data offset {offset} exceeds file length {file_len}"),
            });
        }
        if let Some(length) = reference.length
            && length > file_len - offset
        {
            return Err(OnnxBundleError::GraphExternalData {
                graph: kind,
                path: path.clone(),
                message: format!(
                    "external-data range offset={offset} length={length} exceeds file length {file_len}"
                ),
            });
        }
    }

    Ok(())
}

fn push_graph_files(files: &mut Vec<FileAsset>, graph: &GraphAsset) {
    files.push(FileAsset {
        path: graph.path.clone(),
        sha256: graph.sha256,
    });
    files.extend(graph.external_data.iter().cloned());
}

fn verify_file(root: &Path, file: &FileAsset) -> Result<(), OnnxBundleError> {
    let path = file.path.resolve(root);
    verify_regular_path(root, &file.path, false)?;

    let found = Sha256Digest::of_file(&path).map_err(|source| OnnxBundleError::FileRead {
        path: path.clone(),
        source,
    })?;
    if found != file.sha256 {
        return Err(OnnxBundleError::DigestMismatch {
            path,
            expected: file.sha256,
            found,
        });
    }

    Ok(())
}

fn verify_closure(
    root: &Path,
    directory: &Path,
    declared: &BTreeSet<BundlePath>,
) -> Result<(), OnnxBundleError> {
    let entries = fs::read_dir(directory).map_err(|source| OnnxBundleError::DirectoryRead {
        path: directory.to_path_buf(),
        source,
    })?;

    for entry in entries {
        let entry = entry.map_err(|source| OnnxBundleError::DirectoryRead {
            path: directory.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|source| OnnxBundleError::DirectoryRead {
                path: path.clone(),
                source,
            })?;

        if file_type.is_dir() {
            verify_closure(root, &path, declared)?;
            continue;
        }

        if !file_type.is_file() {
            return Err(OnnxBundleError::NonRegularFile { path });
        }

        let relative = relative_bundle_path(root, &path);
        let is_manifest = directory == root && relative.as_deref() == Some(MANIFEST_FILE_NAME);
        let is_declared = relative
            .as_deref()
            .is_some_and(|relative| declared.iter().any(|path| path.as_str() == relative));
        if !is_manifest && !is_declared {
            return Err(OnnxBundleError::UndeclaredFile { path });
        }
    }

    Ok(())
}

fn verify_regular_path(
    root: &Path,
    relative: &BundlePath,
    directory: bool,
) -> Result<(), OnnxBundleError> {
    let mut current = root.to_path_buf();
    for (index, component) in relative.as_str().split('/').enumerate() {
        current.push(component);
        let metadata = match fs::symlink_metadata(&current) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(OnnxBundleError::MissingFile { path: current });
            }
            Err(error) => {
                return Err(OnnxBundleError::FileRead {
                    path: current,
                    source: error,
                });
            }
        };
        let final_component = index + 1 == relative.as_str().split('/').count();
        if final_component {
            let is_regular = if directory {
                metadata.file_type().is_dir()
            } else {
                metadata.file_type().is_file()
            };
            if !is_regular {
                return Err(OnnxBundleError::NonRegularFile { path: current });
            }
        } else if !metadata.file_type().is_dir() {
            return Err(OnnxBundleError::NonRegularFile { path: current });
        }
    }
    Ok(())
}

fn relative_bundle_path(root: &Path, path: &Path) -> Option<String> {
    let relative = path.strip_prefix(root).ok()?;
    let mut parts = Vec::new();
    for component in relative.components() {
        let Component::Normal(part) = component else {
            return None;
        };
        parts.push(part.to_str()?.to_owned());
    }
    Some(parts.join("/"))
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{BundlePath, GitRevision, OnnxBundleManifest};
    use crate::onnx::error::OnnxBundleError;

    #[test]
    fn bundle_paths_must_be_relative_and_normalized() {
        for bad in [
            "",
            "/abs.onnx",
            "../escape.onnx",
            "./dot.onnx",
            "nested//empty.onnx",
            "back\\slash.onnx",
            "manifest.json",
        ] {
            assert!(
                BundlePath::try_from(bad.to_owned()).is_err(),
                "path `{bad}` must be rejected"
            );
        }
        let path = BundlePath::try_from("graphs/encoder.onnx".to_owned()).unwrap();
        assert_eq!(
            path.resolve(Path::new("/bundle")),
            Path::new("/bundle/graphs/encoder.onnx")
        );
    }

    #[test]
    fn git_revisions_must_be_full_lowercase_hashes() {
        assert!(GitRevision::try_from("abc".to_owned()).is_err());
        assert!(
            GitRevision::try_from("AE9AD07059C7C739FFAF932226A8FE64AE2620B0".to_owned()).is_err()
        );
        assert!(
            GitRevision::try_from("ae9ad07059c7c739ffaf932226a8fe64ae2620b0".to_owned()).is_ok()
        );
    }

    #[test]
    fn unknown_schema_is_reported_before_strict_parsing() {
        let error = OnnxBundleManifest::parse(
            r#"{"schema": "avencera.scriptrs.parakeet-onnx-bundle/2", "anything": 1}"#,
            Path::new("manifest.json"),
        )
        .expect_err("unknown schema must fail");
        assert!(matches!(
            error,
            OnnxBundleError::UnsupportedSchema { found, .. }
                if found == "avencera.scriptrs.parakeet-onnx-bundle/2"
        ));
    }

    #[test]
    fn manifest_without_schema_is_a_parse_error() {
        let error = OnnxBundleManifest::parse(r#"{"deployment": "x"}"#, Path::new("manifest.json"))
            .expect_err("missing schema must fail");
        assert!(matches!(error, OnnxBundleError::ManifestParse { .. }));
    }
}
