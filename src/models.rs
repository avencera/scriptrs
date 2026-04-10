use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::error::TranscriptionError;

const ENCODER_DIR: &str = "parakeet-v2/encoder.mlmodelc";
const ENCODER_V2_DIR: &str = "parakeet-v2/encoder-v2.mlmodelc";
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
    #[cfg(feature = "long-form")]
    vad_dir: PathBuf,
}

#[derive(Debug, Clone)]
pub(crate) struct EncoderModelSpec {
    pub(crate) input_name: String,
    pub(crate) length_name: String,
    pub(crate) output_name: String,
    pub(crate) output_length_name: String,
    pub(crate) max_frames: usize,
}

impl ModelBundle {
    /// Resolve the expected model layout from a local directory
    ///
    /// This only resolves paths. File existence is checked later by
    /// [`Self::validate_base`] or [`Self::validate_long_form`].
    pub fn from_dir(models_dir: impl Into<PathBuf>) -> Self {
        let root = models_dir.into();
        let encoder_v2_dir = root.join(ENCODER_V2_DIR);
        Self {
            encoder_dir: if encoder_v2_dir.exists() {
                encoder_v2_dir
            } else {
                root.join(ENCODER_DIR)
            },
            decoder_dir: root.join(DECODER_DIR),
            joint_decision_dir: root.join(JOINT_DECISION_DIR),
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
            self.decoder_dir.as_path(),
            self.joint_decision_dir.as_path(),
            self.vocab_path.as_path(),
        ] {
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

    pub(crate) fn encoder_spec(&self) -> Result<EncoderModelSpec, TranscriptionError> {
        let metadata_path = self.encoder_dir.join("metadata.json");
        if !metadata_path.exists() {
            return Err(TranscriptionError::MissingModelAsset {
                path: metadata_path,
            });
        }

        let metadata = std::fs::read_to_string(&metadata_path)?;
        parse_encoder_spec(&metadata).map_err(|error| {
            TranscriptionError::CoreMl(format!(
                "failed to parse encoder metadata at {}: {error}",
                metadata_path.display()
            ))
        })
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
    ///
    /// By default this resolves models from `avencera/scriptrs-models`. Set
    /// `SCRIPTRS_MODELS_DIR` to force a local bundle or `SCRIPTRS_MODELS_REPO`
    /// to override the Hugging Face repo.
    pub fn from_pretrained() -> Result<Self, hf_hub::api::sync::ApiError> {
        if let Ok(models_dir) = std::env::var(SCRIPTRS_MODELS_DIR_ENV) {
            return Ok(Self::from_dir(models_dir));
        }
        ModelManager::new()?.ensure_base()
    }

    #[cfg(all(feature = "online", feature = "long-form"))]
    /// Download the Parakeet and VAD model bundle from Hugging Face
    ///
    /// By default this resolves models from `avencera/scriptrs-models`. Set
    /// `SCRIPTRS_MODELS_DIR` to force a local bundle or `SCRIPTRS_MODELS_REPO`
    /// to override the Hugging Face repo.
    pub fn from_pretrained_long_form() -> Result<Self, hf_hub::api::sync::ApiError> {
        if let Ok(models_dir) = std::env::var(SCRIPTRS_MODELS_DIR_ENV) {
            return Ok(Self::from_dir(models_dir));
        }
        ModelManager::new()?.ensure_long_form()
    }
}

fn parse_encoder_spec(metadata: &str) -> Result<EncoderModelSpec, String> {
    let parsed: Value = serde_json::from_str(metadata).map_err(|error| error.to_string())?;
    let schema = parsed
        .as_array()
        .and_then(|items| items.first())
        .ok_or_else(|| "encoder metadata did not contain a top-level schema array".to_owned())?;
    let inputs = schema_entry_map(schema, "inputSchema")?;
    let outputs = schema_entry_map(schema, "outputSchema")?;

    let input_name = first_present(&inputs, &["mel", "audio_signal"])?;
    let length_name = first_present(&inputs, &["mel_length", "length"])?;
    let output_name = first_present(&outputs, &["encoder", "encoder_output"])?;
    let output_length_name = first_present(&outputs, &["encoder_length", "encoder_output_length"])?;
    let input_shape = parse_shape(
        inputs
            .get(input_name)
            .and_then(|value| value.get("shape"))
            .ok_or_else(|| format!("missing shape for encoder input `{input_name}`"))?,
    )?;
    let [_, _, max_frames] = input_shape.as_slice() else {
        return Err(format!("unexpected encoder input shape: {input_shape:?}"));
    };

    Ok(EncoderModelSpec {
        input_name: input_name.to_owned(),
        length_name: length_name.to_owned(),
        output_name: output_name.to_owned(),
        output_length_name: output_length_name.to_owned(),
        max_frames: *max_frames,
    })
}

fn schema_entry_map<'a>(
    schema: &'a Value,
    field: &str,
) -> Result<std::collections::HashMap<&'a str, &'a Value>, String> {
    let entries = schema
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing `{field}` array in encoder metadata"))?;
    let mut map = std::collections::HashMap::with_capacity(entries.len());
    for entry in entries {
        let name = entry
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("entry in `{field}` missing string name"))?;
        map.insert(name, entry);
    }
    Ok(map)
}

fn first_present<'a>(
    entries: &std::collections::HashMap<&'a str, &'a Value>,
    names: &[&'a str],
) -> Result<&'a str, String> {
    names
        .iter()
        .copied()
        .find(|name| entries.contains_key(name))
        .ok_or_else(|| format!("expected one of {names:?}, found {:?}", entries.keys()))
}

fn parse_shape(shape: &Value) -> Result<Vec<usize>, String> {
    match shape {
        Value::Array(values) => values
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .map(|value| value as usize)
                    .ok_or_else(|| format!("shape value was not an integer: {value}"))
            })
            .collect(),
        Value::String(value) => {
            let parsed: Value = serde_json::from_str(value).map_err(|error| error.to_string())?;
            parse_shape(&parsed)
        }
        other => Err(format!("unsupported shape value: {other}")),
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

    /// Ensure the base Parakeet model bundle is cached locally
    ///
    /// The returned [`ModelBundle`] points at the resolved snapshot directory
    /// inside the Hugging Face cache.
    pub fn ensure_base(&self) -> Result<ModelBundle, hf_hub::api::sync::ApiError> {
        self.ensure_unified(false)
    }

    #[cfg(feature = "long-form")]
    /// Ensure the Parakeet and VAD model bundle is cached locally
    ///
    /// The returned [`ModelBundle`] points at the resolved snapshot directory
    /// inside the Hugging Face cache.
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
    use super::{
        DECODER_DIR, ENCODER_DIR, JOINT_DECISION_DIR, ModelBundle, VOCAB_FILE, base_repo_files,
        snapshot_root_from_cached_file,
    };
    use crate::error::TranscriptionError;
    use std::fs;
    use std::path::Path;
    use tempfile::tempdir;

    #[test]
    fn base_repo_files_include_required_assets() {
        let files = base_repo_files();
        assert!(files.iter().any(|file| file == VOCAB_FILE));
        assert!(
            files
                .iter()
                .any(|file| file == &format!("{ENCODER_DIR}/model.mil"))
        );
        assert!(
            files
                .iter()
                .any(|file| file == &format!("{DECODER_DIR}/model.mil"))
        );
        assert!(
            files
                .iter()
                .any(|file| file == &format!("{JOINT_DECISION_DIR}/model.mil"))
        );
        assert!(
            !files
                .iter()
                .any(|file| file.contains("decoder-joint.mlmodelc"))
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
    fn validate_base_requires_split_bundles() {
        let temp_dir = tempdir().expect("temp dir should be created");
        let root = temp_dir.path();
        fs::create_dir_all(root.join(ENCODER_DIR)).expect("encoder dir should be created");
        fs::create_dir_all(root.join(DECODER_DIR)).expect("decoder dir should be created");
        fs::write(root.join(VOCAB_FILE), []).expect("vocab file should be created");

        let bundle = ModelBundle::from_dir(root);
        let error = bundle
            .validate_base()
            .expect_err("missing joint bundle should fail");
        match error {
            TranscriptionError::MissingModelAsset { path } => {
                assert_eq!(path, root.join(JOINT_DECISION_DIR));
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn validate_base_rejects_fused_only_bundle() {
        let temp_dir = tempdir().expect("temp dir should be created");
        let root = temp_dir.path();
        fs::create_dir_all(root.join(ENCODER_DIR)).expect("encoder dir should be created");
        fs::create_dir_all(root.join("parakeet-v2/decoder-joint.mlmodelc"))
            .expect("fused decoder dir should be created");
        fs::write(root.join(VOCAB_FILE), []).expect("vocab file should be created");

        let bundle = ModelBundle::from_dir(root);
        let error = bundle
            .validate_base()
            .expect_err("fused-only bundle should fail");
        match error {
            TranscriptionError::MissingModelAsset { path } => {
                assert_eq!(path, root.join(DECODER_DIR));
            }
            other => panic!("unexpected error: {other}"),
        }
    }
}

#[cfg(test)]
mod encoder_spec_tests {
    use super::{ENCODER_DIR, ENCODER_V2_DIR, ModelBundle, parse_encoder_spec};
    use tempfile::tempdir;

    #[test]
    fn parse_encoder_spec_supports_staged_encoder_names() {
        let spec = parse_encoder_spec(
            r#"[{
                "inputSchema": [
                    {"name": "mel", "shape": "[1, 128, 1501]"},
                    {"name": "mel_length", "shape": "[1]"}
                ],
                "outputSchema": [
                    {"name": "encoder", "shape": "[1, 1024, 188]"},
                    {"name": "encoder_length", "shape": "[1]"}
                ]
            }]"#,
        )
        .expect("staged encoder metadata should parse");
        assert_eq!(spec.input_name, "mel");
        assert_eq!(spec.length_name, "mel_length");
        assert_eq!(spec.output_name, "encoder");
        assert_eq!(spec.output_length_name, "encoder_length");
        assert_eq!(spec.max_frames, 1501);
    }

    #[test]
    fn parse_encoder_spec_supports_encoder_v2_names() {
        let spec = parse_encoder_spec(
            r#"[{
                "inputSchema": [
                    {"name": "audio_signal", "shape": [1, 128, 1001]},
                    {"name": "length", "shape": [1]}
                ],
                "outputSchema": [
                    {"name": "encoder_output", "shape": "[1, 126, 1024]"},
                    {"name": "encoder_output_length", "shape": "[1]"}
                ]
            }]"#,
        )
        .expect("encoder-v2 metadata should parse");
        assert_eq!(spec.input_name, "audio_signal");
        assert_eq!(spec.length_name, "length");
        assert_eq!(spec.output_name, "encoder_output");
        assert_eq!(spec.output_length_name, "encoder_output_length");
        assert_eq!(spec.max_frames, 1001);
    }

    #[test]
    fn bundle_prefers_encoder_v2_when_present() {
        let temp_dir = tempdir().expect("temp dir should be created");
        let root = temp_dir.path();
        std::fs::create_dir_all(root.join(ENCODER_DIR)).expect("encoder dir should be created");
        std::fs::create_dir_all(root.join(ENCODER_V2_DIR))
            .expect("encoder-v2 dir should be created");

        let bundle = ModelBundle::from_dir(root);
        assert_eq!(bundle.encoder_dir(), root.join(ENCODER_V2_DIR));
    }
}
