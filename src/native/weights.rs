use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

use crate::deployment::ParakeetDeployment;
use crate::error::TranscriptionError;
use crate::native::spec::ParakeetSpec;

const FORMAT: &str = "scriptrs-native";
/// Every tensor in FP32, as `dev-tools/nemo/export_native.py` writes it
pub(crate) const FORMAT_VERSION: &str = "2";
/// The Conformer layers' linear weights in the format the `linear_weights` metadata
/// names, as `write_native_weights` writes them; every other tensor stays FP32
#[cfg(feature = "low-precision")]
pub(crate) const LINEAR_FORMAT_VERSION: &str = "3";
/// Metadata naming the format of the linear weights in a version 3 file
#[cfg(feature = "low-precision")]
pub(crate) const LINEAR_WEIGHTS_KEY: &str = "linear_weights";

/// Memory-mapped native weight file written by `dev-tools/nemo/export_native.py`
///
/// Tensors keep their NeMo names and stay fp32. Engines read them as borrowed
/// slices and apply their own layout changes when uploading. With the
/// `low-precision` feature it may also be a version 3 file from
/// `write_native_weights`, whose linear weights are FP16 or quantized
pub(crate) struct WeightFile {
    mmap: Mmap,
    tensors: HashMap<String, TensorEntry>,
    metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
struct TensorEntry {
    dtype: Dtype,
    shape: Vec<usize>,
    /// byte range inside the mapped file
    start: usize,
    end: usize,
}

impl std::fmt::Debug for WeightFile {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WeightFile")
            .field("tensors", &self.tensors.len())
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

impl WeightFile {
    pub(crate) fn open(path: &Path) -> Result<Self, TranscriptionError> {
        let file = File::open(path).map_err(|error| match error.kind() {
            std::io::ErrorKind::NotFound => TranscriptionError::MissingModelAsset {
                path: path.to_path_buf(),
            },
            _ => TranscriptionError::Io(error),
        })?;
        // SAFETY: the mapping is read-only; a model file truncated or rewritten
        // while mapped is outside the supported use, as for any memory-mapped
        // weight loader
        let mmap = unsafe { Mmap::map(&file) }?;

        let (tensors, metadata) = {
            let (header_len, header) = SafeTensors::read_metadata(&mmap)
                .map_err(|error| invalid(format!("{}: {error}", path.display())))?;
            // tensor offsets are relative to the data section after the header
            let data_start = 8 + header_len;
            let parsed = SafeTensors::deserialize(&mmap)
                .map_err(|error| invalid(format!("{}: {error}", path.display())))?;

            let mut tensors = HashMap::new();
            for (name, view) in parsed.tensors() {
                #[cfg(not(feature = "low-precision"))]
                if view.dtype() != Dtype::F32 {
                    return Err(invalid(format!(
                        "tensor `{name}` is {:?}, not f32",
                        view.dtype()
                    )));
                }
                #[cfg(feature = "low-precision")]
                if !matches!(view.dtype(), Dtype::F32 | Dtype::F16 | Dtype::U8) {
                    return Err(invalid(format!(
                        "tensor `{name}` is {:?}, not f32, f16 or u8",
                        view.dtype()
                    )));
                }
                let info = header
                    .info(&name)
                    .ok_or_else(|| invalid(format!("tensor `{name}` has no header entry")))?;
                tensors.insert(
                    name,
                    TensorEntry {
                        dtype: view.dtype(),
                        shape: view.shape().to_vec(),
                        start: data_start + info.data_offsets.0,
                        end: data_start + info.data_offsets.1,
                    },
                );
            }
            (tensors, header.metadata().clone().unwrap_or_default())
        };

        let weights = Self {
            mmap,
            tensors,
            metadata,
        };
        weights.expect_metadata("format", FORMAT)?;
        #[cfg(not(feature = "low-precision"))]
        weights.expect_metadata("format_version", FORMAT_VERSION)?;
        #[cfg(feature = "low-precision")]
        if weights.metadata("format_version") == Some(LINEAR_FORMAT_VERSION) {
            if weights.linear_weights().is_none() {
                return Err(invalid(format!(
                    "metadata `{LINEAR_WEIGHTS_KEY}` is missing"
                )));
            }
        } else {
            weights.expect_metadata("format_version", FORMAT_VERSION)?;
        }
        // only the linear weights of a version 3 file may be other than f32, and
        // `tensor` rejects those, so a stray f16 tensor fails where it is read
        Ok(weights)
    }

    /// The format of the Conformer layers' linear weights in a version 3 file, or
    /// `None` for an all-FP32 version 2 file
    #[cfg(feature = "low-precision")]
    pub(crate) fn linear_weights(&self) -> Option<&str> {
        match self.metadata("format_version") {
            Some(LINEAR_FORMAT_VERSION) => self.metadata(LINEAR_WEIGHTS_KEY),
            _ => None,
        }
    }

    /// Every tensor's name, dtype and shape, sorted by name
    #[cfg(feature = "low-precision")]
    pub(crate) fn tensors(&self) -> Vec<(&str, Dtype, &[usize])> {
        let mut tensors: Vec<_> = self
            .tensors
            .iter()
            .map(|(name, entry)| (name.as_str(), entry.dtype, entry.shape.as_slice()))
            .collect();
        tensors.sort_by_key(|(name, _, _)| *name);
        tensors
    }

    /// Every metadata entry
    #[cfg(feature = "low-precision")]
    pub(crate) fn all_metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    /// The dtype of tensor `name`
    #[cfg(feature = "low-precision")]
    pub(crate) fn dtype(&self, name: &str) -> Result<Dtype, TranscriptionError> {
        self.entry(name).map(|entry| entry.dtype)
    }

    /// One string from the safetensors header metadata
    pub(crate) fn metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }

    pub(crate) fn expect_metadata(
        &self,
        key: &str,
        expected: &str,
    ) -> Result<(), TranscriptionError> {
        match self.metadata(key) {
            Some(found) if found == expected => Ok(()),
            Some(found) => Err(invalid(format!(
                "metadata `{key}` is `{found}`, expected `{expected}`"
            ))),
            None => Err(invalid(format!("metadata `{key}` is missing"))),
        }
    }

    /// Parse the embedded model config and check it against `deployment`
    pub(crate) fn parakeet_spec(
        &self,
        deployment: &ParakeetDeployment,
    ) -> Result<ParakeetSpec, TranscriptionError> {
        let config = self
            .metadata
            .get("config")
            .ok_or_else(|| invalid("metadata `config` is missing".to_owned()))?;
        let spec = ParakeetSpec::from_config_json(config)?;
        spec.validate_for(deployment)?;
        Ok(spec)
    }

    fn entry(&self, name: &str) -> Result<&TensorEntry, TranscriptionError> {
        self.tensors
            .get(name)
            .ok_or_else(|| invalid(format!("tensor `{name}` is missing")))
    }

    /// The bytes of tensor `name` after checking its dtype and shape
    fn data(&self, name: &str, dtype: Dtype, shape: &[usize]) -> Result<&[u8], TranscriptionError> {
        let entry = self.entry(name)?;
        if entry.dtype != dtype {
            return Err(invalid(format!(
                "tensor `{name}` is {:?}, expected {dtype:?}",
                entry.dtype
            )));
        }
        if entry.shape != shape {
            return Err(invalid(format!(
                "tensor `{name}` has shape {:?}, expected {shape:?}",
                entry.shape
            )));
        }
        if cfg!(target_endian = "big") {
            return Err(invalid("safetensors data is little-endian".to_owned()));
        }
        Ok(&self.mmap[entry.start..entry.end])
    }

    /// Borrow an f32 tensor after checking its shape
    pub(crate) fn tensor(&self, name: &str, shape: &[usize]) -> Result<&[f32], TranscriptionError> {
        let bytes = self.data(name, Dtype::F32, shape)?;
        // SAFETY: any bit pattern is a valid f32; `align_to` only yields the
        // aligned middle, and the empty-prefix check rejects a misaligned start
        let (prefix, values, suffix) = unsafe { bytes.align_to::<f32>() };
        if !prefix.is_empty() || !suffix.is_empty() {
            return Err(invalid(format!("tensor `{name}` is not 4-byte aligned")));
        }
        Ok(values)
    }

    /// Borrow an f16 tensor as its bits after checking its shape
    #[cfg(feature = "low-precision")]
    pub(crate) fn tensor_f16(
        &self,
        name: &str,
        shape: &[usize],
    ) -> Result<&[u16], TranscriptionError> {
        let bytes = self.data(name, Dtype::F16, shape)?;
        // SAFETY: any bit pattern is a valid u16; as in `tensor`
        let (prefix, values, suffix) = unsafe { bytes.align_to::<u16>() };
        if !prefix.is_empty() || !suffix.is_empty() {
            return Err(invalid(format!("tensor `{name}` is not 2-byte aligned")));
        }
        Ok(values)
    }

    /// Borrow a u8 tensor after checking its shape
    #[cfg(feature = "low-precision")]
    pub(crate) fn tensor_u8(
        &self,
        name: &str,
        shape: &[usize],
    ) -> Result<&[u8], TranscriptionError> {
        self.data(name, Dtype::U8, shape)
    }
}

fn invalid(message: String) -> TranscriptionError {
    TranscriptionError::InvalidWeights(message)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::WeightFile;
    use crate::deployment::ParakeetDeployment;
    use crate::error::TranscriptionError;

    fn native_models() -> PathBuf {
        std::env::var_os("SCRIPTRS_NATIVE_MODELS")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/shared/models/scriptrs-native"))
    }

    #[test]
    #[ignore = "reads exported weights from /shared/models/scriptrs-native"]
    fn nemo_parity_exported_weights_load_for_their_deployment_only() {
        for (dir, deployment, other) in [
            (
                "parakeet-v2",
                ParakeetDeployment::v2(),
                ParakeetDeployment::v3(),
            ),
            (
                "parakeet-v3",
                ParakeetDeployment::v3(),
                ParakeetDeployment::v2(),
            ),
        ] {
            let weights = WeightFile::open(&native_models().join(dir).join("model.safetensors"))
                .expect("exported weights should open");
            let spec = weights
                .parakeet_spec(&deployment)
                .expect("spec should match");
            assert_eq!(spec.encoder.layers, 24);
            assert_eq!(spec.joint.outputs, deployment.blank_id() + 1 + 5);

            let joint = weights
                .tensor(
                    "joint.joint_net.2.weight",
                    &[spec.joint.outputs, spec.joint.hidden],
                )
                .expect("joint output weight should load");
            assert!(joint.iter().all(|value| value.is_finite()));
            assert!(joint.iter().any(|value| *value != 0.0));

            assert!(matches!(
                weights.parakeet_spec(&other),
                Err(TranscriptionError::LayoutMismatch { .. })
            ));
            assert!(matches!(
                weights.tensor("joint.joint_net.2.weight", &[1, 1]),
                Err(TranscriptionError::InvalidWeights(_))
            ));
        }
    }
}
