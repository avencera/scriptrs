//! Readers for NeMo reference tensors written by `dev-tools/nemo/dump_ref.py`
//!
//! Parity tests that use these are `#[ignore]`d because the references live
//! outside the repository; `just parity` runs them. Set `SCRIPTRS_NEMO_REF`
//! to point at a copy other than `/shared/datasets/parakeet-ref`.

use std::fs::File;
use std::path::PathBuf;

use memmap2::Mmap;
use ndarray::{ArrayD, IxDyn};
use safetensors::{Dtype, SafeTensors};

use crate::deployment::ParakeetVersion;

const DEFAULT_REFERENCE_DIR: &str = "/shared/datasets/parakeet-ref";

/// One reference set (`b1` or `b8`) for one Parakeet version
pub(crate) struct NemoReference {
    mmap: Mmap,
}

impl NemoReference {
    pub(crate) fn open(version: ParakeetVersion, set: &str) -> Self {
        let root = std::env::var_os("SCRIPTRS_NEMO_REF")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_REFERENCE_DIR));
        // the profile wrote the v3 sets at the root; later versions get a subdirectory
        let dir = match version {
            ParakeetVersion::V2 => root.join("v2"),
            ParakeetVersion::V3 => root,
        };
        let path = dir.join(format!("{set}.safetensors"));
        let file = File::open(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        // SAFETY: reference files are written once and only read while tests run
        let mmap = unsafe { Mmap::map(&file) }.expect("reference file should map");
        Self { mmap }
    }

    fn tensors(&self) -> SafeTensors<'_> {
        SafeTensors::deserialize(&self.mmap).expect("reference file should parse")
    }

    pub(crate) fn f32(&self, key: &str) -> ArrayD<f32> {
        let tensors = self.tensors();
        let view = tensors
            .tensor(key)
            .unwrap_or_else(|error| panic!("reference key `{key}`: {error}"));
        assert_eq!(view.dtype(), Dtype::F32, "`{key}` is not f32");
        let values = view
            .data()
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("4-byte chunk")))
            .collect();
        ArrayD::from_shape_vec(IxDyn(view.shape()), values).expect("shape should match data")
    }

    pub(crate) fn i32(&self, key: &str) -> Vec<i32> {
        let tensors = self.tensors();
        let view = tensors
            .tensor(key)
            .unwrap_or_else(|error| panic!("reference key `{key}`: {error}"));
        assert_eq!(view.dtype(), Dtype::I32, "`{key}` is not i32");
        view.data()
            .chunks_exact(4)
            .map(|bytes| i32::from_le_bytes(bytes.try_into().expect("4-byte chunk")))
            .collect()
    }

    pub(crate) fn i64(&self, key: &str) -> Vec<i64> {
        let tensors = self.tensors();
        let view = tensors
            .tensor(key)
            .unwrap_or_else(|error| panic!("reference key `{key}`: {error}"));
        assert_eq!(view.dtype(), Dtype::I64, "`{key}` is not i64");
        view.data()
            .chunks_exact(8)
            .map(|bytes| i64::from_le_bytes(bytes.try_into().expect("8-byte chunk")))
            .collect()
    }
}
