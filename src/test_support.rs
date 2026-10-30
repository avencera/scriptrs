//! Readers for NeMo reference tensors written by `dev-tools/nemo/dump_ref.py`
//!
//! Parity tests that use these are `#[ignore]`d because the references live
//! outside the repository; `just parity` runs them. Set `SCRIPTRS_NEMO_REF`
//! to point at a copy other than `/shared/datasets/parakeet-ref`, and
//! `SCRIPTRS_STREAM_REF` for the streaming references `dev-tools/nemo/stream_ref.py`
//! writes, other than `/shared/datasets/nemotron-stream-ref`.

use std::fs::File;
use std::path::PathBuf;

use memmap2::Mmap;
use ndarray::{ArrayD, IxDyn};
use safetensors::{Dtype, SafeTensors};

use crate::deployment::ParakeetVersion;

const DEFAULT_REFERENCE_DIR: &str = "/shared/datasets/parakeet-ref";

#[cfg(feature = "cuda")]
const DEFAULT_STREAM_REFERENCE_DIR: &str = "/shared/datasets/nemotron-stream-ref";

/// One reference set (`b1` or `b8`) for one Parakeet version
pub(crate) struct NemoReference {
    mmap: Mmap,
    /// The set's `.json` metadata, with NeMo's transcript per utterance
    #[cfg(feature = "cuda")]
    meta_path: PathBuf,
}

fn reference_root() -> PathBuf {
    std::env::var_os("SCRIPTRS_NEMO_REF")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_REFERENCE_DIR))
}

impl NemoReference {
    pub(crate) fn open(version: ParakeetVersion, set: &str) -> Self {
        let root = reference_root();
        // the profile wrote the v3 sets at the root; later versions get a subdirectory
        let dir = match version {
            ParakeetVersion::V2 => root.join("v2"),
            ParakeetVersion::V3 => root,
        };
        Self::open_in(dir, set)
    }

    /// Silero VAD v5 probabilities for one LibriSpeech file (`silero-v5/vad`)
    #[cfg(all(feature = "cuda", feature = "vad"))]
    pub(crate) fn silero() -> Self {
        Self::open_in(reference_root().join("silero-v5"), "vad")
    }

    /// The tensor dump of NeMo's streaming run of `model` (`nemotron-en` or
    /// `nemotron-3.5`) at `lookahead`
    #[cfg(feature = "cuda")]
    pub(crate) fn streaming(model: &str, lookahead: usize) -> Self {
        let root = std::env::var_os("SCRIPTRS_STREAM_REF")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_STREAM_REFERENCE_DIR));
        Self::open_in(root.join(model), &format!("r{lookahead}.dump"))
    }

    fn open_in(dir: PathBuf, set: &str) -> Self {
        let path = dir.join(format!("{set}.safetensors"));
        let file = File::open(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        // SAFETY: reference files are written once and only read while tests run
        let mmap = unsafe { Mmap::map(&file) }.expect("reference file should map");
        Self {
            mmap,
            #[cfg(feature = "cuda")]
            meta_path: dir.join(format!("{set}.json")),
        }
    }

    /// NeMo's transcript of each utterance in the set
    #[cfg(feature = "cuda")]
    pub(crate) fn nemo_texts(&self) -> Vec<String> {
        let text = std::fs::read_to_string(&self.meta_path)
            .unwrap_or_else(|error| panic!("{}: {error}", self.meta_path.display()));
        let meta: serde_json::Value = serde_json::from_str(&text).expect("reference metadata");
        meta["utterances"]
            .as_array()
            .expect("utterances")
            .iter()
            .map(|utterance| {
                utterance["nemo_text"]
                    .as_str()
                    .expect("nemo_text")
                    .to_owned()
            })
            .collect()
    }

    fn tensors(&self) -> SafeTensors<'_> {
        SafeTensors::deserialize(&self.mmap).expect("reference file should parse")
    }

    /// Whether the reference holds `key`
    #[cfg(feature = "cuda")]
    pub(crate) fn contains(&self, key: &str) -> bool {
        self.tensors().tensor(key).is_ok()
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

    #[cfg(feature = "cuda")]
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
