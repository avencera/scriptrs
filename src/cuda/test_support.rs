use std::ops::{Deref, DerefMut};
use std::path::PathBuf;

use ndarray::{ArrayD, Axis, Ix3};

use crate::cuda::arena::Arena;
use crate::cuda::encoder::{CudaEncoder, EncoderWorkspace};
use crate::cuda::kernels::CudaRuntime;
use crate::deployment::{ParakeetDeployment, ParakeetVersion};
use crate::models::{ModelBundle, Precision};
use crate::native::weights::WeightFile;

/// Directory of the exported native weights; set `SCRIPTRS_NATIVE_MODELS` to use a
/// copy other than `/shared/models/scriptrs-native`
pub(crate) fn native_root() -> PathBuf {
    std::env::var_os("SCRIPTRS_NATIVE_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/shared/models/scriptrs-native"))
}

/// Native bundle from [`native_root`] in FP32, the precision the NeMo references
/// were decoded in; the faster precisions only stay near them
pub(crate) fn nemo_parity_bundle(deployment: ParakeetDeployment) -> ModelBundle {
    native_bundle(deployment, Precision::Fp32)
}

/// Native bundle from [`native_root`] at `precision`
pub(crate) fn native_bundle(deployment: ParakeetDeployment, precision: Precision) -> ModelBundle {
    ModelBundle::from_native_dir(native_root(), deployment).with_precision(precision)
}

/// Exported native weights for `version`, from [`native_root`]
pub(crate) fn weights(version: ParakeetVersion) -> WeightFile {
    let root = native_root();
    let dir = match version {
        ParakeetVersion::V2 => "parakeet-v2",
        ParakeetVersion::V3 => "parakeet-v3",
    };
    WeightFile::open(&root.join(dir).join("model.safetensors")).expect("native weights")
}

/// The exported Nemotron Speech Streaming EN 0.6B weights, from [`native_root`]
pub(crate) fn streaming_weights() -> WeightFile {
    let path = native_root()
        .join("nemotron-speech-streaming-en-0.6b")
        .join("model.safetensors");
    WeightFile::open(&path).expect("native streaming weights")
}

/// Relative L2 error and max abs difference over the valid frames of `[batch,
/// frames, d_model]` outputs
///
/// Panics on a non-finite value in either side, which `f64::max` would skip
pub(crate) fn compare(ours: &[f32], reference: &ArrayD<f32>, lens: &[usize]) -> (f64, f64) {
    let reference = reference
        .view()
        .into_dimensionality::<Ix3>()
        .expect("[batch, frames, d]");
    assert_eq!(ours.len(), reference.len(), "output size");
    let (frames, width) = (reference.shape()[1], reference.shape()[2]);
    let (mut error, mut norm, mut max_abs) = (0f64, 0f64, 0f64);
    for (b, row) in reference.axis_iter(Axis(0)).enumerate() {
        for t in 0..lens[b] {
            for c in 0..width {
                let expected = f64::from(row[[t, c]]);
                let value = f64::from(ours[(b * frames + t) * width + c]);
                assert!(
                    value.is_finite() && expected.is_finite(),
                    "row {b} frame {t} channel {c}: {value} against {expected}"
                );
                let diff = value - expected;
                error += diff * diff;
                norm += expected * expected;
                max_abs = max_abs.max(diff.abs());
            }
        }
    }
    ((error / norm).sqrt(), max_abs)
}

/// An encoder workspace on an arena of its own, for tests that run one shape alone
pub(crate) struct LoneWorkspace {
    // the workspace drops before the arena memory it aliases
    workspace: EncoderWorkspace,
    _arena: Arena,
}

impl Deref for LoneWorkspace {
    type Target = EncoderWorkspace;

    fn deref(&self) -> &EncoderWorkspace {
        &self.workspace
    }
}

impl DerefMut for LoneWorkspace {
    fn deref_mut(&mut self) -> &mut EncoderWorkspace {
        &mut self.workspace
    }
}

/// A workspace for `batch` inputs of up to `mel_frames` frames on a fresh arena
pub(crate) fn lone_workspace(
    runtime: &CudaRuntime,
    encoder: &CudaEncoder,
    batch: usize,
    mel_frames: usize,
) -> LoneWorkspace {
    let mut arena = Arena::new();
    // SAFETY: the arena is new, so nothing aliases it yet
    unsafe { arena.reallocate(runtime.stream(), encoder.scratch_bytes(batch, mel_frames)) }
        .expect("arena");
    // SAFETY: the arena backs this workspace alone, and drops after it
    let workspace =
        unsafe { encoder.workspace(runtime, &arena, batch, mel_frames) }.expect("workspace");
    LoneWorkspace {
        workspace,
        _arena: arena,
    }
}
