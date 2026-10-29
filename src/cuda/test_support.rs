use std::path::PathBuf;

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
