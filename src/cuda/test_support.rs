use std::path::PathBuf;

use crate::deployment::ParakeetVersion;
use crate::native::weights::WeightFile;

/// Exported native weights; set `SCRIPTRS_NATIVE_MODELS` to use a copy other than
/// `/shared/models/scriptrs-native`
pub(crate) fn weights(version: ParakeetVersion) -> WeightFile {
    let root = std::env::var_os("SCRIPTRS_NATIVE_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/shared/models/scriptrs-native"));
    let dir = match version {
        ParakeetVersion::V2 => "parakeet-v2",
        ParakeetVersion::V3 => "parakeet-v3",
    };
    WeightFile::open(&root.join(dir).join("model.safetensors")).expect("native weights")
}
