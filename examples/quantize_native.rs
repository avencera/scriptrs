//! Writes a native weight file's Conformer linear weights in a lower precision
//!
//! Reads an FP32 `model.safetensors` from `dev-tools/nemo/export_native.py` and
//! writes `model-<precision>.safetensors` beside it (or at `--out`), which a native
//! bundle loaded at that precision reads instead of the FP32 file. `<precision>` is
//! one of `fp16`, `fp8`, `int8` and `nvfp4`
//!
//! ```sh
//! cargo run --release --features low-precision --example quantize_native -- \
//!     /shared/models/scriptrs-native/parakeet-v3/model.safetensors fp8
//! ```

use std::env;
use std::path::PathBuf;

use eyre::{Result, bail, eyre};
use scriptrs::{Precision, write_native_weights};

const PRECISIONS: [Precision; 4] = [
    Precision::Fp16,
    Precision::Fp8,
    Precision::Int8,
    Precision::Nvfp4,
];

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let (Some(input), Some(name)) = (args.next(), args.next()) else {
        bail!("usage: quantize_native <model.safetensors> <precision> [--out <path>]");
    };
    let precision = PRECISIONS
        .into_iter()
        .find(|precision| precision.weights_name() == Some(name.as_str()))
        .ok_or_else(|| eyre!("unknown precision `{name}`"))?;
    let input = PathBuf::from(input);
    let output = match (args.next().as_deref(), args.next()) {
        (Some("--out"), Some(path)) => PathBuf::from(path),
        (None, _) => input.with_file_name(format!("model-{name}.safetensors")),
        _ => return Err(eyre!("expected `--out <path>`")),
    };
    if let Some(extra) = args.next() {
        bail!("unexpected argument `{extra}`");
    }

    write_native_weights(&input, &output, precision)?;
    println!("{}", output.display());
    Ok(())
}
