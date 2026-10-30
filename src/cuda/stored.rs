//! The Conformer layers' linear weights as native weight files store them, and the
//! writer of version 3 files
//!
//! A version 2 file holds every tensor in FP32 under its NeMo name. A version 3 file
//! ([`write_native_weights`]) holds the linear weights in the format its
//! `linear_weights` metadata names, each `[out, inputs]` and with `linear_q`, `linear_k`
//! and `linear_v` stacked to `self_attn.linear_qkv`, since per-tensor scales cover the
//! stacked weight; every other tensor stays FP32 and keeps its name. A quantized
//! weight `<name>.weight` holds the scheme's bytes, `<name>.weight_scale` its scales
//! (FP32, or NVFP4's tiled block scales as bytes) and, for NVFP4,
//! `<name>.weight_scale_2` the tensor scale

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;

use safetensors::Dtype;
use safetensors::tensor::TensorView;

use crate::cuda::encoder::EncoderMath;
use crate::cuda::math;
use crate::cuda::quant::{
    INPUT_MULTIPLE, INT8_GROUP, NV_BLOCK, QuantizedWeight, ScaleLayout, Scheme, f16_bits,
};
use crate::error::TranscriptionError;
use crate::models::Precision;
use crate::native::spec::{EncoderSpec, ParakeetSpec};
use crate::native::weights::{
    FORMAT_VERSION, LINEAR_FORMAT_VERSION, LINEAR_WEIGHTS_KEY, WeightFile,
};

/// The `linear_weights` value of a version 3 file with FP16 linear weights
const FP16: &str = "fp16";

/// Each Conformer layer's linear weights by name under `encoder.layers.<i>.`, and
/// the shape a version 2 file stores them in, `[out, inputs]` and a trailing 1 for the
/// pointwise convs; `self_attn.linear_qkv` stands for the stacked `linear_q`,
/// `linear_k` and `linear_v`
fn linears(spec: EncoderSpec) -> [(&'static str, Vec<usize>); 9] {
    let (d, hidden) = (spec.d_model, spec.ff_hidden);
    [
        ("feed_forward1.linear1", vec![hidden, d]),
        ("feed_forward1.linear2", vec![d, hidden]),
        ("self_attn.linear_qkv", vec![3 * d, d]),
        ("self_attn.linear_pos", vec![d, d]),
        ("self_attn.linear_out", vec![d, d]),
        ("conv.pointwise_conv1", vec![2 * d, d, 1]),
        ("conv.pointwise_conv2", vec![d, d, 1]),
        ("feed_forward2.linear1", vec![hidden, d]),
        ("feed_forward2.linear2", vec![d, hidden]),
    ]
}

/// The quantization scheme of `precision`, `None` for the float precisions
pub(crate) fn scheme(precision: Precision) -> Option<Scheme> {
    match math(precision) {
        EncoderMath::Quantized(scheme) => Some(scheme),
        _ => None,
    }
}

/// A linear weight as its file stores it
pub(crate) enum StoredLinear<'a> {
    F32(Cow<'a, [f32]>),
    /// FP16 bits
    F16(Cow<'a, [u16]>),
    Quantized(QuantizedWeight),
}

/// The scheme a version 3 file's linear weights are quantized with, `None` for a
/// version 2 or FP16 file; fails on a format this build does not know
pub(crate) fn file_scheme(file: &WeightFile) -> Result<Option<Scheme>, TranscriptionError> {
    match file.linear_weights() {
        None | Some(FP16) => Ok(None),
        Some(name) => Scheme::from_name(name).map(Some).ok_or_else(|| {
            TranscriptionError::InvalidWeights(format!(
                "linear weights are `{name}`, which this scriptrs does not read"
            ))
        }),
    }
}

/// Reads linear weight `weight`, which a version 2 file stores in `shape` (see
/// [`linears`]) and a version 3 file as `[out, inputs]`
pub(crate) fn read_linear<'a>(
    file: &'a WeightFile,
    weight: &str,
    shape: &[usize],
) -> Result<StoredLinear<'a>, TranscriptionError> {
    let (out, inputs) = (shape[0], shape[1]);
    if file.linear_weights().is_none() {
        // version 2: NeMo's names and shapes, in FP32
        return Ok(StoredLinear::F32(
            match weight.strip_suffix("self_attn.linear_qkv.weight") {
                Some(prefix) => {
                    let mut qkv = Vec::with_capacity(out * inputs);
                    for part in ["linear_q", "linear_k", "linear_v"] {
                        qkv.extend_from_slice(file.tensor(
                            &format!("{prefix}self_attn.{part}.weight"),
                            &[out / 3, inputs],
                        )?);
                    }
                    Cow::Owned(qkv)
                }
                None => Cow::Borrowed(file.tensor(weight, shape)?),
            },
        ));
    }

    if file.dtype(weight)? == Dtype::F16 {
        return Ok(StoredLinear::F16(Cow::Borrowed(
            file.tensor_f16(weight, &[out, inputs])?,
        )));
    }
    let scheme = file_scheme(file)?.ok_or_else(|| {
        TranscriptionError::InvalidWeights(format!("tensor `{weight}` is not f16"))
    })?;
    let scale = format!("{weight}_scale");
    let floats = |len: usize| file.tensor(&scale, &[len]).map(<[f32]>::to_vec);
    let bytes = |len: usize| file.tensor_u8(weight, &[out, len]).map(<[u8]>::to_vec);
    Ok(StoredLinear::Quantized(match scheme {
        Scheme::Fp8 => QuantizedWeight::Fp8 {
            values: bytes(inputs)?,
            scale: floats(1)?[0],
        },
        Scheme::Nvfp4 => QuantizedWeight::Nvfp4 {
            values: bytes(inputs / 2)?,
            scales: file
                .tensor_u8(&scale, &[ScaleLayout::new(out, inputs, NV_BLOCK).bytes()])?
                .to_vec(),
            global: file.tensor(&format!("{weight}_scale_2"), &[1])?[0],
        },
        Scheme::Int8 => QuantizedWeight::Int8 {
            values: bytes(inputs)?,
            scales: floats(out * inputs / INT8_GROUP)?,
        },
    }))
}

/// Whether `name` is a linear weight a version 3 file stores differently: the
/// tensors [`read_linear`] reads from a version 2 file
fn is_linear(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("encoder.layers.") else {
        return false;
    };
    let Some((_, rest)) = rest.split_once('.') else {
        return false;
    };
    let linear = rest.strip_suffix(".weight").unwrap_or("");
    [
        "feed_forward1.linear1",
        "feed_forward1.linear2",
        "feed_forward2.linear1",
        "feed_forward2.linear2",
        "self_attn.linear_q",
        "self_attn.linear_k",
        "self_attn.linear_v",
        "self_attn.linear_pos",
        "self_attn.linear_out",
        "conv.pointwise_conv1",
        "conv.pointwise_conv2",
    ]
    .contains(&linear)
}

/// Bytes of one tensor of a file being written
struct Tensor {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl Tensor {
    fn f32(shape: Vec<usize>, values: &[f32]) -> Self {
        Self {
            dtype: Dtype::F32,
            shape,
            data: values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        }
    }

    fn u8(shape: Vec<usize>, data: Vec<u8>) -> Self {
        Self {
            dtype: Dtype::U8,
            shape,
            data,
        }
    }
}

/// Writes a copy of the version 2 native weight file `input` with the Conformer
/// layers' linear weights in `precision`'s format, quantized as
/// [`Precision`] quantizes them at load
///
/// Loading the written file with the same precision gives the same encoder as
/// loading `input` with it, with a smaller download and no quantization at load;
/// [`ModelBundle::from_native_dir`](crate::ModelBundle::from_native_dir) picks it
/// up when it is saved as `model-<name>.safetensors` beside `model.safetensors`,
/// with `<name>` from [`Precision::weights_name`]. [`Precision::Fp16`] stores the
/// linear weights in FP16, half of FP32's bytes, for the same transcripts.
/// [`Precision::Tf32`] and [`Precision::Fp32`] read `input` itself and fail here
pub fn write_native_weights(
    input: &Path,
    output: &Path,
    precision: Precision,
) -> Result<(), TranscriptionError> {
    let file = WeightFile::open(input)?;
    if file.metadata("format_version") != Some(FORMAT_VERSION) {
        return Err(TranscriptionError::InvalidWeights(format!(
            "{} is not an FP32 native weight file",
            input.display()
        )));
    }
    if matches!(precision, Precision::Fp32 | Precision::Tf32) {
        return Err(TranscriptionError::InvalidConfiguration(format!(
            "{precision:?} reads the FP32 weights as they are"
        )));
    }
    let scheme = scheme(precision);
    let config = file.metadata("config").ok_or_else(|| {
        TranscriptionError::InvalidWeights("metadata `config` is missing".to_owned())
    })?;
    let spec = ParakeetSpec::from_config_json(config)?.encoder;
    // whole INT8 groups and NVFP4 scale tiles, as loading at that precision requires
    if scheme.is_some()
        && (spec.d_model % INPUT_MULTIPLE != 0 || spec.ff_hidden % INPUT_MULTIPLE != 0)
    {
        return Err(TranscriptionError::InvalidConfiguration(format!(
            "{precision:?} needs layer widths that are multiples of {INPUT_MULTIPLE}"
        )));
    }

    let mut tensors: Vec<(String, Tensor)> = Vec::new();
    for (name, dtype, shape) in file.tensors() {
        if is_linear(name) {
            continue;
        }
        if dtype != Dtype::F32 {
            return Err(TranscriptionError::InvalidWeights(format!(
                "tensor `{name}` is {dtype:?}, not f32"
            )));
        }
        tensors.push((
            name.to_owned(),
            Tensor::f32(shape.to_vec(), file.tensor(name, shape)?),
        ));
    }
    for layer in 0..spec.layers {
        for (name, shape) in linears(spec) {
            let (out, inputs) = (shape[0], shape[1]);
            let weight = format!("encoder.layers.{layer}.{name}.weight");
            let StoredLinear::F32(values) = read_linear(&file, &weight, &shape)? else {
                unreachable!("a version 2 file stores FP32");
            };
            let Some(scheme) = scheme else {
                let data = values
                    .iter()
                    .flat_map(|value| f16_bits(*value).to_le_bytes())
                    .collect();
                tensors.push((
                    weight,
                    Tensor {
                        dtype: Dtype::F16,
                        shape: vec![out, inputs],
                        data,
                    },
                ));
                continue;
            };
            let quantized = scheme.weight(&values, out, inputs);
            let scale = format!("{weight}_scale");
            let (values, scales) = match quantized {
                QuantizedWeight::Fp8 { values, scale } => (
                    Tensor::u8(vec![out, inputs], values),
                    Tensor::f32(vec![1], &[scale]),
                ),
                QuantizedWeight::Int8 { values, scales } => (
                    Tensor::u8(vec![out, inputs], values),
                    Tensor::f32(vec![scales.len()], &scales),
                ),
                QuantizedWeight::Nvfp4 {
                    values,
                    scales,
                    global,
                } => {
                    tensors.push((format!("{weight}_scale_2"), Tensor::f32(vec![1], &[global])));
                    (
                        Tensor::u8(vec![out, inputs / 2], values),
                        Tensor::u8(vec![scales.len()], scales),
                    )
                }
            };
            tensors.push((weight, values));
            tensors.push((scale, scales));
        }
    }

    let mut metadata: HashMap<String, String> = file.all_metadata().clone();
    metadata.insert(
        "format_version".to_owned(),
        LINEAR_FORMAT_VERSION.to_owned(),
    );
    metadata.insert(
        LINEAR_WEIGHTS_KEY.to_owned(),
        scheme.map_or(FP16, Scheme::name).to_owned(),
    );
    let views = tensors
        .iter()
        .map(|(name, tensor)| {
            TensorView::new(tensor.dtype, tensor.shape.clone(), &tensor.data)
                .map(|view| (name.as_str(), view))
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| TranscriptionError::InvalidWeights(error.to_string()))?;
    safetensors::serialize_to_file(views, Some(metadata), output)
        .map_err(|error| TranscriptionError::InvalidWeights(error.to_string()))?;

    // safetensors writes through a temporary file, which is created 0600; weight
    // files are meant to be shared and published
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        std::fs::set_permissions(output, std::fs::Permissions::from_mode(0o644))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{StoredLinear, linears, read_linear, scheme, write_native_weights};
    use crate::cuda::quant::f16_bits;
    use crate::cuda::test_support::native_root;
    use crate::deployment::ParakeetDeployment;
    use crate::models::Precision;
    use crate::native::weights::WeightFile;

    /// Writes v3 at each precision into `SCRIPTRS_QUANT_OUT` (or a temporary
    /// directory), reopens it, and checks every linear weight against the load-time
    /// quantization of the FP32 file and every other tensor against the FP32 file;
    /// prints each file's size
    #[test]
    #[ignore = "reads exported weights from /shared/models/scriptrs-native and writes ~3 GB"]
    fn written_weights_match_load_time_quantization() {
        let input = native_root().join("parakeet-v3/model.safetensors");
        let source = WeightFile::open(&input).expect("FP32 weights");
        let spec = source
            .parakeet_spec(&ParakeetDeployment::v3())
            .expect("spec")
            .encoder;
        let dir = std::env::var_os("SCRIPTRS_QUANT_OUT")
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join("scriptrs-quant-weights"));
        std::fs::create_dir_all(&dir).expect("output directory");
        println!(
            "fp32: {} bytes",
            std::fs::metadata(&input).expect("size").len()
        );

        for precision in [
            Precision::Fp16,
            Precision::Fp8,
            Precision::Int8,
            Precision::Nvfp4,
        ] {
            let name = precision.weights_name().expect("a file name");
            let output = dir.join(format!("model-{name}.safetensors"));
            write_native_weights(&input, &output, precision).expect("write");
            println!(
                "{name}: {} bytes",
                std::fs::metadata(&output).expect("size").len()
            );
            let written = WeightFile::open(&output).expect("written weights open");
            written
                .parakeet_spec(&ParakeetDeployment::v3())
                .expect("written spec");

            for layer in 0..spec.layers {
                for (linear, shape) in linears(spec) {
                    let (out, inputs) = (shape[0], shape[1]);
                    let weight = format!("encoder.layers.{layer}.{linear}.weight");
                    let Ok(StoredLinear::F32(original)) = read_linear(&source, &weight, &shape)
                    else {
                        panic!("FP32 source");
                    };
                    let stored = read_linear(&written, &weight, &shape).expect("read");
                    match (precision, stored) {
                        (Precision::Fp16, StoredLinear::F16(bits)) => {
                            assert!(
                                original
                                    .iter()
                                    .zip(bits.iter())
                                    .all(|(value, bits)| f16_bits(*value) == *bits),
                                "{name} layer {layer} {linear}"
                            );
                        }
                        (_, StoredLinear::Quantized(weight)) => assert_eq!(
                            weight,
                            scheme(precision)
                                .expect("quantized")
                                .weight(&original, out, inputs),
                            "{name} layer {layer} {linear}"
                        ),
                        _ => panic!("{name} layer {layer} {linear}: unexpected storage"),
                    }
                }
            }
            for (tensor, _, shape) in written.tensors() {
                if source.dtype(tensor).is_ok() && !super::is_linear(tensor) {
                    assert_eq!(
                        written.tensor(tensor, shape).ok(),
                        source.tensor(tensor, shape).ok(),
                        "{name} {tensor}"
                    );
                }
            }
        }
    }
}
