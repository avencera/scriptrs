//! The Conformer layers' linear weights as native weight files store them, and the
//! writer of version 3 files
//!
//! A version 2 file holds every tensor in FP32 under its NeMo name. A version 3 file
//! ([`write_native_weights`]) holds the linear weights in the format its
//! `linear_weights` metadata names, each `[out, inputs]` and with `linear_q`, `linear_k`
//! and `linear_v` stacked to `self_attn.linear_qkv`, since per-tensor scales cover the
//! stacked weight; every other tensor stays FP32 and keeps its name. A quantized
//! weight `<name>.weight` holds the scheme's bytes, `<name>.weight_scale` its scales
//! (FP32, or tiled block scales as bytes) and, for NVFP4, `<name>.weight_scale_2` the
//! tensor scale. A linear weight a mixed recipe keeps at FP16 is stored FP16

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;

use safetensors::Dtype;
use safetensors::tensor::TensorView;

use crate::cuda::encoder::LinearKind;
use crate::cuda::quant::{MX_BLOCK, NV_BLOCK, QuantizedWeight, ScaleLayout, Scheme, f16_bits};
use crate::cuda::scheme;
use crate::error::TranscriptionError;
use crate::models::Precision;
use crate::native::spec::{EncoderSpec, ParakeetSpec};
use crate::native::weights::{
    FORMAT_VERSION, LINEAR_FORMAT_VERSION, LINEAR_WEIGHTS_KEY, WeightFile,
};

/// The `linear_weights` value of a version 3 file with FP16 linear weights
const FP16: &str = "fp16";

/// Each Conformer layer's linear weights, by name under `encoder.layers.<i>.`
pub(crate) const LINEARS: [(&str, LinearKind); 9] = [
    ("feed_forward1.linear1", LinearKind::FeedForwardUp),
    ("feed_forward1.linear2", LinearKind::FeedForwardDown),
    ("self_attn.linear_qkv", LinearKind::Qkv),
    ("self_attn.linear_pos", LinearKind::Position),
    ("self_attn.linear_out", LinearKind::AttentionOut),
    ("conv.pointwise_conv1", LinearKind::Pointwise1),
    ("conv.pointwise_conv2", LinearKind::Pointwise2),
    ("feed_forward2.linear1", LinearKind::FeedForwardUp),
    ("feed_forward2.linear2", LinearKind::FeedForwardDown),
];

/// The `[out, inputs]` shape of a linear kind
pub(crate) fn linear_shape(spec: EncoderSpec, kind: LinearKind) -> (usize, usize) {
    let (d, hidden) = (spec.d_model, spec.ff_hidden);
    match kind {
        LinearKind::FeedForwardUp => (hidden, d),
        LinearKind::FeedForwardDown => (d, hidden),
        LinearKind::Qkv => (3 * d, d),
        LinearKind::Pointwise1 => (2 * d, d),
        LinearKind::Position | LinearKind::AttentionOut | LinearKind::Pointwise2 => (d, d),
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

/// Reads linear `name` of layer `layer`, `[out, inputs]`
pub(crate) fn read_linear<'a>(
    file: &'a WeightFile,
    layer: usize,
    name: &str,
    out: usize,
    inputs: usize,
) -> Result<StoredLinear<'a>, TranscriptionError> {
    let prefix = format!("encoder.layers.{layer}");
    if file.linear_weights().is_none() {
        // version 2: NeMo's names and shapes, in FP32
        return Ok(StoredLinear::F32(match name {
            "self_attn.linear_qkv" => {
                let mut qkv = Vec::with_capacity(out * inputs);
                for part in ["linear_q", "linear_k", "linear_v"] {
                    qkv.extend_from_slice(file.tensor(
                        &format!("{prefix}.self_attn.{part}.weight"),
                        &[out / 3, inputs],
                    )?);
                }
                Cow::Owned(qkv)
            }
            _ if name.starts_with("conv.") => {
                Cow::Borrowed(file.tensor(&format!("{prefix}.{name}.weight"), &[out, inputs, 1])?)
            }
            _ => Cow::Borrowed(file.tensor(&format!("{prefix}.{name}.weight"), &[out, inputs])?),
        }));
    }

    let weight = format!("{prefix}.{name}.weight");
    if file.dtype(&weight)? == Dtype::F16 {
        return Ok(StoredLinear::F16(Cow::Borrowed(
            file.tensor_f16(&weight, &[out, inputs])?,
        )));
    }
    let scheme = file_scheme(file)?.ok_or_else(|| {
        TranscriptionError::InvalidWeights(format!("tensor `{weight}` is not f16"))
    })?;
    let scale = format!("{weight}_scale");
    let floats = |len: usize| file.tensor(&scale, &[len]).map(<[f32]>::to_vec);
    let blocks = |block: usize| {
        file.tensor_u8(&scale, &[ScaleLayout::new(out, inputs, block).bytes()])
            .map(<[u8]>::to_vec)
    };
    let bytes = |len: usize| file.tensor_u8(&weight, &[out, len]).map(<[u8]>::to_vec);
    Ok(StoredLinear::Quantized(match scheme {
        Scheme::Fp8Tensor => QuantizedWeight::Fp8 {
            values: bytes(inputs)?,
            scales: floats(1)?,
        },
        Scheme::Fp8Channel | Scheme::Fp8Token => QuantizedWeight::Fp8 {
            values: bytes(inputs)?,
            scales: floats(out)?,
        },
        Scheme::Mxfp8 => QuantizedWeight::Mxfp8 {
            values: bytes(inputs)?,
            scales: blocks(MX_BLOCK)?,
        },
        Scheme::Nvfp4 => QuantizedWeight::Nvfp4 {
            values: bytes(inputs / 2)?,
            scales: blocks(NV_BLOCK)?,
            global: file.tensor(&format!("{weight}_scale_2"), &[1])?[0],
        },
        Scheme::Int8Weights => QuantizedWeight::Int8 {
            values: bytes(inputs)?,
            scales: floats(out)?,
        },
        Scheme::Int4Weights { group } => QuantizedWeight::Int4 {
            values: bytes(inputs / 2)?,
            scales: floats(out * inputs / group)?,
            group,
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
    linear == "self_attn.linear_q"
        || linear == "self_attn.linear_k"
        || linear == "self_attn.linear_v"
        || LINEARS.iter().any(|(name, _)| *name == linear)
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
        for (name, kind) in LINEARS {
            let (out, inputs) = linear_shape(spec, kind);
            let StoredLinear::F32(values) = read_linear(&file, layer, name, out, inputs)? else {
                unreachable!("a version 2 file stores FP32");
            };
            let weight = format!("encoder.layers.{layer}.{name}.weight");
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
                QuantizedWeight::Fp8 { values, scales }
                | QuantizedWeight::Int8 { values, scales } => (
                    Tensor::u8(vec![out, inputs], values),
                    Tensor::f32(vec![scales.len()], &scales),
                ),
                QuantizedWeight::Int4 { values, scales, .. } => (
                    Tensor::u8(vec![out, inputs / 2], values),
                    Tensor::f32(vec![scales.len()], &scales),
                ),
                QuantizedWeight::Mxfp8 { values, scales } => (
                    Tensor::u8(vec![out, inputs], values),
                    Tensor::u8(vec![scales.len()], scales),
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
        scheme.map_or_else(|| FP16.to_owned(), Scheme::name),
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
        .map_err(|error| TranscriptionError::InvalidWeights(error.to_string()))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{LINEARS, StoredLinear, linear_shape, read_linear, write_native_weights};
    use crate::cuda::quant::f16_bits;
    use crate::cuda::scheme;
    use crate::cuda::test_support::native_root;
    use crate::deployment::ParakeetDeployment;
    use crate::models::Precision;
    use crate::native::weights::WeightFile;

    /// Writes v3 at each precision into `SCRIPTRS_QUANT_OUT` (or a temporary
    /// directory), reopens it, and checks every linear weight against the load-time
    /// quantization of the FP32 file and every other tensor against the FP32 file;
    /// prints each file's size
    #[test]
    #[ignore = "reads exported weights from /shared/models/scriptrs-native and writes ~10 GB"]
    fn nemo_parity_written_weights_match_load_time_quantization() {
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
            Precision::Fp8PerTensor,
            Precision::Fp8PerToken,
            Precision::Mxfp8,
            Precision::Nvfp4,
            Precision::Int8Weights,
            Precision::Int4Weights,
            Precision::Int4Weights64,
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
                for (linear, kind) in LINEARS {
                    let (out, inputs) = linear_shape(spec, kind);
                    let Ok(StoredLinear::F32(original)) =
                        read_linear(&source, layer, linear, out, inputs)
                    else {
                        panic!("FP32 source");
                    };
                    let stored = read_linear(&written, layer, linear, out, inputs).expect("read");
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
