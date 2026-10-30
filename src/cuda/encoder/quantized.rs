//! The encoder's quantized linear layers, under [`EncoderMath::Quantized`]
//!
//! FP8 and NVFP4 quantize each product's FP16 activation on the GPU just before a
//! cuBLASLt product; INT8 dequantizes the weight to FP16 before the FP16 product. The
//! weights are quantized on the host at load, or read as a quantized weight file
//! stores them

use std::borrow::Cow;
use std::sync::Mutex;

use cudakit::cudarc::driver::{CudaSlice, CudaView, DevicePtr, DevicePtrMut};
use cudakit::{DeviceTensor, Error, Sgemm, check_len, element_count};

use super::{EncoderMath, Linear, mixed_precision};
use crate::cuda::fp16::linear_fp16;
use crate::cuda::kernels::{CudaRuntime, EncoderKernels};
use crate::cuda::lt::{Alpha, LtFormat, LtGemm, LtProduct};
use crate::cuda::quant::{
    AMAX_PARTIALS, INPUT_MULTIPLE, NV_BLOCK, QuantKernels, QuantizedWeight, ScaleLayout, Scheme,
    f16_value,
};
use crate::cuda::stored::{StoredLinear, file_scheme, read_linear};
use crate::cuda::zeros;
use crate::error::TranscriptionError;
use crate::native::spec::EncoderSpec;
use crate::native::weights::WeightFile;

/// A quantized `[out, inputs]` weight on the device
pub(super) struct QuantLinear {
    out: usize,
    inputs: usize,
    values: CudaSlice<u8>,
    scales: DeviceScales,
}

/// The scales of a [`QuantLinear`], as its [`QuantizedWeight`] holds them
enum DeviceScales {
    /// The FP8 scale, or the INT8 group scales
    Float(CudaSlice<f32>),
    /// NVFP4 block scales and the tensor scale
    Nvfp4 { blocks: CudaSlice<u8>, global: f32 },
}

/// What [`EncoderMath::Quantized`] adds to the encoder
pub(super) struct QuantState {
    kernels: QuantKernels,
    backend: QuantBackend,
}

/// How a scheme's products run
enum QuantBackend {
    /// FP8 and NVFP4: each product quantizes its activation for a cuBLASLt product
    Lt { gemm: LtGemm, format: LtFormat },
    /// INT8: each product dequantizes its weight to FP16 here, sized for the
    /// largest, for an FP16 product
    // one buffer serves every product: each queues its dequantize and its GEMM on
    // the runtime's one stream, so the next dequantize, from any thread or graph
    // replay, runs only after the last GEMM read the buffer. The lock only lets the
    // encoder be shared between threads
    Dequant(Mutex<CudaSlice<u16>>),
}

/// One quantized activation at a time: each product quantizes its input here just
/// before it runs, on the same stream
pub(super) struct QuantInputs {
    /// E4M3 bits, or E2M1 pairs
    values: CudaSlice<u8>,
    /// The FP8 activation scale
    scale: CudaSlice<f32>,
    /// Tiled NVFP4 block scales
    blocks: CudaSlice<u8>,
    /// [`AMAX_PARTIALS`] partial maxima of the activation
    partials: CudaSlice<f32>,
    /// The device alpha of an NVFP4 product
    alpha: CudaSlice<f32>,
}

impl QuantInputs {
    pub(super) fn bytes(&self) -> usize {
        self.values.num_bytes()
            + self.scale.num_bytes()
            + self.blocks.num_bytes()
            + self.partials.num_bytes()
            + self.alpha.num_bytes()
    }
}

/// The cuBLASLt format of a scheme's products, `None` for INT8's FP16 products
fn format(scheme: Scheme) -> Option<LtFormat> {
    match scheme {
        Scheme::Fp8 => Some(LtFormat::Fp8),
        Scheme::Nvfp4 => Some(LtFormat::Nvfp4),
        Scheme::Int8 => None,
    }
}

/// Uploads linear weight `name` of the shape `shape` a version 2 file stores it in, as
/// the file stores it, for a GEMM at `math`
///
/// FP32 weights go through [`Linear::new`] or are quantized on the host; FP16 weights
/// upload as they are for an FP16 GEMM; a quantized weight uploads as it is, and
/// [`QuantState::load`] has checked that `math` uses its scheme
pub(super) fn load_linear(
    runtime: &CudaRuntime,
    kernels: &EncoderKernels,
    file: &WeightFile,
    name: &str,
    shape: &[usize],
    math: EncoderMath,
) -> Result<Linear, TranscriptionError> {
    let (out, inputs) = (shape[0], shape[1]);
    let stream = runtime.stream();
    let values = match read_linear(file, name, shape)? {
        StoredLinear::Quantized(weight) => {
            let EncoderMath::Quantized(_) = math else {
                return Err(mixed_precision());
            };
            return QuantLinear::upload(runtime, &weight, out, inputs).map(Linear::Quantized);
        }
        StoredLinear::F16(bits) if math == EncoderMath::Fp16 => {
            return Ok(Linear::Fp16(stream.clone_htod(bits.as_ref())?));
        }
        StoredLinear::F16(bits) => Cow::Owned(bits.iter().map(|bits| f16_value(*bits)).collect()),
        StoredLinear::F32(values) => values,
    };
    match math {
        EncoderMath::Quantized(scheme) => {
            let weight = scheme.weight(&values, out, inputs);
            QuantLinear::upload(runtime, &weight, out, inputs).map(Linear::Quantized)
        }
        math => Linear::new(
            runtime,
            kernels,
            DeviceTensor::upload(stream, &values, shape)?,
            math,
        ),
    }
}

impl QuantLinear {
    fn upload(
        runtime: &CudaRuntime,
        weight: &QuantizedWeight,
        out: usize,
        inputs: usize,
    ) -> Result<Self, TranscriptionError> {
        let stream = runtime.stream();
        let (values, scales) = match weight {
            QuantizedWeight::Fp8 { values, scale } => {
                (values, DeviceScales::Float(stream.clone_htod(&[*scale])?))
            }
            QuantizedWeight::Int8 { values, scales } => {
                (values, DeviceScales::Float(stream.clone_htod(scales)?))
            }
            QuantizedWeight::Nvfp4 {
                values,
                scales,
                global,
            } => (
                values,
                DeviceScales::Nvfp4 {
                    blocks: stream.clone_htod(scales)?,
                    global: *global,
                },
            ),
        };
        Ok(Self {
            out,
            inputs,
            values: stream.clone_htod(values)?,
            scales,
        })
    }
}

impl QuantState {
    /// The quantized products of `math`, or `None` for an unquantized math; fails if
    /// `file`'s linear weights are quantized in another scheme, or if this GPU or
    /// cuBLASLt does not run the scheme's products
    pub(super) fn load(
        runtime: &CudaRuntime,
        file: &WeightFile,
        spec: EncoderSpec,
        math: EncoderMath,
    ) -> Result<Option<Self>, TranscriptionError> {
        let wanted = match math {
            EncoderMath::Quantized(scheme) => Some(scheme),
            _ => None,
        };
        if let Some(stored) = file_scheme(file)?
            && wanted != Some(stored)
        {
            return Err(TranscriptionError::InvalidConfiguration(format!(
                "the weights are quantized as `{}`, which only the matching precision loads",
                stored.name()
            )));
        }
        let Some(scheme) = wanted else {
            return Ok(None);
        };
        // whole INT8 groups and NVFP4 scale tiles along every linear layer's inputs
        if !spec.d_model.is_multiple_of(INPUT_MULTIPLE)
            || !spec.ff_hidden.is_multiple_of(INPUT_MULTIPLE)
        {
            return Err(TranscriptionError::InvalidConfiguration(format!(
                "`{}` needs layer widths that are multiples of {INPUT_MULTIPLE}",
                scheme.name()
            )));
        }

        let backend = match format(scheme) {
            Some(format) => {
                let gemm = LtGemm::new(runtime)?;
                if !gemm.supports(runtime, format) {
                    return Err(TranscriptionError::InvalidConfiguration(format!(
                        "`{}` needs cuBLASLt {format:?} products, which this GPU or cuBLAS \
                         does not run",
                        scheme.name()
                    )));
                }
                QuantBackend::Lt { gemm, format }
            }
            None => {
                let largest = (spec.ff_hidden * spec.d_model).max(3 * spec.d_model * spec.d_model);
                QuantBackend::Dequant(Mutex::new(zeros(runtime.stream(), largest)?))
            }
        };
        Ok(Some(Self {
            kernels: QuantKernels::load(runtime)?,
            backend,
        }))
    }

    /// The quantized-activation buffers for a workspace of `rows` encoder rows and a
    /// position table of `positions` rows, or none for INT8
    pub(super) fn inputs(
        &self,
        runtime: &CudaRuntime,
        spec: EncoderSpec,
        rows: usize,
        positions: usize,
    ) -> Result<Option<QuantInputs>, TranscriptionError> {
        let QuantBackend::Lt { format, .. } = self.backend else {
            return Ok(None);
        };
        let stream = runtime.stream();
        let (d, hidden) = (spec.d_model, spec.ff_hidden);
        // every product's activation: the normed or merged rows, the SiLU output and
        // the relative position table
        let operands = [(rows, d), (rows, hidden), (positions, d)];
        let largest = |size: &dyn Fn(usize, usize) -> usize| {
            operands
                .iter()
                .map(|&(rows, width)| size(rows, width))
                .max()
                .unwrap_or(0)
        };
        // FP8 reads no block scales, but keeps a one-byte buffer so every workspace
        // has the same fields
        let blocks = match format {
            LtFormat::Nvfp4 => {
                largest(&|rows, width| ScaleLayout::new(rows, width, NV_BLOCK).bytes())
            }
            LtFormat::Fp8 => 1,
        };
        Ok(Some(QuantInputs {
            values: zeros(stream, largest(&|rows, width| rows * width))?,
            scale: zeros(stream, 1)?,
            blocks: zeros(stream, blocks)?,
            partials: zeros(stream, AMAX_PARTIALS)?,
            alpha: zeros(stream, 1)?,
        }))
    }

    /// Runs `spec` on the FP16 activation `a` and the quantized weight `w`, quantizing
    /// `a` into `inputs` first for FP8 and NVFP4
    pub(super) fn linear<C: DevicePtrMut<f32>>(
        &self,
        runtime: &CudaRuntime,
        spec: Sgemm,
        a: &CudaView<'_, u16>,
        w: &QuantLinear,
        c: &mut C,
        inputs: Option<&mut QuantInputs>,
    ) -> Result<(), TranscriptionError> {
        if (spec.n, spec.k) != (w.out, w.inputs) {
            return Err(TranscriptionError::InvalidModelOutput(format!(
                "a {} x {} product reads a {} x {} weight",
                spec.n, spec.k, w.out, w.inputs
            )));
        }
        let stream = runtime.stream();
        let (lt, format) = match &self.backend {
            QuantBackend::Dequant(dequant) => {
                let mut weight = dequant.lock().map_err(|_| {
                    TranscriptionError::InvalidModelOutput(
                        "the dequantize lock was poisoned".to_owned(),
                    )
                })?;
                let mut weight = weight.slice_mut(..w.out * w.inputs);
                let DeviceScales::Float(scales) = &w.scales else {
                    return Err(mixed_precision());
                };
                self.kernels.dequant_i8(
                    stream,
                    &w.values.slice(..),
                    &scales.slice(..),
                    &mut weight,
                )?;
                linear_fp16(runtime, spec, a, &weight, c)?;
                return Ok(());
            }
            QuantBackend::Lt { gemm, format } => (gemm, *format),
        };
        // a workspace of this encoder holds the quantized activations
        let Some(quant) = inputs else {
            return Err(mixed_precision());
        };
        // the activation is quantized for the weight's scales and the product reads
        // it in the backend's format, so the two must agree for the checks below
        if matches!(
            (&w.scales, format),
            (DeviceScales::Float(_), LtFormat::Nvfp4) | (DeviceScales::Nvfp4 { .. }, LtFormat::Fp8)
        ) {
            return Err(mixed_precision());
        }
        check_lt_buffers(
            spec,
            format,
            LtBuffers {
                a: a.len(),
                c: c.len(),
                values: quant.values.len(),
                blocks: quant.blocks.len(),
            },
        )?;
        let (rows, width) = (spec.m, spec.k);
        let n = rows * width;
        let kernels = &self.kernels;
        kernels.amax(stream, a, &mut quant.partials.slice_mut(..))?;
        // keep the device alpha's guard alive until the matmul is queued
        let (alpha, _alpha_sync) = match &w.scales {
            DeviceScales::Float(_) => {
                kernels.fp8(
                    stream,
                    a,
                    &quant.partials.slice(..),
                    &mut quant.values.slice_mut(..n),
                    &mut quant.scale.slice_mut(..),
                )?;
                (Alpha::Host(spec.alpha), None)
            }
            DeviceScales::Nvfp4 { global, .. } => {
                let layout = ScaleLayout::new(rows, width, NV_BLOCK);
                kernels.nvfp4(
                    stream,
                    a,
                    &quant.partials.slice(..),
                    &mut quant.values.slice_mut(..n / 2),
                    &mut quant.blocks.slice_mut(..layout.bytes()),
                    &mut quant.alpha.slice_mut(..),
                    width,
                    spec.alpha * global,
                )?;
                let (alpha, sync) = quant.alpha.device_ptr(stream);
                (Alpha::Device(alpha), Some(sync))
            }
        };

        let (w_ptr, _w_sync) = w.values.device_ptr(stream);
        let (w_scale, _w_scale_sync) = match &w.scales {
            DeviceScales::Float(scales) => scales.device_ptr(stream),
            DeviceScales::Nvfp4 { blocks, .. } => blocks.device_ptr(stream),
        };
        let (a_ptr, _a_sync) = quant.values.device_ptr(stream);
        let (a_scale, _a_scale_sync) = match format {
            LtFormat::Fp8 => quant.scale.device_ptr(stream),
            LtFormat::Nvfp4 => quant.blocks.device_ptr(stream),
        };
        let (c_ptr, _c_sync) = c.device_ptr_mut(stream);
        // SAFETY: the activation and its scales were checked, sized and written above
        // for `rows x width`, the weight holds `out x inputs` in the scheme's format
        // with matching scales, and `c` was checked to hold `rows x out`; all are
        // ordered on the runtime's stream
        unsafe {
            lt.run(
                runtime,
                LtProduct {
                    format,
                    rows,
                    out: w.out,
                    inputs: w.inputs,
                    w: w_ptr,
                    w_scale,
                    a: a_ptr,
                    a_scale,
                    c: c_ptr,
                    alpha,
                    beta: spec.beta,
                },
            )
        }?;
        Ok(())
    }
}

/// Element counts of the buffers a cuBLASLt product reads and writes
#[derive(Debug, Clone, Copy)]
struct LtBuffers {
    /// The FP16 activation
    a: usize,
    /// The FP32 output
    c: usize,
    /// The quantized-activation workspace, sized for the largest product
    values: usize,
    /// The NVFP4 block-scale workspace, sized for the largest product
    blocks: usize,
}

/// Checks the buffers of `spec` in `format` before the activation is quantized
///
/// cuBLASLt reads and writes them through raw pointers with no length checks, and
/// the workspaces are only sliced to the product's size, which would panic
fn check_lt_buffers(spec: Sgemm, format: LtFormat, buffers: LtBuffers) -> Result<(), Error> {
    let activation = element_count("quantized linear a", &[spec.m, spec.k])?;
    check_len("quantized linear a", activation, buffers.a)?;
    check_len(
        "quantized linear c",
        element_count("quantized linear c", &[spec.m, spec.n])?,
        buffers.c,
    )?;
    let (values, blocks) = match format {
        LtFormat::Fp8 => (activation, 0),
        // two E2M1 values per byte
        LtFormat::Nvfp4 => (
            activation / 2,
            ScaleLayout::new(spec.m, spec.k, NV_BLOCK).bytes(),
        ),
    };
    for (context, needed, actual) in [
        ("quantized activation", values, buffers.values),
        ("quantized activation scales", blocks, buffers.blocks),
    ] {
        if actual < needed {
            return Err(Error::BufferLength {
                context,
                expected: needed,
                actual,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use cudakit::{Library, Math, Options};
    use ndarray::{ArrayD, Ix3};

    use super::{DeviceScales, LtBuffers, QuantBackend, QuantLinear, QuantState, check_lt_buffers};
    use crate::cuda::encoder::tests::time_graph;
    use crate::cuda::encoder::{CudaEncoder, EncoderMath, Linear};
    use crate::cuda::kernels::CudaRuntime;
    use crate::cuda::linear;
    use crate::cuda::lt::LtFormat;
    use crate::cuda::quant::test_support::e4m3_value;
    use crate::cuda::quant::{
        NV_BLOCK, QuantizedWeight, ScaleLayout, Scheme, f16_bits, f16_value, fp8_tensor, nvfp4,
    };
    use crate::cuda::test_support::weights;
    use crate::deployment::{ParakeetDeployment, ParakeetVersion};
    use crate::error::TranscriptionError;
    use crate::test_support::{NemoReference, deployment};

    impl CudaEncoder {
        /// Device bytes of the Conformer layers' linear weights and their scales
        fn linear_bytes(&self) -> usize {
            self.layers
                .iter()
                .flat_map(|layer| {
                    [
                        &layer.ff1.linear1,
                        &layer.ff1.linear2,
                        &layer.attention.qkv,
                        &layer.attention.pos,
                        &layer.attention.out,
                        &layer.conv.pointwise1,
                        &layer.conv.pointwise2,
                        &layer.ff2.linear1,
                        &layer.ff2.linear2,
                    ]
                })
                .map(Linear::bytes)
                .sum()
        }
    }

    impl Linear {
        /// Device bytes of the weight and any scales
        fn bytes(&self) -> usize {
            match self {
                Self::Single(weight) => weight.data().num_bytes(),
                Self::Fp16(bits) => bits.num_bytes(),
                Self::Quantized(weight) => weight.bytes(),
            }
        }
    }

    impl QuantLinear {
        /// Device bytes of the weight and its scales
        fn bytes(&self) -> usize {
            self.values.num_bytes()
                + match &self.scales {
                    DeviceScales::Float(scales) => scales.num_bytes(),
                    DeviceScales::Nvfp4 { blocks, .. } => blocks.num_bytes(),
                }
        }
    }

    /// Every quantization scheme, as the quantized precisions name them
    const SCHEMES: [(&str, Scheme); 3] = [
        ("FP8", Scheme::Fp8),
        ("INT8", Scheme::Int8),
        ("NVFP4", Scheme::Nvfp4),
    ];

    /// Whether `error` is only this GPU or cuBLAS lacking a scheme's products, the
    /// one load failure the tests skip instead of failing
    fn unsupported_here(error: &TranscriptionError) -> bool {
        matches!(error, TranscriptionError::InvalidConfiguration(message) if message.contains("does not run"))
    }

    /// One NeMo reference input: the time-major mel, its valid lengths and the
    /// expected encoder output, `[batch, frames, d_model]`
    struct NemoInput {
        batch: usize,
        mel_frames: usize,
        lens: Vec<usize>,
        time_major: Vec<f32>,
        expected: ArrayD<f32>,
    }

    impl NemoInput {
        fn open(version: ParakeetVersion, set: &str) -> Self {
            let reference = NemoReference::open(version, set);
            let mel = reference
                .f32("mel")
                .into_dimensionality::<Ix3>()
                .expect("[batch, features, frames]");
            let (batch, mel_frames) = (mel.shape()[0], mel.shape()[2]);
            let lens = reference
                .i64("mel_len")
                .iter()
                .map(|len| *len as usize)
                .collect();
            let time_major = mel.permuted_axes([0, 2, 1]).iter().copied().collect();
            let expected = reference.f32("encoder.out").permuted_axes(vec![0, 2, 1]);
            Self {
                batch,
                mel_frames,
                lens,
                time_major,
                expected: expected.as_standard_layout().into_owned(),
            }
        }

        /// The encoder output and its valid frames per row
        fn encode(&self, runtime: &CudaRuntime, encoder: &CudaEncoder) -> (Vec<f32>, Vec<usize>) {
            let mut ws = encoder
                .workspace(runtime, self.batch, self.mel_frames)
                .expect("workspace");
            encoder
                .set_input(runtime, &mut ws, &self.lens, |mel| {
                    mel.copy_from_slice(&self.time_major)
                })
                .expect("input");
            encoder.enqueue(runtime, &mut ws).expect("encode");
            let output = runtime.stream().clone_dtoh(ws.output()).expect("download");
            (output, ws.output_lens().to_vec())
        }
    }

    /// Relative L2 distance of `ours` from `theirs` over the valid frames of `[batch,
    /// frames, width]` outputs
    fn distance(ours: &[f32], theirs: &[f32], frames: usize, width: usize, lens: &[usize]) -> f64 {
        let (mut error, mut norm) = (0f64, 0f64);
        for (b, len) in lens.iter().enumerate() {
            for t in 0..*len {
                for c in 0..width {
                    let at = (b * frames + t) * width + c;
                    let (a, b) = (f64::from(ours[at]), f64::from(theirs[at]));
                    assert!(a.is_finite(), "non-finite output {a} at {at}");
                    error += (a - b) * (a - b);
                    norm += b * b;
                }
            }
        }
        (error / norm).sqrt()
    }

    /// The host quantization of an activation in `scheme`, as the product reads it:
    /// its bytes and the values they stand for
    fn quantized_activation(scheme: Scheme, values: &[f32], width: usize) -> (Vec<u8>, Vec<f32>) {
        let rows = values.len() / width;
        match scheme {
            Scheme::Fp8 => {
                let (bits, scale) = fp8_tensor(values);
                let restored = bits.iter().map(|bits| e4m3_value(*bits) * scale).collect();
                (bits, restored)
            }
            Scheme::Nvfp4 => {
                let (values, scales, global) = nvfp4(values, width);
                let restored = QuantizedWeight::Nvfp4 {
                    values: values.clone(),
                    scales,
                    global,
                }
                .dequantize(rows, width);
                (values, restored)
            }
            Scheme::Int8 => (Vec::new(), values.to_vec()),
        }
    }

    /// A buffer short of the product's size is an error before the quantizer
    /// slices the workspaces or cuBLASLt reads past them
    #[test]
    fn quantized_products_reject_short_buffers() {
        let (rows, out, inputs) = (130, 32, 64);
        let spec = linear(rows, out, inputs, 1.0, 0.0, Math::Tf32);
        let blocks = ScaleLayout::new(rows, inputs, NV_BLOCK).bytes();
        for (format, values, blocks) in [
            (LtFormat::Fp8, rows * inputs, 1),
            (LtFormat::Nvfp4, rows * inputs / 2, blocks),
        ] {
            let sized = LtBuffers {
                a: rows * inputs,
                c: rows * out,
                values,
                blocks,
            };
            check_lt_buffers(spec, format, sized).expect("buffers sized for the product");

            let mut short = vec![
                LtBuffers {
                    a: sized.a - 1,
                    ..sized
                },
                LtBuffers {
                    c: sized.c - 1,
                    ..sized
                },
                LtBuffers {
                    values: sized.values - 1,
                    ..sized
                },
            ];
            if format == LtFormat::Nvfp4 {
                short.push(LtBuffers {
                    blocks: sized.blocks - 1,
                    ..sized
                });
            }
            for buffers in short {
                assert!(
                    check_lt_buffers(spec, format, buffers).is_err(),
                    "{format:?} accepted {buffers:?}"
                );
            }
        }
    }

    /// Runs each scheme's product on real v3 weights and random activations at odd
    /// row counts, and holds it against an FP64 product of the same quantized values
    /// computed on the host. The GPU's quantized activation must match the host
    /// quantizer bit for bit, so a misplaced scale, nibble or alpha shows up here
    /// rather than as fluent but wrong transcripts
    #[test]
    #[ignore = "needs a CUDA GPU and the native weights; `just gpu-parity` runs it on gpu0"]
    fn gpu_quantized_products_match_the_host() {
        let runtime = CudaRuntime::open(0, Options::default()).expect("CUDA runtime");
        runtime.prepare(Library::Cublas).expect("cuBLAS");
        let file = weights(ParakeetVersion::V3);
        let spec = file
            .parakeet_spec(&ParakeetDeployment::v3())
            .expect("spec")
            .encoder;
        let d = spec.d_model;
        let products = [
            (
                "encoder.layers.0.feed_forward1.linear2.weight",
                d,
                spec.ff_hidden,
                0.5,
                1.0,
            ),
            (
                "encoder.layers.5.feed_forward1.linear1.weight",
                spec.ff_hidden,
                d,
                1.0,
                0.0,
            ),
            (
                "encoder.layers.11.self_attn.linear_out.weight",
                d,
                d,
                1.0,
                1.0,
            ),
        ];
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut random = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };

        for (name, scheme) in SCHEMES {
            let math = EncoderMath::Quantized(scheme);
            let quant = match QuantState::load(&runtime, &file, spec, math) {
                Ok(quant) => quant.expect("a quantized math"),
                Err(error) if unsupported_here(&error) => {
                    println!("{name}: unsupported here: {error}");
                    continue;
                }
                Err(error) => panic!("{name}: {error}"),
            };
            for (tensor, out, inputs, alpha, beta) in products {
                let weight = file.tensor(tensor, &[out, inputs]).expect("weight");
                let quantized = scheme.weight(weight, out, inputs);
                let device =
                    QuantLinear::upload(&runtime, &quantized, out, inputs).expect("upload");
                let mut restored = quantized.dequantize(out, inputs);
                if matches!(quant.backend, QuantBackend::Dequant(_)) {
                    // the dequantize kernel rounds the weight to FP16 for the product
                    for value in &mut restored {
                        *value = f16_value(f16_bits(*value));
                    }
                }

                for rows in [1, 77, 301] {
                    let bits: Vec<u16> = (0..rows * inputs)
                        .map(|i| {
                            let outlier = if i % 509 == 0 { 20.0 } else { 3.0 };
                            f16_bits(random() * outlier)
                        })
                        .collect();
                    let values: Vec<f32> = bits.iter().map(|bits| f16_value(*bits)).collect();
                    let initial: Vec<f32> = (0..rows * out).map(|_| random()).collect();
                    let (host_bits, activation) = quantized_activation(scheme, &values, inputs);

                    let stream = runtime.stream();
                    let a = stream.clone_htod(&bits).expect("activation");
                    let mut c = stream.clone_htod(&initial).expect("output");
                    let mut inputs_ws = quant
                        .inputs(&runtime, spec, rows.max(1), rows.max(1))
                        .expect("quantized inputs");
                    quant
                        .linear(
                            &runtime,
                            linear(rows, out, inputs, alpha, beta, Math::Tf32),
                            &a.slice(..),
                            &device,
                            &mut c,
                            inputs_ws.as_mut(),
                        )
                        .expect("product");
                    let ours = stream.clone_dtoh(&c).expect("download");

                    if let Some(inputs_ws) = &inputs_ws {
                        let device_bits = stream.clone_dtoh(&inputs_ws.values).expect("bits");
                        let differ = host_bits
                            .iter()
                            .zip(&device_bits)
                            .filter(|(host, device)| host != device)
                            .count();
                        assert_eq!(
                            differ,
                            0,
                            "{name} {tensor} rows {rows}: {differ} of {} quantized activation bytes differ from the host",
                            host_bits.len()
                        );
                    }

                    let mut expected = vec![0f32; rows * out];
                    for r in 0..rows {
                        let a_row = &activation[r * inputs..(r + 1) * inputs];
                        for o in 0..out {
                            let w_row = &restored[o * inputs..(o + 1) * inputs];
                            let dot: f64 = a_row
                                .iter()
                                .zip(w_row)
                                .map(|(a, w)| f64::from(*a) * f64::from(*w))
                                .sum();
                            let at = r * out + o;
                            expected[at] = (f64::from(alpha) * dot
                                + f64::from(beta) * f64::from(initial[at]))
                                as f32;
                        }
                    }
                    let error = distance(&ours, &expected, rows, out, &[rows]);
                    println!("{name} {tensor} rows {rows}: relative error {error:.2e}");
                    assert!(error < 1e-4, "{name} {tensor} rows {rows}: {error:.2e}");
                }
            }
        }
    }

    /// Runs each quantized encoder end to end on the NeMo references and prints its
    /// distance from NeMo FP32 and from the FP16 encoder, with its linear weights'
    /// device bytes. WER is the accuracy gate; this catches a broken product, which
    /// lands at order 1
    #[test]
    #[ignore = "needs a CUDA GPU and the NeMo references; `just gpu-parity` runs it on gpu0"]
    fn gpu_quantized_encoders_stay_near_nemo() {
        let runtime = CudaRuntime::open(0, Options::default()).expect("CUDA runtime");
        runtime.prepare(Library::Cublas).expect("cuBLAS");
        for version in [ParakeetVersion::V2, ParakeetVersion::V3] {
            let file = weights(version);
            let spec = file
                .parakeet_spec(&deployment(version))
                .expect("spec")
                .encoder;
            let inputs: Vec<_> = ["b1", "b8", "long-full"]
                .into_iter()
                .map(|set| (set, NemoInput::open(version, set)))
                .collect();
            let fp16 = CudaEncoder::load(&runtime, &file, spec, EncoderMath::Fp16).expect("fp16");
            let baseline: Vec<_> = inputs
                .iter()
                .map(|(_, input)| input.encode(&runtime, &fp16).0)
                .collect();
            println!(
                "{version:?} FP16 linear weights: {} bytes",
                fp16.linear_bytes()
            );
            drop(fp16);

            for (name, scheme) in SCHEMES {
                let math = EncoderMath::Quantized(scheme);
                let encoder = match CudaEncoder::load(&runtime, &file, spec, math) {
                    Ok(encoder) => encoder,
                    Err(error) if unsupported_here(&error) => {
                        println!("{version:?} {name}: unsupported here: {error}");
                        continue;
                    }
                    Err(error) => panic!("{version:?} {name}: {error}"),
                };
                println!(
                    "{version:?} {name} linear weights: {} bytes",
                    encoder.linear_bytes()
                );
                for ((set, input), fp16) in inputs.iter().zip(&baseline) {
                    let (output, lens) = input.encode(&runtime, &encoder);
                    let frames = input.expected.shape()[1];
                    let expected = input.expected.as_slice().expect("contiguous");
                    let to_nemo = distance(&output, expected, frames, spec.d_model, &lens);
                    let to_fp16 = distance(&output, fp16, frames, spec.d_model, &lens);
                    println!(
                        "{version:?} {set} {name}: rel to NeMo {to_nemo:.2e}, to FP16 {to_fp16:.2e}"
                    );
                    assert!(to_nemo < 0.5, "{version:?} {set} {name}: {to_nemo:.2e}");
                }
            }
        }
    }

    /// Checks a graph replay of each quantized encoder against its eager pass, which
    /// also shows that the eager pass makes every cuBLASLt plan before the capture
    #[test]
    #[ignore = "needs a CUDA GPU and the NeMo references; `just gpu-parity` runs it on gpu0"]
    fn gpu_quantized_graphs_replay_the_eager_pass() {
        let runtime = CudaRuntime::open(0, Options::default()).expect("CUDA runtime");
        runtime.prepare(Library::Cublas).expect("cuBLAS");
        let file = weights(ParakeetVersion::V3);
        let spec = file.parakeet_spec(&ParakeetDeployment::v3()).expect("spec");
        for (name, scheme) in SCHEMES {
            let math = EncoderMath::Quantized(scheme);
            let Ok(encoder) = CudaEncoder::load(&runtime, &file, spec.encoder, math) else {
                println!("v3 {name}: unsupported here");
                continue;
            };
            for set in ["b1", "b8"] {
                time_graph(&runtime, &encoder, name, set);
            }
        }
    }
}
