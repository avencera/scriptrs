//! FastConformer encoder on cuBLAS plus the `encoder` kernel area
//!
//! The forward pass follows NeMo's `ConformerEncoder` operation by operation in FP32,
//! so each stage can be held against the reference activations. Everything a pass
//! touches lives in an [`EncoderWorkspace`] allocated once per input shape, which
//! keeps [`CudaEncoder::enqueue`] free of allocations and host syncs and therefore
//! capturable in a CUDA graph.
//!
//! [`EncoderMath::Fp16`] stores the Conformer layers' linear weights in FP16 and has
//! the kernels before each of those GEMMs write an FP16 copy of their output, so the
//! GEMMs read half the bytes and run at the FP16 tensor-core rate. The residual
//! stream, the GEMM outputs and their accumulation stay FP32
//!
//! [`EncoderMath::Quantized`] keeps the FP16 activations and quantizes the same
//! linear layers further (see [`Scheme`]): FP8 and block-scaled formats quantize the
//! activation on the GPU before a cuBLASLt product, and weight-only formats
//! dequantize the weight to FP16 before the FP16 product

use std::sync::Mutex;

use cudakit::cudarc::driver::{CudaSlice, CudaView, CudaViewMut, DevicePtr, DevicePtrMut};
use cudakit::{DeviceTensor, Math, Sgemm, SgemmBatched};

use crate::cuda::fp16::linear_fp16;
use crate::cuda::kernels::{CudaRuntime, EncoderKernels, FeatureMap, GemmOperand, Heads};
use crate::cuda::lt::{Alpha, LtFormat, LtGemm, LtProduct};
use crate::cuda::quant::{
    AMAX_PARTIALS, MX_BLOCK, NV_BLOCK, QuantKernels, QuantizedWeight, ScaleLayout, Scheme,
    f16_value,
};
use crate::cuda::stored::{StoredLinear, file_scheme, linear_shape, read_linear};
use crate::cuda::{Direction, HostStaging, linear, upload, upload_host, zeros};
use crate::error::TranscriptionError;
use crate::native::spec::EncoderSpec;
use crate::native::weights::WeightFile;

/// PyTorch's default `eps` for `LayerNorm` and `BatchNorm1d`, which NeMo keeps
const NORM_EPS: f32 = 1e-5;

/// Every `dw_striding` convolution is 3x3 with stride 2 and padding 1
fn subsampled(len: usize) -> usize {
    // NeMo's `calculate_conv_output_size`: (len + 1 + 1 - 3) // 2 + 1
    (len + 2).saturating_sub(3) / 2 + 1
}

/// Precision of the encoder's GEMMs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EncoderMath {
    /// Every GEMM on FP32 operands in this cuBLAS math
    Single(Math),
    /// The Conformer layers' linear layers on FP16 operands with FP32 output and
    /// accumulation. The attention score and mix products and the subsampling module
    /// keep FP32 operands in TF32: they are a few percent of the work, and their
    /// operands would need conversions of their own
    Fp16,
    /// [`Self::Fp16`] with the linear layers quantized further, except those the
    /// quantization keeps at FP16
    Quantized(Quantization),
}

impl EncoderMath {
    /// The math of the GEMMs that keep FP32 operands
    fn single(self) -> Math {
        match self {
            Self::Single(math) => math,
            Self::Fp16 | Self::Quantized(_) => Math::Tf32,
        }
    }

    /// Whether the kernels write FP16 copies of the linear layers' inputs
    fn fp16(self) -> bool {
        !matches!(self, Self::Single(_))
    }
}

/// A [`Scheme`] for the Conformer layers' linear layers, and the layers it leaves at
/// FP16
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Quantization {
    pub(crate) scheme: Scheme,
    pub(crate) keep: KeepFp16,
}

impl Quantization {
    /// Every linear layer quantized with `scheme`
    pub(crate) fn all(scheme: Scheme) -> Self {
        Self {
            scheme,
            keep: KeepFp16::NONE,
        }
    }
}

/// Linear layers left at FP16 under [`EncoderMath::Quantized`]: every linear layer of
/// the layers in `layers` (bit `i` for layer `i`), and the layers of the kinds in
/// `kinds` (bit [`LinearKind`] `as u8`) in every layer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct KeepFp16 {
    pub(crate) layers: u64,
    pub(crate) kinds: u8,
}

impl KeepFp16 {
    pub(crate) const NONE: Self = Self {
        layers: 0,
        kinds: 0,
    };

    /// The layers the experimental `SCRIPTRS_CUDA_KEEP_FP16` variable names, for
    /// measuring mixed recipes against weights quantized at load: comma-separated
    /// layer indices, `first-last` ranges and kinds (`ff-up`, `ff-down`, `qkv`, `pos`,
    /// `out`, `pw1`, `pw2`), as in `12-18,ff-up`
    pub(crate) fn from_env() -> Result<Self, TranscriptionError> {
        let Ok(value) = std::env::var("SCRIPTRS_CUDA_KEEP_FP16") else {
            return Ok(Self::NONE);
        };
        let invalid = |token: &str| {
            TranscriptionError::InvalidConfiguration(format!(
                "SCRIPTRS_CUDA_KEEP_FP16 names no layer or kind `{token}`"
            ))
        };
        let layer = |token: &str| match token.parse::<u32>() {
            Ok(layer) if layer < 64 => Ok(layer),
            _ => Err(invalid(token)),
        };
        let mut keep = Self::NONE;
        for token in value
            .split(',')
            .map(str::trim)
            .filter(|token| !token.is_empty())
        {
            let kind = match token {
                "ff-up" => Some(LinearKind::FeedForwardUp),
                "ff-down" => Some(LinearKind::FeedForwardDown),
                "qkv" => Some(LinearKind::Qkv),
                "pos" => Some(LinearKind::Position),
                "out" => Some(LinearKind::AttentionOut),
                "pw1" => Some(LinearKind::Pointwise1),
                "pw2" => Some(LinearKind::Pointwise2),
                _ => None,
            };
            if let Some(kind) = kind {
                keep.kinds |= 1 << kind as u8;
                continue;
            }

            let (first, last) = match token.split_once('-') {
                Some((first, last)) => (layer(first)?, layer(last)?),
                None => (layer(token)?, layer(token)?),
            };
            for layer in first..=last {
                keep.layers |= 1 << layer;
            }
        }
        Ok(keep)
    }

    fn keeps(self, layer: usize, kind: LinearKind) -> bool {
        (layer < 64 && self.layers >> layer & 1 == 1) || self.kinds >> kind as u8 & 1 == 1
    }
}

/// The linear layers of a Conformer layer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LinearKind {
    /// `linear1` of both feed-forward modules
    FeedForwardUp,
    /// `linear2` of both feed-forward modules
    FeedForwardDown,
    Qkv,
    Position,
    AttentionOut,
    Pointwise1,
    Pointwise2,
}

/// A linear layer's `[out, in]` weight, in the precision its GEMM reads
enum Linear {
    Single(DeviceTensor),
    /// FP16 bits
    Fp16(CudaSlice<u16>),
    /// Read with FP16 activations, which the product quantizes to match
    Quantized(QuantLinear),
}

/// A quantized `[out, inputs]` weight on the device
struct QuantLinear {
    scheme: Scheme,
    out: usize,
    inputs: usize,
    values: CudaSlice<u8>,
    weight: DeviceScales,
}

/// The scales of a [`QuantLinear`], as its [`QuantizedWeight`] holds them
enum DeviceScales {
    /// FP8, INT8 and INT4 scales
    Float(CudaSlice<f32>),
    /// MXFP8 block scales
    Blocks(CudaSlice<u8>),
    /// NVFP4 block scales and the tensor scale
    Nvfp4 { blocks: CudaSlice<u8>, global: f32 },
}

/// What [`EncoderMath::Quantized`] adds to the encoder
struct QuantState {
    quantization: Quantization,
    kernels: QuantKernels,
    /// cuBLASLt products, for every scheme but the weight-only ones
    lt: Option<LtGemm>,
    /// Without cuBLASLt's outer-vector scales, [`Scheme::Fp8Channel`] multiplies with
    /// a weight scale of 1 into [`QuantInputs::product`] and applies the channel scales
    /// after
    scale_after: bool,
    /// `[1.0]`, the weight scale of those products
    one: CudaSlice<f32>,
    /// The dequantized FP16 weight of a weight-only product, sized for the largest
    dequant: Option<Mutex<CudaSlice<u16>>>,
}

/// The activation a linear layer reads, matching its [`Linear`] weight
enum Operand<'a> {
    Single(CudaView<'a, f32>),
    Fp16(CudaView<'a, u16>),
}

struct Norm {
    weight: DeviceTensor,
    bias: DeviceTensor,
}

struct FeedForward {
    linear1: Linear,
    linear2: Linear,
}

struct Attention {
    /// `linear_q`, `linear_k` and `linear_v` stacked to `[3 * d_model, d_model]`
    qkv: Linear,
    pos: Linear,
    out: Linear,
    bias_u: DeviceTensor,
    bias_v: DeviceTensor,
}

struct ConvModule {
    pointwise1: Linear,
    depthwise: DeviceTensor,
    /// Eval-mode BatchNorm folded to `x * scale + shift`
    scale: DeviceTensor,
    shift: DeviceTensor,
    pointwise2: Linear,
}

struct Layer {
    norm_ff1: Norm,
    ff1: FeedForward,
    norm_att: Norm,
    attention: Attention,
    norm_conv: Norm,
    conv: ConvModule,
    norm_ff2: Norm,
    ff2: FeedForward,
    norm_out: Norm,
}

struct Subsampling {
    conv_in: (DeviceTensor, DeviceTensor),
    depthwise1: (DeviceTensor, DeviceTensor),
    pointwise1: (DeviceTensor, DeviceTensor),
    depthwise2: (DeviceTensor, DeviceTensor),
    pointwise2: (DeviceTensor, DeviceTensor),
    out: (DeviceTensor, DeviceTensor),
}

/// Encoder weights on the GPU and the kernels that run them
pub(crate) struct CudaEncoder {
    spec: EncoderSpec,
    math: EncoderMath,
    subsampling: Subsampling,
    layers: Vec<Layer>,
    kernels: EncoderKernels,
    quant: Option<QuantState>,
}

/// Device buffers for one `(batch, mel frames)` input shape
pub(crate) struct EncoderWorkspace {
    batch: usize,
    mel_frames: usize,
    /// Encoder frames after 8x subsampling
    frames: usize,
    /// `[batch, mel_frames, features]`
    mel: CudaSlice<f32>,
    /// Pinned host copy of `mel` that [`CudaEncoder::set_input`] fills
    mel_host: HostStaging<f32>,
    /// Valid frames per batch row before and after each subsampling stage
    lens: [CudaSlice<u32>; 4],
    host_lens: Vec<usize>,
    /// Subsampling feature maps, ping-ponged: `[0]` fits the first stage, `[1]` the
    /// second
    maps: [CudaSlice<f32>; 2],
    flat: CudaSlice<f32>,
    /// The residual stream, `[batch * frames, d_model]`
    x: CudaSlice<f32>,
    normed: CudaSlice<f32>,
    hidden: CudaSlice<f32>,
    qkv: CudaSlice<f32>,
    q_u: CudaSlice<f32>,
    q_v: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    pos_emb: CudaSlice<f32>,
    pos: CudaSlice<f32>,
    pos_heads: CudaSlice<f32>,
    scores: CudaSlice<f32>,
    bd: CudaSlice<f32>,
    context: CudaSlice<f32>,
    merged: CudaSlice<f32>,
    /// FP16 GEMM inputs under [`EncoderMath::Fp16`] and [`EncoderMath::Quantized`]
    fp16: Option<Fp16Inputs>,
    /// Quantized GEMM inputs under [`EncoderMath::Quantized`] for the schemes that
    /// quantize activations
    quant: Option<QuantInputs>,
}

/// One quantized activation at a time: each product quantizes its input here just
/// before it runs, on the same stream
struct QuantInputs {
    /// E4M3 bits, or E2M1 pairs
    values: CudaSlice<u8>,
    /// FP32 activation scales: one, or one per row
    scales: CudaSlice<f32>,
    /// Tiled block scales
    blocks: CudaSlice<u8>,
    /// [`AMAX_PARTIALS`] partial maxima of the activation
    partials: CudaSlice<f32>,
    /// The device alpha of an NVFP4 product
    alpha: CudaSlice<f32>,
    /// The unscaled output of an FP8 product whose channel scales apply after
    product: Option<CudaSlice<f32>>,
}

/// FP16 copies of the activations the FP16 linear layers read
struct Fp16Inputs {
    /// `[batch * frames, d_model]` LayerNorm output
    normed: CudaSlice<u16>,
    /// `[batch * frames, ff_hidden]` SiLU output
    hidden: CudaSlice<u16>,
    /// `[batch * frames, d_model]` attention context or conv module output
    merged: CudaSlice<u16>,
    /// `[2 * frames - 1, d_model]` relative position table
    pos_emb: CudaSlice<u16>,
}

impl EncoderWorkspace {
    /// Mel frames per batch row of the input
    pub(crate) fn mel_frames(&self) -> usize {
        self.mel_frames
    }

    pub(crate) fn frames(&self) -> usize {
        self.frames
    }

    /// Valid encoder frames per batch row of the current input
    pub(crate) fn output_lens(&self) -> &[usize] {
        &self.host_lens
    }

    /// The encoder output, `[batch * frames, d_model]`
    pub(crate) fn output(&self) -> &CudaSlice<f32> {
        &self.x
    }

    /// Device bytes held by the workspace's buffers
    pub(crate) fn bytes(&self) -> usize {
        let lens: usize = self.lens.iter().map(CudaSlice::num_bytes).sum();
        let buffers = [
            &self.mel,
            &self.maps[0],
            &self.maps[1],
            &self.flat,
            &self.x,
            &self.normed,
            &self.hidden,
            &self.qkv,
            &self.q_u,
            &self.q_v,
            &self.k,
            &self.v,
            &self.pos_emb,
            &self.pos,
            &self.pos_heads,
            &self.scores,
            &self.bd,
            &self.context,
            &self.merged,
        ];
        let fp16 = self.fp16.as_ref().map_or(0, |fp16| {
            [&fp16.normed, &fp16.hidden, &fp16.merged, &fp16.pos_emb]
                .iter()
                .map(|buffer| buffer.num_bytes())
                .sum()
        });
        let quant = self.quant.as_ref().map_or(0, |quant| {
            quant.values.num_bytes()
                + quant.scales.num_bytes()
                + quant.blocks.num_bytes()
                + quant.partials.num_bytes()
                + quant.alpha.num_bytes()
                + quant.product.as_ref().map_or(0, CudaSlice::num_bytes)
        });
        lens + fp16
            + quant
            + buffers
                .iter()
                .map(|buffer| buffer.num_bytes())
                .sum::<usize>()
    }
}

impl CudaEncoder {
    /// Uploads the encoder weights and loads the kernels; the GEMMs run at `math`
    pub(crate) fn load(
        runtime: &CudaRuntime,
        file: &WeightFile,
        spec: EncoderSpec,
        math: EncoderMath,
    ) -> Result<Self, TranscriptionError> {
        let kernels = EncoderKernels::load(runtime)?;
        let d = spec.d_model;
        let channels = spec.subsampling_channels;
        let flat = channels * spec.features / spec.subsampling_factor;
        let pair = |name: &str, shape: &[usize]| -> Result<_, TranscriptionError> {
            Ok((
                upload(
                    runtime,
                    file,
                    &format!("encoder.pre_encode.{name}.weight"),
                    shape,
                )?,
                upload(
                    runtime,
                    file,
                    &format!("encoder.pre_encode.{name}.bias"),
                    &shape[..1],
                )?,
            ))
        };
        let subsampling = Subsampling {
            conv_in: pair("conv.0", &[channels, 1, 3, 3])?,
            depthwise1: pair("conv.2", &[channels, 1, 3, 3])?,
            pointwise1: pair("conv.3", &[channels, channels, 1, 1])?,
            depthwise2: pair("conv.5", &[channels, 1, 3, 3])?,
            pointwise2: pair("conv.6", &[channels, channels, 1, 1])?,
            out: pair("out", &[d, flat])?,
        };

        let wanted = match math {
            EncoderMath::Quantized(quantization) => Some(quantization.scheme),
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
        let quant = match math {
            EncoderMath::Quantized(quantization) => {
                Some(QuantState::load(runtime, quantization, spec)?)
            }
            _ => None,
        };
        let mut layers = Vec::with_capacity(spec.layers);
        for index in 0..spec.layers {
            layers.push(Self::load_layer(
                runtime, &kernels, file, spec, math, index,
            )?);
        }

        Ok(Self {
            spec,
            math,
            subsampling,
            layers,
            kernels,
            quant,
        })
    }

    fn load_layer(
        runtime: &CudaRuntime,
        kernels: &EncoderKernels,
        file: &WeightFile,
        spec: EncoderSpec,
        math: EncoderMath,
        index: usize,
    ) -> Result<Layer, TranscriptionError> {
        let d = spec.d_model;
        let prefix = format!("encoder.layers.{index}");
        let tensor = |name: &str, shape: &[usize]| file.tensor(&format!("{prefix}.{name}"), shape);
        let load =
            |name: &str, shape: &[usize]| upload(runtime, file, &format!("{prefix}.{name}"), shape);
        let linear = |name: &str, kind: LinearKind| {
            let math = match math {
                EncoderMath::Quantized(quantization) if quantization.keep.keeps(index, kind) => {
                    EncoderMath::Fp16
                }
                math => math,
            };
            let (out, inputs) = linear_shape(spec, kind);
            let stored = read_linear(file, index, name, out, inputs)?;
            Linear::from_stored(runtime, kernels, stored, out, inputs, math)
        };
        let norm = |name: &str| -> Result<Norm, TranscriptionError> {
            Ok(Norm {
                weight: load(&format!("{name}.weight"), &[d])?,
                bias: load(&format!("{name}.bias"), &[d])?,
            })
        };
        let feed_forward = |name: &str| -> Result<FeedForward, TranscriptionError> {
            Ok(FeedForward {
                linear1: linear(&format!("{name}.linear1"), LinearKind::FeedForwardUp)?,
                linear2: linear(&format!("{name}.linear2"), LinearKind::FeedForwardDown)?,
            })
        };

        let head_dim = d / spec.heads;
        let attention = Attention {
            qkv: linear("self_attn.linear_qkv", LinearKind::Qkv)?,
            pos: linear("self_attn.linear_pos", LinearKind::Position)?,
            out: linear("self_attn.linear_out", LinearKind::AttentionOut)?,
            bias_u: load("self_attn.pos_bias_u", &[spec.heads, head_dim])?,
            bias_v: load("self_attn.pos_bias_v", &[spec.heads, head_dim])?,
        };

        // fold in f64 so the folded constants round once
        let bn = |name: &str| tensor(&format!("conv.batch_norm.{name}"), &[d]);
        let (gamma, beta, mean, var) = (
            bn("weight")?,
            bn("bias")?,
            bn("running_mean")?,
            bn("running_var")?,
        );
        let mut scale = Vec::with_capacity(d);
        let mut shift = Vec::with_capacity(d);
        for c in 0..d {
            let s = f64::from(gamma[c]) / (f64::from(var[c]) + f64::from(NORM_EPS)).sqrt();
            scale.push(s as f32);
            shift.push((f64::from(beta[c]) - f64::from(mean[c]) * s) as f32);
        }
        let conv = ConvModule {
            pointwise1: linear("conv.pointwise_conv1", LinearKind::Pointwise1)?,
            depthwise: load("conv.depthwise_conv.weight", &[d, 1, spec.conv_kernel])?,
            scale: upload_host(runtime, &scale)?,
            shift: upload_host(runtime, &shift)?,
            pointwise2: linear("conv.pointwise_conv2", LinearKind::Pointwise2)?,
        };

        Ok(Layer {
            norm_ff1: norm("norm_feed_forward1")?,
            ff1: feed_forward("feed_forward1")?,
            norm_att: norm("norm_self_att")?,
            attention,
            norm_conv: norm("norm_conv")?,
            conv,
            norm_ff2: norm("norm_feed_forward2")?,
            ff2: feed_forward("feed_forward2")?,
            norm_out: norm("norm_out")?,
        })
    }

    /// Allocates the buffers for `batch` inputs of up to `mel_frames` frames, and the
    /// relative position table for that length
    pub(crate) fn workspace(
        &self,
        runtime: &CudaRuntime,
        batch: usize,
        mel_frames: usize,
    ) -> Result<EncoderWorkspace, TranscriptionError> {
        let stream = runtime.stream();
        let spec = self.spec;
        let (d, heads) = (spec.d_model, spec.heads);
        let channels = spec.subsampling_channels;
        let maps = self.feature_maps(batch, mel_frames);
        let frames = maps[3].frames;
        let rows = batch * frames;
        let positions = 2 * frames - 1;
        let pos_emb = stream.clone_htod(&relative_positions(frames, d))?;
        let fp16 = match self.math {
            EncoderMath::Single(_) => None,
            EncoderMath::Fp16 | EncoderMath::Quantized(_) => {
                let mut table = zeros(stream, pos_emb.len())?;
                self.kernels
                    .to_f16(stream, &pos_emb.slice(..), &mut table.slice_mut(..))?;
                Some(Fp16Inputs {
                    normed: zeros(stream, rows * d)?,
                    hidden: zeros(stream, rows * spec.ff_hidden)?,
                    merged: zeros(stream, rows * d)?,
                    pos_emb: table,
                })
            }
        };

        Ok(EncoderWorkspace {
            batch,
            mel_frames,
            frames,
            mel: zeros(stream, batch * mel_frames * spec.features)?,
            mel_host: HostStaging::new(
                stream,
                batch * mel_frames * spec.features,
                Direction::Upload,
            )?,
            lens: [
                zeros(stream, batch)?,
                zeros(stream, batch)?,
                zeros(stream, batch)?,
                zeros(stream, batch)?,
            ],
            host_lens: vec![0; batch],
            maps: [zeros(stream, maps[1].len())?, zeros(stream, maps[2].len())?],
            flat: zeros(stream, rows * channels * maps[3].features)?,
            x: zeros(stream, rows * d)?,
            normed: zeros(stream, rows * d)?,
            hidden: zeros(stream, rows * spec.ff_hidden.max(2 * d))?,
            qkv: zeros(stream, rows * 3 * d)?,
            q_u: zeros(stream, rows * d)?,
            q_v: zeros(stream, rows * d)?,
            k: zeros(stream, rows * d)?,
            v: zeros(stream, rows * d)?,
            pos_emb,
            pos: zeros(stream, positions * d)?,
            pos_heads: zeros(stream, positions * d)?,
            scores: zeros(stream, batch * heads * frames * frames)?,
            bd: zeros(stream, rows * heads * positions)?,
            context: zeros(stream, rows * d)?,
            merged: zeros(stream, rows * d)?,
            fp16,
            quant: match &self.quant {
                Some(quant) => quant.inputs(runtime, spec, rows, positions)?,
                None => None,
            },
        })
    }

    /// Input and subsampled feature maps: `[0]` is the mel, `[3]` the last stage
    fn feature_maps(&self, batch: usize, mel_frames: usize) -> [FeatureMap; 4] {
        let channels = self.spec.subsampling_channels;
        let mel = FeatureMap {
            batch,
            channels: 1,
            frames: mel_frames,
            features: self.spec.features,
        };
        let next = |map: FeatureMap| FeatureMap {
            batch,
            channels,
            frames: subsampled(map.frames),
            features: subsampled(map.features),
        };
        let first = next(mel);
        let second = next(first);
        [mel, first, second, next(second)]
    }

    /// Queues the upload of a padded batch: `fill` writes every value of the
    /// `[batch, mel_frames, features]` (time-major) input, and `lens` holds the valid
    /// mel frames of each row
    ///
    /// `fill` writes straight into pinned memory, so the upload queues behind earlier
    /// work instead of blocking the host
    pub(crate) fn set_input(
        &self,
        runtime: &CudaRuntime,
        workspace: &mut EncoderWorkspace,
        lens: &[usize],
        fill: impl FnOnce(&mut [f32]),
    ) -> Result<(), TranscriptionError> {
        if lens.len() != workspace.batch {
            return Err(TranscriptionError::InvalidConfiguration(format!(
                "encoder input has {} rows; the workspace takes {}",
                lens.len(),
                workspace.batch
            )));
        }
        if let Some(len) = lens
            .iter()
            .find(|len| **len == 0 || **len > workspace.mel_frames)
        {
            return Err(TranscriptionError::InvalidConfiguration(format!(
                "encoder input row has {len} valid frames of {}",
                workspace.mel_frames
            )));
        }

        let stream = runtime.stream();
        fill(workspace.mel_host.write()?);
        workspace.mel_host.upload(stream, &mut workspace.mel)?;
        let mut stage: Vec<usize> = lens.to_vec();
        for buffer in &mut workspace.lens {
            let values: Vec<u32> = stage.iter().map(|len| *len as u32).collect();
            stream.memcpy_htod(&values, buffer)?;
            stage = stage.into_iter().map(subsampled).collect();
        }
        // the loop subsampled once more than there are stages
        workspace.host_lens = lens
            .iter()
            .map(|len| subsampled(subsampled(subsampled(*len))))
            .collect();
        Ok(())
    }

    /// Queues the whole encoder on the runtime's stream
    pub(crate) fn enqueue(
        &self,
        runtime: &CudaRuntime,
        workspace: &mut EncoderWorkspace,
    ) -> Result<(), TranscriptionError> {
        self.enqueue_subsampling(runtime, workspace)?;
        for index in 0..self.layers.len() {
            self.enqueue_layer(runtime, workspace, index)?;
        }
        Ok(())
    }

    /// `pre_encode`: three stride-2 convolutions, then the projection to `d_model`
    pub(crate) fn enqueue_subsampling(
        &self,
        runtime: &CudaRuntime,
        ws: &mut EncoderWorkspace,
    ) -> Result<(), TranscriptionError> {
        let stream = runtime.stream();
        let kernels = &self.kernels;
        let sub = &self.subsampling;
        let maps = self.feature_maps(ws.batch, ws.mel_frames);
        let [mel_map, first, second, third] = maps;
        let channels = self.spec.subsampling_channels;
        let [large, small] = &mut ws.maps;

        kernels.subsample_conv_in(
            stream,
            &ws.mel.slice(..),
            &ws.lens[0].slice(..),
            &ws.lens[1].slice(..),
            &sub.conv_in.0.data().slice(..),
            &sub.conv_in.1.data().slice(..),
            &mut large.slice_mut(..),
            mel_map,
            first,
        )?;
        kernels.subsample_depthwise(
            stream,
            &large.slice(..),
            &ws.lens[2].slice(..),
            &sub.depthwise1.0.data().slice(..),
            &sub.depthwise1.1.data().slice(..),
            &mut small.slice_mut(..),
            first,
            second,
        )?;
        self.pointwise(runtime, &sub.pointwise1, small, large, second, &ws.lens[2])?;
        kernels.subsample_depthwise(
            stream,
            &large.slice(..second.len()),
            &ws.lens[3].slice(..),
            &sub.depthwise2.0.data().slice(..),
            &sub.depthwise2.1.data().slice(..),
            &mut small.slice_mut(..third.len()),
            second,
            third,
        )?;
        self.pointwise(runtime, &sub.pointwise2, small, large, third, &ws.lens[3])?;
        kernels.subsample_flatten(
            stream,
            &large.slice(..third.len()),
            &mut ws.flat.slice_mut(..),
            third,
        )?;

        let rows = ws.batch * ws.frames;
        let (weight, bias) = &sub.out;
        kernels.fill_rows(stream, &bias.data().slice(..), &mut ws.x.slice_mut(..))?;
        runtime.sgemm(
            linear(
                rows,
                self.spec.d_model,
                channels * third.features,
                1.0,
                1.0,
                self.math.single(),
            ),
            &ws.flat,
            weight.data(),
            &mut ws.x,
        )?;
        Ok(())
    }

    /// A pointwise subsampling convolution, `[channels, channels]` over each row's
    /// `[channels, frames * features]`, then bias, ReLU and the time mask
    fn pointwise(
        &self,
        runtime: &CudaRuntime,
        (weight, bias): &(DeviceTensor, DeviceTensor),
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        map: FeatureMap,
        lens: &CudaSlice<u32>,
    ) -> Result<(), TranscriptionError> {
        let plane = map.frames * map.features;
        let per_row = map.channels * plane;
        let gemm = Sgemm {
            math: self.math.single(),
            ..Sgemm::new(map.channels, plane, map.channels)
        };
        runtime.sgemm_batched(
            SgemmBatched {
                gemm,
                batch: map.batch,
                stride_a: 0,
                stride_b: per_row,
                stride_c: per_row,
            },
            weight.data(),
            &input.slice(..map.len()),
            &mut output.slice_mut(..map.len()),
        )?;
        self.kernels.subsample_bias_relu(
            runtime.stream(),
            &mut output.slice_mut(..map.len()),
            &bias.data().slice(..),
            &lens.slice(..),
            map,
        )?;
        Ok(())
    }

    /// One Conformer layer on the residual stream
    ///
    /// Layers run in order after [`Self::enqueue_subsampling`]: a layer's closing
    /// norm also writes the next layer's first norm to `normed` (or its FP16 copy),
    /// so every layer but the first starts from it
    pub(crate) fn enqueue_layer(
        &self,
        runtime: &CudaRuntime,
        ws: &mut EncoderWorkspace,
        index: usize,
    ) -> Result<(), TranscriptionError> {
        let layer = &self.layers[index];
        if index == 0 {
            self.norm(runtime, ws, &layer.norm_ff1)?;
        }
        self.feed_forward(runtime, ws, &layer.ff1)?;
        self.norm(runtime, ws, &layer.norm_att)?;
        self.attention(runtime, ws, &layer.attention)?;
        self.norm(runtime, ws, &layer.norm_conv)?;
        self.conv_module(runtime, ws, &layer.conv)?;
        self.norm(runtime, ws, &layer.norm_ff2)?;
        self.feed_forward(runtime, ws, &layer.ff2)?;

        // the final norm is not residual, so it runs in place, and the next layer's
        // first norm reads its result before it leaves the registers
        let next = self.layers.get(index + 1).map(|next| {
            (
                next.norm_ff1.weight.data().slice(..),
                next.norm_ff1.bias.data().slice(..),
            )
        });
        self.kernels.layer_norm_pair(
            runtime.stream(),
            &mut ws.x.slice_mut(..),
            (
                &layer.norm_out.weight.data().slice(..),
                &layer.norm_out.bias.data().slice(..),
            ),
            next.as_ref().map(|(gamma, beta)| (gamma, beta)),
            self.gemm_output(
                &mut ws.normed,
                ws.fp16.as_mut().map(|fp16| &mut fp16.normed),
            )?,
            self.spec.d_model,
            NORM_EPS,
        )?;
        Ok(())
    }

    /// LayerNorm of the residual stream for the next linear layer: into `ws.normed`,
    /// or its FP16 copy under [`EncoderMath::Fp16`]
    fn norm(
        &self,
        runtime: &CudaRuntime,
        ws: &mut EncoderWorkspace,
        norm: &Norm,
    ) -> Result<(), TranscriptionError> {
        let out = self.gemm_output(
            &mut ws.normed,
            ws.fp16.as_mut().map(|fp16| &mut fp16.normed),
        )?;
        self.kernels.layer_norm(
            runtime.stream(),
            &ws.x.slice(..),
            &norm.weight.data().slice(..),
            &norm.bias.data().slice(..),
            out,
            self.spec.d_model,
            NORM_EPS,
        )?;
        Ok(())
    }

    /// Where a kernel writes the activation a linear layer reads next: `single`, or
    /// `fp16`, its FP16 copy, under [`EncoderMath::Fp16`]
    fn gemm_output<'a>(
        &self,
        single: &'a mut CudaSlice<f32>,
        fp16: Option<&'a mut CudaSlice<u16>>,
    ) -> Result<GemmOperand<'a>, TranscriptionError> {
        match (self.math.fp16(), fp16) {
            (false, _) => Ok(GemmOperand::Single(single.slice_mut(..))),
            (true, Some(fp16)) => Ok(GemmOperand::Fp16(fp16.slice_mut(..))),
            (true, None) => Err(mixed_precision()),
        }
    }

    /// The activation a linear layer reads: `single`, or `fp16`, its FP16 copy, under
    /// [`EncoderMath::Fp16`]
    fn gemm_input<'a>(
        &self,
        single: CudaView<'a, f32>,
        fp16: Option<CudaView<'a, u16>>,
    ) -> Result<Operand<'a>, TranscriptionError> {
        match (self.math.fp16(), fp16) {
            (false, _) => Ok(Operand::Single(single)),
            (true, Some(fp16)) => Ok(Operand::Fp16(fp16)),
            (true, None) => Err(mixed_precision()),
        }
    }

    /// Runs `spec`, a [`linear`] product, on `a` and `weight` in their shared precision;
    /// a quantized weight quantizes `a` into `quant` first
    fn linear(
        &self,
        runtime: &CudaRuntime,
        spec: Sgemm,
        a: Operand<'_>,
        weight: &Linear,
        c: &mut CudaViewMut<'_, f32>,
        quant: Option<&mut QuantInputs>,
    ) -> Result<(), TranscriptionError> {
        match (a, weight) {
            (Operand::Single(a), Linear::Single(w)) => {
                runtime.sgemm(spec, &a, w.data(), c)?;
            }
            (Operand::Fp16(a), Linear::Fp16(w)) => {
                linear_fp16(runtime, spec, &a, w, c)?;
            }
            (Operand::Fp16(a), Linear::Quantized(w)) => {
                let state = self.quant.as_ref().ok_or_else(mixed_precision)?;
                state.linear(runtime, spec, &a, w, c, quant)?;
            }
            _ => return Err(mixed_precision()),
        }
        Ok(())
    }

    /// `x += 0.5 * linear2(silu(linear1(normed)))`, with `normed` the sublayer's
    /// norm of `x`
    fn feed_forward(
        &self,
        runtime: &CudaRuntime,
        ws: &mut EncoderWorkspace,
        weights: &FeedForward,
    ) -> Result<(), TranscriptionError> {
        let (d, hidden) = (self.spec.d_model, self.spec.ff_hidden);
        let math = self.math.single();
        let rows = ws.batch * ws.frames;
        let normed = self.gemm_input(
            ws.normed.slice(..),
            ws.fp16.as_ref().map(|fp16| fp16.normed.slice(..)),
        )?;
        self.linear(
            runtime,
            linear(rows, hidden, d, 1.0, 0.0, math),
            normed,
            &weights.linear1,
            &mut ws.hidden.slice_mut(..rows * hidden),
            ws.quant.as_mut(),
        )?;
        let stream = runtime.stream();
        match ws.fp16.as_mut() {
            Some(fp16) => self.kernels.silu_f16(
                stream,
                &ws.hidden.slice(..rows * hidden),
                &mut fp16.hidden.slice_mut(..),
            )?,
            None => self
                .kernels
                .silu(stream, &mut ws.hidden.slice_mut(..rows * hidden))?,
        }
        let activated = self.gemm_input(
            ws.hidden.slice(..rows * hidden),
            ws.fp16.as_ref().map(|fp16| fp16.hidden.slice(..)),
        )?;
        // NeMo scales the feed-forward residual by 0.5 (`fc_factor`)
        self.linear(
            runtime,
            linear(rows, d, hidden, 0.5, 1.0, math),
            activated,
            &weights.linear2,
            &mut ws.x.slice_mut(..),
            ws.quant.as_mut(),
        )?;
        Ok(())
    }

    /// `x += linear_out(rel_pos_attention(normed))`, with `normed` the sublayer's
    /// norm of `x`
    fn attention(
        &self,
        runtime: &CudaRuntime,
        ws: &mut EncoderWorkspace,
        weights: &Attention,
    ) -> Result<(), TranscriptionError> {
        let stream = runtime.stream();
        let spec = self.spec;
        let (d, heads, head_dim) = (spec.d_model, spec.heads, spec.d_model / spec.heads);
        let (batch, frames) = (ws.batch, ws.frames);
        let rows = batch * frames;
        let positions = 2 * frames - 1;
        let shape = Heads {
            batch,
            frames,
            heads,
            head_dim,
        };

        let math = self.math.single();
        let normed = self.gemm_input(
            ws.normed.slice(..),
            ws.fp16.as_ref().map(|fp16| fp16.normed.slice(..)),
        )?;
        self.linear(
            runtime,
            linear(rows, 3 * d, d, 1.0, 0.0, math),
            normed,
            &weights.qkv,
            &mut ws.qkv.slice_mut(..),
            ws.quant.as_mut(),
        )?;
        self.kernels.split_qkv(
            stream,
            &ws.qkv.slice(..),
            &weights.bias_u.data().slice(..),
            &weights.bias_v.data().slice(..),
            &mut ws.q_u.slice_mut(..),
            &mut ws.q_v.slice_mut(..),
            &mut ws.k.slice_mut(..),
            &mut ws.v.slice_mut(..),
            shape,
        )?;
        let table = self.gemm_input(
            ws.pos_emb.slice(..),
            ws.fp16.as_ref().map(|fp16| fp16.pos_emb.slice(..)),
        )?;
        self.linear(
            runtime,
            linear(positions, d, d, 1.0, 0.0, math),
            table,
            &weights.pos,
            &mut ws.pos.slice_mut(..),
            ws.quant.as_mut(),
        )?;
        self.kernels.split_heads(
            stream,
            &ws.pos.slice(..),
            &mut ws.pos_heads.slice_mut(..),
            positions,
            heads,
            head_dim,
        )?;

        // content scores: (q + u) · kᵀ per (batch, head)
        let content = Sgemm {
            b_transposed: true,
            math,
            ..Sgemm::new(frames, frames, head_dim)
        };
        runtime.sgemm_batched(
            SgemmBatched::packed(content, batch * heads),
            &ws.q_u,
            &ws.k,
            &mut ws.scores,
        )?;
        // position scores: (q + v) · pᵀ per head, every utterance against one table
        let position = Sgemm {
            b_transposed: true,
            math,
            ..Sgemm::new(rows, positions, head_dim)
        };
        runtime.sgemm_batched(
            SgemmBatched::packed(position, heads),
            &ws.q_v,
            &ws.pos_heads,
            &mut ws.bd,
        )?;
        self.kernels.rel_pos_softmax(
            stream,
            &mut ws.scores.slice_mut(..),
            &ws.bd.slice(..),
            &ws.lens[3].slice(..),
            shape,
        )?;

        let mix = Sgemm {
            math,
            ..Sgemm::new(frames, head_dim, frames)
        };
        runtime.sgemm_batched(
            SgemmBatched::packed(mix, batch * heads),
            &ws.scores,
            &ws.v,
            &mut ws.context,
        )?;
        let merged = self.gemm_output(
            &mut ws.merged,
            ws.fp16.as_mut().map(|fp16| &mut fp16.merged),
        )?;
        self.kernels
            .merge_heads(stream, &ws.context.slice(..), merged, shape)?;
        let merged = self.gemm_input(
            ws.merged.slice(..),
            ws.fp16.as_ref().map(|fp16| fp16.merged.slice(..)),
        )?;
        self.linear(
            runtime,
            linear(rows, d, d, 1.0, 1.0, math),
            merged,
            &weights.out,
            &mut ws.x.slice_mut(..),
            ws.quant.as_mut(),
        )?;
        Ok(())
    }

    /// `x += pointwise2(silu(bn(depthwise(mask(glu(pointwise1(normed)))))))`, with
    /// `normed` the sublayer's norm of `x`
    fn conv_module(
        &self,
        runtime: &CudaRuntime,
        ws: &mut EncoderWorkspace,
        weights: &ConvModule,
    ) -> Result<(), TranscriptionError> {
        let d = self.spec.d_model;
        let math = self.math.single();
        let rows = ws.batch * ws.frames;
        let pw1_len = rows * 2 * d;
        let normed = self.gemm_input(
            ws.normed.slice(..),
            ws.fp16.as_ref().map(|fp16| fp16.normed.slice(..)),
        )?;
        self.linear(
            runtime,
            linear(rows, 2 * d, d, 1.0, 0.0, math),
            normed,
            &weights.pointwise1,
            &mut ws.hidden.slice_mut(..pw1_len),
            ws.quant.as_mut(),
        )?;
        let glued = self.gemm_output(
            &mut ws.merged,
            ws.fp16.as_mut().map(|fp16| &mut fp16.merged),
        )?;
        self.kernels.conv_glue(
            runtime.stream(),
            &ws.hidden.slice(..pw1_len),
            &ws.lens[3].slice(..),
            &weights.depthwise.data().slice(..),
            &weights.scale.data().slice(..),
            &weights.shift.data().slice(..),
            glued,
            ws.batch,
            ws.frames,
            d,
            self.spec.conv_kernel,
        )?;
        let glued = self.gemm_input(
            ws.merged.slice(..),
            ws.fp16.as_ref().map(|fp16| fp16.merged.slice(..)),
        )?;
        self.linear(
            runtime,
            linear(rows, d, d, 1.0, 1.0, math),
            glued,
            &weights.pointwise2,
            &mut ws.x.slice_mut(..),
            ws.quant.as_mut(),
        )?;
        Ok(())
    }
}

impl Linear {
    /// Uploads the `[out, inputs]` weight `values` for an FP32 or TF32 GEMM, rounds it
    /// to FP16 on the device for [`EncoderMath::Fp16`] and frees the FP32 copy, or
    /// quantizes it on the host for [`EncoderMath::Quantized`]
    fn new(
        runtime: &CudaRuntime,
        kernels: &EncoderKernels,
        values: &[f32],
        out: usize,
        inputs: usize,
        math: EncoderMath,
    ) -> Result<Self, TranscriptionError> {
        let stream = runtime.stream();
        match math {
            EncoderMath::Single(_) => Ok(Self::Single(DeviceTensor::upload(
                stream,
                values,
                &[out, inputs],
            )?)),
            EncoderMath::Fp16 => {
                let weight = DeviceTensor::upload(stream, values, &[out, inputs])?;
                let mut bits = zeros(stream, values.len())?;
                kernels.to_f16(stream, &weight.data().slice(..), &mut bits.slice_mut(..))?;
                Ok(Self::Fp16(bits))
            }
            EncoderMath::Quantized(quantization) => {
                let scheme = quantization.scheme;
                QuantLinear::upload(
                    runtime,
                    scheme,
                    &scheme.weight(values, out, inputs),
                    out,
                    inputs,
                )
                .map(Self::Quantized)
            }
        }
    }

    /// Uploads a linear weight as its file stores it for a GEMM at `math`: FP32 or FP16
    /// weights go through [`Linear::new`], except that FP16 weights for an FP16 GEMM
    /// upload as they are; a quantized weight uploads as it is, and
    /// [`CudaEncoder::load`] has checked that `math` uses its scheme
    fn from_stored(
        runtime: &CudaRuntime,
        kernels: &EncoderKernels,
        stored: StoredLinear,
        out: usize,
        inputs: usize,
        math: EncoderMath,
    ) -> Result<Self, TranscriptionError> {
        match (stored, math) {
            (StoredLinear::F32(values), math) => {
                Self::new(runtime, kernels, &values, out, inputs, math)
            }
            (StoredLinear::F16(bits), EncoderMath::Fp16) => {
                Ok(Self::Fp16(runtime.stream().clone_htod(bits.as_ref())?))
            }
            (StoredLinear::F16(bits), math) => {
                let values: Vec<f32> = bits.iter().map(|bits| f16_value(*bits)).collect();
                Self::new(runtime, kernels, &values, out, inputs, math)
            }
            (StoredLinear::Quantized(weight), EncoderMath::Quantized(quantization)) => {
                QuantLinear::upload(runtime, quantization.scheme, &weight, out, inputs)
                    .map(Self::Quantized)
            }
            (StoredLinear::Quantized(_), _) => Err(TranscriptionError::InvalidConfiguration(
                "a quantized linear weight needs its own precision, without FP16 layers".to_owned(),
            )),
        }
    }
}

impl QuantLinear {
    fn upload(
        runtime: &CudaRuntime,
        scheme: Scheme,
        weight: &QuantizedWeight,
        out: usize,
        inputs: usize,
    ) -> Result<Self, TranscriptionError> {
        let stream = runtime.stream();
        let (values, weight) = match weight {
            QuantizedWeight::Fp8 { values, scales }
            | QuantizedWeight::Int8 { values, scales }
            | QuantizedWeight::Int4 { values, scales, .. } => {
                (values, DeviceScales::Float(stream.clone_htod(scales)?))
            }
            QuantizedWeight::Mxfp8 { values, scales } => {
                (values, DeviceScales::Blocks(stream.clone_htod(scales)?))
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
            scheme,
            out,
            inputs,
            values: stream.clone_htod(values)?,
            weight,
        })
    }
}

impl QuantState {
    fn load(
        runtime: &CudaRuntime,
        quantization: Quantization,
        spec: EncoderSpec,
    ) -> Result<Self, TranscriptionError> {
        let scheme = quantization.scheme;
        let stream = runtime.stream();
        let kernels = QuantKernels::load(runtime)?;
        let unsupported = |format: LtFormat| {
            TranscriptionError::InvalidConfiguration(format!(
                "{scheme:?} needs cuBLASLt {format:?} products, which this GPU or cuBLAS \
                 does not run"
            ))
        };
        let (lt, scale_after) = if scheme.weight_only() {
            (None, false)
        } else {
            let lt = LtGemm::new(runtime)?;
            let scale_after = match scheme {
                Scheme::Fp8Tensor => lt.supports(runtime, LtFormat::Fp8).then_some(false),
                Scheme::Fp8Channel if lt.supports(runtime, LtFormat::Fp8OuterVector) => Some(false),
                Scheme::Fp8Channel => lt.supports(runtime, LtFormat::Fp8).then_some(true),
                Scheme::Fp8Token => lt
                    .supports(runtime, LtFormat::Fp8OuterVector)
                    .then_some(false),
                Scheme::Mxfp8 => lt.supports(runtime, LtFormat::Mxfp8).then_some(false),
                Scheme::Nvfp4 => lt.supports(runtime, LtFormat::Nvfp4).then_some(false),
                Scheme::Int8Weights { .. } | Scheme::Int4Weights { .. } => Some(false),
            }
            .ok_or_else(|| unsupported(Self::format(scheme, false)))?;
            (Some(lt), scale_after)
        };
        let largest = (spec.ff_hidden * spec.d_model).max(3 * spec.d_model * spec.d_model);
        let dequant = match scheme.weight_only() {
            true => Some(Mutex::new(zeros(stream, largest)?)),
            false => None,
        };
        Ok(Self {
            quantization,
            kernels,
            lt,
            scale_after,
            one: stream.clone_htod(&[1.0f32])?,
            dequant,
        })
    }

    /// The cuBLASLt format of a scheme's products
    fn format(scheme: Scheme, scale_after: bool) -> LtFormat {
        match scheme {
            Scheme::Fp8Channel if scale_after => LtFormat::Fp8,
            Scheme::Fp8Tensor => LtFormat::Fp8,
            Scheme::Fp8Channel | Scheme::Fp8Token => LtFormat::Fp8OuterVector,
            Scheme::Mxfp8 => LtFormat::Mxfp8,
            Scheme::Nvfp4 | Scheme::Int8Weights { .. } | Scheme::Int4Weights { .. } => {
                LtFormat::Nvfp4
            }
        }
    }

    /// The quantized-activation buffers for a workspace of `rows` encoder rows and a
    /// position table of `positions` rows, or none for a weight-only scheme
    fn inputs(
        &self,
        runtime: &CudaRuntime,
        spec: EncoderSpec,
        rows: usize,
        positions: usize,
    ) -> Result<Option<QuantInputs>, TranscriptionError> {
        let scheme = self.quantization.scheme;
        if scheme.weight_only() {
            return Ok(None);
        }
        let stream = runtime.stream();
        let (d, hidden) = (spec.d_model, spec.ff_hidden);
        // every product's activation: the normed or merged rows, the SiLU output and
        // the relative position table
        let operands = [(rows, d), (rows, hidden), (positions, d)];
        let block = match scheme {
            Scheme::Mxfp8 => MX_BLOCK,
            _ => NV_BLOCK,
        };
        let largest = |size: &dyn Fn(usize, usize) -> usize| {
            operands
                .iter()
                .map(|&(rows, width)| size(rows, width))
                .max()
                .unwrap_or(0)
        };
        let blocks = match scheme {
            Scheme::Mxfp8 | Scheme::Nvfp4 => {
                largest(&|rows, width| ScaleLayout::new(rows, width, block).bytes())
            }
            _ => 1,
        };
        let product = match self.scale_after {
            true => Some(zeros(
                stream,
                (rows * hidden.max(3 * d)).max(positions * d),
            )?),
            false => None,
        };
        Ok(Some(QuantInputs {
            values: zeros(stream, largest(&|rows, width| rows * width))?,
            scales: zeros(stream, rows.max(positions))?,
            blocks: zeros(stream, blocks)?,
            partials: zeros(stream, AMAX_PARTIALS)?,
            alpha: zeros(stream, 1)?,
            product,
        }))
    }

    /// Runs `spec` on the FP16 activation `a` and the quantized weight `w`
    fn linear(
        &self,
        runtime: &CudaRuntime,
        spec: Sgemm,
        a: &CudaView<'_, u16>,
        w: &QuantLinear,
        c: &mut CudaViewMut<'_, f32>,
        quant: Option<&mut QuantInputs>,
    ) -> Result<(), TranscriptionError> {
        if (spec.n, spec.k) != (w.out, w.inputs) {
            return Err(TranscriptionError::InvalidModelOutput(format!(
                "a {} x {} product reads a {} x {} weight",
                spec.n, spec.k, w.out, w.inputs
            )));
        }
        let stream = runtime.stream();
        if let Some(dequant) = &self.dequant {
            let mut weight = dequant.lock().map_err(|_| {
                TranscriptionError::InvalidModelOutput(
                    "the dequantize lock was poisoned".to_owned(),
                )
            })?;
            let mut weight = weight.slice_mut(..w.out * w.inputs);
            let DeviceScales::Float(scales) = &w.weight else {
                return Err(mixed_precision());
            };
            match w.scheme {
                Scheme::Int4Weights { group } => self.kernels.dequant_i4(
                    stream,
                    &w.values.slice(..),
                    &scales.slice(..),
                    &mut weight,
                    group,
                )?,
                _ => self.kernels.dequant_i8(
                    stream,
                    &w.values.slice(..),
                    &scales.slice(..),
                    &mut weight,
                    w.out * w.inputs / scales.len(),
                )?,
            }
            linear_fp16(runtime, spec, a, &weight, c)?;
            return Ok(());
        }

        let (Some(lt), Some(quant)) = (&self.lt, quant) else {
            return Err(mixed_precision());
        };
        let (rows, width) = (spec.m, spec.k);
        let n = rows * width;
        let kernels = &self.kernels;
        let mut values = quant.values.slice_mut(..n);
        let mut alpha = Alpha::Host(spec.alpha);
        let mut beta = spec.beta;
        // the activation scale a product reads: one, or one per row
        let mut a_scales = 1;
        match w.scheme {
            Scheme::Fp8Tensor | Scheme::Fp8Channel => {
                kernels.amax(stream, a, &mut quant.partials.slice_mut(..))?;
                if w.scheme == Scheme::Fp8Channel && !self.scale_after {
                    a_scales = rows;
                }
                kernels.fp8(
                    stream,
                    a,
                    &quant.partials.slice(..),
                    &mut values,
                    &mut quant.scales.slice_mut(..a_scales),
                )?;
            }
            Scheme::Fp8Token => {
                a_scales = rows;
                kernels.fp8_rows(
                    stream,
                    a,
                    &mut values,
                    &mut quant.scales.slice_mut(..rows),
                    width,
                )?;
            }
            Scheme::Mxfp8 => {
                let layout = ScaleLayout::new(rows, width, MX_BLOCK);
                kernels.mxfp8(
                    stream,
                    a,
                    &mut values,
                    &mut quant.blocks.slice_mut(..layout.bytes()),
                    width,
                )?;
            }
            Scheme::Nvfp4 => {
                let DeviceScales::Nvfp4 { global, .. } = &w.weight else {
                    return Err(mixed_precision());
                };
                let layout = ScaleLayout::new(rows, width, NV_BLOCK);
                kernels.amax(stream, a, &mut quant.partials.slice_mut(..))?;
                kernels.nvfp4(
                    stream,
                    a,
                    &quant.partials.slice(..),
                    &mut values.slice_mut(..n / 2),
                    &mut quant.blocks.slice_mut(..layout.bytes()),
                    &mut quant.alpha.slice_mut(..),
                    width,
                    spec.alpha * global,
                )?;
                let (alpha_ptr, _alpha_sync) = quant.alpha.device_ptr(stream);
                alpha = Alpha::Device(alpha_ptr);
            }
            Scheme::Int8Weights { .. } | Scheme::Int4Weights { .. } => {
                return Err(mixed_precision());
            }
        }

        let format = Self::format(w.scheme, self.scale_after);
        let (w_ptr, _w_sync) = w.values.device_ptr(stream);
        let (w_scale, _w_scale_sync) = match (&w.weight, self.scale_after) {
            (DeviceScales::Float(_), true) => self.one.device_ptr(stream),
            (DeviceScales::Float(scales), false) => scales.device_ptr(stream),
            (DeviceScales::Blocks(blocks) | DeviceScales::Nvfp4 { blocks, .. }, _) => {
                blocks.device_ptr(stream)
            }
        };
        let (a_ptr, _a_sync) = quant.values.device_ptr(stream);
        let (a_scale, _a_scale_sync) = match format {
            LtFormat::Mxfp8 | LtFormat::Nvfp4 => quant.blocks.device_ptr(stream),
            LtFormat::Fp8 | LtFormat::Fp8OuterVector => quant.scales.device_ptr(stream),
        };
        debug_assert!(a_scales <= quant.scales.len());
        let scaled_after = self.scale_after && w.scheme == Scheme::Fp8Channel;
        let (c_ptr, _c_sync) = match (scaled_after, quant.product.as_mut()) {
            (true, Some(product)) => {
                alpha = Alpha::Host(1.0);
                beta = 0.0;
                product.device_ptr_mut(stream)
            }
            (true, None) => return Err(mixed_precision()),
            (false, _) => c.device_ptr_mut(stream),
        };
        // SAFETY: the operands were sized and written above for `rows x width`, the
        // weight holds `out x inputs` in the scheme's format with matching scales, and
        // `c` holds `rows x out`; all are ordered on the runtime's stream
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
                    beta,
                },
            )
        }?;
        drop(_c_sync);

        if scaled_after {
            let (Some(product), DeviceScales::Float(scales)) = (&quant.product, &w.weight) else {
                return Err(mixed_precision());
            };
            kernels.scale_columns(
                stream,
                &product.slice(..rows * w.out),
                &scales.slice(..),
                c,
                spec.alpha,
                spec.beta,
            )?;
        }
        Ok(())
    }
}

/// A workspace or weight in the wrong precision for the encoder's math, which
/// [`CudaEncoder::load`] and [`CudaEncoder::workspace`] never build
fn mixed_precision() -> TranscriptionError {
    TranscriptionError::InvalidModelOutput(
        "an encoder GEMM's operands differ in precision".to_owned(),
    )
}

/// NeMo's `RelPositionalEncoding` rows for positions `frames - 1` down to
/// `-(frames - 1)`, `[2 * frames - 1, d_model]`, computed in FP32 as NeMo does
fn relative_positions(frames: usize, d_model: usize) -> Vec<f32> {
    let scale = -(10000f64.ln() / d_model as f64) as f32;
    let div_term: Vec<f32> = (0..d_model)
        .step_by(2)
        .map(|i| (i as f32 * scale).exp())
        .collect();
    let mut table = Vec::with_capacity((2 * frames - 1) * d_model);
    for row in 0..2 * frames - 1 {
        let position = (frames - 1) as f32 - row as f32;
        for term in &div_term {
            let angle = position * term;
            table.push(angle.sin());
            table.push(angle.cos());
        }
    }
    table
}

#[cfg(test)]
mod tests {
    use cudakit::{Library, Math, Options};
    use ndarray::{ArrayD, Axis, Ix3};

    use super::{
        CudaEncoder, DeviceScales, EncoderMath, KeepFp16, Linear, LinearKind, QuantLinear,
        QuantState, Quantization, relative_positions,
    };
    use crate::cuda::kernels::CudaRuntime;
    use crate::cuda::linear;
    use crate::cuda::quant::test_support::e4m3_value;
    use crate::cuda::quant::{
        QuantizedWeight, Scheme, f16_bits, f16_value, fp8_rows, fp8_tensor, mxfp8, nvfp4,
    };
    use crate::cuda::test_support::weights;
    use crate::deployment::{ParakeetDeployment, ParakeetVersion};
    use crate::test_support::NemoReference;

    impl LinearKind {
        const ALL: [Self; 7] = [
            Self::FeedForwardUp,
            Self::FeedForwardDown,
            Self::Qkv,
            Self::Position,
            Self::AttentionOut,
            Self::Pointwise1,
            Self::Pointwise2,
        ];
    }

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
                + match &self.weight {
                    DeviceScales::Float(scales) => scales.num_bytes(),
                    DeviceScales::Blocks(blocks) | DeviceScales::Nvfp4 { blocks, .. } => {
                        blocks.num_bytes()
                    }
                }
        }
    }

    /// Every quantization scheme, as the quantized precisions name them
    const SCHEMES: [(&str, Scheme); 9] = [
        ("FP8", Scheme::Fp8Channel),
        ("FP8 per tensor", Scheme::Fp8Tensor),
        ("FP8 per token", Scheme::Fp8Token),
        ("MXFP8", Scheme::Mxfp8),
        ("NVFP4", Scheme::Nvfp4),
        ("INT8 weights", Scheme::Int8Weights { group: None }),
        (
            "INT8 weights g128",
            Scheme::Int8Weights { group: Some(128) },
        ),
        ("INT4 weights g128", Scheme::Int4Weights { group: 128 }),
        ("INT4 weights g64", Scheme::Int4Weights { group: 64 }),
    ];

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
            Scheme::Fp8Tensor | Scheme::Fp8Channel => {
                let (bits, scale) = fp8_tensor(values);
                let restored = bits.iter().map(|bits| e4m3_value(*bits) * scale).collect();
                (bits, restored)
            }
            Scheme::Fp8Token => {
                let (bits, scales) = fp8_rows(values, width);
                let restored = bits
                    .iter()
                    .enumerate()
                    .map(|(i, bits)| e4m3_value(*bits) * scales[i / width])
                    .collect();
                (bits, restored)
            }
            Scheme::Mxfp8 => {
                let (values, scales) = mxfp8(values, width);
                let restored = QuantizedWeight::Mxfp8 {
                    values: values.clone(),
                    scales,
                }
                .dequantize(rows, width);
                (values, restored)
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
            Scheme::Int8Weights { .. } | Scheme::Int4Weights { .. } => {
                (Vec::new(), values.to_vec())
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
            let quant = match QuantState::load(&runtime, Quantization::all(scheme), spec) {
                Ok(quant) => quant,
                Err(error) => {
                    println!("{name}: unsupported here: {error}");
                    continue;
                }
            };
            println!(
                "{name}: per-channel scales after the product: {}",
                quant.scale_after
            );
            for (tensor, out, inputs, alpha, beta) in products {
                let weight = file.tensor(tensor, &[out, inputs]).expect("weight");
                let quantized = scheme.weight(weight, out, inputs);
                let device =
                    QuantLinear::upload(&runtime, scheme, &quantized, out, inputs).expect("upload");
                let mut restored = quantized.dequantize(out, inputs);
                if scheme.weight_only() {
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
                            &mut c.slice_mut(..),
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
            let deployment = match version {
                ParakeetVersion::V2 => ParakeetDeployment::v2(),
                ParakeetVersion::V3 => ParakeetDeployment::v3(),
            };
            let file = weights(version);
            let spec = file.parakeet_spec(&deployment).expect("spec").encoder;
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
                let math = EncoderMath::Quantized(Quantization::all(scheme));
                let encoder = match CudaEncoder::load(&runtime, &file, spec, math) {
                    Ok(encoder) => encoder,
                    Err(error) => {
                        println!("{version:?} {name}: unsupported here: {error}");
                        continue;
                    }
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

    /// For each scheme, quantizes one layer, then one kind of linear layer, at a time
    /// with the rest at FP16, and prints the v3 encoder output's distance from the
    /// FP16 encoder's: the most sensitive layers are the ones a mixed recipe keeps at
    /// FP16
    #[test]
    #[ignore = "needs a CUDA GPU and the NeMo references; run on gpu0 when choosing a mixed recipe"]
    fn gpu_quantization_sensitivity_by_layer() {
        let runtime = CudaRuntime::open(0, Options::default()).expect("CUDA runtime");
        runtime.prepare(Library::Cublas).expect("cuBLAS");
        let file = weights(ParakeetVersion::V3);
        let spec = file
            .parakeet_spec(&ParakeetDeployment::v3())
            .expect("spec")
            .encoder;
        let inputs: Vec<_> = ["b8", "long-full"]
            .into_iter()
            .map(|set| NemoInput::open(ParakeetVersion::V3, set))
            .collect();
        let fp16 = CudaEncoder::load(&runtime, &file, spec, EncoderMath::Fp16).expect("fp16");
        let baseline: Vec<_> = inputs
            .iter()
            .map(|input| input.encode(&runtime, &fp16).0)
            .collect();
        drop(fp16);
        let all_layers = (1u64 << spec.layers) - 1;
        let all_kinds = (1u8 << LinearKind::ALL.len()) - 1;
        let measure = |scheme: Scheme, keep: KeepFp16| -> Option<f64> {
            let math = EncoderMath::Quantized(Quantization { scheme, keep });
            let encoder = CudaEncoder::load(&runtime, &file, spec, math).ok()?;
            let mut worst = 0f64;
            for (input, fp16) in inputs.iter().zip(&baseline) {
                let (output, lens) = input.encode(&runtime, &encoder);
                let frames = input.expected.shape()[1];
                worst = worst.max(distance(&output, fp16, frames, spec.d_model, &lens));
            }
            Some(worst)
        };

        for (name, scheme) in SCHEMES {
            let Some(full) = measure(scheme, KeepFp16::NONE) else {
                println!("{name}: unsupported here");
                continue;
            };
            println!("{name} all layers: {full:.3e}");
            for layer in 0..spec.layers {
                let keep = KeepFp16 {
                    layers: all_layers & !(1 << layer),
                    kinds: 0,
                };
                let error = measure(scheme, keep).expect("loaded once");
                println!("{name} layer {layer:2} only: {error:.3e}");
            }
            for (bit, kind) in LinearKind::ALL.iter().enumerate() {
                let keep = KeepFp16 {
                    layers: 0,
                    kinds: all_kinds & !(1 << bit),
                };
                let error = measure(scheme, keep).expect("loaded once");
                println!("{name} {kind:?} only: {error:.3e}");
            }
        }
    }

    /// Relative L2 error and max abs difference over the valid frames of `[batch,
    /// frames, d_model]` outputs
    ///
    /// Panics on a non-finite value in either side, which `f64::max` would skip
    fn compare(ours: &[f32], reference: &ArrayD<f32>, lens: &[usize]) -> (f64, f64) {
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

    #[test]
    #[should_panic(expected = "row 0 frame 1")]
    fn compare_rejects_nan_output() {
        let reference = ArrayD::from_elem(vec![1, 2, 2], 1.0f32);
        compare(&[1.0, 1.0, f32::NAN, 1.0], &reference, &[2]);
    }

    #[test]
    #[ignore = "reads NeMo references from /shared/datasets/parakeet-ref"]
    fn nemo_parity_relative_positions() {
        let reference = NemoReference::open(ParakeetVersion::V2, "b8");
        let table = reference.f32("encoder.layers.0.self_attn.linear_pos.in");
        let positions = table.shape()[1];
        let ours = relative_positions(positions.div_ceil(2), table.shape()[2]);
        assert_eq!(ours.len(), table.len(), "table size");
        assert!(
            ours.iter()
                .chain(table.iter())
                .all(|value| value.is_finite()),
            "non-finite relative position"
        );
        let max_abs = ours
            .iter()
            .zip(table.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        // sin and cos of arguments up to a few thousand radians differ by an ulp or so
        assert!(
            max_abs < 1e-3,
            "relative position table differs by {max_abs}"
        );
    }

    /// Runs the encoder stage by stage and holds every stage against NeMo FP32
    ///
    /// Needs a CUDA GPU: run it through the shared queue with `just gpu-parity`
    #[test]
    #[ignore = "needs a CUDA GPU and the NeMo references; `just gpu-parity` runs it on gpu0"]
    fn gpu_encoder_matches_nemo_layer_by_layer() {
        let runtime = CudaRuntime::open(0, Options::default()).expect("CUDA runtime");
        runtime.prepare(Library::Cublas).expect("cuBLAS");
        let mut worst = 0f64;
        for version in [ParakeetVersion::V2, ParakeetVersion::V3] {
            let deployment = match version {
                ParakeetVersion::V2 => ParakeetDeployment::v2(),
                ParakeetVersion::V3 => ParakeetDeployment::v3(),
            };
            let file = weights(version);
            let spec = file.parakeet_spec(&deployment).expect("spec");
            let encoder = CudaEncoder::load(
                &runtime,
                &file,
                spec.encoder,
                EncoderMath::Single(Math::Fp32),
            )
            .expect("encoder");

            for set in ["b1", "b8", "long-full"] {
                let reference = NemoReference::open(version, set);
                let mel = reference
                    .f32("mel")
                    .into_dimensionality::<Ix3>()
                    .expect("[batch, features, frames]");
                let (batch, mel_frames) = (mel.shape()[0], mel.shape()[2]);
                let lens: Vec<usize> = reference
                    .i64("mel_len")
                    .iter()
                    .map(|len| *len as usize)
                    .collect();
                // NeMo's featurizer is feature-major; the encoder takes time-major rows
                let time_major: Vec<f32> = mel.permuted_axes([0, 2, 1]).iter().copied().collect();

                let mut ws = encoder
                    .workspace(&runtime, batch, mel_frames)
                    .expect("workspace");
                encoder
                    .set_input(&runtime, &mut ws, &lens, |mel| {
                        mel.copy_from_slice(&time_major)
                    })
                    .expect("input");
                let out_lens: Vec<usize> = reference
                    .i64("encoder.out_len")
                    .iter()
                    .map(|len| *len as usize)
                    .collect();
                assert_eq!(
                    ws.output_lens(),
                    out_lens.as_slice(),
                    "{version:?} {set} encoder lengths"
                );

                let download = |ws: &super::EncoderWorkspace| {
                    runtime.stream().clone_dtoh(ws.output()).expect("download")
                };
                encoder
                    .enqueue_subsampling(&runtime, &mut ws)
                    .expect("subsampling");
                let (rel, max_abs) = compare(
                    &download(&ws),
                    &reference.f32("encoder.pre_encode.out"),
                    &out_lens,
                );
                println!("{version:?} {set} pre_encode rel {rel:.2e} max {max_abs:.2e}");
                for index in 0..spec.encoder.layers {
                    encoder
                        .enqueue_layer(&runtime, &mut ws, index)
                        .expect("layer");
                    let (rel, max_abs) = compare(
                        &download(&ws),
                        &reference.f32(&format!("encoder.layers.{index}.out")),
                        &out_lens,
                    );
                    println!("{version:?} {set} layer {index:2} rel {rel:.2e} max {max_abs:.2e}");
                }

                let expected = reference.f32("encoder.out").permuted_axes(vec![0, 2, 1]);
                let (rel, max_abs) = compare(
                    &download(&ws),
                    &expected.as_standard_layout().into_owned(),
                    &out_lens,
                );
                println!("{version:?} {set} encoder.out rel {rel:.2e} max {max_abs:.2e}");
                worst = worst.max(rel);
            }
        }

        // NeMo FP32 with TF32 off against cuBLAS FP32: summation order differs only
        assert!(worst < 1e-4, "encoder output relative L2 {worst:.2e}");
    }

    /// Runs the TF32 and FP16 encoders end to end and holds each near NeMo FP32 and
    /// finite, which an FP16 overflow would break. Each output must differ from
    /// the FP32 encoder's, or its GEMMs never left the FP32 path
    #[test]
    #[ignore = "needs a CUDA GPU and the NeMo references; `just gpu-parity` runs it on gpu0"]
    fn gpu_fast_encoders_stay_near_nemo() {
        // a 10-bit mantissa per product through 24 layers lands at 6e-4 to 1.3e-2 in
        // TF32 and 7e-4 to 6e-3 in FP16 on an RTX 5090; a broken GEMM is off by order
        // 1. WER is the accuracy gate
        let fast = [
            ("TF32", EncoderMath::Single(Math::Tf32)),
            ("FP16", EncoderMath::Fp16),
        ];
        let runtime = CudaRuntime::open(0, Options::default()).expect("CUDA runtime");
        runtime.prepare(Library::Cublas).expect("cuBLAS");
        let mut worst = [0f64; 2];
        for version in [ParakeetVersion::V2, ParakeetVersion::V3] {
            let deployment = match version {
                ParakeetVersion::V2 => ParakeetDeployment::v2(),
                ParakeetVersion::V3 => ParakeetDeployment::v3(),
            };
            let file = weights(version);
            let spec = file.parakeet_spec(&deployment).expect("spec");
            let load =
                |math| CudaEncoder::load(&runtime, &file, spec.encoder, math).expect("encoder");
            let fp32 = load(EncoderMath::Single(Math::Fp32));

            for set in ["b1", "b8", "long-full"] {
                let reference = NemoReference::open(version, set);
                let mel = reference
                    .f32("mel")
                    .into_dimensionality::<Ix3>()
                    .expect("[batch, features, frames]");
                let (batch, mel_frames) = (mel.shape()[0], mel.shape()[2]);
                let lens: Vec<usize> = reference
                    .i64("mel_len")
                    .iter()
                    .map(|len| *len as usize)
                    .collect();
                let time_major: Vec<f32> = mel.permuted_axes([0, 2, 1]).iter().copied().collect();
                let encode = |encoder: &CudaEncoder| {
                    let mut ws = encoder
                        .workspace(&runtime, batch, mel_frames)
                        .expect("workspace");
                    encoder
                        .set_input(&runtime, &mut ws, &lens, |mel| {
                            mel.copy_from_slice(&time_major)
                        })
                        .expect("input");
                    encoder.enqueue(&runtime, &mut ws).expect("encode");
                    let output = runtime.stream().clone_dtoh(ws.output()).expect("download");
                    (output, ws.output_lens().to_vec())
                };
                let (full, out_lens) = encode(&fp32);
                let expected = reference.f32("encoder.out").permuted_axes(vec![0, 2, 1]);
                let expected = expected.as_standard_layout().into_owned();
                for (index, (name, math)) in fast.iter().enumerate() {
                    // loaded per set so only one extra copy of the weights is resident
                    let (output, _) = encode(&load(*math));
                    assert!(
                        full != output,
                        "{version:?} {set}: {name} matches FP32 bit for bit"
                    );
                    let (rel, max_abs) = compare(&output, &expected, &out_lens);
                    println!(
                        "{version:?} {set} {name} encoder.out rel {rel:.2e} max {max_abs:.2e}"
                    );
                    worst[index] = worst[index].max(rel);
                }
            }
        }

        for ((name, _), worst) in fast.iter().zip(worst) {
            assert!(
                worst < 5e-2,
                "{name} encoder output relative L2 {worst:.2e}"
            );
        }
    }

    /// Captures the encoder in a CUDA graph at each precision, checks the replay
    /// against the eager pass bit for bit, and prints replay times for the v3
    /// reference shapes
    #[test]
    #[ignore = "needs a CUDA GPU and the NeMo references; `just gpu-parity` runs it on gpu0"]
    fn gpu_encoder_graph_replays_the_eager_pass() {
        let runtime = CudaRuntime::open(0, Options::default()).expect("CUDA runtime");
        runtime.prepare(Library::Cublas).expect("cuBLAS");
        let file = weights(ParakeetVersion::V3);
        let spec = file.parakeet_spec(&ParakeetDeployment::v3()).expect("spec");
        let precisions = [
            ("FP32", EncoderMath::Single(Math::Fp32)),
            ("TF32", EncoderMath::Single(Math::Tf32)),
            ("FP16", EncoderMath::Fp16),
        ]
        .into_iter()
        .chain(
            SCHEMES
                .iter()
                .map(|(name, scheme)| (*name, EncoderMath::Quantized(Quantization::all(*scheme)))),
        );

        for (name, math) in precisions {
            // a quantized precision this GPU does not run fails to load
            let Ok(encoder) = CudaEncoder::load(&runtime, &file, spec.encoder, math) else {
                println!("v3 {name}: unsupported here");
                continue;
            };
            for set in ["b1", "b8"] {
                time_graph(&runtime, &encoder, name, set);
            }
        }
    }

    /// Checks one graph replay of `encoder` on the v3 reference `set` against its
    /// eager pass and prints the mean replay time
    fn time_graph(runtime: &CudaRuntime, encoder: &CudaEncoder, name: &str, set: &str) {
        const REPLAYS: u32 = 100;
        let reference = NemoReference::open(ParakeetVersion::V3, set);
        let mel = reference
            .f32("mel")
            .into_dimensionality::<Ix3>()
            .expect("[batch, features, frames]");
        let (batch, mel_frames) = (mel.shape()[0], mel.shape()[2]);
        let lens: Vec<usize> = reference
            .i64("mel_len")
            .iter()
            .map(|len| *len as usize)
            .collect();
        let time_major: Vec<f32> = mel.permuted_axes([0, 2, 1]).iter().copied().collect();
        let mut ws = encoder
            .workspace(runtime, batch, mel_frames)
            .expect("workspace");
        encoder
            .set_input(runtime, &mut ws, &lens, |mel| {
                mel.copy_from_slice(&time_major)
            })
            .expect("input");

        // the eager pass also finishes cuBLAS's lazy setup before capture
        encoder.enqueue(runtime, &mut ws).expect("eager");
        let eager = runtime.stream().clone_dtoh(ws.output()).expect("download");
        // SAFETY: the graph is declared after `ws` and `encoder` outlives this call, so
        // it drops before the buffers it points at, which stay in place on the
        // runtime's stream
        let graph = unsafe { runtime.capture_nonempty(|| encoder.enqueue(runtime, &mut ws)) }
            .expect("capture");
        graph.launch().expect("replay");
        let replayed = runtime.stream().clone_dtoh(ws.output()).expect("download");
        assert!(
            eager == replayed,
            "v3 {set} {name}: graph replay differs from the eager pass"
        );

        let start = std::time::Instant::now();
        for _ in 0..REPLAYS {
            graph.launch().expect("replay");
        }
        runtime.synchronize().expect("sync");
        let replay_ms = start.elapsed().as_secs_f64() * 1e3 / f64::from(REPLAYS);
        let audio_s: f64 = reference
            .i64("audio_len")
            .iter()
            .map(|len| *len as f64 / 16000.0)
            .sum();
        println!(
            "v3 {set} {name}: batch {batch}, {audio_s:.1} s audio, encoder graph replay {replay_ms:.3} ms"
        );
    }
}
