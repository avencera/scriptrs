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
//! With the `low-precision` feature, `EncoderMath::Quantized` keeps the FP16
//! activations and quantizes the same linear layers further (see the `quantized`
//! module). The streaming encoder does not run it
//!
//! On `sm_80` and newer, the FP16 and TF32 encoders run self-attention as one fused
//! kernel ([`AttentionKernels`]) for the input shapes [`FusedShapes`] picks: the kernel reads the QKV and position projections and computes the scores,
//! the relative-position term, the softmax and the value product tile by tile, so the
//! `frames x frames` score matrices are never stored. Under FP16 the projections and
//! the merged context are FP16 too. The FP32 encoder keeps the cuBLAS attention, as its
//! FP32 products are what the parity tests hold every other path against, and so does
//! the quantized one, whose projections the kernel cannot read

use cudakit::cudarc::driver::{CudaSlice, CudaView, DevicePtrMut};
use cudakit::{DeviceTensor, Math, Sgemm, SgemmBatched};

use crate::cuda::fp16::{linear_fp16, linear_fp16_to_fp16};
use crate::cuda::kernels::{
    ATTENTION_HEAD_DIM, AttentionKernels, CudaRuntime, EncoderKernels, FeatureMap, GemmOperand,
    Heads, StreamKernels,
};
#[cfg(feature = "low-precision")]
use crate::cuda::quant::Scheme;
use crate::cuda::{Direction, HostStaging, linear, upload, upload_host, zeros};
use crate::error::TranscriptionError;
use crate::native::spec::{ConvNorm, EncoderSpec};
use crate::native::weights::WeightFile;

#[cfg(feature = "low-precision")]
mod quantized;
mod stream;

pub(crate) use stream::{
    ChunkRow, MEL_CONTEXT, StreamCaches, StreamWorkspace, chunk_mel, first_chunk_mel,
};

/// PyTorch's default `eps` for `LayerNorm` and `BatchNorm1d`, which NeMo keeps
const NORM_EPS: f32 = 1e-5;

/// Input shapes the FP16 encoder runs the fused attention on: on an RTX 5090 it is 3
/// to 8 % faster than cuBLAS up to 50 frames at any batch, within 1.5 % either way at
/// 100, and 5 to 19 % faster from 640 rows, but up to 15 % slower on a single input of
/// 376 or more frames
const FUSED_FP16: FusedShapes = FusedShapes {
    short_frames: 96,
    min_rows: 640,
};

/// Input shapes the TF32 encoder runs the fused attention on: its FP32 tiles fill
/// shared memory twice as fast, so it needs about twice the rows, and from 50 frames it
/// is up to 35 % slower than cuBLAS on a single input
const FUSED_TF32: FusedShapes = FusedShapes {
    short_frames: 32,
    min_rows: 1200,
};

/// Every `dw_striding` convolution is 3x3 with stride 2 and padding 1
fn subsampled(len: usize) -> usize {
    // NeMo's `calculate_conv_output_size`: (len + 1 + 1 - 3) // 2 + 1
    (len + 2).saturating_sub(3) / 2 + 1
}

/// A causal `dw_striding` convolution pads two frames before and one after instead
fn causal_subsampled(len: usize) -> usize {
    // (len + 2 + 1 - 3) // 2 + 1
    len / 2 + 1
}

/// Features left after the three stride-2 convolutions
fn subsampled_features(spec: EncoderSpec) -> usize {
    let stage = if spec.causal {
        causal_subsampled
    } else {
        subsampled
    };
    stage(stage(stage(spec.features)))
}

/// Precision of the encoder's GEMMs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EncoderMath {
    /// Every GEMM on FP32 operands in this cuBLAS math
    Single(Math),
    /// The Conformer layers' linear layers on FP16 operands with FP32 accumulation and
    /// FP32 output, except for the QKV and position projections the fused attention
    /// reads, which it takes in FP16 as it writes its context. Without the fused
    /// attention the score and mix products keep FP32 operands in TF32, as the
    /// subsampling module does: they are a few percent of the work, and their operands
    /// would need conversions of their own
    Fp16,
    /// [`Self::Fp16`] with the linear layers' weights, and for every scheme but INT8
    /// their activations, quantized in the scheme
    #[cfg(feature = "low-precision")]
    Quantized(Scheme),
}

impl EncoderMath {
    /// The math of the GEMMs that keep FP32 operands
    fn single(self) -> Math {
        match self {
            Self::Single(math) => math,
            Self::Fp16 => Math::Tf32,
            #[cfg(feature = "low-precision")]
            Self::Quantized(_) => Math::Tf32,
        }
    }

    /// Whether the kernels write FP16 copies of the linear layers' inputs
    fn fp16(self) -> bool {
        !matches!(self, Self::Single(_))
    }
}

/// A linear layer's `[out, in]` weight, in the precision its GEMM reads
enum Linear {
    Single(DeviceTensor),
    /// FP16 bits
    Fp16(CudaSlice<u16>),
    #[cfg(feature = "low-precision")]
    Quantized(quantized::QuantLinear),
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
    /// The eval-mode BatchNorm folded to `x * scale + shift`, or the LayerNorm's
    /// `gamma` and `beta` under [`ConvNorm::Layer`]
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
    stream_kernels: StreamKernels,
    #[cfg(feature = "low-precision")]
    quant: Option<quantized::QuantState>,
    /// The fused attention, for the FP16 and TF32 encoders on a GPU the kernels ship
    /// for
    fused: Option<FusedAttention>,
}

/// The fused attention kernel and the inputs it takes over
struct FusedAttention {
    kernels: AttentionKernels,
    shapes: FusedShapes,
}

/// Input shapes, by batch rows and encoder frames, that run the fused attention; the
/// rest keep the cuBLAS path
///
/// Short inputs gain from the launches and tiny GEMMs the kernel replaces. Longer ones
/// need many rows: the kernel runs `batch * heads * frames / 16` warps, and one block
/// takes most of an sm_120 SM's shared memory, so a few long rows leave the SMs a
/// warp or two each, while cuBLAS still fills them
#[derive(Clone, Copy, Debug)]
struct FusedShapes {
    /// Longest input, in encoder frames, that runs it at any batch
    short_frames: usize,
    /// Fewest `batch * frames` rows that run it at any length
    min_rows: usize,
}

impl FusedShapes {
    fn takes(self, batch: usize, frames: usize) -> bool {
        frames <= self.short_frames || batch * frames >= self.min_rows
    }
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
    act: Activations,
    /// `[2 * frames - 1, d_model]` relative position table
    pos_emb: CudaSlice<f32>,
    /// FP16 copy of `pos_emb` under [`EncoderMath::Fp16`]
    pos_emb_f16: Option<CudaSlice<u16>>,
    attention: AttentionBuffers,
}

/// Self-attention buffers for the path [`CudaEncoder::workspace`] picked, beside the
/// QKV projection in [`Activations`]
enum AttentionBuffers {
    /// The head split, cuBLAS score and value products, rel-shift softmax and head
    /// merge
    Unfused(Box<UnfusedAttention>),
    /// The FP16 kernel's operands
    FusedFp16 {
        /// `[batch * frames, 3 * d_model]` QKV projection in FP16, which the kernel
        /// reads in place of the FP32 one the streaming encoder shares
        qkv: CudaSlice<u16>,
        /// `[2 * frames - 1, d_model]` projected position table
        pos: CudaSlice<u16>,
    },
    /// The TF32 kernel's `[2 * frames - 1, d_model]` projected position table
    FusedTf32 { pos: CudaSlice<f32> },
}

struct UnfusedAttention {
    /// `[batch, heads, frames, head_dim]`, `q_v` `[heads, batch, frames, head_dim]`
    q_u: CudaSlice<f32>,
    q_v: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    /// `[2 * frames - 1, d_model]` projected position table, and `[heads, 2 * frames -
    /// 1, head_dim]`
    pos: CudaSlice<f32>,
    pos_heads: CudaSlice<f32>,
    /// `[batch, heads, frames, frames]` content scores, then weights
    scores: CudaSlice<f32>,
    /// `[heads, batch, frames, 2 * frames - 1]` position scores
    bd: CudaSlice<f32>,
    /// `[batch, heads, frames, head_dim]`
    context: CudaSlice<f32>,
}

impl AttentionBuffers {
    fn bytes(&self) -> usize {
        match self {
            Self::Unfused(buffers) => [
                &buffers.q_u,
                &buffers.q_v,
                &buffers.k,
                &buffers.v,
                &buffers.pos,
                &buffers.pos_heads,
                &buffers.scores,
                &buffers.bd,
                &buffers.context,
            ]
            .iter()
            .map(|buffer| buffer.num_bytes())
            .sum(),
            Self::FusedFp16 { qkv, pos } => qkv.num_bytes() + pos.num_bytes(),
            Self::FusedTf32 { pos } => pos.num_bytes(),
        }
    }
}

/// The residual stream and the activations every layer's sublayers pass between
/// their GEMMs, `rows` frames each
struct Activations {
    rows: usize,
    /// The residual stream, `[rows, d_model]`
    x: CudaSlice<f32>,
    normed: CudaSlice<f32>,
    /// `[rows, max(ff_hidden, 2 * d_model)]`: the feed-forward hidden layer or the
    /// conv module's first pointwise output
    hidden: CudaSlice<f32>,
    qkv: CudaSlice<f32>,
    /// `[rows, d_model]` attention context or conv module output
    merged: CudaSlice<f32>,
    /// FP16 GEMM inputs under [`EncoderMath::Fp16`]
    fp16: Option<Fp16Inputs>,
    quant: QuantScratch,
}

/// The buffers the quantized linear layers quantize their activations into, empty
/// without the `low-precision` feature and in a streaming workspace
#[derive(Default)]
struct QuantScratch {
    #[cfg(feature = "low-precision")]
    inputs: Option<quantized::QuantInputs>,
}

impl QuantScratch {
    fn bytes(&self) -> usize {
        #[cfg(feature = "low-precision")]
        if let Some(inputs) = &self.inputs {
            return inputs.bytes();
        }
        0
    }
}

/// FP16 copies of the activations the FP16 linear layers read
struct Fp16Inputs {
    /// `[rows, d_model]` LayerNorm output
    normed: CudaSlice<u16>,
    /// `[rows, ff_hidden]` SiLU output
    hidden: CudaSlice<u16>,
    /// `[rows, d_model]` attention context or conv module output
    merged: CudaSlice<u16>,
}

impl Activations {
    fn new(
        runtime: &CudaRuntime,
        spec: EncoderSpec,
        math: EncoderMath,
        rows: usize,
    ) -> Result<Self, TranscriptionError> {
        let stream = runtime.stream();
        let d = spec.d_model;
        // the quantized maths also read FP16 activations
        let fp16 = match math.fp16() {
            false => None,
            true => Some(Fp16Inputs {
                normed: zeros(stream, rows * d)?,
                hidden: zeros(stream, rows * spec.ff_hidden)?,
                merged: zeros(stream, rows * d)?,
            }),
        };
        Ok(Self {
            rows,
            x: zeros(stream, rows * d)?,
            normed: zeros(stream, rows * d)?,
            hidden: zeros(stream, rows * spec.ff_hidden.max(2 * d))?,
            qkv: zeros(stream, rows * 3 * d)?,
            merged: zeros(stream, rows * d)?,
            fp16,
            quant: QuantScratch::default(),
        })
    }

    fn bytes(&self) -> usize {
        let single = [&self.x, &self.normed, &self.hidden, &self.qkv, &self.merged]
            .iter()
            .map(|buffer| buffer.num_bytes())
            .sum::<usize>();
        let fp16 = self.fp16.as_ref().map_or(0, |fp16| {
            [&fp16.normed, &fp16.hidden, &fp16.merged]
                .iter()
                .map(|buffer| buffer.num_bytes())
                .sum()
        });
        single + fp16 + self.quant.bytes()
    }

    /// Where a kernel writes `merged` for the next linear layer
    fn merged_out(&mut self, math: EncoderMath) -> Result<GemmOperand<'_>, TranscriptionError> {
        gemm_output(
            math,
            &mut self.merged,
            self.fp16.as_mut().map(|fp16| &mut fp16.merged),
        )
    }
}

/// `normed` as the next linear layer reads it, borrowing only those fields so the
/// layer can write another
fn normed_in<'a>(
    math: EncoderMath,
    normed: &'a CudaSlice<f32>,
    fp16: Option<&'a Fp16Inputs>,
) -> Result<Operand<'a>, TranscriptionError> {
    gemm_input(
        math,
        normed.slice(..),
        fp16.map(|fp16| fp16.normed.slice(..)),
    )
}

/// `merged` as the next linear layer reads it, as [`normed_in`]
fn merged_in<'a>(
    math: EncoderMath,
    merged: &'a CudaSlice<f32>,
    fp16: Option<&'a Fp16Inputs>,
) -> Result<Operand<'a>, TranscriptionError> {
    gemm_input(
        math,
        merged.slice(..),
        fp16.map(|fp16| fp16.merged.slice(..)),
    )
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
        &self.act.x
    }

    /// Device bytes held by the workspace's buffers
    pub(crate) fn bytes(&self) -> usize {
        let lens: usize = self.lens.iter().map(CudaSlice::num_bytes).sum();
        let buffers = [
            &self.mel,
            &self.maps[0],
            &self.maps[1],
            &self.flat,
            &self.pos_emb,
        ];
        let pos_emb_f16 = self.pos_emb_f16.as_ref().map_or(0, CudaSlice::num_bytes);
        lens + pos_emb_f16
            + self.act.bytes()
            + self.attention.bytes()
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
        let flat = channels * subsampled_features(spec);
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

        #[cfg(feature = "low-precision")]
        let quant = quantized::QuantState::load(runtime, file, spec, math)?;
        let mut layers = Vec::with_capacity(spec.layers);
        for index in 0..spec.layers {
            layers.push(Self::load_layer(
                runtime, &kernels, file, spec, math, index,
            )?);
        }

        // the kernels are unrolled for one head width, and cuBLAS takes any other
        let shapes = match math {
            _ if spec.d_model != spec.heads * ATTENTION_HEAD_DIM => None,
            EncoderMath::Fp16 => Some(FUSED_FP16),
            EncoderMath::Single(Math::Tf32) => Some(FUSED_TF32),
            EncoderMath::Single(_) => None,
            #[cfg(feature = "low-precision")]
            EncoderMath::Quantized(_) => None,
        };
        let fused = match shapes {
            Some(shapes) => {
                AttentionKernels::load(runtime)?.map(|kernels| FusedAttention { kernels, shapes })
            }
            None => None,
        };

        Ok(Self {
            spec,
            math,
            subsampling,
            layers,
            kernels,
            stream_kernels: StreamKernels::load(runtime)?,
            #[cfg(feature = "low-precision")]
            quant,
            fused,
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
        #[cfg(not(feature = "low-precision"))]
        let to_linear = |weight: DeviceTensor| Linear::new(runtime, kernels, weight, math);
        #[cfg(not(feature = "low-precision"))]
        let load_linear = |name: &str, shape: &[usize]| to_linear(load(name, shape)?);
        // the linear weights may be stored FP16 or quantized
        #[cfg(feature = "low-precision")]
        let load_linear = |name: &str, shape: &[usize]| {
            let name = format!("{prefix}.{name}");
            quantized::load_linear(runtime, kernels, file, &name, shape, math)
        };
        let norm = |name: &str| -> Result<Norm, TranscriptionError> {
            Ok(Norm {
                weight: load(&format!("{name}.weight"), &[d])?,
                bias: load(&format!("{name}.bias"), &[d])?,
            })
        };
        let feed_forward = |name: &str| -> Result<FeedForward, TranscriptionError> {
            Ok(FeedForward {
                linear1: load_linear(&format!("{name}.linear1.weight"), &[spec.ff_hidden, d])?,
                linear2: load_linear(&format!("{name}.linear2.weight"), &[d, spec.ff_hidden])?,
            })
        };

        #[cfg(not(feature = "low-precision"))]
        let mut qkv = Vec::with_capacity(3 * d * d);
        #[cfg(not(feature = "low-precision"))]
        for name in ["linear_q", "linear_k", "linear_v"] {
            qkv.extend_from_slice(tensor(&format!("self_attn.{name}.weight"), &[d, d])?);
        }
        let head_dim = d / spec.heads;
        let attention = Attention {
            #[cfg(not(feature = "low-precision"))]
            qkv: to_linear(DeviceTensor::upload(runtime.stream(), &qkv, &[3 * d, d])?)?,
            // the reader stacks the three weights of a file that keeps them apart
            #[cfg(feature = "low-precision")]
            qkv: load_linear("self_attn.linear_qkv.weight", &[3 * d, d])?,
            pos: load_linear("self_attn.linear_pos.weight", &[d, d])?,
            out: load_linear("self_attn.linear_out.weight", &[d, d])?,
            bias_u: load("self_attn.pos_bias_u", &[spec.heads, head_dim])?,
            bias_v: load("self_attn.pos_bias_v", &[spec.heads, head_dim])?,
        };

        // NeMo names the conv norm `batch_norm` whichever norm it is
        let bn = |name: &str| tensor(&format!("conv.batch_norm.{name}"), &[d]);
        let (scale, shift) = match spec.conv_norm {
            ConvNorm::Batch => {
                // fold in f64 so the folded constants round once
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
                (upload_host(runtime, &scale)?, upload_host(runtime, &shift)?)
            }
            ConvNorm::Layer => (
                load("conv.batch_norm.weight", &[d])?,
                load("conv.batch_norm.bias", &[d])?,
            ),
        };
        let conv = ConvModule {
            pointwise1: load_linear("conv.pointwise_conv1.weight", &[2 * d, d, 1])?,
            depthwise: load("conv.depthwise_conv.weight", &[d, 1, spec.conv_kernel])?,
            scale,
            shift,
            pointwise2: load_linear("conv.pointwise_conv2.weight", &[d, d, 1])?,
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
        // the offline pass pads symmetrically and attends everywhere, which a
        // streaming checkpoint was not trained for
        if spec.causal || spec.conv_norm != ConvNorm::Batch {
            return Err(TranscriptionError::InvalidConfiguration(
                "a streaming encoder runs chunk by chunk, not over whole inputs".to_owned(),
            ));
        }
        let (d, heads) = (spec.d_model, spec.heads);
        let channels = spec.subsampling_channels;
        let maps = self.feature_maps(batch, mel_frames);
        let frames = maps[3].frames;
        let rows = batch * frames;
        let positions = 2 * frames - 1;
        let pos_emb = stream.clone_htod(&relative_positions(frames, d))?;
        let pos_emb_f16 = self.fp16_copy(runtime, &pos_emb)?;
        #[cfg_attr(not(feature = "low-precision"), allow(unused_mut))]
        let mut act = Activations::new(runtime, spec, self.math, rows)?;
        #[cfg(feature = "low-precision")]
        if let Some(quant) = &self.quant {
            act.quant.inputs = quant.inputs(runtime, spec, rows, positions)?;
        }

        let fused = self
            .fused
            .as_ref()
            .is_some_and(|fused| fused.shapes.takes(batch, frames));
        let attention = match self.math {
            EncoderMath::Fp16 if fused => AttentionBuffers::FusedFp16 {
                qkv: zeros(stream, rows * 3 * d)?,
                pos: zeros(stream, positions * d)?,
            },
            EncoderMath::Single(_) if fused => AttentionBuffers::FusedTf32 {
                pos: zeros(stream, positions * d)?,
            },
            _ => AttentionBuffers::Unfused(Box::new(UnfusedAttention {
                q_u: zeros(stream, rows * d)?,
                q_v: zeros(stream, rows * d)?,
                k: zeros(stream, rows * d)?,
                v: zeros(stream, rows * d)?,
                pos: zeros(stream, positions * d)?,
                pos_heads: zeros(stream, positions * d)?,
                scores: zeros(stream, batch * heads * frames * frames)?,
                bd: zeros(stream, rows * heads * positions)?,
                context: zeros(stream, rows * d)?,
            })),
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
            act,
            pos_emb,
            pos_emb_f16,
            attention,
        })
    }

    /// An FP16 copy of `table` for an FP16 GEMM under [`EncoderMath::Fp16`]
    fn fp16_copy(
        &self,
        runtime: &CudaRuntime,
        table: &CudaSlice<f32>,
    ) -> Result<Option<CudaSlice<u16>>, TranscriptionError> {
        if !self.math.fp16() {
            return Ok(None);
        }
        let stream = runtime.stream();
        let mut bits = zeros(stream, table.len())?;
        self.kernels
            .to_f16(stream, &table.slice(..), &mut bits.slice_mut(..))?;
        Ok(Some(bits))
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
        kernels.fill_rows(stream, &bias.data().slice(..), &mut ws.act.x.slice_mut(..))?;
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
            &mut ws.act.x,
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
        self.pointwise_gemm(runtime, weight, input, output, map)?;
        self.kernels.subsample_bias_relu(
            runtime.stream(),
            &mut output.slice_mut(..map.len()),
            &bias.data().slice(..),
            &lens.slice(..),
            map,
        )?;
        Ok(())
    }

    /// The GEMM of a pointwise subsampling convolution, without its epilogue
    fn pointwise_gemm(
        &self,
        runtime: &CudaRuntime,
        weight: &DeviceTensor,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        map: FeatureMap,
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
            self.norm(runtime, &mut ws.act, &layer.norm_ff1)?;
        }
        self.feed_forward(runtime, &mut ws.act, &layer.ff1)?;
        self.norm(runtime, &mut ws.act, &layer.norm_att)?;
        self.attention(runtime, ws, &layer.attention)?;
        self.norm(runtime, &mut ws.act, &layer.norm_conv)?;
        self.conv_module(runtime, ws, &layer.conv)?;
        self.close_layer(runtime, &mut ws.act, index)
    }

    /// A layer's second feed-forward module and closing norm, which also writes the
    /// next layer's first norm
    fn close_layer(
        &self,
        runtime: &CudaRuntime,
        act: &mut Activations,
        index: usize,
    ) -> Result<(), TranscriptionError> {
        let layer = &self.layers[index];
        self.norm(runtime, act, &layer.norm_ff2)?;
        self.feed_forward(runtime, act, &layer.ff2)?;

        // the final norm is not residual, so it runs in place, and the next layer's
        // first norm reads its result before it leaves the registers
        let next = self.layers.get(index + 1).map(|next| {
            (
                next.norm_ff1.weight.data().slice(..),
                next.norm_ff1.bias.data().slice(..),
            )
        });
        let Activations {
            x, normed, fp16, ..
        } = act;
        self.kernels.layer_norm_pair(
            runtime.stream(),
            &mut x.slice_mut(..),
            (
                &layer.norm_out.weight.data().slice(..),
                &layer.norm_out.bias.data().slice(..),
            ),
            next.as_ref().map(|(gamma, beta)| (gamma, beta)),
            gemm_output(
                self.math,
                normed,
                fp16.as_mut().map(|fp16| &mut fp16.normed),
            )?,
            self.spec.d_model,
            NORM_EPS,
        )?;
        Ok(())
    }

    /// LayerNorm of the residual stream for the next linear layer: into `normed`, or
    /// its FP16 copy under [`EncoderMath::Fp16`]
    fn norm(
        &self,
        runtime: &CudaRuntime,
        act: &mut Activations,
        norm: &Norm,
    ) -> Result<(), TranscriptionError> {
        let Activations {
            x, normed, fp16, ..
        } = act;
        self.kernels.layer_norm(
            runtime.stream(),
            &x.slice(..),
            &norm.weight.data().slice(..),
            &norm.bias.data().slice(..),
            gemm_output(
                self.math,
                normed,
                fp16.as_mut().map(|fp16| &mut fp16.normed),
            )?,
            self.spec.d_model,
            NORM_EPS,
        )?;
        Ok(())
    }

    /// Runs `spec`, a [`linear`] product, on `a` and `weight` in their shared precision
    fn linear<C: DevicePtrMut<f32>>(
        &self,
        runtime: &CudaRuntime,
        spec: Sgemm,
        a: Operand<'_>,
        weight: &Linear,
        c: &mut C,
    ) -> Result<(), TranscriptionError> {
        match (a, weight, self.math) {
            (Operand::Single(a), Linear::Single(w), EncoderMath::Single(_)) => {
                runtime.sgemm(spec, &a, w.data(), c)?;
            }
            (Operand::Fp16(a), Linear::Fp16(w), EncoderMath::Fp16) => {
                linear_fp16(runtime, spec, &a, w, c)?;
            }
            _ => return Err(mixed_precision()),
        }
        Ok(())
    }

    /// [`Self::linear`], or for a quantized weight its product, which quantizes `a`
    /// into `quant` first; only the offline pass runs quantized products
    fn linear_or_quantized<C: DevicePtrMut<f32>>(
        &self,
        runtime: &CudaRuntime,
        spec: Sgemm,
        a: Operand<'_>,
        weight: &Linear,
        c: &mut C,
        #[cfg_attr(not(feature = "low-precision"), allow(unused_variables))]
        quant: &mut QuantScratch,
    ) -> Result<(), TranscriptionError> {
        #[cfg(feature = "low-precision")]
        if let (Operand::Fp16(a), Linear::Quantized(w), EncoderMath::Quantized(_)) =
            (&a, weight, self.math)
        {
            let state = self.quant.as_ref().ok_or_else(mixed_precision)?;
            return state.linear(runtime, spec, a, w, c, quant.inputs.as_mut());
        }
        self.linear(runtime, spec, a, weight, c)
    }

    /// `x += 0.5 * linear2(silu(linear1(normed)))`, with `normed` the sublayer's
    /// norm of `x`
    fn feed_forward(
        &self,
        runtime: &CudaRuntime,
        act: &mut Activations,
        weights: &FeedForward,
    ) -> Result<(), TranscriptionError> {
        let (d, hidden) = (self.spec.d_model, self.spec.ff_hidden);
        let math = self.math.single();
        let rows = act.rows;
        self.linear_or_quantized(
            runtime,
            linear(rows, hidden, d, 1.0, 0.0, math),
            normed_in(self.math, &act.normed, act.fp16.as_ref())?,
            &weights.linear1,
            &mut act.hidden.slice_mut(..rows * hidden),
            &mut act.quant,
        )?;
        let stream = runtime.stream();
        match act.fp16.as_mut() {
            Some(fp16) => self.kernels.silu_f16(
                stream,
                &act.hidden.slice(..rows * hidden),
                &mut fp16.hidden.slice_mut(..),
            )?,
            None => self
                .kernels
                .silu(stream, &mut act.hidden.slice_mut(..rows * hidden))?,
        }
        let activated = gemm_input(
            self.math,
            act.hidden.slice(..rows * hidden),
            act.fp16.as_ref().map(|fp16| fp16.hidden.slice(..)),
        )?;
        // NeMo scales the feed-forward residual by 0.5 (`fc_factor`)
        self.linear_or_quantized(
            runtime,
            linear(rows, d, hidden, 0.5, 1.0, math),
            activated,
            &weights.linear2,
            &mut act.x,
            &mut act.quant,
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
        match &self.fused {
            Some(fused) if !matches!(ws.attention, AttentionBuffers::Unfused(_)) => {
                self.fused_attention(runtime, ws, weights, fused)?;
            }
            _ => self.unfused_attention(runtime, ws, weights)?,
        }

        let d = self.spec.d_model;
        self.linear_or_quantized(
            runtime,
            linear(ws.act.rows, d, d, 1.0, 1.0, self.math.single()),
            merged_in(self.math, &ws.act.merged, ws.act.fp16.as_ref())?,
            &weights.out,
            &mut ws.act.x,
            &mut ws.act.quant,
        )?;
        Ok(())
    }

    /// The attention context into `merged`, or its FP16 copy, through the projections
    /// and the fused kernel of the encoder's precision
    fn fused_attention(
        &self,
        runtime: &CudaRuntime,
        ws: &mut EncoderWorkspace,
        weights: &Attention,
        fused: &FusedAttention,
    ) -> Result<(), TranscriptionError> {
        let stream = runtime.stream();
        let spec = self.spec;
        let (d, heads) = (spec.d_model, spec.heads);
        let rows = ws.act.rows;
        let positions = 2 * ws.frames - 1;
        let shape = Heads {
            batch: ws.batch,
            frames: ws.frames,
            heads,
            head_dim: d / heads,
        };
        let math = self.math.single();
        let (qkv_spec, pos_spec) = (
            linear(rows, 3 * d, d, 1.0, 0.0, math),
            linear(positions, d, d, 1.0, 0.0, math),
        );
        let (bias_u, bias_v) = (
            weights.bias_u.data().slice(..),
            weights.bias_v.data().slice(..),
        );
        let lens = ws.lens[3].slice(..);

        match (
            &mut ws.attention,
            ws.act.fp16.as_mut(),
            ws.pos_emb_f16.as_ref(),
            &weights.qkv,
            &weights.pos,
        ) {
            (
                AttentionBuffers::FusedFp16 { qkv, pos },
                Some(fp16),
                Some(table),
                Linear::Fp16(qkv_weight),
                Linear::Fp16(pos_weight),
            ) => {
                linear_fp16_to_fp16(runtime, qkv_spec, &fp16.normed, qkv_weight, qkv)?;
                linear_fp16_to_fp16(runtime, pos_spec, table, pos_weight, pos)?;
                fused.kernels.rel_pos_f16(
                    stream,
                    &qkv.slice(..),
                    &pos.slice(..),
                    &bias_u,
                    &bias_v,
                    &lens,
                    &mut fp16.merged.slice_mut(..),
                    shape,
                )?;
            }
            (
                AttentionBuffers::FusedTf32 { pos },
                None,
                None,
                Linear::Single(qkv_weight),
                Linear::Single(pos_weight),
            ) => {
                runtime.sgemm(qkv_spec, &ws.act.normed, qkv_weight.data(), &mut ws.act.qkv)?;
                runtime.sgemm(pos_spec, &ws.pos_emb, pos_weight.data(), pos)?;
                fused.kernels.rel_pos_tf32(
                    stream,
                    &ws.act.qkv.slice(..),
                    &pos.slice(..),
                    &bias_u,
                    &bias_v,
                    &lens,
                    &mut ws.act.merged.slice_mut(..),
                    shape,
                )?;
            }
            _ => return Err(mixed_precision()),
        }
        Ok(())
    }

    /// The attention context into `merged` (or its FP16 copy) through the head split,
    /// cuBLAS score and value products, the rel-shift softmax and the head merge
    fn unfused_attention(
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
        let AttentionBuffers::Unfused(buffers) = &mut ws.attention else {
            return Err(mixed_precision());
        };

        let math = self.math.single();
        self.linear_or_quantized(
            runtime,
            linear(rows, 3 * d, d, 1.0, 0.0, math),
            normed_in(self.math, &ws.act.normed, ws.act.fp16.as_ref())?,
            &weights.qkv,
            &mut ws.act.qkv,
            &mut ws.act.quant,
        )?;
        self.kernels.split_qkv(
            stream,
            &ws.act.qkv.slice(..),
            &weights.bias_u.data().slice(..),
            &weights.bias_v.data().slice(..),
            &mut buffers.q_u.slice_mut(..),
            &mut buffers.q_v.slice_mut(..),
            &mut buffers.k.slice_mut(..),
            &mut buffers.v.slice_mut(..),
            shape,
        )?;
        let table = gemm_input(
            self.math,
            ws.pos_emb.slice(..),
            ws.pos_emb_f16.as_ref().map(|table| table.slice(..)),
        )?;
        self.linear_or_quantized(
            runtime,
            linear(positions, d, d, 1.0, 0.0, math),
            table,
            &weights.pos,
            &mut buffers.pos,
            &mut ws.act.quant,
        )?;
        self.kernels.split_heads(
            stream,
            &buffers.pos.slice(..),
            &mut buffers.pos_heads.slice_mut(..),
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
            &buffers.q_u,
            &buffers.k,
            &mut buffers.scores,
        )?;
        // position scores: (q + v) · pᵀ per head, every utterance against one table
        let position = Sgemm {
            b_transposed: true,
            math,
            ..Sgemm::new(rows, positions, head_dim)
        };
        runtime.sgemm_batched(
            SgemmBatched::packed(position, heads),
            &buffers.q_v,
            &buffers.pos_heads,
            &mut buffers.bd,
        )?;
        self.kernels.rel_pos_softmax(
            stream,
            &mut buffers.scores.slice_mut(..),
            &buffers.bd.slice(..),
            &ws.lens[3].slice(..),
            shape,
        )?;

        let mix = Sgemm {
            math,
            ..Sgemm::new(frames, head_dim, frames)
        };
        runtime.sgemm_batched(
            SgemmBatched::packed(mix, batch * heads),
            &buffers.scores,
            &buffers.v,
            &mut buffers.context,
        )?;
        self.kernels.merge_heads(
            stream,
            &buffers.context.slice(..),
            ws.act.merged_out(self.math)?,
            shape,
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
        let act = &mut ws.act;
        self.linear_or_quantized(
            runtime,
            linear(rows, 2 * d, d, 1.0, 0.0, math),
            normed_in(self.math, &act.normed, act.fp16.as_ref())?,
            &weights.pointwise1,
            &mut act.hidden.slice_mut(..pw1_len),
            &mut act.quant,
        )?;
        let Activations {
            hidden,
            merged,
            fp16,
            ..
        } = act;
        let glued = gemm_output(
            self.math,
            merged,
            fp16.as_mut().map(|fp16| &mut fp16.merged),
        )?;
        self.kernels.conv_glue(
            runtime.stream(),
            &hidden.slice(..pw1_len),
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
        self.linear_or_quantized(
            runtime,
            linear(rows, d, d, 1.0, 1.0, math),
            merged_in(self.math, &act.merged, act.fp16.as_ref())?,
            &weights.pointwise2,
            &mut act.x,
            &mut act.quant,
        )?;
        Ok(())
    }
}

impl Linear {
    /// Keeps `weight` for an FP32 or TF32 GEMM, or rounds it to FP16 on the device for
    /// [`EncoderMath::Fp16`] and frees the FP32 copy
    fn new(
        runtime: &CudaRuntime,
        kernels: &EncoderKernels,
        weight: DeviceTensor,
        math: EncoderMath,
    ) -> Result<Self, TranscriptionError> {
        if !math.fp16() {
            return Ok(Self::Single(weight));
        }
        let stream = runtime.stream();
        let mut bits = zeros(stream, weight.data().len())?;
        kernels.to_f16(stream, &weight.data().slice(..), &mut bits.slice_mut(..))?;
        Ok(Self::Fp16(bits))
    }
}

/// Where a kernel writes the activation a linear layer reads next: `single`, or
/// `fp16`, its FP16 copy, under [`EncoderMath::Fp16`]
fn gemm_output<'a>(
    math: EncoderMath,
    single: &'a mut CudaSlice<f32>,
    fp16: Option<&'a mut CudaSlice<u16>>,
) -> Result<GemmOperand<'a>, TranscriptionError> {
    match (math.fp16(), fp16) {
        (false, _) => Ok(GemmOperand::Single(single.slice_mut(..))),
        (true, Some(fp16)) => Ok(GemmOperand::Fp16(fp16.slice_mut(..))),
        (true, None) => Err(mixed_precision()),
    }
}

/// The activation a linear layer reads: `single`, or `fp16`, its FP16 copy, under
/// [`EncoderMath::Fp16`]
fn gemm_input<'a>(
    math: EncoderMath,
    single: CudaView<'a, f32>,
    fp16: Option<CudaView<'a, u16>>,
) -> Result<Operand<'a>, TranscriptionError> {
    match (math.fp16(), fp16) {
        (false, _) => Ok(Operand::Single(single)),
        (true, Some(fp16)) => Ok(Operand::Fp16(fp16)),
        (true, None) => Err(mixed_precision()),
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
    position_rows(frames - 1, 2 * frames - 1, d_model)
}

/// `rows` rows of NeMo's relative position table for positions `highest` downward
fn position_rows(highest: usize, rows: usize, d_model: usize) -> Vec<f32> {
    let scale = -(10000f64.ln() / d_model as f64) as f32;
    let div_term: Vec<f32> = (0..d_model)
        .step_by(2)
        .map(|i| (i as f32 * scale).exp())
        .collect();
    let mut table = Vec::with_capacity(rows * d_model);
    for row in 0..rows {
        let position = highest as f32 - row as f32;
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
    use ndarray::{ArrayD, Ix3};

    use super::{CudaEncoder, EncoderMath, relative_positions};
    use crate::cuda::kernels::CudaRuntime;
    use crate::cuda::test_support::{compare, weights};
    use crate::deployment::{ParakeetDeployment, ParakeetVersion};
    use crate::test_support::{NemoReference, deployment};

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
            let file = weights(version);
            let spec = file.parakeet_spec(&deployment(version)).expect("spec");
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
            let file = weights(version);
            let spec = file.parakeet_spec(&deployment(version)).expect("spec");
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

    /// IEEE half bits to `f32`, for reading the FP16 kernel's buffers on the host
    fn f16_to_f32(bits: u16) -> f32 {
        let sign = u32::from(bits >> 15) << 31;
        let exponent = u32::from((bits >> 10) & 0x1f);
        let mantissa = u32::from(bits & 0x3ff);
        let magnitude = match exponent {
            0 => mantissa as f32 * 2f32.powi(-24),
            0x1f if mantissa == 0 => f32::INFINITY,
            0x1f => f32::NAN,
            _ => f32::from_bits(((exponent + 112) << 23) | (mantissa << 13)),
        };
        f32::from_bits(magnitude.to_bits() | sign)
    }

    /// NeMo's relative-position attention in f64: `qkv` is `[batch * frames, 3 *
    /// d_model]`, `pos` `[2 * frames - 1, d_model]`, and the result the merged `[batch *
    /// frames, d_model]` context, 0 on rows past each length
    fn attention_reference(
        qkv: &[f32],
        pos: &[f32],
        [bias_u, bias_v]: [&[f32]; 2],
        lens: &[usize],
        frames: usize,
        heads: usize,
    ) -> Vec<f32> {
        let d = bias_u.len();
        let head_dim = d / heads;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let mut out = vec![0f32; lens.len() * frames * d];
        let mut scores = vec![0f64; frames];
        for (b, &len) in lens.iter().enumerate() {
            let row = |i: usize| &qkv[(b * frames + i) * 3 * d..][..3 * d];
            for h in 0..heads {
                let c0 = h * head_dim;
                for i in 0..len {
                    let q = &row(i)[c0..c0 + head_dim];
                    let mut max = f64::NEG_INFINITY;
                    for (j, score) in scores.iter_mut().enumerate().take(len) {
                        let k = &row(j)[d + c0..d + c0 + head_dim];
                        let p = &pos[(frames - 1 - i + j) * d + c0..][..head_dim];
                        let mut sum = 0f64;
                        for c in 0..head_dim {
                            let q = f64::from(q[c]);
                            sum += (q + f64::from(bias_u[c0 + c])) * f64::from(k[c])
                                + (q + f64::from(bias_v[c0 + c])) * f64::from(p[c]);
                        }
                        *score = sum * scale;
                        max = max.max(*score);
                    }
                    let mut total = 0f64;
                    for score in scores.iter_mut().take(len) {
                        *score = (*score - max).exp();
                        total += *score;
                    }
                    let context = &mut out[(b * frames + i) * d + c0..][..head_dim];
                    for (c, value) in context.iter_mut().enumerate() {
                        let mut sum = 0f64;
                        for (j, weight) in scores.iter().enumerate().take(len) {
                            sum += weight * f64::from(row(j)[2 * d + c0 + c]);
                        }
                        *value = (sum / total) as f32;
                    }
                }
            }
        }
        out
    }

    /// Relative L2 error of `ours` against `expected` over the valid rows, in total,
    /// per head, and per row inside a warp's 16 query rows; padded rows must be 0
    fn attention_error(
        ours: &[f32],
        expected: &[f32],
        lens: &[usize],
        frames: usize,
        heads: usize,
    ) -> (f64, Vec<f64>, Vec<f64>) {
        let d = expected.len() / (lens.len() * frames);
        let head_dim = d / heads;
        let mut per_head = vec![[0f64; 2]; heads];
        let mut per_row = vec![[0f64; 2]; 16];
        for (b, &len) in lens.iter().enumerate() {
            for i in 0..len {
                for c in 0..d {
                    let at = (b * frames + i) * d + c;
                    let (value, reference) = (f64::from(ours[at]), f64::from(expected[at]));
                    assert!(value.is_finite(), "row {b} frame {i} channel {c}: {value}");
                    let error = (value - reference).powi(2);
                    let norm = reference.powi(2);
                    for bucket in [&mut per_head[c / head_dim], &mut per_row[i % 16]] {
                        bucket[0] += error;
                        bucket[1] += norm;
                    }
                }
            }
            // NeMo zeroes the weights of padded query rows
            for i in len..frames {
                let row = &ours[(b * frames + i) * d..(b * frames + i + 1) * d];
                assert!(
                    row.iter().all(|value| *value == 0.0),
                    "row {b} padded frame {i}"
                );
            }
        }
        let rel = |[error, norm]: [f64; 2]| (error / norm).sqrt();
        let total = rel(per_head
            .iter()
            .fold([0.0; 2], |a, b| [a[0] + b[0], a[1] + b[1]]));
        (
            total,
            per_head.into_iter().map(rel).collect(),
            per_row.into_iter().map(rel).collect(),
        )
    }

    /// Both fused attention kernels and the conversions their inputs take
    struct FusedKernels {
        runtime: CudaRuntime,
        attention: crate::cuda::kernels::AttentionKernels,
        encoder: crate::cuda::kernels::EncoderKernels,
    }

    /// One kernel's output and the f64 attention on the operands it read
    struct FusedRun {
        name: &'static str,
        out: Vec<f32>,
        reference: Vec<f32>,
    }

    impl FusedKernels {
        /// The kernels on GPU 0, or `None` where the fused attention does not load
        fn open() -> Option<Self> {
            use crate::cuda::kernels::{AttentionKernels, EncoderKernels};

            let runtime = CudaRuntime::open(0, Options::default()).expect("CUDA runtime");
            let attention = AttentionKernels::load(&runtime).expect("attention kernels")?;
            let encoder = EncoderKernels::load(&runtime).expect("encoder kernels");
            Some(Self {
                runtime,
                attention,
                encoder,
            })
        }

        /// Runs both kernels on FP32 `qkv` and `pos`: FP16 on FP16 copies of them,
        /// which its reference reads too, and TF32 on them as they are
        fn run(
            &self,
            qkv: &[f32],
            pos: &[f32],
            biases: [&[f32]; 2],
            lens: &[usize],
            frames: usize,
            heads: usize,
        ) -> [FusedRun; 2] {
            use crate::cuda::kernels::Heads;
            use cudakit::cudarc::driver::CudaSlice;

            let stream = self.runtime.stream();
            let shape = Heads {
                batch: lens.len(),
                frames,
                heads,
                head_dim: biases[0].len() / heads,
            };
            let rows = qkv.len() / 3;
            let qkv_device = stream.clone_htod(qkv).expect("upload");
            let pos_device = stream.clone_htod(pos).expect("upload");
            let bias_u = stream.clone_htod(biases[0]).expect("upload");
            let bias_v = stream.clone_htod(biases[1]).expect("upload");
            let lens_device = stream
                .clone_htod(&lens.iter().map(|len| *len as u32).collect::<Vec<_>>())
                .expect("upload");

            let mut qkv_f16 = stream.alloc_zeros::<u16>(qkv.len()).expect("alloc");
            let mut pos_f16 = stream.alloc_zeros::<u16>(pos.len()).expect("alloc");
            self.encoder
                .to_f16(stream, &qkv_device.slice(..), &mut qkv_f16.slice_mut(..))
                .expect("to fp16");
            self.encoder
                .to_f16(stream, &pos_device.slice(..), &mut pos_f16.slice_mut(..))
                .expect("to fp16");
            let decode = |buffer: &CudaSlice<u16>| {
                stream
                    .clone_dtoh(buffer)
                    .expect("download")
                    .into_iter()
                    .map(f16_to_f32)
                    .collect::<Vec<f32>>()
            };

            // NaN until written, so a row the kernels skip shows
            let mut out_f16 = stream.clone_htod(&vec![0x7e00u16; rows]).expect("upload");
            self.attention
                .rel_pos_f16(
                    stream,
                    &qkv_f16.slice(..),
                    &pos_f16.slice(..),
                    &bias_u.slice(..),
                    &bias_v.slice(..),
                    &lens_device.slice(..),
                    &mut out_f16.slice_mut(..),
                    shape,
                )
                .expect("fp16 kernel");
            let mut out_tf32 = stream.clone_htod(&vec![f32::NAN; rows]).expect("upload");
            self.attention
                .rel_pos_tf32(
                    stream,
                    &qkv_device.slice(..),
                    &pos_device.slice(..),
                    &bias_u.slice(..),
                    &bias_v.slice(..),
                    &lens_device.slice(..),
                    &mut out_tf32.slice_mut(..),
                    shape,
                )
                .expect("tf32 kernel");

            [
                FusedRun {
                    name: "FP16",
                    out: decode(&out_f16),
                    reference: attention_reference(
                        &decode(&qkv_f16),
                        &decode(&pos_f16),
                        biases,
                        lens,
                        frames,
                        heads,
                    ),
                },
                FusedRun {
                    name: "TF32",
                    out: stream.clone_dtoh(&out_tf32).expect("download"),
                    reference: attention_reference(qkv, pos, biases, lens, frames, heads),
                },
            ]
        }
    }

    impl FusedRun {
        /// Holds the output to its reference, adding to `failures` when it is off by
        /// more than rounding
        fn hold(
            &self,
            set: &str,
            lens: &[usize],
            frames: usize,
            heads: usize,
            failures: &mut Vec<String>,
        ) {
            let (own, per_head, per_row) =
                attention_error(&self.out, &self.reference, lens, frames, heads);
            let name = self.name;
            println!("{set} {name}: rel {own:.2e} against f64 on its inputs");
            let list = |values: &[f64]| {
                values
                    .iter()
                    .map(|value| format!("{value:.1e}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            println!("  per head {}", list(&per_head));
            println!("  per row in warp {}", list(&per_row));
            // 10-bit operands and FP16 weights give about 1e-3; a wrong fragment,
            // shift or mask is off by order 1
            if own > 1e-2 {
                failures.push(format!("{set} {name}: {own:.2e}"));
            }
        }
    }

    /// Feeds NeMo's projections of the first and last layer to each fused kernel and
    /// holds the merged context against the same attention in f64 on the operands the
    /// kernel read, which separates the kernel's own error from its input rounding, and
    /// that f64 attention against NeMo's
    #[test]
    #[ignore = "needs a CUDA GPU and the NeMo references; `just gpu-parity` runs it on gpu0"]
    fn gpu_fused_attention_kernels_match_nemo() {
        let Some(kernels) = FusedKernels::open() else {
            println!("no fused attention on this GPU");
            return;
        };
        let file = weights(ParakeetVersion::V3);
        let spec = file.parakeet_spec(&ParakeetDeployment::v3()).expect("spec");
        let (d, heads) = (spec.encoder.d_model, spec.encoder.heads);
        let mut failures = Vec::new();
        // the references dump the attention operands of the first and the last layer
        for (index, set) in [0, 23]
            .into_iter()
            .flat_map(|index| ["b1", "b8", "long-full"].map(|set| (index, set)))
        {
            let tensor = |name: &str| {
                file.tensor(
                    &format!("encoder.layers.{index}.self_attn.{name}"),
                    &[heads, d / heads],
                )
                .expect("bias")
                .to_vec()
            };
            let (bias_u, bias_v) = (tensor("pos_bias_u"), tensor("pos_bias_v"));

            let reference = NemoReference::open(ParakeetVersion::V3, set);
            let layer = |name: &str| {
                reference
                    .f32(&format!("encoder.layers.{index}.self_attn.{name}"))
                    .iter()
                    .copied()
                    .collect::<Vec<f32>>()
            };
            let set = format!("layer {index} {set}");
            let (q, k, v) = (
                layer("linear_q.out"),
                layer("linear_k.out"),
                layer("linear_v.out"),
            );
            let pos = layer("linear_pos.out");
            let expected = layer("linear_out.in");
            let lens: Vec<usize> = reference
                .i64("encoder.out_len")
                .iter()
                .map(|len| *len as usize)
                .collect();
            let frames = q.len() / (lens.len() * d);
            let mut qkv = Vec::with_capacity(3 * q.len());
            for row in 0..lens.len() * frames {
                for part in [&q, &k, &v] {
                    qkv.extend_from_slice(&part[row * d..(row + 1) * d]);
                }
            }

            let runs = kernels.run(&qkv, &pos, [&bias_u, &bias_v], &lens, frames, heads);
            // TF32 reads the projections as they are, so its reference is NeMo's
            // attention in f64
            let (rel, _, _) = attention_error(&runs[1].reference, &expected, &lens, frames, heads);
            println!("{set}: f64 reference against NeMo rel {rel:.2e}");
            // NeMo's FP32 attention lands about 1e-6 from it, so the kernels' own bound
            // holds them to NeMo's mapping, not only to a shared misreading of it
            if rel > 1e-5 {
                failures.push(format!("{set} reference: {rel:.2e}"));
            }
            for run in &runs {
                let (nemo, _, _) = attention_error(&run.out, &expected, &lens, frames, heads);
                println!("{set} {}: rel {nemo:.2e} against NeMo", run.name);
                run.hold(&set, &lens, frames, heads, &mut failures);
            }
        }
        assert!(failures.is_empty(), "kernel error: {failures:?}");
    }

    /// Runs both fused kernels on random inputs from 1 frame to past one key tile,
    /// which the NeMo sets never reach and the encoder runs fused at any batch, with
    /// mixed lengths so blocks see partial query and key tiles and padded keys
    #[test]
    #[ignore = "needs a CUDA GPU; `just gpu-parity` runs it on gpu0"]
    fn gpu_fused_attention_kernels_handle_short_inputs() {
        let Some(kernels) = FusedKernels::open() else {
            println!("no fused attention on this GPU");
            return;
        };
        let (heads, d) = (8, 8 * crate::cuda::kernels::ATTENTION_HEAD_DIM);
        // xorshift, so a failure reproduces
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut uniform = |scale: f32, len: usize| {
            (0..len)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    ((state >> 40) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
                })
                .collect::<Vec<f32>>()
        };
        let mut failures = Vec::new();
        // the key tiles are 64 frames in FP16 and 32 in TF32, a warp's query tile 16
        for frames in [1usize, 7, 16, 31, 33, 64, 65, 96] {
            let lens = [frames, frames.div_ceil(2), 1];
            let qkv = uniform(2.0, lens.len() * frames * 3 * d);
            let pos = uniform(2.0, (2 * frames - 1) * d);
            let (bias_u, bias_v) = (uniform(0.5, d), uniform(0.5, d));
            let set = format!("{frames} frames");
            for run in kernels.run(&qkv, &pos, [&bias_u, &bias_v], &lens, frames, heads) {
                run.hold(&set, &lens, frames, heads, &mut failures);
            }
        }
        assert!(failures.is_empty(), "kernel error: {failures:?}");
    }

    /// The closing Conformer layers, which amplify any rounding difference several
    /// times over, so two 10-bit paths diverge there by chance
    const AMPLIFYING_LAYERS: usize = 4;

    /// Fused attention at every shape and at none, to compare the two paths
    const ALWAYS: super::FusedShapes = super::FusedShapes {
        short_frames: usize::MAX,
        min_rows: 0,
    };
    const NEVER: super::FusedShapes = super::FusedShapes {
        short_frames: 0,
        min_rows: usize::MAX,
    };

    /// Precisions with a fused attention kernel
    const FUSED_PRECISIONS: [(&str, EncoderMath); 2] = [
        ("FP16", EncoderMath::Fp16),
        ("TF32", EncoderMath::Single(Math::Tf32)),
    ];

    /// Runs the FP16 and TF32 encoders with the fused attention and with the cuBLAS
    /// attention on short, batched and long inputs, and holds the fused one to the
    /// cuBLAS one's error against NeMo FP32 layer by layer until the closing layers
    /// amplify rounding, and to the fast encoders' bound at the output
    #[test]
    #[ignore = "needs a CUDA GPU and the NeMo references; `just gpu-parity` runs it on gpu0"]
    fn gpu_fused_attention_matches_cublas() {
        let runtime = CudaRuntime::open(0, Options::default()).expect("CUDA runtime");
        runtime.prepare(Library::Cublas).expect("cuBLAS");
        let file = weights(ParakeetVersion::V3);
        let spec = file.parakeet_spec(&ParakeetDeployment::v3()).expect("spec");
        let mut failures = Vec::new();
        for (name, math) in FUSED_PRECISIONS {
            let mut encoder =
                CudaEncoder::load(&runtime, &file, spec.encoder, math).expect("encoder");
            if encoder.fused.is_none() {
                println!("no fused attention on this GPU");
                return;
            }

            for set in ["b1", "b8", "long-full"] {
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
                let expected = reference.f32("encoder.out").permuted_axes(vec![0, 2, 1]);
                let expected = expected.as_standard_layout().into_owned();
                let mut encode = |fused: bool| {
                    if let Some(attention) = encoder.fused.as_mut() {
                        attention.shapes = if fused { ALWAYS } else { NEVER };
                    }
                    let mut ws = encoder
                        .workspace(&runtime, batch, mel_frames)
                        .expect("workspace");
                    assert_eq!(
                        !matches!(ws.attention, super::AttentionBuffers::Unfused(_)),
                        fused,
                        "{name} {set}: attention path"
                    );
                    encoder
                        .set_input(&runtime, &mut ws, &lens, |mel| {
                            mel.copy_from_slice(&time_major)
                        })
                        .expect("input");
                    encoder
                        .enqueue_subsampling(&runtime, &mut ws)
                        .expect("subsampling");
                    let mut layers = Vec::with_capacity(spec.encoder.layers);
                    for index in 0..spec.encoder.layers {
                        encoder
                            .enqueue_layer(&runtime, &mut ws, index)
                            .expect("layer");
                        let output = runtime.stream().clone_dtoh(ws.output()).expect("download");
                        let layer = reference.f32(&format!("encoder.layers.{index}.out"));
                        layers.push(compare(&output, &layer, ws.output_lens()).0);
                    }

                    let output = runtime.stream().clone_dtoh(ws.output()).expect("download");
                    (compare(&output, &expected, ws.output_lens()), layers)
                };
                let ((fused_rel, fused_max), fused_layers) = encode(true);
                let ((cublas_rel, cublas_max), cublas_layers) = encode(false);
                println!(
                    "v3 {set} {name} encoder.out against NeMo: fused rel {fused_rel:.2e} max {fused_max:.2e}, cuBLAS rel {cublas_rel:.2e} max {cublas_max:.2e}"
                );
                // both paths round to 10 bits, so they track each other within 5 %
                // until the last layers, which multiply any difference about 3x
                // each and leave which path ends closer to chance (2.6e-2 against
                // 6.1e-3 on FP16 b1, 1.8e-3 against 3.1e-3 on FP16 b8, RTX 5090);
                // a wrong score, mask or shift is off by order 1 from layer 0
                let tracked = spec.encoder.layers - AMPLIFYING_LAYERS;
                for (index, (fused, cublas)) in fused_layers.iter().zip(&cublas_layers).enumerate()
                {
                    println!("  layer {index:2}: fused {fused:.2e} cuBLAS {cublas:.2e}");
                    if index < tracked && *fused > 1.25 * cublas + 1e-5 {
                        failures.push(format!(
                            "{name} {set} layer {index}: fused {fused:.2e} against cuBLAS {cublas:.2e}"
                        ));
                    }
                }
                // the bound `gpu_fast_encoders_stay_near_nemo` holds the cuBLAS path to
                if fused_rel >= 5e-2 {
                    failures.push(format!("{name} {set}: fused relative L2 {fused_rel:.2e}"));
                }
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }

    /// Times FP16 and TF32 encoder graph replays with the cuBLAS and the fused
    /// attention over a grid of batch rows and input lengths, to place
    /// [`FUSED_FP16`] and [`FUSED_TF32`], and marks the shapes where they pick the
    /// slower path
    ///
    /// [`FUSED_FP16`]: super::FUSED_FP16
    /// [`FUSED_TF32`]: super::FUSED_TF32
    #[test]
    #[ignore = "a measurement on a CUDA GPU; run it alone on gpu0 with --nocapture"]
    fn bench_fused_attention_crossover() {
        let runtime = CudaRuntime::open(0, Options::default()).expect("CUDA runtime");
        runtime.prepare(Library::Cublas).expect("cuBLAS");
        let file = weights(ParakeetVersion::V3);
        let spec = file.parakeet_spec(&ParakeetDeployment::v3()).expect("spec");
        // LibriSpeech utterances run one at a time up to about 3500 mel frames, the
        // queue batches up to 16 of them, and long-form windows are 3008
        let batches = [1, 2, 3, 4, 6, 8, 12, 16];
        let lengths = [200, 400, 800, 1200, 1600, 2400, 3008, 3600];
        for (name, math) in FUSED_PRECISIONS {
            let mut encoder =
                CudaEncoder::load(&runtime, &file, spec.encoder, math).expect("encoder");
            let Some(configured) = encoder.fused.as_ref().map(|fused| fused.shapes) else {
                panic!("no fused attention on this GPU");
            };
            println!("{name}: fused / cuBLAS replay time, * where {configured:?} picks the slower");
            println!(
                "batch {}",
                lengths.map(|mel| format!("{:>9}", mel / 8)).join("")
            );
            for batch in batches {
                let mut row = format!("{batch:5}");
                for mel_frames in lengths {
                    let mut times = [0f64; 2];
                    for (slot, shapes) in [(0, NEVER), (1, ALWAYS)] {
                        if let Some(attention) = encoder.fused.as_mut() {
                            attention.shapes = shapes;
                        }
                        times[slot] = replay_ms(&runtime, &encoder, batch, mel_frames);
                    }
                    let frames = encoder.feature_maps(batch, mel_frames)[3].frames;
                    let fused_faster = times[1] < times[0];
                    let wrong = configured.takes(batch, frames) != fused_faster;
                    let mark = if wrong { '*' } else { ' ' };
                    row += &format!("  {:6.3}{mark}", times[1] / times[0]);
                }
                println!("{row}");
            }
        }
    }

    /// Mean graph replay time of `encoder` on `batch` rows of `mel_frames` synthetic
    /// mel frames, in milliseconds
    fn replay_ms(
        runtime: &CudaRuntime,
        encoder: &CudaEncoder,
        batch: usize,
        mel_frames: usize,
    ) -> f64 {
        const REPLAYS: u32 = 50;
        let mut ws = encoder
            .workspace(runtime, batch, mel_frames)
            .expect("workspace");
        // a fixed LCG keeps the input the same for both paths
        let mut state = 0x2545_f491_u32;
        encoder
            .set_input(runtime, &mut ws, &vec![mel_frames; batch], |mel| {
                for value in mel {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    *value = (state >> 8) as f32 / (1u32 << 24) as f32 * 4.0 - 2.0;
                }
            })
            .expect("input");
        encoder.enqueue(runtime, &mut ws).expect("eager");
        // SAFETY: as in `time_graph`, the graph drops before `ws`
        let graph = unsafe { runtime.capture_nonempty(|| encoder.enqueue(runtime, &mut ws)) }
            .expect("capture");
        graph.launch().expect("warm-up");
        runtime.synchronize().expect("sync");
        let start = std::time::Instant::now();
        for _ in 0..REPLAYS {
            graph.launch().expect("replay");
        }
        runtime.synchronize().expect("sync");
        start.elapsed().as_secs_f64() * 1e3 / f64::from(REPLAYS)
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
        ];

        for (name, math) in precisions {
            let encoder = CudaEncoder::load(&runtime, &file, spec.encoder, math).expect("encoder");
            for set in ["b1", "b8"] {
                time_graph(&runtime, &encoder, name, set);
            }
        }
    }

    /// Checks one graph replay of `encoder` on the v3 reference `set` against its
    /// eager pass and prints the mean replay time
    pub(super) fn time_graph(runtime: &CudaRuntime, encoder: &CudaEncoder, name: &str, set: &str) {
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
