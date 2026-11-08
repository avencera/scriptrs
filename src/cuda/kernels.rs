//! Kernel registry and typed launchers for `crates/scriptrs-cuda-kernels`
//!
//! cuda-oxide passes every slice parameter as a device pointer followed by its `u64`
//! length, so each launcher pushes both, in the kernel's parameter order. Kernels write
//! through unchecked indexes, so every launcher first checks that each buffer holds
//! exactly the elements its shape implies

use cudakit::cudarc::driver::{
    CudaFunction, CudaStream, CudaView, CudaViewMut, LaunchArgs, LaunchConfig, PushKernelArg,
};
use cudakit::{
    AreaPtx, EmbeddedPtx, Error, ExtraSymbols, KernelArea, LibraryPolicy, LoadedKernels, Profile,
    PtxTier, Runtime, TierSet, check_len, element_count,
};

use crate::cuda::fp16;

mod attention;
mod decode;
mod stream;

pub(crate) use attention::{ATTENTION_HEAD_DIM, AttentionKernels};
pub(crate) use decode::{
    DecodeFilter, DecodeKernels, DecodeOutput, DecodeShape, DecodeState, LstmInput, LstmState,
    LstmWeights, MAX_FILTER_TOP_K, Projection,
};
pub(crate) use stream::{StreamGeometry, StreamKernels, StreamRows};

/// Build-wide CUDA policy: cuBLAS is allowed while the encoder is library-first. The
/// `sm_75` baseline ships every area; `sm_80` ships only the fused attention, which
/// needs `mma.sync`, so older GPUs keep the cuBLAS attention. The FP16 encoder GEMMs
/// call `cublasGemmEx` directly
pub(crate) static PROFILE: Profile = Profile {
    consumer: "scriptrs",
    compiled: TierSet::of(&[PtxTier::Sm75, PtxTier::Sm80]),
    shipped: TierSet::of(&[PtxTier::Sm75, PtxTier::Sm80]),
    // `cuda` embeds every shipped tier; naming it for a tier above them would tell a
    // user who already enabled `cuda` to enable it, so those get cudakit's generic hint
    tier_features: &[(PtxTier::Sm75, "cuda"), (PtxTier::Sm80, "cuda")],
    libraries: LibraryPolicy::Allowed,
    extra_symbols: ExtraSymbols {
        cublas: fp16::CUBLAS_SYMBOLS,
        ..ExtraSymbols::NONE
    },
};

/// One PTX module per kernel area
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum KernelModule {
    Encoder,
    Decode,
    #[cfg(feature = "low-precision")]
    Quant,
    /// The fused attention, `sm_80` only
    Attention,
}

impl KernelArea for KernelModule {
    const PROFILE: &'static Profile = &PROFILE;

    fn name(self) -> &'static str {
        match self {
            Self::Encoder => "encoder",
            Self::Decode => "decode",
            #[cfg(feature = "low-precision")]
            Self::Quant => "quant",
            Self::Attention => "attention",
        }
    }

    // one copy of the embedded PTX, not one per calling codegen unit
    #[inline(never)]
    fn variants(self) -> AreaPtx {
        // hand-built with `just cuda-kernels` until cudakit-build renders tier crates
        match self {
            Self::Encoder => AreaPtx::baseline(Some(EmbeddedPtx {
                text: include_str!("ptx/encoder.sm75.ptx"),
                cubins: &[],
            })),
            Self::Decode => AreaPtx::baseline(Some(EmbeddedPtx {
                text: include_str!("ptx/decode.sm75.ptx"),
                cubins: &[],
            })),
            #[cfg(feature = "low-precision")]
            Self::Quant => AreaPtx::baseline(Some(EmbeddedPtx {
                text: include_str!("ptx/quant.sm75.ptx"),
                cubins: &[],
            })),
            Self::Attention => AreaPtx::EMPTY.with(
                PtxTier::Sm80,
                Some(EmbeddedPtx {
                    text: include_str!("ptx/attention.sm80.ptx"),
                    cubins: &[],
                }),
            ),
        }
    }
}

pub(crate) type CudaRuntime = Runtime<KernelModule>;

/// Threads per block for the one-thread-per-element kernels
pub(crate) const ELEMENT_THREADS: u32 = 256;

/// Threads per block for the row kernels (LayerNorm, softmax)
const ROW_THREADS: u32 = 256;

/// Row values each LayerNorm thread holds in registers, the kernels' `ROW_VALUES`
const ROW_VALUES: usize = 8;

/// Checks a LayerNorm over rows of `width` against `x`, its parameters and `out`,
/// and returns the launch: one block per row
fn layer_norm_rows(
    x: usize,
    gamma: usize,
    beta: usize,
    out: usize,
    width: usize,
) -> Result<LaunchConfig, Error> {
    if width == 0 || !x.is_multiple_of(width) {
        return Err(Error::BufferLength {
            context: "layer norm input",
            expected: width.max(1) * x.div_ceil(width.max(1)),
            actual: x,
        });
    }
    // each thread holds its columns in registers, so a wider row would drop some
    let widest = ROW_VALUES * ROW_THREADS as usize;
    if width > widest {
        return Err(Error::BufferLength {
            context: "layer norm width",
            expected: widest,
            actual: width,
        });
    }
    check_len("layer norm gamma", width, gamma)?;
    check_len("layer norm beta", width, beta)?;
    check_len("layer norm output", x, out)?;
    rows(x / width)
}

pub(crate) fn elements(count: usize) -> Result<LaunchConfig, Error> {
    let count = u32_of("kernel grid", count)?;
    Ok(LaunchConfig {
        grid_dim: (count.div_ceil(ELEMENT_THREADS), 1, 1),
        block_dim: (ELEMENT_THREADS, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn rows(count: usize) -> Result<LaunchConfig, Error> {
    let count = u32_of("row grid", count)?;
    Ok(LaunchConfig {
        grid_dim: (count, 1, 1),
        block_dim: (ROW_THREADS, 1, 1),
        shared_mem_bytes: 0,
    })
}

pub(crate) fn u32_of(context: &'static str, value: usize) -> Result<u32, Error> {
    u32::try_from(value).map_err(|_| Error::DimensionOverflow { context, value })
}

pub(crate) fn len<T>(view: &CudaView<'_, T>) -> u64 {
    view.len() as u64
}

pub(crate) fn len_mut<T>(view: &CudaViewMut<'_, T>) -> u64 {
    view.len() as u64
}

/// Attention head geometry shared by the head reshuffles
#[derive(Debug, Clone, Copy)]
pub(crate) struct Heads {
    pub batch: usize,
    pub frames: usize,
    pub heads: usize,
    pub head_dim: usize,
}

impl Heads {
    /// Elements in one `[batch, heads, frames, head_dim]` tensor
    fn len(self) -> Result<usize, Error> {
        element_count(
            "attention heads",
            &[self.batch, self.frames, self.heads, self.head_dim],
        )
    }
}

/// Checks the buffers of a 3x3 subsampling convolution, offline or streaming,
/// against its output map
fn check_subsampling(
    input: FeatureMap,
    output: FeatureMap,
    weight: usize,
    bias: usize,
    out: usize,
) -> Result<(), Error> {
    check_len("subsampling batch", output.batch, input.batch)?;
    check_len(
        "subsampling weight",
        element_count("subsampling weight", &[output.channels, 9])?,
        weight,
    )?;
    check_len("subsampling bias", output.channels, bias)?;
    check_len("subsampling output", output.len(), out)
}

/// The dimensions a 3x3 subsampling convolution kernel takes, in its parameter
/// order: the output's batch and channels, then the input's and the output's frames
/// and features
fn subsampling_dims(input: FeatureMap, output: FeatureMap) -> Result<[u32; 6], Error> {
    Ok([
        u32_of("batch", output.batch)?,
        u32_of("channels", output.channels)?,
        u32_of("frames", input.frames)?,
        u32_of("features", input.features)?,
        u32_of("frames", output.frames)?,
        u32_of("features", output.features)?,
    ])
}

/// The output of a kernel that feeds a GEMM: FP32 for an FP32 or TF32 GEMM, or FP16
/// bits for an FP16 one, which the kernel's `_f16` twin writes from the same
/// parameters
pub(crate) enum GemmOperand<'a> {
    Single(CudaViewMut<'a, f32>),
    Fp16(CudaViewMut<'a, u16>),
}

impl GemmOperand<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Single(view) => view.len(),
            Self::Fp16(view) => view.len(),
        }
    }

    /// The kernel for this output: `single` or its `_f16` twin
    fn pick<'k>(&self, single: &'k CudaFunction, fp16: &'k CudaFunction) -> &'k CudaFunction {
        match self {
            Self::Single(_) => single,
            Self::Fp16(_) => fp16,
        }
    }

    /// Pushes the output buffer and its length
    fn push<'l>(&'l mut self, launch: &mut LaunchArgs<'l>, length: &'l u64) {
        match self {
            Self::Single(view) => launch.arg(view).arg(length),
            Self::Fp16(view) => launch.arg(view).arg(length),
        };
    }
}

/// A subsampling feature map `[batch, channels, frames, features]`
#[derive(Debug, Clone, Copy)]
pub(crate) struct FeatureMap {
    pub batch: usize,
    pub channels: usize,
    pub frames: usize,
    pub features: usize,
}

impl FeatureMap {
    pub(crate) fn len(self) -> usize {
        self.batch * self.channels * self.frames * self.features
    }
}

/// The `encoder` area's kernels
pub(crate) struct EncoderKernels {
    layer_norm: CudaFunction,
    layer_norm_f16: CudaFunction,
    layer_norm_pair: CudaFunction,
    layer_norm_pair_f16: CudaFunction,
    silu: CudaFunction,
    silu_f16: CudaFunction,
    to_f16: CudaFunction,
    split_qkv: CudaFunction,
    split_heads: CudaFunction,
    merge_heads: CudaFunction,
    merge_heads_f16: CudaFunction,
    rel_pos_softmax: CudaFunction,
    conv_glue: CudaFunction,
    conv_glue_f16: CudaFunction,
    fill_rows: CudaFunction,
    subsample_conv_in: CudaFunction,
    subsample_depthwise: CudaFunction,
    subsample_bias_relu: CudaFunction,
    subsample_flatten: CudaFunction,
}

impl EncoderKernels {
    pub(crate) fn load(runtime: &CudaRuntime) -> Result<Self, Error> {
        let module: LoadedKernels<KernelModule> = runtime.load_embedded(KernelModule::Encoder)?;
        Ok(Self {
            layer_norm: module.function("encoder_layer_norm")?,
            layer_norm_f16: module.function("encoder_layer_norm_f16")?,
            layer_norm_pair: module.function("encoder_layer_norm_pair")?,
            layer_norm_pair_f16: module.function("encoder_layer_norm_pair_f16")?,
            silu: module.function("encoder_silu")?,
            silu_f16: module.function("encoder_silu_f16")?,
            to_f16: module.function("encoder_to_f16")?,
            split_qkv: module.function("encoder_split_qkv")?,
            split_heads: module.function("encoder_split_heads")?,
            merge_heads: module.function("encoder_merge_heads")?,
            merge_heads_f16: module.function("encoder_merge_heads_f16")?,
            rel_pos_softmax: module.function("encoder_rel_pos_softmax")?,
            conv_glue: module.function("encoder_conv_glue")?,
            conv_glue_f16: module.function("encoder_conv_glue_f16")?,
            fill_rows: module.function("encoder_fill_rows")?,
            subsample_conv_in: module.function("encoder_subsample_conv_in")?,
            subsample_depthwise: module.function("encoder_subsample_depthwise")?,
            subsample_bias_relu: module.function("encoder_subsample_bias_relu")?,
            subsample_flatten: module.function("encoder_subsample_flatten")?,
        })
    }

    /// `out = LayerNorm(x)` over rows of `width`
    #[expect(
        clippy::too_many_arguments,
        reason = "launchers mirror their kernel's parameter list so each launch reads against it"
    )]
    pub(crate) fn layer_norm(
        &self,
        stream: &CudaStream,
        x: &CudaView<'_, f32>,
        gamma: &CudaView<'_, f32>,
        beta: &CudaView<'_, f32>,
        mut out: GemmOperand<'_>,
        width: usize,
        eps: f32,
    ) -> Result<(), Error> {
        let config = layer_norm_rows(x.len(), gamma.len(), beta.len(), out.len(), width)?;
        let lens = [len(x), len(gamma), len(beta), out.len() as u64];
        let width = u32_of("layer norm width", width)?;
        let mut launch = stream.launch_builder(out.pick(&self.layer_norm, &self.layer_norm_f16));
        launch
            .arg(x)
            .arg(&lens[0])
            .arg(gamma)
            .arg(&lens[1])
            .arg(beta)
            .arg(&lens[2]);
        out.push(&mut launch, &lens[3]);
        launch.arg(&width).arg(&eps);
        // SAFETY: the arguments match the parameters of `encoder_layer_norm` and its
        // `_f16` twin in order, and one block per row of `width` stays inside `x` and
        // `out`
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// `x = LayerNorm(x)` in place with `norm`, then with `next` the next layer's
    /// first LayerNorm of the result into `out`, in FP32 or as FP16 bits for an FP16
    /// GEMM; without `next`, `out` is untouched
    #[expect(
        clippy::too_many_arguments,
        reason = "launchers mirror their kernel's parameter list so each launch reads against it"
    )]
    pub(crate) fn layer_norm_pair(
        &self,
        stream: &CudaStream,
        x: &mut CudaViewMut<'_, f32>,
        norm: (&CudaView<'_, f32>, &CudaView<'_, f32>),
        next: Option<(&CudaView<'_, f32>, &CudaView<'_, f32>)>,
        mut out: GemmOperand<'_>,
        width: usize,
        eps: f32,
    ) -> Result<(), Error> {
        let (gamma, beta) = norm;
        let config = layer_norm_rows(x.len(), gamma.len(), beta.len(), out.len(), width)?;
        // without a next norm the kernel returns before it reads the second set, so
        // `x`'s own parameters stand in
        let (next_gamma, next_beta) = next.unwrap_or((gamma, beta));
        check_len("next layer norm gamma", width, next_gamma.len())?;
        check_len("next layer norm beta", width, next_beta.len())?;
        let lens = [
            len_mut(x),
            len(gamma),
            len(beta),
            len(next_gamma),
            len(next_beta),
            out.len() as u64,
        ];
        let width = u32_of("layer norm width", width)?;
        let flag = u32::from(next.is_some());
        let mut launch =
            stream.launch_builder(out.pick(&self.layer_norm_pair, &self.layer_norm_pair_f16));
        launch
            .arg(x)
            .arg(&lens[0])
            .arg(gamma)
            .arg(&lens[1])
            .arg(beta)
            .arg(&lens[2]);
        launch
            .arg(next_gamma)
            .arg(&lens[3])
            .arg(next_beta)
            .arg(&lens[4]);
        out.push(&mut launch, &lens[5]);
        launch.arg(&width).arg(&eps).arg(&flag);
        // SAFETY: matches `encoder_layer_norm_pair` and its `_f16` twin; one block per
        // row of `width` stays inside `x` and `out`, which are sized alike above
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// `x = x * sigmoid(x)` in place
    pub(crate) fn silu(
        &self,
        stream: &CudaStream,
        x: &mut CudaViewMut<'_, f32>,
    ) -> Result<(), Error> {
        let config = elements(x.len())?;
        let length = len_mut(x);
        let mut launch = stream.launch_builder(&self.silu);
        launch.arg(x).arg(&length);
        // SAFETY: matches `encoder_silu`; the kernel bounds-checks its index
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// `out = x * sigmoid(x)` rounded to FP16
    pub(crate) fn silu_f16(
        &self,
        stream: &CudaStream,
        x: &CudaView<'_, f32>,
        out: &mut CudaViewMut<'_, u16>,
    ) -> Result<(), Error> {
        self.elementwise_f16(stream, &self.silu_f16, "silu", x, out)
    }

    /// `out = x` rounded to FP16
    pub(crate) fn to_f16(
        &self,
        stream: &CudaStream,
        x: &CudaView<'_, f32>,
        out: &mut CudaViewMut<'_, u16>,
    ) -> Result<(), Error> {
        self.elementwise_f16(stream, &self.to_f16, "to fp16", x, out)
    }

    /// Launches `function`, which maps each element of `x` to the same element of
    /// `out` as `(x, out)`
    fn elementwise_f16(
        &self,
        stream: &CudaStream,
        function: &CudaFunction,
        context: &'static str,
        x: &CudaView<'_, f32>,
        out: &mut CudaViewMut<'_, u16>,
    ) -> Result<(), Error> {
        check_len(context, x.len(), out.len())?;
        let config = elements(x.len())?;
        let lens = [len(x), len_mut(out)];
        let mut launch = stream.launch_builder(function);
        launch.arg(x).arg(&lens[0]).arg(out).arg(&lens[1]);
        // SAFETY: matches `encoder_silu_f16` and `encoder_to_f16`; `x` holds every
        // index of `out`, and the kernel bounds-checks its index against `out`
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Splits the fused QKV projection into `q + u`, `q + v`, `k` and `v`
    #[expect(
        clippy::too_many_arguments,
        reason = "launchers mirror their kernel's parameter list so each launch reads against it"
    )]
    pub(crate) fn split_qkv(
        &self,
        stream: &CudaStream,
        qkv: &CudaView<'_, f32>,
        bias_u: &CudaView<'_, f32>,
        bias_v: &CudaView<'_, f32>,
        q_u: &mut CudaViewMut<'_, f32>,
        q_v: &mut CudaViewMut<'_, f32>,
        k: &mut CudaViewMut<'_, f32>,
        v: &mut CudaViewMut<'_, f32>,
        shape: Heads,
    ) -> Result<(), Error> {
        let n = shape.len()?;
        let width = element_count("qkv width", &[shape.heads, shape.head_dim])?;
        check_len("qkv input", element_count("qkv input", &[3, n])?, qkv.len())?;
        check_len("pos_bias_u", width, bias_u.len())?;
        check_len("pos_bias_v", width, bias_v.len())?;
        for (context, buffer) in [("q + u", &*q_u), ("q + v", &*q_v), ("k", &*k), ("v", &*v)] {
            check_len(context, n, buffer.len())?;
        }
        let config = elements(n)?;
        let lens = [
            len(qkv),
            len(bias_u),
            len(bias_v),
            len_mut(q_u),
            len_mut(q_v),
            len_mut(k),
            len_mut(v),
        ];
        let dims = [
            u32_of("batch", shape.batch)?,
            u32_of("frames", shape.frames)?,
            u32_of("heads", shape.heads)?,
            u32_of("head dim", shape.head_dim)?,
        ];
        let mut launch = stream.launch_builder(&self.split_qkv);
        launch
            .arg(qkv)
            .arg(&lens[0])
            .arg(bias_u)
            .arg(&lens[1])
            .arg(bias_v)
            .arg(&lens[2]);
        launch
            .arg(q_u)
            .arg(&lens[3])
            .arg(q_v)
            .arg(&lens[4])
            .arg(k)
            .arg(&lens[5]);
        launch.arg(v).arg(&lens[6]);
        launch
            .arg(&dims[0])
            .arg(&dims[1])
            .arg(&dims[2])
            .arg(&dims[3]);
        // SAFETY: matches `encoder_split_qkv`; every output holds `batch * frames *
        // heads * head_dim` elements, which the kernel's slice indexing checks
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// `[rows, heads * head_dim]` to `[heads, rows, head_dim]`
    pub(crate) fn split_heads(
        &self,
        stream: &CudaStream,
        x: &CudaView<'_, f32>,
        out: &mut CudaViewMut<'_, f32>,
        rows_count: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<(), Error> {
        let n = element_count("split heads", &[rows_count, heads, head_dim])?;
        check_len("split heads input", n, x.len())?;
        check_len("split heads output", n, out.len())?;
        let config = elements(n)?;
        let lens = [len(x), len_mut(out)];
        let dims = [
            u32_of("rows", rows_count)?,
            u32_of("heads", heads)?,
            u32_of("head dim", head_dim)?,
        ];
        let mut launch = stream.launch_builder(&self.split_heads);
        launch.arg(x).arg(&lens[0]).arg(out).arg(&lens[1]);
        launch.arg(&dims[0]).arg(&dims[1]).arg(&dims[2]);
        // SAFETY: matches `encoder_split_heads`; indexing is checked in the kernel
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// `[batch, heads, frames, head_dim]` to `[batch * frames, heads * head_dim]`
    pub(crate) fn merge_heads(
        &self,
        stream: &CudaStream,
        x: &CudaView<'_, f32>,
        mut out: GemmOperand<'_>,
        shape: Heads,
    ) -> Result<(), Error> {
        let n = shape.len()?;
        check_len("merge heads input", n, x.len())?;
        check_len("merge heads output", n, out.len())?;
        let config = elements(n)?;
        let lens = [len(x), out.len() as u64];
        let dims = [
            u32_of("batch", shape.batch)?,
            u32_of("frames", shape.frames)?,
            u32_of("heads", shape.heads)?,
            u32_of("head dim", shape.head_dim)?,
        ];
        let mut launch = stream.launch_builder(out.pick(&self.merge_heads, &self.merge_heads_f16));
        launch.arg(x).arg(&lens[0]);
        out.push(&mut launch, &lens[1]);
        launch
            .arg(&dims[0])
            .arg(&dims[1])
            .arg(&dims[2])
            .arg(&dims[3]);
        // SAFETY: matches `encoder_merge_heads` and its `_f16` twin; indexing is
        // checked in the kernel
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Turns the content scores into rel-pos attention weights in place
    pub(crate) fn rel_pos_softmax(
        &self,
        stream: &CudaStream,
        scores: &mut CudaViewMut<'_, f32>,
        bd: &CudaView<'_, f32>,
        lens_frames: &CudaView<'_, u32>,
        shape: Heads,
    ) -> Result<(), Error> {
        let (batch, heads, frames) = (shape.batch, shape.heads, shape.frames);
        let score_rows = element_count("score rows", &[batch, heads, frames])?;
        check_len(
            "scores",
            element_count("scores", &[score_rows, frames])?,
            scores.len(),
        )?;
        // one row of 2T - 1 relative positions per query
        let positions = (2 * frames).saturating_sub(1);
        check_len(
            "position scores",
            element_count("position scores", &[score_rows, positions])?,
            bd.len(),
        )?;
        check_len("frame lengths", batch, lens_frames.len())?;
        let config = rows(score_rows)?;
        let lens = [len_mut(scores), len(bd), len(lens_frames)];
        let dims = [
            u32_of("batch", shape.batch)?,
            u32_of("heads", shape.heads)?,
            u32_of("frames", shape.frames)?,
        ];
        // NeMo divides by `math.sqrt(d_k)` rounded to FP32
        let sqrt_dk = (shape.head_dim as f64).sqrt() as f32;
        let mut launch = stream.launch_builder(&self.rel_pos_softmax);
        launch
            .arg(scores)
            .arg(&lens[0])
            .arg(bd)
            .arg(&lens[1])
            .arg(lens_frames)
            .arg(&lens[2]);
        launch
            .arg(&dims[0])
            .arg(&dims[1])
            .arg(&dims[2])
            .arg(&sqrt_dk);
        // SAFETY: matches `encoder_rel_pos_softmax`; one block per score row, and
        // indexing is checked in the kernel
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// GLU, pad mask, depthwise convolution, folded BatchNorm and SiLU
    #[expect(
        clippy::too_many_arguments,
        reason = "launchers mirror their kernel's parameter list so each launch reads against it"
    )]
    pub(crate) fn conv_glue(
        &self,
        stream: &CudaStream,
        pw1: &CudaView<'_, f32>,
        lens_frames: &CudaView<'_, u32>,
        weight: &CudaView<'_, f32>,
        scale: &CudaView<'_, f32>,
        shift: &CudaView<'_, f32>,
        mut out: GemmOperand<'_>,
        batch: usize,
        frames: usize,
        channels: usize,
        kernel_size: usize,
    ) -> Result<(), Error> {
        let n = element_count("conv module", &[batch, frames, channels])?;
        check_len(
            "conv pw1 output",
            element_count("conv pw1 output", &[2, n])?,
            pw1.len(),
        )?;
        check_len("frame lengths", batch, lens_frames.len())?;
        check_len(
            "depthwise weight",
            element_count("depthwise weight", &[channels, kernel_size])?,
            weight.len(),
        )?;
        check_len("batch norm scale", channels, scale.len())?;
        check_len("batch norm shift", channels, shift.len())?;
        check_len("conv module output", n, out.len())?;
        let config = elements(n)?;
        let lens = [
            len(pw1),
            len(lens_frames),
            len(weight),
            len(scale),
            len(shift),
            out.len() as u64,
        ];
        let dims = [
            u32_of("batch", batch)?,
            u32_of("frames", frames)?,
            u32_of("channels", channels)?,
            u32_of("kernel size", kernel_size)?,
        ];
        let mut launch = stream.launch_builder(out.pick(&self.conv_glue, &self.conv_glue_f16));
        launch
            .arg(pw1)
            .arg(&lens[0])
            .arg(lens_frames)
            .arg(&lens[1])
            .arg(weight)
            .arg(&lens[2]);
        launch.arg(scale).arg(&lens[3]).arg(shift).arg(&lens[4]);
        out.push(&mut launch, &lens[5]);
        launch
            .arg(&dims[0])
            .arg(&dims[1])
            .arg(&dims[2])
            .arg(&dims[3]);
        // SAFETY: matches `encoder_conv_glue` and its `_f16` twin; indexing is checked
        // in the kernel
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Fills each row of `out` with `bias`
    pub(crate) fn fill_rows(
        &self,
        stream: &CudaStream,
        bias: &CudaView<'_, f32>,
        out: &mut CudaViewMut<'_, f32>,
    ) -> Result<(), Error> {
        if bias.is_empty() || out.len() % bias.len() != 0 {
            return Err(Error::BufferLength {
                context: "fill rows output",
                expected: bias.len().max(1) * out.len().div_ceil(bias.len().max(1)),
                actual: out.len(),
            });
        }
        let config = elements(out.len())?;
        let lens = [len(bias), len_mut(out)];
        let width = u32_of("bias width", bias.len())?;
        let mut launch = stream.launch_builder(&self.fill_rows);
        launch
            .arg(bias)
            .arg(&lens[0])
            .arg(out)
            .arg(&lens[1])
            .arg(&width);
        // SAFETY: matches `encoder_fill_rows`; the kernel bounds-checks its index
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// First subsampling convolution with bias, mask and ReLU
    #[expect(
        clippy::too_many_arguments,
        reason = "launchers mirror their kernel's parameter list so each launch reads against it"
    )]
    pub(crate) fn subsample_conv_in(
        &self,
        stream: &CudaStream,
        mel: &CudaView<'_, f32>,
        in_lens: &CudaView<'_, u32>,
        out_lens: &CudaView<'_, u32>,
        weight: &CudaView<'_, f32>,
        bias: &CudaView<'_, f32>,
        out: &mut CudaViewMut<'_, f32>,
        input: FeatureMap,
        output: FeatureMap,
    ) -> Result<(), Error> {
        // the first convolution reads one channel
        check_len(
            "subsampling input",
            element_count(
                "subsampling input",
                &[input.batch, input.frames, input.features],
            )?,
            mel.len(),
        )?;
        check_subsampling(input, output, weight.len(), bias.len(), out.len())?;
        check_len("subsampling lengths", output.batch, in_lens.len())?;
        check_len("input lengths", output.batch, out_lens.len())?;
        let config = elements(output.len())?;
        let lens = [
            len(mel),
            len(in_lens),
            len(out_lens),
            len(weight),
            len(bias),
            len_mut(out),
        ];
        let dims = subsampling_dims(input, output)?;
        let mut launch = stream.launch_builder(&self.subsample_conv_in);
        launch
            .arg(mel)
            .arg(&lens[0])
            .arg(in_lens)
            .arg(&lens[1])
            .arg(out_lens)
            .arg(&lens[2]);
        launch
            .arg(weight)
            .arg(&lens[3])
            .arg(bias)
            .arg(&lens[4])
            .arg(out)
            .arg(&lens[5]);
        for dim in &dims {
            launch.arg(dim);
        }
        // SAFETY: matches `encoder_subsample_conv_in`; indexing is checked in the kernel
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Depthwise 3x3 stride-2 subsampling convolution with bias and mask
    #[expect(
        clippy::too_many_arguments,
        reason = "launchers mirror their kernel's parameter list so each launch reads against it"
    )]
    pub(crate) fn subsample_depthwise(
        &self,
        stream: &CudaStream,
        x: &CudaView<'_, f32>,
        out_lens: &CudaView<'_, u32>,
        weight: &CudaView<'_, f32>,
        bias: &CudaView<'_, f32>,
        out: &mut CudaViewMut<'_, f32>,
        input: FeatureMap,
        output: FeatureMap,
    ) -> Result<(), Error> {
        check_len("depthwise input", input.len(), x.len())?;
        check_subsampling(input, output, weight.len(), bias.len(), out.len())?;
        check_len("subsampling lengths", output.batch, out_lens.len())?;
        let config = elements(output.len())?;
        let lens = [len(x), len(out_lens), len(weight), len(bias), len_mut(out)];
        let dims = subsampling_dims(input, output)?;
        let mut launch = stream.launch_builder(&self.subsample_depthwise);
        launch
            .arg(x)
            .arg(&lens[0])
            .arg(out_lens)
            .arg(&lens[1])
            .arg(weight)
            .arg(&lens[2]);
        launch.arg(bias).arg(&lens[3]).arg(out).arg(&lens[4]);
        for dim in &dims {
            launch.arg(dim);
        }
        // SAFETY: matches `encoder_subsample_depthwise`; indexing is checked in the
        // kernel
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Bias, ReLU and time mask after a pointwise subsampling GEMM, in place
    pub(crate) fn subsample_bias_relu(
        &self,
        stream: &CudaStream,
        x: &mut CudaViewMut<'_, f32>,
        bias: &CudaView<'_, f32>,
        lens_frames: &CudaView<'_, u32>,
        map: FeatureMap,
    ) -> Result<(), Error> {
        check_len("pointwise output", map.len(), x.len())?;
        check_len("pointwise bias", map.channels, bias.len())?;
        check_len("frame lengths", map.batch, lens_frames.len())?;
        let config = elements(map.len())?;
        let lens = [len_mut(x), len(bias), len(lens_frames)];
        let dims = [
            u32_of("channels", map.channels)?,
            u32_of("frames", map.frames)?,
            u32_of("features", map.features)?,
        ];
        let mut launch = stream.launch_builder(&self.subsample_bias_relu);
        launch
            .arg(x)
            .arg(&lens[0])
            .arg(bias)
            .arg(&lens[1])
            .arg(lens_frames)
            .arg(&lens[2]);
        launch.arg(&dims[0]).arg(&dims[1]).arg(&dims[2]);
        // SAFETY: matches `encoder_subsample_bias_relu`; indexing is checked in the
        // kernel
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// `[batch, channels, frames, features]` to `[batch * frames, channels * features]`
    pub(crate) fn subsample_flatten(
        &self,
        stream: &CudaStream,
        x: &CudaView<'_, f32>,
        out: &mut CudaViewMut<'_, f32>,
        map: FeatureMap,
    ) -> Result<(), Error> {
        check_len("flatten input", map.len(), x.len())?;
        check_len("flatten output", map.len(), out.len())?;
        let config = elements(map.len())?;
        let lens = [len(x), len_mut(out)];
        let dims = [
            u32_of("channels", map.channels)?,
            u32_of("frames", map.frames)?,
            u32_of("features", map.features)?,
        ];
        let mut launch = stream.launch_builder(&self.subsample_flatten);
        launch.arg(x).arg(&lens[0]).arg(out).arg(&lens[1]);
        launch.arg(&dims[0]).arg(&dims[1]).arg(&dims[2]);
        // SAFETY: matches `encoder_subsample_flatten`; indexing is checked in the
        // kernel
        unsafe { launch.launch(config) }?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use cudakit::{Error, Options};

    use super::{CudaRuntime, EncoderKernels, GemmOperand};

    /// A buffer one row short must fail on the host, before the kernel's unchecked
    /// writes run past it
    #[test]
    #[ignore = "needs a CUDA GPU; `just gpu-parity` runs it on gpu0"]
    fn gpu_launchers_reject_short_buffers() {
        let runtime = CudaRuntime::open(0, Options::default()).expect("CUDA runtime");
        let kernels = EncoderKernels::load(&runtime).expect("kernels");
        let stream = runtime.stream();
        let x = stream.alloc_zeros::<f32>(2 * 4).expect("alloc");
        let gamma = stream.alloc_zeros::<f32>(4).expect("alloc");
        let mut out = stream.alloc_zeros::<f32>(4).expect("alloc");

        let result = kernels.layer_norm(
            stream,
            &x.slice(..),
            &gamma.slice(..),
            &gamma.slice(..),
            GemmOperand::Single(out.slice_mut(..)),
            4,
            1e-5,
        );
        assert!(
            matches!(result, Err(Error::BufferLength { .. })),
            "{result:?}"
        );
    }
}
