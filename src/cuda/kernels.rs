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

/// Build-wide CUDA policy: cuBLAS is allowed while the encoder is library-first, and
/// only the `sm_75` baseline PTX ships. The FP16 encoder GEMMs call `cublasGemmEx`
/// directly
pub(crate) static PROFILE: Profile = Profile {
    consumer: "scriptrs",
    compiled: TierSet::of(&[PtxTier::Sm75]),
    shipped: TierSet::of(&[PtxTier::Sm75]),
    // only `cuda` embeds PTX, and only sm75; naming it for a higher tier would tell a
    // user who already enabled `cuda` to enable it, so those get cudakit's generic hint
    tier_features: &[(PtxTier::Sm75, "cuda")],
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
}

impl KernelArea for KernelModule {
    const PROFILE: &'static Profile = &PROFILE;

    fn name(self) -> &'static str {
        match self {
            Self::Encoder => "encoder",
            Self::Decode => "decode",
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
        }
    }
}

pub(crate) type CudaRuntime = Runtime<KernelModule>;

/// Threads per block for the one-thread-per-element kernels
const ELEMENT_THREADS: u32 = 256;

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
    if width == 0 || x % width != 0 {
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

/// Threads per block of `decode_decide`, one block per batch row: a decode step
/// has only `batch` blocks, so a wide block hides the latency of each pass over
/// the class logits
const DECIDE_THREADS: u32 = 1024;

/// Class logits each `decode_decide` thread holds in registers, the kernel's
/// `DECIDE_VALUES`
const DECIDE_VALUES: usize = 12;

/// Threads per block of `decode_lstm` and `decode_project`, one block per output,
/// the kernels' `ROW_WARPS` warps
const WEIGHT_ROW_THREADS: u32 = 4 * 32;

/// Weight columns each of those threads holds in registers, the kernels'
/// `ROW_STEPS`, which bounds the predictor's hidden width
const WEIGHT_ROW_STEPS: usize = 8;

/// Fails unless a weight row of `hidden` fits the registers of one
/// `decode_lstm` or `decode_project` block
fn check_weight_row(hidden: usize) -> Result<(), Error> {
    let limit = WEIGHT_ROW_STEPS * WEIGHT_ROW_THREADS as usize;
    if hidden == 0 || hidden > limit {
        return Err(Error::BufferLength {
            context: "predictor hidden width",
            expected: limit,
            actual: hidden,
        });
    }
    Ok(())
}

fn elements(count: usize) -> Result<LaunchConfig, Error> {
    let count = u32::try_from(count).map_err(|_| Error::DimensionOverflow {
        context: "kernel grid",
        value: count,
    })?;
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

fn u32_of(context: &'static str, value: usize) -> Result<u32, Error> {
    u32::try_from(value).map_err(|_| Error::DimensionOverflow { context, value })
}

/// Elements in `shape`, failing on overflow
fn count(context: &'static str, shape: &[usize]) -> Result<usize, Error> {
    element_count(context, shape)
}

fn len<T>(view: &CudaView<'_, T>) -> u64 {
    view.len() as u64
}

fn len_mut<T>(view: &CudaViewMut<'_, T>) -> u64 {
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
        count(
            "attention heads",
            &[self.batch, self.frames, self.heads, self.head_dim],
        )
    }
}

/// Checks the buffers of a 3x3 subsampling convolution against its output map
fn check_subsampling(
    input: FeatureMap,
    output: FeatureMap,
    lens: usize,
    weight: usize,
    bias: usize,
    out: usize,
) -> Result<(), Error> {
    if input.batch != output.batch {
        return Err(Error::BufferLength {
            context: "subsampling batch",
            expected: output.batch,
            actual: input.batch,
        });
    }
    check_len("subsampling lengths", output.batch, lens)?;
    check_len(
        "subsampling weight",
        count("subsampling weight", &[output.channels, 9])?,
        weight,
    )?;
    check_len("subsampling bias", output.channels, bias)?;
    check_len("subsampling output", output.len(), out)
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
    #[allow(
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
    #[allow(
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
    #[allow(
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
        let width = count("qkv width", &[shape.heads, shape.head_dim])?;
        check_len("qkv input", count("qkv input", &[3, n])?, qkv.len())?;
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
        let n = count("split heads", &[rows_count, heads, head_dim])?;
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
        let score_rows = count("score rows", &[batch, heads, frames])?;
        check_len(
            "scores",
            count("scores", &[score_rows, frames])?,
            scores.len(),
        )?;
        // one row of 2T - 1 relative positions per query
        let positions = (2 * frames).saturating_sub(1);
        check_len(
            "position scores",
            count("position scores", &[score_rows, positions])?,
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
    #[allow(
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
        let n = count("conv module", &[batch, frames, channels])?;
        check_len(
            "conv pw1 output",
            count("conv pw1 output", &[2, n])?,
            pw1.len(),
        )?;
        check_len("frame lengths", batch, lens_frames.len())?;
        check_len(
            "depthwise weight",
            count("depthwise weight", &[channels, kernel_size])?,
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
    #[allow(
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
            count(
                "subsampling input",
                &[input.batch, input.frames, input.features],
            )?,
            mel.len(),
        )?;
        check_subsampling(
            input,
            output,
            in_lens.len(),
            weight.len(),
            bias.len(),
            out.len(),
        )?;
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
        let dims = [
            u32_of("batch", output.batch)?,
            u32_of("channels", output.channels)?,
            u32_of("frames", input.frames)?,
            u32_of("features", input.features)?,
            u32_of("frames", output.frames)?,
            u32_of("features", output.features)?,
        ];
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
    #[allow(
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
        check_subsampling(
            input,
            output,
            out_lens.len(),
            weight.len(),
            bias.len(),
            out.len(),
        )?;
        let config = elements(output.len())?;
        let lens = [len(x), len(out_lens), len(weight), len(bias), len_mut(out)];
        let dims = [
            u32_of("batch", output.batch)?,
            u32_of("channels", output.channels)?,
            u32_of("frames", input.frames)?,
            u32_of("features", input.features)?,
            u32_of("frames", output.frames)?,
            u32_of("features", output.features)?,
        ];
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

/// Per-row decode state, `[batch]` each
pub(crate) struct DecodeState<'a> {
    pub frame: CudaViewMut<'a, u32>,
    pub emitted: CudaViewMut<'a, u32>,
    pub label: CudaViewMut<'a, u32>,
    pub emit: CudaViewMut<'a, u32>,
    pub counts: CudaViewMut<'a, u32>,
    pub faults: CudaViewMut<'a, u32>,
}

/// Emitted tokens, `[batch, capacity]` each
pub(crate) struct DecodeOutput<'a> {
    pub tokens: CudaViewMut<'a, u32>,
    pub frames: CudaViewMut<'a, u32>,
    pub durations: CudaViewMut<'a, u32>,
    pub confidences: CudaViewMut<'a, f32>,
}

/// Widest language filter `decode_decide` takes, its `MAX_FILTER_TOP_K`
pub(crate) const MAX_FILTER_TOP_K: usize = 256;

/// Language filter of a decode step, as `decode_decide` reads it
pub(crate) struct DecodeFilter<'a> {
    /// `[classes]`, nonzero for the token ids the filter allows
    pub allowed: CudaView<'a, u32>,
    /// `[1]`, the filter's top-K width, or 0 for no filter
    pub top_k: CudaView<'a, u32>,
}

/// Joint and duration geometry of one decode step
#[derive(Debug, Clone, Copy)]
pub(crate) struct DecodeShape {
    pub batch: usize,
    /// Token classes including the blank, which is the last
    pub classes: usize,
    pub max_symbols: usize,
    /// Output slots per row
    pub capacity: usize,
}

/// One predictor LSTM layer's weights, `[4 * hidden, hidden]` each and the summed
/// biases `[4 * hidden]`
pub(crate) struct LstmWeights<'a> {
    pub weight_ih: CudaView<'a, f32>,
    pub weight_hh: CudaView<'a, f32>,
    pub bias: CudaView<'a, f32>,
}

/// Where a predictor layer reads its input rows
pub(crate) enum LstmInput<'a> {
    /// Row `label[b]` of the embedding table `[classes, hidden]`
    Embedded {
        table: CudaView<'a, f32>,
        label: CudaView<'a, u32>,
    },
    /// Row `b` of the layer below's new state, `[batch, hidden]`
    Below(CudaView<'a, f32>),
}

/// One predictor layer's state, `[batch, hidden]` each: the state before the step,
/// the new state, and the cell state updated in place
pub(crate) struct LstmState<'a> {
    pub h: CudaView<'a, f32>,
    pub h_next: CudaViewMut<'a, f32>,
    pub c: CudaViewMut<'a, f32>,
}

/// The buffers `decode_project` reads and writes
pub(crate) struct Projection<'a> {
    /// `[joint_hidden, hidden]` and `[joint_hidden]`
    pub weight: CudaView<'a, f32>,
    pub bias: CudaView<'a, f32>,
    /// The top layer's new state, `[batch, hidden]`
    pub top: CudaView<'a, f32>,
    /// `[batch, frames, joint_hidden]`
    pub enc_proj: CudaView<'a, f32>,
    pub frame: CudaView<'a, u32>,
    pub lens: CudaView<'a, u32>,
    /// `[outputs]`
    pub joint_bias: CudaView<'a, f32>,
    /// Every layer's new and current state, `[layers, batch, hidden]`
    pub h_next: CudaView<'a, f32>,
    pub h: CudaViewMut<'a, f32>,
    /// `[batch, joint_hidden]`
    pub pred_proj: CudaViewMut<'a, f32>,
    pub joint_in: CudaViewMut<'a, f32>,
    /// `[batch, outputs]`
    pub logits: CudaViewMut<'a, f32>,
}

/// The `decode` area's kernels
pub(crate) struct DecodeKernels {
    decide: CudaFunction,
    lstm: CudaFunction,
    project: CudaFunction,
}

impl DecodeKernels {
    pub(crate) fn load(runtime: &CudaRuntime) -> Result<Self, Error> {
        let module: LoadedKernels<KernelModule> = runtime.load_embedded(KernelModule::Decode)?;
        Ok(Self {
            decide: module.function("decode_decide")?,
            lstm: module.function("decode_lstm")?,
            project: module.function("decode_project")?,
        })
    }

    /// Token decision and frame advance for every unfinished row
    ///
    /// A nonzero `filter.top_k` must be at most `shape.classes` and
    /// [`MAX_FILTER_TOP_K`]; the kernel clamps it to stay in bounds, so a wider one
    /// would search fewer candidates than asked
    #[allow(
        clippy::too_many_arguments,
        reason = "launchers mirror their kernel's parameter list so each launch reads against it"
    )]
    pub(crate) fn decide(
        &self,
        stream: &CudaStream,
        logits: &CudaView<'_, f32>,
        lens_frames: &CudaView<'_, u32>,
        durations: &CudaView<'_, u32>,
        filter: &DecodeFilter<'_>,
        state: &mut DecodeState<'_>,
        output: &mut DecodeOutput<'_>,
        shape: DecodeShape,
    ) -> Result<(), Error> {
        let batch = shape.batch;
        let row = shape.classes + durations.len();
        check_len(
            "joint logits",
            count("joint logits", &[batch, row])?,
            logits.len(),
        )?;
        check_len("frame lengths", batch, lens_frames.len())?;
        check_len("allowed tokens", shape.classes, filter.allowed.len())?;
        check_len("language filter", 1, filter.top_k.len())?;
        for (context, buffer) in [
            ("frame cursors", &state.frame),
            ("emitted counts", &state.emitted),
            ("labels", &state.label),
            ("emit flags", &state.emit),
            ("token counts", &state.counts),
            ("faults", &state.faults),
        ] {
            check_len(context, batch, buffer.len())?;
        }
        let slots = count("decode output", &[batch, shape.capacity])?;
        for (context, buffer) in [
            ("tokens", &output.tokens),
            ("token frames", &output.frames),
            ("token durations", &output.durations),
        ] {
            check_len(context, slots, buffer.len())?;
        }
        check_len("confidences", slots, output.confidences.len())?;
        if shape.classes == 0 || durations.is_empty() {
            return Err(Error::BufferLength {
                context: "joint classes and durations",
                expected: 1,
                actual: 0,
            });
        }
        // each thread holds its class logits in registers, and one lane of warp 0
        // reads each duration logit, so wider rows would drop some
        let widest = DECIDE_VALUES * DECIDE_THREADS as usize;
        if shape.classes > widest {
            return Err(Error::BufferLength {
                context: "joint classes",
                expected: widest,
                actual: shape.classes,
            });
        }
        if durations.len() > 32 {
            return Err(Error::BufferLength {
                context: "joint durations",
                expected: 32,
                actual: durations.len(),
            });
        }

        let config = LaunchConfig {
            block_dim: (DECIDE_THREADS, 1, 1),
            ..rows(batch)?
        };
        let lens = [
            len(logits),
            len(lens_frames),
            len(durations),
            len(&filter.allowed),
            len(&filter.top_k),
            len_mut(&state.frame),
            len_mut(&state.emitted),
            len_mut(&state.label),
            len_mut(&state.emit),
            len_mut(&state.counts),
            len_mut(&state.faults),
            len_mut(&output.tokens),
            len_mut(&output.frames),
            len_mut(&output.durations),
            len_mut(&output.confidences),
        ];
        let dims = [
            u32_of("classes", shape.classes)?,
            u32_of("blank", shape.classes - 1)?,
            u32_of("max symbols", shape.max_symbols)?,
            u32_of("capacity", shape.capacity)?,
        ];
        let mut launch = stream.launch_builder(&self.decide);
        launch
            .arg(logits)
            .arg(&lens[0])
            .arg(lens_frames)
            .arg(&lens[1])
            .arg(durations)
            .arg(&lens[2]);
        launch
            .arg(&filter.allowed)
            .arg(&lens[3])
            .arg(&filter.top_k)
            .arg(&lens[4]);
        launch
            .arg(&mut state.frame)
            .arg(&lens[5])
            .arg(&mut state.emitted)
            .arg(&lens[6])
            .arg(&mut state.label)
            .arg(&lens[7]);
        launch
            .arg(&mut state.emit)
            .arg(&lens[8])
            .arg(&mut state.counts)
            .arg(&lens[9])
            .arg(&mut state.faults)
            .arg(&lens[10]);
        launch
            .arg(&mut output.tokens)
            .arg(&lens[11])
            .arg(&mut output.frames)
            .arg(&lens[12])
            .arg(&mut output.durations)
            .arg(&lens[13]);
        launch.arg(&mut output.confidences).arg(&lens[14]);
        for dim in &dims {
            launch.arg(dim);
        }
        // SAFETY: matches `decode_decide`; one block per row writes only its row's
        // state slots and its `capacity` output slots, all sized above
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// One predictor LSTM layer for the rows with `emit` set; the others copy their
    /// state to `state.h_next`
    pub(crate) fn lstm(
        &self,
        stream: &CudaStream,
        weights: &LstmWeights<'_>,
        input: &LstmInput<'_>,
        emit: &CudaView<'_, u32>,
        state: &mut LstmState<'_>,
        hidden: usize,
    ) -> Result<(), Error> {
        check_weight_row(hidden)?;
        let batch = emit.len();
        let n = count("lstm state", &[batch, hidden])?;
        let gates = count("lstm gates", &[4, hidden])?;
        let weight = count("lstm weight", &[gates, hidden])?;
        check_len("lstm input weight", weight, weights.weight_ih.len())?;
        check_len("lstm state weight", weight, weights.weight_hh.len())?;
        check_len("lstm bias", gates, weights.bias.len())?;
        check_len("lstm hidden", n, state.h.len())?;
        check_len("lstm next hidden", n, state.h_next.len())?;
        check_len("lstm cell", n, state.c.len())?;
        // the kernel's input reads are checked slice indexes, so a label past the
        // embedding table traps instead of reading out of bounds
        let (rows, label, embedded) = match input {
            LstmInput::Embedded { table, label } => {
                if hidden == 0 || table.len() % hidden != 0 {
                    return Err(Error::BufferLength {
                        context: "embedding table",
                        expected: hidden.max(1) * table.len().div_ceil(hidden.max(1)),
                        actual: table.len(),
                    });
                }
                check_len("labels", batch, label.len())?;
                (table, label, 1u32)
            }
            // the label is unread without `embedded`; `emit` stands in for it
            LstmInput::Below(below) => {
                check_len("lstm input", n, below.len())?;
                (below, emit, 0u32)
            }
        };

        let config = LaunchConfig {
            grid_dim: (u32_of("lstm units", hidden)?, 1, 1),
            block_dim: (WEIGHT_ROW_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let lens = [
            len(&weights.weight_ih),
            len(&weights.weight_hh),
            len(&weights.bias),
            len(rows),
            len(label),
            len(emit),
            len(&state.h),
            len_mut(&state.h_next),
            len_mut(&state.c),
        ];
        let hidden = u32_of("hidden", hidden)?;
        let mut launch = stream.launch_builder(&self.lstm);
        launch
            .arg(&weights.weight_ih)
            .arg(&lens[0])
            .arg(&weights.weight_hh)
            .arg(&lens[1])
            .arg(&weights.bias)
            .arg(&lens[2]);
        launch
            .arg(rows)
            .arg(&lens[3])
            .arg(label)
            .arg(&lens[4])
            .arg(emit)
            .arg(&lens[5]);
        launch
            .arg(&state.h)
            .arg(&lens[6])
            .arg(&mut state.h_next)
            .arg(&lens[7])
            .arg(&mut state.c)
            .arg(&lens[8]);
        launch.arg(&hidden).arg(&embedded);
        // SAFETY: matches `decode_lstm`; one block per unit writes only that unit's
        // column of `h_next` and `c`, which are sized above; its unchecked reads are
        // within the weights checked above and the rows it slices with a check
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// The predictor's joint projection for the rows with `emit` set, the next
    /// step's joint input for every row, the joint bias into `logits`, and every
    /// layer's new state back into `h`
    pub(crate) fn project(
        &self,
        stream: &CudaStream,
        buffers: &mut Projection<'_>,
        emit: &CudaView<'_, u32>,
        frames: usize,
        hidden: usize,
    ) -> Result<(), Error> {
        check_weight_row(hidden)?;
        let batch = emit.len();
        let joint_hidden = buffers.bias.len();
        let outputs = buffers.joint_bias.len();
        let rows = count("joint input", &[batch, joint_hidden])?;
        if joint_hidden == 0 || outputs == 0 {
            return Err(Error::BufferLength {
                context: "joint widths",
                expected: 1,
                actual: 0,
            });
        }
        check_len(
            "predictor projection weight",
            count("predictor projection weight", &[joint_hidden, hidden])?,
            buffers.weight.len(),
        )?;
        check_len(
            "top state",
            count("top state", &[batch, hidden])?,
            buffers.top.len(),
        )?;
        check_len(
            "encoder projection",
            count("encoder projection", &[rows, frames])?,
            buffers.enc_proj.len(),
        )?;
        check_len("frame cursors", batch, buffers.frame.len())?;
        check_len("frame lengths", batch, buffers.lens.len())?;
        check_len("predictor states", buffers.h.len(), buffers.h_next.len())?;
        check_len("predictor projection", rows, buffers.pred_proj.len())?;
        check_len("joint input", rows, buffers.joint_in.len())?;
        check_len(
            "joint logits",
            count("joint logits", &[batch, outputs])?,
            buffers.logits.len(),
        )?;

        let config = LaunchConfig {
            grid_dim: (u32_of("projection grid", joint_hidden)?, 1, 1),
            block_dim: (WEIGHT_ROW_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let lens = [
            len(&buffers.weight),
            len(&buffers.bias),
            len(&buffers.top),
            len(emit),
            len(&buffers.enc_proj),
            len(&buffers.frame),
            len(&buffers.lens),
            len(&buffers.joint_bias),
            len(&buffers.h_next),
            len_mut(&buffers.h),
            len_mut(&buffers.pred_proj),
            len_mut(&buffers.joint_in),
            len_mut(&buffers.logits),
        ];
        let dims = [u32_of("frames", frames)?, u32_of("hidden", hidden)?];
        let mut launch = stream.launch_builder(&self.project);
        launch
            .arg(&buffers.weight)
            .arg(&lens[0])
            .arg(&buffers.bias)
            .arg(&lens[1])
            .arg(&buffers.top)
            .arg(&lens[2]);
        launch
            .arg(emit)
            .arg(&lens[3])
            .arg(&buffers.enc_proj)
            .arg(&lens[4])
            .arg(&buffers.frame)
            .arg(&lens[5]);
        launch
            .arg(&buffers.lens)
            .arg(&lens[6])
            .arg(&buffers.joint_bias)
            .arg(&lens[7])
            .arg(&buffers.h_next)
            .arg(&lens[8]);
        launch
            .arg(&mut buffers.h)
            .arg(&lens[9])
            .arg(&mut buffers.pred_proj)
            .arg(&lens[10])
            .arg(&mut buffers.joint_in)
            .arg(&lens[11]);
        launch.arg(&mut buffers.logits).arg(&lens[12]);
        launch.arg(&dims[0]).arg(&dims[1]);
        // SAFETY: matches `decode_project`; each block writes only its column of
        // `pred_proj` and `joint_in`, the side copies stay below `logits.len()` and
        // `h.len()`, every buffer is sized above, and its unchecked weight reads are
        // within the weight checked above
        unsafe { launch.launch(config) }?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use cudakit::cudarc::driver::CudaSlice;
    use cudakit::{Error, Options};

    use super::{
        CudaRuntime, DecodeKernels, EncoderKernels, GemmOperand, LstmInput, LstmState, LstmWeights,
        Projection,
    };

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

    /// Predictor sizes for the decode kernel test: 17 rows cross two 8-row passes,
    /// and a hidden width that is not a multiple of a block's 128 threads leaves
    /// some threads a column short
    const BATCH: usize = 17;
    const HIDDEN: usize = 200;
    const LAYERS: usize = 2;
    const CLASSES: usize = 11;
    const JOINT_HIDDEN: usize = 24;
    const OUTPUTS: usize = 7;
    const FRAMES: usize = 3;

    /// Deterministic values in `[-scale, scale)`
    struct Values(u64);

    impl Values {
        fn next(&mut self, scale: f32) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        }

        fn vec(&mut self, len: usize, scale: f32) -> Vec<f32> {
            (0..len).map(|_| self.next(scale)).collect()
        }

        fn below(&mut self, bound: usize) -> u32 {
            ((self.next(1.0) + 1.0) * 0.5 * bound as f32) as u32 % bound as u32
        }
    }

    /// The predictor's weights and the state a step carries over, on the host
    struct Predictor {
        weight_ih: Vec<Vec<f32>>,
        weight_hh: Vec<Vec<f32>>,
        bias: Vec<Vec<f32>>,
        embed: Vec<f32>,
        proj_weight: Vec<f32>,
        proj_bias: Vec<f32>,
        enc_proj: Vec<f32>,
        joint_bias: Vec<f32>,
        h: Vec<f32>,
        c: Vec<f32>,
        pred_proj: Vec<f32>,
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    impl Predictor {
        /// One step on the host: the LSTM layers and the projection for the rows
        /// with `emit` set, then every row's joint input and bias-filled logits
        fn step(&mut self, emit: &[u32], label: &[u32], frame: &[u32], lens: &[u32]) -> Step {
            let n = BATCH * HIDDEN;
            let mut h_next = vec![0.0; LAYERS * n];
            for layer in 0..LAYERS {
                for b in 0..BATCH {
                    let row = layer * n + b * HIDDEN;
                    if emit[b] == 0 {
                        h_next[row..row + HIDDEN].copy_from_slice(&self.h[row..row + HIDDEN]);
                        continue;
                    }

                    let x: Vec<f32> = if layer == 0 {
                        let at = label[b] as usize * HIDDEN;
                        self.embed[at..at + HIDDEN].to_vec()
                    } else {
                        let at = (layer - 1) * n + b * HIDDEN;
                        h_next[at..at + HIDDEN].to_vec()
                    };
                    let gate = |g: usize, unit: usize| {
                        let j = g * HIDDEN + unit;
                        (0..HIDDEN)
                            .map(|k| {
                                self.weight_ih[layer][j * HIDDEN + k] * x[k]
                                    + self.weight_hh[layer][j * HIDDEN + k] * self.h[row + k]
                            })
                            .sum::<f32>()
                            + self.bias[layer][j]
                    };
                    for unit in 0..HIDDEN {
                        let cell = sigmoid(gate(1, unit)) * self.c[row + unit]
                            + sigmoid(gate(0, unit)) * gate(2, unit).tanh();
                        self.c[row + unit] = cell;
                        h_next[row + unit] = sigmoid(gate(3, unit)) * cell.tanh();
                    }
                }
            }

            let top = (LAYERS - 1) * n;
            let mut joint_in = vec![0.0; BATCH * JOINT_HIDDEN];
            for b in 0..BATCH {
                let t = frame[b].min(lens[b] - 1) as usize;
                for col in 0..JOINT_HIDDEN {
                    let i = b * JOINT_HIDDEN + col;
                    if emit[b] != 0 {
                        self.pred_proj[i] = (0..HIDDEN)
                            .map(|k| {
                                self.proj_weight[col * HIDDEN + k] * h_next[top + b * HIDDEN + k]
                            })
                            .sum::<f32>()
                            + self.proj_bias[col];
                    }
                    let encoded = self.enc_proj[(b * FRAMES + t) * JOINT_HIDDEN + col];
                    joint_in[i] = (encoded + self.pred_proj[i]).max(0.0);
                }
            }
            self.h = h_next;

            Step {
                h: self.h.clone(),
                c: self.c.clone(),
                pred_proj: self.pred_proj.clone(),
                joint_in,
                logits: (0..BATCH * OUTPUTS)
                    .map(|i| self.joint_bias[i % OUTPUTS])
                    .collect(),
            }
        }
    }

    /// The state a step leaves for the next one
    struct Step {
        h: Vec<f32>,
        c: Vec<f32>,
        pred_proj: Vec<f32>,
        joint_in: Vec<f32>,
        logits: Vec<f32>,
    }

    /// Holds a device buffer of rows `width` wide to the host's, within float
    /// rounding of a different summation order
    fn assert_close(name: &str, step: usize, width: usize, device: &[f32], host: &[f32]) {
        assert_eq!(device.len(), host.len(), "{name} length");
        for (i, (d, h)) in device.iter().zip(host).enumerate() {
            assert!(
                (d - h).abs() <= 1e-5 + 1e-4 * h.abs(),
                "step {step} {name}[{i}] (row {}): device {d}, host {h}",
                (i / width) % BATCH
            );
        }
    }

    /// `decode_lstm` and `decode_project` over two steps against a host predictor:
    /// rows past the first 8-row pass, rows that skip a step keeping their state and
    /// projection, and the new state copied back for the next step
    #[test]
    #[ignore = "needs a CUDA GPU; `just gpu-parity` runs it on gpu0"]
    fn gpu_predictor_step_matches_the_host() {
        let runtime = CudaRuntime::open(0, Options::default()).expect("CUDA runtime");
        let kernels = DecodeKernels::load(&runtime).expect("kernels");
        let stream = runtime.stream();
        let mut values = Values(7);
        let gates = 4 * HIDDEN;
        let n = BATCH * HIDDEN;
        let mut host = Predictor {
            weight_ih: (0..LAYERS)
                .map(|_| values.vec(gates * HIDDEN, 0.1))
                .collect(),
            weight_hh: (0..LAYERS)
                .map(|_| values.vec(gates * HIDDEN, 0.1))
                .collect(),
            bias: (0..LAYERS).map(|_| values.vec(gates, 0.5)).collect(),
            embed: values.vec(CLASSES * HIDDEN, 1.0),
            proj_weight: values.vec(JOINT_HIDDEN * HIDDEN, 0.1),
            proj_bias: values.vec(JOINT_HIDDEN, 0.5),
            enc_proj: values.vec(BATCH * FRAMES * JOINT_HIDDEN, 1.0),
            joint_bias: values.vec(OUTPUTS, 1.0),
            h: values.vec(LAYERS * n, 0.5),
            c: values.vec(LAYERS * n, 0.5),
            pred_proj: values.vec(BATCH * JOINT_HIDDEN, 0.5),
        };

        let upload = |data: &[f32]| stream.clone_htod(data).expect("upload");
        let weight_ih: Vec<CudaSlice<f32>> = host.weight_ih.iter().map(|w| upload(w)).collect();
        let weight_hh: Vec<CudaSlice<f32>> = host.weight_hh.iter().map(|w| upload(w)).collect();
        let bias: Vec<CudaSlice<f32>> = host.bias.iter().map(|b| upload(b)).collect();
        let embed = upload(&host.embed);
        let proj_weight = upload(&host.proj_weight);
        let proj_bias = upload(&host.proj_bias);
        let enc_proj = upload(&host.enc_proj);
        let joint_bias = upload(&host.joint_bias);
        let mut h = upload(&host.h);
        let mut c = upload(&host.c);
        let mut pred_proj = upload(&host.pred_proj);
        // stale values a step must overwrite
        let mut h_next = upload(&vec![9.0; LAYERS * n]);
        let mut joint_in = upload(&vec![9.0; BATCH * JOINT_HIDDEN]);
        let mut logits = upload(&vec![9.0; BATCH * OUTPUTS]);

        // different rows emit on each step, and the last row of the second pass
        // emits on both
        let emits: [fn(usize) -> bool; 2] = [|b| b % 2 == 0, |b| b % 3 == 0 || b == BATCH - 1];
        for (step, emits) in emits.into_iter().enumerate() {
            let emit: Vec<u32> = (0..BATCH).map(|b| u32::from(emits(b))).collect();
            let label: Vec<u32> = (0..BATCH).map(|_| values.below(CLASSES)).collect();
            let lens: Vec<u32> = (0..BATCH).map(|_| values.below(FRAMES) + 1).collect();
            // past the last frame on some rows, which the projection clamps
            let frame: Vec<u32> = (0..BATCH).map(|_| values.below(FRAMES + 1)).collect();
            let expected = host.step(&emit, &label, &frame, &lens);

            let emit_device = stream.clone_htod(&emit).expect("upload");
            let label_device = stream.clone_htod(&label).expect("upload");
            let frame_device = stream.clone_htod(&frame).expect("upload");
            let lens_device = stream.clone_htod(&lens).expect("upload");
            for layer in 0..LAYERS {
                let weights = LstmWeights {
                    weight_ih: weight_ih[layer].slice(..),
                    weight_hh: weight_hh[layer].slice(..),
                    bias: bias[layer].slice(..),
                };
                let (below, mut rest) = h_next.split_at_mut(layer * n);
                let input = match layer {
                    0 => LstmInput::Embedded {
                        table: embed.slice(..),
                        label: label_device.slice(..),
                    },
                    _ => LstmInput::Below(below.slice((layer - 1) * n..)),
                };
                let mut state = LstmState {
                    h: h.slice(layer * n..(layer + 1) * n),
                    h_next: rest.slice_mut(..n),
                    c: c.slice_mut(layer * n..(layer + 1) * n),
                };
                kernels
                    .lstm(
                        stream,
                        &weights,
                        &input,
                        &emit_device.slice(..),
                        &mut state,
                        HIDDEN,
                    )
                    .expect("lstm");
            }

            let mut buffers = Projection {
                weight: proj_weight.slice(..),
                bias: proj_bias.slice(..),
                top: h_next.slice((LAYERS - 1) * n..),
                enc_proj: enc_proj.slice(..),
                frame: frame_device.slice(..),
                lens: lens_device.slice(..),
                joint_bias: joint_bias.slice(..),
                h_next: h_next.slice(..),
                h: h.slice_mut(..),
                pred_proj: pred_proj.slice_mut(..),
                joint_in: joint_in.slice_mut(..),
                logits: logits.slice_mut(..),
            };
            kernels
                .project(stream, &mut buffers, &emit_device.slice(..), FRAMES, HIDDEN)
                .expect("project");

            let download = |data: &CudaSlice<f32>| stream.clone_dtoh(data).expect("download");
            let device = Step {
                h: download(&h),
                c: download(&c),
                pred_proj: download(&pred_proj),
                joint_in: download(&joint_in),
                logits: download(&logits),
            };
            assert_close("h", step, HIDDEN, &device.h, &expected.h);
            assert_close("c", step, HIDDEN, &device.c, &expected.c);
            assert_close(
                "pred_proj",
                step,
                JOINT_HIDDEN,
                &device.pred_proj,
                &expected.pred_proj,
            );
            assert_close(
                "joint_in",
                step,
                JOINT_HIDDEN,
                &device.joint_in,
                &expected.joint_in,
            );
            assert_close("logits", step, OUTPUTS, &device.logits, &expected.logits);
        }
    }
}
