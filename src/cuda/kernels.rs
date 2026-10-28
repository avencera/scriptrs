//! Kernel registry and typed launchers for `crates/scriptrs-cuda-kernels`
//!
//! cuda-oxide passes every slice parameter as a device pointer followed by its `u64`
//! length, so each launcher pushes both, in the kernel's parameter order. Kernels write
//! through unchecked indexes, so every launcher first checks that each buffer holds
//! exactly the elements its shape implies

use cudakit::cudarc::driver::{
    CudaFunction, CudaStream, CudaView, CudaViewMut, LaunchConfig, PushKernelArg,
};
use cudakit::{
    AreaPtx, EmbeddedPtx, Error, ExtraSymbols, KernelArea, LibraryPolicy, LoadedKernels, Profile,
    PtxTier, Runtime, TierSet, check_len, element_count,
};

/// Build-wide CUDA policy: cuBLAS is allowed while the encoder is library-first, and
/// only the `sm_75` baseline PTX ships
pub(crate) static PROFILE: Profile = Profile {
    consumer: "scriptrs",
    compiled: TierSet::of(&[PtxTier::Sm75]),
    shipped: TierSet::of(&[PtxTier::Sm75]),
    // only `cuda` embeds PTX, and only sm75; naming it for a higher tier would tell a
    // user who already enabled `cuda` to enable it, so those get cudakit's generic hint
    tier_features: &[(PtxTier::Sm75, "cuda")],
    libraries: LibraryPolicy::Allowed,
    extra_symbols: ExtraSymbols::NONE,
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

/// Threads per block of `decode_decide`, one block per batch row: a decode step
/// has only `batch` blocks, so a wide block hides the latency of each pass over
/// the class logits
const DECIDE_THREADS: u32 = 1024;

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
    silu: CudaFunction,
    split_qkv: CudaFunction,
    split_heads: CudaFunction,
    merge_heads: CudaFunction,
    rel_pos_softmax: CudaFunction,
    conv_glue: CudaFunction,
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
            silu: module.function("encoder_silu")?,
            split_qkv: module.function("encoder_split_qkv")?,
            split_heads: module.function("encoder_split_heads")?,
            merge_heads: module.function("encoder_merge_heads")?,
            rel_pos_softmax: module.function("encoder_rel_pos_softmax")?,
            conv_glue: module.function("encoder_conv_glue")?,
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
        out: &mut CudaViewMut<'_, f32>,
        width: usize,
        eps: f32,
    ) -> Result<(), Error> {
        if width == 0 || x.len() % width != 0 {
            return Err(Error::BufferLength {
                context: "layer norm input",
                expected: width.max(1) * x.len().div_ceil(width.max(1)),
                actual: x.len(),
            });
        }
        check_len("layer norm gamma", width, gamma.len())?;
        check_len("layer norm beta", width, beta.len())?;
        check_len("layer norm output", x.len(), out.len())?;
        let config = rows(x.len() / width)?;
        let lens = [len(x), len(gamma), len(beta), len_mut(out)];
        let width = u32_of("layer norm width", width)?;
        let mut launch = stream.launch_builder(&self.layer_norm);
        launch
            .arg(x)
            .arg(&lens[0])
            .arg(gamma)
            .arg(&lens[1])
            .arg(beta)
            .arg(&lens[2]);
        launch.arg(out).arg(&lens[3]).arg(&width).arg(&eps);
        // SAFETY: the arguments match `encoder_layer_norm`'s parameters in order, and
        // one block per row of `width` stays inside `x` and `out`
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
        out: &mut CudaViewMut<'_, f32>,
        shape: Heads,
    ) -> Result<(), Error> {
        let n = shape.len()?;
        check_len("merge heads input", n, x.len())?;
        check_len("merge heads output", n, out.len())?;
        let config = elements(n)?;
        let lens = [len(x), len_mut(out)];
        let dims = [
            u32_of("batch", shape.batch)?,
            u32_of("frames", shape.frames)?,
            u32_of("heads", shape.heads)?,
            u32_of("head dim", shape.head_dim)?,
        ];
        let mut launch = stream.launch_builder(&self.merge_heads);
        launch.arg(x).arg(&lens[0]).arg(out).arg(&lens[1]);
        launch
            .arg(&dims[0])
            .arg(&dims[1])
            .arg(&dims[2])
            .arg(&dims[3]);
        // SAFETY: matches `encoder_merge_heads`; indexing is checked in the kernel
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
        out: &mut CudaViewMut<'_, f32>,
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
            len_mut(out),
        ];
        let dims = [
            u32_of("batch", batch)?,
            u32_of("frames", frames)?,
            u32_of("channels", channels)?,
            u32_of("kernel size", kernel_size)?,
        ];
        let mut launch = stream.launch_builder(&self.conv_glue);
        launch
            .arg(pw1)
            .arg(&lens[0])
            .arg(lens_frames)
            .arg(&lens[1])
            .arg(weight)
            .arg(&lens[2]);
        launch
            .arg(scale)
            .arg(&lens[3])
            .arg(shift)
            .arg(&lens[4])
            .arg(out)
            .arg(&lens[5]);
        launch
            .arg(&dims[0])
            .arg(&dims[1])
            .arg(&dims[2])
            .arg(&dims[3]);
        // SAFETY: matches `encoder_conv_glue`; indexing is checked in the kernel
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

/// The `decode` area's kernels
pub(crate) struct DecodeKernels {
    joint_input: CudaFunction,
    decide: CudaFunction,
    embed: CudaFunction,
    lstm_cell: CudaFunction,
}

impl DecodeKernels {
    pub(crate) fn load(runtime: &CudaRuntime) -> Result<Self, Error> {
        let module: LoadedKernels<KernelModule> = runtime.load_embedded(KernelModule::Decode)?;
        Ok(Self {
            joint_input: module.function("decode_joint_input")?,
            decide: module.function("decode_decide")?,
            embed: module.function("decode_embed")?,
            lstm_cell: module.function("decode_lstm_cell")?,
        })
    }

    /// `out[b] = relu(enc_proj[b, frame[b]] + pred_proj[b])`
    #[allow(
        clippy::too_many_arguments,
        reason = "launchers mirror their kernel's parameter list so each launch reads against it"
    )]
    pub(crate) fn joint_input(
        &self,
        stream: &CudaStream,
        enc_proj: &CudaView<'_, f32>,
        pred_proj: &CudaView<'_, f32>,
        frame: &CudaView<'_, u32>,
        lens_frames: &CudaView<'_, u32>,
        out: &mut CudaViewMut<'_, f32>,
        frames: usize,
        hidden: usize,
    ) -> Result<(), Error> {
        let batch = lens_frames.len();
        let n = count("joint input", &[batch, hidden])?;
        check_len(
            "encoder projection",
            count("encoder projection", &[n, frames])?,
            enc_proj.len(),
        )?;
        check_len("predictor projection", n, pred_proj.len())?;
        check_len("frame cursors", batch, frame.len())?;
        check_len("joint input", n, out.len())?;
        let config = elements(n)?;
        let lens = [
            len(enc_proj),
            len(pred_proj),
            len(frame),
            len(lens_frames),
            len_mut(out),
        ];
        let dims = [u32_of("frames", frames)?, u32_of("hidden", hidden)?];
        let mut launch = stream.launch_builder(&self.joint_input);
        launch
            .arg(enc_proj)
            .arg(&lens[0])
            .arg(pred_proj)
            .arg(&lens[1])
            .arg(frame)
            .arg(&lens[2]);
        launch.arg(lens_frames).arg(&lens[3]).arg(out).arg(&lens[4]);
        launch.arg(&dims[0]).arg(&dims[1]);
        // SAFETY: matches `decode_joint_input`; the kernel bounds-checks its index
        // against `out` and the checks above size every input for it
        unsafe { launch.launch(config) }?;
        Ok(())
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

    /// `out[b] = embed[label[b]]`
    pub(crate) fn embed(
        &self,
        stream: &CudaStream,
        embed: &CudaView<'_, f32>,
        label: &CudaView<'_, u32>,
        out: &mut CudaViewMut<'_, f32>,
        hidden: usize,
    ) -> Result<(), Error> {
        check_len(
            "embedding output",
            count("embedding output", &[label.len(), hidden])?,
            out.len(),
        )?;
        if hidden == 0 || embed.len() % hidden != 0 {
            return Err(Error::BufferLength {
                context: "embedding table",
                expected: hidden.max(1) * embed.len().div_ceil(hidden.max(1)),
                actual: embed.len(),
            });
        }
        let config = elements(out.len())?;
        let lens = [len(embed), len(label), len_mut(out)];
        let hidden = u32_of("hidden", hidden)?;
        let mut launch = stream.launch_builder(&self.embed);
        launch
            .arg(embed)
            .arg(&lens[0])
            .arg(label)
            .arg(&lens[1])
            .arg(out)
            .arg(&lens[2]);
        launch.arg(&hidden);
        // SAFETY: matches `decode_embed`; the kernel bounds-checks its index against
        // `out`, and the table read is a checked slice index, so a label past the
        // table traps instead of reading out of bounds
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// LSTM cell update in place for the rows that emitted
    pub(crate) fn lstm_cell(
        &self,
        stream: &CudaStream,
        gates: &CudaView<'_, f32>,
        emit: &CudaView<'_, u32>,
        h: &mut CudaViewMut<'_, f32>,
        c: &mut CudaViewMut<'_, f32>,
        hidden: usize,
    ) -> Result<(), Error> {
        let n = count("lstm state", &[emit.len(), hidden])?;
        check_len("lstm gates", count("lstm gates", &[n, 4])?, gates.len())?;
        check_len("lstm hidden", n, h.len())?;
        check_len("lstm cell", n, c.len())?;
        let config = elements(n)?;
        let lens = [len(gates), len(emit), len_mut(h), len_mut(c)];
        let hidden = u32_of("hidden", hidden)?;
        let mut launch = stream.launch_builder(&self.lstm_cell);
        launch
            .arg(gates)
            .arg(&lens[0])
            .arg(emit)
            .arg(&lens[1])
            .arg(h)
            .arg(&lens[2]);
        launch.arg(c).arg(&lens[3]).arg(&hidden);
        // SAFETY: matches `decode_lstm_cell`; the kernel bounds-checks its index
        // against `h`, and `gates` and `c` are sized for it above
        unsafe { launch.launch(config) }?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use cudakit::{Error, Options};

    use super::{CudaRuntime, EncoderKernels};

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
            &mut out.slice_mut(..),
            4,
            1e-5,
        );
        assert!(
            matches!(result, Err(Error::BufferLength { .. })),
            "{result:?}"
        );
    }
}
