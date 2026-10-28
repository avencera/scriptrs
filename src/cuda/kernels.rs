//! Kernel registry and typed launchers for `crates/scriptrs-cuda-kernels`
//!
//! cuda-oxide passes every slice parameter as a device pointer followed by its `u64`
//! length, so each launcher pushes both, in the kernel's parameter order

use cudakit::cudarc::driver::{
    CudaFunction, CudaStream, CudaView, CudaViewMut, LaunchConfig, PushKernelArg,
};
use cudakit::{
    AreaPtx, EmbeddedPtx, Error, ExtraSymbols, KernelArea, LibraryPolicy, LoadedKernels, Profile,
    PtxTier, Runtime, TierSet,
};

/// Build-wide CUDA policy: cuBLAS is allowed while the encoder is library-first, and
/// only the `sm_75` baseline PTX ships
pub(crate) static PROFILE: Profile = Profile {
    consumer: "scriptrs",
    compiled: TierSet::of(&[PtxTier::Sm75]),
    shipped: TierSet::of(&[PtxTier::Sm75]),
    // one feature builds every tier until per-tier kernels exist
    tier_features: ["cuda", "cuda", "cuda", "cuda"],
    libraries: LibraryPolicy::Allowed,
    extra_symbols: ExtraSymbols::NONE,
};

/// One PTX module per kernel area
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum KernelModule {
    Encoder,
}

impl KernelArea for KernelModule {
    const PROFILE: &'static Profile = &PROFILE;

    fn name(self) -> &'static str {
        match self {
            Self::Encoder => "encoder",
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
        }
    }
}

pub(crate) type CudaRuntime = Runtime<KernelModule>;

/// Threads per block for the one-thread-per-element kernels
const ELEMENT_THREADS: u32 = 256;

/// Threads per block for the row kernels (LayerNorm, softmax)
const ROW_THREADS: u32 = 256;

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
        let config = elements(shape.batch * shape.frames * shape.heads * shape.head_dim)?;
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
        let config = elements(rows_count * heads * head_dim)?;
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
        let config = elements(shape.batch * shape.frames * shape.heads * shape.head_dim)?;
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
        let config = rows(shape.batch * shape.heads * shape.frames)?;
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
        let config = elements(batch * frames * channels)?;
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
