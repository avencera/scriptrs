//! Launchers for the streaming encoder kernels in `encoder/stream.rs` of the kernel
//! crate
//!
//! The kernels index the cache arenas by the slot each batch row names. Slots live in
//! device buffers so a captured graph replays for any streams, which leaves them
//! unchecked here: the streaming engine gives each row a distinct slot below the
//! arena's count before it uploads them

use cudakit::cudarc::driver::{
    CudaFunction, CudaStream, CudaView, CudaViewMut, LaunchConfig, PushKernelArg,
};
use cudakit::{Error, LoadedKernels, check_len, element_count};

use super::{
    CudaRuntime, FeatureMap, GemmOperand, KernelModule, ROW_VALUES, check_subsampling, elements,
    len, len_mut, subsampling_dims, u32_of,
};

/// Threads per stream attention block while the grid is small, the kernels'
/// `ATTENTION_THREADS`: a warp scores one key at a time, and groups of `head_dim`
/// threads sum the values of every so many keys. A single stream has only
/// `heads * frames` blocks, so each needs many threads to keep enough loads in flight
const ATTENTION_THREADS: u32 = 16 * 32;

/// Grid size from which attention blocks drop to [`NARROW_ATTENTION_THREADS`]: by
/// then the blocks alone fill an RTX 5090, and wide blocks' extra partial sums cost
/// more than they hide (128 streams of 160 ms chunks ran 3% faster with 4 warps
/// than 16, and 64 streams 8% slower)
const NARROW_ATTENTION_BLOCKS: u32 = 2048;

/// Threads per stream attention block once the grid has [`NARROW_ATTENTION_BLOCKS`],
/// or more for a head wider than that
const NARROW_ATTENTION_THREADS: u32 = 4 * 32;

/// Threads per stream conv block, one per channel of a 1024-channel model; a single
/// stream has only a block per chunk frame
const CONV_THREADS: u32 = 1024;

/// Keys a stream attention block scores, the kernels' `MAX_KEYS`
const MAX_KEYS: usize = 256;

/// Widest head a stream attention block holds, the kernels' `MAX_HEAD_DIM`
const MAX_HEAD_DIM: usize = 256;

// every element of the widest head needs its own thread in an attention block
const _: () = assert!(MAX_HEAD_DIM <= ATTENTION_THREADS as usize);

/// The geometry of one streaming batch and its cache arenas
#[derive(Debug, Clone, Copy)]
pub(crate) struct StreamGeometry {
    /// Streams in the batch
    pub batch: usize,
    /// Encoder frames per chunk
    pub frames: usize,
    pub heads: usize,
    pub head_dim: usize,
    /// Cached left-context frames per layer
    pub context: usize,
    pub layers: usize,
    /// Streams the arenas hold
    pub slots: usize,
}

impl StreamGeometry {
    fn d_model(self) -> usize {
        self.heads * self.head_dim
    }

    /// Rows of the chunk's activations, `batch * frames`
    fn rows(self) -> usize {
        self.batch * self.frames
    }

    /// Elements in the key-value arena, `[slots, layers, 2, context, d_model]`
    pub(crate) fn kv_len(self) -> Result<usize, Error> {
        element_count(
            "key-value cache",
            &[self.slots, self.layers, 2, self.context, self.d_model()],
        )
    }

    /// Elements in the conv arena, `[slots, layers, kernel - 1, d_model]`
    pub(crate) fn conv_len(self, kernel_size: usize) -> Result<usize, Error> {
        element_count(
            "conv cache",
            &[
                self.slots,
                self.layers,
                kernel_size.saturating_sub(1),
                self.d_model(),
            ],
        )
    }
}

/// Per-row stream state, `[batch]` each
pub(crate) struct StreamRows<'a> {
    /// Each row's slot in the cache arenas
    pub slots: CudaView<'a, u32>,
    /// Physical index where each ring's next frames go: its oldest frame once the
    /// ring is full, its first unused frame before
    pub rings: CudaView<'a, u32>,
    /// Valid frames in each ring, the newest; 0 on a stream's first chunk
    pub cached: CudaView<'a, u32>,
    /// Real frames in each row's chunk
    pub valid: CudaView<'a, u32>,
}

impl StreamRows<'_> {
    fn check(&self, batch: usize) -> Result<(), Error> {
        check_len("stream slots", batch, self.slots.len())?;
        check_len("stream rings", batch, self.rings.len())?;
        check_len("stream cached frames", batch, self.cached.len())?;
        check_len("stream valid frames", batch, self.valid.len())
    }
}

/// The streaming encoder's kernels, from the `encoder` area
pub(crate) struct StreamKernels {
    conv_in: CudaFunction,
    depthwise: CudaFunction,
    bias_relu: CudaFunction,
    flatten: CudaFunction,
    attention: CudaFunction,
    attention_f16: CudaFunction,
    kv_append: CudaFunction,
    conv: CudaFunction,
    conv_f16: CudaFunction,
    conv_cache: CudaFunction,
    gather: CudaFunction,
    scatter: CudaFunction,
    prompt: CudaFunction,
}

impl StreamKernels {
    pub(crate) fn load(runtime: &CudaRuntime) -> Result<Self, Error> {
        let module: LoadedKernels<KernelModule> = runtime.load_embedded(KernelModule::Encoder)?;
        Ok(Self {
            conv_in: module.function("encoder_stream_conv_in")?,
            depthwise: module.function("encoder_stream_depthwise")?,
            bias_relu: module.function("encoder_stream_bias_relu")?,
            flatten: module.function("encoder_stream_flatten")?,
            attention: module.function("encoder_stream_attention")?,
            attention_f16: module.function("encoder_stream_attention_f16")?,
            kv_append: module.function("encoder_stream_kv_append")?,
            conv: module.function("encoder_stream_conv")?,
            conv_f16: module.function("encoder_stream_conv_f16")?,
            conv_cache: module.function("encoder_stream_conv_cache")?,
            gather: module.function("encoder_stream_gather")?,
            scatter: module.function("encoder_stream_scatter")?,
            prompt: module.function("encoder_stream_prompt")?,
        })
    }

    /// First causal subsampling convolution with bias, window mask and ReLU
    #[allow(
        clippy::too_many_arguments,
        reason = "launchers mirror their kernel's parameter list so each launch reads against it"
    )]
    pub(crate) fn conv_in(
        &self,
        stream: &CudaStream,
        mel: &CudaView<'_, f32>,
        in_windows: &CudaView<'_, u32>,
        out_windows: &CudaView<'_, u32>,
        weight: &CudaView<'_, f32>,
        bias: &CudaView<'_, f32>,
        out: &mut CudaViewMut<'_, f32>,
        input: FeatureMap,
        output: FeatureMap,
    ) -> Result<(), Error> {
        check_len(
            "subsampling input",
            element_count(
                "subsampling input",
                &[input.batch, input.frames, input.features],
            )?,
            mel.len(),
        )?;
        check_windows(output.batch, in_windows)?;
        check_windows(output.batch, out_windows)?;
        check_subsampling(input, output, weight.len(), bias.len(), out.len())?;
        let config = elements(output.len())?;
        let lens = [
            len(mel),
            len(in_windows),
            len(out_windows),
            len(weight),
            len(bias),
            len_mut(out),
        ];
        let dims = subsampling_dims(input, output)?;
        let mut launch = stream.launch_builder(&self.conv_in);
        launch
            .arg(mel)
            .arg(&lens[0])
            .arg(in_windows)
            .arg(&lens[1])
            .arg(out_windows)
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
        // SAFETY: matches `encoder_stream_conv_in`; indexing is checked in the kernel
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Causal depthwise subsampling convolution with bias and window mask
    #[allow(
        clippy::too_many_arguments,
        reason = "launchers mirror their kernel's parameter list so each launch reads against it"
    )]
    pub(crate) fn depthwise(
        &self,
        stream: &CudaStream,
        x: &CudaView<'_, f32>,
        out_windows: &CudaView<'_, u32>,
        weight: &CudaView<'_, f32>,
        bias: &CudaView<'_, f32>,
        out: &mut CudaViewMut<'_, f32>,
        input: FeatureMap,
        output: FeatureMap,
    ) -> Result<(), Error> {
        check_len("depthwise input", input.len(), x.len())?;
        check_windows(output.batch, out_windows)?;
        check_subsampling(input, output, weight.len(), bias.len(), out.len())?;
        let config = elements(output.len())?;
        let lens = [
            len(x),
            len(out_windows),
            len(weight),
            len(bias),
            len_mut(out),
        ];
        let dims = subsampling_dims(input, output)?;
        let mut launch = stream.launch_builder(&self.depthwise);
        launch
            .arg(x)
            .arg(&lens[0])
            .arg(out_windows)
            .arg(&lens[1])
            .arg(weight)
            .arg(&lens[2]);
        launch.arg(bias).arg(&lens[3]).arg(out).arg(&lens[4]);
        for dim in &dims {
            launch.arg(dim);
        }
        // SAFETY: matches `encoder_stream_depthwise`; indexing is checked in the kernel
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Bias, ReLU and window mask after a pointwise subsampling GEMM, in place
    pub(crate) fn bias_relu(
        &self,
        stream: &CudaStream,
        x: &mut CudaViewMut<'_, f32>,
        bias: &CudaView<'_, f32>,
        windows: &CudaView<'_, u32>,
        map: FeatureMap,
    ) -> Result<(), Error> {
        check_len("pointwise output", map.len(), x.len())?;
        check_len("pointwise bias", map.channels, bias.len())?;
        check_windows(map.batch, windows)?;
        let config = elements(map.len())?;
        let lens = [len_mut(x), len(bias), len(windows)];
        let dims = [
            u32_of("channels", map.channels)?,
            u32_of("frames", map.frames)?,
            u32_of("features", map.features)?,
        ];
        let mut launch = stream.launch_builder(&self.bias_relu);
        launch
            .arg(x)
            .arg(&lens[0])
            .arg(bias)
            .arg(&lens[1])
            .arg(windows)
            .arg(&lens[2]);
        launch.arg(&dims[0]).arg(&dims[1]).arg(&dims[2]);
        // SAFETY: matches `encoder_stream_bias_relu`; indexing is checked in the kernel
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Frames `drop..drop + out_frames` of `map` to `[batch * out_frames, channels *
    /// features]`
    pub(crate) fn flatten(
        &self,
        stream: &CudaStream,
        x: &CudaView<'_, f32>,
        out: &mut CudaViewMut<'_, f32>,
        map: FeatureMap,
        out_frames: usize,
        drop: usize,
    ) -> Result<(), Error> {
        if drop + out_frames > map.frames {
            return Err(Error::BufferLength {
                context: "flattened frames",
                expected: map.frames,
                actual: drop + out_frames,
            });
        }
        check_len("flatten input", map.len(), x.len())?;
        check_len(
            "flatten output",
            element_count(
                "flatten output",
                &[map.batch, out_frames, map.channels, map.features],
            )?,
            out.len(),
        )?;
        let config = elements(out.len())?;
        let lens = [len(x), len_mut(out)];
        let dims = [
            u32_of("channels", map.channels)?,
            u32_of("frames", map.frames)?,
            u32_of("features", map.features)?,
            u32_of("frames", out_frames)?,
            u32_of("drop", drop)?,
        ];
        let mut launch = stream.launch_builder(&self.flatten);
        launch.arg(x).arg(&lens[0]).arg(out).arg(&lens[1]);
        for dim in &dims {
            launch.arg(dim);
        }
        // SAFETY: matches `encoder_stream_flatten`; indexing is checked in the kernel
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Chunked rel-pos attention over the cached and chunk frames of `layer`, into
    /// the merged heads `out`, `[batch * frames, d_model]`
    #[allow(
        clippy::too_many_arguments,
        reason = "launchers mirror their kernel's parameter list so each launch reads against it"
    )]
    pub(crate) fn attention(
        &self,
        stream: &CudaStream,
        qkv: &CudaView<'_, f32>,
        kv: &CudaView<'_, f32>,
        pos: &CudaView<'_, f32>,
        bias_u: &CudaView<'_, f32>,
        bias_v: &CudaView<'_, f32>,
        rows: &StreamRows<'_>,
        mut out: GemmOperand<'_>,
        geometry: StreamGeometry,
        layer: usize,
    ) -> Result<(), Error> {
        let g = geometry;
        let d = g.d_model();
        check_layer(g, layer)?;
        if g.context + g.frames > MAX_KEYS {
            return Err(Error::BufferLength {
                context: "stream attention window",
                expected: MAX_KEYS,
                actual: g.context + g.frames,
            });
        }
        if g.heads == 0 || g.head_dim == 0 || g.head_dim > MAX_HEAD_DIM {
            return Err(Error::BufferLength {
                context: "stream attention head dim",
                expected: MAX_HEAD_DIM,
                actual: g.head_dim,
            });
        }
        rows.check(g.batch)?;
        check_len("qkv", element_count("qkv", &[g.rows(), 3, d])?, qkv.len())?;
        check_len("key-value cache", g.kv_len()?, kv.len())?;
        // positions `context + frames - 1` down to `-(frames - 1)`
        check_len(
            "projected positions",
            element_count("projected positions", &[g.context + 2 * g.frames - 1, d])?,
            pos.len(),
        )?;
        check_len("pos_bias_u", d, bias_u.len())?;
        check_len("pos_bias_v", d, bias_v.len())?;
        check_len(
            "attention output",
            element_count("attention output", &[g.rows(), d])?,
            out.len(),
        )?;
        let blocks = u32_of(
            "attention grid",
            element_count("attention grid", &[g.rows(), g.heads])?,
        )?;
        let threads = if blocks < NARROW_ATTENTION_BLOCKS {
            ATTENTION_THREADS
        } else {
            // each element of a head needs its own thread
            NARROW_ATTENTION_THREADS.max(u32_of("head dim", g.head_dim.next_multiple_of(32))?)
        };
        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let lens = [
            len(qkv),
            len(kv),
            len(pos),
            len(bias_u),
            len(bias_v),
            len(&rows.slots),
            len(&rows.rings),
            len(&rows.cached),
            len(&rows.valid),
            out.len() as u64,
        ];
        let dims = [
            u32_of("frames", g.frames)?,
            u32_of("heads", g.heads)?,
            u32_of("head dim", g.head_dim)?,
            u32_of("context", g.context)?,
            u32_of("layers", g.layers)?,
            u32_of("layer", layer)?,
        ];
        // NeMo divides by `math.sqrt(d_k)` rounded to FP32
        let sqrt_dk = (g.head_dim as f64).sqrt() as f32;
        let mut launch = stream.launch_builder(out.pick(&self.attention, &self.attention_f16));
        launch
            .arg(qkv)
            .arg(&lens[0])
            .arg(kv)
            .arg(&lens[1])
            .arg(pos)
            .arg(&lens[2]);
        launch.arg(bias_u).arg(&lens[3]).arg(bias_v).arg(&lens[4]);
        launch
            .arg(&rows.slots)
            .arg(&lens[5])
            .arg(&rows.rings)
            .arg(&lens[6])
            .arg(&rows.cached)
            .arg(&lens[7])
            .arg(&rows.valid)
            .arg(&lens[8]);
        out.push(&mut launch, &lens[9]);
        for dim in &dims {
            launch.arg(dim);
        }
        launch.arg(&sqrt_dk);
        // SAFETY: matches `encoder_stream_attention` and its `_f16` twin; one block
        // per `(row, head)` query of whole warps, at least `head_dim` and at most
        // `ATTENTION_THREADS` threads, the window fits the shared arrays (checked
        // above), and the engine keeps every slot inside the arena
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Writes the chunk's keys and values of `layer` into each row's ring
    pub(crate) fn kv_append(
        &self,
        stream: &CudaStream,
        qkv: &CudaView<'_, f32>,
        kv: &mut CudaViewMut<'_, f32>,
        rows: &StreamRows<'_>,
        geometry: StreamGeometry,
        layer: usize,
    ) -> Result<(), Error> {
        let g = geometry;
        let d = g.d_model();
        check_layer(g, layer)?;
        if g.frames > g.context {
            return Err(Error::BufferLength {
                context: "chunk frames within the cached context",
                expected: g.context,
                actual: g.frames,
            });
        }
        rows.check(g.batch)?;
        check_len("qkv", element_count("qkv", &[g.rows(), 3, d])?, qkv.len())?;
        check_len("key-value cache", g.kv_len()?, kv.len())?;
        let config = elements(element_count("key-value append", &[g.rows(), 2, d])?)?;
        let lens = [len(qkv), len_mut(kv), len(&rows.slots), len(&rows.rings)];
        let dims = [
            u32_of("batch", g.batch)?,
            u32_of("frames", g.frames)?,
            u32_of("d_model", d)?,
            u32_of("context", g.context)?,
            u32_of("layers", g.layers)?,
            u32_of("layer", layer)?,
        ];
        let mut launch = stream.launch_builder(&self.kv_append);
        launch
            .arg(qkv)
            .arg(&lens[0])
            .arg(kv)
            .arg(&lens[1])
            .arg(&rows.slots)
            .arg(&lens[2])
            .arg(&rows.rings)
            .arg(&lens[3]);
        for dim in &dims {
            launch.arg(dim);
        }
        // SAFETY: matches `encoder_stream_kv_append`; the engine keeps every slot
        // inside the arena and distinct across rows
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// GLU, causal depthwise convolution over the cached frames, LayerNorm and SiLU
    #[allow(
        clippy::too_many_arguments,
        reason = "launchers mirror their kernel's parameter list so each launch reads against it"
    )]
    pub(crate) fn conv(
        &self,
        stream: &CudaStream,
        pw1: &CudaView<'_, f32>,
        cache: &CudaView<'_, f32>,
        rows: &StreamRows<'_>,
        weight: &CudaView<'_, f32>,
        norm: (&CudaView<'_, f32>, &CudaView<'_, f32>),
        mut out: GemmOperand<'_>,
        geometry: StreamGeometry,
        kernel_size: usize,
        layer: usize,
        eps: f32,
    ) -> Result<(), Error> {
        let g = geometry;
        let d = g.d_model();
        let (gamma, beta) = norm;
        check_layer(g, layer)?;
        check_conv_module(g, kernel_size)?;
        rows.check(g.batch)?;
        check_len(
            "conv pw1 output",
            element_count("conv pw1 output", &[g.rows(), 2, d])?,
            pw1.len(),
        )?;
        check_len("conv cache", g.conv_len(kernel_size)?, cache.len())?;
        check_len(
            "depthwise weight",
            element_count("depthwise weight", &[d, kernel_size])?,
            weight.len(),
        )?;
        check_len("conv norm gamma", d, gamma.len())?;
        check_len("conv norm beta", d, beta.len())?;
        check_len(
            "conv module output",
            element_count("conv module output", &[g.rows(), d])?,
            out.len(),
        )?;
        let config = LaunchConfig {
            grid_dim: (u32_of("conv grid", g.rows())?, 1, 1),
            block_dim: (CONV_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let lens = [
            len(pw1),
            len(cache),
            len(&rows.slots),
            len(&rows.cached),
            len(weight),
            len(gamma),
            len(beta),
            out.len() as u64,
        ];
        let dims = [
            u32_of("frames", g.frames)?,
            u32_of("channels", d)?,
            u32_of("kernel size", kernel_size)?,
            u32_of("layers", g.layers)?,
            u32_of("layer", layer)?,
        ];
        let mut launch = stream.launch_builder(out.pick(&self.conv, &self.conv_f16));
        launch
            .arg(pw1)
            .arg(&lens[0])
            .arg(cache)
            .arg(&lens[1])
            .arg(&rows.slots)
            .arg(&lens[2])
            .arg(&rows.cached)
            .arg(&lens[3]);
        launch
            .arg(weight)
            .arg(&lens[4])
            .arg(gamma)
            .arg(&lens[5])
            .arg(beta)
            .arg(&lens[6]);
        out.push(&mut launch, &lens[7]);
        for dim in &dims {
            launch.arg(dim);
        }
        launch.arg(&eps);
        // SAFETY: matches `encoder_stream_conv` and its `_f16` twin; one block per
        // row, `d_model` fits the threads' registers (checked above), and the engine
        // keeps every slot inside the arena
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Moves each row's conv cache of `layer` on by its chunk's GLU frames
    #[allow(
        clippy::too_many_arguments,
        reason = "launchers mirror their kernel's parameter list so each launch reads against it"
    )]
    pub(crate) fn conv_cache(
        &self,
        stream: &CudaStream,
        pw1: &CudaView<'_, f32>,
        cache: &mut CudaViewMut<'_, f32>,
        rows: &StreamRows<'_>,
        geometry: StreamGeometry,
        kernel_size: usize,
        layer: usize,
    ) -> Result<(), Error> {
        let g = geometry;
        let d = g.d_model();
        check_layer(g, layer)?;
        check_conv_module(g, kernel_size)?;
        rows.check(g.batch)?;
        check_len(
            "conv pw1 output",
            element_count("conv pw1 output", &[g.rows(), 2, d])?,
            pw1.len(),
        )?;
        check_len("conv cache", g.conv_len(kernel_size)?, cache.len())?;
        let config = elements(element_count("conv cache update", &[g.batch, d])?)?;
        let lens = [
            len(pw1),
            len_mut(cache),
            len(&rows.slots),
            len(&rows.cached),
        ];
        let dims = [
            u32_of("batch", g.batch)?,
            u32_of("frames", g.frames)?,
            u32_of("channels", d)?,
            u32_of("kernel size", kernel_size)?,
            u32_of("layers", g.layers)?,
            u32_of("layer", layer)?,
        ];
        let mut launch = stream.launch_builder(&self.conv_cache);
        launch
            .arg(pw1)
            .arg(&lens[0])
            .arg(cache)
            .arg(&lens[1])
            .arg(&rows.slots)
            .arg(&lens[2])
            .arg(&rows.cached)
            .arg(&lens[3]);
        for dim in &dims {
            launch.arg(dim);
        }
        // SAFETY: matches `encoder_stream_conv_cache`; the engine keeps every slot
        // inside the arena and distinct across rows
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// `x[b, t, :] = relu(x[b, t, :] + table[prompts[b], :])` over `x` of `[batch,
    /// frames, width]`: the prompt projector's first layer after its GEMM
    pub(crate) fn prompt(
        &self,
        stream: &CudaStream,
        x: &mut CudaViewMut<'_, f32>,
        table: &CudaView<'_, f32>,
        prompts: &CudaView<'_, u32>,
        frames: usize,
    ) -> Result<(), Error> {
        let batch = prompts.len();
        let rows = element_count("prompt rows", &[batch, frames])?;
        let width = x.len().checked_div(rows).unwrap_or(0);
        check_len(
            "prompt input",
            element_count("prompt input", &[rows, width])?,
            x.len(),
        )?;
        if width == 0 || table.len() % width != 0 {
            return Err(Error::BufferLength {
                context: "prompt table",
                expected: width,
                actual: table.len(),
            });
        }
        let config = elements(x.len())?;
        let lens = [len_mut(x), len(table), len(prompts)];
        let dims = [u32_of("frames", frames)?, u32_of("width", width)?];
        let mut launch = stream.launch_builder(&self.prompt);
        launch
            .arg(x)
            .arg(&lens[0])
            .arg(table)
            .arg(&lens[1])
            .arg(prompts)
            .arg(&lens[2]);
        launch.arg(&dims[0]).arg(&dims[1]);
        // SAFETY: matches `encoder_stream_prompt`; the kernel indexes its slices
        // checked, so a prompt past the table traps instead of reading past it
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// `rows[g, b, :] = slots_state[slots[b], g, :]`: each row's state from its slot,
    /// for `rows` of `[groups, batch, width]`
    pub(crate) fn gather(
        &self,
        stream: &CudaStream,
        slots_state: &CudaView<'_, f32>,
        slots: &CudaView<'_, u32>,
        rows: &mut CudaViewMut<'_, f32>,
        groups: usize,
    ) -> Result<(), Error> {
        let (batch, width) = state_shape(slots.len(), groups, rows.len(), slots_state.len())?;
        let config = elements(rows.len())?;
        let lens = [len(slots_state), len(slots), len_mut(rows)];
        let dims = [
            u32_of("batch", batch)?,
            u32_of("groups", groups)?,
            u32_of("width", width)?,
        ];
        let mut launch = stream.launch_builder(&self.gather);
        launch
            .arg(slots_state)
            .arg(&lens[0])
            .arg(slots)
            .arg(&lens[1])
            .arg(rows)
            .arg(&lens[2]);
        launch.arg(&dims[0]).arg(&dims[1]).arg(&dims[2]);
        // SAFETY: matches `encoder_stream_gather`; the kernel indexes its slices
        // checked
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// `slots_state[slots[b], g, :] = rows[g, b, :]`: each row's state back to its
    /// slot
    pub(crate) fn scatter(
        &self,
        stream: &CudaStream,
        rows: &CudaView<'_, f32>,
        slots: &CudaView<'_, u32>,
        slots_state: &mut CudaViewMut<'_, f32>,
        groups: usize,
    ) -> Result<(), Error> {
        let (batch, width) = state_shape(slots.len(), groups, rows.len(), slots_state.len())?;
        let config = elements(rows.len())?;
        let lens = [len(rows), len(slots), len_mut(slots_state)];
        let dims = [
            u32_of("batch", batch)?,
            u32_of("groups", groups)?,
            u32_of("width", width)?,
        ];
        let mut launch = stream.launch_builder(&self.scatter);
        launch
            .arg(rows)
            .arg(&lens[0])
            .arg(slots)
            .arg(&lens[1])
            .arg(slots_state)
            .arg(&lens[2]);
        launch.arg(&dims[0]).arg(&dims[1]).arg(&dims[2]);
        // SAFETY: matches `encoder_stream_scatter`; the caller keeps every slot inside
        // `slots_state` and distinct across rows
        unsafe { launch.launch(config) }?;
        Ok(())
    }
}

/// `(batch, width)` of a gather or scatter between `rows`, `[groups, batch, width]`,
/// and a whole number of slots of `[groups, width]`
fn state_shape(
    batch: usize,
    groups: usize,
    rows: usize,
    slots_state: usize,
) -> Result<(usize, usize), Error> {
    let per_row = element_count("state rows", &[groups, batch])?;
    if per_row == 0 || !rows.is_multiple_of(per_row) {
        return Err(Error::BufferLength {
            context: "state rows",
            expected: per_row,
            actual: rows,
        });
    }
    let width = rows / per_row;
    let slot = element_count("state slot", &[groups, width])?;
    if !slots_state.is_multiple_of(slot) {
        return Err(Error::BufferLength {
            context: "state slots",
            expected: slot * slots_state.div_ceil(slot),
            actual: slots_state,
        });
    }
    Ok((batch, width))
}

/// `[batch, 2]` frame windows
fn check_windows(batch: usize, windows: &CudaView<'_, u32>) -> Result<(), Error> {
    check_len(
        "frame windows",
        element_count("frame windows", &[batch, 2])?,
        windows.len(),
    )
}

fn check_layer(geometry: StreamGeometry, layer: usize) -> Result<(), Error> {
    if layer >= geometry.layers {
        return Err(Error::BufferLength {
            context: "encoder layer",
            expected: geometry.layers,
            actual: layer,
        });
    }
    Ok(())
}

/// A conv module the stream conv kernels can run: a kernel with past taps, and
/// channels that fit one row block's registers
fn check_conv_module(geometry: StreamGeometry, kernel_size: usize) -> Result<(), Error> {
    let widest = ROW_VALUES * CONV_THREADS as usize;
    if kernel_size < 2 || geometry.d_model() > widest {
        return Err(Error::BufferLength {
            context: "stream conv module",
            expected: widest,
            actual: geometry.d_model(),
        });
    }
    Ok(())
}
