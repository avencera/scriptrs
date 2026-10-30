//! FastConformer encoder glue around the cuBLAS GEMMs
//!
//! Every kernel computes in FP32 and mirrors one NeMo operation, so the reference
//! activations in `/shared/datasets/parakeet-ref` check them one at a time. The `_f16`
//! variants write the same values rounded to FP16 bits for the FP16 GEMMs that read
//! them. Layouts are row-major: activations `[batch * frames, width]`, attention heads
//! `[batch, heads, frames, head_dim]` and subsampling feature maps `[batch, channels,
//! frames, features]`. Lengths are per batch row, in frames of the tensor they mask

use cuda_device::{DisjointSlice, SharedArray, kernel, thread};

use crate::common::{MAX_WARPS, add, block_reduce, from_f16, sigmoid, to_f16};

pub mod stream;

fn max(a: f32, b: f32) -> f32 {
    a.max(b)
}

/// Row values each thread of a LayerNorm block holds in registers, so a row of up
/// to `ROW_VALUES * blockDim.x` values is read from memory once
pub const ROW_VALUES: usize = 8;

/// Loads this thread's columns of a row, `tid + n * threads` for slot `n`, with 0
/// past the row
#[inline(always)]
fn load_row(x: &[f32], row: usize, width: usize) -> [f32; ROW_VALUES] {
    let tid = thread::threadIdx_x() as usize;
    let threads = thread::blockDim_x() as usize;
    let mut values = [0.0f32; ROW_VALUES];
    let mut n = 0;
    while n < ROW_VALUES {
        let i = tid + n * threads;
        if i < width {
            values[n] = x[row + i];
        }
        n += 1;
    }
    values
}

/// The row's mean and `1 / sqrt(var + eps)` from every thread's `load_row` values;
/// the variance is biased and taken around the mean in a second pass. Each thread
/// adds its columns in index order, as a strided loop over the row would
#[inline(always)]
fn row_moments(
    values: &[f32; ROW_VALUES],
    width: usize,
    eps: f32,
    scratch: *mut f32,
) -> (f32, f32) {
    let tid = thread::threadIdx_x() as usize;
    let threads = thread::blockDim_x() as usize;
    let mut sum = 0.0f32;
    let mut n = 0;
    while n < ROW_VALUES && tid + n * threads < width {
        sum += values[n];
        n += 1;
    }
    let mean = block_reduce(sum, scratch, add) / width as f32;

    let mut squares = 0.0f32;
    let mut n = 0;
    while n < ROW_VALUES && tid + n * threads < width {
        let centered = values[n] - mean;
        squares += centered * centered;
        n += 1;
    }
    let inv_std = 1.0 / (block_reduce(squares, scratch, add) / width as f32 + eps).sqrt();
    (mean, inv_std)
}

/// LayerNorm over rows of `width`: `out = (x - mean) / sqrt(var + eps) * gamma + beta`
///
/// One block per row, 32 to 1024 threads in multiples of 32, and `width` at most
/// `ROW_VALUES` per thread
#[kernel]
pub fn encoder_layer_norm(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    mut out: DisjointSlice<f32>,
    width: u32,
    eps: f32,
) {
    static mut SCRATCH: SharedArray<f32, MAX_WARPS> = SharedArray::UNINIT;
    // SAFETY: a raw pointer to the block's shared array; only `block_reduce` touches
    // it, between barriers
    let scratch = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };
    let width = width as usize;
    let row = thread::blockIdx_x() as usize * width;
    let tid = thread::threadIdx_x() as usize;
    let threads = thread::blockDim_x() as usize;

    let values = load_row(x, row, width);
    let (mean, inv_std) = row_moments(&values, width, eps, scratch);
    let mut n = 0;
    while n < ROW_VALUES && tid + n * threads < width {
        let i = tid + n * threads;
        // SAFETY: rows are disjoint across blocks and `i` is distinct within the block
        let slot = unsafe { out.get_unchecked_mut(row + i) };
        *slot = (values[n] - mean) * inv_std * gamma[i] + beta[i];
        n += 1;
    }
}

/// [`encoder_layer_norm`] with its output rounded to FP16
#[kernel]
pub fn encoder_layer_norm_f16(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    mut out: DisjointSlice<u16>,
    width: u32,
    eps: f32,
) {
    static mut SCRATCH: SharedArray<f32, MAX_WARPS> = SharedArray::UNINIT;
    // SAFETY: a raw pointer to the block's shared array; only `block_reduce` touches
    // it, between barriers
    let scratch = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };
    let width = width as usize;
    let row = thread::blockIdx_x() as usize * width;
    let tid = thread::threadIdx_x() as usize;
    let threads = thread::blockDim_x() as usize;

    let values = load_row(x, row, width);
    let (mean, inv_std) = row_moments(&values, width, eps, scratch);
    let mut n = 0;
    while n < ROW_VALUES && tid + n * threads < width {
        let i = tid + n * threads;
        // SAFETY: rows are disjoint across blocks and `i` is distinct within the block
        let slot = unsafe { out.get_unchecked_mut(row + i) };
        *slot = to_f16((values[n] - mean) * inv_std * gamma[i] + beta[i]);
        n += 1;
    }
}

/// A layer's closing LayerNorm of row `row` of `x` in place, as the LayerNorm pair
/// kernels run it, returning this thread's normalized slots
#[inline(always)]
fn closing_norm(
    x: &mut DisjointSlice<f32>,
    gamma: &[f32],
    beta: &[f32],
    row: usize,
    width: usize,
    eps: f32,
    scratch: *mut f32,
) -> [f32; ROW_VALUES] {
    let tid = thread::threadIdx_x() as usize;
    let threads = thread::blockDim_x() as usize;
    let mut values = [0.0f32; ROW_VALUES];
    let mut n = 0;
    while n < ROW_VALUES && tid + n * threads < width {
        // SAFETY: rows are disjoint across blocks and `tid + n * threads` is distinct
        // within the block, so each thread reads and later writes only its own slots
        values[n] = unsafe { *x.get_unchecked_mut(row + tid + n * threads) };
        n += 1;
    }
    let (mean, inv_std) = row_moments(&values, width, eps, scratch);
    let mut n = 0;
    while n < ROW_VALUES && tid + n * threads < width {
        let i = tid + n * threads;
        values[n] = (values[n] - mean) * inv_std * gamma[i] + beta[i];
        // SAFETY: as above
        unsafe { *x.get_unchecked_mut(row + i) = values[n] };
        n += 1;
    }
    values
}

/// A layer's closing LayerNorm in place, `x = LayerNorm(x)` with `gamma` and `beta`,
/// then with `next` set the next layer's first LayerNorm of the result into `out`
/// with `next_gamma` and `next_beta`
///
/// Saves the next layer a launch and a pass over the residual stream, and the
/// closing norm the copy back from a scratch row. Each norm is computed as
/// `encoder_layer_norm` computes it, with the same launch shape
#[kernel]
pub fn encoder_layer_norm_pair(
    mut x: DisjointSlice<f32>,
    gamma: &[f32],
    beta: &[f32],
    next_gamma: &[f32],
    next_beta: &[f32],
    mut out: DisjointSlice<f32>,
    width: u32,
    eps: f32,
    next: u32,
) {
    static mut SCRATCH: SharedArray<f32, MAX_WARPS> = SharedArray::UNINIT;
    // SAFETY: a raw pointer to the block's shared array; only `block_reduce` touches
    // it, between barriers
    let scratch = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };
    let width = width as usize;
    let row = thread::blockIdx_x() as usize * width;
    let tid = thread::threadIdx_x() as usize;
    let threads = thread::blockDim_x() as usize;

    let values = closing_norm(&mut x, gamma, beta, row, width, eps, scratch);
    if next == 0 {
        return;
    }

    let (mean, inv_std) = row_moments(&values, width, eps, scratch);
    let mut n = 0;
    while n < ROW_VALUES && tid + n * threads < width {
        let i = tid + n * threads;
        // SAFETY: rows are disjoint across blocks and `i` is distinct within the
        // block; the host sizes `out` like `x`
        let slot = unsafe { out.get_unchecked_mut(row + i) };
        *slot = (values[n] - mean) * inv_std * next_gamma[i] + next_beta[i];
        n += 1;
    }
}

/// [`encoder_layer_norm_pair`] with the next layer's norm rounded to FP16 for the
/// FP16 GEMM that reads it
#[kernel]
pub fn encoder_layer_norm_pair_f16(
    mut x: DisjointSlice<f32>,
    gamma: &[f32],
    beta: &[f32],
    next_gamma: &[f32],
    next_beta: &[f32],
    mut out: DisjointSlice<u16>,
    width: u32,
    eps: f32,
    next: u32,
) {
    static mut SCRATCH: SharedArray<f32, MAX_WARPS> = SharedArray::UNINIT;
    // SAFETY: a raw pointer to the block's shared array; only `block_reduce` touches
    // it, between barriers
    let scratch = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };
    let width = width as usize;
    let row = thread::blockIdx_x() as usize * width;
    let tid = thread::threadIdx_x() as usize;
    let threads = thread::blockDim_x() as usize;

    let values = closing_norm(&mut x, gamma, beta, row, width, eps, scratch);
    if next == 0 {
        return;
    }

    let (mean, inv_std) = row_moments(&values, width, eps, scratch);
    let mut n = 0;
    while n < ROW_VALUES && tid + n * threads < width {
        let i = tid + n * threads;
        // SAFETY: rows are disjoint across blocks and `i` is distinct within the
        // block; the host sizes `out` like `x`
        let slot = unsafe { out.get_unchecked_mut(row + i) };
        *slot = to_f16((values[n] - mean) * inv_std * next_gamma[i] + next_beta[i]);
        n += 1;
    }
}

/// SiLU (Swish) in place, `x * sigmoid(x)`, one thread per element
#[kernel]
pub fn encoder_silu(mut x: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(slot) = x.get_mut(idx) {
        let value = *slot;
        *slot = value * sigmoid(value);
    }
}

/// SiLU of FP16 `x` in place, computed in FP32, one thread per element: the FP16
/// first linear of a feed-forward module writes FP16 directly, so its activation
/// never round-trips through FP32 memory
#[kernel]
pub fn encoder_silu_h16(mut x: DisjointSlice<u16>) {
    let idx = thread::index_1d();
    if let Some(slot) = x.get_mut(idx) {
        let value = from_f16(*slot);
        *slot = to_f16(value * sigmoid(value));
    }
}

/// Rounds `x` to FP16 into `out`, one thread per element
#[kernel]
pub fn encoder_to_f16(x: &[f32], mut out: DisjointSlice<u16>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(slot) = out.get_mut(idx) {
        *slot = to_f16(x[i]);
    }
}

/// Splits the fused `[q | k | v]` projection into per-head operands for the
/// attention GEMMs and adds the positional biases
///
/// `qkv` is `[batch * frames, 3 * heads * head_dim]`, each third laid out
/// `[heads, head_dim]` as NeMo's `view(batch, -1, heads, d_k)`. Writes
/// `q_u = q + pos_bias_u` and `k`, `v` as `[batch, heads, frames, head_dim]`, and
/// `q_v = q + pos_bias_v` as `[heads, batch, frames, head_dim]` so one strided batch
/// over heads multiplies every utterance by the shared position table. One thread
/// per element of `q`
#[kernel]
pub fn encoder_split_qkv(
    qkv: &[f32],
    bias_u: &[f32],
    bias_v: &[f32],
    mut q_u: DisjointSlice<f32>,
    mut q_v: DisjointSlice<f32>,
    mut k: DisjointSlice<f32>,
    mut v: DisjointSlice<f32>,
    batch: u32,
    frames: u32,
    heads: u32,
    head_dim: u32,
) {
    let (batch, frames, heads, head_dim) =
        (batch as usize, frames as usize, heads as usize, head_dim as usize);
    let width = heads * head_dim;
    let i = thread::index_1d().get();
    if i >= batch * frames * width {
        return;
    }

    let column = i % width;
    let row = i / width;
    let (b, t) = (row / frames, row % frames);
    let (h, d) = (column / head_dim, column % head_dim);
    let source = row * 3 * width + column;
    let q = qkv[source];
    let by_batch = ((b * heads + h) * frames + t) * head_dim + d;
    let by_head = ((h * batch + b) * frames + t) * head_dim + d;

    // SAFETY: `(b, t, h, d)` is a bijection on the thread index, so each thread owns
    // one element of every output
    unsafe {
        *q_u.get_unchecked_mut(by_batch) = q + bias_u[column];
        *q_v.get_unchecked_mut(by_head) = q + bias_v[column];
        *k.get_unchecked_mut(by_batch) = qkv[source + width];
        *v.get_unchecked_mut(by_batch) = qkv[source + 2 * width];
    }
}

/// Reorders the projected position table `[positions, heads * head_dim]` to
/// `[heads, positions, head_dim]`, one thread per element
#[kernel]
pub fn encoder_split_heads(
    x: &[f32],
    mut out: DisjointSlice<f32>,
    rows: u32,
    heads: u32,
    head_dim: u32,
) {
    let (rows, heads, head_dim) = (rows as usize, heads as usize, head_dim as usize);
    let width = heads * head_dim;
    let i = thread::index_1d().get();
    if i >= rows * width {
        return;
    }

    let (row, column) = (i / width, i % width);
    let (h, d) = (column / head_dim, column % head_dim);
    // SAFETY: `(row, h, d)` is a bijection on the thread index
    unsafe { *out.get_unchecked_mut((h * rows + row) * head_dim + d) = x[i] };
}

/// Index in `[batch, heads, frames, head_dim]` of element `i` of the merged
/// `[batch * frames, heads * head_dim]`
#[inline(always)]
fn merged_source(i: usize, frames: usize, heads: usize, head_dim: usize) -> usize {
    let width = heads * head_dim;
    let (row, column) = (i / width, i % width);
    let (b, t) = (row / frames, row % frames);
    let (h, d) = (column / head_dim, column % head_dim);
    ((b * heads + h) * frames + t) * head_dim + d
}

/// Merges `[batch, heads, frames, head_dim]` back to `[batch * frames, heads *
/// head_dim]` for the output projection, one thread per element
#[kernel]
pub fn encoder_merge_heads(
    x: &[f32],
    mut out: DisjointSlice<f32>,
    batch: u32,
    frames: u32,
    heads: u32,
    head_dim: u32,
) {
    let (batch, frames, heads, head_dim) =
        (batch as usize, frames as usize, heads as usize, head_dim as usize);
    let i = thread::index_1d().get();
    if i >= batch * frames * heads * head_dim {
        return;
    }

    // SAFETY: `(b, t, h, d)` is a bijection on the thread index
    unsafe { *out.get_unchecked_mut(i) = x[merged_source(i, frames, heads, head_dim)] };
}

/// [`encoder_merge_heads`] with its output rounded to FP16
#[kernel]
pub fn encoder_merge_heads_f16(
    x: &[f32],
    mut out: DisjointSlice<u16>,
    batch: u32,
    frames: u32,
    heads: u32,
    head_dim: u32,
) {
    let (batch, frames, heads, head_dim) =
        (batch as usize, frames as usize, heads as usize, head_dim as usize);
    let i = thread::index_1d().get();
    if i >= batch * frames * heads * head_dim {
        return;
    }

    let value = x[merged_source(i, frames, heads, head_dim)];
    // SAFETY: `(b, t, h, d)` is a bijection on the thread index
    unsafe { *out.get_unchecked_mut(i) = to_f16(value) };
}

/// Relative-position attention weights, in place over the content scores
///
/// `scores` holds `ac = q_u · kᵀ` as `[batch, heads, frames, frames]`; `bd` holds
/// `q_v · pᵀ` as `[heads, batch, frames, 2 * frames - 1]`, with position row `r`
/// meaning relative distance `frames - 1 - r`. NeMo's `rel_shift` reads `bd` at
/// column `frames - 1 - i + j` for query `i` and key `j`. The row becomes
/// `softmax((ac + bd') / sqrt_dk)` over the keys before `lens[b]` and 0 elsewhere,
/// and a query row at or past `lens[b]` becomes all zeros. NeMo masks with -10000
/// before the softmax, and `exp(-10000 - max)` is exactly 0 in FP32, so skipping the
/// masked keys gives the same weights.
///
/// One block per `(batch, head, query)` row, 32 to 1024 threads in multiples of 32
#[kernel]
pub fn encoder_rel_pos_softmax(
    mut scores: DisjointSlice<f32>,
    bd: &[f32],
    lens: &[u32],
    batch: u32,
    heads: u32,
    frames: u32,
    sqrt_dk: f32,
) {
    static mut SCRATCH: SharedArray<f32, MAX_WARPS> = SharedArray::UNINIT;
    // SAFETY: a raw pointer to the block's shared array; only `block_reduce` touches
    // it, between barriers
    let scratch = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };
    let (batch, heads, frames) = (batch as usize, heads as usize, frames as usize);
    let row = thread::blockIdx_x() as usize;
    let query = row % frames;
    let b = row / (heads * frames);
    let h = (row / frames) % heads;
    let len = lens[b] as usize;
    let tid = thread::threadIdx_x() as usize;
    let threads = thread::blockDim_x() as usize;
    let base = row * frames;

    if query >= len {
        let mut j = tid;
        while j < frames {
            // SAFETY: rows are disjoint across blocks and `j` is distinct within the
            // block
            unsafe { *scores.get_unchecked_mut(base + j) = 0.0 };
            j += threads;
        }
        return;
    }

    let positions = 2 * frames - 1;
    let bd_row = ((h * batch + b) * frames + query) * positions + frames - 1 - query;
    let mut row_max = f32::NEG_INFINITY;
    let mut j = tid;
    while j < frames {
        // SAFETY: each thread owns columns `tid + n * threads` of this block's row
        let slot = unsafe { scores.get_unchecked_mut(base + j) };
        if j < len {
            let score = (*slot + bd[bd_row + j]) / sqrt_dk;
            *slot = score;
            row_max = row_max.max(score);
        } else {
            *slot = 0.0;
        }
        j += threads;
    }
    let row_max = block_reduce(row_max, scratch, max);

    let mut sum = 0.0f32;
    let mut j = tid;
    while j < len {
        // SAFETY: as above
        let slot = unsafe { scores.get_unchecked_mut(base + j) };
        let e = (*slot - row_max).exp();
        *slot = e;
        sum += e;
        j += threads;
    }
    let sum = block_reduce(sum, scratch, add);

    let mut j = tid;
    while j < len {
        // SAFETY: as above
        let slot = unsafe { scores.get_unchecked_mut(base + j) };
        *slot /= sum;
        j += threads;
    }
}

/// Channels per conv-glue block, one warp across them so rows load coalesced
pub const CONV_CHANNELS: usize = 32;

/// Output frames per conv-glue block
pub const CONV_FRAMES: usize = 16;

/// Widest depthwise kernel the conv-glue tile holds
pub const CONV_MAX_KERNEL: usize = 31;

/// Threads per conv-glue block
const CONV_THREADS: usize = 256;

/// The conv module between its two pointwise convolutions for one tile of
/// [`CONV_FRAMES`] frames and [`CONV_CHANNELS`] channels, through `store(index,
/// value)`
///
/// GLU depends only on its own frame, so the block computes it once per frame of
/// the tile and its halo into shared memory, and each output then sums its taps
/// from there. The taps add in the same order as a direct sum, and a masked tap adds
/// an exact zero, so the result does not depend on the tiling
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn conv_glue_tile(
    pw1: impl Fn(usize) -> f32,
    lens: &[u32],
    weight: &[f32],
    scale: &[f32],
    shift: &[f32],
    frames: usize,
    channels: usize,
    kernel_size: usize,
    mut store: impl FnMut(usize, f32),
) {
    static mut TILE: SharedArray<f32, { (CONV_FRAMES + CONV_MAX_KERNEL - 1) * CONV_CHANNELS }> =
        SharedArray::UNINIT;
    // SAFETY: a raw pointer to the block's shared array; the loops below index it
    // below `(CONV_FRAMES + kernel_size - 1) * CONV_CHANNELS`, and the host keeps
    // `kernel_size <= CONV_MAX_KERNEL`
    let tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TILE) };
    let c0 = thread::blockIdx_x() as usize * CONV_CHANNELS;
    let t0 = thread::blockIdx_y() as usize * CONV_FRAMES;
    let b = thread::blockIdx_z() as usize;
    let tid = thread::threadIdx_x() as usize;
    let len = (lens[b] as usize).min(frames);
    let half = kernel_size / 2;

    // GLU of frames t0 - half .. t0 + CONV_FRAMES + half, zero outside [0, len) as
    // NeMo's `masked_fill` and the convolution's zero padding
    let span = (CONV_FRAMES + kernel_size - 1) * CONV_CHANNELS;
    let mut j = tid;
    while j < span {
        let (r, cc) = (j / CONV_CHANNELS, j % CONV_CHANNELS);
        let c = c0 + cc;
        let mut glu = 0.0f32;
        if t0 + r >= half && t0 + r - half < len && c < channels {
            let row = b * frames + t0 + r - half;
            let source = row * 2 * channels + c;
            glu = pw1(source) * sigmoid(pw1(source + channels));
        }
        // SAFETY: `j < span`, inside the tile as above
        unsafe { *tile.add(j) = glu };
        j += CONV_THREADS;
    }
    thread::sync_threads();

    let cc = tid % CONV_CHANNELS;
    let c = c0 + cc;
    let mut r = tid / CONV_CHANNELS;
    while r < CONV_FRAMES {
        let t = t0 + r;
        if t < frames && c < channels {
            let mut acc = 0.0f32;
            let mut tap = 0;
            while tap < kernel_size {
                // SAFETY: `r + tap < CONV_FRAMES + kernel_size - 1`, inside the tile
                let glu = unsafe { *tile.add((r + tap) * CONV_CHANNELS + cc) };
                acc += weight[c * kernel_size + tap] * glu;
                tap += 1;
            }
            let normed = acc * scale[c] + shift[c];
            store((b * frames + t) * channels + c, normed * sigmoid(normed));
        }
        r += CONV_THREADS / CONV_CHANNELS;
    }
}

/// The conv module between its two pointwise convolutions: GLU, the pad mask, the
/// depthwise convolution, BatchNorm and SiLU
///
/// `pw1` is `[batch * frames, 2 * channels]` from the first pointwise convolution.
/// GLU gives `a * sigmoid(g)` from its two halves, and frames at or past `lens[b]`
/// become 0 before the depthwise convolution, as NeMo's `masked_fill`. The depthwise
/// convolution has `kernel_size` taps (odd, centered, zero padded, at most
/// [`CONV_MAX_KERNEL`]) and no bias; `scale` and `shift` are the eval-mode BatchNorm
/// folded per channel. Every output frame is written, including padded ones, which
/// NeMo does not mask either. One block of 256 threads per tile of [`CONV_FRAMES`]
/// frames by [`CONV_CHANNELS`] channels, on a grid of (channel tiles, frame tiles,
/// batch)
#[kernel]
pub fn encoder_conv_glue(
    pw1: &[f32],
    lens: &[u32],
    weight: &[f32],
    scale: &[f32],
    shift: &[f32],
    mut out: DisjointSlice<f32>,
    frames: u32,
    channels: u32,
    kernel_size: u32,
) {
    conv_glue_tile(
        |i| pw1[i],
        lens,
        weight,
        scale,
        shift,
        frames as usize,
        channels as usize,
        kernel_size as usize,
        // SAFETY: the tile writes only indexes of valid frames and channels, inside
        // `[batch * frames, channels]`, which the host checks `out` holds
        |i, value| unsafe { *out.get_unchecked_mut(i) = value },
    );
}

/// [`encoder_conv_glue`] with its output rounded to FP16
#[kernel]
pub fn encoder_conv_glue_f16(
    pw1: &[f32],
    lens: &[u32],
    weight: &[f32],
    scale: &[f32],
    shift: &[f32],
    mut out: DisjointSlice<u16>,
    frames: u32,
    channels: u32,
    kernel_size: u32,
) {
    conv_glue_tile(
        |i| pw1[i],
        lens,
        weight,
        scale,
        shift,
        frames as usize,
        channels as usize,
        kernel_size as usize,
        // SAFETY: as in `encoder_conv_glue`
        |i, value| unsafe { *out.get_unchecked_mut(i) = to_f16(value) },
    );
}

/// Fills every row of `out` (`[rows, width]`) with `bias`, so a following GEMM with
/// `beta = 1` adds a linear layer's bias, one thread per element
#[kernel]
pub fn encoder_fill_rows(bias: &[f32], mut out: DisjointSlice<f32>, width: u32) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(slot) = out.get_mut(idx) {
        *slot = bias[i % width as usize];
    }
}

/// First subsampling convolution: 1 to `channels`, 3x3, stride 2, padding 1, with
/// bias, then the time mask and ReLU
///
/// `mel` is `[batch, in_frames, in_features]` (time-major, as NeMo transposes the
/// featurizer output) and frames at or past `in_lens[b]` count as 0. Writes `[batch,
/// channels, out_frames, out_features]` with frames at or past `out_lens[b]` zeroed.
/// One thread per output element
#[kernel]
pub fn encoder_subsample_conv_in(
    mel: &[f32],
    in_lens: &[u32],
    out_lens: &[u32],
    weight: &[f32],
    bias: &[f32],
    mut out: DisjointSlice<f32>,
    batch: u32,
    channels: u32,
    in_frames: u32,
    in_features: u32,
    out_frames: u32,
    out_features: u32,
) {
    let (batch, channels) = (batch as usize, channels as usize);
    let (in_frames, in_features) = (in_frames as usize, in_features as usize);
    let (out_frames, out_features) = (out_frames as usize, out_features as usize);
    let idx = thread::index_1d();
    let i = idx.get();
    if i >= batch * channels * out_frames * out_features {
        return;
    }

    let f = i % out_features;
    let t = (i / out_features) % out_frames;
    let c = (i / (out_features * out_frames)) % channels;
    let b = i / (out_features * out_frames * channels);
    let mut value = 0.0f32;
    if t < out_lens[b] as usize {
        let in_len = (in_lens[b] as usize).min(in_frames);
        let mut acc = bias[c];
        let mut kt = 0;
        while kt < 3 {
            let mut kf = 0;
            while kf < 3 {
                // input index 2t - 1 + k, with padding 1 on both sides
                let (ti, fi) = (2 * t + kt, 2 * f + kf);
                if ti >= 1 && ti - 1 < in_len && fi >= 1 && fi - 1 < in_features {
                    let sample = mel[(b * in_frames + ti - 1) * in_features + fi - 1];
                    acc += weight[(c * 3 + kt) * 3 + kf] * sample;
                }
                kf += 1;
            }
            kt += 1;
        }
        value = acc.max(0.0);
    }

    if let Some(slot) = out.get_mut(idx) {
        *slot = value;
    }
}

/// Depthwise subsampling convolution: 3x3, stride 2, padding 1, with bias, then the
/// time mask
///
/// `x` is `[batch, channels, in_frames, in_features]`, already masked. Writes
/// `[batch, channels, out_frames, out_features]` with frames at or past `out_lens[b]`
/// zeroed. One thread per output element
#[kernel]
pub fn encoder_subsample_depthwise(
    x: &[f32],
    out_lens: &[u32],
    weight: &[f32],
    bias: &[f32],
    mut out: DisjointSlice<f32>,
    batch: u32,
    channels: u32,
    in_frames: u32,
    in_features: u32,
    out_frames: u32,
    out_features: u32,
) {
    let (batch, channels) = (batch as usize, channels as usize);
    let (in_frames, in_features) = (in_frames as usize, in_features as usize);
    let (out_frames, out_features) = (out_frames as usize, out_features as usize);
    let idx = thread::index_1d();
    let i = idx.get();
    if i >= batch * channels * out_frames * out_features {
        return;
    }

    let f = i % out_features;
    let t = (i / out_features) % out_frames;
    let plane = i / (out_features * out_frames);
    let (b, c) = (plane / channels, plane % channels);
    let mut value = 0.0f32;
    if t < out_lens[b] as usize {
        let input = plane * in_frames * in_features;
        let mut acc = bias[c];
        let mut kt = 0;
        while kt < 3 {
            let mut kf = 0;
            while kf < 3 {
                let (ti, fi) = (2 * t + kt, 2 * f + kf);
                if ti >= 1 && ti - 1 < in_frames && fi >= 1 && fi - 1 < in_features {
                    let sample = x[input + (ti - 1) * in_features + fi - 1];
                    acc += weight[(c * 3 + kt) * 3 + kf] * sample;
                }
                kf += 1;
            }
            kt += 1;
        }
        value = acc;
    }

    if let Some(slot) = out.get_mut(idx) {
        *slot = value;
    }
}

/// Pointwise subsampling epilogue in place: bias, ReLU and the time mask over
/// `[batch, channels, frames, features]`, one thread per element
///
/// The pointwise GEMM turns zeroed padding into `bias`, so the mask runs again here,
/// as NeMo masks before the next layer
#[kernel]
pub fn encoder_subsample_bias_relu(
    mut x: DisjointSlice<f32>,
    bias: &[f32],
    lens: &[u32],
    channels: u32,
    frames: u32,
    features: u32,
) {
    let (channels, frames, features) = (channels as usize, frames as usize, features as usize);
    let idx = thread::index_1d();
    let i = idx.get();
    let t = (i / features) % frames;
    let c = (i / (features * frames)) % channels;
    let b = i / (features * frames * channels);
    if let Some(slot) = x.get_mut(idx) {
        *slot = if t < lens[b] as usize {
            (*slot + bias[c]).max(0.0)
        } else {
            0.0
        };
    }
}

/// Flattens `[batch, channels, frames, features]` to `[batch * frames, channels *
/// features]` for the subsampling output projection, as NeMo's `transpose(1, 2)`
/// and `reshape`, one thread per element
#[kernel]
pub fn encoder_subsample_flatten(
    x: &[f32],
    mut out: DisjointSlice<f32>,
    channels: u32,
    frames: u32,
    features: u32,
) {
    let (channels, frames, features) = (channels as usize, frames as usize, features as usize);
    let idx = thread::index_1d();
    let i = idx.get();
    let width = channels * features;
    let (row, column) = (i / width, i % width);
    let (b, t) = (row / frames, row % frames);
    let (c, f) = (column / features, column % features);
    if let Some(slot) = out.get_mut(idx) {
        *slot = x[((b * channels + c) * frames + t) * features + f];
    }
}
