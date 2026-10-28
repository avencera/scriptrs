//! FastConformer encoder glue around the cuBLAS GEMMs
//!
//! Every kernel is FP32 and mirrors one NeMo operation, so the reference activations
//! in `/shared/datasets/parakeet-ref` check them one at a time. Layouts are row-major:
//! activations `[batch * frames, width]`, attention heads `[batch, heads, frames,
//! head_dim]` and subsampling feature maps `[batch, channels, frames, features]`.
//! Lengths are per batch row, in frames of the tensor they mask

use cuda_device::{DisjointSlice, SharedArray, kernel, thread};

use crate::common::{MAX_WARPS, add, block_reduce, sigmoid};

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
    if next == 0 {
        return;
    }

    let (mean, inv_std) = row_moments(&values, width, eps, scratch);
    let mut n = 0;
    while n < ROW_VALUES && tid + n * threads < width {
        let i = tid + n * threads;
        // SAFETY: as above; the host sizes `out` like `x`
        let slot = unsafe { out.get_unchecked_mut(row + i) };
        *slot = (values[n] - mean) * inv_std * next_gamma[i] + next_beta[i];
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
    let width = heads * head_dim;
    let i = thread::index_1d().get();
    if i >= batch * frames * width {
        return;
    }

    let (row, column) = (i / width, i % width);
    let (b, t) = (row / frames, row % frames);
    let (h, d) = (column / head_dim, column % head_dim);
    // SAFETY: `(b, t, h, d)` is a bijection on the thread index
    unsafe { *out.get_unchecked_mut(i) = x[((b * heads + h) * frames + t) * head_dim + d] };
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

/// The conv module between its two pointwise convolutions: GLU, the pad mask, the
/// depthwise convolution, BatchNorm and SiLU
///
/// `pw1` is `[batch * frames, 2 * channels]` from the first pointwise convolution.
/// GLU gives `a * sigmoid(g)` from its two halves, and frames at or past `lens[b]`
/// become 0 before the depthwise convolution, as NeMo's `masked_fill`. The depthwise
/// convolution has `kernel_size` taps (odd, centered, zero padded) and no bias;
/// `scale` and `shift` are the eval-mode BatchNorm folded per channel. Every output
/// frame is written, including padded ones, which NeMo does not mask either. One
/// thread per output element of `[batch * frames, channels]`
#[kernel]
pub fn encoder_conv_glue(
    pw1: &[f32],
    lens: &[u32],
    weight: &[f32],
    scale: &[f32],
    shift: &[f32],
    mut out: DisjointSlice<f32>,
    batch: u32,
    frames: u32,
    channels: u32,
    kernel_size: u32,
) {
    let (batch, frames, channels, kernel_size) =
        (batch as usize, frames as usize, channels as usize, kernel_size as usize);
    let idx = thread::index_1d();
    let i = idx.get();
    if i >= batch * frames * channels {
        return;
    }

    let (row, c) = (i / channels, i % channels);
    let (b, t) = (row / frames, row % frames);
    let len = lens[b] as usize;
    let half = kernel_size / 2;
    let mut acc = 0.0f32;
    let mut tap = 0;
    while tap < kernel_size {
        // frames before 0 or at or past the length are zero after the mask
        if t + tap >= half && t + tap - half < len {
            let source = (b * frames + t + tap - half) * 2 * channels + c;
            let glu = pw1[source] * sigmoid(pw1[source + channels]);
            acc += weight[c * kernel_size + tap] * glu;
        }
        tap += 1;
    }

    let normed = acc * scale[c] + shift[c];
    if let Some(slot) = out.get_mut(idx) {
        *slot = normed * sigmoid(normed);
    }
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
