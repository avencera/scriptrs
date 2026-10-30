//! Cache-aware streaming FastConformer: the kernels that differ from the offline
//! encoder because a chunk carries state from the chunks before it
//!
//! A batch holds one chunk of `frames` encoder frames from each of `batch` streams.
//! Each stream owns a slot in two cache arenas:
//!
//! - `kv`, `[slots, layers, 2, context, d_model]`: every layer's projected keys and
//!   values for its last `context` frames, as a ring. Logical frame `m` (oldest
//!   first) of a stream sits at physical `(ring[b] + m) % context`, and only the
//!   newest `cached[b]` frames are valid
//! - `conv`, `[slots, layers, kernel - 1, d_model]`: every conv module's last
//!   `kernel - 1` GLU frames, oldest first, which its causal depthwise convolution
//!   reads before the chunk's own. A stream's first chunk (`cached[b] == 0`) reads
//!   zeros instead, NeMo's initial cache
//!
//! Subsampling runs on a chunk of mel frames whose valid frames per row and stage are
//! a window `[start, end)` in `windows`, `[batch, 2]`: the frames outside it are
//! zero, as if the chunk had been run alone without padding

use cuda_device::{DisjointSlice, SharedArray, kernel, thread, warp};

use super::{ROW_VALUES, max, row_moments};
use crate::common::{MAX_WARPS, add, block_reduce, sigmoid, to_f16};

/// Attention keys a stream attention block holds scores for: cached plus chunk frames
pub const MAX_KEYS: usize = 256;

/// Widest attention head a stream attention block holds in shared memory
pub const MAX_HEAD_DIM: usize = 256;

/// Most threads in a stream attention block, at least [`MAX_HEAD_DIM`]
pub const ATTENTION_THREADS: usize = 16 * 32;

/// Whether frame `t` lies in row `b`'s `[start, end)` window
#[inline(always)]
fn in_window(windows: &[u32], b: usize, t: usize) -> bool {
    t >= windows[2 * b] as usize && t < windows[2 * b + 1] as usize
}

/// Sum over the warp, returned to every lane
#[inline(always)]
fn warp_sum(value: f32) -> f32 {
    let mut total = value;
    let mut offset = 16u32;
    while offset > 0 {
        total += warp::shuffle_xor_f32(total, offset);
        offset /= 2;
    }
    total
}

/// First causal subsampling convolution: 1 to `channels`, 3x3, stride 2, padded two
/// frames and features before and one after, with bias and ReLU
///
/// `mel` is `[batch, in_frames, in_features]`; samples outside `in_windows` count as
/// 0. Writes `[batch, channels, out_frames, out_features]` with frames outside
/// `out_windows` zeroed. One thread per output element
#[kernel]
pub fn encoder_stream_conv_in(
    mel: &[f32],
    in_windows: &[u32],
    out_windows: &[u32],
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
    if in_window(out_windows, b, t) {
        let (start, end) = (in_windows[2 * b] as usize, in_windows[2 * b + 1] as usize);
        let mut acc = bias[c];
        let mut kt = 0;
        while kt < 3 {
            let mut kf = 0;
            while kf < 3 {
                // input index 2t - 2 + k: two padding frames before, one after
                let (ti, fi) = (2 * t + kt, 2 * f + kf);
                if ti >= start + 2 && ti < end + 2 && fi >= 2 && fi - 2 < in_features {
                    let sample = mel[(b * in_frames + ti - 2) * in_features + fi - 2];
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

/// Causal depthwise subsampling convolution: 3x3, stride 2, padded as
/// [`encoder_stream_conv_in`], with bias
///
/// `x` is `[batch, channels, in_frames, in_features]`, zero outside its windows.
/// Writes `[batch, channels, out_frames, out_features]` with frames outside
/// `out_windows` zeroed. One thread per output element
#[kernel]
pub fn encoder_stream_depthwise(
    x: &[f32],
    out_windows: &[u32],
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
    if in_window(out_windows, b, t) {
        let input = plane * in_frames * in_features;
        let mut acc = bias[c];
        let mut kt = 0;
        while kt < 3 {
            let mut kf = 0;
            while kf < 3 {
                let (ti, fi) = (2 * t + kt, 2 * f + kf);
                if ti >= 2 && ti - 2 < in_frames && fi >= 2 && fi - 2 < in_features {
                    let sample = x[input + (ti - 2) * in_features + fi - 2];
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

/// Pointwise subsampling epilogue in place: bias, ReLU and the window mask over
/// `[batch, channels, frames, features]`, one thread per element
#[kernel]
pub fn encoder_stream_bias_relu(
    mut x: DisjointSlice<f32>,
    bias: &[f32],
    windows: &[u32],
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
        *slot = if in_window(windows, b, t) {
            (*slot + bias[c]).max(0.0)
        } else {
            0.0
        };
    }
}

/// Flattens frames `drop..drop + out_frames` of `[batch, channels, frames, features]`
/// to `[batch * out_frames, channels * features]` for the subsampling output
/// projection, one thread per output element
///
/// NeMo drops the first subsampled frames of every chunk after the first, which the
/// cached mel frames before the chunk produce, and the first chunk's window starts
/// as many frames in, so every row drops the same count
#[kernel]
pub fn encoder_stream_flatten(
    x: &[f32],
    mut out: DisjointSlice<f32>,
    channels: u32,
    frames: u32,
    features: u32,
    out_frames: u32,
    drop: u32,
) {
    let (channels, frames, features) = (channels as usize, frames as usize, features as usize);
    let (out_frames, drop) = (out_frames as usize, drop as usize);
    let idx = thread::index_1d();
    let i = idx.get();
    let width = channels * features;
    let (row, column) = (i / width, i % width);
    let (b, t) = (row / out_frames, row % out_frames);
    let (c, f) = (column / features, column % features);
    if let Some(slot) = out.get_mut(idx) {
        *slot = x[((b * channels + c) * frames + t + drop) * features + f];
    }
}

/// Offset of key `j` of row `b`'s attention window and whether it is in the `kv`
/// cache rather than the chunk's `qkv`; its value is `context * d_model` further in
/// the cache, or `d_model` further in `qkv`
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn key_offset(
    j: usize,
    b: usize,
    slot: usize,
    ring: usize,
    frames: usize,
    d_model: usize,
    context: usize,
    layers: usize,
    layer: usize,
) -> (bool, usize) {
    if j < context {
        let physical = (ring + j) % context;
        let base = ((slot * layers + layer) * 2) * context * d_model;
        (true, base + physical * d_model)
    } else {
        let row = b * frames + j - context;
        (false, row * 3 * d_model + d_model)
    }
}

/// Fills `weights[j]` for the keys of query `i` of row `b`, head `h`: softmax over the
/// valid keys of `((q + u) · k_j + (q + v) · p_{frames - 1 - i + j}) / sqrt_dk`, 0
/// for the rest. Returns false for a query past the row's valid frames, which gets no
/// weights
///
/// Keys are the `context` cached frames then the chunk's `frames`, so query `i` sits
/// at window position `context + i` and key `j` at relative distance
/// `context + i - j`, NeMo's `rel_shift` with a cache. `pos` holds the projected
/// positions `context + frames - 1` down to `-(frames - 1)`, `[context + 2 * frames
/// - 1, d_model]`. A block of whole warps per query; a warp scores one key at a time
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn attention_weights(
    qkv: &[f32],
    kv: &[f32],
    pos: &[f32],
    bias_u: &[f32],
    bias_v: &[f32],
    slots: &[u32],
    rings: &[u32],
    cached: &[u32],
    valid: &[u32],
    frames: usize,
    heads: usize,
    head_dim: usize,
    context: usize,
    layers: usize,
    layer: usize,
    sqrt_dk: f32,
    query_u: *mut f32,
    query_v: *mut f32,
    weights: *mut f32,
    scratch: *mut f32,
) -> bool {
    let block = thread::blockIdx_x() as usize;
    let i = block % frames;
    let h = (block / frames) % heads;
    let b = block / (frames * heads);
    if i >= valid[b] as usize {
        return false;
    }

    let d_model = heads * head_dim;
    let tid = thread::threadIdx_x() as usize;
    let threads = thread::blockDim_x() as usize;
    let q_row = (b * frames + i) * 3 * d_model + h * head_dim;
    let mut d = tid;
    while d < head_dim {
        let q = qkv[q_row + d];
        // SAFETY: `d < head_dim <= MAX_HEAD_DIM`, which the host checks, and each
        // thread writes its own `d`
        unsafe {
            *query_u.add(d) = q + bias_u[h * head_dim + d];
            *query_v.add(d) = q + bias_v[h * head_dim + d];
        }
        d += threads;
    }
    thread::sync_threads();

    let (slot, ring) = (slots[b] as usize, rings[b] as usize);
    let first = context - cached[b] as usize;
    let keys = context + valid[b] as usize;
    let lane = warp::lane_id() as usize;
    let warp_index = tid / 32;
    let warps = threads / 32;
    let mut j = warp_index;
    while j < context + frames {
        let mut score = f32::NEG_INFINITY;
        if j >= first && j < keys {
            let (from_cache, key) =
                key_offset(j, b, slot, ring, frames, d_model, context, layers, layer);
            let p_row = (frames - 1 - i + j) * d_model + h * head_dim;
            let mut content = 0.0f32;
            let mut position = 0.0f32;
            let mut d = lane;
            while d < head_dim {
                let k = if from_cache {
                    kv[key + h * head_dim + d]
                } else {
                    qkv[key + h * head_dim + d]
                };
                // SAFETY: `d < head_dim`, filled above
                unsafe {
                    content += *query_u.add(d) * k;
                    position += *query_v.add(d) * pos[p_row + d];
                }
                d += 32;
            }
            score = (warp_sum(content) + warp_sum(position)) / sqrt_dk;
        }
        if lane == 0 {
            // SAFETY: `j < context + frames <= MAX_KEYS`, which the host checks, and
            // one warp owns each `j`
            unsafe { *weights.add(j) = score };
        }
        j += warps;
    }
    thread::sync_threads();

    let mut row_max = f32::NEG_INFINITY;
    let mut j = first + tid;
    while j < keys {
        // SAFETY: written above
        row_max = row_max.max(unsafe { *weights.add(j) });
        j += threads;
    }
    let row_max = block_reduce(row_max, scratch, max);

    let mut sum = 0.0f32;
    let mut j = tid;
    while j < context + frames {
        // SAFETY: each thread owns its `j`
        unsafe {
            let e = if j >= first && j < keys {
                (*weights.add(j) - row_max).exp()
            } else {
                0.0
            };
            *weights.add(j) = e;
            sum += e;
        }
        j += threads;
    }
    let sum = block_reduce(sum, scratch, add);

    let mut j = tid;
    while j < context + frames {
        // SAFETY: as above
        unsafe { *weights.add(j) /= sum };
        j += threads;
    }
    thread::sync_threads();
    true
}

/// Element `d` of head `h`'s output for the query `attention_weights` scored, over
/// keys `first, first + step, ...` below `keys`: their weighted sum of values
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn attention_value(
    qkv: &[f32],
    kv: &[f32],
    weights: *mut f32,
    b: usize,
    slot: usize,
    ring: usize,
    h: usize,
    d: usize,
    first: usize,
    step: usize,
    keys: usize,
    frames: usize,
    d_model: usize,
    head_dim: usize,
    context: usize,
    layers: usize,
    layer: usize,
) -> f32 {
    let mut acc = 0.0f32;
    let mut j = first;
    while j < keys {
        let (from_cache, key) = key_offset(j, b, slot, ring, frames, d_model, context, layers, layer);
        let column = h * head_dim + d;
        let value = if from_cache {
            kv[key + context * d_model + column]
        } else {
            qkv[key + d_model + column]
        };
        // SAFETY: `j < keys <= MAX_KEYS`, filled by `attention_weights`
        acc += unsafe { *weights.add(j) } * value;
        j += step;
    }
    acc
}

/// Element `threadIdx.x` of the block's output head, for the first `head_dim`
/// threads; 0 for a query `attention_weights` did not score
///
/// One thread per element summing every key waits on one load after another, so
/// the block's threads split into `blockDim.x / head_dim` groups that each sum every
/// so many keys into `partial`, and the first group adds them up
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn attention_output(
    qkv: &[f32],
    kv: &[f32],
    slots: &[u32],
    rings: &[u32],
    cached: &[u32],
    valid: &[u32],
    weights: *mut f32,
    partial: *mut f32,
    scored: bool,
    frames: usize,
    heads: usize,
    head_dim: usize,
    context: usize,
    layers: usize,
    layer: usize,
) -> Option<f32> {
    let block = thread::blockIdx_x() as usize;
    let (h, b) = ((block / frames) % heads, block / (frames * heads));
    let tid = thread::threadIdx_x() as usize;
    let groups = thread::blockDim_x() as usize / head_dim;
    let (group, d) = (tid / head_dim, tid % head_dim);
    if scored && group < groups {
        let (slot, ring) = (slots[b] as usize, rings[b] as usize);
        let first = context - cached[b] as usize;
        let keys = context + valid[b] as usize;
        let sum = attention_value(
            qkv,
            kv,
            weights,
            b,
            slot,
            ring,
            h,
            d,
            first + group,
            groups,
            keys,
            frames,
            heads * head_dim,
            head_dim,
            context,
            layers,
            layer,
        );
        // SAFETY: `tid < blockDim.x <= ATTENTION_THREADS`, which the host launches,
        // and each thread writes its own
        unsafe { *partial.add(tid) = sum };
    }
    thread::sync_threads();
    if tid >= head_dim {
        return None;
    }

    let mut value = 0.0f32;
    if scored {
        let mut group = 0;
        while group < groups {
            // SAFETY: written above, before the barrier
            value += unsafe { *partial.add(group * head_dim + d) };
            group += 1;
        }
    }
    Some(value)
}

/// Chunked relative-position attention over each stream's cached keys and values
/// plus its chunk's own, writing the merged heads `[batch * frames, d_model]`
///
/// `qkv` is the chunk's fused projection `[batch * frames, 3 * d_model]` and `kv` the
/// whole key-value arena; `valid[b]` counts the row's real frames, and queries past
/// it get zeros. One block per `(batch, head, query)` of whole warps, at least
/// `head_dim` and at most `ATTENTION_THREADS` threads; NeMo's `chunked_limited` mask
/// lets every query see the whole window, because the cache is a whole number of
/// chunks
#[kernel]
pub fn encoder_stream_attention(
    qkv: &[f32],
    kv: &[f32],
    pos: &[f32],
    bias_u: &[f32],
    bias_v: &[f32],
    slots: &[u32],
    rings: &[u32],
    cached: &[u32],
    valid: &[u32],
    mut out: DisjointSlice<f32>,
    frames: u32,
    heads: u32,
    head_dim: u32,
    context: u32,
    layers: u32,
    layer: u32,
    sqrt_dk: f32,
) {
    static mut SCRATCH: SharedArray<f32, MAX_WARPS> = SharedArray::UNINIT;
    static mut QUERY_U: SharedArray<f32, MAX_HEAD_DIM> = SharedArray::UNINIT;
    static mut QUERY_V: SharedArray<f32, MAX_HEAD_DIM> = SharedArray::UNINIT;
    static mut WEIGHTS: SharedArray<f32, MAX_KEYS> = SharedArray::UNINIT;
    static mut PARTIAL: SharedArray<f32, ATTENTION_THREADS> = SharedArray::UNINIT;
    // SAFETY: raw pointers to the block's shared arrays, written and read between
    // barriers in `attention_weights` and `attention_output`
    let (scratch, query_u, query_v, weights, partial) = unsafe {
        (
            SharedArray::as_raw_mut_ptr(&raw mut SCRATCH),
            SharedArray::as_raw_mut_ptr(&raw mut QUERY_U),
            SharedArray::as_raw_mut_ptr(&raw mut QUERY_V),
            SharedArray::as_raw_mut_ptr(&raw mut WEIGHTS),
            SharedArray::as_raw_mut_ptr(&raw mut PARTIAL),
        )
    };
    let (frames, heads, head_dim) = (frames as usize, heads as usize, head_dim as usize);
    let (context, layers, layer) = (context as usize, layers as usize, layer as usize);
    let d_model = heads * head_dim;
    let block = thread::blockIdx_x() as usize;
    let (i, h, b) = (block % frames, (block / frames) % heads, block / (frames * heads));
    let scored = attention_weights(
        qkv, kv, pos, bias_u, bias_v, slots, rings, cached, valid, frames, heads, head_dim,
        context, layers, layer, sqrt_dk, query_u, query_v, weights, scratch,
    );

    let output = attention_output(
        qkv, kv, slots, rings, cached, valid, weights, partial, scored, frames, heads, head_dim,
        context, layers, layer,
    );
    if let Some(value) = output {
        let row = (b * frames + i) * d_model + h * head_dim;
        let d = thread::threadIdx_x() as usize;
        // SAFETY: `(b, i, h)` is this block's own and `d` this thread's
        unsafe { *out.get_unchecked_mut(row + d) = value };
    }
}

/// [`encoder_stream_attention`] with its output rounded to FP16
#[kernel]
pub fn encoder_stream_attention_f16(
    qkv: &[f32],
    kv: &[f32],
    pos: &[f32],
    bias_u: &[f32],
    bias_v: &[f32],
    slots: &[u32],
    rings: &[u32],
    cached: &[u32],
    valid: &[u32],
    mut out: DisjointSlice<u16>,
    frames: u32,
    heads: u32,
    head_dim: u32,
    context: u32,
    layers: u32,
    layer: u32,
    sqrt_dk: f32,
) {
    static mut SCRATCH: SharedArray<f32, MAX_WARPS> = SharedArray::UNINIT;
    static mut QUERY_U: SharedArray<f32, MAX_HEAD_DIM> = SharedArray::UNINIT;
    static mut QUERY_V: SharedArray<f32, MAX_HEAD_DIM> = SharedArray::UNINIT;
    static mut WEIGHTS: SharedArray<f32, MAX_KEYS> = SharedArray::UNINIT;
    static mut PARTIAL: SharedArray<f32, ATTENTION_THREADS> = SharedArray::UNINIT;
    // SAFETY: as in `encoder_stream_attention`
    let (scratch, query_u, query_v, weights, partial) = unsafe {
        (
            SharedArray::as_raw_mut_ptr(&raw mut SCRATCH),
            SharedArray::as_raw_mut_ptr(&raw mut QUERY_U),
            SharedArray::as_raw_mut_ptr(&raw mut QUERY_V),
            SharedArray::as_raw_mut_ptr(&raw mut WEIGHTS),
            SharedArray::as_raw_mut_ptr(&raw mut PARTIAL),
        )
    };
    let (frames, heads, head_dim) = (frames as usize, heads as usize, head_dim as usize);
    let (context, layers, layer) = (context as usize, layers as usize, layer as usize);
    let d_model = heads * head_dim;
    let block = thread::blockIdx_x() as usize;
    let (i, h, b) = (block % frames, (block / frames) % heads, block / (frames * heads));
    let scored = attention_weights(
        qkv, kv, pos, bias_u, bias_v, slots, rings, cached, valid, frames, heads, head_dim,
        context, layers, layer, sqrt_dk, query_u, query_v, weights, scratch,
    );

    let output = attention_output(
        qkv, kv, slots, rings, cached, valid, weights, partial, scored, frames, heads, head_dim,
        context, layers, layer,
    );
    if let Some(value) = output {
        let row = (b * frames + i) * d_model + h * head_dim;
        let d = thread::threadIdx_x() as usize;
        // SAFETY: as in `encoder_stream_attention`
        unsafe { *out.get_unchecked_mut(row + d) = to_f16(value) };
    }
}

/// Writes each row's chunk keys and values from `qkv` into the `frames` frames of
/// its ring in `kv` from the ring's start, one thread per element of `[batch,
/// frames, 2, d_model]`: over the oldest frames once the ring is full, into unused
/// ones before
///
/// Runs after the layer's attention has read the ring. The host then moves each
/// ring start `frames` on, so the new frames become the newest
#[kernel]
pub fn encoder_stream_kv_append(
    qkv: &[f32],
    mut kv: DisjointSlice<f32>,
    slots: &[u32],
    rings: &[u32],
    batch: u32,
    frames: u32,
    d_model: u32,
    context: u32,
    layers: u32,
    layer: u32,
) {
    let (batch, frames, d_model) = (batch as usize, frames as usize, d_model as usize);
    let (context, layers, layer) = (context as usize, layers as usize, layer as usize);
    let i = thread::index_1d().get();
    if i >= batch * frames * 2 * d_model {
        return;
    }

    let c = i % d_model;
    let which = (i / d_model) % 2;
    let row = i / (2 * d_model);
    let (b, t) = (row / frames, row % frames);
    let physical = (rings[b] as usize + t) % context;
    let target = (((slots[b] as usize * layers + layer) * 2 + which) * context + physical)
        * d_model
        + c;
    // SAFETY: slots are distinct across rows and `(t, which, c)` is distinct within
    // one, so each thread owns one element of the arena
    unsafe { *kv.get_unchecked_mut(target) = qkv[row * 3 * d_model + (1 + which) * d_model + c] };
}

/// GLU of chunk frame `s` of row `b`, channel `c`, from the first pointwise
/// convolution's `[batch * frames, 2 * channels]`
#[inline(always)]
fn glu(pw1: &[f32], b: usize, s: usize, c: usize, frames: usize, channels: usize) -> f32 {
    let source = (b * frames + s) * 2 * channels + c;
    pw1[source] * sigmoid(pw1[source + channels])
}

/// This thread's channels of the causal depthwise convolution for chunk frame `t` of
/// row `b`, reading the cached GLU frames before the chunk
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn causal_depthwise(
    pw1: &[f32],
    cache: &[f32],
    weight: &[f32],
    b: usize,
    t: usize,
    cache_base: Option<usize>,
    frames: usize,
    channels: usize,
    kernel_size: usize,
) -> [f32; ROW_VALUES] {
    let tid = thread::threadIdx_x() as usize;
    let threads = thread::blockDim_x() as usize;
    let past = kernel_size - 1;
    let mut values = [0.0f32; ROW_VALUES];
    let mut n = 0;
    while n < ROW_VALUES && tid + n * threads < channels {
        let c = tid + n * threads;
        let mut acc = 0.0f32;
        let mut tap = 0;
        while tap < kernel_size {
            // tap `tap` reads frame `t + tap - past`, before the chunk for `t + tap < past`
            let sample = if t + tap >= past {
                glu(pw1, b, t + tap - past, c, frames, channels)
            } else {
                match cache_base {
                    Some(base) => cache[base + (t + tap) * channels + c],
                    None => 0.0,
                }
            };
            acc += weight[c * kernel_size + tap] * sample;
            tap += 1;
        }
        values[n] = acc;
        n += 1;
    }
    values
}

/// The causal conv module between its pointwise convolutions: GLU, the depthwise
/// convolution over the cached and chunk frames, LayerNorm over channels, and SiLU
///
/// `pw1` is `[batch * frames, 2 * channels]`. The depthwise convolution has
/// `kernel_size` taps ending at the output frame and no bias. Frames past a row's
/// valid ones are never read by its valid outputs, so they need no mask. One block
/// per output row, with `channels` at most `ROW_VALUES` per thread
#[kernel]
pub fn encoder_stream_conv(
    pw1: &[f32],
    cache: &[f32],
    slots: &[u32],
    cached: &[u32],
    weight: &[f32],
    gamma: &[f32],
    beta: &[f32],
    mut out: DisjointSlice<f32>,
    frames: u32,
    channels: u32,
    kernel_size: u32,
    layers: u32,
    layer: u32,
    eps: f32,
) {
    static mut SCRATCH: SharedArray<f32, MAX_WARPS> = SharedArray::UNINIT;
    // SAFETY: a raw pointer to the block's shared array; only `block_reduce` touches
    // it, between barriers
    let scratch = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };
    let (frames, channels, kernel_size) = (frames as usize, channels as usize, kernel_size as usize);
    let row = thread::blockIdx_x() as usize;
    let (b, t) = (row / frames, row % frames);
    let cache_base = (cached[b] > 0).then(|| {
        (slots[b] as usize * layers as usize + layer as usize) * (kernel_size - 1) * channels
    });
    let values = causal_depthwise(pw1, cache, weight, b, t, cache_base, frames, channels, kernel_size);

    let (mean, inv_std) = row_moments(&values, channels, eps, scratch);
    let tid = thread::threadIdx_x() as usize;
    let threads = thread::blockDim_x() as usize;
    let mut n = 0;
    while n < ROW_VALUES && tid + n * threads < channels {
        let c = tid + n * threads;
        let normed = (values[n] - mean) * inv_std * gamma[c] + beta[c];
        // SAFETY: rows are disjoint across blocks and `c` is distinct within the block
        unsafe { *out.get_unchecked_mut(row * channels + c) = normed * sigmoid(normed) };
        n += 1;
    }
}

/// [`encoder_stream_conv`] with its output rounded to FP16
#[kernel]
pub fn encoder_stream_conv_f16(
    pw1: &[f32],
    cache: &[f32],
    slots: &[u32],
    cached: &[u32],
    weight: &[f32],
    gamma: &[f32],
    beta: &[f32],
    mut out: DisjointSlice<u16>,
    frames: u32,
    channels: u32,
    kernel_size: u32,
    layers: u32,
    layer: u32,
    eps: f32,
) {
    static mut SCRATCH: SharedArray<f32, MAX_WARPS> = SharedArray::UNINIT;
    // SAFETY: as in `encoder_stream_conv`
    let scratch = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };
    let (frames, channels, kernel_size) = (frames as usize, channels as usize, kernel_size as usize);
    let row = thread::blockIdx_x() as usize;
    let (b, t) = (row / frames, row % frames);
    let cache_base = (cached[b] > 0).then(|| {
        (slots[b] as usize * layers as usize + layer as usize) * (kernel_size - 1) * channels
    });
    let values = causal_depthwise(pw1, cache, weight, b, t, cache_base, frames, channels, kernel_size);

    let (mean, inv_std) = row_moments(&values, channels, eps, scratch);
    let tid = thread::threadIdx_x() as usize;
    let threads = thread::blockDim_x() as usize;
    let mut n = 0;
    while n < ROW_VALUES && tid + n * threads < channels {
        let c = tid + n * threads;
        let normed = (values[n] - mean) * inv_std * gamma[c] + beta[c];
        // SAFETY: as in `encoder_stream_conv`
        unsafe { *out.get_unchecked_mut(row * channels + c) = to_f16(normed * sigmoid(normed)) };
        n += 1;
    }
}

/// Moves each row's conv cache on by its chunk: the last `kernel_size - 1` frames of
/// the cached frames followed by the chunk's GLU frames, one thread per `(batch,
/// channel)`
///
/// Runs after the layer's [`encoder_stream_conv`] has read the cache. A first chunk
/// (`cached[b] == 0`) counts its cache as zeros
#[kernel]
pub fn encoder_stream_conv_cache(
    pw1: &[f32],
    mut cache: DisjointSlice<f32>,
    slots: &[u32],
    cached: &[u32],
    batch: u32,
    frames: u32,
    channels: u32,
    kernel_size: u32,
    layers: u32,
    layer: u32,
) {
    let (batch, frames, channels) = (batch as usize, frames as usize, channels as usize);
    let past = kernel_size as usize - 1;
    let i = thread::index_1d().get();
    if i >= batch * channels {
        return;
    }

    let (b, c) = (i / channels, i % channels);
    let base = (slots[b] as usize * layers as usize + layer as usize) * past * channels;
    let mut m = 0;
    while m < past {
        // frame `m` of the new cache is frame `frames + m` of cache then chunk
        let source = frames + m;
        let value = if source >= past {
            glu(pw1, b, source - past, c, frames, channels)
        } else if cached[b] > 0 {
            // SAFETY: this thread owns channel `c` of the slot's frames, and frame
            // `source > m` has not been overwritten yet
            unsafe { *cache.get_unchecked_mut(base + source * channels + c) }
        } else {
            0.0
        };
        // SAFETY: as above
        unsafe { *cache.get_unchecked_mut(base + m * channels + c) = value };
        m += 1;
    }
}

/// Gathers each row's state from its slot, `dst[g, b, :] = src[slots[b], g, :]`, one
/// thread per element of `dst`, `[groups, batch, width]`
///
/// Moves a stream's per-slot state, such as its predictor's LSTM layers, into the
/// batch-major rows a step reads
#[kernel]
pub fn encoder_stream_gather(
    src: &[f32],
    slots: &[u32],
    mut dst: DisjointSlice<f32>,
    batch: u32,
    groups: u32,
    width: u32,
) {
    let (batch, groups, width) = (batch as usize, groups as usize, width as usize);
    let idx = thread::index_1d();
    let i = idx.get();
    let c = i % width;
    let b = (i / width) % batch;
    let g = i / (width * batch);
    if g < groups
        && let Some(slot) = dst.get_mut(idx)
    {
        *slot = src[(slots[b] as usize * groups + g) * width + c];
    }
}

/// Scatters each row's state back to its slot, `dst[slots[b], g, :] = src[g, b, :]`,
/// one thread per element of `src`, `[groups, batch, width]`
#[kernel]
pub fn encoder_stream_scatter(
    src: &[f32],
    slots: &[u32],
    mut dst: DisjointSlice<f32>,
    batch: u32,
    groups: u32,
    width: u32,
) {
    let (batch, groups, width) = (batch as usize, groups as usize, width as usize);
    let i = thread::index_1d().get();
    if i >= groups * batch * width {
        return;
    }

    let c = i % width;
    let b = (i / width) % batch;
    let g = i / (width * batch);
    // SAFETY: slots are distinct across rows, so each thread owns one element
    unsafe { *dst.get_unchecked_mut((slots[b] as usize * groups + g) * width + c) = src[i] };
}

/// The prompt projector's first layer after its GEMM, in place: `x[b, t, :] =
/// relu(x[b, t, :] + table[prompts[b], :])`, one thread per element of `x`,
/// `[batch, frames, width]`
///
/// The layer reads the encoder output joined with a one-hot prompt, so the prompt
/// only selects one column of its weight; `table` holds each prompt's column plus
/// the bias
#[kernel]
pub fn encoder_stream_prompt(
    mut x: DisjointSlice<f32>,
    table: &[f32],
    prompts: &[u32],
    frames: u32,
    width: u32,
) {
    let (frames, width) = (frames as usize, width as usize);
    let idx = thread::index_1d();
    let i = idx.get();
    let c = i % width;
    let b = i / (width * frames);
    if let Some(value) = x.get_mut(idx) {
        let biased = *value + table[prompts[b] as usize * width + c];
        // torch's ReLU keeps NaN, which `max` would turn into 0
        *value = if biased < 0.0 { 0.0 } else { biased };
    }
}
