//! Greedy TDT decode around the cuBLAS GEMMs of the predictor and joint
//!
//! One decode step for a whole batch is: joint input, joint GEMM, [`decode_decide`],
//! then the predictor for the rows that emitted. The frame advance lives on the
//! device, so the host can replay a captured step many times and only read back the
//! frame cursors. Rows are batch rows; per-row state is `[batch]` and activations
//! are `[batch, width]`, row-major

use cuda_device::{DisjointSlice, SharedArray, kernel, thread, warp};

use crate::common::{MAX_WARPS, add, block_reduce, sigmoid};

/// Encoder frame the next joint step reads for row `b`, clamped so finished rows
/// still read inside their own frames
#[inline(always)]
fn joint_frame(frame: u32, len: u32) -> usize {
    if len == 0 {
        0
    } else {
        frame.min(len - 1) as usize
    }
}

/// `out = relu(enc_proj[b, frame[b]] + pred_proj[b])`, NeMo's joint input after its
/// two projections
///
/// One thread per element of `[batch, hidden]`; `enc_proj` is `[batch, frames,
/// hidden]`
#[kernel]
pub fn decode_joint_input(
    enc_proj: &[f32],
    pred_proj: &[f32],
    frame: &[u32],
    lens: &[u32],
    mut out: DisjointSlice<f32>,
    frames: u32,
    hidden: u32,
) {
    let i = thread::index_1d().get();
    if i >= out.len() {
        return;
    }

    let hidden = hidden as usize;
    let (b, j) = (i / hidden, i % hidden);
    let t = joint_frame(frame[b], lens[b]);
    let value = enc_proj[(b * frames as usize + t) * hidden + j] + pred_proj[i];
    // SAFETY: `i < out.len()` was checked above
    unsafe { *out.get_unchecked_mut(i) = value.max(0.0) };
}

/// First maximum over the block: the largest value, lowest index on ties
#[inline(always)]
fn block_argmax(value: f32, index: u32, values: *mut f32, indices: *mut u32) -> (f32, u32) {
    let better = |v: f32, i: u32, best_v: f32, best_i: u32| v > best_v || (v == best_v && i < best_i);
    let warp_index = thread::threadIdx_x() as usize / 32;
    let warps = thread::blockDim_x() as usize / 32;
    let (mut best_v, mut best_i) = (value, index);
    let mut offset = 16u32;
    while offset > 0 {
        let v = warp::shuffle_xor_f32(best_v, offset);
        let i = warp::shuffle_xor(best_i, offset);
        if better(v, i, best_v, best_i) {
            best_v = v;
            best_i = i;
        }
        offset /= 2;
    }
    if warp::lane_id() == 0 {
        // SAFETY: `warp_index < MAX_WARPS` because a block has at most 1024 threads
        unsafe {
            *values.add(warp_index) = best_v;
            *indices.add(warp_index) = best_i;
        }
    }
    thread::sync_threads();

    // SAFETY: every warp wrote its slot before the barrier above
    let (mut total_v, mut total_i) = unsafe { (*values, *indices) };
    let mut w = 1;
    while w < warps {
        // SAFETY: as above
        let (v, i) = unsafe { (*values.add(w), *indices.add(w)) };
        if better(v, i, total_v, total_i) {
            total_v = v;
            total_i = i;
        }
        w += 1;
    }

    // the scratch slots are rewritten by the next reduction
    thread::sync_threads();
    (total_v, total_i)
}

/// Greedy TDT decision and frame advance for every unfinished row
///
/// One block per row; `logits` is `[batch, classes + durations]` with the blank as
/// the last class, NeMo's joint layout. The token is the first argmax of the class
/// logits, its confidence the softmax at it, and the duration the first argmax of
/// the duration logits. An emitted token is appended to the row's output (up to
/// `capacity`; `counts` keeps counting past it so the host sees an overflow), feeds
/// the predictor through `label` and `emit`, and counts towards `max_symbols`.
///
/// The advance is NeMo's label-looping rule: a positive duration moves by that many
/// frames; otherwise a blank, or the `max_symbols`-th token at one frame, moves by
/// one. A row whose softmax or duration logits are not finite is marked in `faults`
#[kernel]
pub fn decode_decide(
    logits: &[f32],
    lens: &[u32],
    durations: &[u32],
    mut frame: DisjointSlice<u32>,
    mut emitted: DisjointSlice<u32>,
    mut label: DisjointSlice<u32>,
    mut emit: DisjointSlice<u32>,
    mut counts: DisjointSlice<u32>,
    mut faults: DisjointSlice<u32>,
    mut tokens: DisjointSlice<u32>,
    mut token_frames: DisjointSlice<u32>,
    mut token_durations: DisjointSlice<u32>,
    mut confidences: DisjointSlice<f32>,
    classes: u32,
    blank: u32,
    max_symbols: u32,
    capacity: u32,
) {
    static mut SCRATCH: SharedArray<f32, MAX_WARPS> = SharedArray::UNINIT;
    static mut SCRATCH_INDEX: SharedArray<u32, MAX_WARPS> = SharedArray::UNINIT;
    // SAFETY: raw pointers to the block's shared arrays; only the reductions touch
    // them, between barriers
    let (scratch, scratch_index) = unsafe {
        (
            SharedArray::as_raw_mut_ptr(&raw mut SCRATCH),
            SharedArray::as_raw_mut_ptr(&raw mut SCRATCH_INDEX),
        )
    };
    let b = thread::blockIdx_x() as usize;
    let tid = thread::threadIdx_x() as usize;
    let threads = thread::blockDim_x() as usize;

    // every thread reads the cursor before thread 0 rewrites it below, after the
    // reductions' barriers
    // SAFETY: one block per row, so `b` is this block's own slot
    let t = unsafe { *frame.get_unchecked_mut(b) };
    if t >= lens[b] {
        if tid == 0 {
            // SAFETY: as above
            unsafe { *emit.get_unchecked_mut(b) = 0 };
        }
        return;
    }

    let classes = classes as usize;
    let row = b * (classes + durations.len());
    let mut best_v = f32::NEG_INFINITY;
    let mut best_i = u32::MAX;
    let mut i = tid;
    while i < classes {
        let v = logits[row + i];
        if best_i == u32::MAX || v > best_v {
            best_v = v;
            best_i = i as u32;
        }
        i += threads;
    }
    let (row_max, token) = block_argmax(best_v, best_i, scratch, scratch_index);

    let mut sum = 0.0f32;
    let mut i = tid;
    while i < classes {
        sum += (logits[row + i] - row_max).exp();
        i += threads;
    }
    let sum = block_reduce(sum, scratch, add);
    if tid != 0 {
        return;
    }

    let mut bin = 0usize;
    let mut finite = sum.is_finite() && row_max.is_finite();
    let mut k = 0usize;
    while k < durations.len() {
        let v = logits[row + classes + k];
        finite &= v.is_finite();
        if v > logits[row + classes + bin] {
            bin = k;
        }
        k += 1;
    }
    let duration = durations[bin];
    // the softmax at the argmax is `exp(0) / sum`
    let confidence = (1.0 / sum).clamp(0.0, 1.0);

    // SAFETY: thread 0 of the row's block is the only writer of the row's slots
    unsafe {
        if !finite {
            *faults.get_unchecked_mut(b) = 1;
        }

        let mut count = *emitted.get_unchecked_mut(b);
        if token != blank {
            let n = *counts.get_unchecked_mut(b);
            if n < capacity {
                let slot = b * capacity as usize + n as usize;
                *tokens.get_unchecked_mut(slot) = token;
                *token_frames.get_unchecked_mut(slot) = t;
                *token_durations.get_unchecked_mut(slot) = duration;
                *confidences.get_unchecked_mut(slot) = confidence;
            }
            *counts.get_unchecked_mut(b) = n + 1;
            *label.get_unchecked_mut(b) = token;
            *emit.get_unchecked_mut(b) = 1;
            count += 1;
        } else {
            *emit.get_unchecked_mut(b) = 0;
        }

        let mut next = t;
        if duration > 0 {
            next = t + duration;
            count = 0;
        } else if token == blank || count >= max_symbols {
            next = t + 1;
            count = 0;
        }
        *frame.get_unchecked_mut(b) = next;
        *emitted.get_unchecked_mut(b) = count;
    }
}

/// `out[b] = embed[label[b]]`; the blank row is zero, which is NeMo's start symbol
#[kernel]
pub fn decode_embed(embed: &[f32], label: &[u32], mut out: DisjointSlice<f32>, hidden: u32) {
    let i = thread::index_1d().get();
    if i >= out.len() {
        return;
    }

    let hidden = hidden as usize;
    let (b, j) = (i / hidden, i % hidden);
    // SAFETY: `i < out.len()` was checked above
    unsafe { *out.get_unchecked_mut(i) = embed[label[b] as usize * hidden + j] };
}

/// PyTorch LSTM cell on precomputed gates `[batch, 4 * hidden]` (input, forget,
/// cell, output), updating `h` and `c` in place for the rows with `emit` set
#[kernel]
pub fn decode_lstm_cell(
    gates: &[f32],
    emit: &[u32],
    mut h: DisjointSlice<f32>,
    mut c: DisjointSlice<f32>,
    hidden: u32,
) {
    let i = thread::index_1d().get();
    if i >= h.len() {
        return;
    }

    let hidden = hidden as usize;
    let (b, j) = (i / hidden, i % hidden);
    if emit[b] == 0 {
        return;
    }

    let row = b * 4 * hidden;
    let input = sigmoid(gates[row + j]);
    let forget = sigmoid(gates[row + hidden + j]);
    let cell = gates[row + 2 * hidden + j].tanh();
    let output = sigmoid(gates[row + 3 * hidden + j]);
    // SAFETY: `i < h.len()`, and the host sizes `c` like `h`
    unsafe {
        let c_slot = c.get_unchecked_mut(i);
        let next = forget * *c_slot + input * cell;
        *c_slot = next;
        *h.get_unchecked_mut(i) = output * next.tanh();
    }
}
