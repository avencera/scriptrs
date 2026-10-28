//! Greedy TDT decode around the cuBLAS GEMMs of the predictor and joint
//!
//! One decode step for a whole batch is: joint input, joint GEMM, [`decode_decide`],
//! then the predictor for the rows that emitted. The frame advance lives on the
//! device, so the host can replay a captured step many times and only read back the
//! frame cursors. Rows are batch rows; per-row state is `[batch]` and activations
//! are `[batch, width]`, row-major

use cuda_device::atomic::{AtomicOrdering, BlockAtomicU32};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread, warp};

use crate::common::{MAX_WARPS, add, block_reduce, sigmoid};

/// Widest top-K the language filter searches; the host rejects a wider filter, and
/// the candidates above the K-th logit are gathered into shared memory of this size
pub const MAX_FILTER_TOP_K: usize = 256;

/// Histogram bins of one radix-select pass: an 8-bit digit of a 32-bit key
const RADIX_BINS: usize = 256;

/// Digit of a key outside the radix select's current prefix
const NO_DIGIT: u32 = RADIX_BINS as u32;

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

/// Integer twin of `block_reduce`: reduces `value` over the block with `combine`
/// and returns the result to every thread
#[inline(always)]
fn block_reduce_u64(value: u64, scratch: *mut u64, combine: impl Fn(u64, u64) -> u64) -> u64 {
    let warp_index = thread::threadIdx_x() as usize / 32;
    let warps = thread::blockDim_x() as usize / 32;
    let mut partial = value;
    let mut offset = 16u32;
    while offset > 0 {
        partial = combine(partial, warp::shuffle_xor_u64(partial, offset));
        offset /= 2;
    }
    if warp::lane_id() == 0 {
        // SAFETY: `warp_index < MAX_WARPS` because a block has at most 1024 threads
        unsafe { *scratch.add(warp_index) = partial };
    }
    thread::sync_threads();

    // SAFETY: every warp wrote its slot before the barrier above
    let mut total = unsafe { *scratch };
    let mut w = 1;
    while w < warps {
        // SAFETY: as above
        total = combine(total, unsafe { *scratch.add(w) });
        w += 1;
    }

    // the scratch slots are rewritten by the next reduction
    thread::sync_threads();
    total
}

/// `value` as a key whose unsigned order is `f32::total_cmp`'s, the order the host
/// ranks top-K candidates in
#[inline(always)]
fn order_key(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits >> 31 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

/// Inverse of [`order_key`]
#[inline(always)]
fn key_value(key: u32) -> f32 {
    let bits = if key >> 31 != 0 {
        key & 0x7FFF_FFFF
    } else {
        !key
    };
    f32::from_bits(bits)
}

/// This thread's share of `sum(exp(logit - row_max))` over the row's class logits
///
/// A NaN or +inf logit already makes the sum NaN, but a -inf adds 0; with `strict`
/// any non-finite logit makes it NaN, which marks the row as faulted
#[inline(always)]
fn exp_sum(logits: &[f32], row: usize, classes: usize, row_max: f32, strict: bool) -> f32 {
    let threads = thread::blockDim_x() as usize;
    let mut sum = 0.0f32;
    let mut i = thread::threadIdx_x() as usize;
    while i < classes {
        let v = logits[row + i];
        sum += (v - row_max).exp();
        if strict && !v.is_finite() {
            sum = f32::NAN;
        }
        i += threads;
    }
    sum
}

/// Shared memory the language filter works in
struct FilterScratch {
    /// `[MAX_WARPS]`, for `block_reduce_u64`
    wide: *mut u64,
    /// `[RADIX_BINS]` digit counts of one radix-select pass
    histogram: *mut u32,
    /// The selected digit, the candidates still wanted, and the gather cursor
    slots: *mut u32,
    /// `[MAX_FILTER_TOP_K]` keys of the candidates above the K-th logit
    keys: *mut u32,
    /// `[MAX_FILTER_TOP_K]` their softmax terms, in rank order
    terms: *mut f32,
}

/// The `top_k`-th largest key of the row's class logits, and how many ids with
/// exactly that key the top `top_k` takes
///
/// An exact MSB-first radix select: four passes, each counting the 8-bit digit
/// below the prefix chosen so far and keeping the digit that holds the K-th key.
/// Needs `1 <= top_k <= classes`
#[inline(always)]
fn radix_select(
    logits: &[f32],
    row: usize,
    classes: usize,
    top_k: u32,
    scratch: &FilterScratch,
) -> (u32, u32) {
    let tid = thread::threadIdx_x() as usize;
    let threads = thread::blockDim_x() as usize;
    let lane = warp::lane_id();
    let (histogram, slots) = (scratch.histogram, scratch.slots);
    let mut prefix = 0u32;
    let mut wanted = top_k;
    let mut shift = 32u32;
    while shift > 0 {
        shift -= 8;
        // the bits above the digit, which every remaining candidate shares
        let high = if shift == 24 {
            0
        } else {
            u32::MAX << (shift + 8)
        };
        let mut bin = tid;
        while bin < RADIX_BINS {
            // SAFETY: `bin < RADIX_BINS`; nothing reads the histogram until the
            // barrier below
            unsafe { *histogram.add(bin) = 0 };
            bin += threads;
        }
        thread::sync_threads();

        // every lane runs every iteration, so the whole warp reaches the match
        let mut base = 0;
        while base < classes {
            let i = base + tid;
            let mut digit = NO_DIGIT;
            if i < classes {
                let key = order_key(logits[row + i]);
                if key & high == prefix {
                    digit = (key >> shift) & 0xFF;
                }
            }
            // one atomic per distinct digit in the warp: nearby logits share their
            // top digits, so per-lane atomics would serialize on a few bins
            let peers = warp::match_any_sync(u32::MAX, digit);
            if digit != NO_DIGIT && lane == peers.trailing_zeros() {
                // SAFETY: `digit < RADIX_BINS`, and between the barriers around this
                // loop the histogram is only touched atomically
                let count = unsafe { BlockAtomicU32::from_ptr(histogram.add(digit as usize)) };
                count.fetch_add(peers.count_ones(), AtomicOrdering::Relaxed);
            }
            base += threads;
        }
        thread::sync_threads();

        if tid < 32 {
            // lane `l` owns digits `255 - 8l` down to `248 - 8l`, so an inclusive
            // scan over the lanes counts the candidates at or above each lane's digits
            let top = RADIX_BINS as u32 - 1 - 8 * lane;
            let mut own = 0u32;
            let mut j = 0;
            while j < 8 {
                // SAFETY: `top - j < RADIX_BINS`; the histogram is complete after the
                // barrier above
                own += unsafe { *histogram.add((top - j) as usize) };
                j += 1;
            }
            let mut through = own;
            let mut offset = 1u32;
            while offset < 32 {
                let lower = warp::shuffle_up(through, offset);
                if lane >= offset {
                    through += lower;
                }
                offset *= 2;
            }

            // exactly one lane holds the digit where the count crosses `wanted`,
            // because the candidates sharing `prefix` number at least `wanted`
            let mut above = through - own;
            if above < wanted && wanted <= through {
                let mut digit = top;
                loop {
                    // SAFETY: as above; the crossing lies inside this lane's digits
                    let count = unsafe { *histogram.add(digit as usize) };
                    if above + count >= wanted {
                        break;
                    }
                    above += count;
                    digit -= 1;
                }
                // SAFETY: only the crossing lane writes the slots, and every thread
                // reads them after the barrier below
                unsafe {
                    *slots = digit;
                    *slots.add(1) = wanted - above;
                }
            }
        }
        thread::sync_threads();

        // SAFETY: written by the crossing lane before the barrier above; the next
        // write is after the next pass's histogram barrier, which every thread
        // reaches only after this read
        let (digit, remaining) = unsafe { (*slots, *slots.add(1)) };
        prefix |= digit << shift;
        wanted = remaining;
    }
    (prefix, wanted)
}

/// The v3 language filter for a row whose argmax is not the blank
///
/// The host rule: rank the class logits by value (`f32::total_cmp`) and then id,
/// take the first allowed id among the top `top_k`, and score it with the softmax
/// over those `top_k` logits only. Returns that id and score, the score valid on
/// thread 0, or `None` on every thread when no allowed id ranks in the top K.
/// `row_max` is the largest class logit, which always ranks first
#[inline(always)]
fn language_filter(
    logits: &[f32],
    row: usize,
    classes: usize,
    allowed: &[u32],
    top_k: u32,
    row_max: f32,
    scratch: &FilterScratch,
) -> Option<(u32, f32)> {
    let tid = thread::threadIdx_x() as usize;
    let threads = thread::blockDim_x() as usize;

    // the best allowed id, by key and then lowest id: packed so one max reduction
    // finds it, with 0 for no allowed id (a valid packing has a nonzero low half)
    let mut best = 0u64;
    let mut i = tid;
    while i < classes {
        if allowed[i] != 0 {
            let packed = (u64::from(order_key(logits[row + i])) << 32) | u64::from(!(i as u32));
            best = best.max(packed);
        }
        i += threads;
    }
    let best = block_reduce_u64(best, scratch.wide, |a, b| a.max(b));
    if best == 0 {
        return None;
    }
    let (chosen_key, chosen) = ((best >> 32) as u32, !(best as u32));

    // any allowed id in the top K ranks below this one, so the host's first allowed
    // candidate is this id exactly when it ranks in the top K
    let mut ahead = 0u64;
    let mut i = tid;
    while i < classes {
        let key = order_key(logits[row + i]);
        if key > chosen_key || (key == chosen_key && (i as u32) < chosen) {
            ahead += 1;
        }
        i += threads;
    }
    if block_reduce_u64(ahead, scratch.wide, |a, b| a + b) >= u64::from(top_k) {
        return None;
    }

    let (threshold, ties) = radix_select(logits, row, classes, top_k, scratch);
    // the top K are the ids above the threshold key, then `ties` ids at it
    let above = top_k - ties;
    let (slots, keys, terms) = (scratch.slots, scratch.keys, scratch.terms);
    if tid == 0 {
        // SAFETY: slot 2 is the gather cursor alone, which nothing else touches
        // before the barrier below
        unsafe { *slots.add(2) = 0 };
    }
    thread::sync_threads();

    let mut i = tid;
    while i < classes {
        let key = order_key(logits[row + i]);
        if key > threshold {
            // SAFETY: the cursor is only touched atomically between the barriers
            let cursor = unsafe { BlockAtomicU32::from_ptr(slots.add(2)) };
            let slot = cursor.fetch_add(1, AtomicOrdering::Relaxed) as usize;
            // the select is exact, so `above < top_k <= MAX_FILTER_TOP_K` keys pass;
            // the bound only keeps a broken invariant inside the array
            if slot < MAX_FILTER_TOP_K {
                // SAFETY: `slot` is in bounds and claimed by this thread alone
                unsafe { *keys.add(slot) = key };
            }
        }
        i += threads;
    }
    thread::sync_threads();

    // place each term at its rank so thread 0 adds them in the host's order, which
    // keeps the f32 sums equal up to `exp` rounding
    let above = above as usize;
    let mut s = tid;
    while s < above {
        // SAFETY: `s < above`, and the gather finished before the barrier above
        let key = unsafe { *keys.add(s) };
        let mut rank = 0;
        let mut r = 0;
        while r < above {
            // SAFETY: as above
            let other = unsafe { *keys.add(r) };
            if other > key || (other == key && r < s) {
                rank += 1;
            }
            r += 1;
        }
        // SAFETY: ranks are distinct and below `above`
        unsafe { *terms.add(rank) = (key_value(key) - row_max).exp() };
        s += threads;
    }
    thread::sync_threads();

    // only thread 0 writes the decision, so only it adds the terms up
    if tid != 0 {
        return Some((chosen, 0.0));
    }

    let mut sum = 0.0f32;
    let mut r = 0;
    while r < above {
        // SAFETY: every rank below `above` was written before the barrier above
        sum += unsafe { *terms.add(r) };
        r += 1;
    }
    let tie = (key_value(threshold) - row_max).exp();
    let mut r = 0;
    while r < ties {
        sum += tie;
        r += 1;
    }
    let confidence = ((key_value(chosen_key) - row_max).exp() / sum).clamp(0.0, 1.0);
    Some((chosen, confidence))
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
/// `filter[0]` is the language filter's top-K width, 0 for no filter. With a filter
/// and a non-blank argmax, the host rule of `language_filter` may replace the token
/// with an `allowed` one (`[classes]`, nonzero for allowed ids) and its confidence
/// with the softmax over the top-K logits only.
///
/// The advance is NeMo's label-looping rule: a positive duration moves by that many
/// frames; otherwise a blank, or the `max_symbols`-th token at one frame, moves by
/// one. A row whose softmax or duration logits are not finite is marked in `faults`,
/// and with a filter so is a row with any non-finite class logit, which the host
/// rule rejects
#[kernel]
pub fn decode_decide(
    logits: &[f32],
    lens: &[u32],
    durations: &[u32],
    allowed: &[u32],
    filter: &[u32],
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
    static mut SCRATCH_WIDE: SharedArray<u64, MAX_WARPS> = SharedArray::UNINIT;
    static mut HISTOGRAM: SharedArray<u32, RADIX_BINS> = SharedArray::UNINIT;
    static mut SLOTS: SharedArray<u32, 4> = SharedArray::UNINIT;
    static mut KEYS: SharedArray<u32, MAX_FILTER_TOP_K> = SharedArray::UNINIT;
    static mut TERMS: SharedArray<f32, MAX_FILTER_TOP_K> = SharedArray::UNINIT;
    // SAFETY: raw pointers to the block's shared arrays; only the reductions and the
    // language filter touch them, between barriers
    let (scratch, scratch_index, filter_scratch) = unsafe {
        (
            SharedArray::as_raw_mut_ptr(&raw mut SCRATCH),
            SharedArray::as_raw_mut_ptr(&raw mut SCRATCH_INDEX),
            FilterScratch {
                wide: SharedArray::as_raw_mut_ptr(&raw mut SCRATCH_WIDE),
                histogram: SharedArray::as_raw_mut_ptr(&raw mut HISTOGRAM),
                slots: SharedArray::as_raw_mut_ptr(&raw mut SLOTS),
                keys: SharedArray::as_raw_mut_ptr(&raw mut KEYS),
                terms: SharedArray::as_raw_mut_ptr(&raw mut TERMS),
            },
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

    // the select needs `top_k <= classes`, and the gather `top_k <= MAX_FILTER_TOP_K`;
    // the host rejects anything wider, so the clamp only guards memory
    let top_k = filter[0].min(classes).min(MAX_FILTER_TOP_K as u32);
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

    // constant arguments, so each call inlines to its own loop and the unfiltered
    // one carries no finiteness check
    let sum = if top_k == 0 {
        exp_sum(logits, row, classes, row_max, false)
    } else {
        exp_sum(logits, row, classes, row_max, true)
    };
    let sum = block_reduce(sum, scratch, add);

    // `token` and `top_k` are the same on every thread, so the whole block runs the
    // filter's barriers
    let filtered = if top_k != 0 && token != blank {
        language_filter(
            logits,
            row,
            classes,
            allowed,
            top_k,
            row_max,
            &filter_scratch,
        )
    } else {
        None
    };
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
    let (token, confidence) = match filtered {
        Some(decision) => decision,
        // the softmax at the argmax is `exp(0) / sum`
        None => (token, (1.0 / sum).clamp(0.0, 1.0)),
    };

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
