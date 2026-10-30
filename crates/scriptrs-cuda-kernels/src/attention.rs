//! Fused relative-position self-attention on tensor cores
//!
//! One kernel replaces the head split, the content and position score GEMMs, the
//! rel-shift softmax, the value GEMM and the head merge of NeMo's
//! `RelPositionMultiHeadAttention`. It reads `q`, `k` and `v` from the fused QKV
//! projection `[batch * frames, 3 * d_model]` and the projected position table `[2 *
//! frames - 1, d_model]`, and writes the merged context `[batch * frames, d_model]`,
//! so the `frames x frames` score matrices never reach memory.
//!
//! For query `i` and key `j` the score is `((q_i + u) · k_j + (q_i + v) · p_{T-1-i+j})
//! / sqrt(d_k)`, with position row `r` meaning distance `T - 1 - r`, which is the
//! column NeMo's `rel_shift` reads. Keys at or past the row's length are skipped, as
//! NeMo's -10000 mask gives them exactly 0 weight in FP32, and query rows at or past
//! it come out 0, as NeMo zeroes their weights.
//!
//! A block takes 16 query rows per warp of one head and one utterance and walks the
//! keys in tiles of [`KEY_TILE`] with an online softmax. Each warp multiplies its rows
//! by the [`WINDOW`] position rows its tile can reach and moves the diagonal band the
//! rel-shift picks into the score fragments with shuffles inside each lane quad, so
//! the position term costs `WINDOW / KEY_TILE` of a content product and no shared
//! memory. Each score tile starts from that band and the content product accumulates
//! on top of it. Tiles stream in with `cp.async`:
//! the value tile loads while the scores are computed, and the next key tile and
//! position window while the values are mixed.
//!
//! [`attention_rel_pos_f16`] takes FP16 operands for the FP16 encoder, and
//! [`attention_rel_pos_tf32`] FP32 ones rounded to TF32 for the TF32 encoder. The
//! kernels need `mma.sync` and `cp.async`, so this area compiles for `sm_80`

use cuda_device::async_copy::{cp_async_cg_zfill_16, cp_async_commit_group, cp_async_wait_all};
use cuda_device::convert::{cvt_f16x2_f32, cvt_f32x2_f16x2};
use cuda_device::float::ex2_approx_ftz_f32;
use cuda_device::wmma::{
    ldmatrix_x4, ldmatrix_x4_trans, mma_m16n8k8_f32_tf32, mma_m16n8k16_f32_f16,
};
use cuda_device::{DisjointSlice, DynamicSharedArray, kernel, thread, warp};
use seq_macro::seq;

/// Head width the kernels are unrolled for; the host rejects any other
pub const HEAD_DIM: usize = 128;

/// Keys per tile of the FP16 kernel's online softmax
pub const KEY_TILE: usize = 64;

/// Position rows one warp multiplies per key tile: its 16 query rows reach `KEY_TILE
/// + 15` relative distances, rounded up to whole 8-column MMA tiles
pub const WINDOW: usize = KEY_TILE + 16;

/// Shared row stride of a staged FP16 tile, in halves; the 16-byte pad moves each
/// row's start by four banks, so `ldmatrix` reads its 8 rows without conflicts
pub const TILE_ROW: usize = HEAD_DIM + 8;

/// Dynamic shared memory one FP16 block of `warps` warps needs, in bytes: the key and
/// value tiles and the block's position window
pub const fn shared_bytes_f16(warps: usize) -> usize {
    (3 * KEY_TILE + 16 * warps) * TILE_ROW * 2
}

/// Keys per tile of the TF32 kernel: FP32 tiles take twice the bytes of FP16 ones, and
/// 32 keys keep four warps' tiles inside one SM's shared memory
pub const TF32_KEY_TILE: usize = 32;

/// [`WINDOW`] for [`TF32_KEY_TILE`]
pub const TF32_WINDOW: usize = TF32_KEY_TILE + 16;

/// Shared row stride of a staged FP32 tile, in floats; the 16-byte pad keeps both the
/// `ldmatrix` reads and the value reads conflict-free
pub const TF32_TILE_ROW: usize = HEAD_DIM + 4;

/// Dynamic shared memory one TF32 block of `warps` warps needs, in bytes: the key and
/// value tiles, the block's position window and the head's `v - u` bias difference
pub const fn shared_bytes_tf32(warps: usize) -> usize {
    (3 * TF32_KEY_TILE + 16 * warps) * TF32_TILE_ROW * 4 + HEAD_DIM * 4
}

/// Queues 16-byte copies of `rows` rows of [`HEAD_DIM`] elements into a staged tile of
/// row stride `tile_row`, `per_chunk` elements per copy: tile row `r` takes `src[(first
/// + r) * stride + column..]` when `first + r` is in `[0, limit)`, and zeros otherwise
///
/// A macro, not a function: cuda-oxide types the shared-memory pointer in the
/// shared address space, which a generic pointer parameter does not accept
///
/// Safety: `dst` must hold `rows` staged rows in shared memory, every row in `[0,
/// limit)` must hold `column + HEAD_DIM` elements of `src`, and `column` and `stride`
/// must keep each copy 16-byte aligned
macro_rules! stage_rows {
    (
        $dst:expr, $src:expr, $stride:expr, $column:expr, $first:expr, $rows:expr, $limit:expr,
        $tile_row:expr, $per_chunk:expr
    ) => {{
        let (dst, src, stride, column) = ($dst, $src, $stride, $column);
        let (first, rows, limit): (isize, usize, usize) = ($first, $rows, $limit);
        let (tile_row, per_chunk): (usize, usize) = ($tile_row, $per_chunk);
        let chunks = HEAD_DIM / per_chunk;
        let threads = thread::blockDim_x() as usize;
        let mut chunk = thread::threadIdx_x() as usize;
        while chunk < rows * chunks {
            let (r, part) = (chunk / chunks, chunk % chunks);
            let row = first + r as isize;
            let valid = row >= 0 && (row as usize) < limit;
            // a zero-size copy reads nothing, so out-of-range rows point at the base
            let source = if valid {
                src.add(row as usize * stride + column + part * per_chunk)
            } else {
                src
            };
            cp_async_cg_zfill_16(
                dst.add(r * tile_row + part * per_chunk).cast::<u32>(),
                source.cast::<u8>(),
                if valid { 16 } else { 0 },
            );
            chunk += threads;
        }
    }};
}

/// The `ldmatrix_x4` row this lane addresses in a B operand stored `[n][k]` (keys or
/// positions by head columns), for two 8-wide n-tiles from `row` and two 16-byte
/// column groups of `per_chunk` elements from `column`; a macro for the reason
/// [`stage_rows!`] is
macro_rules! b_address {
    ($tile:expr, $row:expr, $column:expr, $lane:expr, $tile_row:expr, $per_chunk:expr) => {{
        let lane: usize = $lane;
        let r = $row + lane % 8 + (lane / 16) * 8;
        let c = $column + ((lane / 8) % 2) * $per_chunk;
        $tile.add(r * $tile_row + c).cast::<u32>()
    }};
}

/// Packs `(q + bias)` for two adjacent columns into FP16x2, adding in FP32 as NeMo
/// adds the positional biases
#[inline(always)]
fn biased(word: u32, bias: &[f32], column: usize) -> u32 {
    let (lo, hi) = cvt_f32x2_f16x2(word);
    cvt_f16x2_f32(lo + bias[column], hi + bias[column + 1])
}

/// Fused relative-position attention with FP16 operands and FP32 accumulation
///
/// `qkv` is `[batch * frames, 3 * heads * 128]` and `pos` `[2 * frames - 1, heads *
/// 128]`, both FP16 bits; `out` is the merged `[batch * frames, heads * 128]` context
/// in FP16 bits for the output projection. `scale` is `log2(e) / sqrt(d_k)`, since the
/// softmax runs in base 2. Grid `(ceil(frames / (16 * warps)), heads, batch)`, 1 to 8
/// warps per block and [`shared_bytes_f16`] of dynamic shared memory
#[kernel]
#[allow(
    clippy::too_many_arguments,
    clippy::erasing_op,
    clippy::identity_op,
    clippy::eq_op,
    reason = "`seq!` unrolls each fragment loop with literal indices, so `E / 2` or `2 * X + 1` read as constant arithmetic"
)]
pub fn attention_rel_pos_f16(
    qkv: &[u16],
    pos: &[u16],
    bias_u: &[f32],
    bias_v: &[f32],
    lens: &[u32],
    mut out: DisjointSlice<u16>,
    frames: u32,
    heads: u32,
    scale: f32,
) {
    let frames = frames as usize;
    let d_model = heads as usize * HEAD_DIM;
    let h = thread::blockIdx_y() as usize;
    let b = thread::blockIdx_z() as usize;
    let tid = thread::threadIdx_x() as usize;
    let threads = thread::blockDim_x() as usize;
    let warps = threads / 32;
    let block_rows = 16 * warps;
    let first_query = thread::blockIdx_x() as usize * block_rows;
    let len = (lens[b] as usize).min(frames);
    let column = h * HEAD_DIM;
    let out_words = out.as_mut_ptr().cast::<u32>();

    // every row of a block past the length is 0; nothing else to compute
    if first_query >= len {
        let rows = block_rows.min(frames - first_query);
        let mut i = tid;
        while i < rows * HEAD_DIM / 2 {
            let (r, pair) = (i / (HEAD_DIM / 2), i % (HEAD_DIM / 2));
            let word = ((b * frames + first_query + r) * d_model + column) / 2 + pair;
            // SAFETY: blocks own disjoint `(row, head)` slices of the output
            unsafe { *out_words.add(word) = 0 };
            i += threads;
        }
        return;
    }

    let keys: *mut u16 = DynamicSharedArray::<u16>::get();
    let values = unsafe { keys.add(KEY_TILE * TILE_ROW) };
    let window = unsafe { values.add(KEY_TILE * TILE_ROW) };
    let warp_index = tid / 32;
    let lane = tid % 32;
    let (group, quad) = (lane / 4, lane % 4);
    let qkv_stride = 3 * d_model;
    let qkv_rows = qkv.as_ptr();

    // stage q + u across the key and value tiles, which are contiguous, and q + v in
    // the window, then keep this warp's rows as MMA A fragments for the whole pass
    let mut chunk = tid;
    while chunk < block_rows * (HEAD_DIM / 8) {
        let (r, part) = (chunk / (HEAD_DIM / 8), chunk % (HEAD_DIM / 8));
        let i = first_query + r;
        let mut with_u = [0u32; 4];
        let mut with_v = [0u32; 4];
        if i < frames {
            let at = (b * frames + i) * qkv_stride + column + part * 8;
            seq!(W in 0..4 {{
                // SAFETY: the host checks `qkv` holds `batch * frames` rows
                let word = unsafe { *qkv_rows.add(at + 2 * W).cast::<u32>() };
                let c = column + part * 8 + 2 * W;
                with_u[W] = biased(word, bias_u, c);
                with_v[W] = biased(word, bias_v, c);
            }});
        }
        unsafe {
            *keys.add(r * TILE_ROW + part * 8).cast::<[u32; 4]>() = with_u;
            *window.add(r * TILE_ROW + part * 8).cast::<[u32; 4]>() = with_v;
        }
        chunk += threads;
    }
    thread::sync_threads();

    // the fragment loops below are unrolled with `seq!`, so every array index is a
    // constant and the accumulators stay in registers; cuda-oxide's unroller leaves
    // loops this size rolled, which spills them to local memory
    let mut q_u = [[0u32; 4]; HEAD_DIM / 16];
    let mut q_v = [[0u32; 4]; HEAD_DIM / 16];
    let a_row = warp_index * 16 + lane % 16;
    seq!(KS in 0..8 {{
        let at = a_row * TILE_ROW + KS * 16 + (lane / 16) * 8;
        // SAFETY: every lane addresses a 16-byte row inside the staged tiles, which
        // the barrier above made visible
        unsafe {
            q_u[KS] = ldmatrix_x4(keys.add(at).cast::<u32>());
            q_v[KS] = ldmatrix_x4(window.add(at).cast::<u32>());
        }
    }});
    thread::sync_threads();

    let positions = 2 * frames - 1;
    // window row 0 is position `frames - 1 - (last query) + j0`
    let window_first =
        |j0: usize| frames as isize - (first_query + block_rows) as isize + j0 as isize;
    let batch_rows = unsafe { qkv_rows.add(b * frames * qkv_stride) };
    // SAFETY: the host checks `qkv` and `pos` against `frames` and `heads`
    unsafe {
        stage_rows!(
            keys,
            batch_rows,
            qkv_stride,
            d_model + column,
            0,
            KEY_TILE,
            frames,
            TILE_ROW,
            8
        );
        stage_rows!(
            window,
            pos.as_ptr(),
            d_model,
            column,
            window_first(0),
            block_rows + KEY_TILE,
            positions,
            TILE_ROW,
            8
        );
        cp_async_commit_group();
    }

    // this warp's 16 rows reach window rows `band_first..band_first + WINDOW`, and row
    // `group + 8` reads band column `shift + jj`
    let band_first = block_rows - 16 - 16 * warp_index;
    let shift = 7 - group;
    let mut o = [[0f32; 4]; HEAD_DIM / 8];
    let mut row_max = [f32::NEG_INFINITY; 2];
    let mut row_sum = [0f32; 2];
    let mut j0 = 0;
    while j0 < len {
        // the key tile and window are in, and every warp is done with the last values
        unsafe { cp_async_wait_all() };
        thread::sync_threads();
        unsafe {
            stage_rows!(
                values,
                batch_rows,
                qkv_stride,
                2 * d_model + column,
                j0 as isize,
                KEY_TILE,
                frames,
                TILE_ROW,
                8
            );
            cp_async_commit_group();
        }

        // position products (q + v) · pᵀ over this warp's window
        let mut bd = [[0f32; 4]; WINDOW / 8];
        seq!(KS in 0..8 {{
            seq!(NP in 0..5 {{
                // SAFETY: rows stay inside the block's window, which is resident
                let fragment = unsafe {
                    ldmatrix_x4(b_address!(window, band_first + NP * 16, KS * 16, lane, TILE_ROW, 8))
                };
                unsafe {
                    bd[2 * NP] = mma_m16n8k16_f32_f16(bd[2 * NP], q_v[KS], [fragment[0], fragment[1]]);
                    bd[2 * NP + 1] =
                        mma_m16n8k16_f32_f16(bd[2 * NP + 1], q_v[KS], [fragment[2], fragment[3]]);
                }
            }});
        }});

        // the rel-shift: score `(r, jj)` takes band column `15 - r + jj`. Rows `group`
        // and `group + 8` shift by `shift + 8` and `shift`, so every lane of a quad
        // reads its quad, and a sender's element is the receiver's `e` moved by the
        // quad's shared shift parity. `received[h][n][e]` is band tile `n` of row
        // half `h` as receiver element `e` wants it
        let mut received = [[[0f32; 2]; WINDOW / 8]; 2];
        seq!(E in 0..2 {{
            let q = 2 * quad + E + shift;
            let source = (4 * group + (q % 8) / 2) as u32;
            let odd = (E + shift) % 2 == 1;
            seq!(N in 0..10 {{
                let low = if odd { bd[N][1] } else { bd[N][0] };
                let high = if odd { bd[N][3] } else { bd[N][2] };
                received[0][N][E] = warp::shuffle_f32(low, source);
                received[1][N][E] = warp::shuffle_f32(high, source);
            }});
        }});
        let mut s = [[0f32; 4]; KEY_TILE / 8];
        seq!(E in 0..2 {{
            let carry = 2 * quad + E + shift >= 8;
            seq!(NT in 0..8 {{
                s[NT][E] = if carry { received[0][NT + 2][E] } else { received[0][NT + 1][E] };
                s[NT][2 + E] = if carry { received[1][NT + 1][E] } else { received[1][NT][E] };
            }});
        }});

        // content products (q + u) · kᵀ on top of the position term
        seq!(KS in 0..8 {{
            seq!(NP in 0..4 {{
                // SAFETY: rows stay inside the resident key tile
                let fragment = unsafe { ldmatrix_x4(b_address!(keys, NP * 16, KS * 16, lane, TILE_ROW, 8)) };
                unsafe {
                    s[2 * NP] = mma_m16n8k16_f32_f16(s[2 * NP], q_u[KS], [fragment[0], fragment[1]]);
                    s[2 * NP + 1] =
                        mma_m16n8k16_f32_f16(s[2 * NP + 1], q_u[KS], [fragment[2], fragment[3]]);
                }
            }});
        }});

        // scale, mask, and fold into the running softmax
        let mut tile_max = row_max;
        seq!(NT in 0..8 {{
            seq!(E in 0..4 {{
                let jj = NT * 8 + 2 * quad + E % 2;
                s[NT][E] = if j0 + jj < len {
                    s[NT][E] * scale
                } else {
                    f32::NEG_INFINITY
                };
                tile_max[E / 2] = tile_max[E / 2].max(s[NT][E]);
            }});
        }});
        let mut correction = [0f32; 2];
        seq!(X in 0..2 {{
            tile_max[X] = tile_max[X].max(warp::shuffle_xor_f32(tile_max[X], 1));
            tile_max[X] = tile_max[X].max(warp::shuffle_xor_f32(tile_max[X], 2));
            correction[X] = ex2_approx_ftz_f32(row_max[X] - tile_max[X]);
            row_max[X] = tile_max[X];
            row_sum[X] *= correction[X];
        }});
        let mut weights = [[0u32; 4]; KEY_TILE / 16];
        seq!(NT in 0..8 {{
            let p = [
                ex2_approx_ftz_f32(s[NT][0] - row_max[0]),
                ex2_approx_ftz_f32(s[NT][1] - row_max[0]),
                ex2_approx_ftz_f32(s[NT][2] - row_max[1]),
                ex2_approx_ftz_f32(s[NT][3] - row_max[1]),
            ];
            row_sum[0] += p[0] + p[1];
            row_sum[1] += p[2] + p[3];
            // two 8-key C tiles make one 16-key A fragment
            weights[NT / 2][2 * (NT % 2)] = cvt_f16x2_f32(p[0], p[1]);
            weights[NT / 2][2 * (NT % 2) + 1] = cvt_f16x2_f32(p[2], p[3]);
        }});
        seq!(DT in 0..16 {{
            o[DT][0] *= correction[0];
            o[DT][1] *= correction[0];
            o[DT][2] *= correction[1];
            o[DT][3] *= correction[1];
        }});

        // the values are in, and every warp is done with this key tile and window
        unsafe { cp_async_wait_all() };
        thread::sync_threads();
        let next = j0 + KEY_TILE;
        if next < len {
            unsafe {
                stage_rows!(
                    keys,
                    batch_rows,
                    qkv_stride,
                    d_model + column,
                    next as isize,
                    KEY_TILE,
                    frames,
                    TILE_ROW,
                    8
                );
                stage_rows!(
                    window,
                    pos.as_ptr(),
                    d_model,
                    column,
                    window_first(next),
                    block_rows + KEY_TILE,
                    positions,
                    TILE_ROW,
                    8
                );
                cp_async_commit_group();
            }
        }

        // o += weights · v
        seq!(KS in 0..4 {{
            seq!(DP in 0..8 {{
                let r = KS * 16 + lane % 8 + ((lane / 8) % 2) * 8;
                let c = DP * 16 + (lane / 16) * 8;
                // SAFETY: rows stay inside the resident value tile
                let fragment =
                    unsafe { ldmatrix_x4_trans(values.add(r * TILE_ROW + c).cast::<u32>()) };
                unsafe {
                    o[2 * DP] = mma_m16n8k16_f32_f16(o[2 * DP], weights[KS], [fragment[0], fragment[1]]);
                    o[2 * DP + 1] =
                        mma_m16n8k16_f32_f16(o[2 * DP + 1], weights[KS], [fragment[2], fragment[3]]);
                }
            }});
        }});
        j0 = next;
    }

    seq!(X in 0..2 {{
        row_sum[X] += warp::shuffle_xor_f32(row_sum[X], 1);
        row_sum[X] += warp::shuffle_xor_f32(row_sum[X], 2);
    }});
    seq!(HALF in 0..2 {{
        let i = first_query + warp_index * 16 + group + HALF * 8;
        if i < frames {
            // a padded row is written as 0, not scaled to it, so an overflowed FP16
            // projection in that row cannot leave NaN behind
            let valid = i < len;
            let inverse = 1.0 / row_sum[HALF];
            let row_word = ((b * frames + i) * d_model + column) / 2;
            seq!(DT in 0..16 {{
                let word = if valid {
                    cvt_f16x2_f32(o[DT][2 * HALF] * inverse, o[DT][2 * HALF + 1] * inverse)
                } else {
                    0
                };
                // SAFETY: each lane writes its own columns of rows this block owns
                unsafe { *out_words.add(row_word + DT * 4 + quad) = word };
            }});
        }
    }});
}

/// Rounds an FP32 value, given as its bits, to the TF32 operand an MMA takes, to
/// nearest with ties away from zero
///
/// Tensor cores ignore the low 13 bits of a TF32 operand, so adding half of the last
/// kept bit is the whole rounding; ptxas lowers `cvt.rna.tf32.f32` to the same add
/// behind a finiteness test, which doubles the cost and only matters for infinities
/// and NaNs, which poison the softmax either way
#[inline(always)]
fn tf32(bits: u32) -> u32 {
    bits.wrapping_add(0x1000)
}

/// Fused relative-position attention with FP32 operands rounded to TF32 and FP32
/// accumulation, as the TF32 encoder's cuBLAS products compute
///
/// `qkv` is `[batch * frames, 3 * heads * 128]`, `pos` `[2 * frames - 1, heads * 128]`
/// and `out` the merged `[batch * frames, heads * 128]` context, all FP32. The rest is
/// as [`attention_rel_pos_f16`] takes it, with 1 to 4 warps per block and
/// [`shared_bytes_tf32`] of dynamic shared memory
///
/// Only `q + u` stays in registers, since TF32 fragments take twice the registers FP16
/// ones do; each key tile rebuilds `q + v` from it and the shared `v - u`, which rounds
/// `q + v` twice, once to TF32 and once more after adding `v - u`
///
/// The value products read each 8-key score tile as an A fragment directly, taking
/// the keys in the order `0, 2, 4, 6, 1, 3, 5, 7`, and pair 8-column output tiles so
/// each lane reads two adjacent value columns per key and writes four adjacent
/// output columns
#[kernel]
#[allow(
    clippy::too_many_arguments,
    clippy::erasing_op,
    clippy::identity_op,
    clippy::eq_op,
    reason = "`seq!` unrolls each fragment loop with literal indices, so `E / 2` or `2 * X + 1` read as constant arithmetic"
)]
pub fn attention_rel_pos_tf32(
    qkv: &[f32],
    pos: &[f32],
    bias_u: &[f32],
    bias_v: &[f32],
    lens: &[u32],
    mut out: DisjointSlice<f32>,
    frames: u32,
    heads: u32,
    scale: f32,
) {
    let frames = frames as usize;
    let d_model = heads as usize * HEAD_DIM;
    let h = thread::blockIdx_y() as usize;
    let b = thread::blockIdx_z() as usize;
    let tid = thread::threadIdx_x() as usize;
    let threads = thread::blockDim_x() as usize;
    let warps = threads / 32;
    let block_rows = 16 * warps;
    let first_query = thread::blockIdx_x() as usize * block_rows;
    let len = (lens[b] as usize).min(frames);
    let column = h * HEAD_DIM;
    let out_rows = out.as_mut_ptr();

    // every row of a block past the length is 0; nothing else to compute
    if first_query >= len {
        let rows = block_rows.min(frames - first_query);
        let mut i = tid;
        while i < rows * HEAD_DIM {
            let (r, c) = (i / HEAD_DIM, i % HEAD_DIM);
            // SAFETY: blocks own disjoint `(row, head)` slices of the output
            unsafe { *out_rows.add((b * frames + first_query + r) * d_model + column + c) = 0.0 };
            i += threads;
        }
        return;
    }

    let keys: *mut f32 = DynamicSharedArray::<f32>::get();
    let values = unsafe { keys.add(TF32_KEY_TILE * TF32_TILE_ROW) };
    let window = unsafe { values.add(TF32_KEY_TILE * TF32_TILE_ROW) };
    let difference = unsafe { window.add((block_rows + TF32_KEY_TILE) * TF32_TILE_ROW) };
    let warp_index = tid / 32;
    let lane = tid % 32;
    let (group, quad) = (lane / 4, lane % 4);
    let qkv_stride = 3 * d_model;
    let qkv_rows = qkv.as_ptr();

    // stage q + u across the key and value tiles, which are contiguous, and the head's
    // v - u after the window, then keep this warp's rows as MMA A fragments
    let mut chunk = tid;
    while chunk < block_rows * (HEAD_DIM / 4) {
        let (r, part) = (chunk / (HEAD_DIM / 4), chunk % (HEAD_DIM / 4));
        let i = first_query + r;
        let mut with_u = [0f32; 4];
        if i < frames {
            let at = (b * frames + i) * qkv_stride + column + part * 4;
            seq!(E in 0..4 {{
                // SAFETY: the host checks `qkv` holds `batch * frames` rows
                with_u[E] = unsafe { *qkv_rows.add(at + E) } + bias_u[column + part * 4 + E];
            }});
        }
        unsafe { *keys.add(r * TF32_TILE_ROW + part * 4).cast::<[f32; 4]>() = with_u };
        chunk += threads;
    }
    let mut c = tid;
    while c < HEAD_DIM {
        unsafe { *difference.add(c) = bias_v[column + c] - bias_u[column + c] };
        c += threads;
    }
    thread::sync_threads();

    // unrolled with `seq!` for the reason the FP16 kernel's loops are
    let mut q_u = [[0u32; 4]; HEAD_DIM / 8];
    let a_row = warp_index * 16 + lane % 16;
    seq!(KS in 0..16 {{
        // SAFETY: every lane addresses a 16-byte row inside the staged tiles, which
        // the barrier above made visible
        let raw = unsafe {
            ldmatrix_x4(keys.add(a_row * TF32_TILE_ROW + KS * 8 + (lane / 16) * 4).cast::<u32>())
        };
        seq!(J in 0..4 {{
            // masked to the exact TF32 value, as `q + v` is rebuilt from it as a float
            q_u[KS][J] = tf32(raw[J]) & 0xffff_e000;
        }});
    }});
    thread::sync_threads();

    let positions = 2 * frames - 1;
    // window row 0 is position `frames - 1 - (last query) + j0`
    let window_first =
        |j0: usize| frames as isize - (first_query + block_rows) as isize + j0 as isize;
    let batch_rows = unsafe { qkv_rows.add(b * frames * qkv_stride) };
    // SAFETY: the host checks `qkv` and `pos` against `frames` and `heads`
    unsafe {
        stage_rows!(
            keys,
            batch_rows,
            qkv_stride,
            d_model + column,
            0,
            TF32_KEY_TILE,
            frames,
            TF32_TILE_ROW,
            4
        );
        stage_rows!(
            window,
            pos.as_ptr(),
            d_model,
            column,
            window_first(0),
            block_rows + TF32_KEY_TILE,
            positions,
            TF32_TILE_ROW,
            4
        );
        cp_async_commit_group();
    }

    // as in the FP16 kernel
    let band_first = block_rows - 16 - 16 * warp_index;
    let shift = 7 - group;
    let mut o = [[0f32; 4]; HEAD_DIM / 8];
    let mut row_max = [f32::NEG_INFINITY; 2];
    let mut row_sum = [0f32; 2];
    let mut j0 = 0;
    while j0 < len {
        // the key tile and window are in, and every warp is done with the last values
        unsafe { cp_async_wait_all() };
        thread::sync_threads();
        unsafe {
            stage_rows!(
                values,
                batch_rows,
                qkv_stride,
                2 * d_model + column,
                j0 as isize,
                TF32_KEY_TILE,
                frames,
                TF32_TILE_ROW,
                4
            );
            cp_async_commit_group();
        }

        // position products (q + v) · pᵀ over this warp's window
        let mut bd = [[0f32; 4]; TF32_WINDOW / 8];
        seq!(KS in 0..16 {{
            // SAFETY: inside the resident bias difference
            let (low, high) =
                unsafe { (*difference.add(KS * 8 + quad), *difference.add(KS * 8 + quad + 4)) };
            let q_v = [
                tf32((f32::from_bits(q_u[KS][0]) + low).to_bits()),
                tf32((f32::from_bits(q_u[KS][1]) + low).to_bits()),
                tf32((f32::from_bits(q_u[KS][2]) + high).to_bits()),
                tf32((f32::from_bits(q_u[KS][3]) + high).to_bits()),
            ];
            seq!(NP in 0..3 {{
                // SAFETY: rows stay inside the block's window, which is resident
                let fragment = unsafe {
                    ldmatrix_x4(b_address!(window, band_first + NP * 16, KS * 8, lane, TF32_TILE_ROW, 4))
                };
                unsafe {
                    bd[2 * NP] =
                        mma_m16n8k8_f32_tf32(bd[2 * NP], q_v, [tf32(fragment[0]), tf32(fragment[1])]);
                    bd[2 * NP + 1] =
                        mma_m16n8k8_f32_tf32(bd[2 * NP + 1], q_v, [tf32(fragment[2]), tf32(fragment[3])]);
                }
            }});
        }});

        // the FP16 kernel's rel-shift over fewer tiles
        let mut received = [[[0f32; 2]; TF32_WINDOW / 8]; 2];
        seq!(E in 0..2 {{
            let q = 2 * quad + E + shift;
            let source = (4 * group + (q % 8) / 2) as u32;
            let odd = (E + shift) % 2 == 1;
            seq!(N in 0..6 {{
                let low = if odd { bd[N][1] } else { bd[N][0] };
                let high = if odd { bd[N][3] } else { bd[N][2] };
                received[0][N][E] = warp::shuffle_f32(low, source);
                received[1][N][E] = warp::shuffle_f32(high, source);
            }});
        }});
        let mut s = [[0f32; 4]; TF32_KEY_TILE / 8];
        seq!(E in 0..2 {{
            let carry = 2 * quad + E + shift >= 8;
            seq!(NT in 0..4 {{
                s[NT][E] = if carry { received[0][NT + 2][E] } else { received[0][NT + 1][E] };
                s[NT][2 + E] = if carry { received[1][NT + 1][E] } else { received[1][NT][E] };
            }});
        }});

        // content products (q + u) · kᵀ on top of the position term
        seq!(KS in 0..16 {{
            seq!(NP in 0..2 {{
                // SAFETY: rows stay inside the resident key tile
                let fragment = unsafe {
                    ldmatrix_x4(b_address!(keys, NP * 16, KS * 8, lane, TF32_TILE_ROW, 4))
                };
                unsafe {
                    s[2 * NP] =
                        mma_m16n8k8_f32_tf32(s[2 * NP], q_u[KS], [tf32(fragment[0]), tf32(fragment[1])]);
                    s[2 * NP + 1] = mma_m16n8k8_f32_tf32(
                        s[2 * NP + 1],
                        q_u[KS],
                        [tf32(fragment[2]), tf32(fragment[3])],
                    );
                }
            }});
        }});

        // scale, mask, and fold into the running softmax
        let mut tile_max = row_max;
        seq!(NT in 0..4 {{
            seq!(E in 0..4 {{
                let jj = NT * 8 + 2 * quad + E % 2;
                s[NT][E] = if j0 + jj < len {
                    s[NT][E] * scale
                } else {
                    f32::NEG_INFINITY
                };
                tile_max[E / 2] = tile_max[E / 2].max(s[NT][E]);
            }});
        }});
        let mut correction = [0f32; 2];
        seq!(X in 0..2 {{
            tile_max[X] = tile_max[X].max(warp::shuffle_xor_f32(tile_max[X], 1));
            tile_max[X] = tile_max[X].max(warp::shuffle_xor_f32(tile_max[X], 2));
            correction[X] = ex2_approx_ftz_f32(row_max[X] - tile_max[X]);
            row_max[X] = tile_max[X];
            row_sum[X] *= correction[X];
        }});
        let mut weights = [[0u32; 4]; TF32_KEY_TILE / 8];
        seq!(NT in 0..4 {{
            let p = [
                ex2_approx_ftz_f32(s[NT][0] - row_max[0]),
                ex2_approx_ftz_f32(s[NT][1] - row_max[0]),
                ex2_approx_ftz_f32(s[NT][2] - row_max[1]),
                ex2_approx_ftz_f32(s[NT][3] - row_max[1]),
            ];
            row_sum[0] += p[0] + p[1];
            row_sum[1] += p[2] + p[3];
            // in the key order `2t, 2t + 1` for A columns `t, t + 4`
            weights[NT] = [
                tf32(p[0].to_bits()),
                tf32(p[2].to_bits()),
                tf32(p[1].to_bits()),
                tf32(p[3].to_bits()),
            ];
        }});
        seq!(DT in 0..16 {{
            o[DT][0] *= correction[0];
            o[DT][1] *= correction[0];
            o[DT][2] *= correction[1];
            o[DT][3] *= correction[1];
        }});

        // the values are in, and every warp is done with this key tile and window
        unsafe { cp_async_wait_all() };
        thread::sync_threads();
        let next = j0 + TF32_KEY_TILE;
        if next < len {
            unsafe {
                stage_rows!(
                    keys,
                    batch_rows,
                    qkv_stride,
                    d_model + column,
                    next as isize,
                    TF32_KEY_TILE,
                    frames,
                    TF32_TILE_ROW,
                    4
                );
                stage_rows!(
                    window,
                    pos.as_ptr(),
                    d_model,
                    column,
                    window_first(next),
                    block_rows + TF32_KEY_TILE,
                    positions,
                    TF32_TILE_ROW,
                    4
                );
                cp_async_commit_group();
            }
        }

        // o += weights · v, output tiles `2 * DP` and `2 * DP + 1` holding value columns
        // `16 * DP + 2 * n` and `16 * DP + 2 * n + 1` for MMA column `n`
        seq!(NT in 0..4 {{
            let key = NT * 8 + 2 * quad;
            seq!(DP in 0..8 {{
                let at = key * TF32_TILE_ROW + DP * 16 + 2 * group;
                // SAFETY: rows stay inside the resident value tile
                let (even, odd) = unsafe {
                    (
                        *values.add(at).cast::<[u32; 2]>(),
                        *values.add(at + TF32_TILE_ROW).cast::<[u32; 2]>(),
                    )
                };
                unsafe {
                    o[2 * DP] = mma_m16n8k8_f32_tf32(o[2 * DP], weights[NT], [tf32(even[0]), tf32(odd[0])]);
                    o[2 * DP + 1] =
                        mma_m16n8k8_f32_tf32(o[2 * DP + 1], weights[NT], [tf32(even[1]), tf32(odd[1])]);
                }
            }});
        }});
        j0 = next;
    }

    seq!(X in 0..2 {{
        row_sum[X] += warp::shuffle_xor_f32(row_sum[X], 1);
        row_sum[X] += warp::shuffle_xor_f32(row_sum[X], 2);
    }});
    seq!(HALF in 0..2 {{
        let i = first_query + warp_index * 16 + group + HALF * 8;
        if i < frames {
            // as in the FP16 kernel
            let valid = i < len;
            let inverse = 1.0 / row_sum[HALF];
            let row = (b * frames + i) * d_model + column;
            seq!(DP in 0..8 {{
                let columns = if valid {
                    [
                        o[2 * DP][2 * HALF] * inverse,
                        o[2 * DP + 1][2 * HALF] * inverse,
                        o[2 * DP][2 * HALF + 1] * inverse,
                        o[2 * DP + 1][2 * HALF + 1] * inverse,
                    ]
                } else {
                    [0.0; 4]
                };
                // SAFETY: each lane writes its own columns of rows this block owns
                unsafe { *out_rows.add(row + DP * 16 + 4 * quad).cast::<[f32; 4]>() = columns };
            }});
        }
    }});
}
