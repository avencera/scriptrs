//! Low-precision GEMM operands for the encoder's linear layers
//!
//! The `quant` kernels quantize the FP16 activations the encoder kernels write into
//! the formats cuBLASLt's FP8 and NVFP4 products read, and dequantize INT8 weights to
//! FP16 for the FP16 product. Formats are encoded in
//! software with round-to-nearest-even, so the PTX stays at the `sm_75` baseline and
//! matches the host quantizer in `src/cuda/quant.rs` bit for bit.
//!
//! NVFP4 block scales use cuBLASLt's tiled layout: the `[rows, blocks]` scale matrix is cut
//! into 128x4 tiles of 512 bytes, tiles row-major over the matrix, and inside a tile
//! row `r` and column `c` sit at `(r % 32) * 16 + (r / 32) * 4 + c`

use cuda_device::{DisjointSlice, SharedArray, kernel, thread};

use crate::common::{MAX_WARPS, block_reduce};

/// Per-block partial maxima [`quant_amax`] writes; a quantize kernel reading them
/// runs at least this many threads per block
pub const AMAX_PARTIALS: usize = 256;

/// E4M3: 3 mantissa bits, bias 7, largest finite 448
const E4M3: (u32, i32) = (3, 7);
const E4M3_MAX: f32 = 448.0;
/// E2M1: 1 mantissa bit, bias 1, largest 6
const E2M1: (u32, i32) = (1, 1);
const E2M1_MAX: f32 = 6.0;

fn max(a: f32, b: f32) -> f32 {
    a.max(b)
}

#[inline(always)]
fn f16_value(bits: u16) -> f32 {
    f16::from_bits(bits) as f32
}

/// `2^exponent` for `-126 <= exponent <= 127`
#[inline(always)]
fn pow2(exponent: i32) -> f32 {
    f32::from_bits(((exponent + 127) as u32) << 23)
}

/// Encodes `a`, finite, non-negative and at most the format's largest value, in a
/// minifloat with `mbits` mantissa bits and exponent bias `bias`, subnormals included,
/// rounding to nearest even
#[inline(always)]
fn encode(a: f32, (mbits, bias): (u32, i32)) -> u32 {
    let exponent = ((a.to_bits() >> 23) & 0xff) as i32 - 127;
    let emin = 1 - bias;
    let mut ulp = if exponent < emin { emin } else { exponent } - mbits as i32;
    // exact: a power-of-two scale, and the result is below 2^(mbits + 1)
    let scaled = a * pow2(-ulp);
    let floor = scaled as u32;
    let fraction = scaled - floor as f32;
    let mut q = if fraction > 0.5 || (fraction == 0.5 && floor & 1 == 1) {
        floor + 1
    } else {
        floor
    };
    let one = 1u32 << mbits;
    if q == 2 * one {
        q = one;
        ulp += 1;
    }
    if q < one {
        q
    } else {
        (((ulp + mbits as i32 + bias) as u32) << mbits) | (q - one)
    }
}

/// The value of an unsigned minifloat code from [`encode`]
#[inline(always)]
fn decode(code: u32, (mbits, bias): (u32, i32)) -> f32 {
    let exponent = (code >> mbits) as i32;
    let mantissa = code & ((1 << mbits) - 1);
    if exponent == 0 {
        mantissa as f32 * pow2(1 - bias - mbits as i32)
    } else {
        ((1 << mbits) + mantissa) as f32 * pow2(exponent - bias - mbits as i32)
    }
}

/// Signed E4M3 bits of `value`, saturating at ±448
#[inline(always)]
fn e4m3(value: f32) -> u8 {
    let sign = if value < 0.0 { 0x80 } else { 0 };
    (sign | encode(value.abs().min(E4M3_MAX), E4M3)) as u8
}

/// Signed E2M1 bits of `value` in the low nibble, saturating at ±6
#[inline(always)]
fn e2m1(value: f32) -> u8 {
    let sign = if value < 0.0 { 0x8 } else { 0 };
    (sign | encode(value.abs().min(E2M1_MAX), E2M1)) as u8
}

/// Byte offset of block scale `(row, column)` in cuBLASLt's tiled layout, for a
/// matrix of `tile_columns` 4-column tiles
#[inline(always)]
fn scale_offset(row: usize, column: usize, tile_columns: usize) -> usize {
    let tile = (row / 128) * tile_columns + column / 4;
    let r = row % 128;
    tile * 512 + (r % 32) * 16 + (r / 32) * 4 + column % 4
}

/// The tensor's largest magnitude from [`quant_amax`]'s partials, in every thread;
/// every thread of the block must call it
#[inline(always)]
fn tensor_amax(partials: &[f32], scratch: *mut f32) -> f32 {
    let tid = thread::threadIdx_x() as usize;
    let partial = if tid < partials.len() {
        partials[tid]
    } else {
        0.0
    };
    block_reduce(partial, scratch, max)
}

/// Largest magnitude of FP16 `x`, as one partial per block
///
/// Grid-stride over `x` with `partials.len()` blocks; the quantize kernels fold the
/// partials, so no atomics or memset are needed between products
#[kernel]
pub fn quant_amax(x: &[u16], mut partials: DisjointSlice<f32>) {
    static mut SCRATCH: SharedArray<f32, MAX_WARPS> = SharedArray::UNINIT;
    // SAFETY: a raw pointer to the block's shared array; only `block_reduce` touches
    // it, between barriers
    let scratch = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };
    let threads = (thread::gridDim_x() * thread::blockDim_x()) as usize;
    let mut i = thread::index_1d().get();
    let mut local = 0.0f32;
    while i < x.len() {
        local = local.max(f16_value(x[i]).abs());
        i += threads;
    }
    let block = block_reduce(local, scratch, max);
    if thread::threadIdx_x() == 0 {
        let b = thread::blockIdx_x() as usize;
        // SAFETY: one block per partial, and only its first thread writes
        unsafe { *partials.get_unchecked_mut(b) = block };
    }
}

/// FP8 E4M3 of FP16 `x` with one dynamic scale for the tensor, `amax / 448` from
/// [`quant_amax`]'s `partials`, written to `scale[0]`
///
/// One thread per element, at least [`AMAX_PARTIALS`] threads per block
#[kernel]
pub fn quant_fp8(
    x: &[u16],
    partials: &[f32],
    mut out: DisjointSlice<u8>,
    mut scale: DisjointSlice<f32>,
) {
    static mut SCRATCH: SharedArray<f32, MAX_WARPS> = SharedArray::UNINIT;
    // SAFETY: as in `quant_amax`
    let scratch = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };
    let amax = tensor_amax(partials, scratch);
    let value = if amax > 0.0 { amax / E4M3_MAX } else { 1.0 };
    let i = thread::index_1d().get();
    if i == 0 {
        // SAFETY: only the first thread writes
        unsafe { *scale.get_unchecked_mut(0) = value };
    }
    if i < x.len() {
        // SAFETY: one thread per element
        unsafe { *out.get_unchecked_mut(i) = e4m3(f16_value(x[i]) / value) };
    }
}

/// NVFP4 of FP16 `x`, rows of `width`: E2M1 values in blocks of 16 along the row, two
/// to a byte with the even element in the low nibble, each block with an E4M3 scale
/// relative to the tensor scale `g = amax / (448 * 6)` from [`quant_amax`]'s
/// `partials`
///
/// cuBLASLt's block-scaled product has no tensor scale, so `alpha[0]` gets
/// `alpha_base * g`, with `alpha_base` the product's alpha times the weight's tensor
/// scale. One thread per block, at least [`AMAX_PARTIALS`] threads per block;
/// `scales` is the tiled layout for `tile_columns` tiles
#[kernel]
pub fn quant_nvfp4(
    x: &[u16],
    partials: &[f32],
    mut out: DisjointSlice<u8>,
    mut scales: DisjointSlice<u8>,
    mut alpha: DisjointSlice<f32>,
    width: u32,
    tile_columns: u32,
    alpha_base: f32,
) {
    static mut SCRATCH: SharedArray<f32, MAX_WARPS> = SharedArray::UNINIT;
    // SAFETY: as in `quant_amax`
    let scratch = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };
    let amax = tensor_amax(partials, scratch);
    let global = if amax > 0.0 {
        amax / (E4M3_MAX * E2M1_MAX)
    } else {
        1.0
    };
    let b = thread::index_1d().get();
    if b == 0 {
        // SAFETY: only the first thread writes
        unsafe { *alpha.get_unchecked_mut(0) = alpha_base * global };
    }
    let width = width as usize;
    let per_row = width / 16;
    if b * 16 >= x.len() {
        return;
    }
    let (row, column) = (b / per_row, b % per_row);
    let start = b * 16;
    let mut block_amax = 0.0f32;
    let mut i = 0;
    while i < 16 {
        block_amax = block_amax.max(f16_value(x[start + i]).abs());
        i += 1;
    }
    let code = encode((block_amax / E2M1_MAX / global).min(E4M3_MAX), E4M3);
    let block_scale = decode(code, E4M3) * global;
    // SAFETY: each thread owns its block's bytes and scale
    unsafe {
        *scales.get_unchecked_mut(scale_offset(row, column, tile_columns as usize)) = code as u8;
    }
    let mut i = 0;
    while i < 16 {
        let (low, high) = if block_scale > 0.0 {
            (
                e2m1(f16_value(x[start + i]) / block_scale),
                e2m1(f16_value(x[start + i + 1]) / block_scale),
            )
        } else {
            (0, 0)
        };
        // SAFETY: as above
        unsafe { *out.get_unchecked_mut(start / 2 + i / 2) = low | (high << 4) };
        i += 2;
    }
}

/// FP16 of an INT8 weight with one scale per `group` consecutive elements, one thread
/// per element
#[kernel]
pub fn quant_dequant_i8(q: &[u8], scales: &[f32], mut out: DisjointSlice<u16>, group: u32) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(slot) = out.get_mut(idx) {
        let value = q[i] as i8 as f32 * scales[i / group as usize];
        *slot = (value as f16).to_bits();
    }
}
