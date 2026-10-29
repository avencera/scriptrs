//! Block reductions and activations shared by the kernel areas

use cuda_device::{thread, warp};

/// Upper bound on warps per block for the block reductions
pub const MAX_WARPS: usize = 32;

/// Reduces `value` over the block with `combine` and returns the result to every
/// thread
///
/// Every thread folds the warp partials in the same order, so all threads see the
/// same result
#[inline(always)]
pub fn block_reduce(value: f32, scratch: *mut f32, combine: impl Fn(f32, f32) -> f32) -> f32 {
    let warp_index = thread::threadIdx_x() as usize / 32;
    let warps = thread::blockDim_x() as usize / 32;
    let mut partial = value;
    let mut offset = 16u32;
    while offset > 0 {
        partial = combine(partial, warp::shuffle_xor_f32(partial, offset));
        offset /= 2;
    }
    if warp::lane_id() == 0 {
        // SAFETY: `warp_index < MAX_WARPS` because a block has at most 1024 threads
        unsafe { *scratch.add(warp_index) = partial };
    }
    thread::sync_threads();

    // SAFETY: every warp wrote its slot before the barrier above, and there is at
    // least one warp
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

#[cfg(any(feature = "encoder", feature = "decode"))]
pub fn add(a: f32, b: f32) -> f32 {
    a + b
}

/// torch's `sigmoid`, `1 / (1 + exp(-x))`
#[cfg(any(feature = "encoder", feature = "decode"))]
#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Rounds `value` to the nearest FP16, returned as its bits
///
/// Past ±65504 it overflows to infinity, as torch's `half()` does
#[cfg(feature = "encoder")]
#[inline(always)]
pub fn to_f16(value: f32) -> u16 {
    (value as f16).to_bits()
}
