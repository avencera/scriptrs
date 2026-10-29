//! Encoder linear layers with FP16 operands on tensor cores
//!
//! cudakit's [`Sgemm`](cudakit::Sgemm) takes FP32 operands only, so the FP16 products
//! call `cublasGemmEx` on cudakit's shared handle. Operands are FP16 bit patterns in
//! `u16` buffers, which the encoder kernels write; the output and the accumulation
//! stay FP32, so a residual add through `beta = 1` keeps full precision

use std::ffi::{CStr, c_void};

use cudakit::cudarc::cublas::result::gemm_ex;
use cudakit::cudarc::cublas::sys::{
    cublasComputeType_t, cublasGemmAlgo_t, cublasOperation_t, cudaDataType,
};
use cudakit::cudarc::driver::{DevicePtr, DevicePtrMut};
use cudakit::{Error, Math, Sgemm, check_len, to_c_int};

use crate::cuda::kernels::CudaRuntime;

/// cuBLAS entry points the FP16 products call beyond cudakit's own
pub(crate) const CUBLAS_SYMBOLS: &[&CStr] = &[c"cublasGemmEx"];

/// Runs `spec`, a [`linear`](crate::cuda::linear) product `c = alpha * a · wᵀ + beta *
/// c`, with `a` and `w` in FP16 and `c` in FP32, on the runtime's stream
///
/// `spec.math` does not apply: 16-bit operands with FP32 compute run on tensor cores
/// in any math mode
pub(crate) fn linear_fp16<A, W, C>(
    runtime: &CudaRuntime,
    spec: Sgemm,
    a: &A,
    w: &W,
    c: &mut C,
) -> Result<(), Error>
where
    A: DevicePtr<u16>,
    W: DevicePtr<u16>,
    C: DevicePtrMut<f32>,
{
    let context = "fp16 linear";
    if spec.a_transposed || !spec.b_transposed {
        return Err(Error::Unsupported {
            context,
            reason: "only `a · wᵀ` products take FP16 operands".to_owned(),
        });
    }
    check_len("fp16 linear a", spec.m.saturating_mul(spec.k), a.len())?;
    check_len("fp16 linear w", spec.n.saturating_mul(spec.k), w.len())?;
    check_len("fp16 linear c", spec.m.saturating_mul(spec.n), c.len())?;
    let (rows, out, inputs) = (
        to_c_int(context, spec.m)?,
        to_c_int(context, spec.n)?,
        to_c_int(context, spec.k)?,
    );

    let stream = runtime.stream();
    // SAFETY: the call below passes host scalars on the handle's own stream and leaves
    // its pointer mode, math mode, stream and workspace unchanged
    let blas = unsafe { runtime.blas(Math::Fp32) }?;
    let (a_ptr, _a_sync) = a.device_ptr(stream);
    let (w_ptr, _w_sync) = w.device_ptr(stream);
    let (c_ptr, _c_sync) = c.device_ptr_mut(stream);
    // row-major `c [rows, out]` is column-major `cᵀ [out, rows] = w · aᵀ`; the
    // row-major `[out, in]` weight read column-major is `wᵀ`, hence `T` on it
    // SAFETY: the lengths match the dimensions and leading dimensions checked above,
    // the buffers are device allocations ordered on the handle's stream, and `alpha`
    // and `beta` are host FP32 scalars, as FP32 compute in host pointer mode takes
    unsafe {
        gemm_ex(
            *blas.handle(),
            cublasOperation_t::CUBLAS_OP_T,
            cublasOperation_t::CUBLAS_OP_N,
            out,
            rows,
            inputs,
            (&raw const spec.alpha).cast::<c_void>(),
            w_ptr as *const c_void,
            cudaDataType::CUDA_R_16F,
            inputs,
            a_ptr as *const c_void,
            cudaDataType::CUDA_R_16F,
            inputs,
            (&raw const spec.beta).cast::<c_void>(),
            c_ptr as *mut c_void,
            cudaDataType::CUDA_R_32F,
            out,
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    }?;
    Ok(())
}
