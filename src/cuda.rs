//! Native CUDA backend
//!
//! Host code runs on cudakit (driver-only loading, cuBLAS, graph capture); the
//! device code is `crates/scriptrs-cuda-kernels`, compiled to the PTX in `src/cuda/ptx`

pub(crate) mod decoder;
pub(crate) mod encoder;
pub(crate) mod kernels;
pub(crate) mod model;
#[cfg(test)]
pub(crate) mod test_support;

use std::sync::Arc;

use cudakit::cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};
use cudakit::{DeviceTensor, Error, Math, Sgemm};

use crate::cuda::kernels::CudaRuntime;
use crate::error::TranscriptionError;
use crate::native::weights::WeightFile;

pub(crate) fn upload(
    runtime: &CudaRuntime,
    file: &WeightFile,
    name: &str,
    shape: &[usize],
) -> Result<DeviceTensor, TranscriptionError> {
    let values = file.tensor(name, shape)?;
    Ok(DeviceTensor::upload(runtime.stream(), values, shape)?)
}

pub(crate) fn upload_host(
    runtime: &CudaRuntime,
    values: &[f32],
) -> Result<DeviceTensor, TranscriptionError> {
    Ok(DeviceTensor::upload(
        runtime.stream(),
        values,
        &[values.len()],
    )?)
}

pub(crate) fn zeros<T>(stream: &Arc<CudaStream>, len: usize) -> Result<CudaSlice<T>, Error>
where
    T: DeviceRepr + ValidAsZeroBits,
{
    Ok(stream.alloc_zeros(len)?)
}

/// `c = alpha * a · wᵀ + beta * c` for a PyTorch `[out, in]` weight, in FP32
pub(crate) fn linear(rows: usize, out: usize, inputs: usize, alpha: f32, beta: f32) -> Sgemm {
    Sgemm {
        b_transposed: true,
        alpha,
        beta,
        math: Math::Fp32,
        ..Sgemm::new(rows, out, inputs)
    }
}
