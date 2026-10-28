//! Native CUDA backend
//!
//! Host code runs on cudakit (driver-only loading, cuBLAS, graph capture); the
//! device code is `crates/scriptrs-cuda-kernels`, compiled to the PTX in `src/cuda/ptx`

pub(crate) mod encoder;
pub(crate) mod kernels;
