//! Native CUDA backend
//!
//! Host code runs on cudakit (driver-only loading, cuBLAS, graph capture); the
//! device code is `crates/scriptrs-cuda-kernels`, compiled to the PTX in `src/cuda/ptx`

pub(crate) mod batcher;
pub(crate) mod decoder;
pub(crate) mod encoder;
pub(crate) mod fp16;
pub(crate) mod kernels;
#[cfg(feature = "low-precision")]
pub(crate) mod lt;
pub(crate) mod model;
#[cfg(feature = "low-precision")]
pub(crate) mod quant;
#[cfg(feature = "low-precision")]
pub(crate) mod stored;
pub(crate) mod streaming;
#[cfg(test)]
pub(crate) mod test_support;

use std::sync::Arc;

use cudakit::cudarc::driver::sys::CU_MEMHOSTALLOC_WRITECOMBINED;
use cudakit::cudarc::driver::{
    CudaEvent, CudaSlice, CudaStream, DeviceRepr, PinnedHostSlice, ValidAsZeroBits,
};
use cudakit::{DeviceTensor, Error, Graph, Math, Sgemm};

use crate::cuda::encoder::EncoderMath;
use crate::cuda::kernels::CudaRuntime;
use crate::cuda::model::exclusive_device_work;
#[cfg(feature = "low-precision")]
use crate::cuda::quant::Scheme;
use crate::error::TranscriptionError;
use crate::models::Precision;
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

/// Which way a [`HostStaging`] buffer carries data
#[derive(Debug, Clone, Copy)]
pub(crate) enum Direction {
    /// The host writes, the device reads: write-combined, which is fast to fill and
    /// to copy from but slow for the host to read
    Upload,
    /// The device writes, the host reads
    Download,
}

/// Page-locked host memory for copies that queue on the stream without blocking
///
/// Allocating and freeing it can synchronize the device, so workspaces holding it
/// are built and dropped under the model's exclusive device lock, away from other
/// threads' graph captures
///
/// Copies through pageable memory make the driver stage the data, and a download
/// blocks the host until it lands. These copies queue like kernels, and `done`
/// marks when the last one finished. cudarc records the pinned slice's own event
/// only while it manages stream synchronization, which cudakit turns off during
/// graph capture, so `done` keeps the order whatever that setting
pub(crate) struct HostStaging<T> {
    buffer: PinnedHostSlice<T>,
    /// Recorded after each queued copy; the host waits on it before touching
    /// `buffer`
    done: CudaEvent,
}

impl<T: DeviceRepr + ValidAsZeroBits> HostStaging<T> {
    /// A zeroed buffer of `len` values
    pub(crate) fn new(
        stream: &Arc<CudaStream>,
        len: usize,
        direction: Direction,
    ) -> Result<Self, Error> {
        let flags = match direction {
            Direction::Upload => CU_MEMHOSTALLOC_WRITECOMBINED,
            Direction::Download => 0,
        };
        let context = stream.context();
        // SAFETY: the memory is zeroed below before anything reads it, and zero is a
        // valid `T`
        let mut buffer = unsafe { context.alloc_pinned_with_flags::<T>(len, flags) }?;
        let done = context.new_event(None)?;
        // SAFETY: the pointer covers `len` values, nothing else refers to the fresh
        // allocation, and all-zero bytes are a valid `T`
        unsafe { std::ptr::write_bytes(buffer.as_mut_ptr()?, 0, len) };
        Ok(Self { buffer, done })
    }

    /// The buffer for the host to fill, once every queued copy from it is done
    pub(crate) fn write(&mut self) -> Result<&mut [T], Error> {
        self.done.synchronize()?;
        Ok(self.buffer.as_mut_slice()?)
    }

    /// The buffer for the host to read, once every queued copy into it is done
    pub(crate) fn read(&self) -> Result<&[T], Error> {
        self.done.synchronize()?;
        Ok(self.buffer.as_slice()?)
    }

    /// Queues a copy of the whole buffer into `device`
    pub(crate) fn upload(
        &self,
        stream: &Arc<CudaStream>,
        device: &mut CudaSlice<T>,
    ) -> Result<(), Error> {
        stream.memcpy_htod(&self.buffer, device)?;
        Ok(self.done.record(stream)?)
    }

    /// Queues a copy of `device` into the front of the buffer
    pub(crate) fn download(
        &mut self,
        stream: &Arc<CudaStream>,
        device: &CudaSlice<T>,
    ) -> Result<(), Error> {
        stream.memcpy_dtoh(device, &mut self.buffer)?;
        Ok(self.done.record(stream)?)
    }
}

impl<T> Drop for HostStaging<T> {
    fn drop(&mut self) {
        // a copy still queued would read or write the memory freed next
        let _ = self.done.synchronize();
    }
}

/// Captures `enqueue` into a CUDA graph, apart from every other engine's loads,
/// captures and teardown
///
/// # Safety
///
/// [`cudakit::Runtime::capture`]'s contract: every buffer `enqueue` touches must
/// stay allocated and in place for as long as the graph exists, and be used only on
/// the runtime's stream. `enqueue` must return errors rather than panic, since a
/// panic leaves the stream capturing
pub(crate) unsafe fn capture(
    runtime: &CudaRuntime,
    enqueue: impl FnOnce() -> Result<(), TranscriptionError>,
) -> Result<Graph, TranscriptionError> {
    let _exclusive = exclusive_device_work();
    // SAFETY: the caller upholds the contract for `enqueue`
    unsafe { runtime.capture_nonempty(enqueue) }
}

/// Launches `graph`, or the first time runs `enqueue` eagerly, which also finishes
/// cuBLAS's lazy setup, then captures it into `graph` for later calls
///
/// # Safety
///
/// [`capture`]'s contract for `enqueue`, kept for as long as `graph` holds the graph
pub(crate) unsafe fn replay_or_capture(
    runtime: &CudaRuntime,
    graph: &mut Option<Graph>,
    mut enqueue: impl FnMut() -> Result<(), TranscriptionError>,
) -> Result<(), TranscriptionError> {
    match graph {
        Some(captured) => captured.launch()?,
        None => {
            enqueue()?;
            // SAFETY: the caller upholds the contract for `enqueue` and `graph`
            *graph = Some(unsafe { capture(runtime, enqueue) }?);
        }
    }
    Ok(())
}

/// The encoder GEMM precision of a model precision
pub(crate) fn math(precision: Precision) -> EncoderMath {
    match precision {
        Precision::Fp32 => EncoderMath::Single(Math::Fp32),
        Precision::Tf32 => EncoderMath::Single(Math::Tf32),
        Precision::Fp16 => EncoderMath::Fp16,
        #[cfg(feature = "low-precision")]
        Precision::Fp8 => EncoderMath::Quantized(Scheme::Fp8),
        #[cfg(feature = "low-precision")]
        Precision::Int8 => EncoderMath::Quantized(Scheme::Int8),
        #[cfg(feature = "low-precision")]
        Precision::Nvfp4 => EncoderMath::Quantized(Scheme::Nvfp4),
    }
}

/// `c = alpha * a · wᵀ + beta * c` for a PyTorch `[out, in]` weight
pub(crate) fn linear(
    rows: usize,
    out: usize,
    inputs: usize,
    alpha: f32,
    beta: f32,
    math: Math,
) -> Sgemm {
    Sgemm {
        b_transposed: true,
        alpha,
        beta,
        math,
        ..Sgemm::new(rows, out, inputs)
    }
}
