//! One device allocation that every cached shape's per-call buffers alias
//!
//! Most of an encoder workspace holds nothing from one call to the next: each call
//! uploads its input, runs the encoder, and the decode reads the output before the
//! call returns. Calls on one engine run one at a time under its shape cache's lock,
//! so every cached shape can carve those buffers from the same memory. A cached shape
//! then costs its graphs and the few buffers whose contents last, instead of a whole
//! workspace, and a batch-32 pass over LibriSpeech keeps all its shapes cached rather
//! than evicting each before its next use

use std::mem::{ManuallyDrop, size_of};
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use cudakit::cudarc::driver::{
    CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, DeviceSlice, SyncOnDrop, sys,
};

use crate::error::TranscriptionError;

/// Byte alignment of every alias, the same as a buffer allocated on its own, so GEMMs
/// and vectorized kernels see the alignment they would without the arena
const ALIGN: usize = 256;

/// Bytes an alias of `len` values of `T` takes in the arena, padding included
///
/// Every alias is padded to [`ALIGN`], so the bytes a set of aliases needs are the
/// sum of this over the set, in any order
pub(crate) fn alias_bytes<T>(len: usize) -> usize {
    (len * size_of::<T>()).next_multiple_of(ALIGN)
}

/// Device memory that shapes carve their per-call buffers from; empty until the
/// first [`Self::reallocate`]
pub(crate) struct Arena {
    memory: Option<CudaSlice<u8>>,
}

impl Arena {
    pub(crate) const fn new() -> Self {
        Self { memory: None }
    }

    /// Device bytes the arena holds
    pub(crate) fn bytes(&self) -> usize {
        self.memory.as_ref().map_or(0, CudaSlice::len)
    }

    /// Frees the arena's memory, then allocates `bytes` of zeroed memory in its place
    ///
    /// The old memory is freed first, so the allocation can reuse it; a failed
    /// allocation leaves the arena empty.
    ///
    /// # Safety
    ///
    /// No alias of the old memory may be used afterward. Aliases may still be
    /// dropped, since dropping one leaves the memory alone
    pub(crate) unsafe fn reallocate(
        &mut self,
        stream: &Arc<CudaStream>,
        bytes: usize,
    ) -> Result<(), TranscriptionError> {
        // the free is stream ordered, so work still queued on the old memory finishes
        // first
        self.memory = None;
        self.memory = Some(stream.alloc_zeros(bytes)?);
        Ok(())
    }

    /// Hands out aliases from the start of the arena
    ///
    /// # Safety
    ///
    /// Aliases from different carves overlap. The caller must use one carve's aliases
    /// at a time, queue all their work on the arena's stream, and expect nothing
    /// written through them to survive the next use of another carve's
    pub(crate) unsafe fn carve(&self) -> Carve<'_> {
        Carve {
            memory: self.memory.as_ref(),
            offset: 0,
        }
    }
}

/// Non-overlapping aliases from one arena, handed out front to back
pub(crate) struct Carve<'a> {
    memory: Option<&'a CudaSlice<u8>>,
    offset: usize,
}

impl Carve<'_> {
    /// The next `len` values of the arena as a buffer of `T`, which holds whatever
    /// the arena last held there
    pub(crate) fn take<T: DeviceRepr>(
        &mut self,
        len: usize,
    ) -> Result<Alias<T>, TranscriptionError> {
        let bytes = alias_bytes::<T>(len);
        let memory = self
            .memory
            .filter(|memory| self.offset + bytes <= memory.len())
            .ok_or_else(|| {
                TranscriptionError::InvalidModelOutput(format!(
                    "a {bytes}-byte buffer does not fit the CUDA arena past byte {}",
                    self.offset
                ))
            })?;

        let stream = memory.stream();
        let (base, _read) = memory.device_ptr(stream);
        // SAFETY: the range lies inside the arena's allocation, which outlives every
        // alias the arena's owner keeps, and no other alias from this carve overlaps
        // it; `Alias` never frees it
        let slice = unsafe { stream.upgrade_device_ptr::<T>(base + self.offset as u64, len) };
        self.offset += bytes;
        Ok(Alias(ManuallyDrop::new(slice)))
    }

    /// Bytes handed out so far
    pub(crate) fn used(&self) -> usize {
        self.offset
    }
}

/// A buffer inside an [`Arena`], which leaves the memory to the arena when dropped
pub(crate) struct Alias<T>(ManuallyDrop<CudaSlice<T>>);

impl<T> Deref for Alias<T> {
    type Target = CudaSlice<T>;

    fn deref(&self) -> &CudaSlice<T> {
        &self.0
    }
}

impl<T> DerefMut for Alias<T> {
    fn deref_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.0
    }
}

impl<T> Drop for Alias<T> {
    fn drop(&mut self) {
        // SAFETY: the slice is taken only here, once
        let slice = unsafe { ManuallyDrop::take(&mut self.0) };
        // queues the stream behind the alias's pending work, then forgets the pointer
        slice.leak();
    }
}

// the generic GEMM and copy calls take these traits rather than `CudaSlice`
impl<T> DeviceSlice<T> for Alias<T> {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn stream(&self) -> &Arc<CudaStream> {
        self.0.stream()
    }
}

impl<T> DevicePtr<T> for Alias<T> {
    fn device_ptr<'a>(&'a self, stream: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        self.0.device_ptr(stream)
    }
}

impl<T> DevicePtrMut<T> for Alias<T> {
    fn device_ptr_mut<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        self.0.device_ptr_mut(stream)
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Arc;

    use cudakit::cudarc::driver::CudaStream;

    use super::Arena;

    impl Arena {
        /// Fills every byte of the arena with `byte`, so a read of memory the
        /// current call has not written shows in its output
        pub(crate) fn poison(&mut self, stream: &Arc<CudaStream>, byte: u8) {
            if let Some(memory) = self.memory.as_mut() {
                let fill = vec![byte; memory.len()];
                stream.memcpy_htod(&fill, memory).expect("poison the arena");
            }
        }
    }
}
