//! Native weights shared by the CUDA engine
//!
//! The weights are the NeMo checkpoint's own fp32 tensors, exported to
//! safetensors by `dev-tools/nemo/export_native.py`, so every native kernel is
//! checked against the same numbers NeMo runs.

#[cfg(feature = "vad")]
pub(crate) mod silero;
pub(crate) mod spec;
pub(crate) mod weights;
