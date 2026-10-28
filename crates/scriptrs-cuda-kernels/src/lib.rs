//! cuda-oxide device kernels for the scriptrs CUDA backend
//!
//! Each area is a module behind its own feature and compiles to one PTX file in
//! `src/cuda/ptx/<area>.ptx` of the scriptrs tree (`just cuda-kernels`). Entry names
//! carry the area prefix because the host looks kernels up by name

#[cfg(any(feature = "encoder", feature = "decode"))]
mod common;
#[cfg(feature = "decode")]
pub mod decode;
#[cfg(feature = "encoder")]
pub mod encoder;
