//! Boundary between the pipelines and a Parakeet runtime
//!
//! A backend turns a batch of encoder chunks into greedy TDT output: the
//! encoder plus the whole decode loop. Device backends run the loop on the
//! device. Step backends, such as Core ML, supply encoder, decoder and joint
//! calls and share the host loop in [`host_loop`]. Either way, the token rule
//! comes from one [`DecodePolicy`].

#[cfg(any(test, target_os = "macos"))]
mod host_loop;
mod policy;

use ndarray::ArrayView2;

use crate::decode::RawTranscription;
use crate::language::TopKCandidates;

#[cfg(target_os = "macos")]
pub(crate) use host_loop::{HostLoop, StepRuntime};
pub(crate) use policy::DecodePolicy;

/// Log-mel input for one chunk, in `[frames, features]` order
#[derive(Debug, Clone, Copy)]
pub(crate) struct EncoderChunk<'a> {
    features: ArrayView2<'a, f32>,
}

impl<'a> EncoderChunk<'a> {
    pub(crate) fn new(features: ArrayView2<'a, f32>) -> Self {
        Self { features }
    }

    pub(crate) fn features(&self) -> ArrayView2<'a, f32> {
        self.features
    }

    pub(crate) fn frames(&self) -> usize {
        self.features.nrows()
    }
}

/// Greedy TDT output for one chunk
#[derive(Debug, Clone)]
pub(crate) struct ChunkDecode {
    /// Emitted tokens with encoder frames, durations and confidences
    pub raw: RawTranscription,
    /// Joint top-K candidates per emitted token, when the policy asked for them
    #[cfg_attr(not(test), allow(dead_code))]
    pub top_k: Option<Vec<TopKCandidates>>,
}
