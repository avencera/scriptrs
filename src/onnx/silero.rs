use std::path::Path;
use std::sync::Arc;

use crate::constants::{SAMPLE_RATE, VAD_CONTEXT_SAMPLES, VAD_STATE_SIZE, VAD_WINDOW_SAMPLES};
use crate::error::TranscriptionError;
use crate::onnx::contract::{
    GraphContract, SILERO_INPUT, SILERO_OUTPUT, SILERO_SAMPLE_RATE, SILERO_STATE_IN,
    SILERO_STATE_OUT,
};
use crate::onnx::error::{GraphKind, OnnxBundleError};
use crate::onnx::session::{GraphSession, SessionProvider, f32_input, i64_scalar_input};

/// Silero VAD v5 window classifier running through ONNX Runtime
///
/// The graph consumes one window of context plus new samples together with
/// the opaque `[2, 1, 128]` recurrent state and returns the next state. The
/// VAD owner in `long_form::vad` drives chunking, context, tail padding, and
/// state reset; this adapter only moves tensors.
#[derive(Debug, Clone)]
pub(crate) struct SileroOnnxModel {
    session: Arc<GraphSession>,
}

impl SileroOnnxModel {
    pub(crate) fn load(path: &Path, provider: SessionProvider) -> Result<Self, OnnxBundleError> {
        let session = GraphSession::load(
            GraphKind::SileroVad,
            path,
            provider,
            GraphContract::silero_vad(),
        )?;
        Ok(Self {
            session: Arc::new(session),
        })
    }

    /// Classify one window and advance `state` in place
    ///
    /// `window` holds `VAD_CONTEXT_SAMPLES` context samples followed by
    /// `VAD_WINDOW_SAMPLES` new samples. `state` holds the two recurrent
    /// vectors back to back.
    pub(crate) fn run(&self, window: &[f32], state: &mut [f32]) -> Result<f32, TranscriptionError> {
        let window_len = VAD_WINDOW_SAMPLES + VAD_CONTEXT_SAMPLES;
        if window.len() != window_len {
            return Err(TranscriptionError::InvalidModelOutput(format!(
                "VAD window holds {} samples but the graph takes {window_len}",
                window.len()
            )));
        }
        if state.len() != 2 * VAD_STATE_SIZE {
            return Err(TranscriptionError::InvalidModelOutput(format!(
                "VAD state holds {} values but the graph takes {}",
                state.len(),
                2 * VAD_STATE_SIZE
            )));
        }

        let sample_rate = [SAMPLE_RATE as i64];
        let inputs = vec![
            f32_input(SILERO_INPUT, &[1, window_len], window)?,
            f32_input(SILERO_STATE_IN, &[2, 1, VAD_STATE_SIZE], state)?,
            i64_scalar_input(SILERO_SAMPLE_RATE, &sample_rate)?,
        ];

        let (probability, next_state) = self.session.run(inputs, |outputs| {
            let probability = outputs.f32(SILERO_OUTPUT)?.data[0];
            if !(0.0..=1.0).contains(&probability) {
                return Err(TranscriptionError::InvalidModelOutput(format!(
                    "VAD probability {probability} was outside 0..=1"
                )));
            }

            let next_state = outputs.f32(SILERO_STATE_OUT)?.data;
            if next_state.iter().any(|value| !value.is_finite()) {
                return Err(TranscriptionError::InvalidModelOutput(
                    "VAD recurrent state was not finite".to_owned(),
                ));
            }

            Ok((probability, next_state))
        })?;

        state.copy_from_slice(&next_state);
        Ok(probability)
    }
}
