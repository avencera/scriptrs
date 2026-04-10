use std::path::Path;

use crate::constants::{VAD_CONTEXT_SAMPLES, VAD_STATE_SIZE, VAD_WINDOW_SAMPLES};
#[cfg(target_os = "macos")]
use crate::coreml::SileroVadCoreMlModel;
use crate::error::TranscriptionError;
use crate::long_form::planner::VadConfig;

#[derive(Debug, Clone)]
pub(crate) struct SileroVad {
    model: VadModelInner,
}

impl SileroVad {
    pub(crate) fn new(model_path: &Path) -> Result<Self, TranscriptionError> {
        Ok(Self {
            model: VadModelInner::new(model_path)?,
        })
    }

    pub(crate) fn process(
        &self,
        audio: &[f32],
        config: &VadConfig,
    ) -> Result<Vec<f32>, TranscriptionError> {
        self.model.process(audio, config)
    }
}

#[derive(Debug, Clone)]
enum VadModelInner {
    #[cfg(target_os = "macos")]
    CoreMl(SileroVadCoreMlModel),
    #[cfg(not(target_os = "macos"))]
    Unsupported,
}

impl VadModelInner {
    fn new(model_path: &Path) -> Result<Self, TranscriptionError> {
        #[cfg(target_os = "macos")]
        {
            Ok(Self::CoreMl(SileroVadCoreMlModel::new(model_path)?))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = model_path;
            Err(TranscriptionError::UnsupportedPlatform)
        }
    }

    fn process(&self, audio: &[f32], _config: &VadConfig) -> Result<Vec<f32>, TranscriptionError> {
        match self {
            #[cfg(target_os = "macos")]
            Self::CoreMl(model) => process_coreml_vad(model, audio),
            #[cfg(not(target_os = "macos"))]
            Self::Unsupported => Err(TranscriptionError::UnsupportedPlatform),
        }
    }
}

#[cfg(target_os = "macos")]
fn process_coreml_vad(
    model: &SileroVadCoreMlModel,
    audio: &[f32],
) -> Result<Vec<f32>, TranscriptionError> {
    if audio.is_empty() {
        return Ok(Vec::new());
    }

    let mut probabilities = Vec::with_capacity(audio.len().div_ceil(VAD_WINDOW_SAMPLES));
    let mut hidden_state = vec![0.0f32; VAD_STATE_SIZE];
    let mut cell_state = vec![0.0f32; VAD_STATE_SIZE];
    let mut context = vec![0.0f32; VAD_CONTEXT_SAMPLES];
    let mut chunk = vec![0.0f32; VAD_WINDOW_SAMPLES];
    let mut model_input = vec![0.0f32; VAD_WINDOW_SAMPLES + VAD_CONTEXT_SAMPLES];

    for chunk_start in (0..audio.len()).step_by(VAD_WINDOW_SAMPLES) {
        let chunk_end = (chunk_start + VAD_WINDOW_SAMPLES).min(audio.len());
        let chunk_len = chunk_end - chunk_start;
        chunk[..chunk_len].copy_from_slice(&audio[chunk_start..chunk_end]);

        if chunk_len < VAD_WINDOW_SAMPLES {
            let last = chunk[chunk_len.saturating_sub(1)];
            chunk[chunk_len..].fill(last);
        }

        model_input[..VAD_CONTEXT_SAMPLES].copy_from_slice(&context);
        model_input[VAD_CONTEXT_SAMPLES..].copy_from_slice(&chunk);

        let outputs = model.run(&model_input, &hidden_state, &cell_state)?;
        copy_state_vec(&outputs.hidden_state, &mut hidden_state)?;
        copy_state_vec(&outputs.cell_state, &mut cell_state)?;
        probabilities.push(outputs.probability);
        context.copy_from_slice(&chunk[VAD_WINDOW_SAMPLES - VAD_CONTEXT_SAMPLES..]);
    }

    Ok(probabilities)
}

#[cfg(target_os = "macos")]
fn copy_state_vec(values: &[f32], state: &mut [f32]) -> Result<(), TranscriptionError> {
    if values.len() < state.len() {
        return Err(TranscriptionError::CoreMl(format!(
            "invalid recurrent VAD state length: {}",
            values.len()
        )));
    }
    state.copy_from_slice(&values[..state.len()]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::constants::SAMPLE_RATE;

    #[test]
    fn sample_rate_constant_matches_expected_value() {
        assert_eq!(SAMPLE_RATE, 16_000);
    }
}
