use std::path::Path;

use crate::coreml::{CoreMlInput, CoreMlModel, CoreMlTensor};
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
    CoreMl(CoreMlModel),
    #[cfg(not(target_os = "macos"))]
    Unsupported,
}

impl VadModelInner {
    fn new(model_path: &Path) -> Result<Self, TranscriptionError> {
        #[cfg(target_os = "macos")]
        {
            Ok(Self::CoreMl(CoreMlModel::new(model_path)?))
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
fn process_coreml_vad(model: &CoreMlModel, audio: &[f32]) -> Result<Vec<f32>, TranscriptionError> {
    if audio.is_empty() {
        return Ok(Vec::new());
    }

    let chunk_size = 4096usize;
    let context_size = 64usize;
    let state_size = 128usize;
    let mut probabilities = Vec::with_capacity(audio.len().div_ceil(chunk_size));
    let mut hidden_state = vec![0.0f32; state_size];
    let mut cell_state = vec![0.0f32; state_size];
    let mut context = vec![0.0f32; context_size];

    for chunk_start in (0..audio.len()).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(audio.len());
        let mut chunk = audio[chunk_start..chunk_end].to_vec();
        if chunk.len() < chunk_size {
            let last = chunk.last().copied().unwrap_or(0.0);
            chunk.resize(chunk_size, last);
        }

        let next_context = chunk[chunk.len() - context_size..].to_vec();
        let mut audio_input = Vec::with_capacity(chunk_size + context_size);
        audio_input.extend_from_slice(&context);
        audio_input.extend_from_slice(&chunk);

        let outputs = model.predict(
            &[
                CoreMlInput::F32 {
                    name: "audio_input",
                    values: &audio_input,
                    shape: &[1, chunk_size + context_size],
                },
                CoreMlInput::F32 {
                    name: "hidden_state",
                    values: &hidden_state,
                    shape: &[1, state_size],
                },
                CoreMlInput::F32 {
                    name: "cell_state",
                    values: &cell_state,
                    shape: &[1, state_size],
                },
            ],
            &["vad_output", "new_hidden_state", "new_cell_state"],
        )?;

        let probability = outputs
            .get("vad_output")
            .and_then(|tensor| tensor.data.first())
            .copied()
            .ok_or_else(|| {
                TranscriptionError::CoreMl("VAD output `vad_output` was empty".to_owned())
            })?;
        hidden_state = take_state(outputs.get("new_hidden_state"), state_size)?;
        cell_state = take_state(outputs.get("new_cell_state"), state_size)?;
        probabilities.push(probability);
        context = next_context;
    }

    Ok(probabilities)
}

#[cfg(target_os = "macos")]
fn take_state(
    tensor: Option<&CoreMlTensor>,
    state_size: usize,
) -> Result<Vec<f32>, TranscriptionError> {
    let tensor = tensor.ok_or_else(|| {
        TranscriptionError::CoreMl("missing recurrent VAD state output".to_owned())
    })?;
    if tensor.data.len() < state_size {
        return Err(TranscriptionError::CoreMl(format!(
            "invalid recurrent VAD state length: {}",
            tensor.data.len()
        )));
    }
    Ok(tensor.data[..state_size].to_vec())
}

#[cfg(test)]
mod tests {
    use crate::constants::SAMPLE_RATE;

    #[test]
    fn sample_rate_constant_matches_expected_value() {
        assert_eq!(SAMPLE_RATE, 16_000);
    }
}
