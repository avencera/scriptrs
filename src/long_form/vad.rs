use crate::constants::{VAD_CONTEXT_SAMPLES, VAD_STATE_SIZE, VAD_WINDOW_SAMPLES};
#[cfg(target_os = "macos")]
use crate::coreml::SileroVadCoreMlModel;
use crate::error::TranscriptionError;
use crate::long_form::planner::VadConfig;
use crate::models::{ModelBundle, RuntimeBundle};
use crate::preprocessing::TranscriptionControl;

/// Silero VAD owner: window chunking, context carry-over, tail padding, and
/// per-recording recurrent state reset
///
/// Runtimes only classify one prepared window and advance the opaque state.
#[derive(Debug, Clone)]
pub(crate) struct SileroVad {
    model: VadModelInner,
}

impl SileroVad {
    /// Load the VAD model a bundle provides, if it provides one
    pub(crate) fn from_bundle(bundle: &ModelBundle) -> Result<Option<Self>, TranscriptionError> {
        VadModelInner::from_bundle(bundle).map(|model| model.map(|model| Self { model }))
    }

    pub(crate) fn process(
        &self,
        audio: &[f32],
        _config: &VadConfig,
        control: &TranscriptionControl,
    ) -> Result<Vec<f32>, TranscriptionError> {
        self.model.process(audio, control)
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
    fn from_bundle(bundle: &ModelBundle) -> Result<Option<Self>, TranscriptionError> {
        match bundle.runtime() {
            RuntimeBundle::CoreMl(layout) => {
                if !layout.vad_dir().exists() {
                    return Ok(None);
                }

                #[cfg(target_os = "macos")]
                {
                    Ok(Some(Self::CoreMl(SileroVadCoreMlModel::new(
                        layout.vad_dir(),
                    )?)))
                }
                #[cfg(not(target_os = "macos"))]
                {
                    Err(TranscriptionError::UnsupportedPlatform)
                }
            }
        }
    }

    fn process(
        &self,
        audio: &[f32],
        control: &TranscriptionControl,
    ) -> Result<Vec<f32>, TranscriptionError> {
        match self {
            #[cfg(target_os = "macos")]
            Self::CoreMl(model) => process_windows(audio, control, |window, state| {
                let (hidden_state, cell_state) = state.split_at_mut(VAD_STATE_SIZE);
                let outputs = model.run(window, hidden_state, cell_state)?;
                copy_state_vec(&outputs.hidden_state, hidden_state)?;
                copy_state_vec(&outputs.cell_state, cell_state)?;
                Ok(outputs.probability)
            }),
            #[cfg(not(target_os = "macos"))]
            Self::Unsupported => {
                let _ = audio;
                let _ = control;
                Err(TranscriptionError::UnsupportedPlatform)
            }
        }
    }
}

/// Run one classifier over every window of a recording
///
/// The recurrent state starts at zero for each recording. Each window is
/// `VAD_CONTEXT_SAMPLES` from the previous chunk followed by one chunk of
/// `VAD_WINDOW_SAMPLES`; a short final chunk is padded with its last sample.
#[cfg(target_os = "macos")]
fn process_windows(
    audio: &[f32],
    control: &TranscriptionControl,
    mut classify: impl FnMut(&[f32], &mut [f32]) -> Result<f32, TranscriptionError>,
) -> Result<Vec<f32>, TranscriptionError> {
    if audio.is_empty() {
        return Ok(Vec::new());
    }

    let mut probabilities = Vec::with_capacity(audio.len().div_ceil(VAD_WINDOW_SAMPLES));
    let mut state = vec![0.0f32; 2 * VAD_STATE_SIZE];
    let mut context = vec![0.0f32; VAD_CONTEXT_SAMPLES];
    let mut chunk = vec![0.0f32; VAD_WINDOW_SAMPLES];
    let mut model_input = vec![0.0f32; VAD_WINDOW_SAMPLES + VAD_CONTEXT_SAMPLES];

    for chunk_start in (0..audio.len()).step_by(VAD_WINDOW_SAMPLES) {
        control.check()?;
        let chunk_end = (chunk_start + VAD_WINDOW_SAMPLES).min(audio.len());
        let chunk_len = chunk_end - chunk_start;
        chunk[..chunk_len].copy_from_slice(&audio[chunk_start..chunk_end]);

        if chunk_len < VAD_WINDOW_SAMPLES {
            let last = chunk[chunk_len.saturating_sub(1)];
            chunk[chunk_len..].fill(last);
        }

        model_input[..VAD_CONTEXT_SAMPLES].copy_from_slice(&context);
        model_input[VAD_CONTEXT_SAMPLES..].copy_from_slice(&chunk);

        probabilities.push(classify(&model_input, &mut state)?);
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

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::process_windows;
    use crate::constants::{VAD_CONTEXT_SAMPLES, VAD_STATE_SIZE, VAD_WINDOW_SAMPLES};
    use crate::preprocessing::TranscriptionControl;

    fn control() -> TranscriptionControl {
        TranscriptionControl::internal_unbounded()
    }

    #[test]
    fn windows_carry_context_pad_the_tail_and_start_from_zero_state() {
        let mut audio = vec![0.0f32; VAD_WINDOW_SAMPLES + 10];
        audio[VAD_WINDOW_SAMPLES - 1] = 0.5;
        audio[VAD_WINDOW_SAMPLES + 9] = 0.25;

        let mut seen = Vec::new();
        let probabilities = process_windows(&audio, &control(), |window, state| {
            seen.push((window.to_vec(), state.to_vec()));
            state[0] += 1.0;
            Ok(state[0] / 10.0)
        })
        .unwrap();

        assert_eq!(probabilities, vec![0.1, 0.2]);
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[0].1, vec![0.0; 2 * VAD_STATE_SIZE]);
        assert_eq!(seen[1].1[0], 1.0);

        // first window has zero context followed by the first chunk
        assert!(seen[0].0[..VAD_CONTEXT_SAMPLES].iter().all(|v| *v == 0.0));
        assert_eq!(seen[0].0[VAD_CONTEXT_SAMPLES + VAD_WINDOW_SAMPLES - 1], 0.5);

        // second window starts with the last context samples of the first chunk
        assert_eq!(seen[1].0[VAD_CONTEXT_SAMPLES - 1], 0.5);

        // short tail is padded with its own last sample
        assert!(
            seen[1].0[VAD_CONTEXT_SAMPLES + 9..]
                .iter()
                .all(|v| *v == 0.25)
        );
    }

    #[test]
    fn empty_audio_never_invokes_the_classifier() {
        let probabilities =
            process_windows(&[], &control(), |_, _| panic!("classifier must not run")).unwrap();
        assert!(probabilities.is_empty());
    }

    #[test]
    fn cancellation_is_checked_before_each_classifier_call() {
        let audio = vec![0.0f32; VAD_WINDOW_SAMPLES * 3];
        let control = control();
        let mut calls = 0;
        let error = process_windows(&audio, &control, |_, _| {
            calls += 1;
            if calls == 1 {
                control.cancel();
            }
            Ok(0.1)
        })
        .expect_err("cancelled scan");

        assert_eq!(calls, 1);
        assert!(matches!(error, crate::error::TranscriptionError::Cancelled));
    }
}
