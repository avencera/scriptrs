#[cfg(target_os = "macos")]
use crate::constants::VAD_STATE_SIZE;
#[cfg(any(target_os = "macos", feature = "cuda", test))]
use crate::constants::{VAD_CONTEXT_SAMPLES, VAD_WINDOW_SAMPLES};
#[cfg(target_os = "macos")]
use crate::coreml::SileroVadCoreMlModel;
use crate::error::TranscriptionError;
use crate::long_form::VadModel;
use crate::models::{ModelBundle, RuntimeBundle, SileroVersion};
#[cfg(feature = "cuda")]
use crate::native::silero::{SileroModel, SileroState};
use crate::preprocessing::TranscriptionControl;
use std::path::PathBuf;
use std::sync::mpsc;

/// Silero VAD owner: window chunking, context carry-over, tail padding, and
/// per-recording recurrent state reset
///
/// Holds every Silero model the bundle provides and runs the ones a
/// [`VadModel`] names. Runtimes only classify prepared windows and advance the
/// opaque state
#[derive(Debug, Clone)]
pub(crate) struct SileroVad {
    v5: Slot,
    v6: Slot,
}

/// One Silero version: its model, or where the bundle expected it
#[derive(Debug, Clone)]
enum Slot {
    Loaded(VadModelInner),
    Missing(PathBuf),
}

impl SileroVad {
    /// Load every Silero model the bundle provides; a missing one fails only the
    /// runs that need it
    pub(crate) fn from_bundle(bundle: &ModelBundle) -> Result<Self, TranscriptionError> {
        let slot = |version| {
            let path = bundle.runtime().vad_paths().get(version);
            if !path.exists() {
                return Ok(Slot::Missing(path.to_path_buf()));
            }
            VadModelInner::load(bundle.runtime(), version, path).map(Slot::Loaded)
        };
        Ok(Self {
            v5: slot(SileroVersion::V5)?,
            v6: slot(SileroVersion::V6)?,
        })
    }

    /// Speech probability of each window of `audio` under `model`
    pub(crate) fn process(
        &self,
        audio: &[f32],
        model: VadModel,
        control: &TranscriptionControl,
    ) -> Result<Vec<f32>, TranscriptionError> {
        self.scan(audio, model, control, |_| {})
    }

    /// [`Self::process`], calling `progress` with every probability so far each time
    /// more are ready, so the caller can act on the start of a recording while the
    /// rest is scanned
    pub(crate) fn scan(
        &self,
        audio: &[f32],
        model: VadModel,
        control: &TranscriptionControl,
        mut progress: impl FnMut(&[f32]),
    ) -> Result<Vec<f32>, TranscriptionError> {
        let models = match model {
            VadModel::SileroV5 => vec![self.model(SileroVersion::V5)?],
            VadModel::SileroV6 => vec![self.model(SileroVersion::V6)?],
            VadModel::SileroFusion => vec![
                self.model(SileroVersion::V5)?,
                self.model(SileroVersion::V6)?,
            ],
        };

        // each model's recurrence is one thread's critical path, so running both at
        // once costs little more than running one. Core ML models are `Send` but not
        // `Sync`, so each thread gets its own clone, which shares the weights
        std::thread::scope(|scope| {
            let (sender, receiver) = mpsc::channel();
            let scans: Vec<_> = models
                .into_iter()
                .enumerate()
                .map(|(slot, model)| {
                    let model = model.clone();
                    let sender = sender.clone();
                    scope.spawn(move || {
                        // the receiver outlives every scan, so a send cannot fail
                        model.process(audio, control, |block| {
                            let _ = sender.send((slot, block.to_vec()));
                        })
                    })
                })
                .collect();
            drop(sender);

            // fusion keeps each window's higher probability, once both models have
            // scored it
            let mut scores = vec![Vec::new(); scans.len()];
            let mut probabilities = Vec::new();
            for (slot, block) in receiver {
                scores[slot].extend(block);
                let ready = scores.iter().map(Vec::len).min().unwrap_or(0);
                if ready == probabilities.len() {
                    continue;
                }
                // every model has scored a window below `ready`
                probabilities.extend((probabilities.len()..ready).map(|index| {
                    scores
                        .iter()
                        .map(|model_scores| model_scores[index])
                        .reduce(f32::max)
                        .unwrap_or_default()
                }));
                progress(&probabilities);
            }

            // the first model's error wins when both fail
            for scan in scans {
                scan.join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))?;
            }
            Ok(probabilities)
        })
    }

    fn model(&self, version: SileroVersion) -> Result<&VadModelInner, TranscriptionError> {
        let slot = match version {
            SileroVersion::V5 => &self.v5,
            SileroVersion::V6 => &self.v6,
        };
        match slot {
            Slot::Loaded(model) => Ok(model),
            Slot::Missing(path) => {
                Err(TranscriptionError::MissingModelAsset { path: path.clone() })
            }
        }
    }
}

#[derive(Debug, Clone)]
enum VadModelInner {
    #[cfg(target_os = "macos")]
    CoreMl(SileroVadCoreMlModel),
    #[cfg(feature = "cuda")]
    Native(SileroModel),
    // no VAD runtime without Core ML or the native weights; keeps the matches
    // below compiling
    #[cfg(not(any(target_os = "macos", feature = "cuda")))]
    #[expect(dead_code, reason = "load never builds a VAD model here")]
    Unsupported,
}

impl VadModelInner {
    fn load(
        runtime: &RuntimeBundle,
        version: SileroVersion,
        path: &std::path::Path,
    ) -> Result<Self, TranscriptionError> {
        match runtime {
            RuntimeBundle::CoreMl(_) => {
                // every Silero version takes the same Core ML inputs and outputs
                let _ = version;
                #[cfg(target_os = "macos")]
                {
                    Ok(Self::CoreMl(SileroVadCoreMlModel::new(path)?))
                }
                #[cfg(not(target_os = "macos"))]
                {
                    let _ = path;
                    Err(TranscriptionError::UnsupportedPlatform)
                }
            }
            #[cfg(feature = "cuda")]
            RuntimeBundle::Native(_) => Ok(Self::Native(SileroModel::open(path, version)?)),
        }
    }

    /// Speech probability of each window of `audio`, also passed to `report` in
    /// order as they are scored
    fn process(
        &self,
        audio: &[f32],
        control: &TranscriptionControl,
        report: impl FnMut(&[f32]),
    ) -> Result<Vec<f32>, TranscriptionError> {
        match self {
            #[cfg(target_os = "macos")]
            Self::CoreMl(model) => {
                let mut state = vec![0.0f32; 2 * VAD_STATE_SIZE];
                process_blocks(audio, control, 1, report, |window, probabilities| {
                    let (hidden_state, cell_state) = state.split_at_mut(VAD_STATE_SIZE);
                    let outputs = model.run(window, hidden_state, cell_state)?;
                    copy_state_vec(&outputs.hidden_state, hidden_state)?;
                    copy_state_vec(&outputs.cell_state, cell_state)?;
                    probabilities.push(outputs.probability);
                    Ok(())
                })
            }
            #[cfg(feature = "cuda")]
            Self::Native(model) => {
                let mut state = SileroState::new();
                process_blocks(
                    audio,
                    control,
                    model.block_windows(),
                    report,
                    |windows, probabilities| {
                        model.classify(windows, &mut state, probabilities);
                        Ok(())
                    },
                )
            }
            #[cfg(not(any(target_os = "macos", feature = "cuda")))]
            Self::Unsupported => {
                let _ = (audio, control, report);
                Err(TranscriptionError::UnsupportedPlatform)
            }
        }
    }
}

/// Model input width: the carried context, then one chunk
#[cfg(any(target_os = "macos", feature = "cuda", test))]
const WINDOW_INPUT_SAMPLES: usize = VAD_CONTEXT_SAMPLES + VAD_WINDOW_SAMPLES;

/// Probabilities scored between reports, about 8 s of audio; a Core ML block is one
/// window, and reporting each would cost a channel send per 32 ms of audio
#[cfg(any(target_os = "macos", feature = "cuda", test))]
const REPORT_WINDOWS: usize = 256;

/// Run one classifier over every window of a recording, `block_windows` at a time
///
/// The caller starts the recurrent state at zero for each recording, and
/// `classify` pushes one probability per window, in order. Each window is
/// `VAD_CONTEXT_SAMPLES` from the previous chunk (zeros for the first) followed by
/// one chunk of `VAD_WINDOW_SAMPLES`; a short final chunk is padded with its last
/// sample. `report` gets the new probabilities after any block that brings them to
/// [`REPORT_WINDOWS`], and after the last. Cancellation is checked before each block
#[cfg(any(target_os = "macos", feature = "cuda", test))]
fn process_blocks(
    audio: &[f32],
    control: &TranscriptionControl,
    block_windows: usize,
    mut report: impl FnMut(&[f32]),
    mut classify: impl FnMut(&[f32], &mut Vec<f32>) -> Result<(), TranscriptionError>,
) -> Result<Vec<f32>, TranscriptionError> {
    let count = audio.len().div_ceil(VAD_WINDOW_SAMPLES);
    let block_windows = block_windows.max(1);
    let mut probabilities = Vec::with_capacity(count);
    let mut inputs = vec![0.0f32; block_windows.min(count) * WINDOW_INPUT_SAMPLES];
    let mut reported = 0;

    for first in (0..count).step_by(block_windows) {
        control.check()?;
        let windows = block_windows.min(count - first);
        let inputs = &mut inputs[..windows * WINDOW_INPUT_SAMPLES];
        for (offset, input) in inputs
            .as_chunks_mut::<WINDOW_INPUT_SAMPLES>()
            .0
            .iter_mut()
            .enumerate()
        {
            fill_window(audio, first + offset, input);
        }

        let before = probabilities.len();
        classify(inputs, &mut probabilities)?;
        debug_assert_eq!(probabilities.len(), before + windows);
        if probabilities.len() - reported >= REPORT_WINDOWS || probabilities.len() == count {
            report(&probabilities[reported..]);
            reported = probabilities.len();
        }
    }

    Ok(probabilities)
}

/// Write model window `index` of `audio` into `input`
///
/// Every chunk but the last is full, so the context is always the
/// `VAD_CONTEXT_SAMPLES` of audio just before the chunk
#[cfg(any(target_os = "macos", feature = "cuda", test))]
fn fill_window(audio: &[f32], index: usize, input: &mut [f32]) {
    let start = index * VAD_WINDOW_SAMPLES;
    let (context, chunk) = input.split_at_mut(VAD_CONTEXT_SAMPLES);
    match start.checked_sub(VAD_CONTEXT_SAMPLES) {
        Some(context_start) => context.copy_from_slice(&audio[context_start..start]),
        None => context.fill(0.0),
    }

    let end = (start + VAD_WINDOW_SAMPLES).min(audio.len());
    let len = end - start;
    chunk[..len].copy_from_slice(&audio[start..end]);
    chunk[len..].fill(audio[end - 1]);
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
    use super::{REPORT_WINDOWS, WINDOW_INPUT_SAMPLES, process_blocks};
    use crate::constants::{VAD_CONTEXT_SAMPLES, VAD_WINDOW_SAMPLES};
    use crate::preprocessing::TranscriptionControl;

    fn control() -> TranscriptionControl {
        TranscriptionControl::internal_unbounded()
    }

    /// Every window the classifier sees, in order, with `block_windows` per call
    ///
    /// Also checks the reports: they add up to the probabilities, and every one but
    /// the last holds at least `REPORT_WINDOWS`
    fn windows_seen(audio: &[f32], block_windows: usize) -> Vec<Vec<f32>> {
        let mut seen = Vec::new();
        let mut reports = Vec::new();
        let probabilities = process_blocks(
            audio,
            &control(),
            block_windows,
            |block| reports.push(block.to_vec()),
            |inputs, out| {
                assert!(inputs.len() <= block_windows * WINDOW_INPUT_SAMPLES);
                for input in inputs.as_chunks::<WINDOW_INPUT_SAMPLES>().0 {
                    seen.push(input.to_vec());
                    out.push(seen.len() as f32);
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(probabilities.len(), seen.len());
        assert_eq!(reports.concat(), probabilities);
        if let Some((_, earlier)) = reports.split_last() {
            assert!(earlier.iter().all(|report| report.len() >= REPORT_WINDOWS));
        }
        seen
    }

    #[test]
    fn windows_carry_context_and_pad_the_tail() {
        let mut audio = vec![0.0f32; VAD_WINDOW_SAMPLES + 10];
        audio[VAD_WINDOW_SAMPLES - 1] = 0.5;
        audio[VAD_WINDOW_SAMPLES + 9] = 0.25;

        let seen = windows_seen(&audio, 1);
        assert_eq!(seen.len(), 2);

        // first window has zero context followed by the first chunk
        assert!(seen[0][..VAD_CONTEXT_SAMPLES].iter().all(|v| *v == 0.0));
        assert_eq!(seen[0][VAD_CONTEXT_SAMPLES + VAD_WINDOW_SAMPLES - 1], 0.5);

        // second window starts with the last context samples of the first chunk
        assert_eq!(seen[1][VAD_CONTEXT_SAMPLES - 1], 0.5);

        // short tail is padded with its own last sample
        assert!(
            seen[1][VAD_CONTEXT_SAMPLES + 9..]
                .iter()
                .all(|v| *v == 0.25)
        );
    }

    #[test]
    fn block_size_does_not_change_the_windows() {
        let audio: Vec<f32> = (0..VAD_WINDOW_SAMPLES * 7 + 100)
            .map(|index| (index as f32 * 0.01).sin())
            .collect();

        let one_at_a_time = windows_seen(&audio, 1);
        assert_eq!(one_at_a_time.len(), 8);
        // 3 does not divide 8, so the last block is short
        assert_eq!(windows_seen(&audio, 3), one_at_a_time);
        assert_eq!(windows_seen(&audio, 64), one_at_a_time);
    }

    #[test]
    fn reports_cover_every_window_in_order() {
        let audio = vec![0.25f32; VAD_WINDOW_SAMPLES * (2 * REPORT_WINDOWS + 7)];

        // one window per block reports every `REPORT_WINDOWS`, larger blocks after
        // the block that crosses it
        for block_windows in [1, 100, 3 * REPORT_WINDOWS] {
            windows_seen(&audio, block_windows);
        }
    }

    #[test]
    fn empty_audio_never_invokes_the_classifier() {
        let probabilities = process_blocks(
            &[],
            &control(),
            4,
            |_| panic!("nothing to report"),
            |_, _| panic!("classifier must not run"),
        )
        .unwrap();
        assert!(probabilities.is_empty());
    }

    #[test]
    fn cancellation_is_checked_before_each_block() {
        let audio = vec![0.0f32; VAD_WINDOW_SAMPLES * 6];
        let control = control();
        let mut calls = 0;
        let error = process_blocks(
            &audio,
            &control,
            2,
            |_| {},
            |inputs, out| {
                calls += 1;
                if calls == 1 {
                    control.cancel();
                }
                out.extend(
                    inputs
                        .as_chunks::<WINDOW_INPUT_SAMPLES>()
                        .0
                        .iter()
                        .map(|_| 0.1),
                );
                Ok(())
            },
        )
        .expect_err("cancelled scan");

        assert_eq!(calls, 1);
        assert!(matches!(
            error,
            crate::error::TranscriptionError::Control(
                crate::preprocessing::TranscriptionControlError::Cancelled
            )
        ));
    }
}
