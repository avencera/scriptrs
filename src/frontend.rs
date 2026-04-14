use std::cell::RefCell;
use std::f32::consts::PI;
use std::sync::Arc;

use ndarray::Array2;
use realfft::{RealFftPlanner, RealToComplex, num_complex::Complex};

use crate::config::TranscriptionConfig;
use crate::error::TranscriptionError;

pub(crate) struct ParakeetFeatureExtractor {
    config: TranscriptionConfig,
    mel_filterbank: Array2<f32>,
    window: Vec<f32>,
    r2c: Arc<dyn RealToComplex<f32>>,
    stft_workspace: RefCell<StftWorkspace>,
}

impl std::fmt::Debug for ParakeetFeatureExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParakeetFeatureExtractor")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl Clone for ParakeetFeatureExtractor {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            mel_filterbank: self.mel_filterbank.clone(),
            window: self.window.clone(),
            r2c: self.r2c.clone(),
            stft_workspace: RefCell::new(StftWorkspace::new(&self.r2c)),
        }
    }
}

#[derive(Debug, Clone)]
struct StftWorkspace {
    preemphasized: Vec<f32>,
    padded: Vec<f32>,
    input: Vec<f32>,
    output: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
}

impl StftWorkspace {
    fn new(r2c: &Arc<dyn RealToComplex<f32>>) -> Self {
        Self {
            preemphasized: Vec::new(),
            padded: Vec::new(),
            input: r2c.make_input_vec(),
            output: r2c.make_output_vec(),
            scratch: r2c.make_scratch_vec(),
        }
    }

    fn prepare_padded_audio(&mut self, pad_amount: usize) {
        self.padded.clear();
        self.padded.resize(pad_amount, 0.0);
        self.padded.extend_from_slice(&self.preemphasized);
        self.padded
            .resize(self.preemphasized.len() + pad_amount * 2, 0.0);
    }

    fn process_frame(
        &mut self,
        r2c: &Arc<dyn RealToComplex<f32>>,
        window: &[f32],
        start: usize,
        available: usize,
    ) -> Result<(), TranscriptionError> {
        self.input.fill(0.0);
        for (window_idx, window_value) in window.iter().take(available).copied().enumerate() {
            self.input[window_idx] = self.padded[start + window_idx] * window_value;
        }

        let input = &mut self.input;
        let output = &mut self.output;
        let scratch = &mut self.scratch;
        r2c.process_with_scratch(input, output, scratch)
            .map_err(|error| TranscriptionError::InvalidModelOutput(format!("fft failed: {error}")))
    }
}

impl ParakeetFeatureExtractor {
    pub(crate) fn new(config: &TranscriptionConfig) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(config.n_fft);
        Self {
            config: config.clone(),
            mel_filterbank: create_mel_filterbank(
                config.n_fft,
                config.feature_size,
                config.sample_rate,
            ),
            window: hann_window(config.win_length),
            stft_workspace: RefCell::new(StftWorkspace::new(&r2c)),
            r2c,
        }
    }

    pub(crate) fn extract(&self, audio: &[f32]) -> Result<Array2<f32>, TranscriptionError> {
        if audio.is_empty() {
            return Err(TranscriptionError::EmptyAudio);
        }

        let spectrogram = self.stft(audio)?;
        let mel_spectrogram = self.mel_filterbank.dot(&spectrogram);
        let log_zero_guard = 2.0f32.powi(-24);
        let mut features = mel_spectrogram
            .mapv(|value| (value + log_zero_guard).ln())
            .t()
            .to_owned();
        normalize_columns(&mut features);
        Ok(features)
    }

    fn stft(&self, audio: &[f32]) -> Result<Array2<f32>, TranscriptionError> {
        let mut workspace = self.stft_workspace.borrow_mut();
        apply_preemphasis_into(audio, self.config.preemphasis, &mut workspace.preemphasized);

        let pad_amount = self.config.n_fft / 2;
        workspace.prepare_padded_audio(pad_amount);

        let num_frames = (workspace.padded.len() - self.config.n_fft) / self.config.hop_length + 1;
        let freq_bins = self.config.n_fft / 2 + 1;
        let mut spectrogram = Array2::<f32>::zeros((freq_bins, num_frames));

        // reuse the FFT plan and temporary buffers across chunk extractions
        for frame_idx in 0..num_frames {
            let start = frame_idx * self.config.hop_length;
            let available = self
                .config
                .win_length
                .min(workspace.padded.len().saturating_sub(start));

            workspace.process_frame(&self.r2c, &self.window, start, available)?;

            for bin in 0..freq_bins {
                spectrogram[[bin, frame_idx]] = workspace.output[bin].norm_sqr();
            }
        }

        Ok(spectrogram)
    }
}

fn apply_preemphasis_into(audio: &[f32], coefficient: f32, output: &mut Vec<f32>) {
    output.clear();
    if output.capacity() < audio.len() {
        output.reserve(audio.len() - output.capacity());
    }
    output.push(audio[0]);
    for index in 1..audio.len() {
        output.push(audio[index] - coefficient * audio[index - 1]);
    }
}

fn hann_window(window_length: usize) -> Vec<f32> {
    (0..window_length)
        .map(|index| 0.5 - 0.5 * ((2.0 * PI * index as f32) / (window_length as f32 - 1.0)).cos())
        .collect()
}

const F_SP: f64 = 200.0 / 3.0;
const MIN_LOG_HZ: f64 = 1000.0;
const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
const LOG_STEP: f64 = 0.06875177742094912;

fn hz_to_mel_slaney(hz: f64) -> f64 {
    if hz < MIN_LOG_HZ {
        hz / F_SP
    } else {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / LOG_STEP
    }
}

fn mel_to_hz_slaney(mel: f64) -> f64 {
    if mel < MIN_LOG_MEL {
        mel * F_SP
    } else {
        MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * LOG_STEP).exp()
    }
}

fn create_mel_filterbank(n_fft: usize, n_mels: usize, sample_rate: usize) -> Array2<f32> {
    let freq_bins = n_fft / 2 + 1;
    let mut filterbank = Array2::<f32>::zeros((n_mels, freq_bins));
    let fmax = sample_rate as f64 / 2.0;
    let mel_min = hz_to_mel_slaney(0.0);
    let mel_max = hz_to_mel_slaney(fmax);
    let mel_points: Vec<f64> = (0..=n_mels + 1)
        .map(|index| {
            mel_to_hz_slaney(mel_min + (mel_max - mel_min) * index as f64 / (n_mels + 1) as f64)
        })
        .collect();
    let fft_frequencies: Vec<f64> = (0..freq_bins)
        .map(|index| index as f64 * sample_rate as f64 / n_fft as f64)
        .collect();
    let frequency_differences: Vec<f64> = mel_points
        .windows(2)
        .map(|window| window[1] - window[0])
        .collect();

    for mel_idx in 0..n_mels {
        for (fft_idx, frequency) in fft_frequencies.iter().copied().enumerate() {
            let lower = (frequency - mel_points[mel_idx]) / frequency_differences[mel_idx];
            let upper = (mel_points[mel_idx + 2] - frequency) / frequency_differences[mel_idx + 1];
            filterbank[[mel_idx, fft_idx]] = 0.0f64.max(lower.min(upper)) as f32;
        }
    }

    for mel_idx in 0..n_mels {
        let enorm = 2.0 / (mel_points[mel_idx + 2] - mel_points[mel_idx]);
        for fft_idx in 0..freq_bins {
            filterbank[[mel_idx, fft_idx]] *= enorm as f32;
        }
    }

    filterbank
}

fn normalize_columns(features: &mut Array2<f32>) {
    let num_frames = features.shape()[0];
    let num_features = features.shape()[1];
    for feature_idx in 0..num_features {
        let mut column = features.column_mut(feature_idx);
        let mean = column.iter().sum::<f32>() / num_frames as f32;
        let variance = column
            .iter()
            .map(|value| (*value - mean).powi(2))
            .sum::<f32>()
            / (num_frames as f32 - 1.0);
        let std = variance.sqrt() + 1e-5;
        for value in &mut column {
            *value = (*value - mean) / std;
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::{ParakeetFeatureExtractor, TranscriptionConfig};

    fn sine_wave(frequency_hz: f32, sample_rate: usize, num_samples: usize) -> Vec<f32> {
        (0..num_samples)
            .map(|index| {
                (2.0 * std::f32::consts::PI * frequency_hz * index as f32 / sample_rate as f32)
                    .sin()
            })
            .collect()
    }

    #[test]
    fn stft_concentrates_power_at_the_expected_bin() {
        let sample_rate = 16_000;
        let extractor = ParakeetFeatureExtractor::new(&TranscriptionConfig::default());
        let spectrum = extractor
            .stft(&sine_wave(1000.0, sample_rate, sample_rate))
            .unwrap();
        let expected_bin = 32;
        let mut correct_frames = 0;
        for frame in 2..spectrum.shape()[1].saturating_sub(2) {
            let mut max_bin = 0;
            let mut max_power = 0.0f32;
            for bin in 0..spectrum.shape()[0] {
                if spectrum[[bin, frame]] > max_power {
                    max_power = spectrum[[bin, frame]];
                    max_bin = bin;
                }
            }
            if max_bin == expected_bin {
                correct_frames += 1;
            }
        }
        assert!(correct_frames > 90);
    }

    #[test]
    fn feature_extraction_produces_expected_shape() {
        let extractor = ParakeetFeatureExtractor::new(&TranscriptionConfig::default());
        let audio = sine_wave(440.0, 16_000, 16_000);
        let features = extractor.extract(&audio).unwrap();
        assert_eq!(features.shape()[1], 128);
        assert!(features.shape()[0] > 0);
    }

    #[test]
    fn features_are_column_normalized() {
        let extractor = ParakeetFeatureExtractor::new(&TranscriptionConfig::default());
        let audio = sine_wave(440.0, 16_000, 16_000);
        let features = extractor.extract(&audio).unwrap();
        let column = features.column(0);
        let mean = column.iter().sum::<f32>() / column.len() as f32;
        assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-4);
    }
}
