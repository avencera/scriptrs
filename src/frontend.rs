use std::f32::consts::PI;

use ndarray::Array2;
use realfft::RealFftPlanner;

use crate::config::TranscriptionConfig;
use crate::error::TranscriptionError;

#[derive(Debug, Clone)]
pub(crate) struct ParakeetFeatureExtractor {
    config: TranscriptionConfig,
    mel_filterbank: Array2<f32>,
}

impl ParakeetFeatureExtractor {
    pub(crate) fn new(config: &TranscriptionConfig) -> Self {
        Self {
            config: config.clone(),
            mel_filterbank: create_mel_filterbank(
                config.n_fft,
                config.feature_size,
                config.sample_rate,
            ),
        }
    }

    pub(crate) fn extract(&self, audio: &[f32]) -> Result<Array2<f32>, TranscriptionError> {
        if audio.is_empty() {
            return Err(TranscriptionError::EmptyAudio);
        }

        let preemphasized = apply_preemphasis(audio, self.config.preemphasis);
        let spectrogram = stft(
            &preemphasized,
            self.config.n_fft,
            self.config.hop_length,
            self.config.win_length,
        )?;
        let mut features = self.mel_filterbank.dot(&spectrogram);
        let log_zero_guard = 2.0f32.powi(-24);
        features.mapv_inplace(|value| (value + log_zero_guard).ln());
        normalize_features(&mut features);
        Ok(features)
    }
}

fn apply_preemphasis(audio: &[f32], coefficient: f32) -> Vec<f32> {
    let mut output = Vec::with_capacity(audio.len());
    output.push(audio[0]);
    for index in 1..audio.len() {
        output.push(audio[index] - coefficient * audio[index - 1]);
    }
    output
}

fn stft(
    audio: &[f32],
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
) -> Result<Array2<f32>, TranscriptionError> {
    let pad_amount = n_fft / 2;
    let mut padded = vec![0.0f32; pad_amount];
    padded.extend_from_slice(audio);
    padded.resize(padded.len() + pad_amount, 0.0);

    let window = hann_window(win_length);
    let num_frames = (padded.len() - n_fft) / hop_length + 1;
    let freq_bins = n_fft / 2 + 1;
    let mut spectrogram = Array2::<f32>::zeros((freq_bins, num_frames));

    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(n_fft);
    let mut input = vec![0.0f32; n_fft];
    let mut output = r2c.make_output_vec();
    let mut scratch = r2c.make_scratch_vec();

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_length;
        input.fill(0.0);
        for window_idx in 0..win_length.min(padded.len() - start) {
            input[window_idx] = padded[start + window_idx] * window[window_idx];
        }

        r2c.process_with_scratch(&mut input, &mut output, &mut scratch)
            .map_err(|error| {
                TranscriptionError::InvalidModelOutput(format!("fft failed: {error}"))
            })?;

        for bin in 0..freq_bins {
            spectrogram[[bin, frame_idx]] = output[bin].norm_sqr();
        }
    }

    Ok(spectrogram)
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

fn normalize_features(features: &mut Array2<f32>) {
    let num_frames = features.shape()[1];
    let num_features = features.shape()[0];
    for feature_idx in 0..num_features {
        let mut row = features.row_mut(feature_idx);
        let mean = row.iter().sum::<f32>() / num_frames as f32;
        let variance = row.iter().map(|value| (*value - mean).powi(2)).sum::<f32>()
            / (num_frames as f32 - 1.0);
        let std = variance.sqrt() + 1e-5;
        for value in &mut row {
            *value = (*value - mean) / std;
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::{ParakeetFeatureExtractor, TranscriptionConfig, stft};

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
        let spectrum = stft(&sine_wave(1000.0, sample_rate, sample_rate), 512, 160, 400).unwrap();
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
        assert_eq!(features.shape()[0], 128);
        assert!(features.shape()[1] > 0);
    }

    #[test]
    fn features_are_column_normalized() {
        let extractor = ParakeetFeatureExtractor::new(&TranscriptionConfig::default());
        let audio = sine_wave(440.0, 16_000, 16_000);
        let features = extractor.extract(&audio).unwrap();
        let feature = features.row(0);
        let mean = feature.iter().sum::<f32>() / feature.len() as f32;
        assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-4);
    }
}
