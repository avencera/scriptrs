use std::cell::RefCell;
use std::f32::consts::PI;
use std::sync::Arc;

use ndarray::Array2;
use realfft::{RealFftPlanner, RealToComplex, num_complex::Complex};

use crate::config::TranscriptionConfig;
use crate::error::TranscriptionError;

pub(crate) struct ParakeetFeatureExtractor {
    config: TranscriptionConfig,
    mel_filterbank: MelFilterbank,
    window: Vec<f32>,
    r2c: Arc<dyn RealToComplex<f32>>,
    workspace: RefCell<Workspace>,
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
            workspace: RefCell::new(Workspace::new(&self.r2c)),
        }
    }
}

/// Buffers reused across extractions
struct Workspace {
    /// The pre-emphasized audio with the STFT's centering pad
    padded: Vec<f32>,
    input: Vec<f32>,
    output: Vec<Complex<f32>>,
    fft: Vec<Complex<f32>>,
    /// The current frame's power spectrum, used while it is still in cache
    power: Vec<f32>,
}

impl Workspace {
    fn new(r2c: &Arc<dyn RealToComplex<f32>>) -> Self {
        Self {
            padded: Vec::new(),
            input: r2c.make_input_vec(),
            output: r2c.make_output_vec(),
            fft: r2c.make_scratch_vec(),
            power: Vec::new(),
        }
    }
}

impl ParakeetFeatureExtractor {
    pub(crate) fn new(config: &TranscriptionConfig) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(config.n_fft);
        Self {
            config: config.clone(),
            mel_filterbank: MelFilterbank::new(&create_mel_filterbank(
                config.n_fft,
                config.feature_size,
                config.sample_rate,
            )),
            window: hann_window(config.win_length),
            workspace: RefCell::new(Workspace::new(&r2c)),
            r2c,
        }
    }

    pub(crate) fn extract(&self, audio: &[f32]) -> Result<Array2<f32>, TranscriptionError> {
        if audio.is_empty() {
            return Err(TranscriptionError::EmptyAudio);
        }

        let mut workspace = self.workspace.borrow_mut();
        preemphasize_padded(
            audio,
            self.config.preemphasis,
            self.config.n_fft / 2,
            &mut workspace.padded,
        );

        // NeMo counts `len / hop` valid frames; the STFT's extra trailing frame is
        // padding that NeMo masks out and leaves out of normalization. One frame
        // is kept for audio shorter than a hop so the encoder has an input
        let num_frames = (audio.len() / self.config.hop_length).max(1);
        let mut features = Array2::<f32>::zeros((num_frames, self.mel_filterbank.mels()));
        let log_zero_guard = 2.0f32.powi(-24);
        for (frame, mut mels) in features.rows_mut().into_iter().enumerate() {
            self.power(&mut workspace, frame * self.config.hop_length)?;
            let mels = mels
                .as_slice_mut()
                .expect("a fresh standard-layout array has contiguous rows");
            self.mel_filterbank.apply(&workspace.power, mels);
            for value in mels {
                *value = (*value + log_zero_guard).ln();
            }
        }
        normalize_columns(&mut features);
        Ok(features)
    }

    /// Fills `workspace.power` with the power spectrum of the frame at `start` in
    /// the padded audio
    fn power(&self, workspace: &mut Workspace, start: usize) -> Result<(), TranscriptionError> {
        let Workspace {
            padded,
            input,
            output,
            fft,
            power,
        } = workspace;
        // torch.stft centers a window shorter than n_fft inside the frame
        let offset = (input.len() - self.window.len()) / 2;
        input.fill(0.0);
        let frame = &padded[start + offset..start + offset + self.window.len()];
        for ((slot, sample), weight) in input[offset..].iter_mut().zip(frame).zip(&self.window) {
            *slot = sample * weight;
        }

        self.r2c
            .process_with_scratch(input, output, fft)
            .map_err(|error| {
                TranscriptionError::InvalidModelOutput(format!("fft failed: {error}"))
            })?;
        // a real FFT of `n_fft` samples gives exactly the `n_fft / 2 + 1` bins
        power.clear();
        power.extend(output.iter().map(Complex::norm_sqr));
        Ok(())
    }

    /// The power spectrum as `[freq_bins, frames]`
    #[cfg(test)]
    fn stft(&self, audio: &[f32]) -> Result<Array2<f32>, TranscriptionError> {
        let mut workspace = self.workspace.borrow_mut();
        preemphasize_padded(
            audio,
            self.config.preemphasis,
            self.config.n_fft / 2,
            &mut workspace.padded,
        );
        let num_frames = (audio.len() / self.config.hop_length).max(1);
        let freq_bins = self.config.n_fft / 2 + 1;
        let mut power = Array2::zeros((freq_bins, num_frames));
        for (frame, mut column) in power.columns_mut().into_iter().enumerate() {
            self.power(&mut workspace, frame * self.config.hop_length)?;
            column.assign(&ndarray::ArrayView1::from(&workspace.power));
        }
        Ok(power)
    }
}

/// `audio` after pre-emphasis, with `pad` zeros on each side as torch.stft's
/// centered frames read it, in one pass
fn preemphasize_padded(audio: &[f32], coefficient: f32, pad: usize, padded: &mut Vec<f32>) {
    padded.clear();
    padded.resize(audio.len() + 2 * pad, 0.0);
    let body = &mut padded[pad..pad + audio.len()];
    body[0] = audio[0];
    for ((out, current), previous) in body[1..].iter_mut().zip(&audio[1..]).zip(audio) {
        *out = current - coefficient * previous;
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

/// Slaney mel filters kept as the band of bins each one weights
///
/// Each triangular filter covers a few of the 257 bins, so the dense product did 65x
/// the needed work. A filter's band is accumulated in bin order with a fused
/// multiply-add, which is what matrixmultiply's FMA kernel does for each output, so
/// the features stay bit-identical to the dense `filterbank · spectrum` on CPUs with
/// FMA: the skipped zero weights add exactly nothing to a non-negative power sum
#[derive(Debug, Clone)]
struct MelFilterbank {
    /// First weighted bin and the weights from it, per mel
    filters: Vec<(usize, Vec<f32>)>,
}

impl MelFilterbank {
    /// Trims each `[n_mels, freq_bins]` row to its nonzero band
    fn new(dense: &Array2<f32>) -> Self {
        let filters = dense
            .rows()
            .into_iter()
            .map(|row| {
                let first = row.iter().position(|weight| *weight != 0.0);
                let last = row.iter().rposition(|weight| *weight != 0.0);
                match (first, last) {
                    (Some(first), Some(last)) => (
                        first,
                        row.iter()
                            .skip(first)
                            .take(last + 1 - first)
                            .copied()
                            .collect(),
                    ),
                    // an empty filter sums to zero, as its dense row did
                    _ => (0, Vec::new()),
                }
            })
            .collect();
        Self { filters }
    }

    fn mels(&self) -> usize {
        self.filters.len()
    }

    /// Writes the mel energies of one frame's power spectrum into `out`
    ///
    /// Each energy rounds exactly as the dense `matrixmultiply` product this
    /// replaced: its kernels add bins in order, fused on x86-64 with FMA and on
    /// aarch64, and as a separate multiply and add elsewhere
    fn apply(&self, power: &[f32], out: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("fma") {
            // SAFETY: the CPU supports FMA, checked just above
            unsafe { self.apply_fma(power, out) };
            return;
        }

        #[cfg(target_arch = "aarch64")]
        self.apply_with(power, out, f32::mul_add);
        #[cfg(not(target_arch = "aarch64"))]
        self.apply_with(power, out, |weight, bin, sum| sum + weight * bin);
    }

    /// # Safety
    ///
    /// The CPU must support FMA. The function is `unsafe` rather than a safe
    /// `#[target_feature]` function, which needs Rust 1.86
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "fma")]
    unsafe fn apply_fma(&self, power: &[f32], out: &mut [f32]) {
        // `mul_add` is one instruction here; without the feature it would call a
        // slow software fallback
        self.apply_with(power, out, f32::mul_add);
    }

    /// Folds each filter over its bins in order with `step(weight, bin, sum)`
    #[inline(always)]
    fn apply_with(&self, power: &[f32], out: &mut [f32], step: impl Fn(f32, f32, f32) -> f32) {
        for ((first, weights), slot) in self.filters.iter().zip(out) {
            let bins = &power[*first..*first + weights.len()];
            *slot = weights
                .iter()
                .zip(bins)
                .fold(0.0, |sum, (weight, bin)| step(*weight, *bin, sum));
        }
    }
}

/// Normalizes each feature column of `[frames, features]` to zero mean and unit
/// (unbiased) deviation
///
/// Sums run row by row into one accumulator per column, which adds each column's
/// values in the same frame order as a column-wise pass, without striding through
/// memory
fn normalize_columns(features: &mut Array2<f32>) {
    let (num_frames, num_features) = features.dim();
    // plain slices rather than ndarray rows let the compiler vectorize each pass
    let values = features
        .as_slice_mut()
        .expect("a fresh standard-layout array is contiguous");
    let mut sums = vec![0.0f32; num_features];
    for row in values.chunks_exact(num_features) {
        for (sum, value) in sums.iter_mut().zip(row) {
            *sum += *value;
        }
    }
    let means: Vec<f32> = sums.iter().map(|sum| sum / num_frames as f32).collect();

    let mut squares = vec![0.0f32; num_features];
    for row in values.chunks_exact(num_features) {
        for ((sum, value), mean) in squares.iter_mut().zip(row).zip(&means) {
            *sum += (*value - mean).powi(2);
        }
    }
    // unbiased, as NeMo's per-feature normalization; a single frame has zero spread
    let stds: Vec<f32> = squares
        .iter()
        .map(|sum| (sum / num_frames.saturating_sub(1).max(1) as f32).sqrt() + 1e-5)
        .collect();

    for row in values.chunks_exact_mut(num_features) {
        for ((value, mean), std) in row.iter_mut().zip(&means).zip(&stds) {
            *value = (*value - mean) / std;
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::{
        Array2, MelFilterbank, ParakeetFeatureExtractor, TranscriptionConfig, create_mel_filterbank,
    };

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
    fn sparse_filterbank_matches_the_dense_product() {
        let config = TranscriptionConfig::default();
        let dense = create_mel_filterbank(config.n_fft, config.feature_size, config.sample_rate);
        let filterbank = MelFilterbank::new(&dense);
        // non-negative like a power spectrum and different in every bin, with frames
        // spread over several orders of magnitude so rounding differs between them
        let frames = 64;
        let power = Array2::from_shape_fn((dense.ncols(), frames), |(bin, frame)| {
            let scale = 10f32.powi(frame as i32 % 8 - 4);
            scale * (1.0 + (bin as f32 * 0.37 + frame as f32).sin().abs() * bin as f32)
        });
        // the dense product the sparse filterbank replaced, on this CPU's kernels
        let expected = dense.dot(&power);

        assert_eq!(filterbank.mels(), dense.nrows());
        let mut mels = vec![0.0f32; filterbank.mels()];
        for (frame, spectrum) in power.columns().into_iter().enumerate() {
            let spectrum: Vec<f32> = spectrum.to_vec();
            filterbank.apply(&spectrum, &mut mels);
            for (mel, sparse) in mels.iter().enumerate() {
                let expected = expected[[mel, frame]];
                assert!(expected > 0.0, "mel {mel} weights no bin");
                assert_eq!(
                    sparse.to_bits(),
                    expected.to_bits(),
                    "mel {mel} frame {frame}: {sparse} vs {expected}"
                );
            }
        }
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

    /// Frontend output against NeMo's `mel` for the same audio, over NeMo's valid frames
    fn assert_matches_nemo_mel(version: crate::deployment::ParakeetVersion) {
        use crate::deployment::ParakeetDeployment;
        use crate::test_support::NemoReference;

        let reference = NemoReference::open(version, "b1");
        let audio = reference.f32("audio");
        let audio_len = reference.i64("audio_len")[0] as usize;
        let mel = reference.f32("mel");
        let mel_len = reference.i64("mel_len")[0] as usize;

        let deployment = match version {
            crate::deployment::ParakeetVersion::V2 => ParakeetDeployment::v2(),
            crate::deployment::ParakeetVersion::V3 => ParakeetDeployment::v3(),
        };
        let extractor =
            ParakeetFeatureExtractor::new(&TranscriptionConfig::from_deployment(&deployment));
        let samples: Vec<f32> = audio.iter().take(audio_len).copied().collect();
        let features = extractor.extract(&samples).unwrap();

        // NeMo's tensor carries one padded frame past `mel_len`; the frontend must stop at it
        assert_eq!(features.nrows(), mel_len, "frame count");
        assert_eq!(features.ncols(), mel.shape()[1], "mel features");
        let mut max_abs = 0.0f32;
        for frame in 0..mel_len {
            for bin in 0..features.ncols() {
                let value = features[[frame, bin]];
                assert!(value.is_finite(), "frame {frame} bin {bin} is {value}");
                max_abs = max_abs.max((value - mel[[0, bin, frame]]).abs());
            }
        }
        assert!(
            max_abs < 1e-4,
            "max abs mel difference {max_abs} over {mel_len} frames"
        );
    }

    #[test]
    #[ignore = "reads NeMo references from /shared/datasets/parakeet-ref"]
    fn nemo_parity_v2_mel() {
        assert_matches_nemo_mel(crate::deployment::ParakeetVersion::V2);
    }

    #[test]
    #[ignore = "reads NeMo references from /shared/datasets/parakeet-ref"]
    fn nemo_parity_v3_mel() {
        assert_matches_nemo_mel(crate::deployment::ParakeetVersion::V3);
    }
}
