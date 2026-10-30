//! Log-mel features computed as audio arrives, for the cache-aware streaming models
//!
//! NeMo's streaming loop computes the mel of a whole utterance once and slices it
//! into chunks. A frame's centered window reaches 200 samples past its hop, so a
//! frame is final once that much audio has arrived; frames near the end read the
//! STFT's zero padding and wait for [`StreamingFeatures::finish`]. The models skip
//! per-utterance normalization, so a frame never changes once emitted

use std::sync::Arc;

use realfft::{RealFftPlanner, RealToComplex, num_complex::Complex};

use super::{MelFilterbank, create_mel_filterbank, forward_fft, hann_window};
use crate::error::TranscriptionError;
use crate::native::spec::FrontendConfig;

/// The STFT plan, window and filterbank every stream of a model shares
pub(crate) struct MelSetup {
    config: FrontendConfig,
    filterbank: MelFilterbank,
    window: Vec<f32>,
    r2c: Arc<dyn RealToComplex<f32>>,
}

impl std::fmt::Debug for MelSetup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MelSetup")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl MelSetup {
    /// The featurizer `config` describes with `features` mel bins
    pub(crate) fn new(config: FrontendConfig, features: usize) -> Arc<Self> {
        let r2c = RealFftPlanner::<f32>::new().plan_fft_forward(config.n_fft);
        Arc::new(Self {
            config,
            filterbank: MelFilterbank::new(&create_mel_filterbank(
                config.n_fft,
                features,
                config.sample_rate,
            )),
            window: hann_window(config.win_length),
            r2c,
        })
    }

    /// Mel bins per frame
    pub(crate) fn features(&self) -> usize {
        self.filterbank.mels()
    }

    /// Audio samples per frame
    pub(crate) fn hop_length(&self) -> usize {
        self.config.hop_length
    }

    /// First sample frame `index`'s window reads, which may be before the audio
    fn window_start(&self, index: usize) -> isize {
        let config = &self.config;
        // torch.stft centers the frame on its hop and the window inside the frame
        (index * config.hop_length) as isize - (config.win_length / 2) as isize
    }
}

/// One stream's featurizer: pre-emphasized audio not yet consumed, and the frames
/// emitted so far
pub(crate) struct StreamingFeatures {
    setup: Arc<MelSetup>,
    /// Pre-emphasized samples from absolute index `base` on
    samples: Vec<f32>,
    base: usize,
    /// The last raw sample, which pre-emphasis subtracts from the next
    previous: Option<f32>,
    /// Raw samples received
    received: usize,
    /// Frames emitted
    frames: usize,
    finished: bool,
    input: Vec<f32>,
    output: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    power: Vec<f32>,
}

impl StreamingFeatures {
    pub(crate) fn new(setup: Arc<MelSetup>) -> Self {
        let r2c = setup.r2c.clone();
        Self {
            samples: Vec::new(),
            base: 0,
            previous: None,
            received: 0,
            frames: 0,
            finished: false,
            input: r2c.make_input_vec(),
            output: r2c.make_output_vec(),
            scratch: r2c.make_scratch_vec(),
            power: vec![0.0; setup.config.n_fft / 2 + 1],
            setup,
        }
    }

    /// Frames emitted so far
    pub(crate) fn frames(&self) -> usize {
        self.frames
    }

    /// Appends `audio` and writes every frame it completes to `out`, time-major
    pub(crate) fn push(
        &mut self,
        audio: &[f32],
        out: &mut Vec<f32>,
    ) -> Result<(), TranscriptionError> {
        if self.finished {
            return Err(TranscriptionError::InvalidConfiguration(
                "audio pushed to a finished stream".to_owned(),
            ));
        }
        let coefficient = self.setup.config.preemphasis;
        self.samples.reserve(audio.len());
        for sample in audio {
            // NeMo keeps the first sample as is
            let emphasized = match self.previous {
                Some(previous) => sample - coefficient * previous,
                None => *sample,
            };
            self.samples.push(emphasized);
            self.previous = Some(*sample);
        }
        self.received += audio.len();

        let win = self.setup.config.win_length as isize;
        while self.setup.window_start(self.frames) + win <= self.received as isize {
            self.emit_frame(out)?;
        }
        self.drop_consumed();
        Ok(())
    }

    /// Writes the frames that read past the end of the audio, then the zero frame
    /// NeMo's streaming buffer carries after the last valid one
    ///
    /// NeMo counts `len / hop` valid frames; its STFT yields one more, which the
    /// featurizer masks to zero and the streaming buffer still feeds the encoder
    pub(crate) fn finish(&mut self, out: &mut Vec<f32>) -> Result<(), TranscriptionError> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;
        let valid = self.received / self.setup.config.hop_length;
        while self.frames < valid {
            self.emit_frame(out)?;
        }
        out.resize(out.len() + self.setup.features(), 0.0);
        self.frames += 1;
        Ok(())
    }

    /// Computes frame `self.frames` from the samples held, zero outside the audio
    fn emit_frame(&mut self, out: &mut Vec<f32>) -> Result<(), TranscriptionError> {
        let setup = &*self.setup;
        let n_fft = setup.config.n_fft;
        let offset = (n_fft - setup.window.len()) / 2;
        let start = setup.window_start(self.frames);
        self.input.fill(0.0);
        for (k, weight) in setup.window.iter().enumerate() {
            let index = start + k as isize;
            if index >= self.base as isize && ((index as usize) < self.received) {
                self.input[offset + k] = self.samples[index as usize - self.base] * weight;
            }
        }
        forward_fft(
            &setup.r2c,
            &mut self.input,
            &mut self.output,
            &mut self.scratch,
        )?;
        for (power, bin) in self.power.iter_mut().zip(&self.output) {
            *power = bin.norm_sqr();
        }

        let first = out.len();
        out.resize(first + setup.features(), 0.0);
        let mels = &mut out[first..];
        setup.filterbank.apply_log(&self.power, mels);
        self.frames += 1;
        Ok(())
    }

    /// Drops the samples before the next frame's window
    fn drop_consumed(&mut self) {
        let keep_from = self.setup.window_start(self.frames).max(0) as usize;
        if keep_from > self.base {
            let consumed = (keep_from - self.base).min(self.samples.len());
            self.samples.drain(..consumed);
            self.base += consumed;
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Ix2;

    use super::{MelSetup, StreamingFeatures};
    use crate::native::spec::FrontendConfig;
    use crate::test_support::NemoReference;

    /// Nemotron Speech Streaming's featurizer
    const CONFIG: FrontendConfig = FrontendConfig {
        sample_rate: 16_000,
        n_fft: 512,
        win_length: 400,
        hop_length: 160,
        preemphasis: 0.97,
    };

    /// Features streamed in uneven pushes against the mel NeMo's streaming buffer
    /// sliced into chunks: same frame count, the trailing zero frame, same values
    #[test]
    #[ignore = "reads NeMo streaming references from /shared/datasets/nemotron-stream-ref"]
    fn nemo_parity_streaming_mel() {
        let reference = NemoReference::streaming("nemotron-en", 13);
        let setup = MelSetup::new(CONFIG, 128);
        for utterance in 0..2 {
            let audio = reference.f32(&format!("utt{utterance}.audio"));
            let audio = audio.as_slice().expect("contiguous audio");
            let mel = reference
                .f32(&format!("utt{utterance}.mel"))
                .into_dimensionality::<Ix2>()
                .expect("[features, frames]");

            let mut features = StreamingFeatures::new(setup.clone());
            let mut out = Vec::new();
            // pushes of 1 to 1601 samples, so frames complete at every offset
            let mut start = 0;
            let mut step = 1;
            while start < audio.len() {
                let end = (start + step).min(audio.len());
                features.push(&audio[start..end], &mut out).unwrap();
                start = end;
                step = step * 7 % 1601 + 1;
            }
            features.finish(&mut out).unwrap();

            let frames = mel.shape()[1];
            assert_eq!(features.frames(), frames, "utt{utterance} frames");
            assert_eq!(out.len(), frames * 128, "utt{utterance} values");
            let mut max_abs = 0f32;
            for frame in 0..frames {
                for bin in 0..128 {
                    let value = out[frame * 128 + bin];
                    assert!(value.is_finite(), "frame {frame} bin {bin} is {value}");
                    max_abs = max_abs.max((value - mel[[bin, frame]]).abs());
                }
            }
            // log-mel spans about -16 to 16 here; NeMo's FFT rounds differently
            assert!(
                max_abs < 1e-3,
                "utt{utterance} max abs mel difference {max_abs}"
            );
        }
    }
}
