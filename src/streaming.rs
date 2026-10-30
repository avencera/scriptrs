//! Streaming transcription with cache-aware FastConformer models on CUDA
//!
//! A [`StreamingModel`] serves many [`Stream`]s at once. Each stream takes audio as
//! it arrives, and [`StreamingModel::process`] encodes every chunk the streams have
//! audio for, one chunk from each stream per batch, so concurrent streams share the
//! GPU's work. Each stream's encoder caches and decoder state stay on the device
//! between chunks
//!
//! Chunks follow NeMo's cache-aware streaming: the encoder sees `lookahead` encoder
//! frames (80 ms each) past the chunk it emits, so a chunk's tokens arrive
//! `1 + lookahead` frames of audio after it starts, plus the 2.5 ms the last mel
//! frame's window reaches past them

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use crate::constants::ENCODER_SUBSAMPLING;
use crate::cuda::encoder::{MEL_CONTEXT, chunk_mel, first_chunk_mel};
use crate::cuda::streaming::{EngineChunk, StreamEngine};
use crate::decode::{ParakeetTdtDecoder, RawTranscription};
use crate::error::TranscriptionError;
use crate::frontend::stream::{MelSetup, StreamingFeatures};
use crate::models::Precision;
use crate::types::TranscriptionResult;
use crate::vocab::Vocabulary;

/// A chunk's mel frames: `frames` from frame `start` on, before 0 for a later
/// chunk whose context reaches before the audio
#[derive(Debug, Clone, Copy)]
struct ChunkWindow {
    start: isize,
    frames: usize,
}

/// Settings for a [`StreamingModel`]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamingConfig {
    /// Encoder frames (80 ms each) the encoder looks past each chunk, one of the
    /// values the model was trained with: 0, 1, 6 or 13 for Nemotron Speech
    /// Streaming EN, 0, 3, 6 or 13 for Nemotron 3.5 ASR Streaming. A chunk spans
    /// `1 + lookahead` encoder frames, so more lookahead means fewer, larger
    /// chunks, more accuracy and more latency
    pub lookahead: usize,
    /// Most streams open at once; each holds up to about 15 MB of device cache
    pub max_streams: usize,
    /// Most chunks encoded together in one batch
    pub max_batch: usize,
    /// Encoder GEMM precision. [`Precision::Fp32`] gives NeMo's streaming tokens
    /// exactly; FP16, the default, is 1.4 to 1.9 times as fast, with WER on
    /// LibriSpeech within 0.003 points of it. The `low-precision` modes are for
    /// offline transcription only, and loading fails with them
    pub precision: Precision,
}

impl StreamingConfig {
    /// `lookahead` frames of lookahead, room for 64 streams in batches of up to 32
    /// chunks, and FP16 GEMMs
    pub fn new(lookahead: usize) -> Self {
        Self {
            lookahead,
            max_streams: 64,
            max_batch: 32,
            precision: Precision::Fp16,
        }
    }
}

/// What every stream of a model shares
struct Shared {
    engine: StreamEngine,
    mel: Arc<MelSetup>,
    decoder: ParakeetTdtDecoder,
    sample_rate: usize,
    /// Mel frames in a stream's first chunk
    first_mel: usize,
    /// Each language tag's prompt, empty for a single-language checkpoint
    languages: BTreeMap<String, u32>,
    /// Slots no open stream holds
    free: Mutex<Vec<usize>>,
}

/// A cache-aware streaming model on CUDA device 0, Nemotron Speech Streaming EN
/// 0.6B or Nemotron 3.5 ASR Streaming 0.6B, shared by clones
#[derive(Clone)]
pub struct StreamingModel(Arc<Shared>);

impl std::fmt::Debug for StreamingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingModel")
            .field("frames_per_chunk", &self.0.engine.frames())
            .finish_non_exhaustive()
    }
}

impl StreamingModel {
    /// Loads a model exported by `dev-tools/nemo/export_native.py`: `dir` holds its
    /// `model.safetensors` and `vocab.txt`
    pub fn from_native_dir(
        dir: impl AsRef<Path>,
        config: &StreamingConfig,
    ) -> Result<Self, TranscriptionError> {
        let dir = dir.as_ref();
        let (engine, spec) = StreamEngine::load(
            &dir.join("model.safetensors"),
            config.lookahead,
            config.max_streams,
            config.max_batch,
            config.precision,
        )?;
        let vocab = Vocabulary::from_sentencepiece_file(dir.join("vocab.txt"))?;
        let classes = spec.transducer.predictor.vocab_size;
        if vocab.len() != classes + 1 || vocab.blank_id() != classes {
            return Err(TranscriptionError::InvalidVocabulary(format!(
                "{} tokens with the blank at {}; the model has {classes} and the blank after them",
                vocab.len(),
                vocab.blank_id()
            )));
        }

        let languages = spec
            .prompt
            .map(|prompt| {
                prompt
                    .languages
                    .into_iter()
                    .map(|(language, index)| (language, index as u32))
                    .collect()
            })
            .unwrap_or_default();

        Ok(Self(Arc::new(Shared {
            first_mel: first_chunk_mel(engine.frames()),
            languages,
            mel: MelSetup::new(spec.frontend, spec.encoder.features),
            decoder: ParakeetTdtDecoder::new(vocab),
            sample_rate: spec.frontend.sample_rate,
            free: Mutex::new((0..config.max_streams).rev().collect()),
            engine,
        })))
    }

    /// Audio samples per chunk, after a stream's first
    pub fn chunk_samples(&self) -> usize {
        ENCODER_SUBSAMPLING * self.0.engine.frames() * self.0.mel.hop_length()
    }

    /// The language tags [`Self::open_stream_in`] takes, such as `en-US`; empty for
    /// a single-language model
    pub fn languages(&self) -> impl Iterator<Item = &str> {
        self.0.languages.keys().map(String::as_str)
    }

    /// Opens a stream of a single-language model, failing when `max_streams` are
    /// already open
    ///
    /// A multilingual model's streams need a language prompt, so they open with
    /// [`Self::open_stream_in`]
    pub fn open_stream(&self) -> Result<Stream, TranscriptionError> {
        if !self.0.languages.is_empty() {
            return Err(TranscriptionError::InvalidConfiguration(
                "the model is multilingual; open its streams in a language, such as `en-US` or `auto`"
                    .to_owned(),
            ));
        }
        self.open(0)
    }

    /// Opens a stream of a multilingual model in `language`, one of
    /// [`Self::languages`], failing when `max_streams` are already open
    pub fn open_stream_in(&self, language: &str) -> Result<Stream, TranscriptionError> {
        let prompt = self.0.languages.get(language).copied().ok_or_else(|| {
            TranscriptionError::InvalidConfiguration(format!(
                "the model has no `{language}` language prompt"
            ))
        })?;
        self.open(prompt)
    }

    fn open(&self, prompt: u32) -> Result<Stream, TranscriptionError> {
        let slot = self
            .0
            .free
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop()
            .ok_or_else(|| {
                TranscriptionError::InvalidConfiguration(
                    "every stream slot of the model is open".to_owned(),
                )
            })?;
        let features = self.0.mel.features();
        Ok(Stream {
            model: self.0.clone(),
            slot,
            prompt,
            features: StreamingFeatures::new(self.0.mel.clone()),
            // a later chunk's context reaches before the first frame when the first
            // chunk is shorter than it, where NeMo pads with zeros
            mel: vec![0.0; MEL_CONTEXT * features],
            mel_start: -(MEL_CONTEXT as isize),
            chunks: 0,
            encoded: 0,
            state: StreamState::Open,
            raw: RawTranscription::empty(),
        })
    }

    /// Encodes and decodes every chunk `streams` have audio for, in batches of one
    /// chunk per stream, and returns how many chunks ran
    ///
    /// A batch that fails once its streams' device state has started to change
    /// leaves those streams [failed](Stream::is_failed): later calls skip them, and
    /// they keep the transcript of the chunks before it. A batch that fails before
    /// leaves its streams as they were
    pub fn process(&self, streams: &mut [&mut Stream]) -> Result<usize, TranscriptionError> {
        if let Some(stream) = streams
            .iter()
            .find(|stream| !Arc::ptr_eq(&stream.model, &self.0))
        {
            return Err(TranscriptionError::InvalidConfiguration(format!(
                "stream in slot {} belongs to another model",
                stream.slot
            )));
        }

        let engine = &self.0.engine;
        let mut processed = 0;
        loop {
            let ready: Vec<(usize, ChunkWindow)> = streams
                .iter()
                .enumerate()
                .filter_map(|(index, stream)| Some((index, stream.next_chunk()?)))
                .collect();
            if ready.is_empty() {
                return Ok(processed);
            }

            for batch in ready.chunks(engine.max_batch()) {
                let chunks: Vec<EngineChunk<'_>> = batch
                    .iter()
                    .map(|(index, window)| streams[*index].chunk(*window))
                    .collect();
                let decoded = match engine.run(&chunks) {
                    Ok(decoded) => decoded,
                    Err(failure) => {
                        if failure.advanced {
                            for (index, _) in batch {
                                streams[*index].state = StreamState::Failed;
                            }
                        }
                        return Err(failure.error);
                    }
                };
                for ((index, window), raw) in batch.iter().zip(decoded) {
                    streams[*index].advance(*window, raw);
                }
                processed += batch.len();
            }
        }
    }
}

/// One audio stream of a [`StreamingModel`]: push its audio as it arrives, run
/// [`StreamingModel::process`], and read the transcript so far
///
/// Dropping a stream frees its slot for another
pub struct Stream {
    model: Arc<Shared>,
    slot: usize,
    prompt: u32,
    features: StreamingFeatures,
    /// Mel frames from frame `mel_start` on, time-major
    mel: Vec<f32>,
    mel_start: isize,
    /// Chunks encoded
    chunks: usize,
    /// Encoder frames encoded
    encoded: usize,
    state: StreamState,
    /// Tokens so far, with frames counted from the stream's start
    raw: RawTranscription,
}

/// Where a [`Stream`] is in its life
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamState {
    /// Taking audio
    Open,
    /// Its audio has ended
    Finished,
    /// A failed batch left its device state unusable; finishing it later keeps it
    /// failed
    Failed,
}

impl std::fmt::Debug for Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stream")
            .field("slot", &self.slot)
            .field("received", &self.features.received())
            .field("chunks", &self.chunks)
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

impl Stream {
    /// Appends mono 16 kHz audio; a [failed](Self::is_failed) stream takes no more
    pub fn push(&mut self, audio: &[f32]) -> Result<(), TranscriptionError> {
        if self.state == StreamState::Failed {
            // nothing drains a failed stream's mel, so it would only grow
            return Err(TranscriptionError::InvalidConfiguration(
                "audio pushed to a failed stream".to_owned(),
            ));
        }
        if let Some(index) = audio.iter().position(|sample| !sample.is_finite()) {
            return Err(TranscriptionError::InvalidAudioSample {
                index: self.features.received() + index,
            });
        }
        self.features.push(audio, &mut self.mel)
    }

    /// Marks the end of the audio, so the next [`StreamingModel::process`] encodes
    /// the rest
    pub fn finish(&mut self) -> Result<(), TranscriptionError> {
        self.features.finish(&mut self.mel)?;
        if self.state == StreamState::Open {
            self.state = StreamState::Finished;
        }
        Ok(())
    }

    /// Whether the stream is finished and every chunk of it is transcribed
    pub fn is_done(&self) -> bool {
        self.state == StreamState::Finished && self.next_chunk().is_none()
    }

    /// Whether a failed [`StreamingModel::process`] left the stream unable to
    /// continue
    pub fn is_failed(&self) -> bool {
        self.state == StreamState::Failed
    }

    /// The transcript of the chunks processed so far
    ///
    /// Each call decodes every token from the stream's start, so its cost grows
    /// with the stream's length
    pub fn transcript(&self) -> TranscriptionResult {
        let seconds = self.features.received() as f64 / self.model.sample_rate as f64;
        self.model.decoder.decode(&self.raw, seconds)
    }

    /// The next chunk's mel frames, once the stream has the audio for it
    ///
    /// A chunk needs all its frames until the stream finishes; NeMo then encodes a
    /// last first chunk of any length, and a last later chunk of at least one
    /// encoder frame's mel frames
    fn next_chunk(&self) -> Option<ChunkWindow> {
        if self.state == StreamState::Failed {
            return None;
        }
        let finished = self.state == StreamState::Finished;
        let available = self.features.frames();
        let first = self.model.first_mel;
        if self.chunks == 0 {
            let frames = match available {
                _ if available >= first => first,
                0 => return None,
                _ if finished => available,
                _ => return None,
            };
            return Some(ChunkWindow { start: 0, frames });
        }

        let frames = self.model.engine.frames();
        let start = first + ENCODER_SUBSAMPLING * frames * (self.chunks - 1);
        let new = available.saturating_sub(start);
        let take = if new >= ENCODER_SUBSAMPLING * frames {
            ENCODER_SUBSAMPLING * frames
        } else if finished && new >= ENCODER_SUBSAMPLING {
            new
        } else {
            return None;
        };
        debug_assert!(MEL_CONTEXT + take <= chunk_mel(frames));
        Some(ChunkWindow {
            start: start as isize - MEL_CONTEXT as isize,
            frames: MEL_CONTEXT + take,
        })
    }

    /// The engine's view of the chunk at `window`
    fn chunk(&self, window: ChunkWindow) -> EngineChunk<'_> {
        let width = self.model.mel.features();
        let offset = (window.start - self.mel_start) as usize * width;
        EngineChunk {
            slot: self.slot,
            encoded: self.encoded,
            mel: &self.mel[offset..offset + window.frames * width],
            prompt: self.prompt,
        }
    }

    /// Records the chunk at `window` as encoded with `raw`'s tokens, and drops the
    /// mel frames no later chunk reads
    fn advance(&mut self, window: ChunkWindow, raw: RawTranscription) {
        let offset = self.encoded;
        let chunk_frames = self.model.engine.frames();
        self.raw.extend_shifted(raw, offset);
        self.chunks += 1;
        self.encoded += chunk_frames;

        let end = window.start + window.frames as isize;
        let keep = end - MEL_CONTEXT as isize;
        if keep > self.mel_start {
            let width = self.model.mel.features();
            self.mel.drain(..(keep - self.mel_start) as usize * width);
            self.mel_start = keep;
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // a failed stream's slot may hold state a batch left half advanced, which a
        // fresh stream resets on its first chunk anyway
        self.model
            .free
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(self.slot);
    }
}

// streams of one model may live on different threads, and the model is shared
const _: () = {
    const fn thread_safe<T: Send + Sync>() {}
    thread_safe::<StreamingModel>();
    thread_safe::<Stream>();
};

#[cfg(test)]
mod tests {
    use super::{Stream, StreamState, StreamingConfig, StreamingModel};
    use crate::cuda::test_support::native_root;
    #[cfg(feature = "low-precision")]
    use crate::error::TranscriptionError;
    use crate::models::Precision;
    use crate::test_support::NemoReference;

    /// Copies of each reference utterance streamed at once
    const COPIES: usize = 3;

    /// The utterances a NeMo streaming dump holds: audio, then tokens and their
    /// encoder frames
    fn utterances(reference: &NemoReference) -> Vec<(Vec<f32>, Vec<u32>, Vec<usize>)> {
        (0..)
            .take_while(|index| reference.contains(&format!("utt{index}.audio")))
            .map(|index| {
                let key = |name: &str| format!("utt{index}.{name}");
                let audio = reference.f32(&key("audio")).iter().copied().collect();
                let tokens = reference.i64(&key("tokens"));
                let frames = reference.i64(&key("timestamps"));
                (
                    audio,
                    tokens.iter().map(|token| *token as u32).collect(),
                    frames.iter().map(|frame| *frame as usize).collect(),
                )
            })
            .collect()
    }

    /// Streams `COPIES` of every reference utterance at once, each pushing pieces of
    /// its own uneven size and starting a few rounds after the last, in batches of
    /// up to 4 chunks that mix first and later chunks and need padding; holds every
    /// stream's tokens and frames to NeMo's streaming decode, in FP32 and in the FP16
    /// default, whose tokens should not change on this clean audio. The second pass
    /// reuses the freed slots and replays the captured graphs
    #[test]
    #[cfg(feature = "low-precision")]
    fn quantized_precisions_do_not_stream() {
        for precision in [Precision::Fp8, Precision::Int8, Precision::Nvfp4] {
            let config = StreamingConfig {
                precision,
                ..StreamingConfig::new(1)
            };
            // fails before reading any weights, so the directory need not exist
            let Err(error) = StreamingModel::from_native_dir("missing-model", &config) else {
                panic!("{precision:?} must not stream");
            };
            assert!(
                matches!(error, TranscriptionError::InvalidConfiguration(_)),
                "{precision:?}: {error}"
            );
        }
    }

    #[test]
    #[ignore = "needs a CUDA GPU, the native streaming weights and the NeMo streaming references"]
    fn gpu_streams_transcribe_like_nemo() {
        assert_streams_match_nemo(
            "nemotron-en",
            "nemotron-speech-streaming-en-0.6b",
            &[13, 6, 1, 0],
            None,
        );
    }

    /// [`gpu_streams_transcribe_like_nemo`] for the multilingual model with its
    /// `en-US` prompt, the language NeMo's reference ran in
    #[test]
    #[ignore = "needs a CUDA GPU, the native streaming weights and the NeMo streaming references"]
    fn gpu_multilingual_streams_transcribe_like_nemo() {
        assert_streams_match_nemo(
            "nemotron-3.5",
            "nemotron-3.5-asr-streaming-0.6b",
            &[13, 3],
            Some("en-US"),
        );
    }

    fn assert_streams_match_nemo(
        reference_model: &str,
        weights: &str,
        lookaheads: &[usize],
        language: Option<&str>,
    ) {
        let dir = native_root().join(weights);
        let runs = lookaheads.iter().flat_map(|&lookahead| {
            [Precision::Fp32, Precision::Fp16].map(|precision| (lookahead, precision))
        });
        for (lookahead, precision) in runs {
            let reference = NemoReference::streaming(reference_model, lookahead);
            let utterances = utterances(&reference);
            assert!(!utterances.is_empty(), "r{lookahead}: no utterances");
            let config = StreamingConfig {
                max_streams: COPIES * utterances.len(),
                max_batch: 4,
                precision,
                ..StreamingConfig::new(lookahead)
            };
            let model = StreamingModel::from_native_dir(&dir, &config).expect("model");

            for pass in 0..2 {
                let jobs: Vec<(usize, usize)> = (0..COPIES)
                    .flat_map(|copy| (0..utterances.len()).map(move |utt| (copy, utt)))
                    .collect();
                let mut streams: Vec<Stream> = jobs
                    .iter()
                    .map(|_| {
                        match language {
                            Some(language) => model.open_stream_in(language),
                            None => model.open_stream(),
                        }
                        .expect("stream")
                    })
                    .collect();
                let mut cursors = vec![0; jobs.len()];
                let mut round = 0;
                while streams.iter().any(|stream| !stream.is_done()) {
                    for (job, ((copy, utt), stream)) in jobs.iter().zip(&mut streams).enumerate() {
                        let audio = &utterances[*utt].0;
                        if round < job || stream.state == StreamState::Finished {
                            continue;
                        }
                        // 37 ms to about half a second, varying by copy and round
                        let step = 593 + (copy * 2_311 + round * 977) % 7_411;
                        let end = (cursors[job] + step).min(audio.len());
                        stream.push(&audio[cursors[job]..end]).expect("push");
                        cursors[job] = end;
                        if end == audio.len() {
                            stream.finish().expect("finish");
                        }
                    }
                    let mut active: Vec<&mut Stream> = streams.iter_mut().collect();
                    model.process(&mut active).expect("process");
                    round += 1;
                }

                for ((copy, utt), stream) in jobs.iter().zip(&streams) {
                    let (_, tokens, frames) = &utterances[*utt];
                    let label =
                        format!("r{lookahead} {precision:?} pass {pass} utt{utt} copy {copy}");
                    assert_eq!(&stream.raw.ids(), tokens, "{label} tokens");
                    assert_eq!(&stream.raw.frames(), frames, "{label} frames");
                    assert!(!stream.transcript().text.is_empty(), "{label} text");
                }
            }
        }
    }
}
