//! Parakeet on the CUDA engine, behind the backend boundary
//!
//! A batch of chunks is padded to a mel-frame bucket, so inputs of similar length
//! share one set of device buffers and captured graphs. The first batch of a shape
//! runs eagerly, which also finishes cuBLAS's lazy setup, and records graphs that
//! later batches of that shape replay. Shapes are evicted least recently used once
//! their buffers pass a share of device memory

use std::collections::HashMap;
use std::fmt;
use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use cudakit::{Graph, Library, Options};

use crate::backend::{ChunkDecode, DecodePolicy, EncoderChunk, LanguageMask};
use crate::cuda::decoder::{CudaDecoder, DecodeWorkspace};
use crate::cuda::encoder::{CudaEncoder, EncoderWorkspace};
use crate::cuda::kernels::CudaRuntime;
use crate::cuda::math;
use crate::deployment::ParakeetDeployment;
use crate::error::TranscriptionError;
use crate::models::Precision;
use crate::native::weights::WeightFile;

/// Mel frames a batch is padded up to a multiple of: 0.64 s, so padding costs a
/// 10 s utterance at most 6 % more encoder work
const MEL_BUCKET: usize = 64;

/// Decode steps per replay of the decode graph, between host checks of the frame
/// cursors
const STEPS_PER_CHECK: usize = 16;

/// Cached shapes may hold up to this fraction (1 / divisor) of the device memory
/// free once the weights are loaded; the rest stays free for other work on the same
/// GPU, such as diarization
const CACHE_SHARE_DIVISOR: usize = 2;

/// Device buffers and graphs for one `(batch, padded mel frames)` shape
struct Shape {
    // graphs come first so they drop before the buffers they point at
    encoder_graph: Option<Graph>,
    decode_graph: Option<Graph>,
    encoder: EncoderWorkspace,
    decoder: DecodeWorkspace,
    bytes: usize,
    last_used: u64,
}

/// Shapes by `(batch, padded mel frames)`, bounded by the bytes their buffers hold
struct ShapeCache {
    shapes: HashMap<(usize, usize), Shape>,
    bytes: usize,
    budget: usize,
    uses: u64,
}

impl ShapeCache {
    fn new(budget: usize) -> Self {
        Self {
            shapes: HashMap::new(),
            bytes: 0,
            budget,
            uses: 0,
        }
    }

    fn clear(&mut self) {
        self.shapes.clear();
        self.bytes = 0;
    }

    /// Adds `shape`, then evicts the least recently used others while over budget;
    /// a shape larger than the whole budget is still kept, alone
    fn insert(&mut self, key: (usize, usize), shape: Shape) {
        self.bytes += shape.bytes;
        self.shapes.insert(key, shape);
        while self.bytes > self.budget {
            let Some(oldest) = self
                .shapes
                .iter()
                .filter(|(other, _)| **other != key)
                .min_by_key(|(_, shape)| shape.last_used)
                .map(|(other, _)| *other)
            else {
                break;
            };
            if let Some(evicted) = self.shapes.remove(&oldest) {
                self.bytes -= evicted.bytes;
            }
        }
    }

    fn get(&mut self, key: (usize, usize)) -> Option<&mut Shape> {
        self.uses += 1;
        let shape = self.shapes.get_mut(&key)?;
        shape.last_used = self.uses;
        Some(shape)
    }
}

/// Held while any engine in the process loads, captures a graph or frees device
/// resources: its teardown, and a shape cache's additions and evictions
///
/// Each engine opens its own runtime, but all of them share device 0's primary
/// context. Tearing one down while another thread captures fails that capture with
/// `CUBLAS_STATUS_INTERNAL_ERROR`, `CUBLAS_STATUS_EXECUTION_FAILED` or
/// `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`, and loading one fails the load with
/// `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`. Parallel long-form workers each load an
/// engine, so a worker that loads late can meet another's first captures, and the
/// first to finish drops its engine while another may still be capturing the last
/// window's shape. Shape changes free only graphs and stream-ordered buffers, which
/// should leave a capture intact, but they are rare enough to hold apart for
/// nothing, so every free is covered. Eager runs and replays overlap all of these
/// without failing, so they stay outside it
static EXCLUSIVE_DEVICE_WORK: Mutex<()> = Mutex::new(());

fn exclusive_device_work() -> MutexGuard<'static, ()> {
    // the lock guards no data, so a panic while holding it leaves nothing to repair
    EXCLUSIVE_DEVICE_WORK
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

/// An engine that drops under [`EXCLUSIVE_DEVICE_WORK`]
///
/// The lock has to cover the fields' drops, which run after `Drop::drop` returns,
/// so the engine is dropped by hand inside it
struct ExclusiveTeardown(ManuallyDrop<Engine>);

impl Deref for ExclusiveTeardown {
    type Target = Engine;

    fn deref(&self) -> &Engine {
        &self.0
    }
}

impl Drop for ExclusiveTeardown {
    fn drop(&mut self) {
        let _exclusive = exclusive_device_work();
        // SAFETY: the engine is dropped only here, and nothing reads it afterward
        unsafe { ManuallyDrop::drop(&mut self.0) };
    }
}

struct Engine {
    // fields drop in order: graphs and buffers, then weights, then the context
    shapes: Mutex<ShapeCache>,
    encoder: CudaEncoder,
    decoder: CudaDecoder,
    features: usize,
    runtime: CudaRuntime,
}

/// Parakeet encoder and greedy decode on CUDA device 0, shared by clones
///
/// Clones may run on any thread; the shape cache's lock also serializes graph
/// capture, which cudakit requires across threads sharing a runtime, and
/// [`EXCLUSIVE_DEVICE_WORK`] keeps every capture apart from other engines' loads and
/// teardown
#[derive(Clone)]
pub(crate) struct CudaModel(Arc<ExclusiveTeardown>);

// a backend handle that is not thread-safe would quietly make pipelines `!Send`
const _: () = {
    const fn thread_safe<T: Send + Sync>() {}
    thread_safe::<CudaModel>();
};

impl fmt::Debug for CudaModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaModel")
            .field("features", &self.0.features)
            .finish_non_exhaustive()
    }
}

impl CudaModel {
    /// Loads native weights for `deployment` onto CUDA device 0, with the encoder
    /// GEMMs at `precision`
    pub(crate) fn load(
        weights: &Path,
        deployment: &ParakeetDeployment,
        precision: Precision,
    ) -> Result<Self, TranscriptionError> {
        let file = WeightFile::open(weights)?;
        let spec = file.parakeet_spec(deployment)?;
        // taken before the runtime opens, so it outlives everything a failed load
        // drops on its way out, which is the same teardown an engine's drop does
        let _exclusive = exclusive_device_work();
        let runtime = CudaRuntime::open(0, Options::default())?;
        runtime.prepare(Library::Cublas)?;
        let encoder = CudaEncoder::load(&runtime, &file, spec.encoder, math(precision))?;
        let decoder = CudaDecoder::load(&runtime, &file, &spec)?;
        let (free, _) = runtime.stream().context().mem_get_info()?;
        Ok(Self(Arc::new(ExclusiveTeardown(ManuallyDrop::new(
            Engine {
                shapes: Mutex::new(ShapeCache::new(free / CACHE_SHARE_DIVISOR)),
                encoder,
                decoder,
                features: spec.encoder.features,
                runtime,
            },
        )))))
    }

    pub(crate) fn transcribe_chunks(
        &self,
        chunks: &[EncoderChunk<'_>],
        policy: &DecodePolicy<'_>,
    ) -> Result<Vec<ChunkDecode>, TranscriptionError> {
        if policy.keep_top_k() {
            return Err(TranscriptionError::InvalidConfiguration(
                "the CUDA backend does not return top-K candidates".to_owned(),
            ));
        }
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        let engine = &self.0;
        let features = engine.features;
        if let Some(chunk) = chunks
            .iter()
            .find(|chunk| chunk.features().ncols() != features)
        {
            return Err(TranscriptionError::InvalidConfiguration(format!(
                "chunk has {} mel features; the model takes {features}",
                chunk.features().ncols()
            )));
        }

        let lens: Vec<usize> = chunks.iter().map(EncoderChunk::frames).collect();
        let longest = lens.iter().copied().max().unwrap_or(0).max(1);
        let mel_frames = longest.div_ceil(MEL_BUCKET) * MEL_BUCKET;
        let batch = chunks.len();
        let mut shapes = engine.shapes.lock().map_err(|_| {
            TranscriptionError::InvalidModelOutput("the CUDA model lock was poisoned".to_owned())
        })?;
        let key = (batch, mel_frames);
        if !shapes.shapes.contains_key(&key) {
            // adding a shape may free others' graphs and buffers, by eviction or by the
            // retry below, and a failed allocation frees what it had made
            let _exclusive = exclusive_device_work();
            let shape = match engine.shape(batch, mel_frames) {
                Ok(shape) => shape,
                // the allocation may have failed on memory cached shapes hold: free
                // them all and try once more
                Err(_) => {
                    shapes.clear();
                    engine.shape(batch, mel_frames)?
                }
            };
            shapes.insert(key, shape);
        }
        let shape = shapes.get(key).ok_or_else(|| {
            TranscriptionError::InvalidModelOutput("a cached CUDA shape went missing".to_owned())
        })?;
        engine.transcribe(shape, chunks, &lens, policy.language_mask().as_ref())
    }
}

impl Engine {
    fn shape(&self, batch: usize, mel_frames: usize) -> Result<Shape, TranscriptionError> {
        let encoder = self.encoder.workspace(&self.runtime, batch, mel_frames)?;
        let decoder = self
            .decoder
            .workspace(&self.runtime, batch, encoder.frames())?;
        Ok(Shape {
            encoder_graph: None,
            decode_graph: None,
            bytes: encoder.bytes() + decoder.bytes(),
            last_used: 0,
            encoder,
            decoder,
        })
    }

    fn transcribe(
        &self,
        shape: &mut Shape,
        chunks: &[EncoderChunk<'_>],
        lens: &[usize],
        language: Option<&LanguageMask>,
    ) -> Result<Vec<ChunkDecode>, TranscriptionError> {
        let runtime = &self.runtime;
        let row_len = shape.encoder.mel_frames() * self.features;
        self.encoder
            .set_input(runtime, &mut shape.encoder, lens, |mel| {
                // time-major `[batch, mel_frames, features]`, zero past each chunk's frames
                for (row, chunk) in mel.chunks_exact_mut(row_len).zip(chunks) {
                    let (valid, padding) = row.split_at_mut(chunk.frames() * self.features);
                    match chunk.features().as_slice() {
                        Some(values) => valid.copy_from_slice(values),
                        None => {
                            for (slot, value) in valid.iter_mut().zip(chunk.features()) {
                                *slot = *value;
                            }
                        }
                    }
                    padding.fill(0.0);
                }
            })?;
        match &shape.encoder_graph {
            Some(graph) => graph.launch()?,
            None => {
                self.encoder.enqueue(runtime, &mut shape.encoder)?;
                let encoder_ws = &mut shape.encoder;
                let _exclusive = exclusive_device_work();
                // SAFETY: the graph is stored in the same `Shape` as the workspace it
                // records, ahead of it, so it drops first; the workspace's buffers are
                // never reallocated, and every launch is on the runtime's stream. The
                // closure returns errors instead of panicking
                let graph = unsafe {
                    runtime.capture_nonempty(|| self.encoder.enqueue(runtime, encoder_ws))
                }?;
                shape.encoder_graph = Some(graph);
            }
        }

        // written outside capture, so the decode graph reads this call's filter
        self.decoder
            .set_language(runtime, &mut shape.decoder, language)?;
        self.decoder.start(
            runtime,
            &mut shape.decoder,
            shape.encoder.output(),
            shape.encoder.output_lens(),
        )?;
        match &shape.decode_graph {
            Some(graph) => {
                self.decoder
                    .run_with(runtime, &mut shape.decoder, STEPS_PER_CHECK, |_| {
                        Ok(graph.launch()?)
                    })?
            }
            None => {
                self.decoder
                    .run(runtime, &mut shape.decoder, STEPS_PER_CHECK)?;
                let decoder_ws = &mut shape.decoder;
                let _exclusive = exclusive_device_work();
                // SAFETY: as for the encoder graph above
                let graph = unsafe {
                    runtime.capture_nonempty(|| {
                        (0..STEPS_PER_CHECK)
                            .try_for_each(|_| self.decoder.step(runtime, decoder_ws))
                    })
                }?;
                shape.decode_graph = Some(graph);
            }
        }

        Ok(self
            .decoder
            .output(runtime, &mut shape.decoder)?
            .into_iter()
            .map(|raw| ChunkDecode { raw, top_k: None })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use ndarray::{Array2, Axis};

    use super::{CudaModel, MEL_BUCKET};
    use crate::backend::{DecodePolicy, EncoderChunk};
    use crate::config::TranscriptionConfig;
    use crate::cuda::test_support::{native_root, nemo_parity_bundle};
    use crate::decode::RawTranscription;
    use crate::deployment::{ParakeetDeployment, ParakeetVersion};
    use crate::frontend::ParakeetFeatureExtractor;
    use crate::language::{LanguageSelection, ParakeetLanguage};
    use crate::models::{ModelBundle, Precision};
    use crate::pipeline::TranscriptionPipeline;
    use crate::test_support::NemoReference;
    use crate::types::TranscriptionResult;

    /// Runs each b8 utterance alone under a cache budget of two of the largest
    /// shapes, forward, backward and forward again, so calls keep evicting shapes and
    /// capturing their graphs anew, and holds every output to the first, unbounded run
    #[test]
    #[ignore = "needs a CUDA GPU, the native weights and the NeMo references; `just gpu-parity` runs it on gpu0"]
    fn gpu_evicted_shapes_transcribe_the_same() {
        let root = native_root();
        let bundle = ModelBundle::from_native_dir(&root, ParakeetDeployment::v3());
        let deployment = bundle.deployment().clone();
        let vocab = bundle.vocabulary().expect("vocabulary");
        let model = CudaModel::load(
            &root.join(deployment.native_weights_path()),
            &deployment,
            Precision::Fp32,
        )
        .expect("model");
        let policy =
            DecodePolicy::new(&deployment, &vocab, LanguageSelection::Auto).expect("policy");
        let extractor =
            ParakeetFeatureExtractor::new(&TranscriptionConfig::from_deployment(&deployment));

        let reference = NemoReference::open(ParakeetVersion::V3, "b8");
        let audio = reference.f32("audio");
        let features: Vec<Array2<f32>> = reference
            .i64("audio_len")
            .iter()
            .enumerate()
            .map(|(row, len)| {
                let samples: Vec<f32> = audio
                    .index_axis(Axis(0), row)
                    .iter()
                    .take(*len as usize)
                    .copied()
                    .collect();
                extractor.extract(&samples).expect("features")
            })
            .collect();
        let run = |features: &Array2<f32>| -> RawTranscription {
            let mut decoded = model
                .transcribe_chunks(&[EncoderChunk::new(features.view())], &policy)
                .expect("transcribe");
            decoded.remove(0).raw
        };
        let expected: Vec<RawTranscription> = features.iter().map(run).collect();

        let (shapes, largest) = {
            let mut cache = model.0.shapes.lock().expect("cache");
            let shapes = cache.shapes.len();
            let largest = cache.shapes.values().map(|shape| shape.bytes).max();
            cache.clear();
            (shapes, largest.expect("a cached shape"))
        };
        assert!(
            shapes >= 3,
            "the utterances fall in {shapes} shapes; eviction needs 3"
        );
        model.0.shapes.lock().expect("cache").budget = 2 * largest;

        let rows = features.len();
        let order = (0..rows).chain((0..rows).rev()).chain(0..rows);
        for (call, row) in order.enumerate() {
            let raw = run(&features[row]);
            let want = &expected[row];
            assert_eq!(
                raw.token_ids, want.token_ids,
                "call {call} row {row} tokens"
            );
            assert_eq!(
                raw.frame_indices, want.frame_indices,
                "call {call} row {row} frames"
            );
            assert_eq!(
                raw.durations, want.durations,
                "call {call} row {row} durations"
            );
            assert_eq!(
                raw.confidences, want.confidences,
                "call {call} row {row} confidences"
            );

            let cache = model.0.shapes.lock().expect("cache");
            let held: usize = cache.shapes.values().map(|shape| shape.bytes).sum();
            assert_eq!(cache.bytes, held, "call {call}: cache byte count");
            assert!(
                cache.bytes <= cache.budget,
                "call {call}: cache over budget"
            );
        }
    }

    /// Keeps capturing graphs on one model while a second thread loads others and
    /// drops each one during a call, as parallel long-form workers do when one
    /// loads late or finishes first: a load during a capture failed with
    /// `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`, and an engine torn down mid-capture
    /// failed the capture with a cuBLAS or capture-invalidated error
    #[test]
    #[ignore = "needs a CUDA GPU and the native weights; `just gpu-parity` runs it on gpu0"]
    fn gpu_engine_teardown_does_not_break_another_capture() {
        /// Capture is most of a call, so each drop that starts mid-call is likely to
        /// overlap one; this many, each in a different call, make a miss vanishingly
        /// rare
        const TEARDOWNS: usize = 8;
        /// `current_call` once the capture thread has stopped, which lets
        /// the remaining drops through
        const STOPPED: usize = usize::MAX;

        let root = native_root();
        let deployment = ParakeetDeployment::v3();
        let weights = root.join(deployment.native_weights_path());
        let load = || CudaModel::load(&weights, &deployment, Precision::Fp32).expect("model");
        let model = load();
        let vocab = ModelBundle::from_native_dir(&root, deployment.clone())
            .vocabulary()
            .expect("vocabulary");
        let policy =
            DecodePolicy::new(&deployment, &vocab, LanguageSelection::Auto).expect("policy");
        let features = Array2::<f32>::zeros((4 * MEL_BUCKET, model.0.features));
        // the 1-based number of the call in flight, or 0 between calls
        let current_call = AtomicUsize::new(0);

        let failure = std::thread::scope(|scope| {
            let dropper = scope.spawn(|| {
                let mut last_call = 0;
                for _ in 0..TEARDOWNS {
                    let other = load();
                    // start each teardown inside a call no earlier teardown started in
                    loop {
                        let call = current_call.load(Ordering::Acquire);
                        if call == STOPPED || (call != 0 && call != last_call) {
                            last_call = call;
                            break;
                        }
                        std::hint::spin_loop();
                    }
                    drop(other);
                }
            });

            let mut calls = 0;
            while !dropper.is_finished() {
                // an empty cache makes every call capture its graphs anew
                model.0.shapes.lock().expect("cache").clear();
                current_call.store(calls + 1, Ordering::Release);
                let decoded =
                    model.transcribe_chunks(&[EncoderChunk::new(features.view())], &policy);
                let failure = match decoded {
                    Ok(decoded) if decoded.len() == 1 => None,
                    Ok(decoded) => Some(format!("call {calls}: {} outputs", decoded.len())),
                    Err(error) => Some(format!("call {calls}: {error}")),
                };
                if failure.is_some() {
                    // lets the remaining drops through, so the scope can join
                    current_call.store(STOPPED, Ordering::Release);
                    return failure;
                }

                current_call.store(0, Ordering::Release);
                calls += 1;
            }
            None
        });

        assert_eq!(failure, None);
    }

    /// Runs the 120 s long-full audio through native long-form, its six windows
    /// batched into one model call and then one window at a time, and holds the two
    /// to the same tokens
    #[cfg(feature = "long-form")]
    #[test]
    #[ignore = "needs a CUDA GPU, the native weights and the NeMo references; `just gpu-parity` runs it on gpu0"]
    fn gpu_batched_long_form_matches_serial_windows() {
        use crate::long_form::{LongFormConfig, LongFormTranscriptionPipeline};

        let bundle = nemo_parity_bundle(ParakeetDeployment::v3());
        let batched = LongFormConfig::for_bundle(&bundle);
        assert!(batched.batch_size >= 6, "the windows must share one batch");
        let serial = LongFormConfig {
            batch_size: 1,
            worker_count: 1,
            ..batched.clone()
        };
        let pipeline = LongFormTranscriptionPipeline::from_bundle(bundle).expect("pipeline");
        let reference = NemoReference::open(ParakeetVersion::V3, "long-full");
        let audio: Vec<f32> = reference
            .f32("audio")
            .iter()
            .take(reference.i64("audio_len")[0] as usize)
            .copied()
            .collect();

        let batched = pipeline.run_with_config(&audio, &batched).expect("batched");
        let serial = pipeline.run_with_config(&audio, &serial).expect("serial");
        let ids = |result: &crate::types::TranscriptionResult| -> Vec<u32> {
            result.tokens.iter().map(|token| token.token_id).collect()
        };
        assert!(!batched.tokens.is_empty(), "no tokens");
        assert_eq!(ids(&batched), ids(&serial), "token ids");
        assert_eq!(batched.text, serial.text);
    }

    /// Transcribes the reference audio through the public pipeline on a native
    /// bundle, twice per utterance so the second pass replays the captured graphs,
    /// and holds the text against NeMo's
    #[test]
    #[ignore = "needs a CUDA GPU, the native weights and the NeMo references; `just gpu-parity` runs it on gpu0"]
    fn gpu_pipeline_transcribes_like_nemo() {
        for version in [ParakeetVersion::V2, ParakeetVersion::V3] {
            let deployment = match version {
                ParakeetVersion::V2 => ParakeetDeployment::v2(),
                ParakeetVersion::V3 => ParakeetDeployment::v3(),
            };
            let pipeline = TranscriptionPipeline::from_bundle(nemo_parity_bundle(deployment))
                .expect("pipeline");

            for set in ["b1", "b8"] {
                let reference = NemoReference::open(version, set);
                let audio = reference.f32("audio");
                let lens = reference.i64("audio_len");
                for (row, expected) in reference.nemo_texts().iter().enumerate() {
                    let samples: Vec<f32> = audio
                        .index_axis(Axis(0), row)
                        .iter()
                        .take(lens[row] as usize)
                        .copied()
                        .collect();
                    for pass in ["eager", "graph"] {
                        let text = pipeline.run(&samples).expect("transcribe").text;
                        assert_eq!(&text, expected, "{version:?} {set} row {row} ({pass})");
                    }
                }
            }
        }
    }

    /// Transcribes the b8 English audio with the language set to English, whose
    /// Latin-script filter allows every token the automatic run emits: the tokens
    /// stay the same while the confidences become softmaxes over the top-K logits.
    /// The filtered run replays the graphs the automatic run captured, and a last
    /// automatic run holds that the filter turns off again
    #[test]
    #[ignore = "needs a CUDA GPU, the native weights and the NeMo references; `just gpu-parity` runs it on gpu0"]
    fn gpu_english_filter_keeps_the_automatic_tokens() {
        let bundle = nemo_parity_bundle(ParakeetDeployment::v3());
        let deployment = bundle.deployment().clone();
        let vocab = bundle.vocabulary().expect("vocabulary");
        let pipeline = TranscriptionPipeline::from_bundle(bundle).expect("pipeline");
        let automatic = pipeline.config().clone();
        let english = TranscriptionConfig {
            language: LanguageSelection::Language(ParakeetLanguage::English),
            ..automatic.clone()
        };
        let policy = DecodePolicy::new(&deployment, &vocab, english.language).expect("policy");
        let mask = policy
            .language_mask()
            .expect("v3 filters an explicit language");
        let ids = |result: &TranscriptionResult| -> Vec<u32> {
            result.tokens.iter().map(|token| token.token_id).collect()
        };
        let confidences = |result: &TranscriptionResult| -> Vec<f32> {
            result.tokens.iter().map(|token| token.confidence).collect()
        };

        let reference = NemoReference::open(ParakeetVersion::V3, "b8");
        let audio = reference.f32("audio");
        let lens = reference.i64("audio_len");
        let mut rescored = 0;
        for (row, len) in lens.iter().enumerate() {
            let samples: Vec<f32> = audio
                .index_axis(Axis(0), row)
                .iter()
                .take(*len as usize)
                .copied()
                .collect();
            let first = pipeline
                .run_with_config(&samples, &automatic)
                .expect("automatic");
            let filtered = pipeline
                .run_with_config(&samples, &english)
                .expect("english");
            let last = pipeline
                .run_with_config(&samples, &automatic)
                .expect("automatic again");

            // the premise: a filter that allows every argmax never changes a token
            assert!(!first.tokens.is_empty(), "row {row}: no tokens");
            assert!(
                first
                    .tokens
                    .iter()
                    .all(|token| mask.allowed()[token.token_id as usize]),
                "row {row}: the automatic run emitted a non-Latin token"
            );
            assert_eq!(ids(&filtered), ids(&first), "row {row} token ids");
            assert_eq!(filtered.text, first.text, "row {row} text");
            for (index, (filtered, first)) in filtered.tokens.iter().zip(&first.tokens).enumerate()
            {
                // a softmax over a subset of the logits that holds the max scores the
                // max at least as high, up to f32 rounding
                assert!(
                    filtered.confidence >= first.confidence - 1e-6,
                    "row {row} token {index}: {} below {}",
                    filtered.confidence,
                    first.confidence
                );
                if filtered.confidence != first.confidence {
                    rescored += 1;
                }
            }
            assert_eq!(
                ids(&last),
                ids(&first),
                "row {row}: tokens after the filter"
            );
            assert_eq!(
                confidences(&last),
                confidences(&first),
                "row {row}: confidences after the filter"
            );
        }

        println!("{rescored} tokens rescored by the top-K softmax");
        assert!(rescored > 0, "no confidence came from the top-K softmax");
    }
}
