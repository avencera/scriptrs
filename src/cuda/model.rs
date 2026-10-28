//! Parakeet on the CUDA engine, behind the backend boundary
//!
//! A batch of chunks is padded to a mel-frame bucket, so inputs of similar length
//! share one set of device buffers and captured graphs. The first batch of a shape
//! runs eagerly, which also finishes cuBLAS's lazy setup, and records graphs that
//! later batches of that shape replay

use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use std::rc::Rc;
use std::sync::Mutex;

use cudakit::{Graph, Library, Options};

use crate::backend::{ChunkDecode, DecodePolicy, EncoderChunk};
use crate::cuda::decoder::{CudaDecoder, DecodeWorkspace};
use crate::cuda::encoder::{CudaEncoder, EncoderWorkspace};
use crate::cuda::kernels::CudaRuntime;
use crate::deployment::ParakeetDeployment;
use crate::error::TranscriptionError;
use crate::native::weights::WeightFile;

/// Mel frames a batch is padded up to a multiple of: 0.64 s, so padding costs a
/// 10 s utterance at most 6 % more encoder work
const MEL_BUCKET: usize = 64;

/// Decode steps per replay of the decode graph, between host checks of the frame
/// cursors
const STEPS_PER_CHECK: usize = 16;

/// Shapes kept before the cache is cleared; each holds its buffers and graphs
const MAX_SHAPES: usize = 64;

/// Device buffers and graphs for one `(batch, padded mel frames)` shape
struct Shape {
    // graphs come first so they drop before the buffers they point at
    encoder_graph: Option<Graph>,
    decode_graph: Option<Graph>,
    encoder: EncoderWorkspace,
    decoder: DecodeWorkspace,
}

struct Engine {
    // fields drop in order: graphs and buffers, then weights, then the context
    shapes: Mutex<HashMap<(usize, usize), Shape>>,
    encoder: CudaEncoder,
    decoder: CudaDecoder,
    features: usize,
    runtime: CudaRuntime,
}

/// Parakeet encoder and greedy decode on CUDA device 0, shared by clones
///
/// `Rc` because cudakit's runtime is not `Send` yet; clones stay on one thread
#[derive(Clone)]
pub(crate) struct CudaModel(Rc<Engine>);

impl fmt::Debug for CudaModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaModel")
            .field("features", &self.0.features)
            .finish_non_exhaustive()
    }
}

impl CudaModel {
    /// Loads native weights for `deployment` onto CUDA device 0
    pub(crate) fn load(
        weights: &Path,
        deployment: &ParakeetDeployment,
    ) -> Result<Self, TranscriptionError> {
        let file = WeightFile::open(weights)?;
        let spec = file.parakeet_spec(deployment)?;
        let runtime = CudaRuntime::open(0, Options::default())?;
        runtime.prepare(Library::Cublas)?;
        let encoder = CudaEncoder::load(&runtime, &file, spec.encoder)?;
        let decoder = CudaDecoder::load(&runtime, &file, &spec)?;
        Ok(Self(Rc::new(Engine {
            shapes: Mutex::new(HashMap::new()),
            encoder,
            decoder,
            features: spec.encoder.features,
            runtime,
        })))
    }

    pub(crate) fn transcribe_chunks(
        &self,
        chunks: &[EncoderChunk<'_>],
        policy: &DecodePolicy<'_>,
    ) -> Result<Vec<ChunkDecode>, TranscriptionError> {
        if policy.language_mask().is_some() || policy.keep_top_k() {
            return Err(TranscriptionError::InvalidConfiguration(
                "the CUDA backend does not apply the language filter or return top-K candidates yet"
                    .to_owned(),
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
        // time-major `[batch, mel_frames, features]`, zero past each chunk's frames
        let mut mel = vec![0.0f32; batch * mel_frames * features];
        for (row, chunk) in chunks.iter().enumerate() {
            let start = row * mel_frames * features;
            for (frame, values) in chunk.features().rows().into_iter().enumerate() {
                let offset = start + frame * features;
                for (slot, value) in mel[offset..offset + features].iter_mut().zip(values) {
                    *slot = *value;
                }
            }
        }

        let mut shapes = engine.shapes.lock().map_err(|_| {
            TranscriptionError::InvalidModelOutput("the CUDA model lock was poisoned".to_owned())
        })?;
        if !shapes.contains_key(&(batch, mel_frames)) && shapes.len() >= MAX_SHAPES {
            shapes.clear();
        }
        let shape = match shapes.entry((batch, mel_frames)) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let encoder = engine
                    .encoder
                    .workspace(&engine.runtime, batch, mel_frames)?;
                let decoder = engine
                    .decoder
                    .workspace(&engine.runtime, batch, encoder.frames())?;
                entry.insert(Shape {
                    encoder_graph: None,
                    decode_graph: None,
                    encoder,
                    decoder,
                })
            }
        };
        engine.transcribe(shape, &mel, &lens)
    }
}

impl Engine {
    fn transcribe(
        &self,
        shape: &mut Shape,
        mel: &[f32],
        lens: &[usize],
    ) -> Result<Vec<ChunkDecode>, TranscriptionError> {
        let runtime = &self.runtime;
        self.encoder
            .set_input(runtime, &mut shape.encoder, mel, lens)?;
        match &shape.encoder_graph {
            Some(graph) => graph.launch()?,
            None => {
                self.encoder.enqueue(runtime, &mut shape.encoder)?;
                let encoder_ws = &mut shape.encoder;
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
            .output(runtime, &shape.decoder)?
            .into_iter()
            .map(|raw| ChunkDecode { raw, top_k: None })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use crate::deployment::{ParakeetDeployment, ParakeetVersion};
    use crate::models::ModelBundle;
    use crate::pipeline::TranscriptionPipeline;
    use crate::test_support::NemoReference;

    /// Transcribes the reference audio through the public pipeline on a native
    /// bundle, twice per utterance so the second pass replays the captured graphs,
    /// and holds the text against NeMo's
    #[test]
    #[ignore = "needs a CUDA GPU, the native weights and the NeMo references; `just gpu-parity` runs it on gpu0"]
    fn gpu_pipeline_transcribes_like_nemo() {
        let root = std::env::var_os("SCRIPTRS_NATIVE_MODELS")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| "/shared/models/scriptrs-native".into());
        for version in [ParakeetVersion::V2, ParakeetVersion::V3] {
            let deployment = match version {
                ParakeetVersion::V2 => ParakeetDeployment::v2(),
                ParakeetVersion::V3 => ParakeetDeployment::v3(),
            };
            let pipeline =
                TranscriptionPipeline::from_bundle(ModelBundle::from_native_dir(&root, deployment))
                    .expect("pipeline");

            for set in ["b1", "b8"] {
                let reference = NemoReference::open(version, set);
                let audio = reference.f32("audio");
                let lens = reference.i64("audio_len");
                for (row, expected) in reference.nemo_texts().iter().enumerate() {
                    let samples: Vec<f32> = audio
                        .index_axis(ndarray::Axis(0), row)
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
}
