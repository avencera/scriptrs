//! The cache-aware streaming encoder and RNNT decode on CUDA: each call encodes
//! and decodes one chunk from each of a batch of streams
//!
//! Streams keep their attention, conv and predictor state in slot arenas on the
//! device, so a batch is any set of streams at any point in their audio. A batch is
//! padded up to a bucket size with spare slots no stream owns, so a few shapes and
//! their captured graphs cover every batch

mod prompt;

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Mutex, PoisonError};

use cudakit::{Graph, Library, Options};

use crate::cuda::decoder::{CudaDecoder, DecodeRow, DecodeWorkspace, DecoderStates};
use crate::cuda::encoder::{ChunkRow, CudaEncoder, StreamCaches, StreamWorkspace};
use crate::cuda::kernels::CudaRuntime;
use crate::cuda::model::{ExclusiveTeardown, exclusive_device_work};
use crate::cuda::{math, replay_or_capture};
use crate::decode::RawTranscription;
use crate::error::TranscriptionError;
use crate::models::Precision;
use crate::native::spec::StreamingSpec;
use crate::native::weights::WeightFile;
use prompt::{PromptBuffers, PromptProjector};

/// Batches up to this size round up to a power of two, larger ones to a multiple of
/// it: few shapes, and at most this many rows less one of padding
const BUCKET_STEP: usize = 8;

/// Rows a batch of `chunks` chunks runs as, at most `max_batch`
pub(crate) fn bucket(chunks: usize, max_batch: usize) -> usize {
    let rows = if chunks <= BUCKET_STEP {
        chunks.next_power_of_two()
    } else {
        chunks.div_ceil(BUCKET_STEP) * BUCKET_STEP
    };
    rows.min(max_batch)
}

/// One stream's chunk for [`StreamEngine::run`]
#[derive(Debug, Clone, Copy)]
pub(crate) struct EngineChunk<'a> {
    /// The stream's slot, below the engine's `streams`
    pub slot: usize,
    /// Encoder frames of the stream already encoded, 0 for its first chunk
    pub encoded: usize,
    /// The chunk's mel frames, time-major, a later chunk's context frames first
    pub mel: &'a [f32],
    /// The stream's language prompt, for a checkpoint with a prompt projector
    pub prompt: u32,
}

/// Device buffers and graphs for one batch bucket
struct Shape {
    // graphs come first so they drop before the buffers they point at
    encoder_graph: Option<Graph>,
    decode_graph: Option<Graph>,
    encoder: StreamWorkspace,
    prompt: Option<PromptBuffers>,
    decoder: DecodeWorkspace,
}

/// The device state calls change, behind the engine's lock
struct DeviceState {
    // shapes come first: their graphs point at the caches and states
    shapes: HashMap<usize, Shape>,
    caches: StreamCaches,
    states: DecoderStates,
}

struct Engine {
    // fields drop in order: graphs and buffers, then weights, then the context
    device: Mutex<DeviceState>,
    encoder: CudaEncoder,
    prompt: Option<PromptProjector>,
    decoder: CudaDecoder,
    /// Encoder frames per chunk, `1 + lookahead`
    frames: usize,
    /// Cached encoder frames per layer
    context: usize,
    features: usize,
    /// Slots streams may own; the slots past them pad batches
    streams: usize,
    max_batch: usize,
    runtime: CudaRuntime,
}

/// A streaming model on CUDA device 0
pub(crate) struct StreamEngine(ExclusiveTeardown<Engine>);

// the public streaming model shares one engine across threads
const _: () = {
    const fn thread_safe<T: Send + Sync>() {}
    thread_safe::<StreamEngine>();
};

impl StreamEngine {
    /// Loads a streaming checkpoint's native weights for `lookahead` encoder frames
    /// of lookahead, with slots for `streams` streams and batches of up to
    /// `max_batch` chunks
    pub(crate) fn load(
        weights: &Path,
        lookahead: usize,
        streams: usize,
        max_batch: usize,
        precision: Precision,
    ) -> Result<(Self, StreamingSpec), TranscriptionError> {
        // the quantized products exist only in the offline encoder
        #[cfg(feature = "low-precision")]
        if crate::cuda::stored::scheme(precision).is_some() {
            return Err(TranscriptionError::InvalidConfiguration(format!(
                "streaming runs at FP16, TF32 or FP32; {precision:?} is for offline \
                 transcription only"
            )));
        }
        let file = WeightFile::open(weights)?;
        let spec = file.streaming_spec()?;
        if !spec.lookaheads.contains(&lookahead) {
            return Err(TranscriptionError::InvalidConfiguration(format!(
                "{} streams with a lookahead of {:?} encoder frames, not {lookahead}",
                spec.model, spec.lookaheads
            )));
        }
        if streams == 0 || max_batch == 0 {
            return Err(TranscriptionError::InvalidConfiguration(
                "a streaming model needs room for at least one stream and one chunk".to_owned(),
            ));
        }

        let max_batch = max_batch.min(streams);
        let padding = (1..=max_batch)
            .map(|chunks| bucket(chunks, max_batch) - chunks)
            .max()
            .unwrap_or(0);
        let slots = streams + padding;
        let frames = 1 + lookahead;
        let context = spec.left_context;

        // taken before the runtime opens, so it outlives everything a failed load
        // drops on its way out, as in the offline engine
        let _exclusive = exclusive_device_work();
        let runtime = CudaRuntime::open(0, Options::default())?;
        runtime.prepare(Library::Cublas)?;
        let encoder = CudaEncoder::load(&runtime, &file, spec.encoder, math(precision))?;
        let prompt = spec
            .prompt
            .as_ref()
            .map(|prompt| PromptProjector::load(&runtime, &file, prompt, spec.encoder.d_model))
            .transpose()?;
        let decoder = CudaDecoder::load(&runtime, &file, &spec.transducer)?;
        let device = DeviceState {
            shapes: HashMap::new(),
            caches: encoder.stream_caches(&runtime, slots, context)?,
            states: decoder.states(&runtime, slots)?,
        };
        let engine = Engine {
            device: Mutex::new(device),
            encoder,
            prompt,
            decoder,
            frames,
            context,
            features: spec.encoder.features,
            streams,
            max_batch,
            runtime,
        };
        Ok((Self(ExclusiveTeardown::new(engine)), spec))
    }

    /// Encoder frames per chunk
    pub(crate) fn frames(&self) -> usize {
        self.0.frames
    }

    /// Most chunks one [`Self::run`] takes
    pub(crate) fn max_batch(&self) -> usize {
        self.0.max_batch
    }

    /// Encodes and decodes each chunk, advancing its stream's caches and predictor;
    /// returns each chunk's tokens with frames counted from the chunk's start
    ///
    /// The chunks' slots must be distinct. A failure says whether it came after the
    /// streams' device state started to change, in which case they cannot continue
    pub(crate) fn run(
        &self,
        chunks: &[EngineChunk<'_>],
    ) -> Result<Vec<RawTranscription>, RunError> {
        let engine = &*self.0;
        if chunks.is_empty() {
            return Ok(Vec::new());
        }
        if chunks.len() > engine.max_batch {
            return Err(RunError::before(TranscriptionError::InvalidConfiguration(
                format!(
                    "{} chunks for batches of at most {}",
                    chunks.len(),
                    engine.max_batch
                ),
            )));
        }
        if let Some(chunk) = chunks.iter().find(|chunk| chunk.slot >= engine.streams) {
            return Err(RunError::before(TranscriptionError::InvalidConfiguration(
                format!(
                    "stream slot {} is outside the {} stream slots",
                    chunk.slot, engine.streams
                ),
            )));
        }

        let batch = bucket(chunks.len(), engine.max_batch);
        // the lock guards device buffers a panic leaves no worse than an error does
        let mut device = engine.device.lock().unwrap_or_else(PoisonError::into_inner);
        let device = &mut *device;
        if !device.shapes.contains_key(&batch) {
            let _exclusive = exclusive_device_work();
            let shape = engine
                .shape(&device.caches, batch)
                .map_err(RunError::before)?;
            device.shapes.insert(batch, shape);
        }
        let shape = device.shapes.get_mut(&batch).ok_or_else(|| {
            RunError::before(TranscriptionError::InvalidModelOutput(
                "a streaming shape went missing".to_owned(),
            ))
        })?;
        let rows = engine
            .stage(shape, &device.caches, chunks)
            .map_err(RunError::before)?;
        engine
            .execute(
                shape,
                &mut device.caches,
                &mut device.states,
                &rows,
                chunks.len(),
            )
            .map_err(|error| RunError {
                error,
                advanced: true,
            })
    }
}

/// A failed [`StreamEngine::run`]
#[derive(Debug)]
pub(crate) struct RunError {
    pub error: TranscriptionError,
    /// Whether the chunks' streams may have advanced on the device, which leaves
    /// them out of step with their host state
    pub advanced: bool,
}

impl RunError {
    fn before(error: TranscriptionError) -> Self {
        Self {
            error,
            advanced: false,
        }
    }
}

impl Engine {
    fn shape(&self, caches: &StreamCaches, batch: usize) -> Result<Shape, TranscriptionError> {
        let encoder = self.encoder.stream_workspace(
            &self.runtime,
            caches,
            batch,
            self.frames,
            self.context,
        )?;
        let prompt = self
            .prompt
            .as_ref()
            .map(|prompt| prompt.buffers(&self.runtime, batch, self.frames))
            .transpose()?;
        let decoder = self.decoder.workspace(&self.runtime, batch, self.frames)?;
        Ok(Shape {
            encoder_graph: None,
            decode_graph: None,
            encoder,
            prompt,
            decoder,
        })
    }

    /// Queues the encoder over the chunks set in `encoder`, then the prompt
    /// projector over its output; capturable in a CUDA graph
    fn enqueue(
        &self,
        caches: &mut StreamCaches,
        encoder: &mut StreamWorkspace,
        prompt: Option<&mut PromptBuffers>,
    ) -> Result<(), TranscriptionError> {
        self.encoder
            .enqueue_stream(&self.runtime, caches, encoder)?;
        if let (Some(projector), Some(buffers)) = (&self.prompt, prompt) {
            projector.enqueue(&self.runtime, buffers, encoder.output())?;
        }
        Ok(())
    }

    /// Uploads the chunks' mel and rows, padded to the shape's batch, without
    /// touching any stream's device state; returns the rows
    fn stage(
        &self,
        shape: &mut Shape,
        caches: &StreamCaches,
        chunks: &[EngineChunk<'_>],
    ) -> Result<Vec<ChunkRow>, TranscriptionError> {
        if let Some(chunk) = chunks
            .iter()
            .find(|chunk| chunk.mel.len() % self.features != 0)
        {
            return Err(TranscriptionError::InvalidConfiguration(format!(
                "{} mel values do not make whole {}-bin frames",
                chunk.mel.len(),
                self.features
            )));
        }

        let runtime = &self.runtime;
        let batch = shape.encoder.batch();
        // a padding row is a one-frame first chunk on a spare slot, decoded over
        // no frames
        let padding = vec![0.0; self.features];
        let mut rows = Vec::with_capacity(batch);
        let mut mels = Vec::with_capacity(batch);
        for chunk in chunks {
            rows.push(ChunkRow {
                slot: chunk.slot,
                encoded: chunk.encoded,
                mel_frames: chunk.mel.len() / self.features,
            });
            mels.push(chunk.mel);
        }
        for spare in 0..batch - chunks.len() {
            rows.push(ChunkRow {
                slot: self.streams + spare,
                encoded: 0,
                mel_frames: 1,
            });
            mels.push(&padding);
        }
        self.encoder
            .set_chunks(runtime, caches, &mut shape.encoder, &rows, &mels)?;
        if let (Some(projector), Some(buffers)) = (&self.prompt, &mut shape.prompt) {
            // padding rows take prompt 0, whose output no row decodes
            let prompts: Vec<u32> = chunks
                .iter()
                .map(|chunk| chunk.prompt)
                .chain(std::iter::repeat(0))
                .take(batch)
                .collect();
            projector.set_prompts(runtime, buffers, &prompts)?;
        }
        Ok(rows)
    }

    /// Runs the staged batch, of which the first `chunks` rows are real, and
    /// returns their tokens
    fn execute(
        &self,
        shape: &mut Shape,
        caches: &mut StreamCaches,
        states: &mut DecoderStates,
        rows: &[ChunkRow],
        chunks: usize,
    ) -> Result<Vec<RawTranscription>, TranscriptionError> {
        let runtime = &self.runtime;
        // SAFETY: the graph is stored in the same `Shape` as the workspaces it
        // records, ahead of them, and the shapes drop before the caches; none of their
        // buffers is ever reallocated, and every launch is on the runtime's stream.
        // The encoder and projector return errors instead of panicking
        unsafe {
            replay_or_capture(runtime, &mut shape.encoder_graph, || {
                self.enqueue(caches, &mut shape.encoder, shape.prompt.as_mut())
            })
        }?;

        let lens: Vec<usize> = shape
            .encoder
            .output_lens()
            .iter()
            .enumerate()
            .map(|(row, len)| if row < chunks { *len } else { 0 })
            .collect();
        let decode_rows: Vec<DecodeRow> = rows
            .iter()
            .map(|row| DecodeRow {
                slot: row.slot,
                fresh: row.encoded == 0,
            })
            .collect();
        let encoder_out = match &shape.prompt {
            Some(buffers) => buffers.output(),
            None => shape.encoder.output(),
        };
        self.decoder.resume(
            runtime,
            &mut shape.decoder,
            encoder_out,
            &lens,
            states,
            &decode_rows,
        )?;
        // SAFETY: as for the encoder graph above
        unsafe {
            self.decoder
                .run_graphed(runtime, &mut shape.decoder, &mut shape.decode_graph)
        }?;
        self.decoder.save(runtime, &shape.decoder, states)?;

        let mut decoded = self.decoder.output(runtime, &mut shape.decoder)?;
        decoded.truncate(chunks);
        Ok(decoded)
    }
}
