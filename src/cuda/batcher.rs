//! Windows from any number of requests, encoded in length buckets and decoded in
//! one lane whose rows refill as they finish
//!
//! A window's mel frames pick its bucket from a fixed ladder. An encoder batch
//! pads the windows it holds to the next power of two, so each bucket captures a
//! few graphs rather than one per window count, and the decoder lane steps only
//! the rows in use. A blank row costs as much as a window, so batches that fit the
//! work waiting let a lone request run at about the speed of one run alone. cuBLAS
//! picks its kernels by problem shape, so a window's tokens can change in a rare
//! place with the windows beside it

use std::collections::HashMap;
use std::mem::ManuallyDrop;

use cudakit::Graph;
use cudakit::cudarc::driver::CudaSlice;

use crate::backend::{EncoderChunk, LanguageMask};
use crate::cuda::decoder::STEPS_PER_CHECK;
use crate::cuda::decoder::{Admission, DecodeLane};
use crate::cuda::encoder::EncoderWorkspace;
use crate::cuda::model::{
    CACHE_SHARE_DIVISOR, CudaModel, Engine, MEL_BUCKET, exclusive_device_work,
};
use crate::cuda::{capture, zeros};
use crate::error::TranscriptionError;
use crate::model::FinishedRows;

/// Bucket sizes in mel frames (10 ms each): close together for short inputs, so
/// clips of similar length share batches, and about 1.2x apart above 5 s, which
/// pads a window by at most a fifth. 3008 fits the native 30 s long-form window
/// exactly, and 4032 the 40 s single-input limit
const LADDER: [usize; 20] = [
    64, 128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2560, 3008,
    3520, 4032,
];

/// The bucket a window of `mel_frames` frames runs in, in mel frames
pub(crate) fn bucket_frames(mel_frames: usize) -> usize {
    LADDER
        .iter()
        .copied()
        .find(|bucket| *bucket >= mel_frames)
        .unwrap_or_else(|| mel_frames.max(1).div_ceil(MEL_BUCKET) * MEL_BUCKET)
}

/// Encoder buffers and graph for one bucket at one batch size
struct EncoderShape {
    // the graph drops before the buffers it points at
    graph: Option<Graph>,
    workspace: EncoderWorkspace,
    bytes: usize,
    last_used: u64,
}

/// Encoder shapes by `(bucket mel frames, batch)`, bounded by the bytes their
/// buffers hold
struct EncoderCache {
    shapes: HashMap<(usize, usize), EncoderShape>,
    bytes: usize,
    budget: usize,
    uses: u64,
}

/// Everything holding device memory, dropped under the exclusive device lock
struct Parts {
    /// Captured decode steps by span; they drop before the lane's buffers
    lane_graphs: HashMap<usize, Graph>,
    lane: DecodeLane,
    encoders: EncoderCache,
    /// One encoder batch's frames projected for the joint, sized for the largest
    /// bucket batch
    projection: CudaSlice<f32>,
}

/// Encodes windows a bucket batch at a time and decodes them in one lane
pub(crate) struct CudaBatcher {
    parts: ManuallyDrop<Parts>,
    /// Dropped after `parts`, so the engine outlives the buffers on its context
    model: CudaModel,
}

impl Drop for CudaBatcher {
    fn drop(&mut self) {
        // freeing device memory must not overlap another thread's graph capture
        let _exclusive = exclusive_device_work();
        // SAFETY: `parts` is dropped only here, and nothing reads it afterward
        unsafe { ManuallyDrop::drop(&mut self.parts) };
    }
}

impl CudaBatcher {
    /// A lane of `rows` rows for windows of up to `max_mel_frames` mel frames, whose
    /// bucket batches hold at most `max_batch` windows, decoding under `mask`
    pub(crate) fn new(
        model: &CudaModel,
        rows: usize,
        max_mel_frames: usize,
        max_batch: usize,
        mask: Option<&LanguageMask>,
    ) -> Result<Self, TranscriptionError> {
        let engine = model.engine();
        let runtime = &engine.runtime;
        let _runtime = engine.lock_runtime()?;
        let _exclusive = exclusive_device_work();

        // a probe workspace gives the encoder frames of the largest bucket
        let largest = bucket_frames(max_mel_frames);
        let frames = engine.encoder.workspace(runtime, 1, largest)?.frames();
        let lane = engine.decoder.lane(runtime, rows, frames, mask)?;
        let projection = zeros(
            runtime.stream(),
            max_batch * frames * engine.decoder.joint_hidden(),
        )?;
        let (free, _) = runtime.stream().context().mem_get_info()?;
        Ok(Self {
            parts: ManuallyDrop::new(Parts {
                lane_graphs: HashMap::new(),
                lane,
                encoders: EncoderCache {
                    shapes: HashMap::new(),
                    bytes: 0,
                    budget: free / CACHE_SHARE_DIVISOR,
                    uses: 0,
                },
                projection,
            }),
            model: model.clone(),
        })
    }

    pub(crate) fn free_rows(&self) -> usize {
        self.parts.lane.free_rows()
    }

    pub(crate) fn active_rows(&self) -> usize {
        self.parts.lane.active_rows()
    }

    /// Forgets every row in flight, after an error left the device state unknown
    pub(crate) fn reset(&mut self) {
        self.parts.lane.reset();
    }

    /// Encodes `chunks` as one batch padded to `mel_frames` and to the next power
    /// of two windows, at most `max_batch`, and starts each in a free decoder row,
    /// returning the rows in order
    pub(crate) fn admit(
        &mut self,
        mel_frames: usize,
        max_batch: usize,
        chunks: &[EncoderChunk<'_>],
    ) -> Result<Vec<usize>, TranscriptionError> {
        let engine = self.model.engine();
        if chunks.is_empty() || chunks.len() > max_batch || chunks.len() > self.free_rows() {
            return Err(TranscriptionError::InvalidConfiguration(format!(
                "{} windows for a batch of {max_batch} with {} free decoder rows",
                chunks.len(),
                self.free_rows()
            )));
        }
        let batch = chunks.len().next_power_of_two().min(max_batch);
        if let Some(chunk) = chunks.iter().find(|chunk| {
            chunk.frames() == 0
                || chunk.frames() > mel_frames
                || chunk.features().ncols() != engine.features
        }) {
            return Err(TranscriptionError::InvalidConfiguration(format!(
                "a window of {} x {} mel features does not fit a bucket of {mel_frames} x {}",
                chunk.frames(),
                chunk.features().ncols(),
                engine.features
            )));
        }

        let runtime = &engine.runtime;
        let _runtime = engine.lock_runtime()?;
        let parts = &mut *self.parts;
        let shape = parts.encoders.get(engine, mel_frames, batch)?;
        let features = engine.features;
        let row_len = mel_frames * features;
        // blank rows span the whole bucket, so no row is shorter than the encoder's
        // masks expect; their output is never read
        let lens: Vec<usize> = (0..batch)
            .map(|row| chunks.get(row).map_or(mel_frames, EncoderChunk::frames))
            .collect();
        engine
            .encoder
            .set_input(runtime, &mut shape.workspace, &lens, |mel| {
                for (row, slot) in mel.chunks_exact_mut(row_len).enumerate() {
                    let Some(chunk) = chunks.get(row) else {
                        slot.fill(0.0);
                        continue;
                    };
                    let (valid, padding) = slot.split_at_mut(chunk.frames() * features);
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
        match &shape.graph {
            Some(graph) => graph.launch()?,
            None => {
                engine.encoder.enqueue(runtime, &mut shape.workspace)?;
                let workspace = &mut shape.workspace;
                let _exclusive = exclusive_device_work();
                // SAFETY: the graph is stored in the same `EncoderShape` as the
                // workspace it records, ahead of it, so it drops first; the
                // workspace's buffers are never reallocated, and every launch is on
                // the runtime's stream. The closure returns errors instead of
                // panicking
                let graph = unsafe {
                    runtime.capture_nonempty(|| engine.encoder.enqueue(runtime, workspace))
                }?;
                shape.graph = Some(graph);
            }
        }

        // one projection per batch, at the encoder's shape
        let frames = shape.workspace.frames();
        let joint_hidden = engine.decoder.joint_hidden();
        let width = frames * joint_hidden;
        let encoded = shape.workspace.output();
        engine.decoder.project_frames(
            runtime,
            &encoded.slice(..),
            &mut parts.projection.slice_mut(..batch * width),
        )?;
        let output_lens = shape.workspace.output_lens();
        let admissions: Vec<Admission<'_>> = (0..chunks.len())
            .map(|row| Admission {
                projected: parts.projection.slice(row * width..(row + 1) * width),
                frames: output_lens[row],
            })
            .collect();
        engine.decoder.admit(runtime, &mut parts.lane, &admissions)
    }

    /// Decodes until some row may have finished, and returns the rows that have,
    /// each with its tokens or why it failed
    pub(crate) fn advance(&mut self) -> Result<FinishedRows, TranscriptionError> {
        let engine = self.model.engine();
        let (runtime, decoder) = (&engine.runtime, &engine.decoder);
        if self.active_rows() == 0 {
            return Ok(Vec::new());
        }

        let _runtime = engine.lock_runtime()?;
        let Parts {
            lane_graphs, lane, ..
        } = &mut *self.parts;
        let span = lane.span();
        let graph = lane_graphs.get(&span);
        // a span's first steps run eagerly, which also finishes cuBLAS's lazy setup
        // for its joint GEMM before the capture
        // `rows` is the lane's span, which the graph was captured at
        let finished = decoder.advance(runtime, lane, STEPS_PER_CHECK, |ws, rows| match graph {
            Some(graph) => Ok(graph.launch()?),
            None => (0..STEPS_PER_CHECK).try_for_each(|_| decoder.step_rows(runtime, ws, rows)),
        })?;
        if graph.is_none() {
            let ws = lane.workspace();
            // SAFETY: the graphs are stored in `Parts` ahead of the lane whose buffers
            // they record, so they drop first; the buffers are never reallocated, and
            // every launch is on the runtime's stream. Capture records the steps
            // without running them, so the lane's state is unchanged
            let graph = unsafe {
                capture(runtime, || {
                    (0..STEPS_PER_CHECK).try_for_each(|_| decoder.step_rows(runtime, ws, span))
                })
            }?;
            lane_graphs.insert(span, graph);
        }
        Ok(finished)
    }
}

impl EncoderCache {
    /// The encoder shape of a bucket, allocated on first use; evicts the least
    /// recently used others while over budget
    fn get(
        &mut self,
        engine: &Engine,
        mel_frames: usize,
        batch: usize,
    ) -> Result<&mut EncoderShape, TranscriptionError> {
        let key = (mel_frames, batch);
        self.uses += 1;
        let uses = self.uses;
        if !self.shapes.contains_key(&key) {
            // adding a shape may free others' graphs and buffers, by eviction or by the
            // retry below, and a failed allocation frees what it had made
            let _exclusive = exclusive_device_work();
            let workspace = match engine.encoder.workspace(&engine.runtime, batch, mel_frames) {
                Ok(workspace) => workspace,
                // the allocation may have failed on memory cached shapes hold: free
                // them all and try once more
                Err(_) => {
                    self.shapes.clear();
                    self.bytes = 0;
                    engine
                        .encoder
                        .workspace(&engine.runtime, batch, mel_frames)?
                }
            };
            let bytes = workspace.bytes();
            self.bytes += bytes;
            self.shapes.insert(
                key,
                EncoderShape {
                    graph: None,
                    workspace,
                    bytes,
                    last_used: uses,
                },
            );
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

        let shape = self.shapes.get_mut(&key).ok_or_else(|| {
            TranscriptionError::InvalidModelOutput("a cached CUDA shape went missing".to_owned())
        })?;
        shape.last_used = uses;
        Ok(shape)
    }
}
