//! Cache-aware streaming on the [`CudaEncoder`]: one chunk from each of a batch of
//! streams through the causal subsampling and every layer, with each stream's
//! attention and conv caches in slot arenas
//!
//! NeMo's streaming loop (`CacheAwareStreamingAudioBuffer` and
//! `conformer_stream_step`) feeds a stream's first chunk `1 + 8 * lookahead` mel
//! frames, and every later chunk the [`MEL_CONTEXT`] mel frames before it plus `8 *
//! (1 + lookahead)` new ones, dropping the first [`DROP`] subsampled frames those
//! context frames produce. Here every row has the later chunks' shape and a first
//! chunk starts [`FIRST_OFFSET`] frames in, which puts its first subsampled frame at
//! [`DROP`], so one drop serves every row and one CUDA graph every mix of first and
//! later chunks
//!
//! A stream's caches advance by one chunk of encoder frames each step, so its ring
//! position and cached frame count follow from the frames it has encoded

use cudakit::cudarc::driver::{CudaSlice, CudaView};

use super::{
    Activations, Attention, ConvModule, CudaEncoder, NORM_EPS, causal_subsampled, gemm_input,
    gemm_output, linear, merged_in, normed_in, position_rows,
};
use crate::cuda::kernels::{CudaRuntime, FeatureMap, StreamGeometry, StreamRows};
use crate::cuda::{Direction, HostStaging, zeros};
use crate::error::TranscriptionError;
use crate::native::spec::ConvNorm;

/// Mel frames before each chunk after a stream's first that NeMo encodes again
pub(crate) const MEL_CONTEXT: usize = 9;

/// Subsampled frames of [`MEL_CONTEXT`] that NeMo drops from each later chunk
const DROP: usize = 2;

/// Where a stream's first chunk starts in the chunk input: three halvings put its
/// first subsampled frame at [`DROP`]
const FIRST_OFFSET: usize = DROP * 8;

/// Mel frames in a stream's first chunk of `frames` encoder frames
pub(crate) fn first_chunk_mel(frames: usize) -> usize {
    8 * frames - 7
}

/// Mel frames in a later chunk of `frames` encoder frames, its context included
pub(crate) fn chunk_mel(frames: usize) -> usize {
    MEL_CONTEXT + 8 * frames
}

/// Every stream's attention and conv caches, one slot per stream
pub(crate) struct StreamCaches {
    slots: usize,
    /// `[slots, layers, 2, context, d_model]` projected keys and values
    kv: CudaSlice<f32>,
    /// `[slots, layers, conv_kernel - 1, d_model]` GLU frames
    conv: CudaSlice<f32>,
}

/// One stream's chunk in a batch
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChunkRow {
    /// The stream's slot in the [`StreamCaches`]
    pub slot: usize,
    /// Encoder frames of the stream already encoded, 0 for its first chunk
    pub encoded: usize,
    /// Valid mel frames of this chunk: for a later chunk including its
    /// [`MEL_CONTEXT`] frames
    pub mel_frames: usize,
}

impl ChunkRow {
    /// Each subsampling stage's valid `[start, end)` frames, mel first, and the
    /// encoder frames the chunk yields
    fn windows(self) -> ([(usize, usize); 4], usize) {
        let mut start = if self.encoded == 0 { FIRST_OFFSET } else { 0 };
        let mut len = self.mel_frames;
        let mut windows = [(start, start + len); 4];
        for window in windows.iter_mut().skip(1) {
            start /= 2;
            len = causal_subsampled(len);
            *window = (start, start + len);
        }
        let frames = windows[3].1.saturating_sub(DROP);
        (windows, frames)
    }
}

/// Device buffers for a batch of `batch` chunks of `frames` encoder frames
pub(crate) struct StreamWorkspace {
    geometry: StreamGeometry,
    /// `[batch, chunk_mel(frames), features]`, time-major
    mel: CudaSlice<f32>,
    mel_host: HostStaging<f32>,
    /// Per-row state, `[12, batch]`: the four stages' `[batch, 2]` windows, then
    /// slots, ring starts, cached frames and valid frames
    rows: CudaSlice<u32>,
    rows_host: HostStaging<u32>,
    /// Encoder frames each row yields
    host_valid: Vec<usize>,
    /// Subsampling feature maps, ping-ponged as in the offline workspace
    maps: [CudaSlice<f32>; 2],
    flat: CudaSlice<f32>,
    act: Activations,
    /// Every layer's projected relative positions for the chunk's window,
    /// `[layers, context + 2 * frames - 1, d_model]`
    pos: CudaSlice<f32>,
}

impl StreamWorkspace {
    /// Chunks per batch
    pub(crate) fn batch(&self) -> usize {
        self.geometry.batch
    }

    /// Encoder frames each row of the current batch yields
    pub(crate) fn output_lens(&self) -> &[usize] {
        &self.host_valid
    }

    /// The encoder output, `[batch * frames, d_model]`
    pub(crate) fn output(&self) -> &CudaSlice<f32> {
        &self.act.x
    }
}

/// Views of `rows`, a workspace's per-row state, for the kernels: the slots, ring
/// starts, cached frames and valid frames
fn row_state(rows: &CudaSlice<u32>, batch: usize) -> [CudaView<'_, u32>; 4] {
    [0, 1, 2, 3].map(|index| rows.slice((8 + index) * batch..(9 + index) * batch))
}

impl CudaEncoder {
    fn check_streaming(&self) -> Result<(), TranscriptionError> {
        if !self.spec.causal || self.spec.conv_norm != ConvNorm::Layer {
            return Err(TranscriptionError::InvalidConfiguration(
                "the encoder is not a cache-aware streaming one".to_owned(),
            ));
        }
        Ok(())
    }

    /// Zeroed caches for `slots` streams with `context` cached frames per layer
    pub(crate) fn stream_caches(
        &self,
        runtime: &CudaRuntime,
        slots: usize,
        context: usize,
    ) -> Result<StreamCaches, TranscriptionError> {
        self.check_streaming()?;
        let geometry = self.geometry(slots, 1, context, slots);
        let stream = runtime.stream();
        Ok(StreamCaches {
            slots,
            kv: zeros(stream, geometry.kv_len()?)?,
            conv: zeros(stream, geometry.conv_len(self.spec.conv_kernel)?)?,
        })
    }

    fn geometry(
        &self,
        batch: usize,
        frames: usize,
        context: usize,
        slots: usize,
    ) -> StreamGeometry {
        StreamGeometry {
            batch,
            frames,
            heads: self.spec.heads,
            head_dim: self.spec.d_model / self.spec.heads,
            context,
            layers: self.layers.len(),
            slots,
        }
    }

    /// The causal subsampling's feature maps for `batch` chunk inputs of `mel_frames`
    fn stream_maps(&self, batch: usize, mel_frames: usize) -> [FeatureMap; 4] {
        let channels = self.spec.subsampling_channels;
        let mel = FeatureMap {
            batch,
            channels: 1,
            frames: mel_frames,
            features: self.spec.features,
        };
        let next = |map: FeatureMap| FeatureMap {
            batch,
            channels,
            frames: causal_subsampled(map.frames),
            features: causal_subsampled(map.features),
        };
        let first = next(mel);
        let second = next(first);
        [mel, first, second, next(second)]
    }

    /// Allocates the buffers for `batch` chunks of `frames` encoder frames against
    /// `caches`, and projects every layer's relative positions for the window
    pub(crate) fn stream_workspace(
        &self,
        runtime: &CudaRuntime,
        caches: &StreamCaches,
        batch: usize,
        frames: usize,
        context: usize,
    ) -> Result<StreamWorkspace, TranscriptionError> {
        self.check_streaming()?;
        // whole chunks fill the cache, so every chunk's window is the same shape and
        // chunked_limited attention sees all of it
        if frames == 0
            || !context.is_multiple_of(frames)
            || caches.kv.len()
                != self
                    .geometry(caches.slots, frames, context, caches.slots)
                    .kv_len()?
        {
            return Err(TranscriptionError::InvalidConfiguration(format!(
                "a {frames}-frame chunk does not tile the {context}-frame cache"
            )));
        }

        let stream = runtime.stream();
        let spec = self.spec;
        let d = spec.d_model;
        let geometry = self.geometry(batch, frames, context, caches.slots);
        let mel_frames = chunk_mel(frames);
        let maps = self.stream_maps(batch, mel_frames);
        let rows = batch * frames;

        // positions `context + frames - 1` down to `-(frames - 1)` cover every
        // query and key of the window
        let positions = context + 2 * frames - 1;
        let table = stream.clone_htod(&position_rows(context + frames - 1, positions, d))?;
        let table_f16 = self.fp16_copy(runtime, &table)?;
        let mut pos = zeros(stream, self.layers.len() * positions * d)?;
        for (index, layer) in self.layers.iter().enumerate() {
            let table = gemm_input(
                self.math,
                table.slice(..),
                table_f16.as_ref().map(|table| table.slice(..)),
            )?;
            self.linear(
                runtime,
                linear(positions, d, d, 1.0, 0.0, self.math.single()),
                table,
                &layer.attention.pos,
                &mut pos.slice_mut(index * positions * d..(index + 1) * positions * d),
            )?;
        }

        Ok(StreamWorkspace {
            geometry,
            mel: zeros(stream, maps[0].len())?,
            mel_host: HostStaging::new(stream, maps[0].len(), Direction::Upload)?,
            rows: zeros(stream, 12 * batch)?,
            rows_host: HostStaging::new(stream, 12 * batch, Direction::Upload)?,
            host_valid: vec![0; batch],
            maps: [zeros(stream, maps[1].len())?, zeros(stream, maps[2].len())?],
            flat: zeros(stream, rows * spec.subsampling_channels * maps[3].features)?,
            act: Activations::new(runtime, spec, self.math, rows)?,
            pos,
        })
    }

    /// Queues the upload of one chunk per row: `mels[b]` holds row `b`'s
    /// `rows[b].mel_frames` valid mel frames, time-major
    ///
    /// Each row's slot must be distinct and inside `caches`, which the kernels rely
    /// on to index the arenas
    pub(crate) fn set_chunks(
        &self,
        runtime: &CudaRuntime,
        caches: &StreamCaches,
        ws: &mut StreamWorkspace,
        rows: &[ChunkRow],
        mels: &[&[f32]],
    ) -> Result<(), TranscriptionError> {
        let batch = ws.geometry.batch;
        let (frames, context) = (ws.geometry.frames, ws.geometry.context);
        let features = self.spec.features;
        let mel_frames = chunk_mel(frames);
        let invalid = |message: String| Err(TranscriptionError::InvalidConfiguration(message));
        if rows.len() != batch || mels.len() != batch {
            return invalid(format!(
                "{} chunks for a {batch}-chunk workspace",
                rows.len()
            ));
        }
        let mut seen = vec![false; caches.slots];
        for (row, mel) in rows.iter().zip(mels) {
            let longest = if row.encoded == 0 {
                first_chunk_mel(frames)
            } else {
                mel_frames
            };
            // a later chunk needs one mel frame past its context to yield a frame
            let shortest = if row.encoded == 0 { 1 } else { MEL_CONTEXT + 1 };
            if row.mel_frames < shortest || row.mel_frames > longest {
                return invalid(format!(
                    "a chunk of {} mel frames is outside {shortest}..={longest}",
                    row.mel_frames
                ));
            }
            if mel.len() != row.mel_frames * features {
                return invalid(format!(
                    "a chunk of {} values for {} mel frames",
                    mel.len(),
                    row.mel_frames
                ));
            }
            if row.encoded % frames != 0 {
                return invalid(format!(
                    "a stream that encoded {} frames ended an earlier chunk short",
                    row.encoded
                ));
            }
            match seen.get_mut(row.slot) {
                Some(seen) if !*seen => *seen = true,
                _ => {
                    return invalid(format!(
                        "slot {} is repeated or outside the {} cache slots",
                        row.slot, caches.slots
                    ));
                }
            }
        }

        let stream = runtime.stream();
        let input = ws.mel_host.write()?;
        input.fill(0.0);
        let state = ws.rows_host.write()?;
        for (b, (row, mel)) in rows.iter().zip(mels).enumerate() {
            let (windows, valid) = row.windows();
            let start = (b * mel_frames + windows[0].0) * features;
            input[start..start + mel.len()].copy_from_slice(mel);
            for (stage, (begin, end)) in windows.into_iter().enumerate() {
                state[2 * (stage * batch + b)] = begin as u32;
                state[2 * (stage * batch + b) + 1] = end as u32;
            }
            let ring = row.encoded % context;
            let cached = row.encoded.min(context);
            for (index, value) in [row.slot, ring, cached, valid].into_iter().enumerate() {
                state[(8 + index) * batch + b] = value as u32;
            }
            ws.host_valid[b] = valid;
        }
        ws.mel_host.upload(stream, &mut ws.mel)?;
        ws.rows_host.upload(stream, &mut ws.rows)?;
        Ok(())
    }

    /// Queues the whole encoder over the chunks [`Self::set_chunks`] uploaded,
    /// reading and advancing their streams' caches
    pub(crate) fn enqueue_stream(
        &self,
        runtime: &CudaRuntime,
        caches: &mut StreamCaches,
        ws: &mut StreamWorkspace,
    ) -> Result<(), TranscriptionError> {
        self.enqueue_stream_subsampling(runtime, ws)?;
        for index in 0..self.layers.len() {
            self.enqueue_stream_layer(runtime, caches, ws, index)?;
        }
        Ok(())
    }

    /// Causal `pre_encode` over each row's window, then the drop and the projection
    fn enqueue_stream_subsampling(
        &self,
        runtime: &CudaRuntime,
        ws: &mut StreamWorkspace,
    ) -> Result<(), TranscriptionError> {
        let stream = runtime.stream();
        let kernels = &self.stream_kernels;
        let sub = &self.subsampling;
        let geometry = ws.geometry;
        let [mel_map, first, second, third] =
            self.stream_maps(geometry.batch, chunk_mel(geometry.frames));
        let channels = self.spec.subsampling_channels;
        let StreamWorkspace {
            mel,
            rows,
            maps: [large, small],
            flat,
            act,
            ..
        } = ws;
        let batch = geometry.batch;
        let windows = |stage: usize| rows.slice(2 * stage * batch..2 * (stage + 1) * batch);

        kernels.conv_in(
            stream,
            &mel.slice(..),
            &windows(0),
            &windows(1),
            &sub.conv_in.0.data().slice(..),
            &sub.conv_in.1.data().slice(..),
            &mut large.slice_mut(..),
            mel_map,
            first,
        )?;
        kernels.depthwise(
            stream,
            &large.slice(..),
            &windows(2),
            &sub.depthwise1.0.data().slice(..),
            &sub.depthwise1.1.data().slice(..),
            &mut small.slice_mut(..),
            first,
            second,
        )?;
        self.pointwise_gemm(runtime, &sub.pointwise1.0, small, large, second)?;
        kernels.bias_relu(
            stream,
            &mut large.slice_mut(..second.len()),
            &sub.pointwise1.1.data().slice(..),
            &windows(2),
            second,
        )?;
        kernels.depthwise(
            stream,
            &large.slice(..second.len()),
            &windows(3),
            &sub.depthwise2.0.data().slice(..),
            &sub.depthwise2.1.data().slice(..),
            &mut small.slice_mut(..third.len()),
            second,
            third,
        )?;
        self.pointwise_gemm(runtime, &sub.pointwise2.0, small, large, third)?;
        kernels.bias_relu(
            stream,
            &mut large.slice_mut(..third.len()),
            &sub.pointwise2.1.data().slice(..),
            &windows(3),
            third,
        )?;
        kernels.flatten(
            stream,
            &large.slice(..third.len()),
            &mut flat.slice_mut(..),
            third,
            geometry.frames,
            DROP,
        )?;

        let (weight, bias) = &sub.out;
        self.kernels
            .fill_rows(stream, &bias.data().slice(..), &mut act.x.slice_mut(..))?;
        runtime.sgemm(
            linear(
                act.rows,
                self.spec.d_model,
                channels * third.features,
                1.0,
                1.0,
                self.math.single(),
            ),
            flat,
            weight.data(),
            &mut act.x,
        )?;
        Ok(())
    }

    /// One Conformer layer on the chunks, as [`Self::enqueue_layer`] with the cached
    /// attention and conv modules
    fn enqueue_stream_layer(
        &self,
        runtime: &CudaRuntime,
        caches: &mut StreamCaches,
        ws: &mut StreamWorkspace,
        index: usize,
    ) -> Result<(), TranscriptionError> {
        let layer = &self.layers[index];
        if index == 0 {
            self.norm(runtime, &mut ws.act, &layer.norm_ff1)?;
        }
        self.feed_forward(runtime, &mut ws.act, &layer.ff1)?;
        self.norm(runtime, &mut ws.act, &layer.norm_att)?;
        self.stream_attention(runtime, caches, ws, &layer.attention, index)?;
        self.norm(runtime, &mut ws.act, &layer.norm_conv)?;
        self.stream_conv(runtime, caches, ws, &layer.conv, index)?;
        self.close_layer(runtime, &mut ws.act, index)
    }

    /// `x += linear_out(attention(normed))` over each stream's cached frames and its
    /// chunk, then the chunk's keys and values into the cache
    fn stream_attention(
        &self,
        runtime: &CudaRuntime,
        caches: &mut StreamCaches,
        ws: &mut StreamWorkspace,
        weights: &Attention,
        index: usize,
    ) -> Result<(), TranscriptionError> {
        let stream = runtime.stream();
        let geometry = ws.geometry;
        let d = self.spec.d_model;
        let math = self.math.single();
        let rows_count = ws.act.rows;
        let positions = geometry.context + 2 * geometry.frames - 1;
        self.linear(
            runtime,
            linear(rows_count, 3 * d, d, 1.0, 0.0, math),
            normed_in(self.math, &ws.act.normed, ws.act.fp16.as_ref())?,
            &weights.qkv,
            &mut ws.act.qkv,
        )?;

        let state = row_state(&ws.rows, geometry.batch);
        let rows = StreamRows {
            slots: &state[0],
            rings: &state[1],
            cached: &state[2],
            valid: &state[3],
        };
        let pos = ws
            .pos
            .slice(index * positions * d..(index + 1) * positions * d);
        let act = &mut ws.act;
        self.stream_kernels.attention(
            stream,
            &act.qkv.slice(..),
            &caches.kv.slice(..),
            &pos,
            &weights.bias_u.data().slice(..),
            &weights.bias_v.data().slice(..),
            &rows,
            gemm_output(
                self.math,
                &mut act.merged,
                act.fp16.as_mut().map(|fp16| &mut fp16.merged),
            )?,
            geometry,
            index,
        )?;
        self.stream_kernels.kv_append(
            stream,
            &act.qkv.slice(..),
            &mut caches.kv.slice_mut(..),
            &rows,
            geometry,
            index,
        )?;
        self.linear(
            runtime,
            linear(rows_count, d, d, 1.0, 1.0, math),
            merged_in(self.math, &act.merged, act.fp16.as_ref())?,
            &weights.out,
            &mut act.x,
        )?;
        Ok(())
    }

    /// `x += pointwise2(silu(layer_norm(causal_depthwise(glu(pointwise1(normed))))))`
    /// with each stream's cached GLU frames, then the chunk's into the cache
    fn stream_conv(
        &self,
        runtime: &CudaRuntime,
        caches: &mut StreamCaches,
        ws: &mut StreamWorkspace,
        weights: &ConvModule,
        index: usize,
    ) -> Result<(), TranscriptionError> {
        let stream = runtime.stream();
        let geometry = ws.geometry;
        let d = self.spec.d_model;
        let math = self.math.single();
        let rows_count = ws.act.rows;
        let pw1_len = rows_count * 2 * d;
        self.linear(
            runtime,
            linear(rows_count, 2 * d, d, 1.0, 0.0, math),
            normed_in(self.math, &ws.act.normed, ws.act.fp16.as_ref())?,
            &weights.pointwise1,
            &mut ws.act.hidden.slice_mut(..pw1_len),
        )?;

        let state = row_state(&ws.rows, geometry.batch);
        let rows = StreamRows {
            slots: &state[0],
            rings: &state[1],
            cached: &state[2],
            valid: &state[3],
        };
        let act = &mut ws.act;
        self.stream_kernels.conv(
            stream,
            &act.hidden.slice(..pw1_len),
            &caches.conv.slice(..),
            &rows,
            &weights.depthwise.data().slice(..),
            (
                &weights.scale.data().slice(..),
                &weights.shift.data().slice(..),
            ),
            gemm_output(
                self.math,
                &mut act.merged,
                act.fp16.as_mut().map(|fp16| &mut fp16.merged),
            )?,
            geometry,
            self.spec.conv_kernel,
            index,
            NORM_EPS,
        )?;
        self.stream_kernels.conv_cache(
            stream,
            &act.hidden.slice(..pw1_len),
            &mut caches.conv.slice_mut(..),
            &rows,
            geometry,
            self.spec.conv_kernel,
            index,
        )?;
        self.linear(
            runtime,
            linear(rows_count, d, d, 1.0, 1.0, math),
            merged_in(self.math, &act.merged, act.fp16.as_ref())?,
            &weights.pointwise2,
            &mut act.x,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use cudakit::{Library, Math, Options};
    use ndarray::{ArrayD, Ix2};

    use super::{ChunkRow, StreamCaches, StreamWorkspace};
    use crate::cuda::encoder::{CudaEncoder, EncoderMath};
    use crate::cuda::kernels::CudaRuntime;
    use crate::cuda::test_support::compare;
    use crate::cuda::test_support::streaming_weights;
    use crate::test_support::NemoReference;

    /// Utterances `stream_ref.py --dump` wrote
    const DUMPED: usize = 2;

    /// One dumped chunk: its time-major mel and NeMo's output of every layer
    struct Chunk {
        mel: Vec<f32>,
        mel_frames: usize,
        layers: Vec<ArrayD<f32>>,
    }

    fn chunks(reference: &NemoReference, utterance: usize, layers: usize) -> Vec<Chunk> {
        let count = reference.i64(&format!("utt{utterance}.chunks"))[0] as usize;
        (0..count)
            .map(|c| {
                let key = |name: &str| format!("utt{utterance}.chunk{c}.{name}");
                let mel = reference
                    .f32(&key("mel"))
                    .into_dimensionality::<Ix2>()
                    .expect("[features, frames]");
                let mel_frames = reference.i64(&key("mel_len"))[0] as usize;
                assert_eq!(mel.shape()[1], mel_frames, "NeMo chunks carry no padding");
                // NeMo's buffer is feature-major; the encoder takes time-major rows
                let mel = mel.t().iter().copied().collect();
                let layers = (0..layers)
                    .map(|index| {
                        let out = reference.f32(&key(&format!("layers.{index}.out")));
                        let shape = out.shape().to_vec();
                        out.into_shape_with_order(vec![1, shape[0], shape[1]])
                            .expect("[1, frames, d_model]")
                    })
                    .collect();
                Chunk {
                    mel,
                    mel_frames,
                    layers,
                }
            })
            .collect()
    }

    /// The first `frames` output rows of batch row `b`, which NeMo produced
    fn row_output(ws: &StreamWorkspace, output: &[f32], b: usize, frames: usize) -> Vec<f32> {
        let width = output.len() / (ws.batch() * ws.geometry.frames);
        let start = b * ws.geometry.frames * width;
        output[start..start + frames * width].to_vec()
    }

    /// Streams the dumped utterances one at a time, holding every layer of every
    /// chunk against NeMo's cache-aware streaming in FP32, then both at once in one
    /// batch, which must give the same outputs
    ///
    /// Needs a CUDA GPU: run it through the shared queue with `just gpu-parity`
    #[test]
    #[ignore = "needs a CUDA GPU and the NeMo streaming references; `just gpu-parity` runs it on gpu0"]
    fn gpu_stream_encoder_matches_nemo_layer_by_layer() {
        let runtime = CudaRuntime::open(0, Options::default()).expect("CUDA runtime");
        runtime.prepare(Library::Cublas).expect("cuBLAS");
        let file = streaming_weights();
        let spec = file.streaming_spec().expect("streaming spec");
        let encoder = CudaEncoder::load(
            &runtime,
            &file,
            spec.encoder,
            EncoderMath::Single(Math::Fp32),
        )
        .expect("encoder");
        let context = spec.left_context;
        let download =
            |ws: &StreamWorkspace| runtime.stream().clone_dtoh(ws.output()).expect("download");

        let mut worst = 0f64;
        for lookahead in [13, 1] {
            let frames = lookahead + 1;
            let reference = NemoReference::streaming("nemotron-en", lookahead);
            let utterances: Vec<Vec<Chunk>> = (0..DUMPED)
                .map(|utterance| chunks(&reference, utterance, spec.encoder.layers))
                .collect();
            let mut caches: StreamCaches = encoder
                .stream_caches(&runtime, DUMPED, context)
                .expect("caches");
            let mut single = encoder
                .stream_workspace(&runtime, &caches, 1, frames, context)
                .expect("workspace");

            for (utterance, chunks) in utterances.iter().enumerate() {
                let mut encoded = 0;
                for (c, chunk) in chunks.iter().enumerate() {
                    let row = ChunkRow {
                        slot: utterance,
                        encoded,
                        mel_frames: chunk.mel_frames,
                    };
                    encoder
                        .set_chunks(&runtime, &caches, &mut single, &[row], &[&chunk.mel])
                        .expect("chunk");
                    let valid = single.output_lens()[0];
                    let expected = chunk.layers[0].shape()[1];
                    assert_eq!(
                        valid, expected,
                        "r{lookahead} utt{utterance} chunk {c} frames"
                    );

                    encoder
                        .enqueue_stream_subsampling(&runtime, &mut single)
                        .expect("subsampling");
                    for (index, layer) in chunk.layers.iter().enumerate() {
                        encoder
                            .enqueue_stream_layer(&runtime, &mut caches, &mut single, index)
                            .expect("layer");
                        let ours = row_output(&single, &download(&single), 0, valid);
                        let (rel, max_abs) = compare(&ours, layer, &[valid]);
                        if index + 1 == chunk.layers.len() || rel > 1e-4 {
                            println!(
                                "r{lookahead} utt{utterance} chunk {c:2} layer {index:2} rel {rel:.2e} max {max_abs:.2e}"
                            );
                        }
                        if index + 1 == chunk.layers.len() {
                            worst = worst.max(rel);
                        }
                    }
                    encoded += valid;
                }
            }

            // both streams in one batch from fresh caches, then the longer one alone
            let mut caches = encoder
                .stream_caches(&runtime, DUMPED, context)
                .expect("caches");
            let mut pair = encoder
                .stream_workspace(&runtime, &caches, DUMPED, frames, context)
                .expect("workspace");
            let mut encoded = [0; DUMPED];
            let longest = utterances.iter().map(Vec::len).max().expect("utterances");
            for c in 0..longest {
                let live: Vec<usize> = (0..DUMPED).filter(|u| c < utterances[*u].len()).collect();
                let ws = if live.len() == DUMPED {
                    &mut pair
                } else {
                    &mut single
                };
                let rows: Vec<ChunkRow> = live
                    .iter()
                    .map(|&u| ChunkRow {
                        slot: u,
                        encoded: encoded[u],
                        mel_frames: utterances[u][c].mel_frames,
                    })
                    .collect();
                let mels: Vec<&[f32]> = live
                    .iter()
                    .map(|&u| utterances[u][c].mel.as_slice())
                    .collect();
                encoder
                    .set_chunks(&runtime, &caches, ws, &rows, &mels)
                    .expect("chunks");
                encoder
                    .enqueue_stream(&runtime, &mut caches, ws)
                    .expect("encoder");
                let output = download(ws);
                for (b, &u) in live.iter().enumerate() {
                    let valid = ws.output_lens()[b];
                    let last = utterances[u][c].layers.last().expect("layers");
                    let (rel, max_abs) =
                        compare(&row_output(ws, &output, b, valid), last, &[valid]);
                    if rel > 1e-4 {
                        println!(
                            "r{lookahead} batched utt{u} chunk {c:2} rel {rel:.2e} max {max_abs:.2e}"
                        );
                    }
                    worst = worst.max(rel);
                    encoded[u] += valid;
                }
            }
        }

        // NeMo FP32 with TF32 off against cuBLAS FP32, as the offline encoder
        assert!(worst < 1e-4, "streaming encoder relative L2 {worst:.2e}");
    }
}
