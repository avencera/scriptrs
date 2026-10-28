//! FastConformer encoder on cuBLAS plus the `encoder` kernel area
//!
//! The forward pass follows NeMo's `ConformerEncoder` operation by operation in FP32,
//! so each stage can be held against the reference activations. Everything a pass
//! touches lives in an [`EncoderWorkspace`] allocated once per input shape, which
//! keeps [`CudaEncoder::enqueue`] free of allocations and host syncs and therefore
//! capturable in a CUDA graph

use std::sync::Arc;

use cudakit::cudarc::driver::{CudaSlice, CudaStream};
use cudakit::{DeviceTensor, Error, Math, Sgemm, SgemmBatched};

use crate::cuda::kernels::{CudaRuntime, EncoderKernels, FeatureMap, Heads};
use crate::error::TranscriptionError;
use crate::native::spec::EncoderSpec;
use crate::native::weights::WeightFile;

/// PyTorch's default `eps` for `LayerNorm` and `BatchNorm1d`, which NeMo keeps
const NORM_EPS: f32 = 1e-5;

/// Every `dw_striding` convolution is 3x3 with stride 2 and padding 1
fn subsampled(len: usize) -> usize {
    // NeMo's `calculate_conv_output_size`: (len + 1 + 1 - 3) // 2 + 1
    (len + 2).saturating_sub(3) / 2 + 1
}

struct Norm {
    weight: DeviceTensor,
    bias: DeviceTensor,
}

struct FeedForward {
    linear1: DeviceTensor,
    linear2: DeviceTensor,
}

struct Attention {
    /// `linear_q`, `linear_k` and `linear_v` stacked to `[3 * d_model, d_model]`
    qkv: DeviceTensor,
    pos: DeviceTensor,
    out: DeviceTensor,
    bias_u: DeviceTensor,
    bias_v: DeviceTensor,
}

struct ConvModule {
    pointwise1: DeviceTensor,
    depthwise: DeviceTensor,
    /// Eval-mode BatchNorm folded to `x * scale + shift`
    scale: DeviceTensor,
    shift: DeviceTensor,
    pointwise2: DeviceTensor,
}

struct Layer {
    norm_ff1: Norm,
    ff1: FeedForward,
    norm_att: Norm,
    attention: Attention,
    norm_conv: Norm,
    conv: ConvModule,
    norm_ff2: Norm,
    ff2: FeedForward,
    norm_out: Norm,
}

struct Subsampling {
    conv_in: (DeviceTensor, DeviceTensor),
    depthwise1: (DeviceTensor, DeviceTensor),
    pointwise1: (DeviceTensor, DeviceTensor),
    depthwise2: (DeviceTensor, DeviceTensor),
    pointwise2: (DeviceTensor, DeviceTensor),
    out: (DeviceTensor, DeviceTensor),
}

/// Encoder weights on the GPU and the kernels that run them
pub(crate) struct CudaEncoder {
    spec: EncoderSpec,
    subsampling: Subsampling,
    layers: Vec<Layer>,
    kernels: EncoderKernels,
}

/// Device buffers for one `(batch, mel frames)` input shape
pub(crate) struct EncoderWorkspace {
    batch: usize,
    mel_frames: usize,
    /// Encoder frames after 8x subsampling
    frames: usize,
    /// `[batch, mel_frames, features]`
    mel: CudaSlice<f32>,
    /// Valid frames per batch row before and after each subsampling stage
    lens: [CudaSlice<u32>; 4],
    host_lens: Vec<usize>,
    /// Subsampling feature maps, ping-ponged: `[0]` fits the first stage, `[1]` the
    /// second
    maps: [CudaSlice<f32>; 2],
    flat: CudaSlice<f32>,
    /// The residual stream, `[batch * frames, d_model]`
    x: CudaSlice<f32>,
    normed: CudaSlice<f32>,
    hidden: CudaSlice<f32>,
    qkv: CudaSlice<f32>,
    q_u: CudaSlice<f32>,
    q_v: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    pos_emb: CudaSlice<f32>,
    pos: CudaSlice<f32>,
    pos_heads: CudaSlice<f32>,
    scores: CudaSlice<f32>,
    bd: CudaSlice<f32>,
    context: CudaSlice<f32>,
    merged: CudaSlice<f32>,
}

impl EncoderWorkspace {
    pub(crate) fn frames(&self) -> usize {
        self.frames
    }

    /// Valid encoder frames per batch row of the current input
    pub(crate) fn output_lens(&self) -> &[usize] {
        &self.host_lens
    }

    /// The encoder output, `[batch * frames, d_model]`
    pub(crate) fn output(&self) -> &CudaSlice<f32> {
        &self.x
    }
}

fn upload(
    runtime: &CudaRuntime,
    file: &WeightFile,
    name: &str,
    shape: &[usize],
) -> Result<DeviceTensor, TranscriptionError> {
    let values = file.tensor(name, shape)?;
    Ok(DeviceTensor::upload(runtime.stream(), values, shape)?)
}

fn upload_host(runtime: &CudaRuntime, values: &[f32]) -> Result<DeviceTensor, TranscriptionError> {
    Ok(DeviceTensor::upload(
        runtime.stream(),
        values,
        &[values.len()],
    )?)
}

fn zeros<T>(stream: &Arc<CudaStream>, len: usize) -> Result<CudaSlice<T>, Error>
where
    T: cudakit::cudarc::driver::DeviceRepr + cudakit::cudarc::driver::ValidAsZeroBits,
{
    Ok(stream.alloc_zeros(len)?)
}

impl CudaEncoder {
    /// Uploads the encoder weights and loads the kernels
    pub(crate) fn load(
        runtime: &CudaRuntime,
        file: &WeightFile,
        spec: EncoderSpec,
    ) -> Result<Self, TranscriptionError> {
        let d = spec.d_model;
        let channels = spec.subsampling_channels;
        let flat = channels * spec.features / spec.subsampling_factor;
        let pair = |name: &str, shape: &[usize]| -> Result<_, TranscriptionError> {
            Ok((
                upload(
                    runtime,
                    file,
                    &format!("encoder.pre_encode.{name}.weight"),
                    shape,
                )?,
                upload(
                    runtime,
                    file,
                    &format!("encoder.pre_encode.{name}.bias"),
                    &shape[..1],
                )?,
            ))
        };
        let subsampling = Subsampling {
            conv_in: pair("conv.0", &[channels, 1, 3, 3])?,
            depthwise1: pair("conv.2", &[channels, 1, 3, 3])?,
            pointwise1: pair("conv.3", &[channels, channels, 1, 1])?,
            depthwise2: pair("conv.5", &[channels, 1, 3, 3])?,
            pointwise2: pair("conv.6", &[channels, channels, 1, 1])?,
            out: pair("out", &[d, flat])?,
        };

        let mut layers = Vec::with_capacity(spec.layers);
        for index in 0..spec.layers {
            layers.push(Self::load_layer(runtime, file, spec, index)?);
        }

        Ok(Self {
            spec,
            subsampling,
            layers,
            kernels: EncoderKernels::load(runtime)?,
        })
    }

    fn load_layer(
        runtime: &CudaRuntime,
        file: &WeightFile,
        spec: EncoderSpec,
        index: usize,
    ) -> Result<Layer, TranscriptionError> {
        let d = spec.d_model;
        let prefix = format!("encoder.layers.{index}");
        let tensor = |name: &str, shape: &[usize]| file.tensor(&format!("{prefix}.{name}"), shape);
        let load =
            |name: &str, shape: &[usize]| upload(runtime, file, &format!("{prefix}.{name}"), shape);
        let norm = |name: &str| -> Result<Norm, TranscriptionError> {
            Ok(Norm {
                weight: load(&format!("{name}.weight"), &[d])?,
                bias: load(&format!("{name}.bias"), &[d])?,
            })
        };
        let feed_forward = |name: &str| -> Result<FeedForward, TranscriptionError> {
            Ok(FeedForward {
                linear1: load(&format!("{name}.linear1.weight"), &[spec.ff_hidden, d])?,
                linear2: load(&format!("{name}.linear2.weight"), &[d, spec.ff_hidden])?,
            })
        };

        let mut qkv = Vec::with_capacity(3 * d * d);
        for name in ["linear_q", "linear_k", "linear_v"] {
            qkv.extend_from_slice(tensor(&format!("self_attn.{name}.weight"), &[d, d])?);
        }
        let head_dim = d / spec.heads;
        let attention = Attention {
            qkv: DeviceTensor::upload(runtime.stream(), &qkv, &[3 * d, d])?,
            pos: load("self_attn.linear_pos.weight", &[d, d])?,
            out: load("self_attn.linear_out.weight", &[d, d])?,
            bias_u: load("self_attn.pos_bias_u", &[spec.heads, head_dim])?,
            bias_v: load("self_attn.pos_bias_v", &[spec.heads, head_dim])?,
        };

        // fold in f64 so the folded constants round once
        let bn = |name: &str| tensor(&format!("conv.batch_norm.{name}"), &[d]);
        let (gamma, beta, mean, var) = (
            bn("weight")?,
            bn("bias")?,
            bn("running_mean")?,
            bn("running_var")?,
        );
        let mut scale = Vec::with_capacity(d);
        let mut shift = Vec::with_capacity(d);
        for c in 0..d {
            let s = f64::from(gamma[c]) / (f64::from(var[c]) + f64::from(NORM_EPS)).sqrt();
            scale.push(s as f32);
            shift.push((f64::from(beta[c]) - f64::from(mean[c]) * s) as f32);
        }
        let conv = ConvModule {
            pointwise1: load("conv.pointwise_conv1.weight", &[2 * d, d, 1])?,
            depthwise: load("conv.depthwise_conv.weight", &[d, 1, spec.conv_kernel])?,
            scale: upload_host(runtime, &scale)?,
            shift: upload_host(runtime, &shift)?,
            pointwise2: load("conv.pointwise_conv2.weight", &[d, d, 1])?,
        };

        Ok(Layer {
            norm_ff1: norm("norm_feed_forward1")?,
            ff1: feed_forward("feed_forward1")?,
            norm_att: norm("norm_self_att")?,
            attention,
            norm_conv: norm("norm_conv")?,
            conv,
            norm_ff2: norm("norm_feed_forward2")?,
            ff2: feed_forward("feed_forward2")?,
            norm_out: norm("norm_out")?,
        })
    }

    /// Allocates the buffers for `batch` inputs of up to `mel_frames` frames, and the
    /// relative position table for that length
    pub(crate) fn workspace(
        &self,
        runtime: &CudaRuntime,
        batch: usize,
        mel_frames: usize,
    ) -> Result<EncoderWorkspace, TranscriptionError> {
        let stream = runtime.stream();
        let spec = self.spec;
        let (d, heads) = (spec.d_model, spec.heads);
        let channels = spec.subsampling_channels;
        let maps = self.feature_maps(batch, mel_frames);
        let frames = maps[3].frames;
        let rows = batch * frames;
        let positions = 2 * frames - 1;

        Ok(EncoderWorkspace {
            batch,
            mel_frames,
            frames,
            mel: zeros(stream, batch * mel_frames * spec.features)?,
            lens: [
                zeros(stream, batch)?,
                zeros(stream, batch)?,
                zeros(stream, batch)?,
                zeros(stream, batch)?,
            ],
            host_lens: vec![0; batch],
            maps: [zeros(stream, maps[1].len())?, zeros(stream, maps[2].len())?],
            flat: zeros(stream, rows * channels * maps[3].features)?,
            x: zeros(stream, rows * d)?,
            normed: zeros(stream, rows * d)?,
            hidden: zeros(stream, rows * spec.ff_hidden.max(2 * d))?,
            qkv: zeros(stream, rows * 3 * d)?,
            q_u: zeros(stream, rows * d)?,
            q_v: zeros(stream, rows * d)?,
            k: zeros(stream, rows * d)?,
            v: zeros(stream, rows * d)?,
            pos_emb: stream.clone_htod(&relative_positions(frames, d))?,
            pos: zeros(stream, positions * d)?,
            pos_heads: zeros(stream, positions * d)?,
            scores: zeros(stream, batch * heads * frames * frames)?,
            bd: zeros(stream, rows * heads * positions)?,
            context: zeros(stream, rows * d)?,
            merged: zeros(stream, rows * d)?,
        })
    }

    /// Input and subsampled feature maps: `[0]` is the mel, `[3]` the last stage
    fn feature_maps(&self, batch: usize, mel_frames: usize) -> [FeatureMap; 4] {
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
            frames: subsampled(map.frames),
            features: subsampled(map.features),
        };
        let first = next(mel);
        let second = next(first);
        [mel, first, second, next(second)]
    }

    /// Uploads a padded batch: `mel` is `[batch, mel_frames, features]` (time-major)
    /// and `lens` the valid mel frames of each row
    pub(crate) fn set_input(
        &self,
        runtime: &CudaRuntime,
        workspace: &mut EncoderWorkspace,
        mel: &[f32],
        lens: &[usize],
    ) -> Result<(), TranscriptionError> {
        if lens.len() != workspace.batch || mel.len() != workspace.mel.len() {
            return Err(TranscriptionError::InvalidConfiguration(format!(
                "encoder input holds {} values for {} rows; the workspace takes {} values for {} rows",
                mel.len(),
                lens.len(),
                workspace.mel.len(),
                workspace.batch
            )));
        }
        if let Some(len) = lens
            .iter()
            .find(|len| **len == 0 || **len > workspace.mel_frames)
        {
            return Err(TranscriptionError::InvalidConfiguration(format!(
                "encoder input row has {len} valid frames of {}",
                workspace.mel_frames
            )));
        }

        let stream = runtime.stream();
        stream.memcpy_htod(mel, &mut workspace.mel)?;
        let mut stage: Vec<usize> = lens.to_vec();
        for buffer in &mut workspace.lens {
            let values: Vec<u32> = stage.iter().map(|len| *len as u32).collect();
            stream.memcpy_htod(&values, buffer)?;
            stage = stage.into_iter().map(subsampled).collect();
        }
        // the loop subsampled once more than there are stages
        workspace.host_lens = lens
            .iter()
            .map(|len| subsampled(subsampled(subsampled(*len))))
            .collect();
        Ok(())
    }

    /// Queues the whole encoder on the runtime's stream
    pub(crate) fn enqueue(
        &self,
        runtime: &CudaRuntime,
        workspace: &mut EncoderWorkspace,
    ) -> Result<(), TranscriptionError> {
        self.enqueue_subsampling(runtime, workspace)?;
        for index in 0..self.layers.len() {
            self.enqueue_layer(runtime, workspace, index)?;
        }
        Ok(())
    }

    /// `pre_encode`: three stride-2 convolutions, then the projection to `d_model`
    pub(crate) fn enqueue_subsampling(
        &self,
        runtime: &CudaRuntime,
        ws: &mut EncoderWorkspace,
    ) -> Result<(), TranscriptionError> {
        let stream = runtime.stream();
        let kernels = &self.kernels;
        let sub = &self.subsampling;
        let maps = self.feature_maps(ws.batch, ws.mel_frames);
        let [mel_map, first, second, third] = maps;
        let channels = self.spec.subsampling_channels;
        let [large, small] = &mut ws.maps;

        kernels.subsample_conv_in(
            stream,
            &ws.mel.slice(..),
            &ws.lens[0].slice(..),
            &ws.lens[1].slice(..),
            &sub.conv_in.0.data().slice(..),
            &sub.conv_in.1.data().slice(..),
            &mut large.slice_mut(..),
            mel_map,
            first,
        )?;
        kernels.subsample_depthwise(
            stream,
            &large.slice(..),
            &ws.lens[2].slice(..),
            &sub.depthwise1.0.data().slice(..),
            &sub.depthwise1.1.data().slice(..),
            &mut small.slice_mut(..),
            first,
            second,
        )?;
        self.pointwise(runtime, &sub.pointwise1, small, large, second, &ws.lens[2])?;
        kernels.subsample_depthwise(
            stream,
            &large.slice(..second.len()),
            &ws.lens[3].slice(..),
            &sub.depthwise2.0.data().slice(..),
            &sub.depthwise2.1.data().slice(..),
            &mut small.slice_mut(..third.len()),
            second,
            third,
        )?;
        self.pointwise(runtime, &sub.pointwise2, small, large, third, &ws.lens[3])?;
        kernels.subsample_flatten(
            stream,
            &large.slice(..third.len()),
            &mut ws.flat.slice_mut(..),
            third,
        )?;

        let rows = ws.batch * ws.frames;
        let (weight, bias) = &sub.out;
        kernels.fill_rows(stream, &bias.data().slice(..), &mut ws.x.slice_mut(..))?;
        runtime.sgemm(
            linear(rows, self.spec.d_model, channels * third.features, 1.0, 1.0),
            &ws.flat,
            weight.data(),
            &mut ws.x,
        )?;
        Ok(())
    }

    /// A pointwise subsampling convolution, `[channels, channels]` over each row's
    /// `[channels, frames * features]`, then bias, ReLU and the time mask
    fn pointwise(
        &self,
        runtime: &CudaRuntime,
        (weight, bias): &(DeviceTensor, DeviceTensor),
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        map: FeatureMap,
        lens: &CudaSlice<u32>,
    ) -> Result<(), TranscriptionError> {
        let plane = map.frames * map.features;
        let per_row = map.channels * plane;
        let gemm = Sgemm::new(map.channels, plane, map.channels);
        runtime.sgemm_batched(
            SgemmBatched {
                gemm,
                batch: map.batch,
                stride_a: 0,
                stride_b: per_row,
                stride_c: per_row,
            },
            weight.data(),
            &input.slice(..map.len()),
            &mut output.slice_mut(..map.len()),
        )?;
        self.kernels.subsample_bias_relu(
            runtime.stream(),
            &mut output.slice_mut(..map.len()),
            &bias.data().slice(..),
            &lens.slice(..),
            map,
        )?;
        Ok(())
    }

    /// One Conformer layer on the residual stream
    pub(crate) fn enqueue_layer(
        &self,
        runtime: &CudaRuntime,
        ws: &mut EncoderWorkspace,
        index: usize,
    ) -> Result<(), TranscriptionError> {
        let layer = &self.layers[index];
        self.feed_forward(runtime, ws, &layer.norm_ff1, &layer.ff1)?;
        self.attention(runtime, ws, &layer.norm_att, &layer.attention)?;
        self.conv_module(runtime, ws, &layer.norm_conv, &layer.conv)?;
        self.feed_forward(runtime, ws, &layer.norm_ff2, &layer.ff2)?;

        // the final norm is not residual; normalize into the scratch row and copy back
        self.norm(runtime, ws, &layer.norm_out)?;
        runtime.stream().memcpy_dtod(&ws.normed, &mut ws.x)?;
        Ok(())
    }

    fn norm(
        &self,
        runtime: &CudaRuntime,
        ws: &mut EncoderWorkspace,
        norm: &Norm,
    ) -> Result<(), TranscriptionError> {
        self.kernels.layer_norm(
            runtime.stream(),
            &ws.x.slice(..),
            &norm.weight.data().slice(..),
            &norm.bias.data().slice(..),
            &mut ws.normed.slice_mut(..),
            self.spec.d_model,
            NORM_EPS,
        )?;
        Ok(())
    }

    /// `x += 0.5 * linear2(silu(linear1(norm(x))))`
    fn feed_forward(
        &self,
        runtime: &CudaRuntime,
        ws: &mut EncoderWorkspace,
        norm: &Norm,
        weights: &FeedForward,
    ) -> Result<(), TranscriptionError> {
        let (d, hidden) = (self.spec.d_model, self.spec.ff_hidden);
        let rows = ws.batch * ws.frames;
        self.norm(runtime, ws, norm)?;
        let mut hidden_view = ws.hidden.slice_mut(..rows * hidden);
        runtime.sgemm(
            linear(rows, hidden, d, 1.0, 0.0),
            &ws.normed,
            weights.linear1.data(),
            &mut hidden_view,
        )?;
        self.kernels.silu(runtime.stream(), &mut hidden_view)?;
        // NeMo scales the feed-forward residual by 0.5 (`fc_factor`)
        runtime.sgemm(
            linear(rows, d, hidden, 0.5, 1.0),
            &ws.hidden.slice(..rows * hidden),
            weights.linear2.data(),
            &mut ws.x,
        )?;
        Ok(())
    }

    /// `x += linear_out(rel_pos_attention(norm(x)))`
    fn attention(
        &self,
        runtime: &CudaRuntime,
        ws: &mut EncoderWorkspace,
        norm: &Norm,
        weights: &Attention,
    ) -> Result<(), TranscriptionError> {
        let stream = runtime.stream();
        let spec = self.spec;
        let (d, heads, head_dim) = (spec.d_model, spec.heads, spec.d_model / spec.heads);
        let (batch, frames) = (ws.batch, ws.frames);
        let rows = batch * frames;
        let positions = 2 * frames - 1;
        let shape = Heads {
            batch,
            frames,
            heads,
            head_dim,
        };

        self.norm(runtime, ws, norm)?;
        runtime.sgemm(
            linear(rows, 3 * d, d, 1.0, 0.0),
            &ws.normed,
            weights.qkv.data(),
            &mut ws.qkv,
        )?;
        self.kernels.split_qkv(
            stream,
            &ws.qkv.slice(..),
            &weights.bias_u.data().slice(..),
            &weights.bias_v.data().slice(..),
            &mut ws.q_u.slice_mut(..),
            &mut ws.q_v.slice_mut(..),
            &mut ws.k.slice_mut(..),
            &mut ws.v.slice_mut(..),
            shape,
        )?;
        runtime.sgemm(
            linear(positions, d, d, 1.0, 0.0),
            &ws.pos_emb,
            weights.pos.data(),
            &mut ws.pos,
        )?;
        self.kernels.split_heads(
            stream,
            &ws.pos.slice(..),
            &mut ws.pos_heads.slice_mut(..),
            positions,
            heads,
            head_dim,
        )?;

        // content scores: (q + u) · kᵀ per (batch, head)
        let content = Sgemm {
            b_transposed: true,
            ..Sgemm::new(frames, frames, head_dim)
        };
        runtime.sgemm_batched(
            SgemmBatched::packed(content, batch * heads),
            &ws.q_u,
            &ws.k,
            &mut ws.scores,
        )?;
        // position scores: (q + v) · pᵀ per head, every utterance against one table
        let position = Sgemm {
            b_transposed: true,
            ..Sgemm::new(rows, positions, head_dim)
        };
        runtime.sgemm_batched(
            SgemmBatched::packed(position, heads),
            &ws.q_v,
            &ws.pos_heads,
            &mut ws.bd,
        )?;
        self.kernels.rel_pos_softmax(
            stream,
            &mut ws.scores.slice_mut(..),
            &ws.bd.slice(..),
            &ws.lens[3].slice(..),
            shape,
        )?;

        let mix = Sgemm::new(frames, head_dim, frames);
        runtime.sgemm_batched(
            SgemmBatched::packed(mix, batch * heads),
            &ws.scores,
            &ws.v,
            &mut ws.context,
        )?;
        self.kernels.merge_heads(
            stream,
            &ws.context.slice(..),
            &mut ws.merged.slice_mut(..),
            shape,
        )?;
        runtime.sgemm(
            linear(rows, d, d, 1.0, 1.0),
            &ws.merged,
            weights.out.data(),
            &mut ws.x,
        )?;
        Ok(())
    }

    /// `x += pointwise2(silu(bn(depthwise(mask(glu(pointwise1(norm(x))))))))`
    fn conv_module(
        &self,
        runtime: &CudaRuntime,
        ws: &mut EncoderWorkspace,
        norm: &Norm,
        weights: &ConvModule,
    ) -> Result<(), TranscriptionError> {
        let d = self.spec.d_model;
        let rows = ws.batch * ws.frames;
        self.norm(runtime, ws, norm)?;
        let pw1_len = rows * 2 * d;
        runtime.sgemm(
            linear(rows, 2 * d, d, 1.0, 0.0),
            &ws.normed,
            weights.pointwise1.data(),
            &mut ws.hidden.slice_mut(..pw1_len),
        )?;
        self.kernels.conv_glue(
            runtime.stream(),
            &ws.hidden.slice(..pw1_len),
            &ws.lens[3].slice(..),
            &weights.depthwise.data().slice(..),
            &weights.scale.data().slice(..),
            &weights.shift.data().slice(..),
            &mut ws.merged.slice_mut(..),
            ws.batch,
            ws.frames,
            d,
            self.spec.conv_kernel,
        )?;
        runtime.sgemm(
            linear(rows, d, d, 1.0, 1.0),
            &ws.merged,
            weights.pointwise2.data(),
            &mut ws.x,
        )?;
        Ok(())
    }
}

/// `c = alpha * a · wᵀ + beta * c` for a PyTorch `[out, in]` weight, in FP32
fn linear(rows: usize, out: usize, inputs: usize, alpha: f32, beta: f32) -> Sgemm {
    Sgemm {
        b_transposed: true,
        alpha,
        beta,
        math: Math::Fp32,
        ..Sgemm::new(rows, out, inputs)
    }
}

/// NeMo's `RelPositionalEncoding` rows for positions `frames - 1` down to
/// `-(frames - 1)`, `[2 * frames - 1, d_model]`, computed in FP32 as NeMo does
fn relative_positions(frames: usize, d_model: usize) -> Vec<f32> {
    let scale = -(10000f64.ln() / d_model as f64) as f32;
    let div_term: Vec<f32> = (0..d_model)
        .step_by(2)
        .map(|i| (i as f32 * scale).exp())
        .collect();
    let mut table = Vec::with_capacity((2 * frames - 1) * d_model);
    for row in 0..2 * frames - 1 {
        let position = (frames - 1) as f32 - row as f32;
        for term in &div_term {
            let angle = position * term;
            table.push(angle.sin());
            table.push(angle.cos());
        }
    }
    table
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use cudakit::{Library, Options};
    use ndarray::{ArrayD, Axis, Ix3};

    use super::{CudaEncoder, relative_positions};
    use crate::cuda::kernels::CudaRuntime;
    use crate::deployment::{ParakeetDeployment, ParakeetVersion};
    use crate::native::weights::WeightFile;
    use crate::test_support::NemoReference;

    fn weights(version: ParakeetVersion) -> WeightFile {
        let root = std::env::var_os("SCRIPTRS_NATIVE_MODELS")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/shared/models/scriptrs-native"));
        let dir = match version {
            ParakeetVersion::V2 => "parakeet-v2",
            ParakeetVersion::V3 => "parakeet-v3",
        };
        WeightFile::open(&root.join(dir).join("model.safetensors")).expect("native weights")
    }

    /// Relative L2 error and max abs difference over the valid frames of `[batch,
    /// frames, d_model]` outputs
    fn compare(ours: &[f32], reference: &ArrayD<f32>, lens: &[usize]) -> (f64, f64) {
        let reference = reference
            .view()
            .into_dimensionality::<Ix3>()
            .expect("[batch, frames, d]");
        let (frames, width) = (reference.shape()[1], reference.shape()[2]);
        let (mut error, mut norm, mut max_abs) = (0f64, 0f64, 0f64);
        for (b, row) in reference.axis_iter(Axis(0)).enumerate() {
            for t in 0..lens[b] {
                for c in 0..width {
                    let expected = f64::from(row[[t, c]]);
                    let diff = f64::from(ours[(b * frames + t) * width + c]) - expected;
                    error += diff * diff;
                    norm += expected * expected;
                    max_abs = max_abs.max(diff.abs());
                }
            }
        }
        ((error / norm).sqrt(), max_abs)
    }

    #[test]
    #[ignore = "reads NeMo references from /shared/datasets/parakeet-ref"]
    fn nemo_parity_relative_positions() {
        let reference = NemoReference::open(ParakeetVersion::V2, "b8");
        let table = reference.f32("encoder.layers.0.self_attn.linear_pos.in");
        let positions = table.shape()[1];
        let ours = relative_positions(positions.div_ceil(2), table.shape()[2]);
        let max_abs = ours
            .iter()
            .zip(table.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        // sin and cos of arguments up to a few thousand radians differ by an ulp or so
        assert!(
            max_abs < 1e-3,
            "relative position table differs by {max_abs}"
        );
    }

    /// Runs the encoder stage by stage and holds every stage against NeMo FP32
    ///
    /// Needs a CUDA GPU: run it through the shared queue with `just gpu-parity`
    #[test]
    #[ignore = "needs a CUDA GPU and the NeMo references; `just gpu-parity` runs it on gpu0"]
    fn gpu_encoder_matches_nemo_layer_by_layer() {
        let runtime = CudaRuntime::open(0, Options::default()).expect("CUDA runtime");
        runtime.prepare(Library::Cublas).expect("cuBLAS");
        let mut worst = 0f64;
        for version in [ParakeetVersion::V2, ParakeetVersion::V3] {
            let deployment = match version {
                ParakeetVersion::V2 => ParakeetDeployment::v2(),
                ParakeetVersion::V3 => ParakeetDeployment::v3(),
            };
            let file = weights(version);
            let spec = file.parakeet_spec(&deployment).expect("spec");
            let encoder = CudaEncoder::load(&runtime, &file, spec.encoder).expect("encoder");

            for set in ["b1", "b8", "long-full"] {
                let reference = NemoReference::open(version, set);
                let mel = reference
                    .f32("mel")
                    .into_dimensionality::<Ix3>()
                    .expect("[batch, features, frames]");
                let (batch, mel_frames) = (mel.shape()[0], mel.shape()[2]);
                let lens: Vec<usize> = reference
                    .i64("mel_len")
                    .iter()
                    .map(|len| *len as usize)
                    .collect();
                // NeMo's featurizer is feature-major; the encoder takes time-major rows
                let time_major: Vec<f32> = mel.permuted_axes([0, 2, 1]).iter().copied().collect();

                let mut ws = encoder
                    .workspace(&runtime, batch, mel_frames)
                    .expect("workspace");
                encoder
                    .set_input(&runtime, &mut ws, &time_major, &lens)
                    .expect("input");
                let out_lens: Vec<usize> = reference
                    .i64("encoder.out_len")
                    .iter()
                    .map(|len| *len as usize)
                    .collect();
                assert_eq!(
                    ws.output_lens(),
                    out_lens.as_slice(),
                    "{version:?} {set} encoder lengths"
                );

                let download = |ws: &super::EncoderWorkspace| {
                    runtime.stream().clone_dtoh(ws.output()).expect("download")
                };
                encoder
                    .enqueue_subsampling(&runtime, &mut ws)
                    .expect("subsampling");
                let (rel, max_abs) = compare(
                    &download(&ws),
                    &reference.f32("encoder.pre_encode.out"),
                    &out_lens,
                );
                println!("{version:?} {set} pre_encode rel {rel:.2e} max {max_abs:.2e}");
                for index in 0..spec.encoder.layers {
                    encoder
                        .enqueue_layer(&runtime, &mut ws, index)
                        .expect("layer");
                    let (rel, max_abs) = compare(
                        &download(&ws),
                        &reference.f32(&format!("encoder.layers.{index}.out")),
                        &out_lens,
                    );
                    println!("{version:?} {set} layer {index:2} rel {rel:.2e} max {max_abs:.2e}");
                }

                let expected = reference.f32("encoder.out").permuted_axes(vec![0, 2, 1]);
                let (rel, max_abs) = compare(
                    &download(&ws),
                    &expected.as_standard_layout().into_owned(),
                    &out_lens,
                );
                println!("{version:?} {set} encoder.out rel {rel:.2e} max {max_abs:.2e}");
                worst = worst.max(rel);
            }
        }

        // NeMo FP32 with TF32 off against cuBLAS FP32: summation order differs only
        assert!(worst < 1e-4, "encoder output relative L2 {worst:.2e}");
    }

    /// Captures the encoder in a CUDA graph, checks the replay against the eager pass
    /// bit for bit, and prints replay times for the v3 reference shapes
    #[test]
    #[ignore = "needs a CUDA GPU and the NeMo references; `just gpu-parity` runs it on gpu0"]
    fn gpu_encoder_graph_replays_the_eager_pass() {
        const REPLAYS: u32 = 100;
        let runtime = CudaRuntime::open(0, Options::default()).expect("CUDA runtime");
        runtime.prepare(Library::Cublas).expect("cuBLAS");
        let file = weights(ParakeetVersion::V3);
        let spec = file.parakeet_spec(&ParakeetDeployment::v3()).expect("spec");
        let encoder = CudaEncoder::load(&runtime, &file, spec.encoder).expect("encoder");

        for set in ["b1", "b8"] {
            let reference = NemoReference::open(ParakeetVersion::V3, set);
            let mel = reference
                .f32("mel")
                .into_dimensionality::<Ix3>()
                .expect("[batch, features, frames]");
            let (batch, mel_frames) = (mel.shape()[0], mel.shape()[2]);
            let lens: Vec<usize> = reference
                .i64("mel_len")
                .iter()
                .map(|len| *len as usize)
                .collect();
            let time_major: Vec<f32> = mel.permuted_axes([0, 2, 1]).iter().copied().collect();
            let mut ws = encoder
                .workspace(&runtime, batch, mel_frames)
                .expect("workspace");
            encoder
                .set_input(&runtime, &mut ws, &time_major, &lens)
                .expect("input");

            // the eager pass also finishes cuBLAS's lazy setup before capture
            encoder.enqueue(&runtime, &mut ws).expect("eager");
            let eager = runtime.stream().clone_dtoh(ws.output()).expect("download");
            let graph = runtime
                .capture_nonempty(|| encoder.enqueue(&runtime, &mut ws))
                .expect("capture");
            graph.launch().expect("replay");
            let replayed = runtime.stream().clone_dtoh(ws.output()).expect("download");
            assert!(
                eager == replayed,
                "v3 {set}: graph replay differs from the eager pass"
            );

            let start = std::time::Instant::now();
            for _ in 0..REPLAYS {
                graph.launch().expect("replay");
            }
            runtime.synchronize().expect("sync");
            let replay_ms = start.elapsed().as_secs_f64() * 1e3 / f64::from(REPLAYS);
            let audio_s: f64 = reference
                .i64("audio_len")
                .iter()
                .map(|len| *len as f64 / 16000.0)
                .sum();
            println!(
                "v3 {set}: batch {batch}, {audio_s:.1} s audio, encoder graph replay {replay_ms:.3} ms"
            );
        }
    }
}
