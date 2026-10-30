//! Self-attention of the offline encoder: the fused kernel for the shapes
//! [`FusedShapes`] picks, else the cuBLAS head split, score products and rel-shift
//! softmax

use cudakit::cudarc::driver::{CudaSlice, CudaView};
use cudakit::{Sgemm, SgemmBatched};

use super::{
    Activations, Attention, CudaEncoder, EncoderWorkspace, Linear, gemm_input, merged_in,
    mixed_precision, normed_in,
};
use crate::cuda::fp16::linear_fp16_to_fp16;
use crate::cuda::kernels::{AttentionKernels, CudaRuntime, Heads};
use crate::cuda::linear;
use crate::error::TranscriptionError;

/// Input shapes the FP16 encoder runs the fused attention on: on an RTX 5090 it is 3
/// to 8 % faster than cuBLAS up to 50 frames at any batch, within 1.5 % either way at
/// 100, and 5 to 19 % faster from 640 rows, but up to 15 % slower on a single input of
/// 376 or more frames
pub(super) const FUSED_FP16: FusedShapes = FusedShapes {
    short_frames: 96,
    min_rows: 640,
};

/// Input shapes the TF32 encoder runs the fused attention on: its FP32 tiles fill
/// shared memory twice as fast, so it needs about twice the rows, and from 50 frames it
/// is up to 35 % slower than cuBLAS on a single input
pub(super) const FUSED_TF32: FusedShapes = FusedShapes {
    short_frames: 32,
    min_rows: 1200,
};

/// The fused attention kernel and the inputs it takes over
pub(super) struct FusedAttention {
    pub(super) kernels: AttentionKernels,
    pub(super) shapes: FusedShapes,
}

/// Input shapes, by batch rows and encoder frames, that run the fused attention; the
/// rest keep the cuBLAS path
///
/// Short inputs gain from the launches and tiny GEMMs the kernel replaces. Longer ones
/// need many rows: the kernel runs `batch * heads * frames / 16` warps, and one block
/// takes most of an sm_120 SM's shared memory, so a few long rows leave the SMs a
/// warp or two each, while cuBLAS still fills them
#[derive(Clone, Copy, Debug)]
pub(super) struct FusedShapes {
    /// Longest input, in encoder frames, that runs it at any batch
    short_frames: usize,
    /// Fewest `batch * frames` rows that run it at any length
    min_rows: usize,
}

impl FusedShapes {
    pub(super) fn takes(self, batch: usize, frames: usize) -> bool {
        frames <= self.short_frames || batch * frames >= self.min_rows
    }
}

/// Self-attention buffers for the path [`CudaEncoder::workspace`] picked, beside the
/// QKV projection in [`Activations`]
pub(super) enum AttentionBuffers {
    /// The head split, cuBLAS score and value products, rel-shift softmax and head
    /// merge
    Unfused(Box<UnfusedAttention>),
    /// The FP16 kernel's operands
    FusedFp16 {
        /// `[batch * frames, 3 * d_model]` QKV projection in FP16, which the kernel
        /// reads in place of the FP32 one the streaming encoder shares
        qkv: CudaSlice<u16>,
        /// `[2 * frames - 1, d_model]` projected position table
        pos: CudaSlice<u16>,
    },
    /// The TF32 kernel's `[2 * frames - 1, d_model]` projected position table
    FusedTf32 { pos: CudaSlice<f32> },
}

pub(super) struct UnfusedAttention {
    /// `[batch, heads, frames, head_dim]`, `q_v` `[heads, batch, frames, head_dim]`
    pub(super) q_u: CudaSlice<f32>,
    pub(super) q_v: CudaSlice<f32>,
    pub(super) k: CudaSlice<f32>,
    pub(super) v: CudaSlice<f32>,
    /// `[2 * frames - 1, d_model]` projected position table, and `[heads, 2 * frames -
    /// 1, head_dim]`
    pub(super) pos: CudaSlice<f32>,
    pub(super) pos_heads: CudaSlice<f32>,
    /// `[batch, heads, frames, frames]` content scores, then weights
    pub(super) scores: CudaSlice<f32>,
    /// `[heads, batch, frames, 2 * frames - 1]` position scores
    pub(super) bd: CudaSlice<f32>,
    /// `[batch, heads, frames, head_dim]`
    pub(super) context: CudaSlice<f32>,
}

/// What an attention sublayer reads from its workspace beside the buffers of its
/// path
struct AttentionInputs<'a> {
    shape: Heads,
    /// Valid encoder frames per batch row
    lens: CudaView<'a, u32>,
    /// `[2 * frames - 1, d_model]` relative position table
    pos_emb: &'a CudaSlice<f32>,
    /// Its FP16 copy, where the linear layers read FP16
    pos_emb_f16: Option<&'a CudaSlice<u16>>,
}

impl AttentionBuffers {
    pub(super) fn bytes(&self) -> usize {
        match self {
            Self::Unfused(buffers) => [
                &buffers.q_u,
                &buffers.q_v,
                &buffers.k,
                &buffers.v,
                &buffers.pos,
                &buffers.pos_heads,
                &buffers.scores,
                &buffers.bd,
                &buffers.context,
            ]
            .iter()
            .map(|buffer| buffer.num_bytes())
            .sum(),
            Self::FusedFp16 { qkv, pos } => qkv.num_bytes() + pos.num_bytes(),
            Self::FusedTf32 { pos } => pos.num_bytes(),
        }
    }
}

impl CudaEncoder {
    /// `x += linear_out(rel_pos_attention(normed))`, with `normed` the sublayer's
    /// norm of `x`, through the attention path the workspace was built for
    pub(super) fn attention(
        &self,
        runtime: &CudaRuntime,
        ws: &mut EncoderWorkspace,
        weights: &Attention,
    ) -> Result<(), TranscriptionError> {
        let spec = self.spec;
        let EncoderWorkspace {
            batch,
            frames,
            lens,
            act,
            pos_emb,
            pos_emb_f16,
            attention,
            ..
        } = ws;
        let inputs = AttentionInputs {
            shape: Heads {
                batch: *batch,
                frames: *frames,
                heads: spec.heads,
                head_dim: spec.d_model / spec.heads,
            },
            lens: lens[3].slice(..),
            pos_emb,
            pos_emb_f16: pos_emb_f16.as_ref(),
        };
        match attention {
            AttentionBuffers::Unfused(buffers) => {
                self.unfused_attention(runtime, act, buffers, &inputs, weights)?;
            }
            fused => self.fused_attention(runtime, act, fused, &inputs, weights)?,
        }

        let d = spec.d_model;
        self.linear_or_quantized(
            runtime,
            linear(act.rows, d, d, 1.0, 1.0, self.math.single()),
            merged_in(&act.merged, act.fp16.as_ref()),
            &weights.out,
            &mut act.x,
            &mut act.quant,
        )?;
        Ok(())
    }

    /// The attention context into `merged`, or its FP16 copy, through the projections
    /// and the fused kernel of the encoder's precision
    fn fused_attention(
        &self,
        runtime: &CudaRuntime,
        act: &mut Activations,
        buffers: &mut AttentionBuffers,
        inputs: &AttentionInputs<'_>,
        weights: &Attention,
    ) -> Result<(), TranscriptionError> {
        // the encoder that built a fused workspace has the kernel
        let fused = self.fused.as_ref().ok_or_else(mixed_precision)?;
        let stream = runtime.stream();
        let d = self.spec.d_model;
        let rows = act.rows;
        let positions = 2 * inputs.shape.frames - 1;
        let math = self.math.single();
        let (qkv_spec, pos_spec) = (
            linear(rows, 3 * d, d, 1.0, 0.0, math),
            linear(positions, d, d, 1.0, 0.0, math),
        );
        let (bias_u, bias_v) = (
            weights.bias_u.data().slice(..),
            weights.bias_v.data().slice(..),
        );

        // the buffers, activations and weights all follow the encoder's precision,
        // which their types do not tie together
        match (
            buffers,
            act.fp16.as_mut(),
            inputs.pos_emb_f16,
            &weights.qkv,
            &weights.pos,
        ) {
            (
                AttentionBuffers::FusedFp16 { qkv, pos },
                Some(fp16),
                Some(table),
                Linear::Fp16(qkv_weight),
                Linear::Fp16(pos_weight),
            ) => {
                linear_fp16_to_fp16(runtime, qkv_spec, &fp16.normed, qkv_weight, qkv)?;
                linear_fp16_to_fp16(runtime, pos_spec, table, pos_weight, pos)?;
                fused.kernels.rel_pos_f16(
                    stream,
                    &qkv.slice(..),
                    &pos.slice(..),
                    &bias_u,
                    &bias_v,
                    &inputs.lens,
                    &mut fp16.merged.slice_mut(..),
                    inputs.shape,
                )?;
            }
            (
                AttentionBuffers::FusedTf32 { pos },
                None,
                None,
                Linear::Single(qkv_weight),
                Linear::Single(pos_weight),
            ) => {
                runtime.sgemm(qkv_spec, &act.normed, qkv_weight.data(), &mut act.qkv)?;
                runtime.sgemm(pos_spec, inputs.pos_emb, pos_weight.data(), pos)?;
                fused.kernels.rel_pos_tf32(
                    stream,
                    &act.qkv.slice(..),
                    &pos.slice(..),
                    &bias_u,
                    &bias_v,
                    &inputs.lens,
                    &mut act.merged.slice_mut(..),
                    inputs.shape,
                )?;
            }
            _ => return Err(mixed_precision()),
        }
        Ok(())
    }

    /// The attention context into `merged` (or its FP16 copy) through the head split,
    /// cuBLAS score and value products, the rel-shift softmax and the head merge
    fn unfused_attention(
        &self,
        runtime: &CudaRuntime,
        act: &mut Activations,
        buffers: &mut UnfusedAttention,
        inputs: &AttentionInputs<'_>,
        weights: &Attention,
    ) -> Result<(), TranscriptionError> {
        let stream = runtime.stream();
        let shape = inputs.shape;
        let Heads {
            batch,
            frames,
            heads,
            head_dim,
        } = shape;
        let d = self.spec.d_model;
        let rows = batch * frames;
        let positions = 2 * frames - 1;

        let math = self.math.single();
        self.linear_or_quantized(
            runtime,
            linear(rows, 3 * d, d, 1.0, 0.0, math),
            normed_in(&act.normed, act.fp16.as_ref()),
            &weights.qkv,
            &mut act.qkv,
            &mut act.quant,
        )?;
        self.kernels.split_qkv(
            stream,
            &act.qkv.slice(..),
            &weights.bias_u.data().slice(..),
            &weights.bias_v.data().slice(..),
            &mut buffers.q_u.slice_mut(..),
            &mut buffers.q_v.slice_mut(..),
            &mut buffers.k.slice_mut(..),
            &mut buffers.v.slice_mut(..),
            shape,
        )?;
        let table = gemm_input(
            inputs.pos_emb.slice(..),
            inputs.pos_emb_f16.map(|table| table.slice(..)),
        );
        self.linear_or_quantized(
            runtime,
            linear(positions, d, d, 1.0, 0.0, math),
            table,
            &weights.pos,
            &mut buffers.pos,
            &mut act.quant,
        )?;
        self.kernels.split_heads(
            stream,
            &buffers.pos.slice(..),
            &mut buffers.pos_heads.slice_mut(..),
            positions,
            heads,
            head_dim,
        )?;

        // content scores: (q + u) · kᵀ per (batch, head)
        let content = Sgemm {
            b_transposed: true,
            math,
            ..Sgemm::new(frames, frames, head_dim)
        };
        runtime.sgemm_batched(
            SgemmBatched::packed(content, batch * heads),
            &buffers.q_u,
            &buffers.k,
            &mut buffers.scores,
        )?;
        // position scores: (q + v) · pᵀ per head, every utterance against one table
        let position = Sgemm {
            b_transposed: true,
            math,
            ..Sgemm::new(rows, positions, head_dim)
        };
        runtime.sgemm_batched(
            SgemmBatched::packed(position, heads),
            &buffers.q_v,
            &buffers.pos_heads,
            &mut buffers.bd,
        )?;
        self.kernels.rel_pos_softmax(
            stream,
            &mut buffers.scores.slice_mut(..),
            &buffers.bd.slice(..),
            &inputs.lens,
            shape,
        )?;

        let mix = Sgemm {
            math,
            ..Sgemm::new(frames, head_dim, frames)
        };
        runtime.sgemm_batched(
            SgemmBatched::packed(mix, batch * heads),
            &buffers.scores,
            &buffers.v,
            &mut buffers.context,
        )?;
        self.kernels
            .merge_heads(stream, &buffers.context.slice(..), act.merged_out(), shape)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use cudakit::{Library, Math, Options};
    use ndarray::Ix3;

    use crate::cuda::encoder::{CudaEncoder, EncoderMath};
    use crate::cuda::kernels::CudaRuntime;
    use crate::cuda::test_support::{compare, weights};
    use crate::deployment::{ParakeetDeployment, ParakeetVersion};
    use crate::test_support::NemoReference;

    /// IEEE half bits to `f32`, for reading the FP16 kernel's buffers on the host
    fn f16_to_f32(bits: u16) -> f32 {
        let sign = u32::from(bits >> 15) << 31;
        let exponent = u32::from((bits >> 10) & 0x1f);
        let mantissa = u32::from(bits & 0x3ff);
        let magnitude = match exponent {
            0 => mantissa as f32 * 2f32.powi(-24),
            0x1f if mantissa == 0 => f32::INFINITY,
            0x1f => f32::NAN,
            _ => f32::from_bits(((exponent + 112) << 23) | (mantissa << 13)),
        };
        f32::from_bits(magnitude.to_bits() | sign)
    }

    /// NeMo's relative-position attention in f64: `qkv` is `[batch * frames, 3 *
    /// d_model]`, `pos` `[2 * frames - 1, d_model]`, and the result the merged `[batch *
    /// frames, d_model]` context, 0 on rows past each length
    fn attention_reference(
        qkv: &[f32],
        pos: &[f32],
        [bias_u, bias_v]: [&[f32]; 2],
        lens: &[usize],
        frames: usize,
        heads: usize,
    ) -> Vec<f32> {
        let d = bias_u.len();
        let head_dim = d / heads;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let mut out = vec![0f32; lens.len() * frames * d];
        let mut scores = vec![0f64; frames];
        for (b, &len) in lens.iter().enumerate() {
            let row = |i: usize| &qkv[(b * frames + i) * 3 * d..][..3 * d];
            for h in 0..heads {
                let c0 = h * head_dim;
                for i in 0..len {
                    let q = &row(i)[c0..c0 + head_dim];
                    let mut max = f64::NEG_INFINITY;
                    for (j, score) in scores.iter_mut().enumerate().take(len) {
                        let k = &row(j)[d + c0..d + c0 + head_dim];
                        let p = &pos[(frames - 1 - i + j) * d + c0..][..head_dim];
                        let mut sum = 0f64;
                        for c in 0..head_dim {
                            let q = f64::from(q[c]);
                            sum += (q + f64::from(bias_u[c0 + c])) * f64::from(k[c])
                                + (q + f64::from(bias_v[c0 + c])) * f64::from(p[c]);
                        }
                        *score = sum * scale;
                        max = max.max(*score);
                    }
                    let mut total = 0f64;
                    for score in scores.iter_mut().take(len) {
                        *score = (*score - max).exp();
                        total += *score;
                    }
                    let context = &mut out[(b * frames + i) * d + c0..][..head_dim];
                    for (c, value) in context.iter_mut().enumerate() {
                        let mut sum = 0f64;
                        for (j, weight) in scores.iter().enumerate().take(len) {
                            sum += weight * f64::from(row(j)[2 * d + c0 + c]);
                        }
                        *value = (sum / total) as f32;
                    }
                }
            }
        }
        out
    }

    /// Relative L2 error of `ours` against `expected` over the valid rows, in total,
    /// per head, and per row inside a warp's 16 query rows; padded rows must be 0
    fn attention_error(
        ours: &[f32],
        expected: &[f32],
        lens: &[usize],
        frames: usize,
        heads: usize,
    ) -> (f64, Vec<f64>, Vec<f64>) {
        let d = expected.len() / (lens.len() * frames);
        let head_dim = d / heads;
        let mut per_head = vec![[0f64; 2]; heads];
        let mut per_row = vec![[0f64; 2]; 16];
        for (b, &len) in lens.iter().enumerate() {
            for i in 0..len {
                for c in 0..d {
                    let at = (b * frames + i) * d + c;
                    let (value, reference) = (f64::from(ours[at]), f64::from(expected[at]));
                    assert!(value.is_finite(), "row {b} frame {i} channel {c}: {value}");
                    let error = (value - reference).powi(2);
                    let norm = reference.powi(2);
                    for bucket in [&mut per_head[c / head_dim], &mut per_row[i % 16]] {
                        bucket[0] += error;
                        bucket[1] += norm;
                    }
                }
            }
            // NeMo zeroes the weights of padded query rows
            for i in len..frames {
                let row = &ours[(b * frames + i) * d..(b * frames + i + 1) * d];
                assert!(
                    row.iter().all(|value| *value == 0.0),
                    "row {b} padded frame {i}"
                );
            }
        }
        let rel = |[error, norm]: [f64; 2]| (error / norm).sqrt();
        let total = rel(per_head
            .iter()
            .fold([0.0; 2], |a, b| [a[0] + b[0], a[1] + b[1]]));
        (
            total,
            per_head.into_iter().map(rel).collect(),
            per_row.into_iter().map(rel).collect(),
        )
    }

    /// Both fused attention kernels and the conversions their inputs take
    struct FusedKernels {
        runtime: CudaRuntime,
        attention: crate::cuda::kernels::AttentionKernels,
        encoder: crate::cuda::kernels::EncoderKernels,
    }

    /// One kernel's output and the f64 attention on the operands it read
    struct FusedRun {
        name: &'static str,
        out: Vec<f32>,
        reference: Vec<f32>,
    }

    impl FusedKernels {
        /// The kernels on GPU 0, or `None` where the fused attention does not load
        fn open() -> Option<Self> {
            use crate::cuda::kernels::{AttentionKernels, EncoderKernels};

            let runtime = CudaRuntime::open(0, Options::default()).expect("CUDA runtime");
            let attention = AttentionKernels::load(&runtime).expect("attention kernels")?;
            let encoder = EncoderKernels::load(&runtime).expect("encoder kernels");
            Some(Self {
                runtime,
                attention,
                encoder,
            })
        }

        /// Runs both kernels on FP32 `qkv` and `pos`: FP16 on FP16 copies of them,
        /// which its reference reads too, and TF32 on them as they are
        fn run(
            &self,
            qkv: &[f32],
            pos: &[f32],
            biases: [&[f32]; 2],
            lens: &[usize],
            frames: usize,
            heads: usize,
        ) -> [FusedRun; 2] {
            use crate::cuda::kernels::Heads;
            use cudakit::cudarc::driver::CudaSlice;

            let stream = self.runtime.stream();
            let shape = Heads {
                batch: lens.len(),
                frames,
                heads,
                head_dim: biases[0].len() / heads,
            };
            let rows = qkv.len() / 3;
            let qkv_device = stream.clone_htod(qkv).expect("upload");
            let pos_device = stream.clone_htod(pos).expect("upload");
            let bias_u = stream.clone_htod(biases[0]).expect("upload");
            let bias_v = stream.clone_htod(biases[1]).expect("upload");
            let lens_device = stream
                .clone_htod(&lens.iter().map(|len| *len as u32).collect::<Vec<_>>())
                .expect("upload");

            let mut qkv_f16 = stream.alloc_zeros::<u16>(qkv.len()).expect("alloc");
            let mut pos_f16 = stream.alloc_zeros::<u16>(pos.len()).expect("alloc");
            self.encoder
                .to_f16(stream, &qkv_device.slice(..), &mut qkv_f16.slice_mut(..))
                .expect("to fp16");
            self.encoder
                .to_f16(stream, &pos_device.slice(..), &mut pos_f16.slice_mut(..))
                .expect("to fp16");
            let decode = |buffer: &CudaSlice<u16>| {
                stream
                    .clone_dtoh(buffer)
                    .expect("download")
                    .into_iter()
                    .map(f16_to_f32)
                    .collect::<Vec<f32>>()
            };

            // NaN until written, so a row the kernels skip shows
            let mut out_f16 = stream.clone_htod(&vec![0x7e00u16; rows]).expect("upload");
            self.attention
                .rel_pos_f16(
                    stream,
                    &qkv_f16.slice(..),
                    &pos_f16.slice(..),
                    &bias_u.slice(..),
                    &bias_v.slice(..),
                    &lens_device.slice(..),
                    &mut out_f16.slice_mut(..),
                    shape,
                )
                .expect("fp16 kernel");
            let mut out_tf32 = stream.clone_htod(&vec![f32::NAN; rows]).expect("upload");
            self.attention
                .rel_pos_tf32(
                    stream,
                    &qkv_device.slice(..),
                    &pos_device.slice(..),
                    &bias_u.slice(..),
                    &bias_v.slice(..),
                    &lens_device.slice(..),
                    &mut out_tf32.slice_mut(..),
                    shape,
                )
                .expect("tf32 kernel");

            [
                FusedRun {
                    name: "FP16",
                    out: decode(&out_f16),
                    reference: attention_reference(
                        &decode(&qkv_f16),
                        &decode(&pos_f16),
                        biases,
                        lens,
                        frames,
                        heads,
                    ),
                },
                FusedRun {
                    name: "TF32",
                    out: stream.clone_dtoh(&out_tf32).expect("download"),
                    reference: attention_reference(qkv, pos, biases, lens, frames, heads),
                },
            ]
        }
    }

    impl FusedRun {
        /// Holds the output to its reference, adding to `failures` when it is off by
        /// more than rounding
        fn hold(
            &self,
            set: &str,
            lens: &[usize],
            frames: usize,
            heads: usize,
            failures: &mut Vec<String>,
        ) {
            let (own, per_head, per_row) =
                attention_error(&self.out, &self.reference, lens, frames, heads);
            let name = self.name;
            println!("{set} {name}: rel {own:.2e} against f64 on its inputs");
            let list = |values: &[f64]| {
                values
                    .iter()
                    .map(|value| format!("{value:.1e}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            println!("  per head {}", list(&per_head));
            println!("  per row in warp {}", list(&per_row));
            // 10-bit operands and FP16 weights give about 1e-3; a wrong fragment,
            // shift or mask is off by order 1
            if own > 1e-2 {
                failures.push(format!("{set} {name}: {own:.2e}"));
            }
        }
    }

    /// Feeds NeMo's projections of the first and last layer to each fused kernel and
    /// holds the merged context against the same attention in f64 on the operands the
    /// kernel read, which separates the kernel's own error from its input rounding, and
    /// that f64 attention against NeMo's
    #[test]
    #[ignore = "needs a CUDA GPU and the NeMo references; `just gpu-parity` runs it on gpu0"]
    fn gpu_fused_attention_kernels_match_nemo() {
        let Some(kernels) = FusedKernels::open() else {
            println!("no fused attention on this GPU");
            return;
        };
        let file = weights(ParakeetVersion::V3);
        let spec = file.parakeet_spec(&ParakeetDeployment::v3()).expect("spec");
        let (d, heads) = (spec.encoder.d_model, spec.encoder.heads);
        let mut failures = Vec::new();
        // the references dump the attention operands of the first and the last layer
        for (index, set) in [0, 23]
            .into_iter()
            .flat_map(|index| ["b1", "b8", "long-full"].map(|set| (index, set)))
        {
            let tensor = |name: &str| {
                file.tensor(
                    &format!("encoder.layers.{index}.self_attn.{name}"),
                    &[heads, d / heads],
                )
                .expect("bias")
                .to_vec()
            };
            let (bias_u, bias_v) = (tensor("pos_bias_u"), tensor("pos_bias_v"));

            let reference = NemoReference::open(ParakeetVersion::V3, set);
            let layer = |name: &str| {
                reference
                    .f32(&format!("encoder.layers.{index}.self_attn.{name}"))
                    .iter()
                    .copied()
                    .collect::<Vec<f32>>()
            };
            let set = format!("layer {index} {set}");
            let (q, k, v) = (
                layer("linear_q.out"),
                layer("linear_k.out"),
                layer("linear_v.out"),
            );
            let pos = layer("linear_pos.out");
            let expected = layer("linear_out.in");
            let lens: Vec<usize> = reference
                .i64("encoder.out_len")
                .iter()
                .map(|len| *len as usize)
                .collect();
            let frames = q.len() / (lens.len() * d);
            let mut qkv = Vec::with_capacity(3 * q.len());
            for row in 0..lens.len() * frames {
                for part in [&q, &k, &v] {
                    qkv.extend_from_slice(&part[row * d..(row + 1) * d]);
                }
            }

            let runs = kernels.run(&qkv, &pos, [&bias_u, &bias_v], &lens, frames, heads);
            // TF32 reads the projections as they are, so its reference is NeMo's
            // attention in f64
            let (rel, _, _) = attention_error(&runs[1].reference, &expected, &lens, frames, heads);
            println!("{set}: f64 reference against NeMo rel {rel:.2e}");
            // NeMo's FP32 attention lands about 1e-6 from it, so the kernels' own bound
            // holds them to NeMo's mapping, not only to a shared misreading of it
            if rel > 1e-5 {
                failures.push(format!("{set} reference: {rel:.2e}"));
            }
            for run in &runs {
                let (nemo, _, _) = attention_error(&run.out, &expected, &lens, frames, heads);
                println!("{set} {}: rel {nemo:.2e} against NeMo", run.name);
                run.hold(&set, &lens, frames, heads, &mut failures);
            }
        }
        assert!(failures.is_empty(), "kernel error: {failures:?}");
    }

    /// Runs both fused kernels on random inputs from 1 frame to past one key tile,
    /// which the NeMo sets never reach and the encoder runs fused at any batch, with
    /// mixed lengths so blocks see partial query and key tiles and padded keys
    #[test]
    #[ignore = "needs a CUDA GPU; `just gpu-parity` runs it on gpu0"]
    fn gpu_fused_attention_kernels_handle_short_inputs() {
        let Some(kernels) = FusedKernels::open() else {
            println!("no fused attention on this GPU");
            return;
        };
        let (heads, d) = (8, 8 * crate::cuda::kernels::ATTENTION_HEAD_DIM);
        // xorshift, so a failure reproduces
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut uniform = |scale: f32, len: usize| {
            (0..len)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    ((state >> 40) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
                })
                .collect::<Vec<f32>>()
        };
        let mut failures = Vec::new();
        // the key tiles are 64 frames in FP16 and 32 in TF32, a warp's query tile 16
        for frames in [1usize, 7, 16, 31, 33, 64, 65, 96] {
            let lens = [frames, frames.div_ceil(2), 1];
            let qkv = uniform(2.0, lens.len() * frames * 3 * d);
            let pos = uniform(2.0, (2 * frames - 1) * d);
            let (bias_u, bias_v) = (uniform(0.5, d), uniform(0.5, d));
            let set = format!("{frames} frames");
            for run in kernels.run(&qkv, &pos, [&bias_u, &bias_v], &lens, frames, heads) {
                run.hold(&set, &lens, frames, heads, &mut failures);
            }
        }
        assert!(failures.is_empty(), "kernel error: {failures:?}");
    }

    /// The closing Conformer layers, which amplify any rounding difference several
    /// times over, so two 10-bit paths diverge there by chance
    const AMPLIFYING_LAYERS: usize = 4;

    /// Fused attention at every shape and at none, to compare the two paths
    const ALWAYS: super::FusedShapes = super::FusedShapes {
        short_frames: usize::MAX,
        min_rows: 0,
    };
    const NEVER: super::FusedShapes = super::FusedShapes {
        short_frames: 0,
        min_rows: usize::MAX,
    };

    /// Precisions with a fused attention kernel
    const FUSED_PRECISIONS: [(&str, EncoderMath); 2] = [
        ("FP16", EncoderMath::Fp16),
        ("TF32", EncoderMath::Single(Math::Tf32)),
    ];

    /// Runs the FP16 and TF32 encoders with the fused attention and with the cuBLAS
    /// attention on short, batched and long inputs, and holds the fused one to the
    /// cuBLAS one's error against NeMo FP32 layer by layer until the closing layers
    /// amplify rounding, and to the fast encoders' bound at the output
    #[test]
    #[ignore = "needs a CUDA GPU and the NeMo references; `just gpu-parity` runs it on gpu0"]
    fn gpu_fused_attention_matches_cublas() {
        let runtime = CudaRuntime::open(0, Options::default()).expect("CUDA runtime");
        runtime.prepare(Library::Cublas).expect("cuBLAS");
        let file = weights(ParakeetVersion::V3);
        let spec = file.parakeet_spec(&ParakeetDeployment::v3()).expect("spec");
        let mut failures = Vec::new();
        for (name, math) in FUSED_PRECISIONS {
            let mut encoder =
                CudaEncoder::load(&runtime, &file, spec.encoder, math).expect("encoder");
            if encoder.fused.is_none() {
                println!("no fused attention on this GPU");
                return;
            }

            for set in ["b1", "b8", "long-full"] {
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
                let expected = reference.f32("encoder.out").permuted_axes(vec![0, 2, 1]);
                let expected = expected.as_standard_layout().into_owned();
                let mut encode = |fused: bool| {
                    if let Some(attention) = encoder.fused.as_mut() {
                        attention.shapes = if fused { ALWAYS } else { NEVER };
                    }
                    let mut ws = encoder
                        .workspace(&runtime, batch, mel_frames)
                        .expect("workspace");
                    assert_eq!(
                        !matches!(ws.attention, super::AttentionBuffers::Unfused(_)),
                        fused,
                        "{name} {set}: attention path"
                    );
                    encoder
                        .set_input(&runtime, &mut ws, &lens, |mel| {
                            mel.copy_from_slice(&time_major)
                        })
                        .expect("input");
                    encoder
                        .enqueue_subsampling(&runtime, &mut ws)
                        .expect("subsampling");
                    let mut layers = Vec::with_capacity(spec.encoder.layers);
                    for index in 0..spec.encoder.layers {
                        encoder
                            .enqueue_layer(&runtime, &mut ws, index)
                            .expect("layer");
                        let output = runtime.stream().clone_dtoh(ws.output()).expect("download");
                        let layer = reference.f32(&format!("encoder.layers.{index}.out"));
                        layers.push(compare(&output, &layer, ws.output_lens()).0);
                    }

                    let output = runtime.stream().clone_dtoh(ws.output()).expect("download");
                    (compare(&output, &expected, ws.output_lens()), layers)
                };
                let ((fused_rel, fused_max), fused_layers) = encode(true);
                let ((cublas_rel, cublas_max), cublas_layers) = encode(false);
                println!(
                    "v3 {set} {name} encoder.out against NeMo: fused rel {fused_rel:.2e} max {fused_max:.2e}, cuBLAS rel {cublas_rel:.2e} max {cublas_max:.2e}"
                );
                // both paths round to 10 bits, so they track each other within 5 %
                // until the last layers, which multiply any difference about 3x
                // each and leave which path ends closer to chance (2.6e-2 against
                // 6.1e-3 on FP16 b1, 1.8e-3 against 3.1e-3 on FP16 b8, RTX 5090);
                // a wrong score, mask or shift is off by order 1 from layer 0
                let tracked = spec.encoder.layers - AMPLIFYING_LAYERS;
                for (index, (fused, cublas)) in fused_layers.iter().zip(&cublas_layers).enumerate()
                {
                    println!("  layer {index:2}: fused {fused:.2e} cuBLAS {cublas:.2e}");
                    if index < tracked && *fused > 1.25 * cublas + 1e-5 {
                        failures.push(format!(
                            "{name} {set} layer {index}: fused {fused:.2e} against cuBLAS {cublas:.2e}"
                        ));
                    }
                }
                // the bound `gpu_fast_encoders_stay_near_nemo` holds the cuBLAS path to
                if fused_rel >= 5e-2 {
                    failures.push(format!("{name} {set}: fused relative L2 {fused_rel:.2e}"));
                }
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }

    /// Times FP16 and TF32 encoder graph replays with the cuBLAS and the fused
    /// attention over a grid of batch rows and input lengths, to place
    /// [`FUSED_FP16`] and [`FUSED_TF32`], and marks the shapes where they pick the
    /// slower path
    ///
    /// [`FUSED_FP16`]: super::FUSED_FP16
    /// [`FUSED_TF32`]: super::FUSED_TF32
    #[test]
    #[ignore = "a measurement on a CUDA GPU; run it alone on gpu0 with --nocapture"]
    fn bench_fused_attention_crossover() {
        let runtime = CudaRuntime::open(0, Options::default()).expect("CUDA runtime");
        runtime.prepare(Library::Cublas).expect("cuBLAS");
        let file = weights(ParakeetVersion::V3);
        let spec = file.parakeet_spec(&ParakeetDeployment::v3()).expect("spec");
        // LibriSpeech utterances run one at a time up to about 3500 mel frames, the
        // queue batches up to 16 of them, and long-form windows are 3008
        let batches = [1, 2, 3, 4, 6, 8, 12, 16];
        let lengths = [200, 400, 800, 1200, 1600, 2400, 3008, 3600];
        for (name, math) in FUSED_PRECISIONS {
            let mut encoder =
                CudaEncoder::load(&runtime, &file, spec.encoder, math).expect("encoder");
            let Some(configured) = encoder.fused.as_ref().map(|fused| fused.shapes) else {
                panic!("no fused attention on this GPU");
            };
            println!("{name}: fused / cuBLAS replay time, * where {configured:?} picks the slower");
            println!(
                "batch {}",
                lengths.map(|mel| format!("{:>9}", mel / 8)).join("")
            );
            for batch in batches {
                let mut row = format!("{batch:5}");
                for mel_frames in lengths {
                    let mut times = [0f64; 2];
                    for (slot, shapes) in [(0, NEVER), (1, ALWAYS)] {
                        if let Some(attention) = encoder.fused.as_mut() {
                            attention.shapes = shapes;
                        }
                        times[slot] = replay_ms(&runtime, &encoder, batch, mel_frames);
                    }
                    let frames = encoder.feature_maps(batch, mel_frames)[3].frames;
                    let fused_faster = times[1] < times[0];
                    let wrong = configured.takes(batch, frames) != fused_faster;
                    let mark = if wrong { '*' } else { ' ' };
                    row += &format!("  {:6.3}{mark}", times[1] / times[0]);
                }
                println!("{row}");
            }
        }
    }

    /// Mean graph replay time of `encoder` on `batch` rows of `mel_frames` synthetic
    /// mel frames, in milliseconds
    fn replay_ms(
        runtime: &CudaRuntime,
        encoder: &CudaEncoder,
        batch: usize,
        mel_frames: usize,
    ) -> f64 {
        const REPLAYS: u32 = 50;
        let mut ws = encoder
            .workspace(runtime, batch, mel_frames)
            .expect("workspace");
        // a fixed LCG keeps the input the same for both paths
        let mut state = 0x2545_f491_u32;
        encoder
            .set_input(runtime, &mut ws, &vec![mel_frames; batch], |mel| {
                for value in mel {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    *value = (state >> 8) as f32 / (1u32 << 24) as f32 * 4.0 - 2.0;
                }
            })
            .expect("input");
        encoder.enqueue(runtime, &mut ws).expect("eager");
        // SAFETY: as in `time_graph`, the graph drops before `ws`
        let graph = unsafe { runtime.capture_nonempty(|| encoder.enqueue(runtime, &mut ws)) }
            .expect("capture");
        graph.launch().expect("warm-up");
        runtime.synchronize().expect("sync");
        let start = std::time::Instant::now();
        for _ in 0..REPLAYS {
            graph.launch().expect("replay");
        }
        runtime.synchronize().expect("sync");
        start.elapsed().as_secs_f64() * 1e3 / f64::from(REPLAYS)
    }
}
