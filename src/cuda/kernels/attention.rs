//! Launchers for the fused relative-position attention in the kernel crate's
//! `attention` area

use cudakit::cudarc::driver::sys::CUfunction_attribute;
use cudakit::cudarc::driver::{
    CudaFunction, CudaStream, CudaView, CudaViewMut, DevicePtr, DeviceRepr, LaunchConfig,
    PushKernelArg,
};
use cudakit::{Error, check_len, element_count};

use super::{CudaRuntime, Heads, KernelModule, len, len_mut, u32_of};

/// Head width the fused attention is unrolled for, the kernels' `HEAD_DIM`
pub(crate) const ATTENTION_HEAD_DIM: usize = 128;

/// One fused attention kernel with its block shape limits
struct FusedKernel {
    function: CudaFunction,
    /// Most warps per block, 16 query rows each: as many as share each key and value
    /// tile in one SM's shared memory on sm_120
    max_warps: usize,
    /// Dynamic shared memory of a block of `warps` warps, the kernels' `shared_bytes_*`
    shared_bytes: fn(usize) -> usize,
}

/// The FP16 kernel's `shared_bytes_f16`: 64-key tiles of FP16 rows padded to 136
const fn attention_shared_bytes_f16(warps: usize) -> usize {
    (3 * 64 + 16 * warps) * (ATTENTION_HEAD_DIM + 8) * 2
}

/// The TF32 kernel's `shared_bytes_tf32`: 32-key tiles of FP32 rows padded to 132,
/// and the bias difference
const fn attention_shared_bytes_tf32(warps: usize) -> usize {
    (3 * 32 + 16 * warps) * (ATTENTION_HEAD_DIM + 4) * 4 + ATTENTION_HEAD_DIM * 4
}

/// The `attention` area's fused relative-position attention, `sm_80` and newer
pub(crate) struct AttentionKernels {
    rel_pos_f16: FusedKernel,
    rel_pos_tf32: FusedKernel,
    multiprocessors: usize,
}

impl AttentionKernels {
    /// Loads the area, or `None` on a GPU it does not ship for, which keeps the cuBLAS
    /// attention
    pub(crate) fn load(runtime: &CudaRuntime) -> Result<Option<Self>, Error> {
        let Some(request) = runtime.embedded_request(KernelModule::Attention)? else {
            return Ok(None);
        };
        let module = runtime.load(request)?;
        let kernel = |name, max_warps, shared_bytes: fn(usize) -> usize| {
            let function = module.function(name)?;
            // above 48 KB a block must opt in to its dynamic shared memory
            function.set_attribute(
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                i32::try_from(shared_bytes(max_warps)).unwrap_or(i32::MAX),
            )?;
            Ok::<_, Error>(FusedKernel {
                function,
                max_warps,
                shared_bytes,
            })
        };

        Ok(Some(Self {
            rel_pos_f16: kernel("attention_rel_pos_f16", 8, attention_shared_bytes_f16)?,
            rel_pos_tf32: kernel("attention_rel_pos_tf32", 4, attention_shared_bytes_tf32)?,
            multiprocessors: runtime.device().multiprocessors().get() as usize,
        }))
    }

    /// Warps per block of `kernel` for `shape`: the most, for the best key and value
    /// reuse, that still gives every SM a block
    fn warps(&self, kernel: &FusedKernel, shape: Heads) -> usize {
        let mut warps = kernel.max_warps;
        while warps > 1 {
            let blocks = shape.frames.div_ceil(16 * warps) * shape.heads * shape.batch;
            if blocks >= self.multiprocessors {
                break;
            }
            warps /= 2;
        }
        warps
    }

    /// Relative-position attention from FP16 projections to the merged FP16 context
    ///
    /// `qkv` is the fused `[batch * frames, 3 * d_model]` projection, `pos` the
    /// projected `[2 * frames - 1, d_model]` position table, and `out` the `[batch *
    /// frames, d_model]` context; `lens` holds the valid frames per batch row
    #[expect(
        clippy::too_many_arguments,
        reason = "launchers mirror their kernel's parameter list so each launch reads against it"
    )]
    pub(crate) fn rel_pos_f16(
        &self,
        stream: &CudaStream,
        qkv: &CudaView<'_, u16>,
        pos: &CudaView<'_, u16>,
        bias_u: &CudaView<'_, f32>,
        bias_v: &CudaView<'_, f32>,
        lens_frames: &CudaView<'_, u32>,
        out: &mut CudaViewMut<'_, u16>,
        shape: Heads,
    ) -> Result<(), Error> {
        self.rel_pos(
            &self.rel_pos_f16,
            stream,
            [qkv, pos],
            [bias_u, bias_v],
            lens_frames,
            out,
            shape,
        )
    }

    /// [`rel_pos_f16`](Self::rel_pos_f16) from FP32 projections to the FP32 context,
    /// with TF32 products
    #[expect(
        clippy::too_many_arguments,
        reason = "launchers mirror their kernel's parameter list so each launch reads against it"
    )]
    pub(crate) fn rel_pos_tf32(
        &self,
        stream: &CudaStream,
        qkv: &CudaView<'_, f32>,
        pos: &CudaView<'_, f32>,
        bias_u: &CudaView<'_, f32>,
        bias_v: &CudaView<'_, f32>,
        lens_frames: &CudaView<'_, u32>,
        out: &mut CudaViewMut<'_, f32>,
        shape: Heads,
    ) -> Result<(), Error> {
        self.rel_pos(
            &self.rel_pos_tf32,
            stream,
            [qkv, pos],
            [bias_u, bias_v],
            lens_frames,
            out,
            shape,
        )
    }

    /// Checks the buffers against `shape` and launches `kernel`, which takes `qkv`,
    /// `pos` and `out` as `T`
    #[expect(
        clippy::too_many_arguments,
        reason = "launchers mirror their kernel's parameter list so each launch reads against it"
    )]
    fn rel_pos<T: DeviceRepr>(
        &self,
        kernel: &FusedKernel,
        stream: &CudaStream,
        [qkv, pos]: [&CudaView<'_, T>; 2],
        [bias_u, bias_v]: [&CudaView<'_, f32>; 2],
        lens_frames: &CudaView<'_, u32>,
        out: &mut CudaViewMut<'_, T>,
        shape: Heads,
    ) -> Result<(), Error> {
        if shape.head_dim != ATTENTION_HEAD_DIM || shape.frames == 0 {
            return Err(Error::Unsupported {
                context: "fused attention",
                reason: format!(
                    "the kernel takes {ATTENTION_HEAD_DIM}-wide heads and at least one frame, not {} by {}",
                    shape.head_dim, shape.frames
                ),
            });
        }
        let n = shape.len()?;
        let d_model = element_count("attention width", &[shape.heads, shape.head_dim])?;
        check_len("fused qkv", element_count("fused qkv", &[3, n])?, qkv.len())?;
        check_len(
            "position table",
            element_count("position table", &[2 * shape.frames - 1, d_model])?,
            pos.len(),
        )?;
        check_len("pos_bias_u", d_model, bias_u.len())?;
        check_len("pos_bias_v", d_model, bias_v.len())?;
        check_len("frame lengths", shape.batch, lens_frames.len())?;
        check_len("attention output", n, out.len())?;
        // `cp.async`, `ldmatrix` and the vector stores move 16 bytes at a time, which an
        // offset view of a whole allocation can break
        for (name, address) in [
            ("fused qkv", qkv.device_ptr(stream).0),
            ("position table", pos.device_ptr(stream).0),
            ("attention output", out.device_ptr(stream).0),
        ] {
            if address % 16 != 0 {
                return Err(Error::Unsupported {
                    context: "fused attention",
                    reason: format!("the {name} must start 16-byte aligned"),
                });
            }
        }

        let warps = self.warps(kernel, shape);
        let config = LaunchConfig {
            grid_dim: (
                u32_of("query blocks", shape.frames.div_ceil(16 * warps))?,
                u32_of("heads", shape.heads)?,
                u32_of("batch", shape.batch)?,
            ),
            block_dim: (32 * warps as u32, 1, 1),
            shared_mem_bytes: (kernel.shared_bytes)(warps) as u32,
        };
        let lens = [
            len(qkv),
            len(pos),
            len(bias_u),
            len(bias_v),
            len(lens_frames),
            len_mut(out),
        ];
        let dims = [
            u32_of("frames", shape.frames)?,
            u32_of("heads", shape.heads)?,
        ];
        // the kernel's softmax runs in base 2, and NeMo divides by `math.sqrt(d_k)`
        let scale = (std::f64::consts::LOG2_E / (shape.head_dim as f64).sqrt()) as f32;
        let mut launch = stream.launch_builder(&kernel.function);
        launch
            .arg(qkv)
            .arg(&lens[0])
            .arg(pos)
            .arg(&lens[1])
            .arg(bias_u)
            .arg(&lens[2]);
        launch
            .arg(bias_v)
            .arg(&lens[3])
            .arg(lens_frames)
            .arg(&lens[4])
            .arg(out)
            .arg(&lens[5]);
        launch.arg(&dims[0]).arg(&dims[1]).arg(&scale);
        // SAFETY: matches `attention_rel_pos_f16` and `attention_rel_pos_tf32`, whose
        // parameter lists differ only in the operand type `T`; the buffers hold exactly
        // the rows `shape` implies, which bound every unchecked access, the grid covers
        // the frames in blocks of `16 * warps` rows, and the shared memory is the block's.
        // `qkv`, `pos` and `out` start 16-byte aligned, as checked above
        unsafe { launch.launch(config) }?;
        Ok(())
    }
}
