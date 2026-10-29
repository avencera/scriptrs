//! FP8 and block-scaled encoder linear layers through cuBLASLt
//!
//! cuBLAS's `cublasGemmEx` takes no operand scales, so the quantized products of
//! [`EncoderMath::Quantized`](crate::cuda::encoder::EncoderMath) call cuBLASLt on
//! a handle of their own. cudakit's cudarc carries the CUDA 12.4 bindings, which
//! predate the block-scale modes and the FP4 and UE8M0 types, so the descriptor
//! attributes and matrix layouts go through the raw entry points with the values of
//! CUDA 13's `cublasLt.h`; the library itself is whatever cudakit loaded.
//!
//! Each product shape gets a plan the first time it runs: the heuristic's candidates
//! are timed into a scratch output and the fastest is kept, as cuBLASLt's first
//! choice underfills the GPU at batch-1 row counts. The encoder's eager pass makes
//! every plan before a graph capture records the product. Accumulation and the
//! output stay FP32

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use cudakit::Error;
use cudakit::cudarc::cublaslt::result as lt;
use cudakit::cudarc::cublaslt::sys;
use cudakit::cudarc::driver::sys::{CUevent_flags, CUstreamCaptureStatus};
use cudakit::cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::cuda::kernels::CudaRuntime;

/// cuBLASLt workspace, which split-K candidates use for partial sums
const WORKSPACE_BYTES: usize = 32 << 20;
/// Heuristic candidates timed per shape
const CANDIDATES: usize = 8;
/// Timed launches per candidate, after as many untimed ones
const TIMED_LAUNCHES: usize = 10;

// `cublasLtMatmulDescAttributes_t` and friends, from CUDA 13's `cublasLt.h`
const DESC_POINTER_MODE: u32 = 2;
const DESC_TRANSA: u32 = 3;
const DESC_TRANSB: u32 = 4;
const DESC_A_SCALE_POINTER: u32 = 17;
const DESC_B_SCALE_POINTER: u32 = 18;
const DESC_A_SCALE_MODE: u32 = 31;
const DESC_B_SCALE_MODE: u32 = 32;
const POINTER_MODE_DEVICE: i32 = 1;
const OP_N: i32 = 0;
const OP_T: i32 = 1;
const SCALE_SCALAR_32F: i32 = 0;
const SCALE_VEC16_UE4M3: i32 = 1;
const SCALE_VEC32_UE8M0: i32 = 2;
const SCALE_OUTER_VEC_32F: i32 = 3;
// `cudaDataType`, from `library_types.h`
const R_32F: u32 = 0;
const R_8F_E4M3: u32 = 28;
const R_4F_E2M1: u32 = 33;

/// Operand format of a cuBLASLt product; both operands share it
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum LtFormat {
    /// E4M3 with one FP32 scale per operand
    Fp8,
    /// E4M3 with FP32 outer-vector scales: one per output channel of the weight and
    /// one per row of the activation
    Fp8OuterVector,
    /// E4M3 with a UE8M0 scale per 32 inputs
    Mxfp8,
    /// E2M1 with an E4M3 scale per 16 inputs
    Nvfp4,
}

impl LtFormat {
    fn data_type(self) -> u32 {
        match self {
            Self::Nvfp4 => R_4F_E2M1,
            _ => R_8F_E4M3,
        }
    }

    fn scale_mode(self) -> i32 {
        match self {
            Self::Fp8 => SCALE_SCALAR_32F,
            Self::Fp8OuterVector => SCALE_OUTER_VEC_32F,
            Self::Mxfp8 => SCALE_VEC32_UE8M0,
            Self::Nvfp4 => SCALE_VEC16_UE4M3,
        }
    }
}

/// The alpha of a product: a host value, or a device FP32 that a quantize kernel
/// wrote earlier on the stream
#[derive(Debug, Clone, Copy)]
pub(crate) enum Alpha {
    Host(f32),
    Device(u64),
}

/// `c = alpha * (a · wᵀ) + beta * c` over quantized operands, as device pointers
///
/// `w` is a PyTorch `[out, inputs]` weight and `a` a `[rows, inputs]` activation, both
/// row-major in `format`, and `c` is FP32 `[rows, out]`. The scale pointers follow
/// the format. With a device alpha, `beta` must be 0 or 1
#[derive(Debug, Clone, Copy)]
pub(crate) struct LtProduct {
    pub(crate) format: LtFormat,
    pub(crate) rows: usize,
    pub(crate) out: usize,
    pub(crate) inputs: usize,
    pub(crate) w: u64,
    pub(crate) w_scale: u64,
    pub(crate) a: u64,
    pub(crate) a_scale: u64,
    pub(crate) c: u64,
    pub(crate) alpha: Alpha,
    pub(crate) beta: f32,
}

/// Raw cuBLASLt entry points whose enum arguments cudakit's bindings lack
struct Raw {
    set_attribute:
        unsafe extern "C" fn(sys::cublasLtMatmulDesc_t, u32, *const c_void, usize) -> i32,
    create_layout:
        unsafe extern "C" fn(*mut sys::cublasLtMatrixLayout_t, u32, u64, u64, i64) -> i32,
}

fn raw() -> Result<&'static Raw, Error> {
    static RAW: OnceLock<Result<Raw, String>> = OnceLock::new();
    RAW.get_or_init(|| {
        // SAFETY: checks for the library without loading symbols
        if !unsafe { sys::is_culib_present() } {
            return Err("libcublasLt is not loadable".to_owned());
        }
        // SAFETY: the library is present, and both symbols have these C signatures in
        // every cuBLASLt from 11 on, with the enums passed as 32-bit integers
        unsafe {
            let library = sys::culib();
            Ok(Raw {
                set_attribute: *library
                    .get(b"cublasLtMatmulDescSetAttribute\0")
                    .map_err(|error| error.to_string())?,
                create_layout: *library
                    .get(b"cublasLtMatrixLayoutCreate\0")
                    .map_err(|error| error.to_string())?,
            })
        }
    })
    .as_ref()
    .map_err(|reason| Error::Unsupported {
        context: "cublasLt",
        reason: reason.clone(),
    })
}

fn status(context: &'static str, code: i32) -> Result<(), Error> {
    match code {
        0 => Ok(()),
        code => Err(Error::Unsupported {
            context,
            reason: format!("cuBLASLt status {code}"),
        }),
    }
}

fn lt_error(context: &'static str) -> impl Fn(lt::CublasError) -> Error {
    move |error| Error::Unsupported {
        context,
        reason: format!("{error:?}"),
    }
}

/// Sets one descriptor attribute to `value`
///
/// # Safety
///
/// `desc` is live and `attribute` takes a `T`
unsafe fn set<T>(desc: sys::cublasLtMatmulDesc_t, attribute: u32, value: &T) -> Result<(), Error> {
    let raw = raw()?;
    // SAFETY: the caller's contract
    let code = unsafe {
        (raw.set_attribute)(
            desc,
            attribute,
            std::ptr::from_ref(value).cast(),
            size_of::<T>(),
        )
    };
    status("cublasLt descriptor", code)
}

/// A column-major layout of `rows` by `cols` elements of `data_type`
fn layout(data_type: u32, rows: usize, cols: usize) -> Result<sys::cublasLtMatrixLayout_t, Error> {
    let raw = raw()?;
    let mut layout = std::ptr::null_mut();
    // SAFETY: `layout` is a valid out-pointer
    let code = unsafe {
        (raw.create_layout)(
            &raw mut layout,
            data_type,
            rows as u64,
            cols as u64,
            rows as i64,
        )
    };
    status("cublasLt layout", code)?;
    Ok(layout)
}

/// The descriptors of one product shape and the algorithm chosen for it
struct Plan {
    desc: sys::cublasLtMatmulDesc_t,
    w: sys::cublasLtMatrixLayout_t,
    a: sys::cublasLtMatrixLayout_t,
    c: sys::cublasLtMatrixLayout_t,
    algo: sys::cublasLtMatmulAlgo_t,
}

impl Drop for Plan {
    fn drop(&mut self) {
        // SAFETY: the plan owns its descriptors, and no queued product reads them
        // after its launch returned
        unsafe {
            let _ = lt::destroy_matrix_layout(self.w);
            let _ = lt::destroy_matrix_layout(self.a);
            let _ = lt::destroy_matrix_layout(self.c);
            let _ = lt::destroy_matmul_desc(self.desc);
        }
    }
}

/// Plan key: format, rows, out, inputs and whether alpha is on the device
type PlanKey = (LtFormat, usize, usize, usize, bool);

/// A cuBLASLt handle, its workspace and the plan chosen for each product shape
pub(crate) struct LtGemm {
    handle: sys::cublasLtHandle_t,
    workspace: CudaSlice<u8>,
    /// `[0, 1]`, the device betas of products with a device alpha
    betas: CudaSlice<f32>,
    plans: Mutex<HashMap<PlanKey, Plan>>,
}

// SAFETY: the handle and descriptors are used only on the runtime's stream, under
// the plan lock, and cuBLASLt handles may move between threads
unsafe impl Send for LtGemm {}
// SAFETY: as above
unsafe impl Sync for LtGemm {}

impl Drop for LtGemm {
    fn drop(&mut self) {
        if let Ok(plans) = self.plans.get_mut() {
            plans.clear();
        }
        // SAFETY: the plans that used the handle are gone
        let _ = unsafe { lt::destroy_handle(self.handle) };
    }
}

impl LtGemm {
    pub(crate) fn new(runtime: &CudaRuntime) -> Result<Self, Error> {
        raw()?;
        let handle = lt::create_handle().map_err(lt_error("cublasLt create"))?;
        let stream = runtime.stream();
        Ok(Self {
            handle,
            workspace: stream.alloc_zeros::<u8>(WORKSPACE_BYTES)?,
            betas: stream.clone_htod(&[0.0f32, 1.0])?,
            plans: Mutex::new(HashMap::new()),
        })
    }

    /// Whether this GPU and cuBLASLt run `format` products with FP32 output, judged by
    /// whether the heuristic offers an algorithm for a typical encoder shape
    pub(crate) fn supports(&self, format: LtFormat) -> bool {
        let built = self.descriptors(format, 128, 1024, 1024, format == LtFormat::Nvfp4);
        built.is_ok_and(|plan| self.candidates(&plan).is_ok_and(|found| !found.is_empty()))
    }

    /// Queues `product` on the runtime's stream
    ///
    /// # Safety
    ///
    /// Every pointer is a live device allocation ordered on the runtime's stream and
    /// holds what `product`'s format and dimensions imply
    pub(crate) unsafe fn run(
        &self,
        runtime: &CudaRuntime,
        product: LtProduct,
    ) -> Result<(), Error> {
        let device_alpha = matches!(product.alpha, Alpha::Device(_));
        if device_alpha && product.beta != 0.0 && product.beta != 1.0 {
            return Err(Error::Unsupported {
                context: "cublasLt product",
                reason: format!("a device alpha takes beta 0 or 1, not {}", product.beta),
            });
        }
        let mut plans = self.plans.lock().map_err(|_| Error::Unsupported {
            context: "cublasLt product",
            reason: "poisoned plan lock".to_owned(),
        })?;
        let key = (
            product.format,
            product.rows,
            product.out,
            product.inputs,
            device_alpha,
        );
        let plan = match plans.entry(key) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(self.plan(runtime, product)?),
        };
        // SAFETY: the caller's contract, and the plan matches the product's key
        unsafe { self.launch(runtime, plan, product) }
    }

    /// # Safety
    ///
    /// As [`Self::run`], with `plan` built for `product`
    unsafe fn launch(
        &self,
        runtime: &CudaRuntime,
        plan: &Plan,
        product: LtProduct,
    ) -> Result<(), Error> {
        let stream = runtime.stream();
        // SAFETY: the descriptor is live; the attributes take device pointers
        unsafe {
            set(plan.desc, DESC_A_SCALE_POINTER, &product.w_scale)?;
            set(plan.desc, DESC_B_SCALE_POINTER, &product.a_scale)?;
        }
        let (workspace, _workspace_sync) = self.workspace.device_ptr(stream);
        let (betas, _betas_sync) = self.betas.device_ptr(stream);
        let (alpha, beta): (*const c_void, *const c_void) = match product.alpha {
            Alpha::Host(alpha) => (
                std::ptr::from_ref(&alpha).cast(),
                std::ptr::from_ref(&product.beta).cast(),
            ),
            Alpha::Device(alpha) => (
                alpha as *const c_void,
                (betas + if product.beta == 0.0 { 0 } else { 4 }) as *const c_void,
            ),
        };
        // SAFETY: the caller's contract; host scalars outlive the call, which reads
        // them before it returns
        unsafe {
            lt::matmul(
                self.handle,
                plan.desc,
                alpha,
                beta,
                product.w as *const c_void,
                plan.w,
                product.a as *const c_void,
                plan.a,
                product.c as *const c_void,
                plan.c,
                product.c as *mut c_void,
                plan.c,
                &raw const plan.algo,
                workspace as *mut c_void,
                WORKSPACE_BYTES,
                stream.cu_stream().cast(),
            )
        }
        .map_err(lt_error("cublasLt matmul"))
    }

    /// The descriptors of a product, with the heuristic's first algorithm unset
    fn descriptors(
        &self,
        format: LtFormat,
        rows: usize,
        out: usize,
        inputs: usize,
        device_alpha: bool,
    ) -> Result<Plan, Error> {
        let desc = lt::create_matmul_desc(
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cudaDataType::CUDA_R_32F,
        )
        .map_err(lt_error("cublasLt descriptor"))?;
        let data_type = format.data_type();
        // column-major `cᵀ [out, rows] = w(T) · aᵀ`, as in the FP16 products: the
        // row-major `[out, inputs]` weight read column-major is `wᵀ`, hence `T`
        let plan = Plan {
            desc,
            w: layout(data_type, inputs, out)?,
            a: layout(data_type, inputs, rows)?,
            c: layout(R_32F, out, rows)?,
            // SAFETY: a plain C struct for which all-zero bits are valid
            algo: unsafe { std::mem::zeroed() },
        };
        // SAFETY: the descriptor is live and each attribute takes a 32-bit enum
        unsafe {
            set(desc, DESC_TRANSA, &OP_T)?;
            set(desc, DESC_TRANSB, &OP_N)?;
            set(desc, DESC_A_SCALE_MODE, &format.scale_mode())?;
            set(desc, DESC_B_SCALE_MODE, &format.scale_mode())?;
            if device_alpha {
                set(desc, DESC_POINTER_MODE, &POINTER_MODE_DEVICE)?;
            }
        }
        Ok(plan)
    }

    /// The heuristic's algorithms for `plan`, best first
    fn candidates(&self, plan: &Plan) -> Result<Vec<sys::cublasLtMatmulAlgo_t>, Error> {
        let err = lt_error("cublasLt heuristic");
        let pref = lt::create_matmul_pref().map_err(&err)?;
        let mut results: [sys::cublasLtMatmulHeuristicResult_t; CANDIDATES] =
            // SAFETY: a plain C struct for which all-zero bits are valid
            unsafe { std::mem::zeroed() };
        let mut found = 0;
        let workspace_bytes = WORKSPACE_BYTES;
        // SAFETY: every descriptor is live, and `results` holds `CANDIDATES` entries
        let queried = unsafe {
            lt::set_matmul_pref_attribute(
                pref,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                (&raw const workspace_bytes).cast(),
                size_of::<usize>(),
            )
            .and_then(|()| {
                sys::cublasLtMatmulAlgoGetHeuristic(
                    self.handle,
                    plan.desc,
                    plan.w,
                    plan.a,
                    plan.c,
                    plan.c,
                    pref,
                    CANDIDATES as i32,
                    results.as_mut_ptr(),
                    &raw mut found,
                )
                .result()
            })
        };
        // SAFETY: the preference is live and unused past this point
        let _ = unsafe { lt::destroy_matmul_pref(pref) };
        queried.map_err(&err)?;
        Ok(results[..found as usize]
            .iter()
            .filter(|result| result.state == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS)
            .map(|result| result.algo)
            .collect())
    }

    /// Builds the plan for `product` and times the heuristic's candidates into a
    /// scratch output, so tuning leaves `c` alone
    fn plan(&self, runtime: &CudaRuntime, product: LtProduct) -> Result<Plan, Error> {
        let device_alpha = matches!(product.alpha, Alpha::Device(_));
        let mut plan = self.descriptors(
            product.format,
            product.rows,
            product.out,
            product.inputs,
            device_alpha,
        )?;
        let candidates = self.candidates(&plan)?;
        let first = *candidates.first().ok_or_else(|| Error::Unsupported {
            context: "cublasLt plan",
            reason: format!(
                "no {:?} algorithm for {} x {} x {}",
                product.format, product.rows, product.out, product.inputs
            ),
        })?;
        plan.algo = first;

        let stream = runtime.stream();
        // a capture can't time anything; keep the heuristic's choice
        if stream.capture_status()? != CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
            return Ok(plan);
        }
        let mut scratch = stream.alloc_zeros::<f32>(product.rows * product.out)?;
        let (c, _c_sync) = scratch.device_ptr_mut(stream);
        let timing = LtProduct {
            c,
            beta: 0.0,
            ..product
        };
        let (mut best, mut best_algo) = (f32::INFINITY, first);
        for algo in candidates {
            plan.algo = algo;
            // SAFETY: the caller's operands, and `scratch` holds `rows * out`
            let run = || unsafe { self.launch(runtime, &plan, timing) };
            // a candidate the launch rejects is skipped
            if run().is_err() {
                continue;
            }
            for _ in 1..TIMED_LAUNCHES {
                run()?;
            }
            let start = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
            for _ in 0..TIMED_LAUNCHES {
                run()?;
            }
            let end = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
            let elapsed = start.elapsed_ms(&end)?;
            if elapsed < best {
                (best, best_algo) = (elapsed, algo);
            }
        }
        plan.algo = best_algo;
        Ok(plan)
    }
}

#[cfg(test)]
mod tests {
    /// Loads the raw entry points, which needs the library but no GPU
    #[test]
    #[ignore = "needs libcublasLt; `just gpu-parity` runs it"]
    fn gpu_raw_entry_points_load() {
        super::raw().expect("cuBLASLt entry points");
    }
}
