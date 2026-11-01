//! Measures the fastest cuBLASLt algorithm for each Parakeet linear-layer shape and
//! row range on this GPU, and writes the table `src/cuda/linears.rs` embeds
//!
//! For each shape, precision and row count on the grid below, every algorithm the
//! heuristic offers runs from a captured graph, as the encoder replays it, and the
//! fastest wins unless it beats the heuristic's first by less than `MARGIN`: a near
//! tie stays with the default rather than with measurement noise. Neighbouring row
//! counts with the same winner merge into one range, which also covers the rows
//! between grid points. Run it on an otherwise idle GPU, then check the speedup with
//! `cuda_bench` and WER before committing the table
//!
//! ```sh
//! cargo run --release --features cuda --example cuda_tune -- \
//!     --out src/cuda/tune/nvidia-geforce-rtx-5090.tsv
//! ```

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use cudakit::cudarc::driver::CudaSlice;
use cudakit::cudarc::driver::sys::CUevent_flags;
use cudakit::{
    AreaPtx, ExtraSymbols, KernelArea, Library, LibraryPolicy, MATMUL_ALIGNMENT, Math, Matmul,
    MatmulAlgo, Operands, Options, Profile, PtxTier, Runtime, TierSet,
};
use eyre::{Result, bail, eyre};

/// A winner must be this much faster than the heuristic's first to replace it
const MARGIN: f64 = 0.03;
/// Algorithms timed per point; the heuristic rarely offers more than this
const CANDIDATES: usize = 16;
/// Products per captured graph, and graph replays per timing, of which the fastest counts
const LAUNCHES: usize = 20;
/// Bytes of weight copies the launches rotate through, past the 5090's 96 MiB L2: the
/// encoder streams 24 layers of weights from DRAM, so a weight the cache still holds
/// from the last launch would favour the wrong algorithm
const ROTATION_BYTES: usize = 256 << 20;
const REPLAYS: usize = 3;

/// A linear layer as `(name, outputs, inputs, precisions, row counts)`
struct Shape {
    name: &'static str,
    n: usize,
    k: usize,
    operands: &'static [Operands],
    rows: fn() -> Vec<usize>,
}

const TF32: Operands = Operands::F32(Math::Tf32);
const FP16: Operands = Operands::F16;

/// Encoder rows are batch rows times frames, and frames come in multiples of 8 (64
/// mel frames per shape bucket, 8x subsampling); the position projection's `2T - 1`
/// rows fall between grid points, where the merged ranges cover them. Long-form runs
/// 16 windows of 376 frames, about 6000 rows
fn encoder_rows() -> Vec<usize> {
    (8..=1024)
        .step_by(8)
        .chain((1088..=8192).step_by(64))
        .collect()
}

/// The joint's output projection runs once per decode step on the batch
fn joint_rows() -> Vec<usize> {
    (1..=16).collect()
}

/// Parakeet TDT 0.6B (v2 and v3): d_model 1024, feed-forward 4096, 256 subsampling
/// channels over 16 mel bins, joint hidden 640. Layer linears run in FP16 under the
/// default precision and TF32 under `Precision::Tf32`; the subsampling output and
/// the joint stay TF32 under both. The subsampling output, 4096 inputs to 1024
/// outputs, has ff2's shape, so ff2's TF32 rows cover it
const SHAPES: &[Shape] = &[
    Shape {
        name: "ff1",
        n: 4096,
        k: 1024,
        operands: &[TF32, FP16],
        rows: encoder_rows,
    },
    Shape {
        name: "ff2",
        n: 1024,
        k: 4096,
        operands: &[TF32, FP16],
        rows: encoder_rows,
    },
    Shape {
        name: "qkv",
        n: 3072,
        k: 1024,
        operands: &[TF32, FP16],
        rows: encoder_rows,
    },
    Shape {
        name: "pw1",
        n: 2048,
        k: 1024,
        operands: &[TF32, FP16],
        rows: encoder_rows,
    },
    // attention output, position projection and second pointwise conv
    Shape {
        name: "d2d",
        n: 1024,
        k: 1024,
        operands: &[TF32, FP16],
        rows: encoder_rows,
    },
    Shape {
        name: "joint encoder",
        n: 640,
        k: 1024,
        operands: &[TF32],
        rows: encoder_rows,
    },
    Shape {
        name: "joint v3",
        n: 8198,
        k: 640,
        operands: &[TF32],
        rows: joint_rows,
    },
    Shape {
        name: "joint v2",
        n: 1030,
        k: 640,
        operands: &[TF32],
        rows: joint_rows,
    },
];

static PROFILE: Profile = Profile {
    consumer: "scriptrs-cuda-tune",
    compiled: TierSet::of(&[PtxTier::Sm75]),
    shipped: TierSet::of(&[PtxTier::Sm75]),
    tier_features: &[],
    libraries: LibraryPolicy::Allowed,
    extra_symbols: ExtraSymbols::NONE,
};

/// The tuner loads no kernels of its own
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct NoKernels;

impl KernelArea for NoKernels {
    const PROFILE: &'static Profile = &PROFILE;

    fn name(self) -> &'static str {
        "none"
    }

    fn variants(self) -> AreaPtx {
        AreaPtx::EMPTY
    }
}

struct Buffers {
    a32: CudaSlice<f32>,
    w32: CudaSlice<f32>,
    a16: CudaSlice<u16>,
    w16: CudaSlice<u16>,
    c: CudaSlice<f32>,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error:?}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let mut args = env::args().skip(1);
    let mut out = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => {
                out = Some(PathBuf::from(
                    args.next().ok_or(eyre!("--out needs a path"))?,
                ))
            }
            other => return Err(eyre!("unknown argument {other}")),
        }
    }
    let out = out.ok_or(eyre!("--out <table.tsv> is required"))?;

    let runtime = Runtime::<NoKernels>::open(0, Options::default())?;
    runtime.prepare(Library::CublasLt)?;
    let device = runtime.device().name().to_owned();
    let mut table = String::new();
    writeln!(
        table,
        "# cuBLASLt algorithms for scriptrs's linear layers, by examples/cuda_tune.rs"
    )?;
    writeln!(
        table,
        "# operands, n, k, first row, last row, algorithm  # config, speedup over the default"
    )?;
    writeln!(table, "device\t{device}")?;
    writeln!(table, "multiprocessors\t{}", runtime.multiprocessor_count())?;
    writeln!(table, "cublaslt\t{}", runtime.cublaslt_version()?)?;

    let stream = runtime.stream();
    let largest = |field: fn(&Shape) -> usize| SHAPES.iter().map(field).max().unwrap_or(0);
    let max_rows = SHAPES
        .iter()
        .flat_map(|shape| (shape.rows)())
        .max()
        .unwrap_or(0);
    let (max_n, max_k) = (largest(|shape| shape.n), largest(|shape| shape.k));
    let fill = |len: usize| -> Vec<f32> {
        (0..len)
            .map(|index| ((index * 7) % 17) as f32 / 8.0 - 1.0)
            .collect()
    };
    // room for the rotation plus two padded copies of the largest weight, the most
    // `fastest` takes when one copy alone passes `ROTATION_BYTES`
    let rotation = |element: usize| ROTATION_BYTES / element + 2 * (max_n * max_k + 8);
    // 0x3c00 is FP16 1.0; the values only need to be finite
    let mut buffers = Buffers {
        a32: stream.clone_htod(&fill(max_rows * max_k))?,
        w32: stream.clone_htod(&fill(rotation(size_of::<f32>())))?,
        a16: stream.clone_htod(&vec![0x3c00_u16; max_rows * max_k])?,
        w16: stream.clone_htod(&vec![0x3c00_u16; rotation(size_of::<u16>())])?,
        c: stream.alloc_zeros(max_rows * max_n)?,
    };

    for shape in SHAPES {
        for &operands in shape.operands {
            let mut winners: Vec<(usize, MatmulAlgo, String)> = Vec::new();
            for rows in (shape.rows)() {
                let (algo, note) = fastest(&runtime, &mut buffers, shape, operands, rows)?;
                eprintln!("{} {:?} rows {rows}: {note}", shape.name, operands);
                winners.push((rows, algo, note));
            }
            let token = match operands {
                Operands::F16 => "fp16",
                Operands::F32(Math::Tf32) => "tf32",
                _ => "fp32",
            };
            // merge runs of one winner; each range starts after the previous one ends
            let mut first = 1;
            let mut index = 0;
            while index < winners.len() {
                let (_, algo, ref note) = winners[index];
                let mut end = index;
                while end + 1 < winners.len() && winners[end + 1].1 == algo {
                    end += 1;
                }
                let last = winners[end].0;
                let words: Vec<String> = algo.0.iter().map(|word| format!("{word:x}")).collect();
                writeln!(
                    table,
                    "{token}\t{}\t{}\t{first}\t{last}\t{}\t# {} {note}",
                    shape.n,
                    shape.k,
                    words.join("."),
                    shape.name
                )?;
                first = last + 1;
                index = end + 1;
            }
        }
    }

    fs::write(&out, table)?;
    eprintln!("wrote {}", out.display());
    Ok(())
}

/// The winning algorithm for one point, and a note on its config and speedup
fn fastest(
    runtime: &Runtime<NoKernels>,
    buffers: &mut Buffers,
    shape: &Shape,
    operands: Operands,
    rows: usize,
) -> Result<(MatmulAlgo, String)> {
    let spec = Matmul::linear(rows, shape.n, shape.k, operands);
    let candidates = runtime.matmul_candidates(spec, CANDIDATES)?;
    let Some(default) = candidates.first() else {
        bail!("cuBLASLt offers no algorithm for {spec:?}");
    };

    let mut timed = Vec::new();
    for candidate in &candidates {
        let Ok(plan) = runtime.matmul_plan_with(spec, candidate.algo) else {
            continue;
        };
        let weights = shape.n * shape.k;
        // whole copies of this shape's weight that fit the rotation, each 16-byte aligned
        let element = match operands {
            Operands::F16 => size_of::<u16>(),
            _ => size_of::<f32>(),
        };
        let copies = (ROTATION_BYTES / (weights * element)).clamp(2, LAUNCHES);
        let stride = weights.next_multiple_of(MATMUL_ALIGNMENT / element);
        let a32 = buffers.a32.slice(..rows * shape.k);
        let a16 = buffers.a16.slice(..rows * shape.k);
        let mut c = buffers.c.slice_mut(..rows * shape.n);
        // SAFETY: every captured buffer lives in `buffers` past the graph's last replay
        let graph = unsafe {
            runtime.capture_nonempty(|| {
                for launch in 0..LAUNCHES {
                    let first = (launch % copies) * stride;
                    match operands {
                        Operands::F16 => {
                            let w16 = buffers.w16.slice(first..first + weights);
                            runtime.matmul(&plan, 1.0, 0.0, &a16, &w16, &mut c)?
                        }
                        _ => {
                            let w32 = buffers.w32.slice(first..first + weights);
                            runtime.matmul(&plan, 1.0, 0.0, &a32, &w32, &mut c)?
                        }
                    }
                }
                Ok::<_, cudakit::Error>(())
            })
        }?;
        graph.launch()?;
        let mut best = f64::INFINITY;
        for _ in 0..REPLAYS {
            let start = runtime
                .stream()
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
            graph.launch()?;
            let end = runtime
                .stream()
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
            best = best.min(f64::from(start.elapsed_ms(&end)?) * 1000.0 / LAUNCHES as f64);
        }
        timed.push((candidate.algo, best));
    }

    let default_time = timed
        .iter()
        .find(|(algo, _)| *algo == default.algo)
        .map(|(_, time)| *time)
        .ok_or(eyre!("the default algorithm for {spec:?} did not run"))?;
    let (algo, time) = timed
        .iter()
        .copied()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap_or((default.algo, default_time));
    let (algo, time) = if time < default_time * (1.0 - MARGIN) {
        (algo, time)
    } else {
        (default.algo, default_time)
    };
    let note = format!(
        "{} {:.2} us {:.2}x",
        runtime.matmul_algo_config(algo)?,
        time,
        default_time / time
    );
    Ok((algo, note))
}
