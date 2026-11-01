//! Linear layers on cuBLASLt, with the algorithm a checked-in table names per GPU
//!
//! cuBLAS runs cuBLASLt's first heuristic, which at batch-1 row counts often leaves
//! most of a large GPU idle: on an RTX 5090, a split-K algorithm further down the list
//! runs many TF32 encoder products 1.5-2.3x faster. `examples/cuda_tune.rs` times
//! every candidate once per shape and row range and writes the winners to
//! `src/cuda/tune/`. The table is fixed per GPU model and cuBLASLt version, so the
//! summation order, and with it every transcript, stays the same from run to run,
//! which choosing by timing at load would not give. Without a matching table, a
//! product runs the heuristic's first algorithm, the one cuBLAS itself would pick

use std::collections::HashMap;
use std::ops::RangeInclusive;
use std::sync::{Arc, Mutex, PoisonError};

use cudakit::cudarc::driver::{DevicePtr, DevicePtrMut};
use cudakit::{
    Error, Library, Math, Matmul, MatmulAlgo, MatmulElement, MatmulPlan, Operands, Sgemm,
};
use tracing::{debug, info};

use crate::cuda::kernels::CudaRuntime;

/// Tables written by `examples/cuda_tune.rs`, one per GPU model
const TABLES: &[&str] = &[include_str!("tune/nvidia-geforce-rtx-5090.tsv")];

/// The GPU and library a table was measured on; a table applies only to an exact match
#[derive(Debug, Clone, PartialEq, Eq)]
struct Identity {
    device: String,
    multiprocessors: usize,
    cublaslt: usize,
}

/// One row range of one product shape and the algorithm it runs
#[derive(Debug, Clone, PartialEq)]
struct Entry {
    operands: Operands,
    n: usize,
    k: usize,
    rows: RangeInclusive<usize>,
    algo: MatmulAlgo,
}

/// A parsed tune table
///
/// The format is tab-separated text, `#` starting a comment:
///
/// ```text
/// device           <name as the driver reports it>
/// multiprocessors  <SM count>
/// cublaslt         <cublasLtGetVersion>
/// <tf32|fp16|fp32> <n> <k> <first row> <last row> <algo>
/// ```
///
/// `n` and `k` are a linear layer's outputs and inputs, the rows bound the product's
/// `m`, and `algo` is the eight 64-bit words of a `cublasLtMatmulAlgo_t` in hex,
/// joined by `.`
#[derive(Debug, Clone, PartialEq)]
struct Table {
    identity: Identity,
    entries: Vec<Entry>,
}

impl Table {
    fn parse(text: &str) -> Result<Self, String> {
        let (mut device, mut multiprocessors, mut cublaslt) = (None, None, None);
        let mut entries = Vec::new();
        for (index, line) in text.lines().enumerate() {
            let line = line.split('#').next().unwrap_or_default().trim();
            if line.is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split('\t').map(str::trim).collect();
            let error = |reason: &str| format!("line {}: {reason}: {line}", index + 1);
            let number = |field: &str| field.parse::<usize>().map_err(|_| error("bad number"));
            match fields.as_slice() {
                ["device", name] => device = Some((*name).to_owned()),
                ["multiprocessors", count] => multiprocessors = Some(number(count)?),
                ["cublaslt", version] => cublaslt = Some(number(version)?),
                [operands, n, k, first, last, algo] => {
                    let operands = match *operands {
                        "tf32" => Operands::F32(Math::Tf32),
                        "fp32" => Operands::F32(Math::Fp32),
                        "fp16" => Operands::F16,
                        _ => return Err(error("unknown operands")),
                    };
                    let words: Vec<u64> = algo
                        .split('.')
                        .map(|word| u64::from_str_radix(word, 16))
                        .collect::<Result<_, _>>()
                        .map_err(|_| error("bad algorithm word"))?;
                    let algo = MatmulAlgo(
                        words
                            .try_into()
                            .map_err(|_| error("an algorithm is eight words"))?,
                    );
                    let rows = number(first)?..=number(last)?;
                    if rows.is_empty() {
                        return Err(error("empty row range"));
                    }
                    entries.push(Entry {
                        operands,
                        n: number(n)?,
                        k: number(k)?,
                        rows,
                        algo,
                    });
                }
                _ => return Err(error("unrecognized line")),
            }
        }

        let identity = Identity {
            device: device.ok_or("no device line")?,
            multiprocessors: multiprocessors.ok_or("no multiprocessors line")?,
            cublaslt: cublaslt.ok_or("no cublaslt line")?,
        };
        Ok(Self { identity, entries })
    }

    /// The algorithm for a product of `rows` rows, if the table covers it
    fn lookup(&self, operands: Operands, n: usize, k: usize, rows: usize) -> Option<MatmulAlgo> {
        self.entries
            .iter()
            .find(|entry| {
                entry.operands == operands
                    && entry.n == n
                    && entry.k == k
                    && entry.rows.contains(&rows)
            })
            .map(|entry| entry.algo)
    }
}

/// cuBLASLt plans for linear products, one per shape, cached as they are first used
pub(crate) struct Linears {
    table: Option<Table>,
    plans: Mutex<HashMap<Matmul, Arc<MatmulPlan>>>,
}

impl Linears {
    /// Prepares cuBLASLt and picks the table measured on this GPU and library version
    pub(crate) fn load(runtime: &CudaRuntime) -> Result<Self, Error> {
        runtime.prepare(Library::CublasLt)?;
        let identity = Identity {
            device: runtime.device().name().to_owned(),
            multiprocessors: runtime.multiprocessor_count(),
            cublaslt: runtime.cublaslt_version()?,
        };
        let tables = TABLES.iter().map(|text| {
            Table::parse(text).map_err(|reason| Error::Unsupported {
                context: "cuBLASLt tune table",
                reason,
            })
        });
        let mut table = None;
        for parsed in tables {
            let parsed = parsed?;
            if parsed.identity == identity {
                table = Some(parsed);
                break;
            }
        }
        match &table {
            Some(table) => debug!(entries = table.entries.len(), "cuBLASLt tune table"),
            None => info!(
                ?identity,
                "no cuBLASLt tune table for this GPU and library; linear layers run \
                 cuBLAS's default algorithms"
            ),
        }
        Ok(Self {
            table,
            plans: Mutex::new(HashMap::new()),
        })
    }

    /// Runs `spec`, a [`linear`](crate::cuda::linear) product `c = alpha * a · wᵀ +
    /// beta * c`, with `a` and `w` in `operands` precision, on the runtime's stream
    ///
    /// `spec.math` does not apply: `operands` carries the precision
    pub(crate) fn run<T, A, W, C>(
        &self,
        runtime: &CudaRuntime,
        spec: Sgemm,
        operands: Operands,
        a: &A,
        w: &W,
        c: &mut C,
    ) -> Result<(), Error>
    where
        T: MatmulElement,
        A: DevicePtr<T>,
        W: DevicePtr<T>,
        C: DevicePtrMut<f32>,
    {
        if spec.a_transposed || !spec.b_transposed {
            return Err(Error::Unsupported {
                context: "linear",
                reason: "only `a · wᵀ` products run as linear layers".to_owned(),
            });
        }
        let plan = self.plan(runtime, Matmul::linear(spec.m, spec.n, spec.k, operands))?;
        runtime.matmul(&plan, spec.alpha, spec.beta, a, w, c)
    }

    fn plan(&self, runtime: &CudaRuntime, spec: Matmul) -> Result<Arc<MatmulPlan>, Error> {
        let mut plans = self.plans.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(plan) = plans.get(&spec) {
            return Ok(plan.clone());
        }

        let stored = self
            .table
            .as_ref()
            .and_then(|table| table.lookup(spec.operands, spec.n, spec.k, spec.m));
        let plan = match stored {
            Some(algo) => runtime.matmul_plan_with(spec, algo).or_else(|error| {
                // a table entry cuBLASLt refuses would be a tuner bug; run the default
                debug!(?spec, %error, "tuned cuBLASLt algorithm refused");
                runtime.matmul_plan(spec)
            })?,
            None => runtime.matmul_plan(spec)?,
        };
        let plan = Arc::new(plan);
        plans.insert(spec, plan.clone());
        Ok(plan)
    }
}

#[cfg(test)]
mod tests {
    use cudakit::{Math, MatmulAlgo, Operands};

    use super::{TABLES, Table};

    #[test]
    fn checked_in_tables_parse_and_cover_each_shape_without_overlap() {
        for text in TABLES {
            let table = Table::parse(text).expect("table parses");
            assert!(!table.entries.is_empty());
            for (index, entry) in table.entries.iter().enumerate() {
                let overlapping = table.entries[index + 1..].iter().find(|other| {
                    other.operands == entry.operands
                        && (other.n, other.k) == (entry.n, entry.k)
                        && other.rows.start() <= entry.rows.end()
                        && entry.rows.start() <= other.rows.end()
                });
                assert!(overlapping.is_none(), "{entry:?} overlaps {overlapping:?}");
            }
        }
    }

    #[test]
    fn lookup_finds_the_range_holding_the_rows() {
        let text = "device\tGPU\nmultiprocessors\t2\ncublaslt\t7\n\
                    tf32\t8\t4\t1\t8\t1.2.3.4.5.6.7.8\t# small\n\
                    tf32\t8\t4\t9\t64\ta.b.c.d.e.f.0.1\n";
        let table = Table::parse(text).expect("table parses");
        let tf32 = Operands::F32(Math::Tf32);
        assert_eq!(
            table.lookup(tf32, 8, 4, 8),
            Some(MatmulAlgo([1, 2, 3, 4, 5, 6, 7, 8]))
        );
        assert_eq!(
            table.lookup(tf32, 8, 4, 33),
            Some(MatmulAlgo([10, 11, 12, 13, 14, 15, 0, 1]))
        );
        assert_eq!(table.lookup(tf32, 8, 4, 65), None);
        assert_eq!(table.lookup(Operands::F16, 8, 4, 8), None);
        assert!(Table::parse("tf32\t8\t4\t1\t8\t1.2").is_err());
    }
}
