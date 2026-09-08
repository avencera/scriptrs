use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ort::ep::{self, ExecutionProvider};
use ort::session::builder::SessionBuilder;
use ort::session::{Session, SessionInputValue, SessionOutputs};
use ort::value::{PrimitiveTensorElementType, TensorRef};

use crate::error::TranscriptionError;
use crate::onnx::contract::GraphContract;
use crate::onnx::error::{GraphKind, OnnxBundleError};

/// CUDA device ordinal that owns every session of one bundle
///
/// The default is device `0`. Sessions use the ONNX Runtime default compute
/// stream of that device; no user stream is shared with other libraries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct CudaDevice(u32);

impl CudaDevice {
    /// Select a CUDA device by ordinal
    pub const fn new(ordinal: u32) -> Self {
        Self(ordinal)
    }

    /// Device ordinal
    pub const fn ordinal(self) -> u32 {
        self.0
    }
}

/// Execution provider a bundle's sessions are registered with
///
/// Production bundles only ever use CUDA. The CPU provider exists for tests
/// that exercise tensor conversion with fixture graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SessionProvider {
    Cuda(CudaDevice),
    #[cfg(test)]
    Cpu,
}

impl SessionProvider {
    /// Refuse before loading any graph when CUDA cannot be registered
    pub(crate) fn ensure_available(self) -> Result<(), OnnxBundleError> {
        match self {
            Self::Cuda(_) => ensure_cuda_available(),
            #[cfg(test)]
            Self::Cpu => Ok(()),
        }
    }

    fn configure(self, builder: SessionBuilder) -> Result<SessionBuilder, String> {
        match self {
            Self::Cuda(device) => {
                let device_id = i32::try_from(device.ordinal()).map_err(|_| {
                    format!("CUDA device ordinal {} is out of range", device.ordinal())
                })?;
                let cuda = ep::CUDA::default().with_device_id(device_id);
                builder
                    .with_execution_providers([cuda.build().error_on_failure()])
                    .map_err(|error| error.to_string())
            }
            #[cfg(test)]
            Self::Cpu => builder
                .with_execution_providers([ep::CPU::default().build().error_on_failure()])
                .map_err(|error| error.to_string()),
        }
    }
}

fn ensure_cuda_available() -> Result<(), OnnxBundleError> {
    let cuda = ep::CUDA::default();
    if !cuda.supported_by_platform() {
        return Err(OnnxBundleError::CudaUnavailable {
            reason: "the CUDA execution provider is not supported on this platform".to_owned(),
        });
    }

    match cuda.is_available() {
        Ok(true) => Ok(()),
        Ok(false) => Err(OnnxBundleError::CudaUnavailable {
            reason: "this ONNX Runtime build does not include the CUDA execution provider"
                .to_owned(),
        }),
        Err(error) => Err(OnnxBundleError::CudaUnavailable {
            reason: format!("provider discovery failed: {error}"),
        }),
    }
}

/// One loaded graph with its validated tensor contract
///
/// Runs are serialized through a mutex because ONNX Runtime sessions are
/// mutated by `run`. Clones of a bundle share the same session.
pub(crate) struct GraphSession {
    kind: GraphKind,
    path: PathBuf,
    contract: GraphContract,
    session: Mutex<Session>,
}

impl fmt::Debug for GraphSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GraphSession")
            .field("kind", &self.kind)
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl GraphSession {
    /// Build a session for `path` and check its declared tensors against `contract`
    pub(crate) fn load(
        kind: GraphKind,
        path: &Path,
        provider: SessionProvider,
        contract: GraphContract,
    ) -> Result<Self, OnnxBundleError> {
        let session_load = |message: String| OnnxBundleError::SessionLoad {
            graph: kind,
            path: path.to_path_buf(),
            message,
        };
        let builder = Session::builder().map_err(|error| session_load(error.to_string()))?;
        let mut builder = provider.configure(builder).map_err(session_load)?;
        let session = builder
            .commit_from_file(path)
            .map_err(|error| session_load(error.to_string()))?;

        contract
            .check_runtime(session.inputs(), session.outputs())
            .map_err(|message| OnnxBundleError::GraphIo {
                graph: kind,
                path: path.to_path_buf(),
                contract: contract.id,
                message,
            })?;

        Ok(Self {
            kind,
            path: path.to_path_buf(),
            contract,
            session: Mutex::new(session),
        })
    }

    /// Run the graph and read its outputs while the session lock is held
    pub(crate) fn run<T>(
        &self,
        inputs: Vec<(&'static str, SessionInputValue<'_>)>,
        read: impl FnOnce(&OutputReader<'_, '_>) -> Result<T, TranscriptionError>,
    ) -> Result<T, TranscriptionError> {
        let mut session = self.session.lock().map_err(|_| {
            TranscriptionError::Onnx(format!(
                "{} graph session lock was poisoned by an earlier panic",
                self.kind
            ))
        })?;
        let outputs = session.run(inputs).map_err(|error| {
            TranscriptionError::Onnx(format!("{} graph run failed: {error}", self.kind))
        })?;
        let reader = OutputReader {
            graph: self,
            outputs: &outputs,
        };

        read(&reader)
    }
}

/// Typed access to the outputs of one run
pub(crate) struct OutputReader<'a, 'r> {
    graph: &'a GraphSession,
    outputs: &'a SessionOutputs<'r>,
}

/// Owned copy of one output tensor with its resolved shape
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct OutputTensor<T> {
    pub(crate) shape: Vec<usize>,
    pub(crate) data: Vec<T>,
}

impl OutputReader<'_, '_> {
    /// Copy a float32 output after checking it against the contract
    pub(crate) fn f32(&self, name: &str) -> Result<OutputTensor<f32>, TranscriptionError> {
        self.extract::<f32>(name)
    }

    /// Copy an int32 output after checking it against the contract
    pub(crate) fn i32(&self, name: &str) -> Result<OutputTensor<i32>, TranscriptionError> {
        self.extract::<i32>(name)
    }

    fn extract<T: PrimitiveTensorElementType + Clone>(
        &self,
        name: &str,
    ) -> Result<OutputTensor<T>, TranscriptionError> {
        let kind = self.graph.kind;
        let invalid = |message: String| {
            TranscriptionError::InvalidModelOutput(format!("{kind} graph output {message}"))
        };
        let spec = self
            .graph
            .contract
            .output(name)
            .ok_or_else(|| invalid(format!("`{name}` is not part of the contract")))?;
        let value = self
            .outputs
            .get(name)
            .ok_or_else(|| invalid(format!("`{name}` was not produced")))?;
        let (shape, data) = value
            .try_extract_tensor::<T>()
            .map_err(|error| invalid(format!("`{name}` could not be read: {error}")))?;
        let shape = spec.resolve_runtime_shape(shape).map_err(invalid)?;

        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(invalid(format!(
                "`{name}` holds {} values for shape {shape:?}",
                data.len()
            )));
        }

        Ok(OutputTensor {
            shape,
            data: data.to_vec(),
        })
    }
}

/// Borrow a float32 buffer as a named graph input
pub(crate) fn f32_input<'a>(
    name: &'static str,
    shape: &[usize],
    data: &'a [f32],
) -> Result<(&'static str, SessionInputValue<'a>), TranscriptionError> {
    let tensor = TensorRef::from_array_view((shape.to_vec(), data))
        .map_err(|error| input_error(name, error))?;
    Ok((name, SessionInputValue::from(tensor)))
}

/// Borrow an int32 buffer as a named graph input
pub(crate) fn i32_input<'a>(
    name: &'static str,
    shape: &[usize],
    data: &'a [i32],
) -> Result<(&'static str, SessionInputValue<'a>), TranscriptionError> {
    let tensor = TensorRef::from_array_view((shape.to_vec(), data))
        .map_err(|error| input_error(name, error))?;
    Ok((name, SessionInputValue::from(tensor)))
}

/// Borrow a single int64 value as a rank-0 graph input
#[cfg(feature = "vad")]
pub(crate) fn i64_scalar_input<'a>(
    name: &'static str,
    data: &'a [i64; 1],
) -> Result<(&'static str, SessionInputValue<'a>), TranscriptionError> {
    let tensor =
        TensorRef::from_array_view(((), &data[..])).map_err(|error| input_error(name, error))?;
    Ok((name, SessionInputValue::from(tensor)))
}

fn input_error(name: &str, error: ort::Error) -> TranscriptionError {
    TranscriptionError::Onnx(format!("failed to build graph input `{name}`: {error}"))
}
