use crate::deployment::{DecoderDimensions, ModelIo, ParakeetDeployment};
use crate::onnx::error::GraphKind;

/// Tensor element types used by the split graph contracts
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ElementType {
    F32,
    I32,
    I64,
}

impl ElementType {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::F32 => "float32",
            Self::I32 => "int32",
            Self::I64 => "int64",
        }
    }
}

/// A declared graph dimension from ONNX or ONNX Runtime
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeclaredDim {
    Static(i64),
    Dynamic,
}

/// Name, element type, and shape metadata for one graph tensor
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TensorMetadata {
    pub(crate) name: String,
    pub(crate) dtype: ElementType,
    pub(crate) shape: Vec<DeclaredDim>,
}

/// Metadata extracted from one serialized graph
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GraphMetadata {
    pub(crate) inputs: Vec<TensorMetadata>,
    pub(crate) outputs: Vec<TensorMetadata>,
}

/// One axis of a contract shape
///
/// `Exact` axes may be declared dynamic by the graph but must resolve to the
/// exact size at run time. `Any` axes are data dependent, such as encoder time
/// or the joint top-K width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DimSpec {
    Exact(usize),
    Any,
}

/// Name, element type, and shape contract for one graph tensor
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TensorSpec {
    pub(crate) name: String,
    pub(crate) dtype: ElementType,
    pub(crate) shape: Vec<DimSpec>,
}

impl TensorSpec {
    fn new(name: &str, dtype: ElementType, shape: &[DimSpec]) -> Self {
        Self {
            name: name.to_owned(),
            dtype,
            shape: shape.to_vec(),
        }
    }

    /// Resolve a concrete run-time shape against this contract
    #[cfg(any(feature = "onnx-cuda", test))]
    pub(crate) fn resolve_runtime_shape(&self, actual: &[i64]) -> Result<Vec<usize>, String> {
        if actual.len() != self.shape.len() {
            return Err(format!(
                "`{}` has rank {} but the contract requires rank {}",
                self.name,
                actual.len(),
                self.shape.len()
            ));
        }

        let mut resolved = Vec::with_capacity(actual.len());
        for (axis, (spec, dim)) in self.shape.iter().zip(actual).enumerate() {
            let dim = usize::try_from(*dim).map_err(|_| {
                format!(
                    "`{}` axis {axis} has negative size {dim} at run time",
                    self.name
                )
            })?;
            if let DimSpec::Exact(expected) = spec
                && dim != *expected
            {
                return Err(format!(
                    "`{}` axis {axis} has size {dim} but the contract requires {expected}",
                    self.name
                ));
            }
            if matches!(spec, DimSpec::Any) && dim == 0 {
                return Err(format!("`{}` axis {axis} has size 0", self.name));
            }
            resolved.push(dim);
        }

        Ok(resolved)
    }

    fn check_declared(&self, tensor: &TensorMetadata) -> Result<(), String> {
        if self.dtype != tensor.dtype {
            return Err(format!(
                "`{}` has element type {} but the contract requires {}",
                self.name,
                tensor.dtype.name(),
                self.dtype.name()
            ));
        }
        if tensor.shape.len() != self.shape.len() {
            return Err(format!(
                "`{}` declares rank {} but the contract requires rank {}",
                self.name,
                tensor.shape.len(),
                self.shape.len()
            ));
        }

        for (axis, (spec, declared)) in self.shape.iter().zip(&tensor.shape).enumerate() {
            let DeclaredDim::Static(declared) = declared else {
                continue;
            };
            if *declared < 0 {
                return Err(format!(
                    "`{}` axis {axis} has negative size {declared}",
                    self.name
                ));
            }
            match spec {
                DimSpec::Exact(expected) if *declared as usize != *expected => {
                    return Err(format!(
                        "`{}` axis {axis} declares size {declared} but the contract requires {expected}",
                        self.name
                    ));
                }
                DimSpec::Any if *declared == 0 => {
                    return Err(format!("`{}` axis {axis} declares size 0", self.name));
                }
                _ => {}
            }
        }

        Ok(())
    }
}

/// Declared tensor contract for one graph
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GraphContract {
    pub(crate) id: &'static str,
    pub(crate) inputs: Vec<TensorSpec>,
    pub(crate) outputs: Vec<TensorSpec>,
}

impl GraphContract {
    /// Build the contract for one graph slot
    pub(crate) fn for_kind(kind: GraphKind, deployment: &ParakeetDeployment) -> Self {
        let io = deployment.model_io();
        let dims = deployment.decoder_dimensions();
        let config = crate::config::TranscriptionConfig::from_deployment(deployment);
        match kind {
            GraphKind::Encoder => {
                Self::encoder(io, dims, config.feature_size, config.max_feature_frames())
            }
            GraphKind::Decoder => Self::decoder(io, dims),
            GraphKind::Joint => Self::joint(io, dims),
            GraphKind::SileroVad => Self::silero_vad(),
        }
    }

    /// Encoder: `mel` and `mel_length` to `encoder` and `encoder_length`
    pub(crate) fn encoder(
        io: ModelIo,
        dims: DecoderDimensions,
        feature_size: usize,
        max_feature_frames: usize,
    ) -> Self {
        let encoder = io.encoder();
        Self {
            id: crate::onnx::manifest::ParakeetGraphContract::SplitV1.id(),
            inputs: vec![
                TensorSpec::new(
                    encoder.input_mel,
                    ElementType::F32,
                    &[
                        DimSpec::Exact(1),
                        DimSpec::Exact(feature_size),
                        DimSpec::Exact(max_feature_frames),
                    ],
                ),
                TensorSpec::new(encoder.input_length, ElementType::I32, &[DimSpec::Exact(1)]),
            ],
            outputs: vec![
                TensorSpec::new(
                    encoder.output_encoder,
                    ElementType::F32,
                    &[
                        DimSpec::Exact(1),
                        DimSpec::Exact(dims.encoder_hidden()),
                        DimSpec::Any,
                    ],
                ),
                TensorSpec::new(
                    encoder.output_length,
                    ElementType::I32,
                    &[DimSpec::Exact(1)],
                ),
            ],
        }
    }

    /// Decoder: token, length, and LSTM state to projected output and next state
    pub(crate) fn decoder(io: ModelIo, dims: DecoderDimensions) -> Self {
        let decoder = io.decoder();
        let state = [
            DimSpec::Exact(dims.decoder_layers()),
            DimSpec::Exact(1),
            DimSpec::Exact(dims.decoder_hidden()),
        ];
        Self {
            id: crate::onnx::manifest::ParakeetGraphContract::SplitV1.id(),
            inputs: vec![
                TensorSpec::new(
                    decoder.targets,
                    ElementType::I32,
                    &[DimSpec::Exact(1), DimSpec::Exact(1)],
                ),
                TensorSpec::new(
                    decoder.target_length,
                    ElementType::I32,
                    &[DimSpec::Exact(1)],
                ),
                TensorSpec::new(decoder.hidden_in, ElementType::F32, &state),
                TensorSpec::new(decoder.cell_in, ElementType::F32, &state),
            ],
            outputs: vec![
                TensorSpec::new(
                    decoder.decoder,
                    ElementType::F32,
                    &[
                        DimSpec::Exact(1),
                        DimSpec::Exact(dims.decoder_hidden()),
                        DimSpec::Exact(1),
                    ],
                ),
                TensorSpec::new(decoder.hidden_out, ElementType::F32, &state),
                TensorSpec::new(decoder.cell_out, ElementType::F32, &state),
            ],
        }
    }

    /// Joint: encoder frame and decoder output to token, probability, duration,
    /// and optional top-K candidates
    pub(crate) fn joint(io: ModelIo, dims: DecoderDimensions) -> Self {
        let joint = io.joint();
        let scalar = [DimSpec::Exact(1), DimSpec::Exact(1), DimSpec::Exact(1)];
        let top_k = [DimSpec::Exact(1), DimSpec::Any];
        let mut outputs = vec![
            TensorSpec::new(joint.token_id, ElementType::I32, &scalar),
            TensorSpec::new(joint.token_prob, ElementType::F32, &scalar),
            TensorSpec::new(joint.duration, ElementType::I32, &scalar),
        ];
        if let Some(name) = joint.top_k_ids {
            outputs.push(TensorSpec::new(name, ElementType::I32, &top_k));
        }
        if let Some(name) = joint.top_k_logits {
            outputs.push(TensorSpec::new(name, ElementType::F32, &top_k));
        }

        Self {
            id: crate::onnx::manifest::ParakeetGraphContract::SplitV1.id(),
            inputs: vec![
                TensorSpec::new(
                    joint.encoder_step,
                    ElementType::F32,
                    &[
                        DimSpec::Exact(1),
                        DimSpec::Exact(dims.encoder_hidden()),
                        DimSpec::Exact(1),
                    ],
                ),
                TensorSpec::new(
                    joint.decoder_step,
                    ElementType::F32,
                    &[
                        DimSpec::Exact(1),
                        DimSpec::Exact(dims.decoder_hidden()),
                        DimSpec::Exact(1),
                    ],
                ),
            ],
            outputs,
        }
    }

    /// Silero VAD v5 16 kHz window classifier
    pub(crate) fn silero_vad() -> Self {
        let state = [DimSpec::Exact(2), DimSpec::Exact(1), DimSpec::Exact(128)];
        Self {
            id: crate::onnx::manifest::SileroGraphContract::V1.id(),
            inputs: vec![
                TensorSpec::new(
                    SILERO_INPUT,
                    ElementType::F32,
                    &[DimSpec::Exact(1), DimSpec::Exact(512 + 64)],
                ),
                TensorSpec::new(SILERO_STATE_IN, ElementType::F32, &state),
                TensorSpec::new(SILERO_SAMPLE_RATE, ElementType::I64, &[]),
            ],
            outputs: vec![
                TensorSpec::new(
                    SILERO_OUTPUT,
                    ElementType::F32,
                    &[DimSpec::Exact(1), DimSpec::Exact(1)],
                ),
                TensorSpec::new(SILERO_STATE_OUT, ElementType::F32, &state),
            ],
        }
    }

    /// Compare serialized graph metadata with this contract
    pub(crate) fn check_metadata(&self, metadata: &GraphMetadata) -> Result<(), String> {
        check_tensors("input", &self.inputs, &metadata.inputs)?;
        check_tensors("output", &self.outputs, &metadata.outputs)
    }

    /// Compare an ONNX Runtime session's metadata with this contract
    #[cfg(feature = "onnx-cuda")]
    pub(crate) fn check_runtime(
        &self,
        inputs: &[ort::value::Outlet],
        outputs: &[ort::value::Outlet],
    ) -> Result<(), String> {
        let inputs = inputs
            .iter()
            .map(runtime_tensor_metadata)
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = outputs
            .iter()
            .map(runtime_tensor_metadata)
            .collect::<Result<Vec<_>, _>>()?;
        self.check_metadata(&GraphMetadata { inputs, outputs })
    }

    /// Contract entry for one output tensor
    #[cfg(feature = "onnx-cuda")]
    pub(crate) fn output(&self, name: &str) -> Option<&TensorSpec> {
        self.outputs.iter().find(|spec| spec.name == name)
    }
}

pub(crate) const SILERO_INPUT: &str = "input";
pub(crate) const SILERO_STATE_IN: &str = "state";
pub(crate) const SILERO_SAMPLE_RATE: &str = "sr";
pub(crate) const SILERO_OUTPUT: &str = "output";
pub(crate) const SILERO_STATE_OUT: &str = "stateN";

fn check_tensors(
    kind: &str,
    specs: &[TensorSpec],
    tensors: &[TensorMetadata],
) -> Result<(), String> {
    let mut names = std::collections::BTreeSet::new();
    for tensor in tensors {
        if !names.insert(&tensor.name) {
            return Err(format!(
                "{kind} `{}` is declared more than once",
                tensor.name
            ));
        }
    }
    for spec in specs {
        let tensor = tensors
            .iter()
            .find(|tensor| tensor.name == spec.name)
            .ok_or_else(|| format!("{kind} `{}` is missing", spec.name))?;
        spec.check_declared(tensor)?;
    }

    for tensor in tensors {
        if !specs.iter().any(|spec| spec.name == tensor.name) {
            return Err(format!(
                "{kind} `{}` is not part of the contract",
                tensor.name
            ));
        }
    }

    Ok(())
}

#[cfg(feature = "onnx-cuda")]
fn runtime_tensor_metadata(outlet: &ort::value::Outlet) -> Result<TensorMetadata, String> {
    use ort::value::{TensorElementType, ValueType};

    let ValueType::Tensor { ty, shape, .. } = outlet.dtype() else {
        return Err(format!("`{}` is not a tensor", outlet.name()));
    };
    let dtype = match ty {
        TensorElementType::Float32 => ElementType::F32,
        TensorElementType::Int32 => ElementType::I32,
        TensorElementType::Int64 => ElementType::I64,
        other => {
            return Err(format!(
                "`{}` has unsupported element type {other:?}",
                outlet.name()
            ));
        }
    };
    let shape = shape
        .iter()
        .map(|dimension| {
            if *dimension < 0 {
                DeclaredDim::Dynamic
            } else {
                DeclaredDim::Static(*dimension)
            }
        })
        .collect();
    Ok(TensorMetadata {
        name: outlet.name().to_owned(),
        dtype,
        shape,
    })
}

#[cfg(test)]
mod tests {
    use super::{DeclaredDim, DimSpec, ElementType, TensorMetadata, TensorSpec};

    #[test]
    fn runtime_shape_resolution_rejects_zero_any_axes() {
        let spec = TensorSpec::new(
            "top_k",
            ElementType::I32,
            &[DimSpec::Exact(1), DimSpec::Any],
        );
        assert!(spec.resolve_runtime_shape(&[1, 0]).is_err());
        assert_eq!(spec.resolve_runtime_shape(&[1, 2]).unwrap(), vec![1, 2]);
    }

    #[test]
    fn declared_metadata_rejects_duplicate_and_unknown_tensors() {
        let spec = TensorSpec::new("input", ElementType::F32, &[DimSpec::Exact(1)]);
        let duplicate = TensorMetadata {
            name: "input".to_owned(),
            dtype: ElementType::F32,
            shape: vec![DeclaredDim::Static(1)],
        };
        let mut tensors = vec![duplicate.clone(), duplicate];
        assert!(super::check_tensors("input", &[spec], &tensors).is_err());

        tensors.truncate(1);
        tensors.push(TensorMetadata {
            name: "extra".to_owned(),
            dtype: ElementType::F32,
            shape: vec![DeclaredDim::Static(1)],
        });
        let spec = TensorSpec::new("input", ElementType::F32, &[DimSpec::Exact(1)]);
        assert!(super::check_tensors("input", &[spec], &tensors).is_err());
    }
}
