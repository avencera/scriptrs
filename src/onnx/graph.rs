use std::collections::BTreeMap;
use std::path::Path;

use onnx_protobuf::Message;
use onnx_protobuf::{
    AttributeProto, FunctionProto, GraphProto, ModelProto, NodeProto, SparseTensorProto,
    TensorProto, ValueInfoProto, tensor_proto,
};
use protobuf::Enum;

use crate::deployment::ParakeetDeployment;
use crate::onnx::contract::{
    DeclaredDim, ElementType, GraphContract, GraphMetadata, TensorMetadata,
};
use crate::onnx::error::{GraphKind, OnnxBundleError};
use crate::onnx::manifest::BundlePath;

/// Parsed graph metadata and all external tensor-data references it owns
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GraphInspection {
    pub(crate) metadata: GraphMetadata,
    pub(crate) default_opset: u32,
    pub(crate) external_data: Vec<ExternalDataReference>,
}

/// One normalized external tensor-data reference
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExternalDataReference {
    pub(crate) path: BundlePath,
    pub(crate) offset: Option<u64>,
    pub(crate) length: Option<u64>,
}

/// Decode one ONNX graph and inspect all TensorProto owners in its model
pub(crate) fn inspect_graph(
    path: &BundlePath,
    kind: GraphKind,
    bytes: &[u8],
) -> Result<GraphInspection, OnnxBundleError> {
    let error_path = path.resolve(Path::new("."));
    let model =
        ModelProto::parse_from_bytes(bytes).map_err(|error| OnnxBundleError::GraphParse {
            graph: kind,
            path: error_path.clone(),
            message: error.to_string(),
        })?;
    let default_opset = default_opset(&model, kind, &error_path)?;
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| OnnxBundleError::GraphParse {
            graph: kind,
            path: error_path.clone(),
            message: "model has no graph".to_owned(),
        })?;
    let metadata = graph_metadata(graph, kind, &error_path)?;
    let mut external_data = Vec::new();
    collect_graph_tensors(graph, path, kind, &error_path, &mut external_data)?;
    for training in &model.training_info {
        if let Some(graph) = training.initialization.as_ref() {
            collect_graph_tensors(graph, path, kind, &error_path, &mut external_data)?;
        }
        if let Some(graph) = training.algorithm.as_ref() {
            collect_graph_tensors(graph, path, kind, &error_path, &mut external_data)?;
        }
    }
    for function in &model.functions {
        collect_function_tensors(function, path, kind, &error_path, &mut external_data)?;
    }

    Ok(GraphInspection {
        metadata,
        default_opset,
        external_data,
    })
}

/// Validate graph metadata and the default-domain opset against its contract
pub(crate) fn validate_graph_contract(
    path: &BundlePath,
    kind: GraphKind,
    inspection: &GraphInspection,
    deployment: &ParakeetDeployment,
    expected_opset: u32,
) -> Result<(), OnnxBundleError> {
    if inspection.default_opset != expected_opset {
        return Err(OnnxBundleError::GraphOpset {
            graph: kind,
            path: path.resolve(Path::new(".")),
            message: format!(
                "graph declares default-domain opset {} but manifest declares {}",
                inspection.default_opset, expected_opset
            ),
        });
    }

    let contract = GraphContract::for_kind(kind, deployment);
    contract
        .check_metadata(&inspection.metadata)
        .map_err(|message| OnnxBundleError::GraphIo {
            graph: kind,
            path: path.resolve(Path::new(".")),
            contract: contract.id,
            message,
        })
}

fn default_opset(model: &ModelProto, kind: GraphKind, path: &Path) -> Result<u32, OnnxBundleError> {
    let mut default = None;
    for import in &model.opset_import {
        if !import.domain.is_empty() && import.domain != "ai.onnx" {
            continue;
        }
        let version = u32::try_from(import.version).map_err(|_| OnnxBundleError::GraphOpset {
            graph: kind,
            path: path.to_path_buf(),
            message: format!(
                "default-domain opset {} is outside the supported range",
                import.version
            ),
        })?;
        if version == 0 {
            return Err(OnnxBundleError::GraphOpset {
                graph: kind,
                path: path.to_path_buf(),
                message: "default-domain opset must be positive".to_owned(),
            });
        }
        if default.replace(version).is_some() {
            return Err(OnnxBundleError::GraphOpset {
                graph: kind,
                path: path.to_path_buf(),
                message: "multiple default-domain opset declarations".to_owned(),
            });
        }
    }
    default.ok_or_else(|| OnnxBundleError::GraphOpset {
        graph: kind,
        path: path.to_path_buf(),
        message: "missing default-domain opset declaration".to_owned(),
    })
}

fn graph_metadata(
    graph: &GraphProto,
    kind: GraphKind,
    path: &Path,
) -> Result<GraphMetadata, OnnxBundleError> {
    Ok(GraphMetadata {
        inputs: graph
            .input
            .iter()
            .map(|value| value_metadata(value, kind, path))
            .collect::<Result<_, _>>()?,
        outputs: graph
            .output
            .iter()
            .map(|value| value_metadata(value, kind, path))
            .collect::<Result<_, _>>()?,
    })
}

fn value_metadata(
    value: &ValueInfoProto,
    kind: GraphKind,
    path: &Path,
) -> Result<TensorMetadata, OnnxBundleError> {
    let Some(type_proto) = value.type_.as_ref() else {
        return Err(OnnxBundleError::GraphIo {
            graph: kind,
            path: path.to_path_buf(),
            contract: "onnx",
            message: format!("{} has no type", value.name),
        });
    };
    if !type_proto.has_tensor_type() {
        return Err(OnnxBundleError::GraphIo {
            graph: kind,
            path: path.to_path_buf(),
            contract: "onnx",
            message: format!("{} is not a tensor", value.name),
        });
    }
    let tensor = type_proto.tensor_type();
    let dtype = match tensor_proto::DataType::from_i32(tensor.elem_type) {
        Some(tensor_proto::DataType::FLOAT) => ElementType::F32,
        Some(tensor_proto::DataType::INT32) => ElementType::I32,
        Some(tensor_proto::DataType::INT64) => ElementType::I64,
        Some(other) => {
            return Err(OnnxBundleError::GraphIo {
                graph: kind,
                path: path.to_path_buf(),
                contract: "onnx",
                message: format!("{} has unsupported element type {other:?}", value.name),
            });
        }
        None => {
            return Err(OnnxBundleError::GraphIo {
                graph: kind,
                path: path.to_path_buf(),
                contract: "onnx",
                message: format!(
                    "{} has unknown element type {}",
                    value.name, tensor.elem_type
                ),
            });
        }
    };
    let Some(shape) = tensor.shape.as_ref() else {
        return Err(OnnxBundleError::GraphIo {
            graph: kind,
            path: path.to_path_buf(),
            contract: "onnx",
            message: format!("{} has no shape", value.name),
        });
    };
    let shape = shape
        .dim
        .iter()
        .map(|dimension| {
            if dimension.has_dim_value() {
                DeclaredDim::Static(dimension.dim_value())
            } else {
                DeclaredDim::Dynamic
            }
        })
        .collect();
    Ok(TensorMetadata {
        name: value.name.clone(),
        dtype,
        shape,
    })
}

fn collect_graph_tensors(
    graph: &GraphProto,
    graph_path: &BundlePath,
    kind: GraphKind,
    error_path: &Path,
    references: &mut Vec<ExternalDataReference>,
) -> Result<(), OnnxBundleError> {
    for tensor in &graph.initializer {
        collect_tensor(tensor, graph_path, kind, error_path, references)?;
    }
    for sparse in &graph.sparse_initializer {
        collect_sparse_tensor(sparse, graph_path, kind, error_path, references)?;
    }
    for node in &graph.node {
        collect_node(node, graph_path, kind, error_path, references)?;
    }
    Ok(())
}

fn collect_function_tensors(
    function: &FunctionProto,
    graph_path: &BundlePath,
    kind: GraphKind,
    error_path: &Path,
    references: &mut Vec<ExternalDataReference>,
) -> Result<(), OnnxBundleError> {
    for attribute in &function.attribute_proto {
        collect_attribute(attribute, graph_path, kind, error_path, references)?;
    }
    for node in &function.node {
        collect_node(node, graph_path, kind, error_path, references)?;
    }
    Ok(())
}

fn collect_node(
    node: &NodeProto,
    graph_path: &BundlePath,
    kind: GraphKind,
    error_path: &Path,
    references: &mut Vec<ExternalDataReference>,
) -> Result<(), OnnxBundleError> {
    for attribute in &node.attribute {
        collect_attribute(attribute, graph_path, kind, error_path, references)?;
    }
    Ok(())
}

fn collect_attribute(
    attribute: &AttributeProto,
    graph_path: &BundlePath,
    kind: GraphKind,
    error_path: &Path,
    references: &mut Vec<ExternalDataReference>,
) -> Result<(), OnnxBundleError> {
    if let Some(tensor) = attribute.t.as_ref() {
        collect_tensor(tensor, graph_path, kind, error_path, references)?;
    }
    for tensor in &attribute.tensors {
        collect_tensor(tensor, graph_path, kind, error_path, references)?;
    }
    if let Some(sparse) = attribute.sparse_tensor.as_ref() {
        collect_sparse_tensor(sparse, graph_path, kind, error_path, references)?;
    }
    for sparse in &attribute.sparse_tensors {
        collect_sparse_tensor(sparse, graph_path, kind, error_path, references)?;
    }
    if let Some(graph) = attribute.g.as_ref() {
        collect_graph_tensors(graph, graph_path, kind, error_path, references)?;
    }
    for graph in &attribute.graphs {
        collect_graph_tensors(graph, graph_path, kind, error_path, references)?;
    }
    Ok(())
}

fn collect_sparse_tensor(
    sparse: &SparseTensorProto,
    graph_path: &BundlePath,
    kind: GraphKind,
    error_path: &Path,
    references: &mut Vec<ExternalDataReference>,
) -> Result<(), OnnxBundleError> {
    if let Some(values) = sparse.values.as_ref() {
        collect_tensor(values, graph_path, kind, error_path, references)?;
    }
    if let Some(indices) = sparse.indices.as_ref() {
        collect_tensor(indices, graph_path, kind, error_path, references)?;
    }
    Ok(())
}

fn collect_tensor(
    tensor: &TensorProto,
    graph_path: &BundlePath,
    kind: GraphKind,
    error_path: &Path,
    references: &mut Vec<ExternalDataReference>,
) -> Result<(), OnnxBundleError> {
    let mut fields = BTreeMap::new();
    for entry in &tensor.external_data {
        if fields
            .insert(entry.key.clone(), entry.value.clone())
            .is_some()
        {
            return Err(external_data_error(
                kind,
                error_path,
                format!("duplicate external-data key {}", entry.key),
            ));
        }
    }
    for key in fields.keys() {
        if !matches!(key.as_str(), "location" | "offset" | "length" | "checksum") {
            return Err(external_data_error(
                kind,
                error_path,
                format!("unknown external-data key {key}"),
            ));
        }
    }
    let location = fields.get("location");
    let is_external = tensor
        .data_location
        .enum_value()
        .map_err(|error| external_data_error(kind, error_path, error.to_string()))?
        == tensor_proto::DataLocation::EXTERNAL;
    if is_external != location.is_some() {
        return Err(external_data_error(
            kind,
            error_path,
            "external data location and data_location disagree".to_owned(),
        ));
    }
    let Some(location) = location else {
        return Ok(());
    };
    let path = resolve_external_path(graph_path, location).map_err(|message| {
        external_data_error(kind, error_path, format!("location {location}: {message}"))
    })?;
    let offset = parse_range_field(fields.get("offset"), "offset", kind, error_path)?;
    let length = parse_range_field(fields.get("length"), "length", kind, error_path)?;
    references.push(ExternalDataReference {
        path,
        offset,
        length,
    });
    Ok(())
}

fn parse_range_field(
    value: Option<&String>,
    name: &str,
    kind: GraphKind,
    error_path: &Path,
) -> Result<Option<u64>, OnnxBundleError> {
    value
        .map(|value| {
            value.parse::<u64>().map_err(|error| {
                external_data_error(
                    kind,
                    error_path,
                    format!("{name} {value} is not a non-negative decimal integer: {error}"),
                )
            })
        })
        .transpose()
}

fn resolve_external_path(
    graph_path: &BundlePath,
    location: &str,
) -> Result<BundlePath, &'static str> {
    if location.is_empty() {
        return Err("location is empty");
    }
    if location.contains('\\') || location.starts_with('/') {
        return Err("location must be a relative forward-slash path");
    }
    if location
        .split('/')
        .next()
        .is_some_and(|component| component.contains(':'))
    {
        return Err("location must not use a drive prefix");
    }
    let parent = graph_path
        .as_str()
        .rsplit_once('/')
        .map_or("", |(parent, _)| parent);
    let joined = if parent.is_empty() {
        location.to_owned()
    } else {
        format!("{parent}/{location}")
    };
    BundlePath::try_from(joined).map_err(|_| "location escapes or is not normalized")
}

fn external_data_error(kind: GraphKind, path: &Path, message: String) -> OnnxBundleError {
    OnnxBundleError::GraphExternalData {
        graph: kind,
        path: path.to_path_buf(),
        message,
    }
}
