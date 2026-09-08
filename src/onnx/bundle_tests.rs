use std::fs;
use std::path::Path;

use onnx_protobuf::{
    AttributeProto, GraphProto, Message, ModelProto, NodeProto, OperatorSetIdProto,
    SparseTensorProto, StringStringEntryProto, TensorProto, tensor_proto,
};
use protobuf::{EnumOrUnknown, MessageField};
use tempfile::{TempDir, tempdir};

use crate::deployment::{ParakeetDeployment, ParakeetVersion};
use crate::onnx::test_support::{FixtureBundle, V2_REVISION, digest, write_manifest};
use crate::onnx::{
    FileAsset, GraphKind, OnnxBundleError, OnnxBundleManifest, validate_onnx_bundle,
};

fn write_fixture(version: ParakeetVersion) -> (TempDir, OnnxBundleManifest) {
    let dir = tempdir().expect("fixture directory should exist");
    let manifest = FixtureBundle::blank(version).write(dir.path());
    (dir, manifest)
}

fn update_encoder(root: &Path, manifest: &mut OnnxBundleManifest, bytes: Vec<u8>) {
    fs::write(root.join("graphs/encoder.onnx"), bytes).expect("encoder should be written");
    manifest.graphs.encoder.sha256 = digest(root, "graphs/encoder.onnx");
}

fn external_tensor(location: &str) -> TensorProto {
    let mut tensor = TensorProto::new();
    tensor.name = "external".to_owned();
    tensor.dims = vec![1];
    tensor.data_type = tensor_proto::DataType::FLOAT as i32;
    tensor.data_location = EnumOrUnknown::new(tensor_proto::DataLocation::EXTERNAL);
    tensor.external_data.push(StringStringEntryProto {
        key: "location".to_owned(),
        value: location.to_owned(),
        ..Default::default()
    });
    tensor
}

fn add_initializer(mut bytes: Vec<u8>, tensor: TensorProto) -> Vec<u8> {
    let mut model = ModelProto::parse_from_bytes(&bytes).expect("fixture graph should parse");
    model
        .graph
        .as_mut()
        .expect("fixture graph should have a graph")
        .initializer
        .push(tensor);
    bytes = model.write_to_bytes().expect("fixture graph should encode");
    bytes
}

#[test]
fn canonical_validation_runs_without_ort() {
    let (dir, _) = write_fixture(ParakeetVersion::V2);
    let evidence = validate_onnx_bundle(dir.path(), ParakeetDeployment::v2())
        .expect("fixture should pass ORT-free validation");
    assert_eq!(
        evidence.manifest().export.opset,
        17,
        "the validator should retain the typed manifest evidence"
    );
    assert_eq!(evidence.vocabulary().blank_id(), 1_024);
}

#[test]
fn malformed_tokenizer_is_rejected_during_canonical_validation() {
    let (dir, mut manifest) = write_fixture(ParakeetVersion::V2);
    fs::write(dir.path().join("vocab.txt"), "not a tokenizer")
        .expect("tokenizer should be written");
    manifest.tokenizer.sha256 = digest(dir.path(), "vocab.txt");
    write_manifest(dir.path(), &manifest);

    let error = validate_onnx_bundle(dir.path(), ParakeetDeployment::v2())
        .expect_err("malformed tokenizer must fail before runtime loading");
    assert!(matches!(error, OnnxBundleError::InvalidTokenizer { .. }));
}

#[test]
fn retained_vocabulary_is_immune_to_post_validation_replacement() {
    let (dir, _) = write_fixture(ParakeetVersion::V2);
    let evidence = validate_onnx_bundle(dir.path(), ParakeetDeployment::v2()).unwrap();
    let original = evidence.vocabulary().token(1).unwrap().to_owned();
    fs::write(dir.path().join("vocab.txt"), "<unk> 0\nchanged 1\n").unwrap();
    assert_eq!(evidence.vocabulary().token(1), Some(original.as_str()));
}

#[test]
fn graph_external_data_must_match_the_graph_manifest_entry() {
    let (dir, mut manifest) = write_fixture(ParakeetVersion::V2);
    let bytes = add_initializer(
        fs::read(dir.path().join("graphs/encoder.onnx")).unwrap(),
        external_tensor("encoder.onnx.data"),
    );
    update_encoder(dir.path(), &mut manifest, bytes);
    write_manifest(dir.path(), &manifest);

    let error = validate_onnx_bundle(dir.path(), ParakeetDeployment::v2())
        .expect_err("an unlisted graph reference must fail");
    assert!(matches!(
        error,
        OnnxBundleError::GraphExternalDataMismatch {
            graph: GraphKind::Encoder,
            ..
        }
    ));
}

#[test]
fn nested_and_sparse_tensor_owners_are_closed_over() {
    let (dir, mut manifest) = write_fixture(ParakeetVersion::V2);
    let mut model =
        ModelProto::parse_from_bytes(&fs::read(dir.path().join("graphs/encoder.onnx")).unwrap())
            .unwrap();
    let graph = model.graph.as_mut().unwrap();
    let nested_graph = GraphProto {
        initializer: vec![external_tensor("encoder.onnx.data")],
        ..Default::default()
    };
    let mut attribute = AttributeProto::new();
    attribute.name = "body".to_owned();
    attribute.g = MessageField::some(nested_graph);
    graph.node.push(NodeProto {
        op_type: "If".to_owned(),
        attribute: vec![attribute],
        ..Default::default()
    });
    graph.sparse_initializer.push(SparseTensorProto {
        values: MessageField::some(external_tensor("encoder.onnx.data")),
        indices: MessageField::some(external_tensor("encoder.onnx.data")),
        dims: vec![1],
        ..Default::default()
    });
    let bytes = model.write_to_bytes().unwrap();
    update_encoder(dir.path(), &mut manifest, bytes);
    fs::write(dir.path().join("graphs/encoder.onnx.data"), [0_u8; 64]).unwrap();
    manifest.graphs.encoder.external_data.push(FileAsset {
        path: crate::onnx::BundlePath::try_from("graphs/encoder.onnx.data".to_owned()).unwrap(),
        sha256: digest(dir.path(), "graphs/encoder.onnx.data"),
    });
    write_manifest(dir.path(), &manifest);

    validate_onnx_bundle(dir.path(), ParakeetDeployment::v2())
        .expect("nested graph and sparse tensor references must be discovered");
}

#[test]
fn default_domain_opset_must_match_manifest_and_be_unique() {
    let (dir, mut manifest) = write_fixture(ParakeetVersion::V2);
    manifest.export.opset = 16;
    write_manifest(dir.path(), &manifest);
    let error = validate_onnx_bundle(dir.path(), ParakeetDeployment::v2()).unwrap_err();
    assert!(matches!(error, OnnxBundleError::GraphOpset { .. }));

    let (dir, mut manifest) = write_fixture(ParakeetVersion::V2);
    let mut model =
        ModelProto::parse_from_bytes(&fs::read(dir.path().join("graphs/encoder.onnx")).unwrap())
            .unwrap();
    model.opset_import.push(OperatorSetIdProto {
        domain: String::new(),
        version: 17,
        ..Default::default()
    });
    update_encoder(
        dir.path(),
        &mut manifest,
        model.write_to_bytes().expect("graph should encode"),
    );
    write_manifest(dir.path(), &manifest);
    let error = validate_onnx_bundle(dir.path(), ParakeetDeployment::v2()).unwrap_err();
    assert!(matches!(error, OnnxBundleError::GraphOpset { .. }));
}

#[cfg(unix)]
#[test]
fn symlinked_manifest_and_declared_paths_are_rejected_before_hashing() {
    use std::os::unix::fs::symlink;

    let (dir, manifest) = write_fixture(ParakeetVersion::V2);
    let outside = tempdir().unwrap();
    fs::write(outside.path().join("manifest.json"), manifest.to_json()).unwrap();
    fs::remove_file(dir.path().join("manifest.json")).unwrap();
    symlink(
        outside.path().join("manifest.json"),
        dir.path().join("manifest.json"),
    )
    .unwrap();
    let error = validate_onnx_bundle(dir.path(), ParakeetDeployment::v2()).unwrap_err();
    assert!(
        matches!(error, OnnxBundleError::NonRegularFile { path } if path.ends_with("manifest.json"))
    );

    let (dir, mut manifest) = write_fixture(ParakeetVersion::V2);
    let outside_graph = dir.path().join("outside.onnx");
    fs::write(
        &outside_graph,
        fs::read(dir.path().join("graphs/encoder.onnx")).unwrap(),
    )
    .unwrap();
    fs::remove_file(dir.path().join("graphs/encoder.onnx")).unwrap();
    symlink(&outside_graph, dir.path().join("graphs/encoder.onnx")).unwrap();
    manifest.graphs.encoder.sha256 = crate::onnx::Sha256Digest::of_file(&outside_graph).unwrap();
    write_manifest(dir.path(), &manifest);
    let error = validate_onnx_bundle(dir.path(), ParakeetDeployment::v2()).unwrap_err();
    assert!(
        matches!(error, OnnxBundleError::NonRegularFile { path } if path.ends_with("encoder.onnx"))
    );

    let (dir, manifest) = write_fixture(ParakeetVersion::V2);
    fs::rename(dir.path().join("graphs"), dir.path().join("graphs-real")).unwrap();
    symlink(dir.path().join("graphs-real"), dir.path().join("graphs")).unwrap();
    write_manifest(dir.path(), &manifest);
    let error = validate_onnx_bundle(dir.path(), ParakeetDeployment::v2()).unwrap_err();
    assert!(matches!(error, OnnxBundleError::NonRegularFile { path } if path.ends_with("graphs")));
}

#[test]
fn canonical_validation_is_deployment_specific() {
    let (dir, _) = write_fixture(ParakeetVersion::V2);
    let error = validate_onnx_bundle(dir.path(), ParakeetDeployment::v3()).unwrap_err();
    assert!(matches!(error, OnnxBundleError::DeploymentMismatch { .. }));
    assert_eq!(V2_REVISION.len(), 40);
}
