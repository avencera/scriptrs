//! Test-only ONNX graphs and bundles encoded by hand
//!
//! The fixtures are tiny graphs that satisfy the split-graph contracts with
//! arithmetic instead of learned weights, so tensor conversion, bounds
//! checks, and the shared decoder can be exercised on the CPU provider.

use std::fs;
use std::path::Path;

#[cfg(feature = "vad")]
use crate::constants::{VAD_CONTEXT_SAMPLES, VAD_STATE_SIZE, VAD_WINDOW_SAMPLES};
use crate::deployment::{ParakeetDeployment, ParakeetVersion};
use crate::onnx::digest::Sha256Digest;
use crate::onnx::manifest::{
    BundlePath, ExportIdentity, FileAsset, GitRevision, GraphAsset, MANIFEST_FILE_NAME,
    ManifestDeployment, ManifestSchema, OnnxBundleManifest, ParakeetGraphContract, ParakeetGraphs,
    SileroGraphContract, SourceCheckpoint, TokenizerAsset, TokenizerFormat, VadAsset,
};

/// Fixture-wide constants that mirror the deployment dimensions
pub(crate) const FEATURE_SIZE: usize = 128;
pub(crate) const ENCODER_HIDDEN: usize = 1_024;
pub(crate) const DECODER_HIDDEN: usize = 640;
pub(crate) const DECODER_LAYERS: usize = 2;

/// Encoder step layout used by the programmable joint fixture
pub(crate) const JOINT_TOKEN_INDEX: usize = 0;
pub(crate) const JOINT_DURATION_INDEX: usize = 1;
pub(crate) const JOINT_PROB_INDEX: usize = 2;
pub(crate) const JOINT_TOP_K_INDEX: usize = 3;

/// Vocabulary id of the Cyrillic token in the generated v3 tokenizer
pub(crate) const V3_CYRILLIC_TOKEN: usize = 100;

pub(crate) const V2_REVISION: &str = "ae9ad07059c7c739ffaf932226a8fe64ae2620b0";
pub(crate) const V3_REVISION: &str = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b";
pub(crate) const VAD_REVISION: &str = "0000000000000000000000000000000000000000";

// ---------------------------------------------------------------------------
// Minimal protobuf writer for onnx.proto

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ElemType {
    Float = 1,
    Int32 = 6,
    Int64 = 7,
}

#[derive(Debug, Clone)]
pub(crate) enum Dim {
    Value(i64),
    Param(&'static str),
}

#[derive(Debug, Clone)]
pub(crate) struct TensorInit {
    name: String,
    dims: Vec<i64>,
    elem: ElemType,
    raw: Vec<u8>,
}

impl TensorInit {
    pub(crate) fn f32(name: &str, dims: &[i64], values: &[f32]) -> Self {
        Self {
            name: name.to_owned(),
            dims: dims.to_vec(),
            elem: ElemType::Float,
            raw: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    }

    pub(crate) fn i32(name: &str, dims: &[i64], values: &[i32]) -> Self {
        Self {
            name: name.to_owned(),
            dims: dims.to_vec(),
            elem: ElemType::Int32,
            raw: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    }

    pub(crate) fn i64(name: &str, dims: &[i64], values: &[i64]) -> Self {
        Self {
            name: name.to_owned(),
            dims: dims.to_vec(),
            elem: ElemType::Int64,
            raw: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    }

    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for dim in &self.dims {
            write_varint_field(&mut out, 1, *dim as u64);
        }
        write_varint_field(&mut out, 2, self.elem as u64);
        write_bytes_field(&mut out, 8, self.name.as_bytes());
        write_bytes_field(&mut out, 9, &self.raw);
        out
    }
}

#[derive(Debug, Clone)]
pub(crate) enum Attr {
    Int(&'static str, i64),
    #[cfg(feature = "vad")]
    Ints(&'static str, Vec<i64>),
}

impl Attr {
    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match self {
            Self::Int(name, value) => {
                write_bytes_field(&mut out, 1, name.as_bytes());
                write_varint_field(&mut out, 3, *value as u64);
                write_varint_field(&mut out, 20, 2);
            }
            #[cfg(feature = "vad")]
            Self::Ints(name, values) => {
                write_bytes_field(&mut out, 1, name.as_bytes());
                for value in values {
                    write_varint_field(&mut out, 8, *value as u64);
                }
                write_varint_field(&mut out, 20, 7);
            }
        }
        out
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Node {
    op: &'static str,
    inputs: Vec<String>,
    outputs: Vec<String>,
    attrs: Vec<Attr>,
}

impl Node {
    pub(crate) fn new(
        op: &'static str,
        inputs: &[&str],
        outputs: &[&str],
        attrs: Vec<Attr>,
    ) -> Self {
        Self {
            op,
            inputs: inputs.iter().map(|s| (*s).to_owned()).collect(),
            outputs: outputs.iter().map(|s| (*s).to_owned()).collect(),
            attrs,
        }
    }

    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for input in &self.inputs {
            write_bytes_field(&mut out, 1, input.as_bytes());
        }
        for output in &self.outputs {
            write_bytes_field(&mut out, 2, output.as_bytes());
        }
        write_bytes_field(&mut out, 4, self.op.as_bytes());
        for attr in &self.attrs {
            write_bytes_field(&mut out, 5, &attr.encode());
        }
        out
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ValueInfo {
    name: String,
    elem: ElemType,
    dims: Vec<Dim>,
}

impl ValueInfo {
    pub(crate) fn new(name: &str, elem: ElemType, dims: &[Dim]) -> Self {
        Self {
            name: name.to_owned(),
            elem,
            dims: dims.to_vec(),
        }
    }

    fn encode(&self) -> Vec<u8> {
        let mut shape = Vec::new();
        for dim in &self.dims {
            let mut dimension = Vec::new();
            match dim {
                Dim::Value(value) => write_varint_field(&mut dimension, 1, *value as u64),
                Dim::Param(param) => write_bytes_field(&mut dimension, 2, param.as_bytes()),
            }
            write_bytes_field(&mut shape, 1, &dimension);
        }

        let mut tensor_type = Vec::new();
        write_varint_field(&mut tensor_type, 1, self.elem as u64);
        // an explicit empty shape message marks a rank-0 tensor
        write_bytes_field(&mut tensor_type, 2, &shape);

        let mut type_proto = Vec::new();
        write_bytes_field(&mut type_proto, 1, &tensor_type);

        let mut out = Vec::new();
        write_bytes_field(&mut out, 1, self.name.as_bytes());
        write_bytes_field(&mut out, 2, &type_proto);
        out
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Graph {
    pub(crate) name: &'static str,
    pub(crate) nodes: Vec<Node>,
    pub(crate) initializers: Vec<TensorInit>,
    pub(crate) inputs: Vec<ValueInfo>,
    pub(crate) outputs: Vec<ValueInfo>,
}

impl Graph {
    /// Encode as an ONNX `ModelProto` with IR version 8 and default opset 17
    pub(crate) fn encode_model(&self) -> Vec<u8> {
        let mut graph = Vec::new();
        for node in &self.nodes {
            write_bytes_field(&mut graph, 1, &node.encode());
        }
        write_bytes_field(&mut graph, 2, self.name.as_bytes());
        for initializer in &self.initializers {
            write_bytes_field(&mut graph, 5, &initializer.encode());
        }
        for input in &self.inputs {
            write_bytes_field(&mut graph, 11, &input.encode());
        }
        for output in &self.outputs {
            write_bytes_field(&mut graph, 12, &output.encode());
        }

        let mut opset = Vec::new();
        write_varint_field(&mut opset, 2, 17);

        let mut model = Vec::new();
        write_varint_field(&mut model, 1, 8);
        write_bytes_field(&mut model, 2, b"scriptrs-test-fixture");
        write_bytes_field(&mut model, 7, &graph);
        write_bytes_field(&mut model, 8, &opset);
        model
    }
}

fn write_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn write_varint_field(out: &mut Vec<u8>, field: u32, value: u64) {
    write_varint(out, u64::from(field) << 3);
    write_varint(out, value);
}

fn write_bytes_field(out: &mut Vec<u8>, field: u32, bytes: &[u8]) {
    write_varint(out, (u64::from(field) << 3) | 2);
    write_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

// ---------------------------------------------------------------------------
// Fixture graphs

/// Knobs for producing encoder graphs that violate the contract on purpose
#[derive(Debug, Clone)]
pub(crate) struct EncoderFixture {
    pub(crate) length_dtype: ElemType,
    pub(crate) encoder_output_name: &'static str,
    pub(crate) hidden: usize,
    pub(crate) length_rank: usize,
}

impl Default for EncoderFixture {
    fn default() -> Self {
        Self {
            length_dtype: ElemType::Int32,
            encoder_output_name: "encoder",
            hidden: ENCODER_HIDDEN,
            length_rank: 1,
        }
    }
}

/// `encoder[0, h, t] = mel[0, h % 128, t]`, `encoder_length = mel_length / 8`
pub(crate) fn encoder_graph(fixture: &EncoderFixture) -> Vec<u8> {
    let repeat = (fixture.hidden / FEATURE_SIZE) as i64;
    let length_dims: Vec<Dim> = vec![Dim::Value(1); fixture.length_rank];
    let eight = match fixture.length_dtype {
        ElemType::Int32 => TensorInit::i32("eight", &[1], &[8]),
        ElemType::Int64 => TensorInit::i64("eight", &[1], &[8]),
        ElemType::Float => TensorInit::f32("eight", &[1], &[8.0]),
    };

    Graph {
        name: "encoder_fixture",
        nodes: vec![
            Node::new(
                "Tile",
                &["mel", "repeats"],
                &[fixture.encoder_output_name],
                Vec::new(),
            ),
            Node::new(
                "Div",
                &["mel_length", "eight"],
                &["encoder_length"],
                Vec::new(),
            ),
        ],
        initializers: vec![TensorInit::i64("repeats", &[3], &[1, repeat, 1]), eight],
        inputs: vec![
            ValueInfo::new(
                "mel",
                ElemType::Float,
                &[
                    Dim::Value(1),
                    Dim::Value(FEATURE_SIZE as i64),
                    Dim::Param("time"),
                ],
            ),
            ValueInfo::new("mel_length", fixture.length_dtype, &length_dims),
        ],
        outputs: vec![
            ValueInfo::new(
                fixture.encoder_output_name,
                ElemType::Float,
                &[
                    Dim::Value(1),
                    Dim::Value(fixture.hidden as i64),
                    Dim::Param("time"),
                ],
            ),
            ValueInfo::new("encoder_length", fixture.length_dtype, &length_dims),
        ],
    }
    .encode_model()
}

/// `h_out = h_in + targets`, `c_out = c_in + 1`, `decoder = h_out[0]` as `[1, 640, 1]`
pub(crate) fn decoder_graph() -> Vec<u8> {
    let state = [
        Dim::Value(DECODER_LAYERS as i64),
        Dim::Value(1),
        Dim::Value(DECODER_HIDDEN as i64),
    ];
    Graph {
        name: "decoder_fixture",
        nodes: vec![
            Node::new(
                "Cast",
                &["targets"],
                &["targets_f"],
                vec![Attr::Int("to", ElemType::Float as i64)],
            ),
            Node::new("Add", &["h_in", "targets_f"], &["h_out"], Vec::new()),
            Node::new("Add", &["c_in", "one"], &["c_out"], Vec::new()),
            Node::new(
                "Slice",
                &["h_out", "layer_start", "layer_end", "layer_axis"],
                &["layer0"],
                Vec::new(),
            ),
            Node::new(
                "Reshape",
                &["layer0", "decoder_shape"],
                &["decoder"],
                Vec::new(),
            ),
        ],
        initializers: vec![
            TensorInit::f32("one", &[], &[1.0]),
            TensorInit::i64("layer_start", &[1], &[0]),
            TensorInit::i64("layer_end", &[1], &[1]),
            TensorInit::i64("layer_axis", &[1], &[0]),
            TensorInit::i64("decoder_shape", &[3], &[1, DECODER_HIDDEN as i64, 1]),
        ],
        inputs: vec![
            ValueInfo::new("targets", ElemType::Int32, &[Dim::Value(1), Dim::Value(1)]),
            ValueInfo::new("target_length", ElemType::Int32, &[Dim::Value(1)]),
            ValueInfo::new("h_in", ElemType::Float, &state),
            ValueInfo::new("c_in", ElemType::Float, &state),
        ],
        outputs: vec![
            ValueInfo::new(
                "decoder",
                ElemType::Float,
                &[
                    Dim::Value(1),
                    Dim::Value(DECODER_HIDDEN as i64),
                    Dim::Value(1),
                ],
            ),
            ValueInfo::new("h_out", ElemType::Float, &state),
            ValueInfo::new("c_out", ElemType::Float, &state),
        ],
    }
    .encode_model()
}

/// How the joint fixture produces its outputs
#[derive(Debug, Clone, Copy)]
pub(crate) enum JointFixture {
    /// Read token, duration, probability, and top-K from the encoder step
    Programmable { top_k: Option<usize> },
    /// Always emit the blank with duration 1
    ConstantBlank {
        blank_id: usize,
        top_k: Option<usize>,
    },
}

pub(crate) fn joint_graph(fixture: JointFixture) -> Vec<u8> {
    let scalar = [Dim::Value(1), Dim::Value(1), Dim::Value(1)];
    let inputs = vec![
        ValueInfo::new(
            "encoder_step",
            ElemType::Float,
            &[
                Dim::Value(1),
                Dim::Value(ENCODER_HIDDEN as i64),
                Dim::Value(1),
            ],
        ),
        ValueInfo::new(
            "decoder_step",
            ElemType::Float,
            &[
                Dim::Value(1),
                Dim::Value(DECODER_HIDDEN as i64),
                Dim::Value(1),
            ],
        ),
    ];
    let mut outputs = vec![
        ValueInfo::new("token_id", ElemType::Int32, &scalar),
        ValueInfo::new("token_prob", ElemType::Float, &scalar),
        ValueInfo::new("duration", ElemType::Int32, &scalar),
    ];
    let mut nodes = Vec::new();
    let mut initializers = vec![TensorInit::i64("axis1", &[1], &[1])];

    match fixture {
        JointFixture::Programmable { top_k } => {
            slice_scalar(&mut nodes, &mut initializers, "token_f", JOINT_TOKEN_INDEX);
            cast_i32(&mut nodes, "token_f", "token_id");
            slice_scalar(
                &mut nodes,
                &mut initializers,
                "token_prob",
                JOINT_PROB_INDEX,
            );
            slice_scalar(
                &mut nodes,
                &mut initializers,
                "duration_f",
                JOINT_DURATION_INDEX,
            );
            cast_i32(&mut nodes, "duration_f", "duration");

            if let Some(k) = top_k {
                let ids_start = JOINT_TOP_K_INDEX;
                let logits_start = JOINT_TOP_K_INDEX + k;
                slice_range(
                    &mut nodes,
                    &mut initializers,
                    "ids_raw",
                    ids_start,
                    ids_start + k,
                );
                slice_range(
                    &mut nodes,
                    &mut initializers,
                    "logits_raw",
                    logits_start,
                    logits_start + k,
                );
                initializers.push(TensorInit::i64("top_k_shape", &[2], &[1, k as i64]));
                nodes.push(Node::new(
                    "Reshape",
                    &["ids_raw", "top_k_shape"],
                    &["ids_f"],
                    Vec::new(),
                ));
                cast_i32(&mut nodes, "ids_f", "top_k_ids");
                nodes.push(Node::new(
                    "Reshape",
                    &["logits_raw", "top_k_shape"],
                    &["top_k_logits"],
                    Vec::new(),
                ));
                push_top_k_outputs(&mut outputs, k);
            }
        }
        JointFixture::ConstantBlank { blank_id, top_k } => {
            initializers.push(TensorInit::i32("blank", &[1, 1, 1], &[blank_id as i32]));
            initializers.push(TensorInit::f32("half", &[1, 1, 1], &[0.5]));
            initializers.push(TensorInit::i32("one_step", &[1, 1, 1], &[1]));
            nodes.push(Node::new("Identity", &["blank"], &["token_id"], Vec::new()));
            nodes.push(Node::new(
                "Identity",
                &["half"],
                &["token_prob"],
                Vec::new(),
            ));
            nodes.push(Node::new(
                "Identity",
                &["one_step"],
                &["duration"],
                Vec::new(),
            ));

            if let Some(k) = top_k {
                let ids: Vec<i32> = (0..k).map(|i| i as i32).collect();
                let logits: Vec<f32> = (0..k).map(|i| -(i as f32)).collect();
                initializers.push(TensorInit::i32("ids_const", &[1, k as i64], &ids));
                initializers.push(TensorInit::f32("logits_const", &[1, k as i64], &logits));
                nodes.push(Node::new(
                    "Identity",
                    &["ids_const"],
                    &["top_k_ids"],
                    Vec::new(),
                ));
                nodes.push(Node::new(
                    "Identity",
                    &["logits_const"],
                    &["top_k_logits"],
                    Vec::new(),
                ));
                push_top_k_outputs(&mut outputs, k);
            }
        }
    }

    Graph {
        name: "joint_fixture",
        nodes,
        initializers,
        inputs,
        outputs,
    }
    .encode_model()
}

fn push_top_k_outputs(outputs: &mut Vec<ValueInfo>, k: usize) {
    let shape = [Dim::Value(1), Dim::Value(k as i64)];
    outputs.push(ValueInfo::new("top_k_ids", ElemType::Int32, &shape));
    outputs.push(ValueInfo::new("top_k_logits", ElemType::Float, &shape));
}

fn slice_scalar(
    nodes: &mut Vec<Node>,
    initializers: &mut Vec<TensorInit>,
    output: &str,
    index: usize,
) {
    slice_range(nodes, initializers, output, index, index + 1);
}

fn slice_range(
    nodes: &mut Vec<Node>,
    initializers: &mut Vec<TensorInit>,
    output: &str,
    start: usize,
    end: usize,
) {
    let start_name = format!("{output}_start");
    let end_name = format!("{output}_end");
    initializers.push(TensorInit::i64(&start_name, &[1], &[start as i64]));
    initializers.push(TensorInit::i64(&end_name, &[1], &[end as i64]));
    nodes.push(Node::new(
        "Slice",
        &["encoder_step", &start_name, &end_name, "axis1"],
        &[output],
        Vec::new(),
    ));
}

fn cast_i32(nodes: &mut Vec<Node>, input: &str, output: &str) {
    nodes.push(Node::new(
        "Cast",
        &[input],
        &[output],
        vec![Attr::Int("to", ElemType::Int32 as i64)],
    ));
}

/// `output = sigmoid(mean(input) + mean(state))`, `stateN = state + mean(input)`
#[cfg(feature = "vad")]
pub(crate) fn silero_graph() -> Vec<u8> {
    let state = [
        Dim::Value(2),
        Dim::Value(1),
        Dim::Value(VAD_STATE_SIZE as i64),
    ];
    Graph {
        name: "silero_fixture",
        nodes: vec![
            Node::new(
                "ReduceMean",
                &["input"],
                &["window_mean"],
                vec![Attr::Ints("axes", vec![1]), Attr::Int("keepdims", 1)],
            ),
            Node::new(
                "ReduceMean",
                &["state"],
                &["state_mean"],
                vec![Attr::Int("keepdims", 0)],
            ),
            Node::new(
                "Add",
                &["window_mean", "state_mean"],
                &["pre_activation"],
                Vec::new(),
            ),
            Node::new("Sigmoid", &["pre_activation"], &["output"], Vec::new()),
            Node::new("Add", &["state", "window_mean"], &["stateN"], Vec::new()),
        ],
        initializers: Vec::new(),
        inputs: vec![
            ValueInfo::new(
                "input",
                ElemType::Float,
                &[
                    Dim::Value(1),
                    Dim::Value((VAD_WINDOW_SAMPLES + VAD_CONTEXT_SAMPLES) as i64),
                ],
            ),
            ValueInfo::new("state", ElemType::Float, &state),
            ValueInfo::new("sr", ElemType::Int64, &[]),
        ],
        outputs: vec![
            ValueInfo::new("output", ElemType::Float, &[Dim::Value(1), Dim::Value(1)]),
            ValueInfo::new("stateN", ElemType::Float, &state),
        ],
    }
    .encode_model()
}

/// Expected fixture probability for one window given the accumulated state mean
#[cfg(feature = "vad")]
pub(crate) fn silero_fixture_probability(window: &[f32], state_mean: f32) -> f32 {
    let window_mean = window.iter().sum::<f32>() / window.len() as f32;
    1.0 / (1.0 + (-(window_mean + state_mean)).exp())
}

// ---------------------------------------------------------------------------
// Tokenizers

pub(crate) fn v2_vocab_txt() -> String {
    let mut text = String::from("<unk> 0\n");
    for id in 1..1_024 {
        text.push_str(&format!("▁t{id} {id}\n"));
    }
    text.push_str("<blk> 1024\n");
    text
}

pub(crate) fn v3_vocab_json() -> String {
    let mut tokens: Vec<String> = (0..8_192).map(|id| format!("▁t{id}")).collect();
    tokens[0] = "<unk>".to_owned();
    tokens[V3_CYRILLIC_TOKEN] = "кот".to_owned();
    tokens[8_191] = "Œ".to_owned();
    serde_json::to_string(&tokens).expect("string vector serializes")
}

// ---------------------------------------------------------------------------
// Bundle writer

/// Description of one fixture bundle on disk
#[derive(Debug, Clone)]
pub(crate) struct FixtureBundle {
    pub(crate) version: ParakeetVersion,
    pub(crate) joint: JointFixture,
    pub(crate) encoder: EncoderFixture,
    pub(crate) vad: bool,
}

impl FixtureBundle {
    pub(crate) fn blank(version: ParakeetVersion) -> Self {
        let deployment = deployment_for(version);
        let top_k = deployment.model_io().joint().top_k_ids.map(|_| 2);
        Self {
            version,
            joint: JointFixture::ConstantBlank {
                blank_id: deployment.blank_id(),
                top_k,
            },
            encoder: EncoderFixture::default(),
            vad: false,
        }
    }

    pub(crate) fn programmable(version: ParakeetVersion) -> Self {
        let deployment = deployment_for(version);
        let top_k = deployment.model_io().joint().top_k_ids.map(|_| 2);
        Self {
            version,
            joint: JointFixture::Programmable { top_k },
            encoder: EncoderFixture::default(),
            vad: false,
        }
    }

    #[cfg(feature = "vad")]
    pub(crate) fn with_vad(mut self) -> Self {
        self.vad = true;
        self
    }

    pub(crate) fn with_encoder(mut self, encoder: EncoderFixture) -> Self {
        self.encoder = encoder;
        self
    }

    /// Write graphs, tokenizer, and manifest under `root`; return the manifest
    pub(crate) fn write(&self, root: &Path) -> OnnxBundleManifest {
        fs::create_dir_all(root.join("graphs")).unwrap();
        fs::write(
            root.join("graphs/encoder.onnx"),
            encoder_graph(&self.encoder),
        )
        .unwrap();
        fs::write(root.join("graphs/decoder.onnx"), decoder_graph()).unwrap();
        fs::write(root.join("graphs/joint.onnx"), joint_graph(self.joint)).unwrap();

        let (tokenizer_name, tokenizer_text, format, revision) = match self.version {
            ParakeetVersion::V2 => (
                "vocab.txt",
                v2_vocab_txt(),
                TokenizerFormat::SentencePieceVocabTxt,
                V2_REVISION,
            ),
            ParakeetVersion::V3 => (
                "parakeet_vocab.json",
                v3_vocab_json(),
                TokenizerFormat::ParakeetVocabJson,
                V3_REVISION,
            ),
        };
        fs::write(root.join(tokenizer_name), tokenizer_text).unwrap();

        let vad = self.vad.then(|| {
            fs::create_dir_all(root.join("vad")).unwrap();
            fs::write(root.join("vad/silero_vad.onnx"), vad_graph_bytes()).unwrap();
            VadAsset {
                graph: graph_asset(root, "vad/silero_vad.onnx"),
                source: SourceCheckpoint {
                    repository: "snakers4/silero-vad".to_owned(),
                    revision: git_revision(VAD_REVISION),
                },
                graph_contract: SileroGraphContract::V1,
            }
        });

        let manifest = OnnxBundleManifest {
            schema: ManifestSchema::ParakeetOnnxBundleV1,
            deployment: ManifestDeployment::from_version(self.version),
            source: SourceCheckpoint {
                repository: match self.version {
                    ParakeetVersion::V2 => "nvidia/parakeet-tdt-0.6b-v2".to_owned(),
                    ParakeetVersion::V3 => "nvidia/parakeet-tdt-0.6b-v3".to_owned(),
                },
                revision: git_revision(revision),
            },
            export: ExportIdentity {
                exporter: "scriptrs-test-fixture".to_owned(),
                exporter_version: "0".to_owned(),
                opset: 17,
                graph_contract: ParakeetGraphContract::SplitV1,
            },
            tokenizer: TokenizerAsset {
                path: bundle_path(tokenizer_name),
                sha256: digest(root, tokenizer_name),
                format,
                source_revision: git_revision(revision),
            },
            graphs: ParakeetGraphs {
                encoder: graph_asset(root, "graphs/encoder.onnx"),
                decoder: graph_asset(root, "graphs/decoder.onnx"),
                joint: graph_asset(root, "graphs/joint.onnx"),
            },
            vad,
        };
        write_manifest(root, &manifest);
        manifest
    }
}

#[cfg(feature = "vad")]
fn vad_graph_bytes() -> Vec<u8> {
    silero_graph()
}

// without the `vad` feature the VAD graph is still a declared, digest-checked
// file; a stand-in body keeps closure tests meaningful
#[cfg(not(feature = "vad"))]
fn vad_graph_bytes() -> Vec<u8> {
    b"not-loaded-without-vad".to_vec()
}

pub(crate) fn deployment_for(version: ParakeetVersion) -> ParakeetDeployment {
    match version {
        ParakeetVersion::V2 => ParakeetDeployment::v2(),
        ParakeetVersion::V3 => ParakeetDeployment::v3(),
    }
}

pub(crate) fn write_manifest(root: &Path, manifest: &OnnxBundleManifest) {
    fs::write(root.join(MANIFEST_FILE_NAME), manifest.to_json()).unwrap();
}

pub(crate) fn bundle_path(path: &str) -> BundlePath {
    BundlePath::try_from(path.to_owned()).unwrap()
}

pub(crate) fn git_revision(revision: &str) -> GitRevision {
    GitRevision::try_from(revision.to_owned()).unwrap()
}

pub(crate) fn digest(root: &Path, relative: &str) -> Sha256Digest {
    Sha256Digest::of_file(&root.join(relative)).unwrap()
}

fn graph_asset(root: &Path, relative: &str) -> GraphAsset {
    GraphAsset {
        path: bundle_path(relative),
        sha256: digest(root, relative),
        external_data: Vec::<FileAsset>::new(),
    }
}
