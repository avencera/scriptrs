use std::fs;
use std::path::Path;

use ndarray::{Array1, Array2, Array3};
use tempfile::{TempDir, tempdir};

use crate::config::TranscriptionConfig;
use crate::deployment::{ParakeetDeployment, ParakeetVersion};
use crate::error::TranscriptionError;
use crate::language::{LanguageSelection, ParakeetLanguage};
use crate::model::ParakeetModel;
use crate::models::ModelBundle;
use crate::onnx::OnnxCudaBundle;
use crate::onnx::error::{GraphKind, OnnxBundleError};
use crate::onnx::manifest::{OnnxBundleManifest, TokenizerFormat};
use crate::onnx::session::CudaDevice;
use crate::onnx::test_support::{
    ElemType, EncoderFixture, FEATURE_SIZE, FixtureBundle, JOINT_DURATION_INDEX, JOINT_PROB_INDEX,
    JOINT_TOKEN_INDEX, JOINT_TOP_K_INDEX, V2_REVISION, V3_CYRILLIC_TOKEN, deployment_for, digest,
    git_revision, write_manifest,
};
use crate::pipeline::TranscriptionPipeline;

fn write_bundle(fixture: &FixtureBundle) -> (TempDir, OnnxBundleManifest) {
    let dir = tempdir().unwrap();
    let manifest = fixture.write(dir.path());
    (dir, manifest)
}

fn open_cpu(root: &Path, version: ParakeetVersion) -> Result<OnnxCudaBundle, OnnxBundleError> {
    OnnxCudaBundle::open_on_cpu(root, deployment_for(version))
}

fn expect_bundle_error(result: Result<OnnxCudaBundle, OnnxBundleError>) -> OnnxBundleError {
    match result {
        Ok(_) => panic!("bundle must be rejected"),
        Err(error) => error,
    }
}

/// Load the shared greedy decoder on top of a CPU fixture bundle
fn fixture_model(root: &Path, version: ParakeetVersion) -> (ParakeetModel, TranscriptionConfig) {
    let onnx = open_cpu(root, version).unwrap();
    let vocab = onnx.vocabulary().clone();
    let bundle = ModelBundle::from_onnx_cuda(onnx);
    let config = TranscriptionConfig::from_deployment(bundle.deployment());
    let model = ParakeetModel::from_bundle(&bundle, &vocab, &config).unwrap();
    (model, config)
}

/// One joint step encoded into an encoder frame for the programmable fixture
#[derive(Debug, Clone, Copy)]
struct Step {
    token: f32,
    duration: f32,
    prob: f32,
    top_k_ids: [f32; 2],
    top_k_logits: [f32; 2],
}

impl Step {
    fn blank(blank_id: usize) -> Self {
        Self {
            token: blank_id as f32,
            duration: 1.0,
            prob: 0.5,
            top_k_ids: [blank_id as f32, 0.0],
            top_k_logits: [1.0, 0.0],
        }
    }

    fn token(token: usize, duration: usize, prob: f32) -> Self {
        Self {
            token: token as f32,
            duration: duration as f32,
            prob,
            top_k_ids: [token as f32, 0.0],
            top_k_logits: [1.0, 0.0],
        }
    }
}

/// Mel features whose first columns program the joint fixture; the encoder
/// fixture tiles these into every hidden block, so column `c` of frame `t`
/// reaches joint index `c`
fn programmed_features(steps: &[Step], frames: usize) -> Array2<f32> {
    let mut features = Array2::<f32>::zeros((frames, FEATURE_SIZE));
    for (frame, step) in steps.iter().enumerate() {
        features[[frame, JOINT_TOKEN_INDEX]] = step.token;
        features[[frame, JOINT_DURATION_INDEX]] = step.duration;
        features[[frame, JOINT_PROB_INDEX]] = step.prob;
        features[[frame, JOINT_TOP_K_INDEX]] = step.top_k_ids[0];
        features[[frame, JOINT_TOP_K_INDEX + 1]] = step.top_k_ids[1];
        features[[frame, JOINT_TOP_K_INDEX + 2]] = step.top_k_logits[0];
        features[[frame, JOINT_TOP_K_INDEX + 3]] = step.top_k_logits[1];
    }
    features
}

// ---------------------------------------------------------------------------
// Startup validation

#[test]
fn cpu_fixture_bundle_opens_for_both_versions() {
    for version in [ParakeetVersion::V2, ParakeetVersion::V3] {
        let (dir, manifest) = write_bundle(&FixtureBundle::blank(version));
        let bundle = open_cpu(dir.path(), version).unwrap();
        assert_eq!(bundle.version(), version);
        assert_eq!(bundle.manifest(), &manifest);
        assert!(!bundle.has_vad());
        assert_eq!(bundle.root(), dir.path());
    }
}

#[test]
fn deployment_mismatch_is_rejected_before_any_file_is_read() {
    let (dir, _) = write_bundle(&FixtureBundle::blank(ParakeetVersion::V2));
    let error = expect_bundle_error(open_cpu(dir.path(), ParakeetVersion::V3));
    assert!(matches!(
        error,
        OnnxBundleError::DeploymentMismatch {
            expected: ParakeetVersion::V3,
            found: ParakeetVersion::V2,
        }
    ));
}

#[test]
fn unsupported_schema_is_rejected() {
    let (dir, manifest) = write_bundle(&FixtureBundle::blank(ParakeetVersion::V2));
    let text = manifest
        .to_json()
        .replace("parakeet-onnx-bundle/1", "parakeet-onnx-bundle/7");
    fs::write(dir.path().join("manifest.json"), text).unwrap();

    let error = expect_bundle_error(open_cpu(dir.path(), ParakeetVersion::V2));
    assert!(matches!(error, OnnxBundleError::UnsupportedSchema { .. }));
}

#[test]
fn unknown_manifest_fields_and_escaping_paths_are_parse_errors() {
    let (dir, manifest) = write_bundle(&FixtureBundle::blank(ParakeetVersion::V2));
    let mut text = manifest.to_json();
    text.insert_str(text.len() - 2, ",\n  \"extra\": true\n");
    fs::write(dir.path().join("manifest.json"), text).unwrap();
    let error = expect_bundle_error(open_cpu(dir.path(), ParakeetVersion::V2));
    assert!(matches!(error, OnnxBundleError::ManifestParse { .. }));

    let text = manifest
        .to_json()
        .replace("graphs/encoder.onnx", "../graphs/encoder.onnx");
    fs::write(dir.path().join("manifest.json"), text).unwrap();
    let error = expect_bundle_error(open_cpu(dir.path(), ParakeetVersion::V2));
    assert!(matches!(
        error,
        OnnxBundleError::ManifestParse { message, .. } if message.contains("invalid bundle path")
    ));
}

#[test]
fn digest_mismatch_is_rejected() {
    let (dir, mut manifest) = write_bundle(&FixtureBundle::blank(ParakeetVersion::V2));
    fs::write(dir.path().join("vocab.txt"), "<unk> 0\n<blk> 1\n").unwrap();
    let error = expect_bundle_error(open_cpu(dir.path(), ParakeetVersion::V2));
    assert!(matches!(
        error,
        OnnxBundleError::DigestMismatch { path, expected, .. }
            if path.ends_with("vocab.txt") && expected == manifest.tokenizer.sha256
    ));

    // repairing the digest but keeping the wrong contents fails later, in the
    // deployment tokenizer contract, never silently
    manifest.tokenizer.sha256 = digest(dir.path(), "vocab.txt");
    write_manifest(dir.path(), &manifest);
    let error = open_cpu(dir.path(), ParakeetVersion::V2)
        .expect_err("tokenizer must fail during bundle open");
    assert!(matches!(error, OnnxBundleError::InvalidTokenizer { .. }));
}

#[test]
fn missing_declared_file_is_rejected() {
    let (dir, _) = write_bundle(&FixtureBundle::blank(ParakeetVersion::V2));
    fs::remove_file(dir.path().join("graphs/joint.onnx")).unwrap();
    let error = expect_bundle_error(open_cpu(dir.path(), ParakeetVersion::V2));
    assert!(matches!(
        error,
        OnnxBundleError::MissingFile { path } if path.ends_with("graphs/joint.onnx")
    ));
}

#[test]
fn undeclared_files_break_the_closure() {
    let (dir, _) = write_bundle(&FixtureBundle::blank(ParakeetVersion::V2));
    fs::create_dir_all(dir.path().join("graphs/extra")).unwrap();
    fs::write(
        dir.path().join("graphs/extra/decoder_joint-model.onnx"),
        b"x",
    )
    .unwrap();
    let error = expect_bundle_error(open_cpu(dir.path(), ParakeetVersion::V2));
    assert!(matches!(
        error,
        OnnxBundleError::UndeclaredFile { path } if path.ends_with("decoder_joint-model.onnx")
    ));
}

#[test]
fn duplicate_paths_are_rejected() {
    let (dir, mut manifest) = write_bundle(&FixtureBundle::blank(ParakeetVersion::V2));
    manifest.graphs.decoder = manifest.graphs.encoder.clone();
    write_manifest(dir.path(), &manifest);
    let error = expect_bundle_error(open_cpu(dir.path(), ParakeetVersion::V2));
    assert!(matches!(
        error,
        OnnxBundleError::DuplicatePath { path } if path == "graphs/encoder.onnx"
    ));
}

#[test]
fn tokenizer_format_and_revision_must_match_the_checkpoint() {
    let (dir, mut manifest) = write_bundle(&FixtureBundle::blank(ParakeetVersion::V2));
    manifest.tokenizer.format = TokenizerFormat::ParakeetVocabJson;
    write_manifest(dir.path(), &manifest);
    let error = expect_bundle_error(open_cpu(dir.path(), ParakeetVersion::V2));
    assert!(matches!(
        error,
        OnnxBundleError::TokenizerFormatMismatch {
            version: ParakeetVersion::V2,
            found: TokenizerFormat::ParakeetVocabJson,
            ..
        }
    ));

    manifest.tokenizer.format = TokenizerFormat::SentencePieceVocabTxt;
    manifest.tokenizer.source_revision = git_revision("541d1f99c6b0c3cd0b11a95167540bb8edefd82b");
    write_manifest(dir.path(), &manifest);
    let error = expect_bundle_error(open_cpu(dir.path(), ParakeetVersion::V2));
    assert!(matches!(
        error,
        OnnxBundleError::TokenizerRevisionMismatch { checkpoint, .. } if checkpoint == V2_REVISION
    ));
}

#[test]
fn graph_io_metadata_is_checked_against_the_contract() {
    let cases: [(EncoderFixture, &str); 4] = [
        (
            EncoderFixture {
                encoder_output_name: "encoded",
                ..EncoderFixture::default()
            },
            "output `encoder` is missing",
        ),
        (
            EncoderFixture {
                length_dtype: ElemType::Int64,
                ..EncoderFixture::default()
            },
            "`mel_length` has element type int64",
        ),
        (
            EncoderFixture {
                length_rank: 2,
                ..EncoderFixture::default()
            },
            "`mel_length` declares rank 2",
        ),
        (
            EncoderFixture {
                hidden: 512,
                ..EncoderFixture::default()
            },
            "`encoder` axis 1 declares size 512",
        ),
    ];

    for (encoder, expected_message) in cases {
        let fixture = FixtureBundle::blank(ParakeetVersion::V2).with_encoder(encoder);
        let (dir, _) = write_bundle(&fixture);
        let error = expect_bundle_error(open_cpu(dir.path(), ParakeetVersion::V2));
        match error {
            OnnxBundleError::GraphIo {
                graph: GraphKind::Encoder,
                message,
                contract,
                ..
            } => {
                assert_eq!(contract, "avencera.parakeet-split/1");
                assert!(
                    message.contains(expected_message),
                    "expected `{expected_message}` in `{message}`"
                );
            }
            other => panic!("unexpected error: {other}"),
        }
    }
}

#[test]
fn graph_in_the_wrong_slot_is_rejected_by_name() {
    let (dir, _) = write_bundle(&FixtureBundle::blank(ParakeetVersion::V2));
    fs::copy(
        dir.path().join("graphs/decoder.onnx"),
        dir.path().join("graphs/joint.onnx"),
    )
    .unwrap();
    let mut manifest = OnnxBundleManifest::read(dir.path()).unwrap();
    manifest.graphs.joint.sha256 = digest(dir.path(), "graphs/joint.onnx");
    write_manifest(dir.path(), &manifest);

    let error = expect_bundle_error(open_cpu(dir.path(), ParakeetVersion::V2));
    assert!(matches!(
        error,
        OnnxBundleError::GraphIo { graph: GraphKind::Joint, message, .. }
            if message.contains("input `encoder_step` is missing")
    ));
}

#[test]
fn production_constructor_refuses_without_cuda() {
    let (dir, _) = write_bundle(&FixtureBundle::blank(ParakeetVersion::V2));
    let result = OnnxCudaBundle::open(dir.path(), ParakeetDeployment::v2(), CudaDevice::new(0));

    // hosts that can never run CUDA must be refused before any graph loads;
    // a real CUDA host is exercised by the ignored GPU test instead
    let cuda_capable_platform = cfg!(any(
        all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        all(target_os = "windows", target_arch = "x86_64")
    ));
    if cuda_capable_platform {
        return;
    }
    assert!(matches!(
        result,
        Err(TranscriptionError::OnnxBundle(
            OnnxBundleError::CudaUnavailable { .. }
        ))
    ));
}

#[test]
#[ignore = "requires a CUDA-capable host with the ONNX Runtime CUDA execution provider"]
fn production_constructor_runs_fixture_graphs_on_cuda() {
    let (dir, _) = write_bundle(&FixtureBundle::blank(ParakeetVersion::V2));
    let bundle =
        OnnxCudaBundle::open(dir.path(), ParakeetDeployment::v2(), CudaDevice::default()).unwrap();
    let pipeline = TranscriptionPipeline::from_bundle(ModelBundle::from_onnx_cuda(bundle)).unwrap();
    let result = pipeline.run(&vec![0.0f32; 16_000]).unwrap();
    assert!(result.text.is_empty());
}

// ---------------------------------------------------------------------------
// Tensor conversion and the shared decoder

#[test]
fn blank_fixture_pipeline_transcribes_silence_to_empty_text() {
    for version in [ParakeetVersion::V2, ParakeetVersion::V3] {
        let (dir, _) = write_bundle(&FixtureBundle::blank(version));
        let bundle = ModelBundle::from_onnx_cuda(open_cpu(dir.path(), version).unwrap());
        bundle.validate_base().unwrap();
        let pipeline = TranscriptionPipeline::from_bundle(bundle).unwrap();
        let result = pipeline.run(&vec![0.0f32; 16_000]).unwrap();
        assert!(result.text.is_empty());
        assert!(result.tokens.is_empty());
        assert_eq!(result.duration_seconds, 1.0);
    }
}

#[test]
fn decoder_graph_threads_targets_and_state_through_the_contract_shapes() {
    let (dir, _) = write_bundle(&FixtureBundle::blank(ParakeetVersion::V2));
    let bundle = open_cpu(dir.path(), ParakeetVersion::V2).unwrap();
    let model = bundle.parakeet_model();
    let dims = ParakeetDeployment::v2().decoder_dimensions();
    let state = Array3::<f32>::from_elem((dims.decoder_layers(), 1, dims.decoder_hidden()), 0.25);

    let outputs = model
        .run_decoder(
            &Array2::from_elem((1, 1), 5),
            &Array1::from_elem(1, 1),
            &state,
            &state,
        )
        .unwrap();

    assert_eq!(outputs.decoder_step.shape(), &[1, dims.decoder_hidden(), 1]);
    assert!(outputs.decoder_step.iter().all(|v| *v == 5.25));
    assert_eq!(
        outputs.hidden_state.shape(),
        &[dims.decoder_layers(), 1, dims.decoder_hidden()]
    );
    assert!(outputs.hidden_state.iter().all(|v| *v == 5.25));
    assert!(outputs.cell_state.iter().all(|v| *v == 1.25));

    // a non-contiguous view must be refused instead of misread
    let transposed = state.clone().permuted_axes([2, 1, 0]);
    let error = model
        .run_decoder(
            &Array2::from_elem((1, 1), 5),
            &Array1::from_elem(1, 1),
            &transposed,
            &state,
        )
        .expect_err("non-contiguous state must be rejected");
    assert!(matches!(
        error,
        TranscriptionError::InvalidModelOutput(message) if message.contains("not contiguous")
    ));
}

#[test]
fn encoder_graph_maps_lengths_and_rejects_oversized_frames() {
    let (dir, _) = write_bundle(&FixtureBundle::blank(ParakeetVersion::V2));
    let model = open_cpu(dir.path(), ParakeetVersion::V2)
        .unwrap()
        .parakeet_model();
    let config = TranscriptionConfig::from_deployment(&ParakeetDeployment::v2());
    let target_frames = config.max_feature_frames();
    let mut mel = vec![0.0f32; FEATURE_SIZE * target_frames];
    // feature 3 at frame 2 must land at hidden 3, 131, 259, ... at frame 2
    mel[3 * target_frames + 2] = 7.0;

    let (encoded, time_steps) = model
        .run_encoder(&mel, FEATURE_SIZE, target_frames, 64)
        .unwrap();
    assert_eq!(time_steps, 8);
    assert_eq!(encoded.shape(), &[1, 1_024, target_frames]);
    assert_eq!(encoded[[0, 3, 2]], 7.0);
    assert_eq!(encoded[[0, 3 + FEATURE_SIZE, 2]], 7.0);
    assert_eq!(encoded[[0, 4, 2]], 0.0);

    let error = model
        .run_encoder(&mel, FEATURE_SIZE, target_frames, target_frames + 1)
        .expect_err("frames past the encoder target must be rejected");
    assert!(matches!(error, TranscriptionError::InvalidModelOutput(_)));

    let error = model
        .run_encoder(&mel[..10], FEATURE_SIZE, target_frames, 64)
        .expect_err("short mel buffers must be rejected");
    assert!(matches!(error, TranscriptionError::InvalidModelOutput(_)));
}

#[test]
fn shared_greedy_decoder_emits_tokens_from_onnx_joint_outputs() {
    let (dir, _) = write_bundle(&FixtureBundle::programmable(ParakeetVersion::V2));
    let (model, config) = fixture_model(dir.path(), ParakeetVersion::V2);
    let blank = ParakeetDeployment::v2().blank_id();

    // t0 emits 5 and advances 1; t1 blank advances 2; t3 emits 7 and advances
    // 4; t7 blank ends the 8 encoder steps produced by 64 mel frames
    let mut steps = vec![Step::blank(blank); 8];
    steps[0] = Step::token(5, 1, 0.9);
    steps[1] = Step {
        duration: 2.0,
        ..Step::blank(blank)
    };
    steps[3] = Step::token(7, 4, 0.6);
    let features = programmed_features(&steps, 64);

    let raw = model
        .transcribe(
            &features,
            64,
            config.max_feature_frames(),
            LanguageSelection::Auto,
        )
        .unwrap();
    assert_eq!(raw.token_ids, vec![5, 7]);
    assert_eq!(raw.frame_indices, vec![0, 3]);
    assert_eq!(raw.durations, vec![1, 4]);
    assert_eq!(raw.confidences, vec![0.9, 0.6]);
}

#[test]
fn v3_top_k_language_filter_runs_on_onnx_candidates() {
    let (dir, _) = write_bundle(&FixtureBundle::programmable(ParakeetVersion::V3));
    let (model, config) = fixture_model(dir.path(), ParakeetVersion::V3);
    let blank = ParakeetDeployment::v3().blank_id();

    let mut steps = vec![Step::blank(blank); 8];
    steps[0] = Step {
        top_k_ids: [V3_CYRILLIC_TOKEN as f32, 5.0],
        top_k_logits: [4.0, 2.0],
        ..Step::token(V3_CYRILLIC_TOKEN, 1, 0.9)
    };
    let features = programmed_features(&steps, 64);

    let auto = model
        .transcribe(
            &features,
            64,
            config.max_feature_frames(),
            LanguageSelection::Auto,
        )
        .unwrap();
    assert_eq!(auto.token_ids, vec![V3_CYRILLIC_TOKEN as u32]);

    let polish = model
        .transcribe(
            &features,
            64,
            config.max_feature_frames(),
            LanguageSelection::Language(ParakeetLanguage::Polish),
        )
        .unwrap();
    assert_eq!(polish.token_ids, vec![5]);
    let expected_probability = (2.0f32 - 4.0).exp() / (1.0 + (2.0f32 - 4.0).exp());
    assert!((polish.confidences[0] - expected_probability).abs() < 1e-5);
}

#[test]
fn joint_output_bounds_are_enforced_at_the_boundary() {
    let (dir, _) = write_bundle(&FixtureBundle::programmable(ParakeetVersion::V3));
    let (model, config) = fixture_model(dir.path(), ParakeetVersion::V3);
    let blank = ParakeetDeployment::v3().blank_id();
    let run = |step: Step| {
        let mut steps = vec![Step::blank(blank); 8];
        steps[0] = step;
        model.transcribe(
            &programmed_features(&steps, 64),
            64,
            config.max_feature_frames(),
            LanguageSelection::Language(ParakeetLanguage::Polish),
        )
    };

    let negative = run(Step {
        token: -1.0,
        ..Step::blank(blank)
    })
    .expect_err("negative token ids must be rejected");
    assert!(matches!(
        negative,
        TranscriptionError::InvalidModelOutput(message) if message.contains("`token_id` was negative")
    ));

    let past_blank = run(Step::token(blank + 1, 1, 0.5)).expect_err("ids past blank are invalid");
    assert!(matches!(
        past_blank,
        TranscriptionError::InvalidModelOutput(message) if message.contains("exceeded blank id")
    ));

    let duration = run(Step::token(5, 9, 0.5)).expect_err("duration bins are deployment owned");
    assert!(matches!(
        duration,
        TranscriptionError::InvalidDurationBin { bin: 9, len: 5 }
    ));

    let nan_prob = run(Step::token(5, 1, f32::NAN)).expect_err("NaN probabilities are invalid");
    assert!(matches!(
        nan_prob,
        TranscriptionError::InvalidModelOutput(message) if message.contains("`token_prob` was not finite")
    ));

    let inf_logit = run(Step {
        top_k_logits: [f32::INFINITY, 1.0],
        ..Step::token(5, 1, 0.5)
    })
    .expect_err("non-finite top-k logits are invalid");
    assert!(matches!(
        inf_logit,
        TranscriptionError::InvalidLanguageLogit { index: 0, .. }
    ));

    let negative_candidate = run(Step {
        top_k_ids: [5.0, -3.0],
        ..Step::token(5, 1, 0.5)
    })
    .expect_err("negative top-k ids are invalid");
    assert!(matches!(
        negative_candidate,
        TranscriptionError::InvalidModelOutput(message) if message.contains("`top_k_ids` was negative")
    ));
}

#[test]
fn greedy_state_resets_between_runs_on_the_same_sessions() {
    let (dir, _) = write_bundle(&FixtureBundle::programmable(ParakeetVersion::V2));
    let (model, config) = fixture_model(dir.path(), ParakeetVersion::V2);
    let blank = ParakeetDeployment::v2().blank_id();
    let mut steps = vec![Step::blank(blank); 8];
    steps[0] = Step::token(5, 1, 0.9);
    steps[2] = Step::token(9, 1, 0.4);
    let features = programmed_features(&steps, 64);

    let first = model
        .transcribe(
            &features,
            64,
            config.max_feature_frames(),
            LanguageSelection::Auto,
        )
        .unwrap();
    let second = model
        .transcribe(
            &features,
            64,
            config.max_feature_frames(),
            LanguageSelection::Auto,
        )
        .unwrap();
    assert_eq!(first.token_ids, vec![5, 9]);
    assert_eq!(first.token_ids, second.token_ids);
    assert_eq!(first.frame_indices, second.frame_indices);
    assert_eq!(first.confidences, second.confidences);
}

// ---------------------------------------------------------------------------
// Silero VAD adapter

#[cfg(feature = "vad")]
mod vad {
    use super::{FixtureBundle, ModelBundle, ParakeetVersion, open_cpu, write_bundle};
    use crate::constants::{VAD_CONTEXT_SAMPLES, VAD_WINDOW_SAMPLES};
    use crate::error::TranscriptionError;
    use crate::long_form::{LongFormConfig, LongFormMode, LongFormTranscriptionPipeline};
    use crate::onnx::error::OnnxBundleError;
    use crate::onnx::test_support::silero_fixture_probability;

    fn probabilities(root: &std::path::Path, audio: &[f32]) -> Vec<f32> {
        let bundle = open_cpu(root, ParakeetVersion::V2).unwrap();
        let model = bundle.vad_model().expect("fixture declares a VAD graph");
        let mut state = vec![0.0f32; 256];
        let mut window = vec![0.0f32; VAD_CONTEXT_SAMPLES + VAD_WINDOW_SAMPLES];
        let mut out = Vec::new();
        for chunk in audio.chunks(VAD_WINDOW_SAMPLES) {
            window[VAD_CONTEXT_SAMPLES..VAD_CONTEXT_SAMPLES + chunk.len()].copy_from_slice(chunk);
            out.push(model.run(&window, &mut state).unwrap());
        }
        out
    }

    #[test]
    fn silero_graph_moves_probability_and_state_through_the_contract() {
        let (dir, _) = write_bundle(&FixtureBundle::blank(ParakeetVersion::V2).with_vad());
        let audio: Vec<f32> = (0..VAD_WINDOW_SAMPLES * 2)
            .map(|i| (i % 7) as f32 / 10.0)
            .collect();
        let observed = probabilities(dir.path(), &audio);

        let first_window: Vec<f32> = std::iter::repeat_n(0.0, VAD_CONTEXT_SAMPLES)
            .chain(audio[..VAD_WINDOW_SAMPLES].iter().copied())
            .collect();
        let first_mean = first_window.iter().sum::<f32>() / first_window.len() as f32;
        let second_window: Vec<f32> = std::iter::repeat_n(0.0, VAD_CONTEXT_SAMPLES)
            .chain(audio[VAD_WINDOW_SAMPLES..].iter().copied())
            .collect();

        let expected = [
            silero_fixture_probability(&first_window, 0.0),
            silero_fixture_probability(&second_window, first_mean),
        ];
        for (observed, expected) in observed.iter().zip(expected) {
            assert!(
                (observed - expected).abs() < 1e-5,
                "{observed} vs {expected}"
            );
        }
    }

    #[test]
    fn vad_owner_carries_context_pads_the_tail_and_resets_state() {
        let (dir, _) = write_bundle(&FixtureBundle::blank(ParakeetVersion::V2).with_vad());
        let bundle =
            ModelBundle::from_onnx_cuda(open_cpu(dir.path(), ParakeetVersion::V2).unwrap());
        bundle.validate_long_form().unwrap();
        let vad = crate::long_form::vad::SileroVad::from_bundle(&bundle)
            .unwrap()
            .expect("fixture declares a VAD graph");
        let config = LongFormConfig::default();

        // two chunks: the second is short and padded with its last sample
        let tail_len = 88;
        let mut audio = vec![0.0f32; VAD_WINDOW_SAMPLES + tail_len];
        audio[VAD_WINDOW_SAMPLES - 1] = 0.5;
        audio[VAD_WINDOW_SAMPLES + tail_len - 1] = 0.3;

        let first_window: Vec<f32> = std::iter::repeat_n(0.0, VAD_CONTEXT_SAMPLES)
            .chain(audio[..VAD_WINDOW_SAMPLES].iter().copied())
            .collect();
        let first_mean = first_window.iter().sum::<f32>() / first_window.len() as f32;
        let mut second_window: Vec<f32> =
            audio[VAD_WINDOW_SAMPLES - VAD_CONTEXT_SAMPLES..].to_vec();
        second_window.resize(VAD_CONTEXT_SAMPLES + VAD_WINDOW_SAMPLES, 0.3);
        let expected = [
            silero_fixture_probability(&first_window, 0.0),
            silero_fixture_probability(&second_window, first_mean),
        ];

        let observed = vad.process(&audio, &config.vad).unwrap();
        assert_eq!(observed.len(), 2);
        for (observed, expected) in observed.iter().zip(expected) {
            assert!(
                (observed - expected).abs() < 1e-5,
                "{observed} vs {expected}"
            );
        }

        // the second window sees the context sample, so moving it out of the
        // context region changes only the second probability
        let mut moved = audio.clone();
        moved[VAD_WINDOW_SAMPLES - 1] = 0.0;
        moved[100] = 0.5;
        let moved_probabilities = vad.process(&moved, &config.vad).unwrap();
        assert!((moved_probabilities[0] - observed[0]).abs() < 1e-6);
        assert!((moved_probabilities[1] - observed[1]).abs() > 1e-4);

        // state resets per recording
        let again = vad.process(&audio, &config.vad).unwrap();
        assert_eq!(again, observed);
        assert!(vad.process(&[], &config.vad).unwrap().is_empty());
    }

    #[test]
    fn vad_mode_requires_a_declared_vad_graph() {
        let (dir, _) = write_bundle(&FixtureBundle::blank(ParakeetVersion::V2));
        let bundle =
            ModelBundle::from_onnx_cuda(open_cpu(dir.path(), ParakeetVersion::V2).unwrap());
        assert!(matches!(
            bundle.validate_long_form(),
            Err(TranscriptionError::OnnxBundle(
                OnnxBundleError::MissingVad { .. }
            ))
        ));

        let pipeline = LongFormTranscriptionPipeline::from_bundle(bundle).unwrap();
        let config = LongFormConfig {
            mode: LongFormMode::Vad,
            ..LongFormConfig::default()
        };
        let error = pipeline
            .run_with_config(&vec![0.0f32; 400_000], &config)
            .expect_err("VAD mode without a VAD graph must fail");
        assert!(matches!(
            error,
            TranscriptionError::OnnxBundle(OnnxBundleError::MissingVad { manifest })
                if manifest.ends_with("manifest.json")
        ));
    }

    #[test]
    fn long_form_vad_planning_runs_on_onnx_sessions() {
        let (dir, _) = write_bundle(&FixtureBundle::blank(ParakeetVersion::V2).with_vad());
        let bundle =
            ModelBundle::from_onnx_cuda(open_cpu(dir.path(), ParakeetVersion::V2).unwrap());
        let pipeline = LongFormTranscriptionPipeline::from_bundle(bundle).unwrap();
        let config = LongFormConfig {
            mode: LongFormMode::Vad,
            worker_count: 2,
            ..LongFormConfig::default()
        };

        // constant positive audio drives the fixture probability above the
        // 0.85 threshold after a few windows, so the planner sees speech and
        // the blank joint yields an empty transcript for every segment
        let audio = vec![1.0f32; 400_000];
        let result = pipeline.run_with_config(&audio, &config).unwrap();
        assert!(result.text.is_empty());
        assert_eq!(result.duration_seconds, 25.0);

        let silence = vec![0.0f32; 400_000];
        let result = pipeline.run_with_config(&silence, &config).unwrap();
        assert!(result.chunks.is_empty());
    }
}
