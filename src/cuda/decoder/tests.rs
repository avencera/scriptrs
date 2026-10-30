use cudakit::cudarc::driver::CudaSlice;
use cudakit::{Library, Math, Options};
use ndarray::Ix3;

use super::{CudaDecoder, DecodeWorkspace, STEPS_PER_CHECK};
use crate::cuda::encoder::{CudaEncoder, EncoderMath};
use crate::cuda::kernels::CudaRuntime;
use crate::cuda::test_support::weights;
use crate::decode::RawTranscription;
use crate::deployment::ParakeetVersion;
use crate::test_support::{NemoReference, deployment};

mod filter;

const SETS: [&str; 3] = ["b1", "b8", "long-full"];

fn runtime() -> CudaRuntime {
    let runtime = CudaRuntime::open(0, Options::default()).expect("CUDA runtime");
    runtime.prepare(Library::Cublas).expect("cuBLAS");
    runtime
}

/// NeMo's encoder output as `[batch * frames, d_model]` on the device, with its
/// valid lengths
fn nemo_encoder_out(
    runtime: &CudaRuntime,
    reference: &NemoReference,
) -> (CudaSlice<f32>, usize, usize, Vec<usize>) {
    let out = reference
        .f32("encoder.out")
        .into_dimensionality::<Ix3>()
        .expect("[batch, d_model, frames]");
    let (batch, frames) = (out.shape()[0], out.shape()[2]);
    let time_major: Vec<f32> = out.permuted_axes([0, 2, 1]).iter().copied().collect();
    let lens = reference
        .i64("encoder.out_len")
        .iter()
        .map(|len| *len as usize)
        .collect();
    let device = runtime.stream().clone_htod(&time_major).expect("upload");
    (device, batch, frames, lens)
}

/// Holds each row against NeMo's batched greedy decode: tokens, frames and
/// durations exactly
fn assert_matches_nemo(
    name: &str,
    reference: &NemoReference,
    decoded: &[RawTranscription],
) -> usize {
    let mut tokens = 0;
    for (row, raw) in decoded.iter().enumerate() {
        let as_u32 = |key: &str| -> Vec<u32> {
            reference
                .i32(&format!("utt{row}.hyp.{key}"))
                .iter()
                .map(|value| *value as u32)
                .collect()
        };
        let frames: Vec<u32> = raw.frames().iter().map(|f| *f as u32).collect();
        let durations: Vec<u32> = raw.durations().iter().map(|d| *d as u32).collect();
        assert_eq!(raw.ids(), as_u32("tokens"), "{name} row {row} tokens");
        assert_eq!(frames, as_u32("timestamps"), "{name} row {row} frames");
        assert_eq!(durations, as_u32("durations"), "{name} row {row} durations");
        assert!(
            raw.tokens
                .iter()
                .all(|token| token.confidence > 0.0 && token.confidence <= 1.0),
            "{name} row {row} confidences"
        );
        tokens += raw.tokens.len();
    }
    tokens
}

fn decode_eager(
    runtime: &CudaRuntime,
    decoder: &CudaDecoder,
    ws: &mut DecodeWorkspace,
    encoder_out: &CudaSlice<f32>,
    lens: &[usize],
) -> Vec<RawTranscription> {
    decoder
        .start(runtime, ws, encoder_out, lens)
        .expect("start");
    decoder.run(runtime, ws, STEPS_PER_CHECK).expect("run");
    decoder.output(runtime, ws).expect("output")
}

/// Decodes NeMo's own encoder output, then this crate's encoder output from NeMo's
/// mel, and holds both against NeMo's decode
#[test]
#[ignore = "needs a CUDA GPU and the NeMo references; `just gpu-parity` runs it on gpu0"]
fn gpu_decoder_matches_nemo_hypotheses() {
    let runtime = runtime();
    for version in [ParakeetVersion::V2, ParakeetVersion::V3] {
        let file = weights(version);
        let spec = file.parakeet_spec(&deployment(version)).expect("spec");
        let decoder = CudaDecoder::load(&runtime, &file, &spec.transducer).expect("decoder");
        let encoder = CudaEncoder::load(
            &runtime,
            &file,
            spec.encoder,
            EncoderMath::Single(Math::Fp32),
        )
        .expect("encoder");

        for set in SETS {
            let reference = NemoReference::open(version, set);
            let (encoder_out, batch, frames, lens) = nemo_encoder_out(&runtime, &reference);
            let mut ws = decoder
                .workspace(&runtime, batch, frames)
                .expect("workspace");
            let decoded = decode_eager(&runtime, &decoder, &mut ws, &encoder_out, &lens);
            let tokens = assert_matches_nemo(
                &format!("{version:?} {set} NeMo encoder"),
                &reference,
                &decoded,
            );

            let mel = reference
                .f32("mel")
                .into_dimensionality::<Ix3>()
                .expect("[batch, features, frames]");
            let mel_frames = mel.shape()[2];
            let mel_lens: Vec<usize> = reference
                .i64("mel_len")
                .iter()
                .map(|len| *len as usize)
                .collect();
            let time_major: Vec<f32> = mel.permuted_axes([0, 2, 1]).iter().copied().collect();
            let mut encoder_ws = encoder
                .workspace(&runtime, batch, mel_frames)
                .expect("encoder workspace");
            encoder
                .set_input(&runtime, &mut encoder_ws, &mel_lens, |mel| {
                    mel.copy_from_slice(&time_major)
                })
                .expect("input");
            encoder.enqueue(&runtime, &mut encoder_ws).expect("encoder");
            assert_eq!(encoder_ws.frames(), frames, "encoder frames");
            let decoded = decode_eager(
                &runtime,
                &decoder,
                &mut ws,
                encoder_ws.output(),
                encoder_ws.output_lens(),
            );
            assert_matches_nemo(
                &format!("{version:?} {set} CUDA encoder"),
                &reference,
                &decoded,
            );
            println!("{version:?} {set}: {tokens} tokens match NeMo from both encoders");
        }
    }
}

/// Replays a captured graph of decode steps, checks it against the eager decode
/// and prints the decode time per set
#[test]
#[ignore = "needs a CUDA GPU and the NeMo references; `just gpu-parity` runs it on gpu0"]
fn gpu_decoder_graph_replays_the_eager_decode() {
    const REPEATS: u32 = 20;
    let runtime = runtime();
    let version = ParakeetVersion::V3;
    let file = weights(version);
    let spec = file.parakeet_spec(&deployment(version)).expect("spec");
    let decoder = CudaDecoder::load(&runtime, &file, &spec.transducer).expect("decoder");

    for set in SETS {
        let reference = NemoReference::open(version, set);
        let (encoder_out, batch, frames, lens) = nemo_encoder_out(&runtime, &reference);
        let mut ws = decoder
            .workspace(&runtime, batch, frames)
            .expect("workspace");
        // the eager decode also finishes cuBLAS's lazy setup before capture
        let eager = decode_eager(&runtime, &decoder, &mut ws, &encoder_out, &lens);

        decoder
            .start(&runtime, &mut ws, &encoder_out, &lens)
            .expect("start");
        // SAFETY: the graph is declared after `ws` and `decoder`, so it drops
        // before the buffers it points at, which stay in place on the runtime's
        // stream; the closure returns errors instead of panicking
        let graph = unsafe {
            runtime.capture_nonempty(|| {
                (0..STEPS_PER_CHECK).try_for_each(|_| decoder.step(&runtime, &mut ws))
            })
        }
        .expect("capture");

        let decode = |ws: &mut DecodeWorkspace| -> Vec<RawTranscription> {
            decoder
                .start(&runtime, ws, &encoder_out, &lens)
                .expect("start");
            decoder
                .run_with(&runtime, ws, STEPS_PER_CHECK, |_| Ok(graph.launch()?))
                .expect("run");
            decoder.output(&runtime, ws).expect("output")
        };
        let replayed = decode(&mut ws);
        for (row, (a, b)) in eager.iter().zip(&replayed).enumerate() {
            assert_eq!(a.ids(), b.ids(), "{set} row {row} tokens");
            assert_eq!(a.frames(), b.frames(), "{set} row {row} frames");
            assert_eq!(
                a.confidences(),
                b.confidences(),
                "{set} row {row} confidences"
            );
        }

        let start = std::time::Instant::now();
        for _ in 0..REPEATS {
            decode(&mut ws);
        }
        let ms = start.elapsed().as_secs_f64() * 1e3 / f64::from(REPEATS);
        let tokens: usize = replayed.iter().map(|raw| raw.tokens.len()).sum();
        let longest = lens.iter().copied().max().unwrap_or(0);
        println!(
            "v3 {set}: batch {batch}, {longest} frames, {tokens} tokens, graph decode {ms:.3} ms"
        );
    }
}
