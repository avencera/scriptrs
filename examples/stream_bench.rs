//! Per-chunk latency and throughput of the CUDA streaming engine on LibriSpeech
//! test-clean, with transcripts for `dev-tools/nemo/wer.py`
//!
//! Audio is preloaded and fed one chunk of samples at a time, as a live source
//! delivers it. With `--streams 1` (the default) the utterances run one after
//! another; with `--streams N`, N streams stay open at once, each starting the
//! next utterance when its last ends, and every round pushes one chunk to each and
//! processes them together. The timed region of a round is the mel of the pushed
//! audio, the encoder, the decode and the readback; a round's time is the latency
//! of each chunk in it. The first `N` utterances run once as warmup, which also
//! captures the graphs. `--language <tag>` opens the streams in that language, which
//! a multilingual model needs, and `--fp32` or `--tf32` runs the encoder GEMMs
//! in that precision instead of FP16. Writes a summary and `<out>.texts.json`
//! (`{id: text}`)
//!
//! ```sh
//! cargo run --release --features cuda --example stream_bench -- \
//!     --model /shared/models/scriptrs-native/nemotron-speech-streaming-en-0.6b \
//!     --lookahead 1 --streams 1 --out bench/stream-r1-s1.json
//! ```

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use eyre::{Result, bail, eyre};
use hound::WavReader;
use scriptrs::{Precision, Stream, StreamingConfig, StreamingModel};
use serde_json::{Value, json};

const DEFAULT_MANIFEST: &str = "/shared/datasets/librispeech-test-clean/manifest.jsonl";
const SAMPLE_RATE: f64 = 16_000.0;

struct Args {
    model: PathBuf,
    lookahead: usize,
    streams: usize,
    manifest: PathBuf,
    limit: Option<usize>,
    out: PathBuf,
    precision: Precision,
    language: Option<String>,
}

struct Utterance {
    id: String,
    audio: Vec<f32>,
}

/// An open stream and the utterance it is on
struct Lane {
    stream: Stream,
    utterance: usize,
    cursor: usize,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error:?}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let args = parse_args()?;
    let mut utterances = load_manifest(&args.manifest)?;
    if let Some(limit) = args.limit {
        utterances.truncate(limit);
    }
    if utterances.is_empty() {
        bail!("no utterances in {}", args.manifest.display());
    }

    let config = StreamingConfig {
        max_streams: args.streams,
        max_batch: args.streams,
        precision: args.precision,
        ..StreamingConfig::new(args.lookahead)
    };
    let model = StreamingModel::from_native_dir(&args.model, &config)?;
    let warmup = args.streams.min(utterances.len());
    let language = args.language.as_deref();
    transcribe(&model, &utterances[..warmup], args.streams, language)?;
    let run = transcribe(&model, &utterances, args.streams, language)?;

    let audio_seconds: f64 = utterances
        .iter()
        .map(|utterance| utterance.audio.len() as f64 / SAMPLE_RATE)
        .sum();
    let total = run.total.as_secs_f64();
    let mut latencies: Vec<f64> = run
        .rounds
        .iter()
        .map(|round| round.as_secs_f64() * 1e3)
        .collect();
    latencies.sort_by(f64::total_cmp);
    let percentile = |p: f64| latencies[((latencies.len() - 1) as f64 * p).round() as usize];
    let chunk_ms = model.chunk_samples() as f64 / SAMPLE_RATE * 1e3;
    let summary = json!({
        "model": args.model,
        "lookahead": args.lookahead,
        "chunk_ms": chunk_ms,
        "streams": args.streams,
        "precision": format!("{:?}", args.precision),
        "language": args.language,
        "utterances": utterances.len(),
        "audio_seconds": audio_seconds,
        "seconds": total,
        "rtfx": audio_seconds / total,
        "chunks": run.chunks,
        "rounds": latencies.len(),
        "round_ms": {
            "mean": latencies.iter().sum::<f64>() / latencies.len() as f64,
            "p50": percentile(0.5),
            "p90": percentile(0.9),
            "p99": percentile(0.99),
            "max": percentile(1.0),
        },
    });
    println!("{}", serde_json::to_string_pretty(&summary)?);

    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.out, serde_json::to_string_pretty(&summary)?)?;
    let texts: BTreeMap<&str, &str> = utterances
        .iter()
        .zip(&run.texts)
        .map(|(utterance, text)| (utterance.id.as_str(), text.as_str()))
        .collect();
    fs::write(
        args.out.with_extension("texts.json"),
        serde_json::to_string_pretty(&texts)?,
    )?;
    Ok(())
}

struct Run {
    texts: Vec<String>,
    /// Wall time of each round that processed at least one chunk
    rounds: Vec<Duration>,
    chunks: usize,
    total: Duration,
}

/// Streams `utterances` through `streams` lanes, one chunk of samples per lane per
/// round
fn transcribe(
    model: &StreamingModel,
    utterances: &[Utterance],
    streams: usize,
    language: Option<&str>,
) -> Result<Run> {
    let step = model.chunk_samples();
    let mut texts = vec![String::new(); utterances.len()];
    let mut rounds = Vec::new();
    let mut chunks = 0;
    let mut next = 0;
    let mut lanes: Vec<Lane> = Vec::with_capacity(streams);
    let started = Instant::now();
    loop {
        while lanes.len() < streams && next < utterances.len() {
            lanes.push(Lane {
                stream: match language {
                    Some(language) => model.open_stream_in(language)?,
                    None => model.open_stream()?,
                },
                utterance: next,
                cursor: 0,
            });
            next += 1;
        }
        if lanes.is_empty() {
            break;
        }

        let round = Instant::now();
        for lane in &mut lanes {
            let audio = &utterances[lane.utterance].audio;
            let end = (lane.cursor + step).min(audio.len());
            lane.stream.push(&audio[lane.cursor..end])?;
            lane.cursor = end;
            if end == audio.len() {
                lane.stream.finish()?;
            }
        }
        let mut active: Vec<&mut Stream> = lanes.iter_mut().map(|lane| &mut lane.stream).collect();
        let processed = model.process(&mut active)?;
        if processed > 0 {
            rounds.push(round.elapsed());
            chunks += processed;
        }

        lanes.retain(|lane| {
            if !lane.stream.is_done() {
                return true;
            }
            texts[lane.utterance] = lane.stream.transcript().text;
            false
        });
    }

    Ok(Run {
        texts,
        rounds,
        chunks,
        total: started.elapsed(),
    })
}

fn parse_args() -> Result<Args> {
    let mut model = None;
    let mut lookahead = None;
    let mut streams = 1;
    let mut manifest = PathBuf::from(DEFAULT_MANIFEST);
    let mut limit = None;
    let mut out = None;
    let mut precision = Precision::default();
    let mut language = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| eyre!("{flag} needs a value"));
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(value()?)),
            "--lookahead" => lookahead = Some(value()?.parse()?),
            "--streams" => streams = value()?.parse()?,
            "--manifest" => manifest = PathBuf::from(value()?),
            "--limit" => limit = Some(value()?.parse()?),
            "--out" => out = Some(PathBuf::from(value()?)),
            "--fp32" => precision = Precision::Fp32,
            "--tf32" => precision = Precision::Tf32,
            "--language" => language = Some(value()?),
            _ => return Err(eyre!("unknown flag {flag}")),
        }
    }
    if streams == 0 {
        bail!("--streams must be at least 1");
    }

    Ok(Args {
        model: model.ok_or_else(|| eyre!("--model is required"))?,
        lookahead: lookahead.ok_or_else(|| eyre!("--lookahead is required"))?,
        streams,
        manifest,
        limit,
        out: out.ok_or_else(|| eyre!("--out is required"))?,
        precision,
        language,
    })
}

fn load_manifest(path: &Path) -> Result<Vec<Utterance>> {
    fs::read_to_string(path)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let row: Value = serde_json::from_str(line)?;
            let field = |key: &str| {
                row[key]
                    .as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| eyre!("manifest row has no {key}"))
            };
            Ok(Utterance {
                id: field("id")?,
                audio: read_wav(Path::new(&field("audio_filepath")?))?,
            })
        })
        .collect()
}

/// 16 kHz mono 16-bit PCM, scaled by 2^15 as soundfile (and so NeMo) reads it
fn read_wav(path: &Path) -> Result<Vec<f32>> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    if spec.sample_rate != 16_000 || spec.channels != 1 || spec.bits_per_sample != 16 {
        bail!("{} is not 16 kHz mono 16-bit PCM", path.display());
    }

    reader
        .samples::<i16>()
        .map(|sample| Ok(f32::from(sample?) / 32_768.0))
        .collect()
}
