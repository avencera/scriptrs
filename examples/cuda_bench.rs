//! Batch-1 speed and transcripts of the CUDA backend on LibriSpeech test-clean
//!
//! Mirrors the NeMo profile's `bench.py`: audio is preloaded, utterances run longest
//! first, the longest and shortest run once as warmup, and the timed region is
//! mel, encoder, decode and detokenization for every utterance. Writes a summary
//! and `<out>.texts.json` (`{id: text}`) for `dev-tools/nemo/wer.py`
//!
//! ```sh
//! cargo run --release --features cuda --example cuda_bench -- \
//!     --models /shared/models/scriptrs-native --version v3 --out bench/v3-bs1.json
//! ```

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use eyre::{Result, bail, eyre};
use hound::WavReader;
use scriptrs::{ModelBundle, ParakeetDeployment, TranscriptionPipeline};
use serde_json::{Value, json};

const DEFAULT_MANIFEST: &str = "/shared/datasets/librispeech-test-clean/manifest.jsonl";

struct Args {
    models: PathBuf,
    deployment: ParakeetDeployment,
    manifest: PathBuf,
    out: PathBuf,
    repeats: usize,
}

struct Utterance {
    id: String,
    audio: Vec<f32>,
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
    let version = format!("{:?}", args.deployment.version());
    let pipeline = TranscriptionPipeline::from_bundle(ModelBundle::from_native_dir(
        &args.models,
        args.deployment,
    ))?;

    let mut utterances = load_manifest(&args.manifest)?;
    utterances.sort_by(|a, b| b.audio.len().cmp(&a.audio.len()).then(a.id.cmp(&b.id)));
    let audio_s: f64 = utterances
        .iter()
        .map(|utterance| utterance.audio.len() as f64 / 16_000.0)
        .sum();
    let (Some(longest), Some(shortest)) = (utterances.first(), utterances.last()) else {
        bail!("the manifest is empty");
    };

    for utterance in [longest, shortest] {
        pipeline.run(&utterance.audio)?;
    }

    let mut walls = Vec::with_capacity(args.repeats);
    let mut texts = BTreeMap::new();
    for _ in 0..args.repeats {
        texts.clear();
        let started = Instant::now();
        for utterance in &utterances {
            let text = pipeline.run(&utterance.audio)?.text;
            texts.insert(utterance.id.clone(), text);
        }
        walls.push(started.elapsed().as_secs_f64());
    }

    let mut sorted = walls.clone();
    sorted.sort_by(f64::total_cmp);
    let summary = json!({
        "backend": "scriptrs-cuda-fp32",
        "version": version,
        "bs": 1,
        "n_utts": utterances.len(),
        "audio_s": audio_s,
        "walls_s": walls,
        "rtfx_best": audio_s / sorted[0],
        "rtfx_median": audio_s / sorted[sorted.len() / 2],
    });
    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.out, serde_json::to_string_pretty(&summary)?)?;
    fs::write(
        args.out.with_extension("texts.json"),
        serde_json::to_string_pretty(&texts)?,
    )?;
    println!("{summary}");
    Ok(())
}

fn parse_args() -> Result<Args> {
    let mut models = None;
    let mut deployment = ParakeetDeployment::v3();
    let mut manifest = PathBuf::from(DEFAULT_MANIFEST);
    let mut out = None;
    let mut repeats = 3;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| eyre!("{flag} needs a value"));
        match flag.as_str() {
            "--models" => models = Some(PathBuf::from(value()?)),
            "--version" => {
                deployment = match value()?.as_str() {
                    "v2" => ParakeetDeployment::v2(),
                    "v3" => ParakeetDeployment::v3(),
                    other => return Err(eyre!("unknown version {other}")),
                }
            }
            "--manifest" => manifest = PathBuf::from(value()?),
            "--out" => out = Some(PathBuf::from(value()?)),
            "--repeats" => repeats = value()?.parse()?,
            other => return Err(eyre!("unknown argument {other}")),
        }
    }

    Ok(Args {
        models: models.ok_or_else(|| eyre!("--models is required"))?,
        deployment,
        manifest,
        out: out.ok_or_else(|| eyre!("--out is required"))?,
        repeats: repeats.max(1),
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

/// 16 kHz mono 16-bit PCM, scaled by 2^15 as soundfile (and so NeMo's bench) reads it
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
