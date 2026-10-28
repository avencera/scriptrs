//! Batch-1 speed and transcripts of the CUDA backend on LibriSpeech test-clean, or
//! on long recordings
//!
//! Mirrors the NeMo profile's `bench.py`: audio is preloaded, utterances run longest
//! first, the longest and shortest run once as warmup, and the timed region is
//! mel, encoder, decode and detokenization for every utterance. Writes a summary
//! and `<out>.texts.json` (`{id: text}`) for `dev-tools/nemo/wer.py`
//!
//! With `--long <wav or directory>` (repeatable), the first recording runs once as
//! warmup, then each runs through the long-form pipeline with the bundle's
//! defaults; the timed region excludes reading the files, and transcripts are keyed
//! by file stem for `dev-tools/nemo/long_wer.py`. `--window` and `--overlap`
//! (seconds) override the overlap plan and `--batch` the windows per model call;
//! at batch 1 the windows run one at a time on one worker, and above it the
//! default workers prepare windows ahead of the model. `--language <code>` decodes
//! with that language's filter instead of automatic language selection, and
//! `--tf32` runs the encoder GEMMs in TF32
//!
//! ```sh
//! cargo run --release --features cuda,long-form --example cuda_bench -- \
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
use scriptrs::{
    LanguageSelection, LongFormConfig, LongFormTranscriptionPipeline, ModelBundle,
    ParakeetDeployment, ParakeetLanguage, Precision, TranscriptionConfig, TranscriptionPipeline,
};
use serde_json::{Value, json};

const DEFAULT_MANIFEST: &str = "/shared/datasets/librispeech-test-clean/manifest.jsonl";

struct Args {
    models: PathBuf,
    deployment: ParakeetDeployment,
    manifest: PathBuf,
    out: PathBuf,
    repeats: usize,
    long: Vec<PathBuf>,
    window_s: Option<f64>,
    overlap_s: Option<f64>,
    batch: Option<usize>,
    language: LanguageSelection,
    precision: Precision,
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
    let bundle = ModelBundle::from_native_dir(&args.models, args.deployment.clone())
        .with_precision(args.precision);
    let (audio_s, walls, texts, batch) = if args.long.is_empty() {
        bench_manifest(bundle, &args.manifest, args.repeats, args.language)?
    } else {
        bench_long(bundle, &args.long, &args)?
    };

    let mut sorted = walls.clone();
    sorted.sort_by(f64::total_cmp);
    let summary = json!({
        "backend": match args.precision {
            Precision::Fp32 => "scriptrs-cuda-fp32",
            Precision::Tf32 => "scriptrs-cuda-tf32",
        },
        "version": version,
        "bs": batch,
        "long": !args.long.is_empty(),
        "n_utts": texts.len(),
        "audio_s": audio_s,
        "walls_s": walls,
        "rtfx_best": audio_s / sorted[0],
        "rtfx_median": audio_s / median(&sorted),
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

/// Median of sorted values, averaging the middle two like `np.median`
fn median(sorted: &[f64]) -> f64 {
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}

/// Audio seconds, wall seconds per timed pass, transcripts by id, and windows per
/// model call
type Bench = (f64, Vec<f64>, BTreeMap<String, String>, usize);

fn bench_manifest(
    bundle: ModelBundle,
    manifest: &Path,
    repeats: usize,
    language: LanguageSelection,
) -> Result<Bench> {
    let pipeline = TranscriptionPipeline::from_bundle(bundle)?;
    let config = TranscriptionConfig {
        language,
        ..pipeline.config().clone()
    };
    let mut utterances = load_manifest(manifest)?;
    utterances.sort_by(|a, b| b.audio.len().cmp(&a.audio.len()).then(a.id.cmp(&b.id)));
    let audio_s: f64 = utterances
        .iter()
        .map(|utterance| utterance.audio.len() as f64 / 16_000.0)
        .sum();
    let (Some(longest), Some(shortest)) = (utterances.first(), utterances.last()) else {
        bail!("the manifest is empty");
    };

    for utterance in [longest, shortest] {
        pipeline.run_with_config(&utterance.audio, &config)?;
    }

    let mut walls = Vec::with_capacity(repeats);
    let mut texts = BTreeMap::new();
    for _ in 0..repeats {
        texts.clear();
        let started = Instant::now();
        for utterance in &utterances {
            let text = pipeline.run_with_config(&utterance.audio, &config)?.text;
            texts.insert(utterance.id.clone(), text);
        }
        walls.push(started.elapsed().as_secs_f64());
    }
    Ok((audio_s, walls, texts, 1))
}

fn bench_long(bundle: ModelBundle, paths: &[PathBuf], args: &Args) -> Result<Bench> {
    let mut config = LongFormConfig::for_bundle(&bundle);
    config.transcription.language = args.language;
    if let Some(batch) = args.batch {
        config.batch_size = batch;
    }
    if config.batch_size == 1 {
        // each worker of the unbatched plan loads its own model
        config.worker_count = 1;
    }
    if let Some(window_s) = args.window_s {
        config.overlap.max_model_samples = (window_s * 16_000.0) as usize;
    }
    if let Some(overlap_s) = args.overlap_s {
        config.overlap.overlap_seconds = overlap_s;
    }
    let pipeline = LongFormTranscriptionPipeline::from_bundle(bundle)?;
    let recordings = long_recordings(paths)?;
    let Some(first) = recordings.first() else {
        bail!("--long found no wav files");
    };
    pipeline.run_with_config(&read_wav(first)?, &config)?;

    // files are read outside the timed region, one at a time, to keep memory flat
    let mut walls = vec![0.0; args.repeats];
    let mut texts = BTreeMap::new();
    let mut audio_s = 0.0;
    for path in &recordings {
        let audio = read_wav(path)?;
        audio_s += audio.len() as f64 / 16_000.0;
        for wall in &mut walls {
            let started = Instant::now();
            let text = pipeline.run_with_config(&audio, &config)?.text;
            *wall += started.elapsed().as_secs_f64();
            texts.insert(stem(path)?, text);
        }
    }
    Ok((audio_s, walls, texts, config.batch_size))
}

/// The wav files named by `--long`, with directories expanded to the wavs in them
fn long_recordings(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut recordings = Vec::new();
    for path in paths {
        if !path.is_dir() {
            recordings.push(path.clone());
            continue;
        }

        for entry in fs::read_dir(path)? {
            let entry = entry?.path();
            if entry
                .extension()
                .is_some_and(|extension| extension == "wav")
            {
                recordings.push(entry);
            }
        }
    }
    recordings.sort();
    Ok(recordings)
}

fn stem(path: &Path) -> Result<String> {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .map(str::to_owned)
        .ok_or_else(|| eyre!("{} has no file stem", path.display()))
}

fn parse_args() -> Result<Args> {
    let mut models = None;
    let mut deployment = ParakeetDeployment::v3();
    let mut manifest = PathBuf::from(DEFAULT_MANIFEST);
    let mut out = None;
    let mut repeats = 3;
    let mut long = Vec::new();
    let mut window_s = None;
    let mut overlap_s = None;
    let mut batch = None;
    let mut language = LanguageSelection::Auto;
    let mut precision = Precision::Fp32;
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
            "--long" => long.push(PathBuf::from(value()?)),
            "--window" => window_s = Some(value()?.parse()?),
            "--overlap" => overlap_s = Some(value()?.parse()?),
            "--batch" => batch = Some(value()?.parse()?),
            "--language" => {
                let code = value()?;
                let parsed = ParakeetLanguage::from_code(&code)
                    .ok_or_else(|| eyre!("unknown language {code}"))?;
                language = LanguageSelection::Language(parsed);
            }
            "--tf32" => precision = Precision::Tf32,
            other => return Err(eyre!("unknown argument {other}")),
        }
    }

    Ok(Args {
        models: models.ok_or_else(|| eyre!("--models is required"))?,
        deployment,
        manifest,
        out: out.ok_or_else(|| eyre!("--out is required"))?,
        repeats: repeats.max(1),
        long,
        window_s,
        overlap_s,
        batch,
        language,
        precision,
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
