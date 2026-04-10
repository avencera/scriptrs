use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use eyre::{Result, bail, eyre};
use hound::WavReader;

#[cfg(feature = "long-form")]
use scriptrs::LongFormTranscriptionPipeline;
use scriptrs::TranscriptionPipeline;

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
    let args = Args::parse(env::args().skip(1))?;
    let audio = read_mono_16khz_wav(&args.audio_path)?;
    let pipeline_started_at = Instant::now();

    #[cfg(feature = "long-form")]
    let result = if args.long_form {
        let pipeline = build_long_form_pipeline(&args)?;
        let pipeline_elapsed = pipeline_started_at.elapsed().as_secs_f64();
        run_long_form_pipeline(&pipeline, &audio, &args, pipeline_elapsed)?
    } else {
        let pipeline = build_pipeline(&args)?;
        let pipeline_elapsed = pipeline_started_at.elapsed().as_secs_f64();
        run_pipeline(&pipeline, &audio, &args, pipeline_elapsed)?
    };

    #[cfg(not(feature = "long-form"))]
    let result = {
        if args.long_form {
            bail!("rebuild with --features long-form to use --long-form")
        }
        let pipeline = build_pipeline(&args)?;
        let pipeline_elapsed = pipeline_started_at.elapsed().as_secs_f64();
        run_pipeline(&pipeline, &audio, &args, pipeline_elapsed)?
    };

    println!("file: {}", args.audio_path.display());
    println!("audio_seconds: {:.1}", audio.len() as f64 / 16_000.0);
    println!("chunks: {}", result.chunks.len());
    println!("tokens: {}", result.tokens.len());
    println!("{}", preview(&result.text, args.preview_chars));
    Ok(())
}

#[derive(Debug, Clone)]
struct Args {
    audio_path: PathBuf,
    models_dir: Option<PathBuf>,
    pretrained: bool,
    long_form: bool,
    warmup_runs: usize,
    benchmark_runs: usize,
    preview_chars: usize,
}

impl Args {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self> {
        let mut args = args.into_iter();
        let mut audio_path = None;
        let mut models_dir = None;
        let mut pretrained = false;
        let mut long_form = false;
        let mut warmup_runs = 1usize;
        let mut benchmark_runs = 0usize;
        let mut preview_chars = 800usize;

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--audio" => audio_path = Some(next_path(&mut args, "--audio")?),
                "--models-dir" => models_dir = Some(next_path(&mut args, "--models-dir")?),
                "--pretrained" => pretrained = true,
                "--long-form" => long_form = true,
                "--warmup-runs" => {
                    warmup_runs = next_value(&mut args, "--warmup-runs")?
                        .parse()
                        .map_err(|error| eyre!("invalid --warmup-runs value: {error}"))?;
                }
                "--benchmark-runs" => {
                    benchmark_runs = next_value(&mut args, "--benchmark-runs")?
                        .parse()
                        .map_err(|error| eyre!("invalid --benchmark-runs value: {error}"))?;
                }
                "--preview-chars" => {
                    preview_chars = next_value(&mut args, "--preview-chars")?
                        .parse()
                        .map_err(|error| eyre!("invalid --preview-chars value: {error}"))?;
                }
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                flag if flag.starts_with('-') => bail!("unknown flag: {flag}"),
                path => {
                    if audio_path.is_some() {
                        bail!("unexpected positional argument: {path}")
                    }
                    audio_path = Some(PathBuf::from(path));
                }
            }
        }

        let Some(audio_path) = audio_path else {
            bail!("missing --audio <path.wav>")
        };
        if pretrained && models_dir.is_some() {
            bail!("use either --pretrained or --models-dir, not both")
        }
        if benchmark_runs > 0 && warmup_runs == 0 {
            bail!("--warmup-runs must be at least 1 when benchmarking")
        }

        Ok(Self {
            audio_path,
            models_dir,
            pretrained,
            long_form,
            warmup_runs,
            benchmark_runs,
            preview_chars,
        })
    }
}

fn print_usage() {
    eprintln!(
        "Usage:
  cargo run --example transcribe_wav -- --audio <path.wav>
  cargo run --example transcribe_wav -- --audio <path.wav> --pretrained
  cargo run --example transcribe_wav --features long-form -- --audio <path.wav> --pretrained --long-form

Options:
  --models-dir <dir>         local scriptrs model bundle directory
  --pretrained               download models via the online feature
  --long-form                use LongFormTranscriptionPipeline
  --warmup-runs <n>          warmup runs before timing (default: 1)
  --benchmark-runs <n>       timed runs after warmup on one loaded pipeline
  --preview-chars <n>        text preview limit"
    );
}

fn run_pipeline(
    pipeline: &TranscriptionPipeline,
    audio: &[f32],
    args: &Args,
    pipeline_elapsed: f64,
) -> Result<scriptrs::TranscriptionResult> {
    if args.benchmark_runs == 0 {
        let started_at = Instant::now();
        let result = pipeline.run(audio)?;
        println!("pipeline_load_seconds: {:.2}", pipeline_elapsed);
        println!("elapsed_seconds: {:.2}", started_at.elapsed().as_secs_f64());
        return Ok(result);
    }

    for _ in 0..args.warmup_runs {
        let _ = pipeline.run(audio)?;
    }

    let mut total_seconds = 0.0;
    let mut result = None;
    for _ in 0..args.benchmark_runs {
        let started_at = Instant::now();
        let run_result = pipeline.run(audio)?;
        total_seconds += started_at.elapsed().as_secs_f64();
        result = Some(run_result);
    }

    let result = result.expect("benchmark_runs should be positive");
    println!("pipeline_load_seconds: {:.2}", pipeline_elapsed);
    println!("warmup_runs: {}", args.warmup_runs);
    println!("benchmark_runs: {}", args.benchmark_runs);
    println!("elapsed_seconds: {:.2}", total_seconds);
    println!(
        "mean_elapsed_seconds: {:.2}",
        total_seconds / args.benchmark_runs as f64
    );
    Ok(result)
}

#[cfg(feature = "long-form")]
fn run_long_form_pipeline(
    pipeline: &LongFormTranscriptionPipeline,
    audio: &[f32],
    args: &Args,
    pipeline_elapsed: f64,
) -> Result<scriptrs::TranscriptionResult> {
    if args.benchmark_runs == 0 {
        let started_at = Instant::now();
        let result = pipeline.run(audio)?;
        println!("pipeline_load_seconds: {:.2}", pipeline_elapsed);
        println!("elapsed_seconds: {:.2}", started_at.elapsed().as_secs_f64());
        return Ok(result);
    }

    for _ in 0..args.warmup_runs {
        let _ = pipeline.run(audio)?;
    }

    let mut total_seconds = 0.0;
    let mut result = None;
    for _ in 0..args.benchmark_runs {
        let started_at = Instant::now();
        let run_result = pipeline.run(audio)?;
        total_seconds += started_at.elapsed().as_secs_f64();
        result = Some(run_result);
    }

    let result = result.expect("benchmark_runs should be positive");
    println!("pipeline_load_seconds: {:.2}", pipeline_elapsed);
    println!("warmup_runs: {}", args.warmup_runs);
    println!("benchmark_runs: {}", args.benchmark_runs);
    println!("elapsed_seconds: {:.2}", total_seconds);
    println!(
        "mean_elapsed_seconds: {:.2}",
        total_seconds / args.benchmark_runs as f64
    );
    Ok(result)
}

fn build_pipeline(args: &Args) -> Result<TranscriptionPipeline> {
    if let Some(models_dir) = &args.models_dir {
        return Ok(TranscriptionPipeline::from_dir(models_dir)?);
    }

    #[cfg(feature = "online")]
    {
        let _ = args.pretrained;
        Ok(TranscriptionPipeline::from_pretrained()?)
    }

    #[cfg(not(feature = "online"))]
    {
        let _ = args.pretrained;
        bail!("rebuild with the default online feature or pass --models-dir")
    }
}

#[cfg(feature = "long-form")]
fn build_long_form_pipeline(args: &Args) -> Result<LongFormTranscriptionPipeline> {
    if let Some(models_dir) = &args.models_dir {
        return Ok(LongFormTranscriptionPipeline::from_dir(models_dir)?);
    }

    #[cfg(feature = "online")]
    {
        let _ = args.pretrained;
        Ok(LongFormTranscriptionPipeline::from_pretrained()?)
    }

    #[cfg(not(feature = "online"))]
    {
        let _ = args.pretrained;
        bail!("rebuild with the default online feature or pass --models-dir")
    }
}

fn next_path(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<PathBuf> {
    Ok(PathBuf::from(next_value(args, flag)?))
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    args.next().ok_or_else(|| eyre!("missing value for {flag}"))
}

fn read_mono_16khz_wav(path: &Path) -> Result<Vec<f32>> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    if spec.sample_rate != 16_000 {
        bail!(
            "expected 16kHz audio, got {} Hz in {}",
            spec.sample_rate,
            path.display()
        )
    }
    if spec.channels != 1 {
        bail!(
            "expected mono audio, got {} channels in {}",
            spec.channels,
            path.display()
        )
    }

    match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .map_err(Into::into),
        (hound::SampleFormat::Int, bits) if (1..=32).contains(&bits) => {
            let scale = ((1_i64 << (bits - 1)) - 1) as f32;
            let samples = reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 / scale))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(samples)
        }
        _ => bail!(
            "unsupported WAV format: {:?} {}-bit",
            spec.sample_format,
            spec.bits_per_sample
        ),
    }
}

fn preview(text: &str, limit: usize) -> String {
    if text.len() <= limit {
        return text.to_owned();
    }
    format!("{}...", &text[..limit])
}
