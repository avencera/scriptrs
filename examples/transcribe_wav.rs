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
    let started_at = Instant::now();

    #[cfg(feature = "long-form")]
    let result = if args.long_form {
        LongFormTranscriptionPipeline::from_dir(&args.models_dir)?.run(&audio)?
    } else {
        TranscriptionPipeline::from_dir(&args.models_dir)?.run(&audio)?
    };

    #[cfg(not(feature = "long-form"))]
    let result = {
        if args.long_form {
            bail!("rebuild with --features long-form to use --long-form")
        }
        TranscriptionPipeline::from_dir(&args.models_dir)?.run(&audio)?
    };

    println!("file: {}", args.audio_path.display());
    println!("audio_seconds: {:.1}", audio.len() as f64 / 16_000.0);
    println!("elapsed_seconds: {:.2}", started_at.elapsed().as_secs_f64());
    println!("chunks: {}", result.chunks.len());
    println!("tokens: {}", result.tokens.len());
    println!("{}", preview(&result.text, args.preview_chars));
    Ok(())
}

#[derive(Debug, Clone)]
struct Args {
    audio_path: PathBuf,
    models_dir: PathBuf,
    long_form: bool,
    preview_chars: usize,
}

impl Args {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self> {
        let mut args = args.into_iter();
        let mut audio_path = None;
        let mut models_dir = PathBuf::from("fixtures/models");
        let mut long_form = false;
        let mut preview_chars = 800usize;

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--audio" => audio_path = Some(next_path(&mut args, "--audio")?),
                "--models-dir" => models_dir = next_path(&mut args, "--models-dir")?,
                "--long-form" => long_form = true,
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

        Ok(Self {
            audio_path,
            models_dir,
            long_form,
            preview_chars,
        })
    }
}

fn print_usage() {
    eprintln!(
        "Usage:
  cargo run --example transcribe_wav -- --audio <path.wav>
  cargo run --example transcribe_wav --features long-form -- --audio <path.wav> --long-form

Options:
  --models-dir <dir>         scriptrs model bundle directory
  --long-form                use LongFormTranscriptionPipeline
  --preview-chars <n>        text preview limit"
    );
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
