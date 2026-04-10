use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use eyre::{Result, bail, eyre};
use parakeet_rs::{ParakeetTDT, TimestampMode, Transcriber};
use scriptrs::{TimedToken, TranscriptionPipeline};

const DEFAULT_DATASET: &str = "voxconverse";
const DEFAULT_MAX_FILES: usize = 3;
const DEFAULT_CHUNKS_PER_FILE: usize = 2;
const DEFAULT_CHUNK_SECONDS: f64 = 15.0;
const DEFAULT_BENCHMARK_WARMUP: usize = 1;
const DEFAULT_BENCHMARK_RUNS: usize = 0;
const COREML_RUNTIME_MODE_ENV: &str = "SCRIPTRS_COREML_RUNTIME_MODE";
const DEV_TOOLS_DIR: &str = env!("CARGO_MANIFEST_DIR");

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
    configure_runtime_mode(args.runtime_mode);
    let scriptrs_pipeline = TranscriptionPipeline::from_dir(&args.models_dir)?;
    let parakeet_dir = PreparedParakeetDir::new(&args.models_dir, &args.onnx_dir)?;
    let mut parakeet = ParakeetTDT::from_pretrained(parakeet_dir.path(), None)?;

    match args.input.clone() {
        InputMode::Audio(audio_path) => {
            let audio = read_mono_16khz_wav(&audio_path)?;
            let chunk = AudioChunk::new(audio_path.display().to_string(), 0, audio);
            let comparison = compare_chunk(&scriptrs_pipeline, &mut parakeet, &chunk)?;
            print_single_report(&args, &comparison);
            maybe_print_benchmark(&args, &scriptrs_pipeline, std::slice::from_ref(&chunk))?;
            if args.strict && !comparison.summary.is_match() {
                bail!("scriptrs and parakeet-rs results differ")
            }
        }
        InputMode::Dataset(dataset_dir) => {
            let chunks = load_dataset_chunks(&args, &dataset_dir)?;
            let report = compare_chunks(&scriptrs_pipeline, &mut parakeet, &chunks)?;
            print_dataset_report(&args, &dataset_dir, &report);
            maybe_print_benchmark(&args, &scriptrs_pipeline, &chunks)?;
            if args.strict && report.mismatch_count > 0 {
                bail!("found {} mismatching chunks", report.mismatch_count)
            }
        }
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct Args {
    input: InputMode,
    models_dir: PathBuf,
    onnx_dir: PathBuf,
    strict: bool,
    token_limit: usize,
    max_files: usize,
    chunks_per_file: usize,
    chunk_seconds: f64,
    benchmark_warmup: usize,
    benchmark_runs: usize,
    runtime_mode: RuntimeMode,
}

#[derive(Debug, Clone)]
enum InputMode {
    Audio(PathBuf),
    Dataset(PathBuf),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuntimeMode {
    Sync,
    AsyncExperiment,
}

impl RuntimeMode {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "sync" => Ok(Self::Sync),
            "async-experiment" => Ok(Self::AsyncExperiment),
            _ => bail!("invalid --runtime-mode value: {value}"),
        }
    }

    fn as_env_value(self) -> &'static str {
        match self {
            Self::Sync => "sync",
            Self::AsyncExperiment => "async-experiment",
        }
    }
}

impl Args {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self> {
        let mut audio_path = None;
        let mut dataset_dir = None;
        let mut speakrs_dataset = None;
        let mut models_dir = repo_root().join("fixtures/models");
        let mut onnx_dir = None;
        let mut strict = false;
        let mut token_limit = 12usize;
        let mut max_files = DEFAULT_MAX_FILES;
        let mut chunks_per_file = DEFAULT_CHUNKS_PER_FILE;
        let mut chunk_seconds = DEFAULT_CHUNK_SECONDS;
        let mut benchmark_warmup = DEFAULT_BENCHMARK_WARMUP;
        let mut benchmark_runs = DEFAULT_BENCHMARK_RUNS;
        let mut runtime_mode = RuntimeMode::Sync;

        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--audio" => audio_path = Some(next_path(&mut args, "--audio")?),
                "--dataset-dir" => dataset_dir = Some(next_path(&mut args, "--dataset-dir")?),
                "--speakrs-dataset" => {
                    speakrs_dataset = Some(next_value(&mut args, "--speakrs-dataset")?)
                }
                "--models-dir" => models_dir = next_path(&mut args, "--models-dir")?,
                "--onnx-dir" => onnx_dir = Some(next_path(&mut args, "--onnx-dir")?),
                "--strict" => strict = true,
                "--token-limit" => {
                    token_limit = next_value(&mut args, "--token-limit")?
                        .parse()
                        .map_err(|error| eyre!("invalid --token-limit value: {error}"))?;
                }
                "--max-files" => {
                    max_files = next_value(&mut args, "--max-files")?
                        .parse()
                        .map_err(|error| eyre!("invalid --max-files value: {error}"))?;
                }
                "--chunks-per-file" => {
                    chunks_per_file = next_value(&mut args, "--chunks-per-file")?
                        .parse()
                        .map_err(|error| eyre!("invalid --chunks-per-file value: {error}"))?;
                }
                "--chunk-seconds" => {
                    chunk_seconds = next_value(&mut args, "--chunk-seconds")?
                        .parse()
                        .map_err(|error| eyre!("invalid --chunk-seconds value: {error}"))?;
                }
                "--benchmark-warmup" => {
                    benchmark_warmup = next_value(&mut args, "--benchmark-warmup")?
                        .parse()
                        .map_err(|error| eyre!("invalid --benchmark-warmup value: {error}"))?;
                }
                "--benchmark-runs" => {
                    benchmark_runs = next_value(&mut args, "--benchmark-runs")?
                        .parse()
                        .map_err(|error| eyre!("invalid --benchmark-runs value: {error}"))?;
                }
                "--runtime-mode" => {
                    runtime_mode = RuntimeMode::parse(&next_value(&mut args, "--runtime-mode")?)?;
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

        if chunk_seconds <= 0.0 {
            bail!("--chunk-seconds must be positive")
        }
        if max_files == 0 {
            bail!("--max-files must be at least 1")
        }
        if chunks_per_file == 0 {
            bail!("--chunks-per-file must be at least 1")
        }
        if benchmark_runs > 0 && benchmark_warmup == 0 {
            bail!("--benchmark-warmup must be at least 1 when benchmarking")
        }

        let input = match (audio_path, dataset_dir, speakrs_dataset) {
            (Some(audio_path), None, None) => InputMode::Audio(audio_path),
            (None, Some(dataset_dir), None) => InputMode::Dataset(dataset_dir),
            (None, None, Some(dataset_id)) => {
                InputMode::Dataset(resolve_speakrs_dataset_dir(&dataset_id))
            }
            (None, None, None) => InputMode::Dataset(resolve_speakrs_dataset_dir(DEFAULT_DATASET)),
            _ => bail!("choose exactly one of --audio, --dataset-dir, or --speakrs-dataset"),
        };
        let onnx_dir = onnx_dir.unwrap_or_else(|| models_dir.join("parakeet-v2/onnx"));

        Ok(Self {
            input,
            models_dir,
            onnx_dir,
            strict,
            token_limit,
            max_files,
            chunks_per_file,
            chunk_seconds,
            benchmark_warmup,
            benchmark_runs,
            runtime_mode,
        })
    }
}

fn print_usage() {
    eprintln!(
        "Usage:
  cargo run --manifest-path dev-tools/Cargo.toml --bin compare_parakeet_rs -- --audio <path.wav>
  cargo run --manifest-path dev-tools/Cargo.toml --bin compare_parakeet_rs -- --speakrs-dataset voxconverse

Options:
  --dataset-dir <dir>        directory of 16k mono benchmark wavs
  --speakrs-dataset <id>     dataset under ../speakrs/fixtures/datasets/<id>/wav
  --models-dir <dir>         scriptrs model bundle directory
  --onnx-dir <dir>           ONNX directory for parakeet-rs
  --max-files <n>            dataset files to sample (default: 3)
  --chunks-per-file <n>      15s chunks sampled per file (default: 2)
  --chunk-seconds <n>        chunk length in seconds (default: 15)
  --benchmark-warmup <n>     warmup passes before timing (default: 1)
  --benchmark-runs <n>       timed passes for scriptrs (default: 0)
  --runtime-mode <mode>      sync or async-experiment (default: sync)
  --token-limit <n>          token preview rows (default: 12)
  --strict                   exit non-zero on mismatch"
    );
}

fn configure_runtime_mode(runtime_mode: RuntimeMode) {
    // safe because this CLI mutates the process environment before starting worker threads
    unsafe {
        env::set_var(COREML_RUNTIME_MODE_ENV, runtime_mode.as_env_value());
    }
}

fn resolve_speakrs_dataset_dir(dataset_id: &str) -> PathBuf {
    repo_root()
        .join("../speakrs/fixtures/datasets")
        .join(dataset_id)
        .join("wav")
}

fn repo_root() -> PathBuf {
    PathBuf::from(DEV_TOOLS_DIR)
        .parent()
        .expect("dev-tools crate should have a parent directory")
        .to_path_buf()
}

fn next_path(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<PathBuf> {
    Ok(PathBuf::from(next_value(args, flag)?))
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    args.next().ok_or_else(|| eyre!("missing value for {flag}"))
}

#[derive(Debug, Clone)]
struct AudioChunk {
    label: String,
    original_samples: usize,
    audio: Vec<f32>,
}

impl AudioChunk {
    fn new(label: String, original_samples: usize, audio: Vec<f32>) -> Self {
        let original_samples = if original_samples == 0 {
            audio.len()
        } else {
            original_samples
        };
        Self {
            label,
            original_samples,
            audio,
        }
    }
}

#[derive(Debug, Clone)]
struct Comparison {
    label: String,
    original_samples: usize,
    scriptrs_text: String,
    parakeet_text: String,
    scriptrs_tokens: Vec<TimedToken>,
    parakeet_tokens: Vec<parakeet_rs::TimedToken>,
    summary: ComparisonSummary,
}

#[derive(Debug, Clone)]
struct DatasetReport {
    comparisons: Vec<Comparison>,
    mismatch_count: usize,
    text_match_count: usize,
    exact_match_count: usize,
    aligned_tokens: usize,
    matching_tokens: usize,
    max_start_delta: f64,
    max_end_delta: f64,
}

#[derive(Debug, Clone)]
struct BenchmarkReport {
    runtime_mode: RuntimeMode,
    warmup_runs: usize,
    timed_runs: usize,
    chunk_count: usize,
    total_duration: Duration,
    mean_duration: Duration,
    p50_duration: Duration,
    p95_duration: Duration,
}

fn load_dataset_chunks(args: &Args, dataset_dir: &Path) -> Result<Vec<AudioChunk>> {
    let wav_paths = discover_wavs(dataset_dir)?;
    if wav_paths.is_empty() {
        bail!("no wav files found in {}", dataset_dir.display())
    }

    let mut chunks = Vec::new();
    let chunk_samples = (args.chunk_seconds * 16_000.0).round() as usize;
    for wav_path in wav_paths.into_iter().take(args.max_files) {
        let audio = read_mono_16khz_wav(&wav_path)?;
        for (chunk_index, chunk) in
            sample_chunks(&wav_path, &audio, chunk_samples, args.chunks_per_file)
        {
            let label = format!("{}#{}", wav_path.display(), chunk_index);
            chunks.push(AudioChunk::new(label, chunk.len(), chunk));
        }
    }

    Ok(chunks)
}

fn compare_chunks(
    scriptrs_pipeline: &TranscriptionPipeline,
    parakeet: &mut ParakeetTDT,
    chunks: &[AudioChunk],
) -> Result<DatasetReport> {
    let mut comparisons = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        comparisons.push(compare_chunk(scriptrs_pipeline, parakeet, chunk)?);
    }

    let mismatch_count = comparisons
        .iter()
        .filter(|comparison| !comparison.summary.is_match())
        .count();
    let text_match_count = comparisons
        .iter()
        .filter(|comparison| comparison.summary.normalized_text_match)
        .count();
    let exact_match_count = comparisons
        .iter()
        .filter(|comparison| comparison.summary.is_match())
        .count();
    let aligned_tokens = comparisons
        .iter()
        .map(|comparison| comparison.summary.aligned_tokens)
        .sum();
    let matching_tokens = comparisons
        .iter()
        .map(|comparison| comparison.summary.matching_tokens)
        .sum();
    let max_start_delta = comparisons
        .iter()
        .map(|comparison| comparison.summary.max_start_delta)
        .fold(0.0, f64::max);
    let max_end_delta = comparisons
        .iter()
        .map(|comparison| comparison.summary.max_end_delta)
        .fold(0.0, f64::max);

    Ok(DatasetReport {
        comparisons,
        mismatch_count,
        text_match_count,
        exact_match_count,
        aligned_tokens,
        matching_tokens,
        max_start_delta,
        max_end_delta,
    })
}

fn maybe_print_benchmark(
    args: &Args,
    scriptrs_pipeline: &TranscriptionPipeline,
    chunks: &[AudioChunk],
) -> Result<()> {
    if args.benchmark_runs == 0 {
        return Ok(());
    }

    let report = benchmark_scriptrs(
        scriptrs_pipeline,
        chunks,
        args.runtime_mode,
        args.benchmark_warmup,
        args.benchmark_runs,
    )?;
    print_benchmark_report(&report);
    Ok(())
}

fn benchmark_scriptrs(
    scriptrs_pipeline: &TranscriptionPipeline,
    chunks: &[AudioChunk],
    runtime_mode: RuntimeMode,
    warmup_runs: usize,
    timed_runs: usize,
) -> Result<BenchmarkReport> {
    let benchmark_inputs: Vec<_> = chunks
        .iter()
        .map(|chunk| {
            pad_audio_for_scriptrs(&chunk.audio, scriptrs_pipeline.config().max_audio_samples)
        })
        .collect();

    for _ in 0..warmup_runs {
        run_benchmark_pass(scriptrs_pipeline, &benchmark_inputs)?;
    }

    let mut durations = Vec::with_capacity(timed_runs);
    for _ in 0..timed_runs {
        durations.push(run_benchmark_pass(scriptrs_pipeline, &benchmark_inputs)?);
    }
    let total_duration = durations.iter().copied().sum::<Duration>();
    let mean_duration = total_duration.div_f64(timed_runs as f64);
    let (p50_duration, p95_duration) = percentile_durations(&durations);

    Ok(BenchmarkReport {
        runtime_mode,
        warmup_runs,
        timed_runs,
        chunk_count: benchmark_inputs.len(),
        total_duration,
        mean_duration,
        p50_duration,
        p95_duration,
    })
}

fn run_benchmark_pass(
    scriptrs_pipeline: &TranscriptionPipeline,
    benchmark_inputs: &[Vec<f32>],
) -> Result<Duration> {
    let start = Instant::now();
    for audio in benchmark_inputs {
        let _ = scriptrs_pipeline.run(audio)?;
    }
    Ok(start.elapsed())
}

fn percentile_durations(durations: &[Duration]) -> (Duration, Duration) {
    let mut sorted = durations.to_vec();
    sorted.sort_unstable();
    let p50_index = percentile_index(sorted.len(), 50);
    let p95_index = percentile_index(sorted.len(), 95);
    (sorted[p50_index], sorted[p95_index])
}

fn percentile_index(len: usize, percentile: usize) -> usize {
    ((len - 1) * percentile).div_ceil(100)
}

fn compare_chunk(
    scriptrs_pipeline: &TranscriptionPipeline,
    parakeet: &mut ParakeetTDT,
    chunk: &AudioChunk,
) -> Result<Comparison> {
    let padded_audio =
        pad_audio_for_scriptrs(&chunk.audio, scriptrs_pipeline.config().max_audio_samples);
    let scriptrs_result = scriptrs_pipeline.run(&padded_audio)?;
    let parakeet_result =
        parakeet.transcribe_samples(padded_audio, 16_000, 1, Some(TimestampMode::Tokens))?;
    let summary = ComparisonSummary::new(&scriptrs_result.tokens, &parakeet_result.tokens)
        .with_text_match(&scriptrs_result.text, &parakeet_result.text);

    Ok(Comparison {
        label: chunk.label.clone(),
        original_samples: chunk.original_samples,
        scriptrs_text: scriptrs_result.text,
        parakeet_text: parakeet_result.text,
        scriptrs_tokens: scriptrs_result.tokens,
        parakeet_tokens: parakeet_result.tokens,
        summary,
    })
}

fn discover_wavs(dataset_dir: &Path) -> Result<Vec<PathBuf>> {
    if !dataset_dir.exists() {
        bail!(
            "dataset directory does not exist: {}",
            dataset_dir.display()
        )
    }

    let mut wavs = fs::read_dir(dataset_dir)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    wavs.retain(|path| {
        path.extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("wav"))
    });
    wavs.sort();
    Ok(wavs)
}

fn sample_chunks(
    _wav_path: &Path,
    audio: &[f32],
    chunk_samples: usize,
    chunks_per_file: usize,
) -> Vec<(usize, Vec<f32>)> {
    if audio.len() <= chunk_samples {
        return vec![(0, audio.to_vec())];
    }

    let max_start = audio.len() - chunk_samples;
    let mut starts = Vec::new();
    if chunks_per_file == 1 || max_start == 0 {
        starts.push(0);
    } else {
        for index in 0..chunks_per_file {
            let numerator = index * max_start;
            let denominator = chunks_per_file - 1;
            let start = numerator / denominator;
            if starts.last().copied() != Some(start) {
                starts.push(start);
            }
        }
    }

    starts
        .into_iter()
        .map(|start| {
            let end = start + chunk_samples;
            (start / 16_000, audio[start..end].to_vec())
        })
        .collect()
}

fn read_mono_16khz_wav(path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
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

fn pad_audio_for_scriptrs(audio: &[f32], target_samples: usize) -> Vec<f32> {
    if audio.len() >= target_samples {
        return audio.to_vec();
    }

    let mut padded = Vec::with_capacity(target_samples);
    padded.extend_from_slice(audio);
    padded.resize(target_samples, 0.0);
    padded
}

#[derive(Debug, Clone)]
struct ComparableToken {
    text: String,
    start: f64,
    end: f64,
}

impl ComparableToken {
    fn normalized_text(&self) -> String {
        normalize_whitespace(&self.text)
    }
}

impl From<&TimedToken> for ComparableToken {
    fn from(token: &TimedToken) -> Self {
        Self {
            text: token.text.clone(),
            start: token.start,
            end: token.end,
        }
    }
}

impl From<&parakeet_rs::TimedToken> for ComparableToken {
    fn from(token: &parakeet_rs::TimedToken) -> Self {
        Self {
            text: token.text.clone(),
            start: token.start as f64,
            end: token.end as f64,
        }
    }
}

#[derive(Debug, Clone)]
struct ComparisonSummary {
    normalized_text_match: bool,
    token_count_match: bool,
    aligned_tokens: usize,
    matching_tokens: usize,
    first_token_mismatch: Option<usize>,
    max_start_delta: f64,
    max_end_delta: f64,
}

impl ComparisonSummary {
    fn new(scriptrs_tokens: &[TimedToken], parakeet_tokens: &[parakeet_rs::TimedToken]) -> Self {
        let left: Vec<_> = scriptrs_tokens.iter().map(ComparableToken::from).collect();
        let right: Vec<_> = parakeet_tokens.iter().map(ComparableToken::from).collect();
        let aligned_tokens = left.len().min(right.len());
        let mut matching_tokens = 0usize;
        let mut first_token_mismatch = None;
        let mut max_start_delta = 0.0f64;
        let mut max_end_delta = 0.0f64;

        for index in 0..aligned_tokens {
            let left_token = &left[index];
            let right_token = &right[index];
            if left_token.normalized_text() == right_token.normalized_text() {
                matching_tokens += 1;
            } else if first_token_mismatch.is_none() {
                first_token_mismatch = Some(index);
            }

            max_start_delta = max_start_delta.max((left_token.start - right_token.start).abs());
            max_end_delta = max_end_delta.max((left_token.end - right_token.end).abs());
        }

        Self {
            normalized_text_match: false,
            token_count_match: left.len() == right.len(),
            aligned_tokens,
            matching_tokens,
            first_token_mismatch,
            max_start_delta,
            max_end_delta,
        }
    }

    fn with_text_match(mut self, scriptrs_text: &str, parakeet_text: &str) -> Self {
        self.normalized_text_match =
            normalize_whitespace(scriptrs_text) == normalize_whitespace(parakeet_text);
        self
    }

    fn is_match(&self) -> bool {
        self.normalized_text_match
            && self.token_count_match
            && self.first_token_mismatch.is_none()
            && self.matching_tokens == self.aligned_tokens
    }
}

fn print_single_report(args: &Args, comparison: &Comparison) {
    println!("models_dir: {}", args.models_dir.display());
    println!("onnx_dir: {}", args.onnx_dir.display());
    println!("runtime_mode: {}", args.runtime_mode.as_env_value());
    println!("label: {}", comparison.label);
    println!(
        "original_duration: {:.3}s",
        comparison.original_samples as f64 / 16_000.0
    );
    println!("comparison_duration: 15.000s");
    println!(
        "normalized_text_match: {}",
        comparison.summary.normalized_text_match
    );
    println!(
        "token_count_match: {}",
        comparison.summary.token_count_match
    );
    println!(
        "scriptrs_text: {}",
        normalize_whitespace(&comparison.scriptrs_text)
    );
    println!(
        "parakeet_rs_text: {}",
        normalize_whitespace(&comparison.parakeet_text)
    );
    println!("scriptrs_tokens: {}", comparison.scriptrs_tokens.len());
    println!("parakeet_rs_tokens: {}", comparison.parakeet_tokens.len());
    println!(
        "aligned_tokens: {} matching_tokens: {}",
        comparison.summary.aligned_tokens, comparison.summary.matching_tokens
    );
    println!(
        "max_start_delta: {:.6}s max_end_delta: {:.6}s",
        comparison.summary.max_start_delta, comparison.summary.max_end_delta
    );

    if let Some(index) = comparison.summary.first_token_mismatch {
        println!("first_token_mismatch: {index}");
    } else {
        println!("first_token_mismatch: none");
    }

    println!("token_preview:");
    print_token_rows(
        &comparison.scriptrs_tokens,
        &comparison.parakeet_tokens,
        args.token_limit,
        comparison.summary.first_token_mismatch,
    );
}

fn print_dataset_report(args: &Args, dataset_dir: &Path, report: &DatasetReport) {
    println!("models_dir: {}", args.models_dir.display());
    println!("onnx_dir: {}", args.onnx_dir.display());
    println!("runtime_mode: {}", args.runtime_mode.as_env_value());
    println!("dataset_dir: {}", dataset_dir.display());
    println!("files_sampled: {}", args.max_files);
    println!("chunks_per_file: {}", args.chunks_per_file);
    println!("chunks_compared: {}", report.comparisons.len());
    println!("text_matches: {}", report.text_match_count);
    println!("exact_matches: {}", report.exact_match_count);
    println!("mismatches: {}", report.mismatch_count);
    println!(
        "token_matches: {}/{}",
        report.matching_tokens, report.aligned_tokens
    );
    println!(
        "max_start_delta: {:.6}s max_end_delta: {:.6}s",
        report.max_start_delta, report.max_end_delta
    );

    let mismatches: Vec<_> = report
        .comparisons
        .iter()
        .filter(|comparison| !comparison.summary.is_match())
        .collect();
    if mismatches.is_empty() {
        println!("all sampled chunks matched");
        return;
    }

    println!("mismatch_preview:");
    for comparison in mismatches.into_iter().take(3) {
        println!("label: {}", comparison.label);
        println!(
            "  text_match={} token_count_match={} aligned_tokens={} matching_tokens={}",
            comparison.summary.normalized_text_match,
            comparison.summary.token_count_match,
            comparison.summary.aligned_tokens,
            comparison.summary.matching_tokens
        );
        println!(
            "  scriptrs={}",
            normalize_whitespace(&comparison.scriptrs_text)
        );
        println!(
            "  parakeet-rs={}",
            normalize_whitespace(&comparison.parakeet_text)
        );
    }
}

fn print_benchmark_report(report: &BenchmarkReport) {
    println!(
        "benchmark_runtime_mode: {}",
        report.runtime_mode.as_env_value()
    );
    println!("benchmark_warmup_runs: {}", report.warmup_runs);
    println!("benchmark_timed_runs: {}", report.timed_runs);
    println!("benchmark_chunks_per_run: {}", report.chunk_count);
    println!(
        "benchmark_total_duration: {}",
        format_duration_secs(report.total_duration)
    );
    println!(
        "benchmark_mean_duration: {}",
        format_duration_secs(report.mean_duration)
    );
    println!(
        "benchmark_p50_duration: {}",
        format_duration_secs(report.p50_duration)
    );
    println!(
        "benchmark_p95_duration: {}",
        format_duration_secs(report.p95_duration)
    );
}

fn print_token_rows(
    scriptrs_tokens: &[TimedToken],
    parakeet_tokens: &[parakeet_rs::TimedToken],
    token_limit: usize,
    first_token_mismatch: Option<usize>,
) {
    let preview_limit = match first_token_mismatch {
        Some(index) => token_limit.max(index + 3),
        None => token_limit,
    };
    let rows = preview_limit.min(scriptrs_tokens.len().max(parakeet_tokens.len()));

    for index in 0..rows {
        let left = scriptrs_tokens.get(index).map(ComparableToken::from);
        let right = parakeet_tokens.get(index).map(ComparableToken::from);
        let status = match (&left, &right) {
            (Some(left), Some(right)) if left.normalized_text() == right.normalized_text() => "=",
            (Some(_), Some(_)) => "!",
            _ => "~",
        };

        println!(
            "{status} [{index:03}] scriptrs={} | parakeet-rs={}",
            format_token(left.as_ref()),
            format_token(right.as_ref())
        );
    }
}

fn format_token(token: Option<&ComparableToken>) -> String {
    let Some(token) = token else {
        return "<missing>".to_owned();
    };
    format!(
        "{:?} [{:.3}, {:.3}]",
        token.normalized_text(),
        token.start,
        token.end
    )
}

fn format_duration_secs(duration: Duration) -> String {
    format!("{:.6}s", duration.as_secs_f64())
}

fn normalize_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

struct PreparedParakeetDir(PathBuf);

impl PreparedParakeetDir {
    fn new(models_dir: &Path, onnx_dir: &Path) -> Result<Self> {
        let staging_dir = PathBuf::from("target/compare-parakeet-rs-models");
        let vocab_path = resolve_vocab_path(models_dir, onnx_dir)?;
        if staging_dir.exists() {
            fs::remove_dir_all(&staging_dir)?;
        }
        fs::create_dir_all(&staging_dir)?;

        for file_name in [
            "encoder-model.onnx",
            "encoder-model.onnx.data",
            "decoder_joint-model.onnx",
        ] {
            let source = onnx_dir.join(file_name);
            if source.exists() {
                link_or_copy_file(&source, &staging_dir.join(file_name))?;
            }
        }
        link_or_copy_file(&vocab_path, &staging_dir.join("vocab.txt"))?;

        Ok(Self(staging_dir))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

fn resolve_vocab_path(models_dir: &Path, onnx_dir: &Path) -> Result<PathBuf> {
    let candidates = [
        onnx_dir.join("vocab.txt"),
        onnx_dir
            .parent()
            .map(|parent| parent.join("vocab.txt"))
            .unwrap_or_else(|| onnx_dir.join("vocab.txt")),
        models_dir.join("parakeet-v2/vocab.txt"),
    ];
    candidates
        .into_iter()
        .find(|path| path.exists())
        .ok_or_else(|| eyre!("unable to locate vocab.txt for parakeet-rs"))
}

fn link_or_copy_file(source: &Path, destination: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        let source = source.canonicalize()?;
        if std::os::unix::fs::symlink(&source, destination).is_ok() {
            return Ok(());
        }
    }

    fs::copy(source, destination)?;
    Ok(())
}
