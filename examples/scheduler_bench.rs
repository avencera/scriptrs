//! Throughput and request latency of the transcription scheduler under concurrent
//! load, against the same requests served one at a time
//!
//! `--clients 1,8,32` runs one pass per client count, starting with one client,
//! the pass the others are held against. Each client sends its next
//! request as soon as the last one returns, until every request has run once; a
//! request's latency runs from its submit to its result. The scheduler batches
//! windows across the requests in flight. The baseline is today's server: one
//! long-form pipeline behind a FIFO queue, running each request's windows in its
//! own batches of 16 while the other clients wait their turn
//!
//! The workload is LibriSpeech test-clean (`--manifest`), Earnings-21 calls
//! (`--long <wav or directory>`, repeatable), or both, the calls spread evenly among
//! the utterances. Every pass's transcripts are held against the scheduler's
//! one-client pass, token for token, and counted against the baseline's. Audio is
//! preloaded, and an untimed pass over every request first captures each bucket's
//! graphs. Writes a JSON summary to `--out` and the transcripts (`{id: text}`) of
//! the scheduler's one-client pass and of the baseline next to it
//!
//! ```sh
//! cargo run --release --features cuda,long-form --example scheduler_bench -- \
//!     --models /shared/models/scriptrs-native --clients 1,8,32,128 --out bench/short.json
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use eyre::{Result, bail, eyre};
use hound::WavReader;
use scriptrs::{
    LongFormConfig, LongFormTranscriptionPipeline, ModelBundle, ParakeetDeployment,
    SchedulerConfig, TranscriptionResult,
};
use serde_json::{Value, json};

const DEFAULT_MANIFEST: &str = "/shared/datasets/librispeech-test-clean/manifest.jsonl";

struct Args {
    models: PathBuf,
    manifest: Option<PathBuf>,
    long: Vec<PathBuf>,
    clients: Vec<usize>,
    out: PathBuf,
    baseline: bool,
    /// Requests taken from the start of the workload, for short profiling runs
    limit: Option<usize>,
    scheduler: SchedulerConfig,
}

struct Request {
    id: String,
    long: bool,
    audio: Arc<[f32]>,
}

/// One pass: wall time, then each request's latency and transcript by index
struct Pass {
    wall: Duration,
    latencies: Vec<Duration>,
    results: Vec<TranscriptionResult>,
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
    let mut requests = load_requests(&args)?;
    if let Some(limit) = args.limit {
        requests.truncate(limit);
    }
    let audio_s: f64 = requests
        .iter()
        .map(|request| request.audio.len() as f64 / 16_000.0)
        .sum();
    eprintln!(
        "{} requests ({} long), {:.2} h of audio",
        requests.len(),
        requests.iter().filter(|request| request.long).count(),
        audio_s / 3600.0
    );

    let bundle = ModelBundle::from_native_dir(&args.models, ParakeetDeployment::v3());
    let scheduler = LongFormTranscriptionPipeline::from_bundle(bundle.clone())?
        .into_scheduler(args.scheduler)?;
    let warmup_clients = args.clients.iter().copied().max().unwrap_or(1);
    let started = Instant::now();
    pass(&requests, warmup_clients, |audio| {
        scheduler.transcribe(audio)
    })?;
    eprintln!("warmup: {:.1} s", started.elapsed().as_secs_f64());

    let mut reference: Option<Pass> = None;
    let mut rows = Vec::new();
    for &clients in &args.clients {
        let scheduled = pass(&requests, clients, |audio| scheduler.transcribe(audio))?;
        let changed = reference
            .as_ref()
            .map_or(0, |reference| changed(&scheduled, reference));
        rows.push(summary(
            "scheduler",
            clients,
            audio_s,
            &requests,
            &scheduled,
            changed,
        ));
        print_row(rows.last().expect("a row"));
        if reference.is_none() {
            reference = Some(scheduled);
        }
    }
    let reference = reference.ok_or_else(|| eyre!("no client counts given"))?;
    drop(scheduler);

    let mut baseline_texts = None;
    if args.baseline {
        let pipeline = LongFormTranscriptionPipeline::from_bundle(bundle.clone())?;
        let config = LongFormConfig::for_bundle(&bundle);
        let server = OneAtATime::start(pipeline, config);
        pass(&requests, 1, |audio| server.transcribe(audio))?;
        for &clients in &args.clients {
            let served = pass(&requests, clients, |audio| server.transcribe(audio))?;
            let changed = changed(&served, &reference);
            rows.push(summary(
                "one at a time",
                clients,
                audio_s,
                &requests,
                &served,
                changed,
            ));
            print_row(rows.last().expect("a row"));
            baseline_texts.get_or_insert_with(|| texts(&requests, &served));
        }
    }

    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent)?;
    }
    let config = &args.scheduler;
    let summary = json!({
        "requests": requests.len(),
        "long_requests": requests.iter().filter(|request| request.long).count(),
        "audio_hours": audio_s / 3600.0,
        "scheduler": {
            "max_requests": config.max_requests,
            "decoder_rows": config.decoder_rows,
            "frames_per_batch": config.frames_per_batch,
            "min_batch": config.min_batch,
            "max_batch": config.max_batch,
            "bucket_capacity": config.bucket_capacity,
            "workers": config.workers,
            "batch_wait_ms": config.batch_wait.as_secs_f64() * 1e3,
        },
        "passes": rows,
    });
    fs::write(&args.out, serde_json::to_string_pretty(&summary)?)?;
    let texts_path = args.out.with_extension("texts.json");
    fs::write(
        &texts_path,
        serde_json::to_string_pretty(&texts(&requests, &reference))?,
    )?;
    if let Some(baseline) = baseline_texts {
        let path = args.out.with_extension("baseline.texts.json");
        fs::write(path, serde_json::to_string_pretty(&baseline)?)?;
    }
    eprintln!("wrote {}", args.out.display());
    Ok(())
}

/// Runs every request once from `clients` closed-loop clients
fn pass<F>(requests: &[Request], clients: usize, transcribe: F) -> Result<Pass>
where
    F: Fn(Arc<[f32]>) -> Result<TranscriptionResult, scriptrs::TranscriptionError> + Sync,
{
    let next = AtomicUsize::new(0);
    let outcomes = Mutex::new(BTreeMap::new());
    let started = Instant::now();
    thread::scope(|scope| {
        for _ in 0..clients.max(1) {
            scope.spawn(|| {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(request) = requests.get(index) else {
                        break;
                    };
                    let sent = Instant::now();
                    let result = transcribe(Arc::clone(&request.audio));
                    let latency = sent.elapsed();
                    outcomes
                        .lock()
                        .expect("outcomes")
                        .insert(index, (latency, result));
                }
            });
        }
    });
    let wall = started.elapsed();

    let mut latencies = Vec::with_capacity(requests.len());
    let mut results = Vec::with_capacity(requests.len());
    for (index, (latency, result)) in outcomes.into_inner().expect("outcomes") {
        let result = result.map_err(|error| eyre!("{}: {error}", requests[index].id))?;
        latencies.push(latency);
        results.push(result);
    }
    Ok(Pass {
        wall,
        latencies,
        results,
    })
}

/// Requests whose transcript differs from `reference` in any token, timing or
/// confidence
fn changed(pass: &Pass, reference: &Pass) -> usize {
    pass.results
        .iter()
        .zip(&reference.results)
        .filter(|(got, want)| {
            got.text != want.text
                || got.tokens.len() != want.tokens.len()
                || got.tokens.iter().zip(&want.tokens).any(|(got, want)| {
                    got.token_id != want.token_id
                        || got.start.to_bits() != want.start.to_bits()
                        || got.end.to_bits() != want.end.to_bits()
                        || got.confidence.to_bits() != want.confidence.to_bits()
                })
        })
        .count()
}

fn summary(
    server: &str,
    clients: usize,
    audio_s: f64,
    requests: &[Request],
    pass: &Pass,
    changed: usize,
) -> Value {
    let class = |long: bool| -> Value {
        let mut latencies: Vec<f64> = requests
            .iter()
            .zip(&pass.latencies)
            .filter(|(request, _)| request.long == long)
            .map(|(_, latency)| latency.as_secs_f64() * 1e3)
            .collect();
        if latencies.is_empty() {
            return Value::Null;
        }
        latencies.sort_by(f64::total_cmp);
        json!({
            "requests": latencies.len(),
            "p50_ms": percentile(&latencies, 0.50),
            "p95_ms": percentile(&latencies, 0.95),
            "max_ms": latencies[latencies.len() - 1],
        })
    };
    let wall = pass.wall.as_secs_f64();
    json!({
        "server": server,
        "clients": clients,
        "wall_s": wall,
        "x_real_time": audio_s / wall,
        "requests_per_s": requests.len() as f64 / wall,
        "short": class(false),
        "long": class(true),
        "changed_transcripts": changed,
    })
}

fn print_row(row: &Value) {
    let latency = |class: &str| -> String {
        let class = &row[class];
        if class.is_null() {
            return "-".to_owned();
        }
        format!(
            "p50 {:.1} / p95 {:.1} ms",
            class["p50_ms"].as_f64().unwrap_or(0.0),
            class["p95_ms"].as_f64().unwrap_or(0.0)
        )
    };
    eprintln!(
        "{:>13} {:>4} clients: {:7.0}x real time, {:7.1} req/s, short {}, long {}, {} changed",
        row["server"].as_str().unwrap_or(""),
        row["clients"],
        row["x_real_time"].as_f64().unwrap_or(0.0),
        row["requests_per_s"].as_f64().unwrap_or(0.0),
        latency("short"),
        latency("long"),
        row["changed_transcripts"],
    );
}

/// Nearest-rank percentile of sorted values
fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let rank = (quantile * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

fn texts(requests: &[Request], pass: &Pass) -> BTreeMap<String, String> {
    requests
        .iter()
        .zip(&pass.results)
        .map(|(request, result)| (request.id.clone(), result.text.clone()))
        .collect()
}

/// One long-form pipeline serving requests in arrival order, one at a time
struct OneAtATime {
    requests: Mutex<Sender<Job>>,
}

type Job = (
    Arc<[f32]>,
    Sender<Result<TranscriptionResult, scriptrs::TranscriptionError>>,
);

impl OneAtATime {
    fn start(pipeline: LongFormTranscriptionPipeline, config: LongFormConfig) -> Self {
        let (sender, receiver) = channel::<Job>();
        thread::spawn(move || {
            for (audio, reply) in receiver {
                let _ = reply.send(pipeline.run_with_config(&audio, &config));
            }
        });
        Self {
            requests: Mutex::new(sender),
        }
    }

    fn transcribe(
        &self,
        audio: Arc<[f32]>,
    ) -> Result<TranscriptionResult, scriptrs::TranscriptionError> {
        let (reply, result) = channel();
        self.requests
            .lock()
            .expect("server queue")
            .send((audio, reply))
            .expect("server running");
        result.recv().expect("server reply")
    }
}

fn load_requests(args: &Args) -> Result<Vec<Request>> {
    let mut short = Vec::new();
    if let Some(manifest) = &args.manifest {
        for line in fs::read_to_string(manifest)?.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let row: Value = serde_json::from_str(line)?;
            let id = row["id"]
                .as_str()
                .ok_or_else(|| eyre!("manifest row without id"))?;
            let path = row["audio_filepath"]
                .as_str()
                .ok_or_else(|| eyre!("manifest row without audio_filepath"))?;
            short.push(Request {
                id: id.to_owned(),
                long: false,
                audio: read_wav(Path::new(path))?.into(),
            });
        }
    }

    let mut files = Vec::new();
    for path in &args.long {
        if path.is_dir() {
            let mut entries: Vec<PathBuf> = fs::read_dir(path)?
                .map(|entry| entry.map(|entry| entry.path()))
                .collect::<Result<_, _>>()?;
            entries.retain(|entry| entry.extension().is_some_and(|ext| ext == "wav"));
            entries.sort();
            files.extend(entries);
        } else {
            files.push(path.clone());
        }
    }
    let mut long = Vec::with_capacity(files.len());
    for file in files {
        let id = file
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| eyre!("bad file name {}", file.display()))?
            .to_owned();
        long.push(Request {
            id,
            long: true,
            audio: read_wav(&file)?.into(),
        });
    }

    // transcripts are saved by id, so a repeated id would hide one of them
    let mut ids = BTreeSet::new();
    if let Some(request) = short
        .iter()
        .chain(&long)
        .find(|request| !ids.insert(&request.id))
    {
        bail!("request id {} appears twice", request.id);
    }

    // spread the long requests evenly among the short ones
    if long.is_empty() || short.is_empty() {
        short.extend(long);
        return Ok(short);
    }
    let stride = short.len().div_ceil(long.len());
    let mut long = long.into_iter();
    let mut requests = Vec::new();
    for (index, request) in short.into_iter().enumerate() {
        if index % stride == 0
            && let Some(call) = long.next()
        {
            requests.push(call);
        }
        requests.push(request);
    }
    requests.extend(long);
    Ok(requests)
}

fn read_wav(path: &Path) -> Result<Vec<f32>> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_rate != 16_000 || spec.bits_per_sample != 16 {
        bail!("{}: expected 16 kHz mono 16-bit", path.display());
    }
    reader
        .samples::<i16>()
        .map(|sample| Ok(f32::from(sample?) / 32_768.0))
        .collect()
}

fn parse_args() -> Result<Args> {
    let mut args = Args {
        models: PathBuf::from("/shared/models/scriptrs-native"),
        manifest: None,
        long: Vec::new(),
        clients: vec![1, 8, 32],
        out: PathBuf::from("bench/scheduler.json"),
        baseline: false,
        limit: None,
        scheduler: SchedulerConfig::default(),
    };
    let mut iter = env::args().skip(1);
    let value = |iter: &mut dyn Iterator<Item = String>, flag: &str| {
        iter.next().ok_or_else(|| eyre!("{flag} needs a value"))
    };
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--models" => args.models = value(&mut iter, &flag)?.into(),
            "--manifest" => args.manifest = Some(value(&mut iter, &flag)?.into()),
            "--librispeech" => args.manifest = Some(DEFAULT_MANIFEST.into()),
            "--long" => args.long.push(value(&mut iter, &flag)?.into()),
            "--clients" => {
                args.clients = value(&mut iter, &flag)?
                    .split(',')
                    .map(str::parse)
                    .collect::<Result<_, _>>()?;
            }
            "--out" => args.out = value(&mut iter, &flag)?.into(),
            "--baseline" => args.baseline = true,
            "--limit" => args.limit = Some(value(&mut iter, &flag)?.parse()?),
            "--rows" => args.scheduler.decoder_rows = value(&mut iter, &flag)?.parse()?,
            "--frames-per-batch" => {
                args.scheduler.frames_per_batch = value(&mut iter, &flag)?.parse()?;
            }
            "--min-batch" => args.scheduler.min_batch = value(&mut iter, &flag)?.parse()?,
            "--max-batch" => args.scheduler.max_batch = value(&mut iter, &flag)?.parse()?,
            "--workers" => args.scheduler.workers = value(&mut iter, &flag)?.parse()?,
            "--batch-wait-ms" => {
                let ms: f64 = value(&mut iter, &flag)?.parse()?;
                args.scheduler.batch_wait = Duration::from_secs_f64(ms / 1e3);
            }
            other => return Err(eyre!("unknown flag: {other}")),
        }
    }
    if args.manifest.is_none() && args.long.is_empty() {
        bail!("give --librispeech, --manifest <jsonl> or --long <wav or directory>");
    }
    if args.clients.first() != Some(&1) {
        bail!("--clients must start with 1, the pass the others are held against");
    }
    Ok(args)
}
