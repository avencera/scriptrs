//! The transcription scheduler on the CUDA backend, against the blocking
//! pipelines on the same inputs
//!
//! The tests need a CUDA GPU, the native weights (`SCRIPTRS_NATIVE_MODELS`, default
//! `/shared/models/scriptrs-native`), LibriSpeech test-clean
//! (`SCRIPTRS_LIBRISPEECH_MANIFEST`) and the Earnings-21 audio
//! (`SCRIPTRS_EARNINGS21_WAV`); `just gpu-parity` runs them on gpu0

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;

use scriptrs::{
    LongFormTranscriptionPipeline, ModelBundle, ParakeetDeployment, SchedulerConfig,
    TranscriptionError, TranscriptionPipeline, TranscriptionResult, TranscriptionScheduler,
};

/// Utterances taken from the manifest, spread over its lengths
const UTTERANCES: usize = 64;

/// Seconds of each Earnings-21 call the long requests take
const CALL_SECONDS: usize = 150;

/// Earnings-21 calls the long requests come from
const CALLS: [&str; 3] = ["4320211", "4341191", "4330115"];

fn bundle() -> ModelBundle {
    let root = std::env::var_os("SCRIPTRS_NATIVE_MODELS").map_or_else(
        || PathBuf::from("/shared/models/scriptrs-native"),
        PathBuf::from,
    );
    ModelBundle::from_native_dir(root, ParakeetDeployment::v3())
}

fn long_form() -> LongFormTranscriptionPipeline {
    LongFormTranscriptionPipeline::from_bundle(bundle()).expect("pipeline")
}

fn scheduler() -> TranscriptionScheduler {
    long_form()
        .into_scheduler(SchedulerConfig::default())
        .expect("scheduler")
}

fn read_wav(path: &Path, max_samples: usize) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("wav");
    reader
        .samples::<i16>()
        .take(max_samples)
        .map(|sample| f32::from(sample.expect("sample")) / 32_768.0)
        .collect()
}

/// Short LibriSpeech utterances spread over the manifest, then the openings of a
/// few Earnings-21 calls, which run as long-form
fn requests() -> Vec<(String, Arc<[f32]>)> {
    let manifest = std::env::var_os("SCRIPTRS_LIBRISPEECH_MANIFEST").map_or_else(
        || PathBuf::from("/shared/datasets/librispeech-test-clean/manifest.jsonl"),
        PathBuf::from,
    );
    let rows: Vec<serde_json::Value> = std::fs::read_to_string(manifest)
        .expect("manifest")
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("manifest row"))
        .collect();
    let step = (rows.len() / UTTERANCES).max(1);
    let mut requests: Vec<(String, Arc<[f32]>)> = rows
        .iter()
        .step_by(step)
        .take(UTTERANCES)
        .map(|row| {
            let id = row["id"].as_str().expect("id").to_owned();
            let path = row["audio_filepath"].as_str().expect("audio path");
            (id, read_wav(Path::new(path), usize::MAX).into())
        })
        .collect();

    let calls = std::env::var_os("SCRIPTRS_EARNINGS21_WAV").map_or_else(
        || {
            let home = std::env::var_os("HOME").expect("HOME");
            PathBuf::from(home).join("code/speakrs/fixtures/datasets/earnings21/wav")
        },
        PathBuf::from,
    );
    for call in CALLS {
        let audio = read_wav(&calls.join(format!("{call}.wav")), CALL_SECONDS * 16_000);
        requests.push((call.to_owned(), audio.into()));
    }
    requests
}

fn token_ids(result: &TranscriptionResult) -> Vec<u32> {
    result.tokens.iter().map(|token| token.token_id).collect()
}

/// Levenshtein distance between two token sequences
fn edit_distance(a: &[u32], b: &[u32]) -> usize {
    let mut previous: Vec<usize> = (0..=b.len()).collect();
    for (i, left) in a.iter().enumerate() {
        let mut current = vec![i + 1; b.len() + 1];
        for (j, right) in b.iter().enumerate() {
            let substitution = previous[j] + usize::from(left != right);
            current[j + 1] = substitution.min(previous[j + 1] + 1).min(current[j] + 1);
        }
        previous = current;
    }
    previous[b.len()]
}

/// Token edits one result may differ by: a flipped token or two, so a result
/// that belongs to another request still fails
fn allowed_edits(want: &[u32]) -> usize {
    1 + want.len() / 10
}

/// Holds each result to the blocking pipeline's tokens, up to rounding
///
/// A batch's shape changes how cuBLAS orders its sums, so a token can flip
/// against a run alone; the tokens must stay within 1 % edit distance overall,
/// and each result within [`allowed_edits`]
fn assert_close(ids: &[&str], got: &[TranscriptionResult], want: &[TranscriptionResult]) {
    let mut edits = 0;
    let mut tokens = 0;
    for ((id, got), want) in ids.iter().zip(got).zip(want) {
        let want = token_ids(want);
        assert!(!want.is_empty(), "{id}: the pipeline produced no tokens");
        let distance = edit_distance(&token_ids(got), &want);
        assert!(
            distance <= allowed_edits(&want),
            "{id}: {distance} token edits against the pipeline"
        );
        if distance > 0 {
            eprintln!("{id}: {distance} token edits against the pipeline");
        }
        edits += distance;
        tokens += want.len();
    }
    assert!(
        edits * 100 <= tokens,
        "{edits} token edits over {tokens} tokens"
    );
}

/// Runs every request through the long-form pipeline one at a time, then through
/// the scheduler from 16 threads at once, short and long mixed, so windows of
/// different requests share encoder batches and decoder rows
#[test]
#[ignore = "needs a CUDA GPU, the native weights, LibriSpeech and Earnings-21; `just gpu-parity` runs it on gpu0"]
fn gpu_concurrent_requests_transcribe_like_the_pipeline() {
    const CALLERS: usize = 16;

    let requests = requests();
    let pipeline = long_form();
    let alone: Vec<TranscriptionResult> = requests
        .iter()
        .map(|(_, audio)| pipeline.run(audio).expect("alone"))
        .collect();
    let scheduler = pipeline
        .into_scheduler(SchedulerConfig::default())
        .expect("scheduler");

    // the long requests go first, so short ones join batches in progress
    let mut order: Vec<usize> = (0..requests.len()).rev().collect();
    order.rotate_left(CALLS.len() / 2);
    let queue = Mutex::new(order);
    let results = Mutex::new(vec![None; requests.len()]);
    thread::scope(|scope| {
        for _ in 0..CALLERS {
            scope.spawn(|| {
                loop {
                    let Some(index) = queue.lock().expect("queue").pop() else {
                        break;
                    };
                    let result = scheduler
                        .transcribe(Arc::clone(&requests[index].1))
                        .expect("concurrent");
                    results.lock().expect("results")[index] = Some(result);
                }
            });
        }
    });

    let results: Vec<TranscriptionResult> = results
        .into_inner()
        .expect("results")
        .into_iter()
        .map(|result| result.expect("every request ran"))
        .collect();
    let ids: Vec<&str> = requests.iter().map(|(id, _)| id.as_str()).collect();
    assert_close(&ids, &results, &alone);
}

/// Short clips submitted all at once to a scheduler over the single-input
/// pipeline, with bad inputs among them: the bad ones fail at submit, and the
/// rest transcribe like `run`
#[test]
#[ignore = "needs a CUDA GPU, the native weights and LibriSpeech; `just gpu-parity` runs it on gpu0"]
fn gpu_short_clips_transcribe_like_run() {
    let pipeline = TranscriptionPipeline::from_bundle(bundle()).expect("pipeline");
    let clips: Vec<(String, Arc<[f32]>)> = requests()
        .into_iter()
        .filter(|(_, audio)| audio.len() <= pipeline.config().max_audio_samples)
        .collect();
    let alone: Vec<TranscriptionResult> = clips
        .iter()
        .map(|(_, audio)| pipeline.run(audio).expect("run"))
        .collect();
    let too_long = vec![0.0f32; pipeline.config().max_audio_samples + 1];
    let scheduler = pipeline
        .into_scheduler(SchedulerConfig::default())
        .expect("scheduler");

    let mut pending = Vec::new();
    for (index, (_, audio)) in clips.iter().enumerate() {
        pending.push(scheduler.submit(Arc::clone(audio)).expect("submit"));
        // the bad inputs sit between good ones, so the scheduler must keep going
        if index == 1 {
            assert!(matches!(
                scheduler.submit(Vec::new()),
                Err(TranscriptionError::EmptyAudio)
            ));
            assert!(matches!(
                scheduler.submit(too_long.clone()),
                Err(TranscriptionError::AudioTooLong { .. })
            ));
        }
    }
    let results: Vec<TranscriptionResult> = pending
        .into_iter()
        .map(|pending| pending.wait().expect("transcribed"))
        .collect();
    let ids: Vec<&str> = clips.iter().map(|(id, _)| id.as_str()).collect();
    assert_close(&ids, &results, &alone);
}

/// A request dropped while it waits is cancelled without disturbing the others,
/// and requests submitted before the scheduler drops still finish
#[test]
#[ignore = "needs a CUDA GPU, the native weights, LibriSpeech and Earnings-21; `just gpu-parity` runs it on gpu0"]
fn gpu_cancelled_and_orphaned_requests() {
    let requests = requests();
    let scheduler = scheduler();
    let short = &requests[0].1;
    let long = &requests[requests.len() - 1].1;
    let want = scheduler.transcribe(Arc::clone(short)).expect("alone");
    // one clip, so a flipped token weighs more than across a set
    let want = token_ids(&want);
    let assert_same = |label: &str, got: TranscriptionResult| {
        let distance = edit_distance(&token_ids(&got), &want);
        assert!(
            distance <= allowed_edits(&want),
            "{label}: {distance} token edits"
        );
    };

    // a long request dropped at once, among short ones that must still finish
    let cancelled = scheduler.submit(Arc::clone(long)).expect("submit");
    let pending: Vec<_> = (0..8)
        .map(|_| scheduler.submit(Arc::clone(short)).expect("submit"))
        .collect();
    drop(cancelled);
    for pending in pending {
        assert_same("after a cancel", pending.wait().expect("wait"));
    }

    // the scheduler's last handle drops with requests still in flight
    let pending: Vec<_> = (0..8)
        .map(|_| scheduler.submit(Arc::clone(short)).expect("submit"))
        .collect();
    drop(scheduler);
    for pending in pending {
        assert_same("orphaned", pending.wait().expect("wait"));
    }
}

/// A full scheduler refuses `try_submit` and makes `submit` wait for room, a
/// dropped request frees its place, and every request it took still finishes
#[test]
#[ignore = "needs a CUDA GPU, the native weights, LibriSpeech and Earnings-21; `just gpu-parity` runs it on gpu0"]
fn gpu_full_scheduler_refuses_or_waits() {
    let requests = requests();
    let calls = &requests[requests.len() - CALLS.len()..];
    let pipeline = long_form();
    let alone: Vec<TranscriptionResult> = calls
        .iter()
        .map(|(_, audio)| pipeline.run(audio).expect("alone"))
        .collect();
    let config = SchedulerConfig {
        max_requests: 2,
        ..SchedulerConfig::default()
    };
    let scheduler = pipeline.into_scheduler(config).expect("scheduler");
    let audio = |index: usize| Arc::clone(&calls[index].1);

    // two calls hold both places for tens of milliseconds, far longer than the
    // next line takes
    let first = scheduler.submit(audio(0)).expect("submit");
    let second = scheduler.submit(audio(1)).expect("submit");
    assert!(matches!(
        scheduler.try_submit(audio(2)),
        Err(TranscriptionError::SchedulerFull)
    ));

    drop(second);
    let third = scheduler
        .try_submit(audio(2))
        .expect("a dropped request frees its place");

    // full again, so this returns only once the first or third has finished
    let second = scheduler.submit(audio(1)).expect("submit waits for room");
    assert!(first.is_finished() || third.is_finished());

    let results = [first, second, third].map(|pending| pending.wait().expect("wait"));
    let ids: Vec<&str> = calls.iter().map(|(id, _)| id.as_str()).collect();
    assert_close(&ids, &results, &alone);
}

/// Requests the planner rejects fail at submit, and do not stop the scheduler
#[test]
#[ignore = "needs a CUDA GPU and the native weights; `just gpu-parity` runs it on gpu0"]
fn gpu_rejected_requests_fail_at_submit() {
    let scheduler = scheduler();
    assert!(matches!(
        scheduler.submit(Vec::new()),
        Err(TranscriptionError::EmptyAudio)
    ));
    let mut audio = vec![0.0f32; 60 * 16_000];
    audio[100] = f32::NAN;
    assert!(matches!(
        scheduler.submit(audio),
        Err(TranscriptionError::InvalidAudioSample { index: 100 })
    ));
    // the scheduler still runs after the rejections
    scheduler
        .transcribe(vec![0.0f32; 16_000])
        .expect("a valid request after rejected ones");
}
