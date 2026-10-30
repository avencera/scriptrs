//! The background transcription queue on the CUDA backend, against the blocking
//! pipeline on the same inputs
//!
//! The tests need a CUDA GPU, the native weights (`SCRIPTRS_NATIVE_MODELS`, default
//! `/shared/models/scriptrs-native`) and LibriSpeech test-clean
//! (`SCRIPTRS_LIBRISPEECH_MANIFEST`); `just gpu-parity` runs them on gpu0

use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use scriptrs::{
    ModelBundle, ParakeetDeployment, QueueConfig, QueueError, QueueSender,
    QueuedTranscriptionRequest, QueuedTranscriptionResult, TranscriptionError,
    TranscriptionPipeline, TranscriptionResult,
};

/// Utterances taken from the manifest, spread over its lengths
const UTTERANCES: usize = 48;

fn pipeline() -> TranscriptionPipeline {
    let root = std::env::var_os("SCRIPTRS_NATIVE_MODELS").map_or_else(
        || PathBuf::from("/shared/models/scriptrs-native"),
        PathBuf::from,
    );
    let bundle = ModelBundle::from_native_dir(root, ParakeetDeployment::v3());
    TranscriptionPipeline::from_bundle(bundle).expect("pipeline")
}

/// Every `n`th LibriSpeech test-clean utterance as `(id, samples)`
fn utterances() -> Vec<(String, Vec<f32>)> {
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
    rows.iter()
        .step_by(step)
        .take(UTTERANCES)
        .map(|row| {
            let id = row["id"].as_str().expect("id").to_owned();
            let path = row["audio_filepath"].as_str().expect("audio path");
            (id, read_wav(Path::new(path)))
        })
        .collect()
}

fn read_wav(path: &Path) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("wav");
    reader
        .samples::<i16>()
        .map(|sample| f32::from(sample.expect("sample")) / 32_768.0)
        .collect()
}

/// Pushes every request in order, waiting out a full queue
fn push_all(sender: &QueueSender, requests: Vec<QueuedTranscriptionRequest>) {
    for mut request in requests {
        loop {
            match sender.try_push(request) {
                Ok(_) => break,
                Err(QueueError::Full(rejected)) => {
                    request = rejected;
                    thread::sleep(Duration::from_millis(1));
                }
                Err(error) => panic!("push failed: {error}"),
            }
        }
    }
}

fn token_ids(result: &TranscriptionResult) -> Vec<u32> {
    result.tokens.iter().map(|token| token.token_id).collect()
}

/// At batch 1 every input runs alone, so each result must equal the blocking
/// pipeline's exactly; a two-slot queue makes the producer wait on a full queue,
/// and bad inputs fail alone without stopping the queue
#[test]
#[ignore = "needs a CUDA GPU, the native weights and LibriSpeech; `just gpu-parity` runs it on gpu0"]
fn gpu_queue_at_batch_one_matches_the_pipeline() {
    let pipeline = pipeline();
    let utterances = utterances();
    let expected: Vec<TranscriptionResult> = utterances
        .iter()
        .map(|(_, audio)| pipeline.run(audio).expect("run"))
        .collect();
    let too_long = vec![0.0; pipeline.deployment().hard_sample_max() + 1];
    let queue = QueueConfig {
        capacity: 2,
        max_batch: 1,
    };
    let config = *pipeline.config();
    let (sender, mut receiver) = pipeline
        .into_queued_with_config(config, queue)
        .expect("queue");
    assert_eq!(sender.capacity(), 2);

    // the bad inputs sit between good ones, so the queue must keep going past them
    let mut requests: Vec<QueuedTranscriptionRequest> = utterances
        .iter()
        .map(|(id, audio)| QueuedTranscriptionRequest::new(id.clone(), audio.clone()))
        .collect();
    requests.insert(1, QueuedTranscriptionRequest::new("empty", Vec::new()));
    requests.insert(3, QueuedTranscriptionRequest::new("too-long", too_long));
    let order: Vec<String> = requests
        .iter()
        .map(|request| request.id().to_owned())
        .collect();
    let producer = thread::spawn(move || push_all(&sender, requests));

    let mut results: Vec<QueuedTranscriptionResult> = Vec::new();
    loop {
        match receiver.recv() {
            Ok(result) => results.push(result),
            Err(QueueError::Closed) => break,
            Err(error) => panic!("receive failed: {error}"),
        }
    }
    producer.join().expect("producer");
    // the end state repeats rather than blocking
    assert!(matches!(receiver.recv(), Err(QueueError::Closed)));
    assert!(matches!(receiver.try_recv(), Err(QueueError::Closed)));

    let ids: Vec<&str> = results.iter().map(|result| result.id.as_str()).collect();
    assert_eq!(ids, order, "results come back in submission order");
    let job_ids: Vec<u64> = results
        .iter()
        .map(|result| result.job_id.as_u64())
        .collect();
    assert!(
        job_ids.windows(2).all(|pair| pair[0] < pair[1]),
        "one sender's job ids increase: {job_ids:?}"
    );

    let mut good =
        results
            .into_iter()
            .filter(|result| match (result.id.as_str(), &result.result) {
                ("empty", Err(TranscriptionError::EmptyAudio)) => false,
                ("too-long", Err(TranscriptionError::AudioTooLong { .. })) => false,
                (id, Err(error)) => panic!("{id} failed: {error}"),
                _ => true,
            });
    for ((id, _), expected) in utterances.iter().zip(&expected) {
        let queued = good.next().expect("a result for every utterance");
        assert_eq!(&queued.id, id);
        let result = queued.result.expect("transcribed");
        assert!(!result.tokens.is_empty(), "{id}: no tokens");
        assert_eq!(&result, expected, "{id}: differs from the pipeline");
    }
    assert!(good.next().is_none(), "unexpected extra results");
}

/// Batches of up to 8, with two producers pushing at once: every request comes
/// back once, under its own job id, with the tokens the blocking pipeline gives
/// up to rounding
///
/// A batch's shape changes how cuBLAS orders its sums, so a token can flip
/// against the batch-1 run (4 of 2620 test-clean transcripts at batch 8); the
/// tokens must stay within 1 % edit distance overall
#[test]
#[ignore = "needs a CUDA GPU, the native weights and LibriSpeech; `just gpu-parity` runs it on gpu0"]
fn gpu_batched_queue_transcribes_like_the_pipeline() {
    let pipeline = pipeline();
    let utterances = utterances();
    let expected: Vec<TranscriptionResult> = utterances
        .iter()
        .map(|(_, audio)| pipeline.run(audio).expect("run"))
        .collect();
    let config = *pipeline.config();
    let queue = QueueConfig {
        max_batch: 8,
        ..QueueConfig::default()
    };
    let (sender, receiver) = pipeline
        .into_queued_with_config(config, queue)
        .expect("queue");

    let (evens, odds): (Vec<_>, Vec<_>) = utterances
        .iter()
        .enumerate()
        .map(|(index, (id, audio))| {
            (
                index,
                QueuedTranscriptionRequest::new(id.clone(), audio.clone()),
            )
        })
        .partition(|(index, _)| index % 2 == 0);
    let producers: Vec<_> = [evens, odds]
        .into_iter()
        .map(|half| {
            let sender = sender.clone();
            let requests = half.into_iter().map(|(_, request)| request).collect();
            thread::spawn(move || push_all(&sender, requests))
        })
        .collect();
    drop(sender);

    let results: Vec<QueuedTranscriptionResult> = receiver
        .into_iter()
        .map(|result| result.expect("queue"))
        .collect();
    for producer in producers {
        producer.join().expect("producer");
    }

    let mut job_ids: Vec<u64> = results
        .iter()
        .map(|result| result.job_id.as_u64())
        .collect();
    job_ids.sort_unstable();
    job_ids.dedup();
    assert_eq!(job_ids.len(), utterances.len(), "job ids are unique");
    assert_eq!(results.len(), utterances.len());

    let mut edits = 0;
    let mut tokens = 0;
    for queued in results {
        let index = utterances
            .iter()
            .position(|(id, _)| *id == queued.id)
            .expect("a known id");
        let result = queued.result.expect("transcribed");
        let expected = token_ids(&expected[index]);
        let distance = edit_distance(&token_ids(&result), &expected);
        if distance > 0 {
            eprintln!("{}: {distance} token edits against the pipeline", queued.id);
        }
        edits += distance;
        tokens += expected.len();
    }
    assert!(tokens > 0, "the pipeline produced no tokens");
    assert!(
        edits * 100 <= tokens,
        "{edits} token edits over {tokens} tokens"
    );
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

/// Dropping the receiver stops the worker, both while it is idle, waiting on a
/// sender that is still alive, and while it is transcribing; pushes then report
/// that the worker is gone
#[test]
#[ignore = "needs a CUDA GPU, the native weights and LibriSpeech; `just gpu-parity` runs it on gpu0"]
fn gpu_dropping_the_receiver_stops_the_worker() {
    let (id, audio) = utterances().swap_remove(0);

    // idle: nothing was ever pushed, so only the receiver's drop can stop it
    let (sender, receiver) = pipeline().into_queued().expect("queue");
    drop(receiver);
    thread::sleep(Duration::from_secs(2));
    assert!(
        matches!(
            sender.try_push(QueuedTranscriptionRequest::new(id.clone(), audio.clone())),
            Err(QueueError::WorkerGone)
        ),
        "the idle worker kept running"
    );

    // busy: the receiver goes away while requests are queued
    let (sender, receiver) = pipeline().into_queued().expect("queue");
    for _ in 0..4 {
        sender
            .try_push(QueuedTranscriptionRequest::new(id.clone(), audio.clone()))
            .expect("push");
    }
    drop(receiver);
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        match sender.try_push(QueuedTranscriptionRequest::new(id.clone(), audio.clone())) {
            Err(QueueError::WorkerGone) => break,
            Ok(_) | Err(QueueError::Full(_)) => {
                assert!(Instant::now() < deadline, "the busy worker kept running");
                thread::sleep(Duration::from_millis(10));
            }
            Err(error) => panic!("push failed: {error}"),
        }
    }
}
