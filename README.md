# scriptrs

Fast [NVIDIA Parakeet TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) speech-to-text in Rust.

- Parakeet TDT 0.6B v2 (English) and v3 (25 European languages)
- Core ML on macOS
- Native CUDA on Linux, with no ONNX Runtime or PyTorch (`cuda` feature)
- Long-form transcription of hour-long recordings, with optional VAD
- Token-level timestamps and confidences

On an RTX 5090, the CUDA backend transcribes LibriSpeech test-clean at 1534x real
time one utterance at a time, 3.1x as fast as NVIDIA NeMo and 6.1x as fast as
parakeet-rs, at the same word error rate. Batched, it is 1.6x as fast as NeMo at
batch 8 and about even at batch 32 (5158x against 5242x), and the scheduler's
own batch sizes run the set at 5451x. On 31.6 h of
multilingual speech the default FP16 mode scores the same WER as the FP32 mode,
which matches NeMo. On six harder English sets it has a lower WER than Whisper
large-v3 at about 20x its speed. See [CUDA](#cuda) for the numbers.

`scriptrs` is early: the public API is still moving.

The base crate exposes a single-chunk `TranscriptionPipeline`. Fast long-audio chunking lives behind the `long-form` feature via `LongFormTranscriptionPipeline`. VAD-backed speech region planning lives behind the `vad` feature, which also enables `long-form`.

## Current scope

- Base pipeline for short audio
- Optional fast long-form pipeline with overlap chunking
- Optional VAD-backed long-form region planning
- Scheduler for many inputs or concurrent requests of any length, batched on CUDA
- Native CoreML inference on macOS
- CUDA inference on Linux (`cuda` feature)
- Streaming transcription of many live streams at once with Nemotron Speech
  Streaming EN 0.6B and Nemotron 3.5 ASR Streaming 0.6B on CUDA
- Hugging Face download support with optional local model loading
- Python package on PyPI, with wheels for macOS arm64 and Linux x86_64

## What it does not do yet

- Windows support
- Other ASR models
- Streaming on Core ML
- Stable public guarantees around model layout or long-form behavior

## Install

```toml
[dependencies]
scriptrs = "0.4"
```

For native CUDA on Linux:

```toml
[dependencies]
scriptrs = { version = "0.4", features = ["cuda"] }
```

For fast long-form transcription:

```toml
[dependencies]
scriptrs = { version = "0.4", features = ["long-form"] }
```

For VAD-backed long-form transcription:

```toml
[dependencies]
scriptrs = { version = "0.4", features = ["vad"] }
```

## Python

```sh
pip install scriptrs
```

```python
import scriptrs

pipeline = scriptrs.Pipeline.from_pretrained("parakeet/v2")  # or "parakeet/v3" for 25 languages
print(pipeline.transcribe_file("meeting.mp3").text)
```

One package carries both models; the model is picked when the pipeline loads,
and only that model downloads. The wheel includes long-form transcription, VAD,
the batching queue, and audio file decoding. See the
[Python README](crates/scriptrs-python/README.md).

## Model downloads

With the default `online` feature, `scriptrs` can resolve models automatically:

- Core ML bundles come from `avencera/scriptrs-models`
- native CUDA weights come from
  [`avencera/parakeet-tdt-0.6b-v2-cuda`](https://huggingface.co/avencera/parakeet-tdt-0.6b-v2-cuda)
  and [`avencera/parakeet-tdt-0.6b-v3-cuda`](https://huggingface.co/avencera/parakeet-tdt-0.6b-v3-cuda),
  and the native Silero VAD from
  [`avencera/silero-vad-v5-safetensors`](https://huggingface.co/avencera/silero-vad-v5-safetensors)

You can force a local bundle when needed:

- `SCRIPTRS_MODELS_DIR=/path/to/models` forces a local bundle

Each download uses an immutable snapshot pinned in the release, such as
`63f6f9cbce43e5561c219229896f86f20c0eb660` of `avencera/scriptrs-models`.

## Local model layout

If you want to use `from_dir(...)` or `SCRIPTRS_MODELS_DIR`, the local bundle should look like this:

```text
models/
  parakeet-v2/
    encoder.mlmodelc/
    decoder.mlmodelc/
    joint-decision.mlmodelc/
    vocab.txt
  parakeet-v3/
    encoder.mlmodelc/
    decoder.mlmodelc/
    joint-decision.mlmodelc/
    parakeet_vocab.json
```

Use `ParakeetDeployment::v2()` or `ParakeetDeployment::v3()` to select a layout. v2 remains the default for `from_dir` and `from_pretrained`.

With `vad`, add:

```text
models/
  vad/
    silero-vad.mlmodelc/
```


## CUDA

The `cuda` feature runs Parakeet on an NVIDIA GPU with no ONNX Runtime or
PyTorch. It uses NVIDIA's weights unchanged, and its word error rate matches
NeMo's. It supports the v3 language filter and, with `vad`, VAD-backed long-form
planning.

It loads native weights rather than Core ML models. With the default `online`
feature, `from_pretrained_native` downloads them from a pinned snapshot of
[`avencera/parakeet-tdt-0.6b-v2-cuda`](https://huggingface.co/avencera/parakeet-tdt-0.6b-v2-cuda)
or [`avencera/parakeet-tdt-0.6b-v3-cuda`](https://huggingface.co/avencera/parakeet-tdt-0.6b-v3-cuda):

```rust,ignore
use scriptrs::{ModelBundle, ParakeetDeployment, TranscriptionPipeline};

let bundle = ModelBundle::from_pretrained_native(ParakeetDeployment::v3())?;
let pipeline = TranscriptionPipeline::from_bundle(bundle)?;
let result = pipeline.run(&audio)?;
```

With `vad`, `from_pretrained_native_long_form` also downloads the Silero VAD
weights from
[`avencera/silero-vad-v5-safetensors`](https://huggingface.co/avencera/silero-vad-v5-safetensors).
To manage the files yourself, `dev-tools/nemo/export_native.py` writes
them from the NeMo checkpoints in this layout, and `from_native_dir` loads them:

```text
models/
  parakeet-v2/
    model.safetensors
    vocab.txt
  parakeet-v3/
    model.safetensors
    parakeet_vocab.json
  vad/
    silero-vad-v5.safetensors    # with `vad`
```

```rust,ignore
let bundle = ModelBundle::from_native_dir("models", ParakeetDeployment::v3());
```

A native bundle accepts single inputs up to 40 s; longer audio goes through
`LongFormTranscriptionPipeline`. For native bundles, long-form defaults to 30 s
windows with 8 s overlap, run 16 at a time through one model. Longer windows or
shorter overlaps lose accuracy on long recordings.

On an RTX 5090, Parakeet v3 on LibriSpeech test-clean (5.4 h), one utterance at a
time:

| | Speed (x real time) | vs NeMo | WER |
|---|---|---|---|
| scriptrs | 1534x | 3.1x | 1.93 % |
| NVIDIA NeMo | 490x | 1x | 1.93 % |
| parakeet-rs 0.4.3 (ONNX Runtime) | 253x | 0.52x | 1.93 % |

Through the [scheduler](#many-inputs-and-concurrent-requests), with all 2620 clips
submitted at once, the same set runs at 5451x. Batching changes one of the 2620
transcripts by a comma, and WER stays 1.93 %.

NeMo gains more from batching, because large batches are dominated by the
encoder's matrix multiplies rather than launch overhead. Against NeMo in bf16,
which like the FP16 default multiplies 16-bit values with FP32 sums, with the
scheduler held to the same batch sizes (`cuda_bench --scheduler --batch N`):

| Batch | scriptrs (scheduler) | NVIDIA NeMo bf16 | vs NeMo |
|---|---|---|---|
| 1 | 1739x | 490x | 3.5x |
| 8 | 4793x | 2958x | 1.6x |
| 32 | 5158x | 5242x | 0.98x |

WER is 1.92–1.93 % for both at every batch size.

The multilingual check is nearly 6 times the size of LibriSpeech: FLEURS and MLS,
31.6 h over ten language sets in six languages (de, es, fr, it, nl, pt), one
utterance at a time with automatic language selection. The FP32 mode matches NeMo
layer by layer and gives NeMo's exact text in the parity tests, and the FP16
default scores the same WER as FP32. Speed is for the FP16 default:

| Set | Speed (x real time) | WER, FP16 | WER, FP32 |
|---|---|---|---|
| FLEURS de, es, fr, it, nl, pt (15.8 h) | 1839x | 4.57 % | 4.57 % |
| MLS es, fr, it, pt (15.8 h) | 1921x | 6.25 % | 6.26 % |

On all 44 Earnings-21 calls (39 h), long-form with the defaults runs at 4011x with
8.24 % pooled WER; NeMo transcribing each whole call with local attention scores
9.58 %.

On harder English audio, long-form with the defaults and English selected has a
lower WER than Whisper large-v3 (run through WhisperX 3.8.6 in float16) on every
set, and runs 16x to 22x as fast:

| Set | Recordings | Hours | scriptrs WER | Whisper WER | scriptrs speed | Whisper speed | vs Whisper |
|---|---:|---:|---:|---:|---:|---:|---:|
| Earnings-21 earnings calls | 44 | 39.3 | **8.2 %** | 9.7 % | **4011x** | 186x | 21.6x |
| AMI meetings, headset mix | 34 | 18.7 | **21.5 %** | 23.8 % | **3917x** | 182x | 21.5x |
| AMI meetings, far-field microphone | 34 | 18.7 | **23.2 %** | 28.3 % | **3912x** | 199x | 19.6x |
| EdAcc accented conversations | 40 | 18.2 | **16.0 %** | 19.2 % | **3949x** | 201x | 19.7x |
| CHiME-6 dinner parties | 4 | 9.7 | **44.0 %** | 44.7 % | **3942x** | 176x | 22.4x |
| AppTek call-center calls, 14 accents | 56 | 9.4 | **11.2 %** | 12.6 % | **3512x** | 224x | 15.7x |
| Mean over sets | 212 | 114 | **20.7 %** | 23.1 % | **3918x** | 191x | 20.5x |

WER is pooled over each set's recordings, both sides through Whisper's English
normalizer. Speed is x real time for transcription alone on one RTX 5090, with
models loaded and audio in memory; WhisperX runs its VAD with batch size 16. The
last row's speed is over all 114 hours.

With `vad`, Silero VAD v5 runs on the CPU, at about 0.3 s per hour of audio. On a
GPU each step of its recurrence costs more in launches than the whole step costs on
the CPU. VAD planning helps when most of a recording is not speech, and costs
accuracy and speed on dense or far-field speech:

| Set | Overlap WER | VAD WER | Overlap speed | VAD speed |
|---|---:|---:|---:|---:|
| Sparse speech, 24 % speech (4.2 h) | 14.8 % | **2.3 %** | 3430x | **3990x** |
| Earnings-21 earnings calls | **8.2 %** | 8.4 % | **3630x** | 2030x |
| AMI meetings, headset mix | 21.5 % | **21.1 %** | **3480x** | 1800x |
| AMI meetings, far-field microphone | **23.2 %** | 26.0 % | **3480x** | 1790x |
| CHiME-6 dinner parties | **44.0 %** | 54.0 % | **3490x** | 1970x |

The sparse set is LibriSpeech test-clean utterances separated by 15 to 75 s of
film music, effects, ambience and low-level noise. VAD runs use the default 0.5
speech threshold and pad each speech region by 0.2 s on both sides, including both
sides of a split inside a region longer than the model window. Silero is slow to
react to a soft first word, such as the "a" in "a cold", and 0.1 s of padding cut
it off. 0.2 s lowered WER on every set; larger padding cost AMI headset accuracy.
The speed columns were measured with 0.1 s padding; in the padding sweep, 0.2 s
was no slower.

By default the encoder layers' linear weights are stored in FP16, which has TF32's
10-bit mantissa, and their matrix multiplies run on FP16 tensor cores with FP32
accumulation. Use this default: it is the fastest and scores the same WER as FP32.

`Precision::Tf32` and `Precision::Fp32` are for compatibility, when you need to
keep matching transcripts you already produced at that precision. `Precision::Tf32`
keeps the weights in FP32 and uses TF32 tensor cores on Ampere and newer GPUs, or
FP32 on older ones. `Precision::Fp32` matches NeMo's FP32 decode token for token on
any GPU:

```rust,ignore
use scriptrs::{ModelBundle, ParakeetDeployment, Precision};

let bundle = ModelBundle::from_native_dir("models", ParakeetDeployment::v3())
    .with_precision(Precision::Fp32);
```

On an RTX 5090 with Parakeet v3 (speed in x real time, then WER):

| | FP16 (default) | TF32 | FP32 |
|---|---|---|---|
| LibriSpeech test-clean, one utterance at a time | 1534x, 1.93 % | 1179x, 1.93 % | 848x, 1.93 % |
| LibriSpeech test-clean, scheduler, all clips at once | 5451x, 1.93 % | 3423x, 1.93 % | — |
| Earnings-21, long-form | 4173x, 8.24 % | 2772x, 8.24 % | — |
| FLEURS de, es, fr, it, nl, pt (15.8 h) | 1839x, 4.57 % | 1450x, 4.57 % | 966x, 4.57 % |
| MLS es, fr, it, pt (15.8 h) | 1921x, 6.25 % | 1524x, 6.26 % | 996x, 6.26 % |

FLEURS and MLS run one utterance at a time with automatic language selection and
are scored with Whisper's basic normalizer. On each of their ten language sets,
FP16 is within one word error of TF32. On LibriSpeech one utterance at a time,
FP16 gives the same transcripts as TF32. Batching changes a few transcripts in
either precision, so the scheduled FP16 and TF32 transcripts differ slightly.

The `low-precision` feature adds FP8, INT8 and NVFP4 modes for offline
transcription, and the weight files that store them. Most users should leave it
off. None of the three is faster than FP16 one utterance at a time, and they save
little GPU memory: 0.4 to 0.7 GB of FP16's peak of about 7 GB on an RTX 5090. FP8
and INT8 only halve the download at FP16's WER, and NVFP4 is faster only in large
batches on Blackwell GPUs, at a WER cost. The `Precision` docs give the GPU
requirements and measurements.

## Usage

### Short audio

Use the base pipeline when your audio already fits in a single Parakeet chunk.

With the default `online` feature, `from_pretrained()` is the intended path:

```rust
use scriptrs::TranscriptionPipeline;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio: Vec<f32> = load_mono_16khz_audio();
    let pipeline = TranscriptionPipeline::from_pretrained()?;
    let result = pipeline.run(&audio)?;

    println!("{}", result.text);
    Ok(())
}

fn load_mono_16khz_audio() -> Vec<f32> {
    Vec::new()
}
```

If the input is too long for the base pipeline, it returns `AudioTooLong`.

If you want to use a local bundle instead:

```rust
use scriptrs::TranscriptionPipeline;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio: Vec<f32> = load_mono_16khz_audio();
    let pipeline = TranscriptionPipeline::from_dir("models")?;
    let result = pipeline.run(&audio)?;

    println!("{}", result.text);
    Ok(())
}

fn load_mono_16khz_audio() -> Vec<f32> {
    Vec::new()
}
```

### Long audio

Enable `long-form` if you want `scriptrs` to own long-audio chunking internally and you care most about speed on clean, dense speech.

```rust
use scriptrs::LongFormTranscriptionPipeline;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio: Vec<f32> = load_mono_16khz_audio();
    let pipeline = LongFormTranscriptionPipeline::from_pretrained()?;
    let result = pipeline.run(&audio)?;

    println!("{}", result.text);
    Ok(())
}

fn load_mono_16khz_audio() -> Vec<f32> {
    Vec::new()
}
```

`LongFormConfig` defaults to the fast overlap-chunking path with `4` workers, each loading its own model. On native CUDA bundles, windows instead run `batch_size` at a time (16 by default) through the pipeline's one model, and the workers prepare windows on the CPU ahead of it; Core ML bundles ignore `batch_size`. `LongFormConfig::for_bundle(&bundle)` gives a bundle's defaults to start from when you change one setting. You can tune the worker count when you want less or more parallelism:

```rust
use scriptrs::{LongFormConfig, LongFormTranscriptionPipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio: Vec<f32> = load_mono_16khz_audio();
    let pipeline = LongFormTranscriptionPipeline::from_pretrained()?;
    let config = LongFormConfig {
        worker_count: 2,
        ..LongFormConfig::default()
    };
    let result = pipeline.run_with_config(&audio, &config)?;

    println!("{}", result.text);
    Ok(())
}

fn load_mono_16khz_audio() -> Vec<f32> {
    Vec::new()
}
```

Enable `vad` when you want VAD-backed speech region planning for sparse speech, long silences, or recordings with a lot of non-speech audio. Windows with a speech probability of at least `config.vad.default_threshold` (0.5 by default) count as speech:

```rust
use scriptrs::{LongFormConfig, LongFormMode, LongFormTranscriptionPipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio: Vec<f32> = load_mono_16khz_audio();
    let pipeline = LongFormTranscriptionPipeline::from_pretrained()?;
    let config = LongFormConfig {
        mode: LongFormMode::Vad,
        ..LongFormConfig::default()
    };
    let result = pipeline.run_with_config(&audio, &config)?;

    println!("{}", result.text);
    Ok(())
}

fn load_mono_16khz_audio() -> Vec<f32> {
    Vec::new()
}
```

### Streaming (CUDA)

`StreamingModel` runs NVIDIA's cache-aware [Nemotron Speech Streaming EN
0.6B](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) and
multilingual [Nemotron 3.5 ASR Streaming
0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) on live audio,
with the `cuda` feature. Push audio to each `Stream` as it arrives and call
`process`: it encodes every chunk the streams have audio for, one
chunk from each stream per batch, and keeps each stream's encoder caches and
decoder state on the GPU between calls. `lookahead` picks the chunk size the model
was trained with: for the English model, 0, 1, 6 or 13 encoder frames of
lookahead give chunks of 80, 160, 560 or 1120 ms, trading latency for accuracy;
Nemotron 3.5 takes 0, 3, 6 or 13. Nemotron 3.5 streams open with
`open_stream_in` in one of `languages()`, such as `de-DE`, or `auto` to let the
model pick; `open_stream` is for the English model.

`dev-tools/nemo/export_native.py --only nemotron-en` (or `--only nemotron-3.5`)
writes the weights:

```rust,ignore
use scriptrs::{StreamingConfig, StreamingModel};

let model = StreamingModel::from_native_dir(
    "models/nemotron-speech-streaming-en-0.6b",
    &StreamingConfig::new(1),
)?;
let mut stream = model.open_stream()?;
for piece in live_audio() {
    stream.push(&piece)?;
    model.process(&mut [&mut stream])?;
    println!("{}", stream.transcript().text);
}
stream.finish()?;
model.process(&mut [&mut stream])?;
println!("{}", stream.transcript().text);
```

`StreamingConfig` also sets the most streams open at once (`max_streams`, 64 by
default; each holds about 15 MB of GPU cache) and the most chunks per batch
(`max_batch`, 32). `examples/stream_bench.rs` measures per-chunk latency and
throughput on LibriSpeech test-clean.

`precision` defaults to FP16 encoder GEMMs, which run 1.4 to 1.9 times as fast as
FP32 and change a few transcripts, with WER within 0.003 points. With
`Precision::Fp32`, streams give NeMo's streaming tokens exactly: at every chunk
size of the English model, and for Nemotron 3.5 in `en-US` at 320 and 1120 ms, the
sizes checked.

On an RTX 5090, Nemotron Speech Streaming EN on LibriSpeech test-clean (5.4 h),
one stream, audio pushed a chunk at a time as fast as it is processed. Each cell
is the mean time per chunk, then speed in x real time. scriptrs times the mel,
encoder and decode of each chunk; NeMo computes the mel up front and times the
rest. [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp) runs its
8-bit `Q8_0` model with batching off:

| Chunk (lookahead) | scriptrs FP32 | scriptrs FP16 | NeMo-Speech.cpp | NeMo | WER |
|---|---|---|---|---|---|
| 80 ms (0) | 2.8 ms, 28x | | | 10.5 ms, 8x | 3.03 % |
| 160 ms (1) | 4.0 ms, 40x | 2.7 ms, 58x | **1.7 ms, 88x** | 10.9 ms, 15x | 2.68 % |
| 560 ms (6) | 4.6 ms, 118x | | | 11.1 ms, 49x | 2.48 % |
| 1120 ms (13) | 5.6 ms, 187x | 3.0 ms, **345x** | **2.9 ms**, 275x | 12.4 ms, 84x | 2.32 % |

The WER column is NeMo's and both scriptrs precisions'; NeMo-Speech.cpp's 8-bit
model scores 2.52 % at 160 ms and 2.43 % at 1120 ms.

With many streams at 160 ms chunks, each cell is total speed in x real time, then
the 99th-percentile time per chunk. The scriptrs bench steps every stream once per
round and batches them all together (`max_batch` set to the stream count);
NeMo-Speech.cpp runs a thread per stream through its batcher, which took about
18 ms per chunk even with one stream:

| Streams | scriptrs FP32 | scriptrs FP16 | NeMo-Speech.cpp |
|---|---|---|---|
| 8 | 241x, 5.3 ms | **414x, 3.1 ms** | 65x, 23.6 ms |
| 16 | 479x, 5.3 ms | **741x, 3.4 ms** | 123x, 25.8 ms |
| 32 | 795x, 10.2 ms | **1163x, 7.0 ms** | 222x, 27.9 ms |
| 64 | 1038x, 13.5 ms | **1573x, 9.1 ms** | 351x, 39.8 ms |
| 128 | 1135x, 21.1 ms | **1884x, 13.0 ms** | 427x, 89.1 ms |
| 256 | 1305x, 33.3 ms | **2069x, 21.4 ms** | 511x, 143.9 ms |

At every stream count, scriptrs FP32 gives NeMo's WER of 2.68 %; FP16 stays
within 0.002 points of it.

### Many inputs and concurrent requests

To transcribe many inputs, or to serve many callers at once from one model, move
the pipeline behind a `TranscriptionScheduler`. Any number of threads submit audio
of any length, short clips and hour-long recordings alike, and each gets its own
result back. A long request is cut into the same windows as
`LongFormTranscriptionPipeline::run` and merged the same way:

```rust,no_run
use std::sync::Arc;

use scriptrs::{LongFormTranscriptionPipeline, SchedulerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pipeline = LongFormTranscriptionPipeline::from_dir("models")?;
    let scheduler = pipeline.into_scheduler(SchedulerConfig::default())?;

    // the scheduler is cheap to clone; each request handler takes its own
    let handlers: Vec<_> = load_requests()
        .into_iter()
        .map(|audio| {
            let scheduler = scheduler.clone();
            std::thread::spawn(move || scheduler.transcribe(audio))
        })
        .collect();
    for handler in handlers {
        println!("{}", handler.join().expect("handler panicked")?.text);
    }
    Ok(())
}

fn load_requests() -> Vec<Arc<[f32]>> {
    Vec::new()
}
```

`submit` returns a `PendingTranscription` to `wait` on later, so one thread can
submit a whole set of files and then collect the results in order:

```rust,no_run
use scriptrs::{SchedulerConfig, TranscriptionPipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pipeline = TranscriptionPipeline::from_dir("models")?;
    let scheduler = pipeline.into_scheduler(SchedulerConfig::default())?;

    let clips = load_clips();
    let pending = clips
        .iter()
        .map(|(_, audio)| scheduler.submit(audio.clone()))
        .collect::<Result<Vec<_>, _>>()?;
    for ((id, _), pending) in clips.iter().zip(pending) {
        println!("{id}: {}", pending.wait()?.text);
    }
    Ok(())
}

fn load_clips() -> Vec<(String, Vec<f32>)> {
    Vec::new()
}
```

Dropping a `PendingTranscription` cancels the request's windows that have not
started; windows already decoding finish and are discarded. Empty or overlong
audio fails at `submit` without affecting other requests, as does audio with a
non-finite sample under the long-form pipeline.
`TranscriptionPipeline::into_scheduler` takes inputs up to the single-input
limit, as `run` does.

The scheduler holds at most `SchedulerConfig::max_requests` unfinished requests
(1024 by default). Past that, `submit` waits for one to finish, which paces a
loop like the one above, and `try_submit` returns
`TranscriptionError::SchedulerFull` at once, which a server can turn into a busy
reply:

```rust,no_run
use scriptrs::{SchedulerConfig, TranscriptionError, TranscriptionPipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pipeline = TranscriptionPipeline::from_dir("models")?;
    let config = SchedulerConfig {
        max_requests: 256,
        ..SchedulerConfig::default()
    };
    let scheduler = pipeline.into_scheduler(config)?;

    match scheduler.try_submit(vec![0.0; 16_000]) {
        Ok(pending) => println!("{}", pending.wait()?.text),
        Err(TranscriptionError::SchedulerFull) => println!("busy, try again later"),
        Err(error) => return Err(error.into()),
    }
    Ok(())
}
```

On CUDA the scheduler batches windows across requests. Windows are sorted into
length buckets, and the encoder runs a bucket's waiting windows together, whichever
requests they came from. The decoder works on up to `decoder_rows` windows at once
(64 by default), and a row that finishes takes the next encoded window between
decode steps instead of waiting for the rest of its batch. Batches grow and shrink
with the work waiting: an encoder batch pads only to the next power of two
windows, and a decode step runs only the rows in use, so a lone request costs
about what `run` does. Core ML bundles run the windows one at a time.

On an RTX 5090 with the defaults, each client below sends its next request as
soon as its last one returns. One at a time is the same pipeline's `run` serving
requests in arrival order, as a server holding one model behind a lock would:

| Workload | Clients | Scheduler | p50 / p95 latency | One at a time | p50 / p95 latency |
|---|---:|---:|---:|---:|---:|
| LibriSpeech test-clean (2620 clips, 5.4 h) | 1 | 1328x | 5 / 10 ms | **1422x** | **5 / 9 ms** |
| | 8 | **2415x** | **25 / 38 ms** | 1474x | 39 / 57 ms |
| | 32 | **3309x** | **68 / 126 ms** | 1475x | 157 / 210 ms |
| | 128 | **4531x** | **197 / 353 ms** | 1439x | 645 / 789 ms |
| Earnings-21 (44 calls, 39.3 h) | 1 | **4292x** | **0.76 / 1.10 s** | 4024x | 0.84 / 1.32 s |
| | 4 | **4661x** | **2.57 / 3.77 s** | 3993x | 3.08 / 4.30 s |
| | 44 | **4435x** | **16.2 / 30.2 s** | 4123x | 19.4 / 32.5 s |
| Both sets, calls spread among the clips | 64 | **4615x** | **clips 0.80 / 1.43 s, calls 1.67 / 2.30 s** | 3368x | clips 1.10 / 1.61 s, calls 1.86 / 2.39 s |

A lone short request runs about 7 % slower than `run`, and a lone long request
runs faster, since its windows share batches. A batch's shape changes how cuBLAS
orders its sums, so a request's transcript can differ in a rare token with the
requests beside it. Against `run`, a lone request gives the same LibriSpeech
transcripts and changes 49 of the 365,842 Earnings-21 words, and with every clip
submitted at once one of the 2620 LibriSpeech transcripts changes by a comma; WER
stays 8.24 % on Earnings-21 and 1.93 % on LibriSpeech. `examples/scheduler_bench.rs`
and `examples/cuda_bench.rs --scheduler` measure this.

## Example

A small WAV example is included:

```bash
cargo run --example transcribe_wav -- --audio /path/to/file.wav --pretrained
cargo run --example transcribe_wav -- --audio /path/to/file.wav --models-dir models
cargo run --example transcribe_wav --features long-form -- --audio /path/to/file.wav --pretrained --long-form
cargo run --example transcribe_wav --features long-form -- --audio /path/to/file.wav --models-dir models --long-form
cargo run --example transcribe_wav --features long-form -- --audio /path/to/file.wav --pretrained --long-form --long-form-workers 2
cargo run --example transcribe_wav --features vad -- --audio /path/to/file.wav --pretrained --long-form --vad-long-form
```

The example expects mono 16kHz WAV input.

## Notes

- The public API is still moving
- `scriptrs` currently targets the exact file layout and model I/O shipped in `avencera/scriptrs-models`; if you swap in a different CoreML Parakeet export, you may need runtime code changes
- Use `long-form` for the fastest path on clean, dense speech
- Add `vad` when you need better robustness on sparse-speech or non-speech-heavy recordings; on dense or far-field speech it costs accuracy

## License

Licensed under the [Apache License, Version 2.0](LICENSE)
