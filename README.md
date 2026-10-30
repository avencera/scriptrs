# scriptrs

Fast [NVIDIA Parakeet TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) speech-to-text in Rust.

- Parakeet TDT 0.6B v2 (English) and v3 (25 European languages)
- Core ML on macOS
- Native CUDA on Linux, with no ONNX Runtime or PyTorch (`cuda` feature)
- Long-form transcription of hour-long recordings, with optional VAD
- Token-level timestamps and confidences

On an RTX 5090, the CUDA backend transcribes LibriSpeech test-clean at 1534x real
time, 3.2x as fast as NVIDIA NeMo and 6.1x as fast as parakeet-rs, at the same
word error rate. Through the batching queue it runs at 4770x. On 31.6 h of
multilingual speech the default FP16 mode scores the same WER as the FP32 mode,
which matches NeMo. On six harder English sets it has a lower WER than Whisper
large-v3 at about 18x its speed. See [CUDA](#cuda) for the numbers.

`scriptrs` is early: the public API is still moving.

The base crate exposes a single-chunk `TranscriptionPipeline`. Fast long-audio chunking lives behind the `long-form` feature via `LongFormTranscriptionPipeline`. VAD-backed speech region planning lives behind the `vad` feature, which also enables `long-form`.

## Current scope

- Base pipeline for short audio
- Optional fast long-form pipeline with overlap chunking
- Optional VAD-backed long-form region planning
- Background queue for many short inputs, batched on CUDA
- Native CoreML inference on macOS
- CUDA inference on Linux (`cuda` feature)
- Hugging Face download support with optional local model loading

## What it does not do yet

- Windows support
- Other ASR models
- Streaming transcription
- Stable public guarantees around model layout or long-form behavior

## Install

```toml
[dependencies]
scriptrs = "0.3"
```

For native CUDA on Linux:

```toml
[dependencies]
scriptrs = { version = "0.3", features = ["cuda"] }
```

For fast long-form transcription:

```toml
[dependencies]
scriptrs = { version = "0.3", features = ["long-form"] }
```

For VAD-backed long-form transcription:

```toml
[dependencies]
scriptrs = { version = "0.3", features = ["vad"] }
```

## Model downloads

With the default `online` feature, `scriptrs` can resolve models automatically:

- it downloads the runtime bundle from `avencera/scriptrs-models`

You can force a local bundle when needed:

- `SCRIPTRS_MODELS_DIR=/path/to/models` forces a local bundle

Online downloads use the immutable `63f6f9cbce43e5561c219229896f86f20c0eb660`
snapshot of `avencera/scriptrs-models`.

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
feature, `from_pretrained_native` downloads them from the same pinned
`avencera/scriptrs-models` snapshot:

```rust,ignore
use scriptrs::{ModelBundle, ParakeetDeployment, TranscriptionPipeline};

let bundle = ModelBundle::from_pretrained_native(ParakeetDeployment::v3())?;
let pipeline = TranscriptionPipeline::from_bundle(bundle)?;
let result = pipeline.run(&audio)?;
```

With `vad`, `from_pretrained_native_long_form` also downloads the Silero VAD
weights. To manage the files yourself, `dev-tools/nemo/export_native.py` writes
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
| scriptrs | 1534x | 3.2x | 1.93 % |
| NVIDIA NeMo | 477x | 1x | 1.93 % |
| parakeet-rs 0.4.3 (ONNX Runtime) | 253x | 0.53x | 1.93 % |

Through the [queue](#many-short-inputs), which prepares the next inputs on the CPU
while the GPU runs, the same set runs at 1761x one at a time with identical
transcripts, and at 4770x in batches of up to 16 (the default). Batching changes 2
of the 2620 transcripts by one word or punctuation mark, and WER stays 1.93 %.

The multilingual check is nearly 6 times the size of LibriSpeech: FLEURS and MLS,
31.6 h over ten language sets in six languages (de, es, fr, it, nl, pt), one
utterance at a time with automatic language selection. The FP32 mode matches NeMo
layer by layer and gives NeMo's exact text in the parity tests, and the FP16
default scores the same WER as FP32. Speed is for the FP16 default:

| Set | Speed (x real time) | WER, FP16 | WER, FP32 |
|---|---|---|---|
| FLEURS de, es, fr, it, nl, pt (15.8 h) | 1839x | 4.57 % | 4.57 % |
| MLS es, fr, it, pt (15.8 h) | 1921x | 6.25 % | 6.26 % |

On all 44 Earnings-21 calls (39 h), long-form with the defaults runs at 3541x with
8.24 % pooled WER; NeMo transcribing each whole call with local attention scores
9.58 %.

On harder English audio, long-form with the defaults and English selected has a
lower WER than Whisper large-v3 (run through WhisperX 3.8.6 in float16) on every
set, and runs 14x to 20x as fast:

| Set | Recordings | Hours | scriptrs WER | Whisper WER | scriptrs speed | Whisper speed | vs Whisper |
|---|---:|---:|---:|---:|---:|---:|---:|
| Earnings-21 earnings calls | 44 | 39.3 | **8.2 %** | 9.7 % | **3541x** | 186x | 19.0x |
| AMI meetings, headset mix | 34 | 18.7 | **21.5 %** | 23.8 % | **3458x** | 182x | 19.0x |
| AMI meetings, far-field microphone | 34 | 18.7 | **23.2 %** | 28.3 % | **3447x** | 199x | 17.3x |
| EdAcc accented conversations | 40 | 18.2 | **16.0 %** | 19.2 % | **3472x** | 201x | 17.3x |
| CHiME-6 dinner parties | 4 | 9.7 | **44.0 %** | 44.7 % | **3474x** | 176x | 19.7x |
| AppTek call-center calls, 14 accents | 56 | 9.4 | **11.2 %** | 12.6 % | **3108x** | 224x | 13.9x |
| Mean over sets | 212 | 114 | **20.7 %** | 23.1 % | **3456x** | 191x | 18.0x |

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
| Sparse speech, 24 % speech (4.2 h) | 14.8 % | **3.1 %** | 3430x | **3990x** |
| Earnings-21 earnings calls | **8.2 %** | 8.6 % | **3630x** | 2030x |
| AMI meetings, headset mix | 21.5 % | **21.2 %** | **3480x** | 1800x |
| AMI meetings, far-field microphone | **23.2 %** | 26.6 % | **3480x** | 1790x |
| CHiME-6 dinner parties | **44.0 %** | 54.8 % | **3490x** | 1970x |

The sparse set is LibriSpeech test-clean utterances separated by 15 to 75 s of
film music, effects, ambience and low-level noise. VAD runs use the default 0.5
speech threshold.

By default the encoder layers' linear weights are stored in FP16, which has TF32's
10-bit mantissa, and their matrix multiplies run on FP16 tensor cores with FP32
accumulation. `Precision::Tf32` keeps the weights in FP32 and uses TF32 tensor
cores on Ampere and newer GPUs, or FP32 on older ones. `Precision::Fp32` matches
NeMo's FP32 decode token for token on any GPU:

```rust,ignore
use scriptrs::{ModelBundle, ParakeetDeployment, Precision};

let bundle = ModelBundle::from_native_dir("models", ParakeetDeployment::v3())
    .with_precision(Precision::Fp32);
```

On an RTX 5090 with Parakeet v3 (speed in x real time, then WER):

| | FP16 (default) | TF32 | FP32 |
|---|---|---|---|
| LibriSpeech test-clean, one utterance at a time | 1534x, 1.93 % | 1179x, 1.93 % | — |
| LibriSpeech test-clean, queue in batches of 16 | 4767x, 1.93 % | 3364x, 1.92 % | — |
| Earnings-21, long-form | 3633x, 8.24 % | 2633x, 8.24 % | — |
| FLEURS de, es, fr, it, nl, pt (15.8 h) | 1839x, 4.57 % | 1450x, 4.57 % | 966x, 4.57 % |
| MLS es, fr, it, pt (15.8 h) | 1921x, 6.25 % | 1524x, 6.26 % | 996x, 6.26 % |

FLEURS and MLS run one utterance at a time with automatic language selection and
are scored with Whisper's basic normalizer. On each of their ten language sets,
FP16 is within one word error of TF32. On LibriSpeech one utterance at a time,
FP16 gives the same transcripts as TF32. Batching changes a few transcripts in
either precision, so the queued FP16 and TF32 transcripts differ slightly.

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

### Many short inputs

To transcribe many independent inputs, such as utterances or clips, move the
pipeline onto a background queue. A preparation thread computes the next inputs'
features while the model runs, and on CUDA the model transcribes up to
`QueueConfig::max_batch` waiting inputs in one pass. Results come back in the order
the queue accepted the requests, each under the id you gave it:

```rust,no_run
use std::time::Duration;

use scriptrs::{QueueError, QueuedTranscriptionRequest, TranscriptionPipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pipeline = TranscriptionPipeline::from_dir("models")?;
    let (sender, receiver) = pipeline.into_queued()?;

    let producer = std::thread::spawn(move || -> Result<(), QueueError> {
        for (id, audio) in load_clips() {
            let mut request = QueuedTranscriptionRequest::new(id, audio);
            loop {
                match sender.try_push(request) {
                    Ok(_) => break,
                    // the queue is at capacity; retry once the model catches up
                    Err(QueueError::Full(rejected)) => {
                        request = rejected;
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    Err(error) => return Err(error),
                }
            }
        }
        // dropping the sender closes the queue once every request is done
        Ok(())
    });

    for queued in receiver {
        let queued = queued?;
        println!("{}: {}", queued.id, queued.result?.text);
    }
    producer.join().expect("producer panicked")?;
    Ok(())
}

fn load_clips() -> Vec<(String, Vec<f32>)> {
    Vec::new()
}
```

`QueueConfig` sets the queue's capacity (64 by default) and the largest batch (16
by default); pass it with `into_queued_with_config`. At `max_batch: 1` each result
is exactly what `run` gives. A failed batch reruns its inputs one at a time, so a
bad input fails only its own request. Core ML bundles run a batch's inputs one
after another.

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
