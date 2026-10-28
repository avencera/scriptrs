# scriptrs

**Work in progress**

`scriptrs` is early and intentionally narrow right now:

- Apple CoreML on macOS
- NVIDIA CUDA on Linux, experimental, behind the `cuda` feature
- Parakeet TDT v2 and v3

Rust transcription with native CoreML Parakeet v2 and v3 inference.

The base crate exposes a single-chunk `TranscriptionPipeline`. Fast long-audio chunking lives behind the `long-form` feature via `LongFormTranscriptionPipeline`. VAD-backed speech region planning lives behind the `vad` feature, which also enables `long-form`.

## Current scope

- Base pipeline for short audio
- Optional fast long-form pipeline with overlap chunking
- Optional VAD-backed long-form region planning
- Native CoreML inference on macOS
- Experimental FP32 CUDA inference on Linux (`cuda` feature)
- Hugging Face download support with optional local model loading

## What it does not do yet

- Windows support
- Other ASR models
- Streaming transcription
- Stable public guarantees around model layout or long-form behavior

## Install

```toml
[dependencies]
scriptrs = "0.1.0"
```

For fast long-form transcription:

```toml
[dependencies]
scriptrs = { version = "0.1.0", features = ["long-form"] }
```

For VAD-backed long-form transcription:

```toml
[dependencies]
scriptrs = { version = "0.1.0", features = ["vad"] }
```

## Model downloads

With the default `online` feature, `scriptrs` can resolve models automatically:

- it downloads the runtime bundle from `avencera/scriptrs-models`

You can force a local bundle when needed:

- `SCRIPTRS_MODELS_DIR=/path/to/models` forces a local bundle

Online downloads use the immutable `5fbef649fc28117307a967e509be696c77b6e521`
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


## CUDA (experimental)

The `cuda` feature runs Parakeet on an NVIDIA GPU, in FP32, with no ONNX Runtime
or PyTorch. On NeMo reference inputs (LibriSpeech utterances and a two-minute
Earnings-21 excerpt) its tokens and timestamps match NVIDIA NeMo's greedy decode
exactly. It is built from this
repository only for now, and does not yet support the v3 language filter, top-K
candidates or VAD.

It loads native weights rather than Core ML models:

```text
models/
  parakeet-v2/
    model.safetensors
    vocab.txt
  parakeet-v3/
    model.safetensors
    parakeet_vocab.json
```

```rust,ignore
use scriptrs::{ModelBundle, ParakeetDeployment, TranscriptionPipeline};

let bundle = ModelBundle::from_native_dir("models", ParakeetDeployment::v3());
let pipeline = TranscriptionPipeline::from_bundle(bundle)?;
let result = pipeline.run(&audio)?;
```

A native bundle accepts single inputs up to 40 s; longer audio goes through
`LongFormTranscriptionPipeline`. For native bundles, long-form defaults to 30 s
windows with 8 s overlap, run 16 at a time through one model. Longer windows or
shorter overlaps lose accuracy on long recordings.

On an RTX 5090, v3 runs LibriSpeech test-clean one utterance at a time at 719x
real time with 1.93 % WER, the same WER as NeMo's FP32 greedy decode at 410x.
On all 44 Earnings-21 calls (39 h), long-form with the defaults runs at 1260x
with 8.25 % pooled WER; NeMo transcribing each whole call with local attention
scores 9.58 %.

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

`LongFormConfig` defaults to the fast overlap-chunking path with `4` workers, each loading its own model. On native CUDA bundles, windows instead run `batch_size` at a time (16 by default) through the pipeline's one model; Core ML bundles ignore `batch_size`. `LongFormConfig::for_bundle(&bundle)` gives a bundle's defaults to start from when you change one setting. You can tune the worker count when you want less or more parallelism:

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

Enable `vad` when you want VAD-backed speech region planning for sparse speech, long silences, or recordings with a lot of non-speech audio:

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
- Add `vad` when you need better robustness on sparse-speech or non-speech-heavy recordings
