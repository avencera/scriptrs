# scriptrs

**Work in progress**

`scriptrs` is early and intentionally narrow right now:

- Apple CoreML on macOS by default
- ONNX Runtime with the CUDA execution provider behind the `onnx-cuda` feature, from a local validated bundle
- Parakeet TDT v2 and v3
- no other backends

Rust transcription with native CoreML Parakeet v2 and v3 inference.

The base crate exposes a single-chunk `TranscriptionPipeline`. Fast long-audio chunking lives behind the `long-form` feature via `LongFormTranscriptionPipeline`. VAD-backed speech region planning lives behind the `vad` feature, which also enables `long-form`.

## Current scope

- Base pipeline for short audio
- Optional fast long-form pipeline with overlap chunking
- Optional VAD-backed long-form region planning
- Native CoreML inference on macOS
- Hugging Face download support with optional local model loading

## What it does not do yet

- Windows support
- Other ASR models
- Streaming transcription
- Stable public guarantees around model layout or long-form behavior

## ONNX Runtime on CUDA

The `onnx-cuda` feature adds a second runtime behind the same model boundary.
It loads a local bundle described by `manifest.json`, verifies every file
digest, checks each graph against the split-graph tensor contract, and runs
the graphs with the ONNX Runtime CUDA execution provider. The frontend,
greedy TDT decoder, language filtering, long-form planner, and VAD chunking
are shared with the CoreML path.

```toml
[dependencies]
scriptrs = { version = "0.2.0", default-features = false, features = ["onnx-cuda", "vad"] }
```

```rust
use scriptrs::{CudaDevice, ModelBundle, OnnxCudaBundle, ParakeetDeployment, TranscriptionPipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bundle = OnnxCudaBundle::open("bundles/parakeet-v3", ParakeetDeployment::v3(), CudaDevice::new(0))?;
    let pipeline = TranscriptionPipeline::from_bundle(ModelBundle::from_onnx_cuda(bundle))?;
    let result = pipeline.run(&load_mono_16khz_audio())?;

    println!("{}", result.text);
    Ok(())
}

fn load_mono_16khz_audio() -> Vec<f32> {
    Vec::new()
}
```

Opening a bundle never downloads anything and never falls back to the CPU
provider; a host without a usable CUDA runtime fails at `open`. The manifest
schema and the exact graph tensor contract are in
[`docs/onnx-cuda-bundle.md`](docs/onnx-cuda-bundle.md). The feature pins
`ort = "=2.0.0-rc.12"` (ONNX Runtime 1.24) so it can share one runtime with
SpeakRS, and it needs Rust 1.88 because of that crate.

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

`LongFormConfig` defaults to the fast overlap-chunking path with `4` workers. You can tune the worker count when you want less or more parallelism:

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
