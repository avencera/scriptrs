# scriptrs

> [!WARNING]
> Work in progress
>
> `scriptrs` is early and intentionally narrow right now:
> - macOS only
> - Apple CoreML only
> - Parakeet TDT v2 only
> - no CUDA
> - no non-macOS backend yet

Rust transcription with native CoreML Parakeet v2 inference.

The base crate exposes a single-chunk [`TranscriptionPipeline`](./src/pipeline.rs). Long-audio chunking, VAD, and overlap fallback live behind the `long-form` feature via [`LongFormTranscriptionPipeline`](./src/long_form.rs).

## Current scope

- Base pipeline for short audio
- Optional long-form pipeline with VAD-based region planning
- Native CoreML inference on macOS
- Hugging Face download support with optional local model loading

## What it does not do yet

- Linux or Windows support
- CUDA support
- Other ASR models
- Streaming transcription
- Stable public guarantees around model layout or long-form behavior

## Install

```toml
[dependencies]
scriptrs = "0.1.0"
```

For long-form transcription:

```toml
[dependencies]
scriptrs = { version = "0.1.0", features = ["long-form"] }
```

## Model downloads

With the default `online` feature, `scriptrs` can resolve models automatically:

- it downloads the runtime bundle from `avencera/scriptrs-models`

You can override either side of that:

- `SCRIPTRS_MODELS_DIR=/path/to/models` forces a local bundle
- `SCRIPTRS_MODELS_REPO=owner/repo` forces a specific Hugging Face model repo layout

## Local model layout

If you want to use `from_dir(...)` or `SCRIPTRS_MODELS_DIR`, the local bundle should look like this:

```text
models/
  parakeet-v2/
    encoder.mlmodelc/
    decoder.mlmodelc/
    joint-decision.mlmodelc/
    vocab.txt
```

With `long-form`, add:

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

Enable `long-form` if you want `scriptrs` to own VAD, chunking, and overlap fallback internally.

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

## Example

A small WAV example is included:

```bash
cargo run --example transcribe_wav -- --audio /path/to/file.wav --pretrained
cargo run --example transcribe_wav -- --audio /path/to/file.wav --models-dir models
cargo run --example transcribe_wav --features long-form -- --audio /path/to/file.wav --pretrained --long-form
cargo run --example transcribe_wav --features long-form -- --audio /path/to/file.wav --models-dir models --long-form
```

The example expects mono 16kHz WAV input.

## Notes

- The public API is still moving
- `scriptrs` currently targets the exact file layout and model I/O shipped in `avencera/scriptrs-models`; if you swap in a different CoreML Parakeet export, you may need runtime code changes
- Long-form is intentionally optional so callers with their own segmentation pipeline do not pay for the extra machinery
