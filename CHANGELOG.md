# Changelog

All notable changes to scriptrs and its Python package are listed here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). The Rust crate and the Python package share
one version.

## [Unreleased]

The first release on PyPI. It has breaking changes to the Rust API; the README's
"Breaking changes in 0.5.0" section lists what to write instead.

### Added

- Python package `scriptrs` on PyPI, with one wheel for every Python version from 3.11:
  Core ML on macOS arm64 and CUDA on Linux x86_64 and Windows x86_64. It includes the
  `scriptrs transcribe` command.
- Windows x86_64 builds of the `cuda` feature. They open `nvcuda.dll`,
  `cublas64_12.dll` and, for `low-precision`, `cublasLt64_12.dll` at run time, and
  the Windows wheel installs cuBLAS from `nvidia-cublas-cu12` and loads it before the
  first model. CI builds and tests them on a runner without a GPU; they haven't run
  on a Windows GPU yet.
- `TranscriptionScheduler` batches windows across concurrent requests on CUDA, for
  servers with many short requests. `SchedulerConfig::max_requests` bounds the requests
  held at once, and `try_submit` returns `TranscriptionError::SchedulerFull` past it.
- Silero VAD 6.2.3. VAD mode now scores each window with both 5.1.2 and 6.2.3, keeps
  the higher speech probability, and uses a 0.6 threshold. `VadModel::SileroV5` and
  `VadModel::SileroV6` run one model alone at 0.5.
- VAD mode transcribes each speech region as soon as the scan passes it, which makes
  it 13 to 36% faster on the benchmark sets with identical transcripts.
- Interrupted model downloads resume where they stopped, and concurrent downloads of
  one file wait for each other through the same lock file as `huggingface_hub`.

### Changed

- `TranscriptionConfig` has only `language`. The frontend settings and the input limit
  come from the model bundle. `from_deployment` and `validate_for` are gone.
- `TranscriptionError::Control` wraps `TranscriptionControlError` in place of the three
  variants that mirrored it. Messages and Python error codes are unchanged.
- `Vocabulary` loads only through a deployment, with `Vocabulary::load` or
  `Vocabulary::from_bytes`. `VocabKind` is gone.
- `VadConfig` is gone. The threshold is `VadSegmentationConfig::threshold`.
- Hub downloads go through [hf-fetch](https://crates.io/crates/hf-fetch). Download
  errors leave out the signed query of storage URLs.
- In Python, `transcribe_many` submits every clip before waiting, its `max_batch`
  default is 32, and the error code `queue` is now `scheduler`.

### Fixed

- Long-form VAD timestamps could be up to 80 ms early in the overlap fallback.
- Clips from 30 to 40 s on native CUDA bundles ran as one window and skipped VAD.
- A failed long-form run cancelled the caller's `TranscriptionControl`.
- A Core ML encoder length of 0 decoded padding as speech.
- Core ML inputs and the FP8 and NVFP4 cuBLASLt buffers are checked before use, in
  place of reads that could go out of bounds.
- A streaming checkpoint with no left attention context panicked instead of failing
  to load.
- A failing Silero scan or worker thread now stops the whole run promptly.

## [0.4.0] - 2026-10-30

### Added

- Python bindings, built as wheels for macOS arm64 and Linux x86_64.
- Cache-aware streaming transcription on CUDA with `StreamingModel`.
- FP8, INT8 and NVFP4 encoder precisions behind the `low-precision` feature.
- Native CUDA weights download from one Hugging Face repository per model.

### Changed

- `Precision` is `#[non_exhaustive]`.
- VAD regions are padded by 0.2 s so soft first words are kept.
- The relative-position attention runs as one fused kernel on CUDA.

[Unreleased]: https://github.com/avencera/scriptrs/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/avencera/scriptrs/releases/tag/v0.4.0
