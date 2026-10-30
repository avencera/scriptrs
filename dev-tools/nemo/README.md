# NeMo parity oracle

NeMo PyTorch is the reference for the native backends. This uv project pins the
stack the 2026-10-23 Parakeet profile measured (NeMo 3.0.0, torch 2.14.1+cu130)
and Silero VAD 5.1.2, whose weights match the published Core ML VAD.

```sh
uv sync
just nemo-export                                 # from dev-tools/
just nemo-refs v2 /shared/datasets/parakeet-ref/v2
```

## `export_native.py`

Writes the native model layout from the `.nemo` checkpoints and the Silero package:

```text
parakeet-v2/model.safetensors   parakeet-v2/vocab.txt
parakeet-v3/model.safetensors   parakeet-v3/parakeet_vocab.json
nemotron-speech-streaming-en-0.6b/model.safetensors   .../vocab.txt
nemotron-3.5-asr-streaming-0.6b/model.safetensors     .../vocab.txt
vad/silero-vad-v5.safetensors
```

`--only` (repeatable: `v2`, `v3`, `nemotron-en`, `nemotron-3.5`, `vad`) exports a
subset. The streaming checkpoints' configs also record the cache-aware settings
(`att_context_size`, causal subsampling and convolution, LayerNorm in the
convolution module), and their `vocab.txt` ends with the `<blk>` token.

Tensors keep their NeMo or Silero names and stay fp32. The safetensors metadata
holds `format`, `format_version` (now `2`), the source checkpoint and its SHA-256,
and the model config the runtime checks. The config lists every setting that
changes the numbers at inference, with keys the checkpoint omits filled from the
NeMo class defaults, so it records what NeMo actually runs. `src/native` (the
`cuda` feature) refuses a file whose config uses an option the native engine does
not implement or does not match the requested deployment. The vocab files are byte-identical to the
published Core ML bundle's.

## `dump_ref.py`

Writes `b1` (one 10.4 s utterance) and `b8` (8 utterances, 2.3–25.5 s, zero
padded) reference sets: the mel, every subsampling stage, all submodules of
layers 0 and 23, every layer output, the encoder output, a per-step greedy TDT
trace, and a bf16-autocast encoder output for tolerance scale. The trace follows
NeMo's label-looping frame advance (the same rules as `HostLoop`), and generation
fails unless its tokens and timestamps equal NeMo's batched decode. `--vad-audio`
adds Silero v5 per-window probabilities. `--long-audio <wav>` writes only
`long-full` and `long-local`: one 120 s excerpt (from 300 s by default) with full
attention, then with NeMo's long-form recipe (`rel_pos_local_attn` [256, 256]),
the only sets where the two attention modes differ. Each is about 3.5 GB. NeMo runs on the GPU in fp32 with TF32
off, so run it through the homebased `gpu0` queue.

The v3 sets at `/shared/datasets/parakeet-ref/` came from the profile's original
script, which this one ports with an unchanged key layout; v2 sets are in `v2/`
and Silero's in `silero-v5/`. `just parity` in the repository root checks the
Rust side against them.

## `stream_ref.py`

Runs NeMo's cache-aware streaming loop (the one in
`speech_to_text_cache_aware_streaming_infer.py`) over LibriSpeech test-clean, one
utterance at a time, for each `--lookahead` (repeatable), and writes under
`<out>/<model>/` `r<R>.texts.json`, `r<R>.hyps.json` (tokens and encoder-frame
timestamps), `r<R>.timing.json` (per-chunk step time) and, with `--dump N`,
`r<R>.dump.safetensors`: the first N utterances' audio, mel, each chunk's mel input
and every layer's output, and the tokens. The Rust streaming tests read the dumps
from `/shared/datasets/nemotron-stream-ref` (or `$SCRIPTRS_STREAM_REF`). NeMo runs
in FP32 with TF32 off on the GPU, so submit it to the `gpu0` queue:

```sh
uv run python stream_ref.py --model nemotron-en --lookahead 13 --lookahead 1 \
    --dump 4 --out /shared/datasets/nemotron-stream-ref
```

`examples/stream_bench.rs` writes the scriptrs side for `wer.py`. `--fp32` gives
NeMo's text exactly; the FP16 default changes a few utterances:

```sh
cargo run --release --features cuda --example stream_bench -- \
    --model /shared/models/scriptrs-native/nemotron-speech-streaming-en-0.6b \
    --lookahead 1 --fp32 --out bench/stream-r1.json
uv run python wer.py bench/stream-r1.texts.json \
    --against /shared/datasets/nemotron-stream-ref/nemotron-en/r1.texts.json
```

## `wer.py`

Scores a `{id: text}` transcript file on LibriSpeech test-clean the way the NeMo
profile bench does: Whisper's English normalizer, empty references dropped, corpus
WER with jiwer. `--against <texts.json>` lists utterances whose raw text differs.
`examples/cuda_bench.rs` writes the transcripts:

```sh
cargo run --release --features cuda,long-form --example cuda_bench -- \
    --models /shared/models/scriptrs-native --version v3 --out bench/v3-bs1.json
uv run python wer.py bench/v3-bs1.texts.json
```

## `long_texts.py` and `long_wer.py`

`long_texts.py` transcribes whole recordings with NeMo's long-form recipe (local
attention [256, 256], FP32, one batch-1 input per file, eval mode after the
attention swap) and writes `{stem: text}`. `long_wer.py` scores any number of
such files against the Earnings-21 `.nlp` references, pooled over the recordings
all of them cover, with `--per-file` for each recording.
`cuda_bench --long <wav dir>` writes the scriptrs side:

```sh
uv run python long_texts.py ~/code/speakrs/fixtures/datasets/earnings21/wav/*.wav --out nemo.texts.json
cargo run --release --features cuda,long-form --example cuda_bench -- \
    --models /shared/models/scriptrs-native --version v3 \
    --long ~/code/speakrs/fixtures/datasets/earnings21/wav --repeats 1 --out bench/e21.json
uv run python long_wer.py nemo.texts.json bench/e21.texts.json
```

`change_attention_model` rebuilds the attention modules in training mode, so a
script that swaps attention and does not call `eval()` afterwards runs with
dropout on. The 2026-10-23 profile's long-file runs did that: its 9.14 % on
Earnings-21 4360674 is 9.34 % in eval mode
