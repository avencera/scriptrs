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
vad/silero-vad-v5.safetensors
```

Tensors keep their NeMo or Silero names and stay fp32. The safetensors metadata
holds `format`, `format_version`, the source checkpoint and its SHA-256, and the
model config the runtime checks. The vocab files are byte-identical to the
published Core ML bundle's.

## `dump_ref.py`

Writes `b1` (one 10.4 s utterance) and `b8` (8 utterances, 2.3–25.5 s, zero
padded) reference sets: the mel, every subsampling stage, all submodules of
layers 0 and 23, every layer output, the encoder output, a per-step greedy TDT
trace, and a bf16-autocast encoder output for tolerance scale. `--vad-audio`
adds Silero v5 per-window probabilities. NeMo runs on the GPU in fp32 with TF32
off, so run it through the homebased `gpu0` queue.

The v3 sets at `/shared/datasets/parakeet-ref/` came from the profile's original
script, which this one ports with an unchanged key layout; v2 sets are in `v2/`
and Silero's in `silero-v5/`. `just parity` in the repository root checks the
Rust side against them.
