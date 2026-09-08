# Parakeet ONNX/CUDA bundle format

This document is the contract between the `scriptrs` `onnx-cuda` feature and
the tool that exports NVIDIA Parakeet TDT checkpoints to ONNX. `scriptrs`
implements the consumer side in `src/onnx/`. An exporter that follows this
document produces a directory that `OnnxCudaBundle::open` accepts.

Schema tag: `avencera.scriptrs.parakeet-onnx-bundle/1`
Graph contract: `avencera.parakeet-split/1`
VAD contract: `avencera.silero-vad/1`

## Runtime

| Item | Value |
| --- | --- |
| Rust crate | `ort = "=2.0.0-rc.12"` with the `cuda` feature |
| ONNX Runtime | 1.24.x, the release `ort-sys 2.0.0-rc.12` binds |
| Execution provider | `CUDAExecutionProvider`, registered with error-on-failure |
| Minimum Rust for the feature | 1.88 (required by `ort`); the base crate stays at 1.85 |

The exact `ort` pin is an ABI requirement, not a preference. `ort-sys` sets
`links = "onnxruntime"`, so a binary can hold only one `ort-sys`; SpeakRS
already resolves `2.0.0-rc.12`, and both crates must share it.

Prebuilt ORT binaries from pyke for `x86_64-unknown-linux-gnu` exist for
`cu12` and `cu13` feature sets. Selecting CUDA 12 or 13 is a build-time
decision of the image, not of this crate. This document does not qualify
any toolchain; the image build must lock exact packages and digests.

## Bundle layout

A bundle is one directory. It contains `manifest.json` and exactly the files
the manifest declares. Nothing else may exist under the directory, including
hidden files and symbolic links. Directory names are free as long as every
path is relative, uses `/`, and contains no `.` or `..` components.

Example:

```text
parakeet-v3/
  manifest.json
  parakeet_vocab.json
  graphs/
    encoder.onnx
    encoder.onnx.data
    decoder.onnx
    joint.onnx
  vad/
    silero_vad.onnx
```

## Manifest schema

`manifest.json` is strict JSON. Unknown fields anywhere are rejected.

```json
{
  "schema": "avencera.scriptrs.parakeet-onnx-bundle/1",
  "deployment": "parakeet-tdt-0.6b-v3",
  "source": {
    "repository": "nvidia/parakeet-tdt-0.6b-v3",
    "revision": "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
  },
  "export": {
    "exporter": "avencera-models-parakeet-export",
    "exporter_version": "<tool version or commit>",
    "opset": 17,
    "graph_contract": "avencera.parakeet-split/1"
  },
  "tokenizer": {
    "path": "parakeet_vocab.json",
    "sha256": "<64 lowercase hex>",
    "format": "parakeet-vocab-json",
    "source_revision": "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
  },
  "graphs": {
    "encoder": {
      "path": "graphs/encoder.onnx",
      "sha256": "<64 lowercase hex>",
      "external_data": [
        { "path": "graphs/encoder.onnx.data", "sha256": "<64 lowercase hex>" }
      ]
    },
    "decoder": { "path": "graphs/decoder.onnx", "sha256": "<64 lowercase hex>" },
    "joint": { "path": "graphs/joint.onnx", "sha256": "<64 lowercase hex>" }
  },
  "vad": {
    "graph": { "path": "vad/silero_vad.onnx", "sha256": "<64 lowercase hex>" },
    "source": {
      "repository": "snakers4/silero-vad",
      "revision": "<40 lowercase hex commit>"
    },
    "graph_contract": "avencera.silero-vad/1"
  }
}
```

| Field | Type | Rule |
| --- | --- | --- |
| `schema` | string | Must equal `avencera.scriptrs.parakeet-onnx-bundle/1` |
| `deployment` | enum | `parakeet-tdt-0.6b-v2` or `parakeet-tdt-0.6b-v3`; must match the requested `ParakeetDeployment` |
| `source.repository` | string | Hugging Face repository of the checkpoint |
| `source.revision` | git commit | 40 lowercase hex characters |
| `export.exporter` | string | Exporter program name |
| `export.exporter_version` | string | Exporter version or commit |
| `export.opset` | integer | Default-domain opset of every graph |
| `export.graph_contract` | enum | `avencera.parakeet-split/1` |
| `tokenizer.path` | bundle path | Tokenizer file |
| `tokenizer.sha256` | digest | SHA-256 of the tokenizer file |
| `tokenizer.format` | enum | `sentencepiece-vocab-txt` for v2, `parakeet-vocab-json` for v3 |
| `tokenizer.source_revision` | git commit | Must equal `source.revision` |
| `graphs.encoder`, `graphs.decoder`, `graphs.joint` | graph asset | Path, digest, optional `external_data` list |
| `graphs.*.external_data[]` | file asset | Path and digest of each ONNX external-data file |
| `vad` | optional | Required only for `LongFormMode::Vad` |
| `vad.graph` | graph asset | Silero graph |
| `vad.source` | checkpoint | Repository and commit the Silero graph was exported from |
| `vad.graph_contract` | enum | `avencera.silero-vad/1` |

Digests are SHA-256 over file bytes, written as 64 lowercase hexadecimal
characters. Uppercase digests are rejected.

### Startup validation order

1. Read and parse the manifest. An unknown `schema` is reported before strict
   parsing.
2. Check `deployment`, tokenizer `format`, and tokenizer `source_revision`
   against the requested deployment and `source.revision`.
3. Reject duplicate paths. Verify every declared file exists as a regular
   file and hash it against the manifest.
4. Walk the directory. Any regular file, symbolic link, or special file that
   is neither `manifest.json` nor declared fails validation.
5. Confirm the CUDA execution provider is supported on the platform and
   compiled into the linked ONNX Runtime.
6. Build one session per graph with CUDA registered as error-on-failure and
   compare each session's declared inputs and outputs with the contract
   below. Names, element types, ranks, and static axes must match; an axis
   the graph declares as dynamic is checked when the tensor is produced.
7. Load the tokenizer through the deployment contract (blank id 1024 for v2,
   8192 for v3; token counts; blank token placement).

Every step is a hard failure. There is no fallback to Core ML, to a sibling
file, or to the CPU provider.

## Graph contract `avencera.parakeet-split/1`

Shared dimensions come from `ParakeetDeployment`: 128 mel features, encoder
hidden 1024, decoder hidden 640, two LSTM layers, 240 000 sample hard
maximum, hop 160, so the padded mel frame count is `240000 / 160 + 1 = 1501`.
Tensor names are the same names the Core ML split bundle uses.

### Encoder

| Direction | Name | dtype | Shape | Notes |
| --- | --- | --- | --- | --- |
| input | `mel` | float32 | `[1, 128, 1501]` | Log-mel features padded with zeros to 1501 frames; the time axis may be declared dynamic but always receives 1501 |
| input | `mel_length` | int32 | `[1]` | Valid frame count |
| output | `encoder` | float32 | `[1, 1024, T]` | Encoded frames, hidden first, `T` data dependent |
| output | `encoder_length` | int32 | `[1]` | Valid encoded frames; must satisfy `0 <= encoder_length <= T` |

### Decoder (prediction network)

| Direction | Name | dtype | Shape | Notes |
| --- | --- | --- | --- | --- |
| input | `targets` | int32 | `[1, 1]` | Previous emitted token; the blank id before the first emission |
| input | `target_length` | int32 | `[1]` | Always 1 |
| input | `h_in` | float32 | `[2, 1, 640]` | LSTM hidden state, zeros at the start of every chunk |
| input | `c_in` | float32 | `[2, 1, 640]` | LSTM cell state, zeros at the start of every chunk |
| output | `decoder` | float32 | `[1, 640, 1]` | Projected decoder output for the joint |
| output | `h_out` | float32 | `[2, 1, 640]` | Next hidden state |
| output | `c_out` | float32 | `[2, 1, 640]` | Next cell state |

`scriptrs` adopts `h_out` and `c_out` only when the joint emits a non-blank
token, exactly as the Core ML path does.

### Joint (single-step decision)

| Direction | Name | dtype | Shape | Notes |
| --- | --- | --- | --- | --- |
| input | `encoder_step` | float32 | `[1, 1024, 1]` | One encoder frame |
| input | `decoder_step` | float32 | `[1, 640, 1]` | Current `decoder` output |
| output | `token_id` | int32 | `[1, 1, 1]` | Argmax over the token logits including blank; `0 <= token_id <= blank` |
| output | `token_prob` | float32 | `[1, 1, 1]` | Softmax probability of `token_id`; must be finite |
| output | `duration` | int32 | `[1, 1, 1]` | Argmax over the TDT duration head; bins map to `[0, 1, 2, 3, 4]` frames |
| output (v3 only) | `top_k_ids` | int32 | `[1, K]` | `K >= 1` highest-logit token ids, blank allowed, each `<= blank` |
| output (v3 only) | `top_k_logits` | float32 | `[1, K]` | Logits parallel to `top_k_ids`; must be finite |

v2 graphs must not declare the top-K outputs. v3 graphs must declare both.
`K` is fixed by the exporter and read from the produced tensor; the language
filter in `scriptrs` computes a softmax over the supplied `K` logits only.

Blank ids: 1024 for v2 (`<blk>` is the last tokenizer entry), 8192 for v3
(one past the 8192-entry JSON table). These are owned by the deployment and
are not configurable through the manifest.

### Run-time bounds enforced by `scriptrs`

- Negative `token_id`, `duration`, or `top_k_ids` values are rejected.
- `token_id` or any `top_k_ids` entry above the blank id is rejected.
- `duration` outside the five deployment bins is rejected.
- Non-finite `token_prob` or `top_k_logits` values are rejected.
- Output shapes are checked against the table for every call, including
  dynamic axes.
- The greedy loop keeps its existing per-frame token cap and duration
  advancement; the graphs do not implement any decoding.

## VAD contract `avencera.silero-vad/1`

Silero VAD v5, 16 kHz graph. `scriptrs` drives chunking: each call carries
64 context samples from the previous chunk followed by 512 new samples, the
last short chunk is padded with its final sample, and the recurrent state is
reset to zeros per recording.

| Direction | Name | dtype | Shape | Notes |
| --- | --- | --- | --- | --- |
| input | `input` | float32 | `[1, 576]` | 64 context samples followed by 512 new samples |
| input | `state` | float32 | `[2, 1, 128]` | Opaque recurrent state; `state[0]` and `state[1]` correspond to the `h` and `c` inputs of the Core ML export |
| input | `sr` | int64 | `[]` (scalar) | Always 16000 |
| output | `output` | float32 | `[1, 1]` | Speech probability in `0..=1` |
| output | `stateN` | float32 | `[2, 1, 128]` | Next recurrent state; must be finite |

The planner constants (threshold 0.85, speech minimum 0.15 s, end silence
0.75 s, padding 0.10 s, split threshold 0.3, hysteresis offset 0.15,
minimum split silence 0.098 s, 2.0 s fallback overlap) are unchanged and
live in `src/long_form/planner.rs`.

### Provenance still owed by the exporter

The Core ML Silero bundle currently staged by `scriptrs` was copied from the
unpinned `aufklarer/Silero-VAD-v5-CoreML` repository. This crate does not
know which `snakers4/silero-vad` commit that export came from. The Models
exporter must select one Silero v5 16 kHz source commit, record it in
`vad.source.revision`, and confirm on fixed tensors that the ONNX graph and
the staged Core ML graph agree before either bundle is called equivalent.

## Exporter obligations

- Load the exact local checkpoint at `source.revision`; never download at
  export time from a moving reference.
- Emit the three graphs with the names, dtypes, and shapes above. Declare
  dynamic axes only where the table allows `T` or `K`.
- Compare each graph with the source model on fixed tensors and fail before
  publication on mismatch.
- Write `manifest.json` last, after hashing every file in place.
- Do not reuse Core ML artifact digests or qualification records for ONNX
  files. Exported file identity and source checkpoint identity are separate
  fields on purpose.

## What this document does not claim

No bundle has been exported or qualified. The CPU fixture tests in
`src/onnx/tests.rs` prove tensor movement, contract validation, bounds, and
the shared decoder on synthetic graphs. Only a run on a CUDA host with real
exported graphs can qualify a bundle, and the ignored test
`production_constructor_runs_fixture_graphs_on_cuda` is the entry point for
that check.
