"""Export Parakeet TDT `.nemo` checkpoints and Silero VAD v5 to scriptrs native weights

Writes, under --out:

    parakeet-v2/model.safetensors, parakeet-v2/vocab.txt
    parakeet-v3/model.safetensors, parakeet-v3/parakeet_vocab.json
    vad/silero-vad-v5.safetensors

Tensors keep their NeMo (or Silero) names and stay fp32, so the file is the parity
oracle's own weights. Layout changes such as BatchNorm folding or transposes happen
at load time in the native runtime, not here. The safetensors header carries the
model config the runtime validates against
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import importlib.resources
import json
import tarfile
from pathlib import Path

import sentencepiece
import torch
import yaml
from safetensors.torch import save_file

FORMAT = "scriptrs-native"
FORMAT_VERSION = "2"

# checkpoint prefixes the native runtime reads; the preprocessor filterbank and
# window are kept so the Rust frontend can be checked against NeMo's own buffers
KEEP_PREFIXES = ("encoder.", "decoder.", "joint.", "preprocessor.featurizer.")

VERSIONS = {
    "v2": {
        "source": "nvidia/parakeet-tdt-0.6b-v2",
        "blank_id": 1024,
        "vocab_file": "vocab.txt",
    },
    "v3": {
        "source": "nvidia/parakeet-tdt-0.6b-v3",
        "blank_id": 8192,
        "vocab_file": "parakeet_vocab.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--v2-nemo",
        type=Path,
        default=Path("/shared/models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo"),
    )
    parser.add_argument(
        "--v3-nemo",
        type=Path,
        default=Path("/shared/models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo"),
    )
    parser.add_argument(
        "--only",
        choices=["v2", "v3", "vad"],
        action="append",
        help="Export only these artifacts (repeatable); default exports all",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def tar_member(archive: tarfile.TarFile, suffix: str) -> tarfile.TarInfo:
    matches = [m for m in archive.getmembers() if m.name.endswith(suffix)]
    if len(matches) != 1:
        raise SystemExit(f"expected one `{suffix}` in the .nemo, found {len(matches)}")
    return matches[0]


def resolved(section: dict, keys: tuple[str, ...]) -> dict:
    """`keys` of one config section, with absent keys filled from the defaults of the
    NeMo class that section builds, so the runtime checks the values NeMo really uses"""
    module_name, class_name = section["_target_"].rsplit(".", 1)
    parameters = inspect.signature(getattr(importlib.import_module(module_name), class_name)).parameters
    values = {}
    for key in keys:
        if key in section:
            values[key] = section[key]
        elif key in parameters and parameters[key].default is not inspect.Parameter.empty:
            values[key] = parameters[key].default
        else:
            raise SystemExit(f"{section['_target_']} has no `{key}` and no default for it")
    return values


def model_config(cfg: dict, version: str) -> dict:
    """The NeMo settings the native runtime checks before trusting the weights

    Includes every setting that changes the numbers at inference, so a checkpoint the
    runtime does not implement fails at load. Dropout and training-only settings are
    left out
    """
    pre, enc, dec, joint = cfg["preprocessor"], cfg["encoder"], cfg["decoder"], cfg["joint"]
    decoding = cfg["decoding"]
    prednet = dec["prednet"]
    decoder = resolved(dec, ("vocab_size", "blank_as_pad", "normalization_mode"))
    decoder.update(
        pred_hidden=prednet["pred_hidden"],
        pred_rnn_layers=prednet["pred_rnn_layers"],
        # RNNTDecoder reads this with `prednet.get("rnn_hidden_size", -1)`; a positive
        # value other than pred_hidden adds an LSTM projection
        rnn_hidden_size=prednet.get("rnn_hidden_size", -1),
    )
    return {
        "version": version,
        "preprocessor": resolved(
            pre,
            (
                "sample_rate",
                "normalize",
                "window_size",
                "window_stride",
                "window",
                "features",
                "n_fft",
                "log",
                "frame_splicing",
                "pad_to",
                "pad_value",
                "preemph",
                "mag_power",
                "log_zero_guard_type",
                "log_zero_guard_value",
                "exact_pad",
                "lowfreq",
                "highfreq",
                "mel_norm",
            ),
        ),
        "encoder": resolved(
            enc,
            (
                "feat_in",
                "feat_out",
                "n_layers",
                "d_model",
                "use_bias",
                "subsampling",
                "subsampling_factor",
                "subsampling_conv_channels",
                "causal_downsampling",
                "ff_expansion_factor",
                "self_attention_model",
                "n_heads",
                "att_context_size",
                "xscaling",
                "untie_biases",
                "pos_emb_max_len",
                "conv_kernel_size",
                "conv_norm_type",
                "conv_context_size",
                "reduction",
                "global_tokens",
            ),
        ),
        "decoder": decoder,
        "joint": {
            **resolved(joint, ("num_classes", "num_extra_outputs")),
            "joint_hidden": joint["jointnet"]["joint_hidden"],
            "activation": joint["jointnet"]["activation"],
        },
        "decoding": {
            "model_type": decoding["model_type"],
            "durations": decoding["durations"],
            "max_symbols": decoding["greedy"]["max_symbols"],
        },
    }


def export_parakeet(version: str, nemo_path: Path, out: Path) -> None:
    spec = VERSIONS[version]
    target = out / f"parakeet-{version}"
    target.mkdir(parents=True, exist_ok=True)
    print(f"[{version}] reading {nemo_path}")

    with tarfile.open(nemo_path) as archive:
        cfg = yaml.safe_load(archive.extractfile(tar_member(archive, "model_config.yaml")))
        state = torch.load(
            archive.extractfile(tar_member(archive, "model_weights.ckpt")),
            map_location="cpu",
            weights_only=True,
        )
        tokenizer_model = archive.extractfile(tar_member(archive, "_tokenizer.model")).read()

    config = model_config(cfg, version)
    vocab_size = config["decoder"]["vocab_size"]
    if vocab_size != spec["blank_id"]:
        raise SystemExit(f"[{version}] vocab size {vocab_size} != blank id {spec['blank_id']}")

    tensors = {}
    for name, tensor in state.items():
        if not name.startswith(KEEP_PREFIXES) or name.endswith("num_batches_tracked"):
            continue
        if not tensor.is_floating_point():
            raise SystemExit(f"[{version}] unexpected non-float tensor {name}")
        tensors[name] = tensor.detach().to(torch.float32).contiguous()

    joint_out = tensors["joint.joint_net.2.weight"].shape[0]
    expected_joint = vocab_size + 1 + len(config["decoding"]["durations"])
    if joint_out != expected_joint:
        raise SystemExit(f"[{version}] joint width {joint_out} != {expected_joint}")

    metadata = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "model": "parakeet-tdt-0.6b",
        "source": spec["source"],
        "source_sha256": sha256_file(nemo_path),
        "exporter": f"nemo-toolkit {importlib.metadata.version('nemo-toolkit')}",
        "config": json.dumps(config, sort_keys=True),
    }
    save_file(tensors, target / "model.safetensors", metadata=metadata)
    print(f"[{version}] wrote {len(tensors)} tensors to {target / 'model.safetensors'}")

    write_vocab(version, tokenizer_model, vocab_size, target / spec["vocab_file"])


def write_vocab(version: str, tokenizer_model: bytes, vocab_size: int, path: Path) -> None:
    processor = sentencepiece.SentencePieceProcessor(model_proto=tokenizer_model)
    if processor.get_piece_size() != vocab_size:
        raise SystemExit(
            f"[{version}] tokenizer has {processor.get_piece_size()} pieces, expected {vocab_size}"
        )
    pieces = [processor.id_to_piece(index) for index in range(vocab_size)]

    if version == "v2":
        # SentencePiece text vocab with the RNNT blank appended at id `vocab_size`
        lines = [f"{piece} {index}" for index, piece in enumerate(pieces)]
        lines.append(f"<blk> {vocab_size}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        # the v3 blank is one past the table, so it is not written; the published
        # `parakeet_vocab.json` spells the SentencePiece word marker as a space,
        # which language filtering and detokenization rely on
        table = {str(index): piece.replace("▁", " ") for index, piece in enumerate(pieces)}
        path.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{version}] wrote {vocab_size} tokens to {path}")


def export_silero(out: Path) -> None:
    target = out / "vad"
    target.mkdir(parents=True, exist_ok=True)
    package_version = importlib.metadata.version("silero-vad")
    if not package_version.startswith("5."):
        raise SystemExit(f"silero-vad {package_version} does not ship the v5 weights")

    jit_path = importlib.resources.files("silero_vad").joinpath("data/silero_vad.jit")
    model = torch.jit.load(str(jit_path), map_location="cpu")
    # `_model` is the 16 kHz branch; `_model_8k` is not used by scriptrs
    tensors = {
        name.removeprefix("_model."): tensor.detach().to(torch.float32).contiguous()
        for name, tensor in model.state_dict().items()
        if name.startswith("_model.")
    }
    metadata = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "model": "silero-vad-v5",
        "source": f"silero-vad {package_version} (16 kHz branch)",
        "source_sha256": sha256_file(Path(str(jit_path))),
        "config": json.dumps(
            {"sample_rate": 16000, "window": 512, "context": 64, "state": 128}, sort_keys=True
        ),
    }
    save_file(tensors, target / "silero-vad-v5.safetensors", metadata=metadata)
    print(f"[vad] wrote {len(tensors)} tensors to {target / 'silero-vad-v5.safetensors'}")


def main() -> None:
    args = parse_args()
    only = set(args.only or ["v2", "v3", "vad"])
    if "v2" in only:
        export_parakeet("v2", args.v2_nemo, args.out)
    if "v3" in only:
        export_parakeet("v3", args.v3_nemo, args.out)
    if "vad" in only:
        export_silero(args.out)


if __name__ == "__main__":
    main()
