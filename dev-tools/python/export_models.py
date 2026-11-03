# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface_hub>=0.31",
# ]
# ///
"""Stage upstream Parakeet and VAD model assets into the local scriptrs layout"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[2]
PARAKEET_COREML_REPO = "FluidInference/parakeet-tdt-0.6b-v2-coreml"
PARAKEET_VOCAB_REPO = "istupakov/parakeet-tdt-0.6b-v2-onnx"
# staged directory: (repo, revision); both builds share one Core ML graph and I/O,
# v6.2.1 has the same weights as Silero 6.2.3
SILERO_VAD_REPOS = {
    "silero-vad.mlmodelc": ("aufklarer/Silero-VAD-v5-CoreML", None),
    "silero-vad-v6.mlmodelc": (
        "aufklarer/Silero-VAD-v6.2.1-CoreML",
        "523876545a57961474fee9df913e833e130560b8",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and stage model assets for scriptrs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "fixtures/models",
        help="Directory where the staged models should be written",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading Parakeet CoreML bundles...")
    parakeet_coreml_dir = download_snapshot(
        PARAKEET_COREML_REPO,
        [
            "Encoder.mlmodelc/*",
            "Decoder.mlmodelc/*",
            "JointDecision.mlmodelc/*",
        ],
    )

    print("Downloading Parakeet ONNX reference assets...")
    parakeet_onnx_dir = download_snapshot(
        PARAKEET_VOCAB_REPO,
        [
            "config.json",
            "vocab.txt",
            "encoder-model.onnx",
            "encoder-model.onnx.data",
            "decoder_joint-model.onnx",
        ],
    )

    print("Downloading Silero VAD CoreML bundles...")
    vad_dirs = {
        name: download_snapshot(repo, ["silero_vad.mlmodelc/*"], revision)
        for name, (repo, revision) in SILERO_VAD_REPOS.items()
    }

    stage_parakeet(parakeet_coreml_dir, parakeet_onnx_dir, output_dir)
    for name, vad_dir in vad_dirs.items():
        copy_bundle(vad_dir / "silero_vad.mlmodelc", output_dir / "vad" / name)
    print(f"Staged models in {output_dir}")


def download_snapshot(
    repo_id: str, allow_patterns: list[str], revision: str | None = None
) -> Path:
    return Path(
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            revision=revision,
        )
    )


def stage_parakeet(coreml_dir: Path, onnx_dir: Path, output_dir: Path) -> None:
    parakeet_dir = output_dir / "parakeet-v2"
    if parakeet_dir.exists():
        shutil.rmtree(parakeet_dir)
    copy_bundle(coreml_dir / "Encoder.mlmodelc", parakeet_dir / "encoder.mlmodelc")
    copy_bundle(coreml_dir / "Decoder.mlmodelc", parakeet_dir / "decoder.mlmodelc")
    copy_bundle(
        coreml_dir / "JointDecision.mlmodelc",
        parakeet_dir / "joint-decision.mlmodelc",
    )
    copy_file(onnx_dir / "vocab.txt", parakeet_dir / "vocab.txt")
    copy_file(onnx_dir / "config.json", parakeet_dir / "config.json")
    copy_file(onnx_dir / "encoder-model.onnx", parakeet_dir / "onnx/encoder-model.onnx")
    copy_file(
        onnx_dir / "encoder-model.onnx.data",
        parakeet_dir / "onnx/encoder-model.onnx.data",
    )
    copy_file(
        onnx_dir / "decoder_joint-model.onnx",
        parakeet_dir / "onnx/decoder_joint-model.onnx",
    )


def copy_bundle(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


if __name__ == "__main__":
    main()
