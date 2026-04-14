# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "coremltools>=9.0",
#   "huggingface_hub>=0.31",
#   "numpy",
#   "onnxruntime>=1.20",
# ]
# ///
"""Benchmark staged and upstream CoreML encoder variants"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import coremltools as ct
import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO_ID = "FluidInference/parakeet-tdt-0.6b-v2-coreml"
DEFAULT_REMOTE_VARIANTS = [
    "ParakeetEncoder_v2.mlmodelc",
    "ParakeetEncoder_4bit_par.mlmodelc",
]


@dataclass(frozen=True)
class EncoderSchema:
    input_name: str
    length_name: str
    output_name: str
    output_length_name: str
    feature_size: int
    max_frames: int


@dataclass(frozen=True)
class EncoderVariant:
    label: str
    path: Path
    schema: EncoderSchema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark local and upstream CoreML encoder variants",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=REPO_ROOT / "fixtures/models",
        help="Directory containing the staged scriptrs models",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face repo that hosts alternate encoder bundles",
    )
    parser.add_argument(
        "--remote-variant",
        action="append",
        dest="remote_variants",
        help="Remote encoder bundle to benchmark in addition to the staged encoder",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup runs per encoder variant",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Timed runs per encoder variant",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for synthetic mel inputs",
    )
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=REPO_ROOT / "fixtures/models/parakeet-v2/onnx",
        help="ONNX directory used for encoder parity checks",
    )
    parser.add_argument(
        "--encoder-atol",
        type=float,
        default=2e-1,
        help="Absolute tolerance for encoder parity checks",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    remote_variants = args.remote_variants or DEFAULT_REMOTE_VARIANTS
    variants = [load_local_variant(args.models_dir)]
    variants.extend(load_remote_variant(args.repo_id, variant) for variant in remote_variants)
    onnx_encoder = ort.InferenceSession(
        str(args.onnx_dir / "encoder-model.onnx"),
        providers=["CPUExecutionProvider"],
    )

    rng = np.random.default_rng(args.seed)
    print(f"repo_id: {args.repo_id}")
    print(f"warmup_runs: {args.warmup}")
    print(f"timed_runs: {args.runs}")

    for variant in variants:
        benchmark_variant(
            variant,
            onnx_encoder,
            rng,
            args.warmup,
            args.runs,
            args.encoder_atol,
        )


def load_local_variant(models_dir: Path) -> EncoderVariant:
    path = models_dir / "parakeet-v2/encoder.mlmodelc"
    return EncoderVariant(
        label="staged",
        path=path,
        schema=load_schema(path),
    )


def load_remote_variant(repo_id: str, variant_name: str) -> EncoderVariant:
    snapshot_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"{variant_name}/*"],
        )
    )
    path = snapshot_dir / variant_name
    return EncoderVariant(
        label=variant_name.removesuffix(".mlmodelc"),
        path=path,
        schema=load_schema(path),
    )


def load_schema(bundle_dir: Path) -> EncoderSchema:
    metadata = json.loads((bundle_dir / "metadata.json").read_text())
    schema = metadata[0]
    input_schema = {entry["name"]: entry for entry in schema["inputSchema"]}
    output_schema = {entry["name"]: entry for entry in schema["outputSchema"]}
    input_name = first_present(input_schema, "mel", "audio_signal")
    length_name = first_present(input_schema, "mel_length", "length")
    output_name = first_present(output_schema, "encoder", "encoder_output")
    output_length_name = first_present(output_schema, "encoder_length", "encoder_output_length")
    input_shape = parse_shape(input_schema[input_name]["shape"])
    feature_size = input_shape[1]
    max_frames = input_shape[2]
    return EncoderSchema(
        input_name=input_name,
        length_name=length_name,
        output_name=output_name,
        output_length_name=output_length_name,
        feature_size=feature_size,
        max_frames=max_frames,
    )


def first_present(entries: dict[str, object], *names: str) -> str:
    for name in names:
        if name in entries:
            return name
    raise KeyError(f"expected one of {names}, found {list(entries)}")


def parse_shape(raw_shape: object) -> list[int]:
    if isinstance(raw_shape, list):
        return [int(value) for value in raw_shape]
    if isinstance(raw_shape, str):
        return [int(value) for value in json.loads(raw_shape)]
    raise TypeError(f"unsupported shape value: {raw_shape!r}")


def benchmark_variant(
    variant: EncoderVariant,
    onnx_encoder: ort.InferenceSession,
    rng: np.random.Generator,
    warmup: int,
    runs: int,
    encoder_atol: float,
) -> None:
    mel = rng.standard_normal(
        (1, variant.schema.feature_size, variant.schema.max_frames),
        dtype=np.float32,
    )
    length = np.array([variant.schema.max_frames], dtype=np.int32)
    model = ct.models.CompiledMLModel(
        str(variant.path),
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )

    inputs = {
        variant.schema.input_name: mel,
        variant.schema.length_name: length,
    }

    for _ in range(warmup):
        model.predict(inputs)

    durations = []
    for _ in range(runs):
        started = time.perf_counter()
        outputs = model.predict(inputs)
        durations.append(time.perf_counter() - started)

    output = np.asarray(outputs[variant.schema.output_name])
    output_length = int(np.asarray(outputs[variant.schema.output_length_name]).reshape(-1)[0])
    onnx_output = np.asarray(
        onnx_encoder.run(
            ["outputs"],
            {
                "audio_signal": mel,
                "length": np.array([variant.schema.max_frames], dtype=np.int64),
            },
        )[0],
        dtype=np.float32,
    )
    coreml_output = align_encoder_output(output.astype(np.float32), onnx_output.shape)
    encoder_max_abs = float(np.max(np.abs(onnx_output - coreml_output)))
    mean_seconds = statistics.fmean(durations)
    p50_seconds = percentile(durations, 0.50)
    p95_seconds = percentile(durations, 0.95)
    micros_per_frame = mean_seconds * 1_000_000.0 / variant.schema.max_frames

    print(f"variant: {variant.label}")
    print(f"  path: {variant.path}")
    print(
        "  schema:"
        f" input={variant.schema.input_name}"
        f" length={variant.schema.length_name}"
        f" output={variant.schema.output_name}"
        f" output_length={variant.schema.output_length_name}"
        f" max_frames={variant.schema.max_frames}"
    )
    print(
        "  outputs:"
        f" shape={list(output.shape)}"
        f" output_length={output_length}"
    )
    print(
        "  parity:"
        f" encoder_max_abs={encoder_max_abs:.6e}"
        f" atol={encoder_atol:.6e}"
        f" passes={encoder_max_abs <= encoder_atol}"
    )
    print(
        "  benchmark:"
        f" mean_ms={mean_seconds * 1000.0:.3f}"
        f" p50_ms={p50_seconds * 1000.0:.3f}"
        f" p95_ms={p95_seconds * 1000.0:.3f}"
        f" us_per_frame={micros_per_frame:.3f}"
    )


def align_encoder_output(output: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    if tuple(output.shape) == target_shape:
        return output
    if output.ndim == 3 and tuple(output.transpose(0, 2, 1).shape) == target_shape:
        return output.transpose(0, 2, 1)
    raise ValueError(f"unable to align encoder output shape {output.shape} to {target_shape}")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    index = int(round((len(ordered) - 1) * fraction))
    return ordered[index]


if __name__ == "__main__":
    main()
