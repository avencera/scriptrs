# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "coremltools>=9.0",
#   "numpy",
#   "onnxruntime>=1.20",
# ]
# ///
"""Compare staged CoreML Parakeet bundles against the ONNX reference"""

from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[2]
DECODER_HIDDEN_SIZE = 640
DECODER_LAYERS = 2
FEATURE_SIZE = 128
MAX_MEL_FRAMES = 1501


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare staged Parakeet CoreML bundles against the ONNX reference",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=REPO_ROOT / "fixtures/models",
        help="Directory containing the staged scriptrs models",
    )
    parser.add_argument(
        "--encoder-atol",
        type=float,
        default=2e-1,
        help="Absolute tolerance for encoder parity",
    )
    parser.add_argument(
        "--state-atol",
        type=float,
        default=5e-2,
        help="Absolute tolerance for decoder recurrent state parity",
    )
    parser.add_argument(
        "--prob-atol",
        type=float,
        default=5e-2,
        help="Absolute tolerance for token probability parity",
    )
    parser.add_argument(
        "--greedy-steps",
        type=int,
        default=64,
        help="Maximum greedy decode steps to compare after encoder parity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parakeet_dir = args.models_dir / "parakeet-v2"
    onnx_dir = parakeet_dir / "onnx"
    vocab_size = count_vocab_entries(parakeet_dir / "vocab.txt")

    encoder_ort = ort.InferenceSession(
        str(onnx_dir / "encoder-model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    decoder_joint_ort = ort.InferenceSession(
        str(onnx_dir / "decoder_joint-model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    encoder_coreml = ct.models.CompiledMLModel(
        str(parakeet_dir / "encoder.mlmodelc"),
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    decoder_coreml = ct.models.CompiledMLModel(
        str(parakeet_dir / "decoder.mlmodelc"),
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    joint_coreml = ct.models.CompiledMLModel(
        str(parakeet_dir / "joint-decision.mlmodelc"),
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )

    np.random.seed(0)
    mel = np.random.randn(1, FEATURE_SIZE, MAX_MEL_FRAMES).astype(np.float32)
    mel_length_coreml = np.array([MAX_MEL_FRAMES], dtype=np.int32)
    mel_length_onnx = np.array([MAX_MEL_FRAMES], dtype=np.int64)

    onnx_encoder = np.asarray(
        encoder_ort.run(
            ["outputs"],
            {
                "audio_signal": mel,
                "length": mel_length_onnx,
            },
        )[0],
        dtype=np.float32,
    )
    coreml_encoder = np.asarray(
        encoder_coreml.predict(
            {
                "mel": mel,
                "mel_length": mel_length_coreml,
            }
        )["encoder"],
        dtype=np.float32,
    )
    encoder_max_abs = max_abs(onnx_encoder, coreml_encoder)
    print(f"encoder max_abs={encoder_max_abs:.6e}")
    if encoder_max_abs > args.encoder_atol:
        raise SystemExit(
            f"encoder parity failed: {encoder_max_abs:.6e} > {args.encoder_atol:.6e}"
        )

    targets_coreml = np.array([[0]], dtype=np.int32)
    targets_onnx = np.array([[0]], dtype=np.int32)
    target_length_coreml = np.array([1], dtype=np.int32)
    target_length_onnx = np.array([1], dtype=np.int32)
    hidden_state = np.zeros((DECODER_LAYERS, 1, DECODER_HIDDEN_SIZE), dtype=np.float32)
    cell_state = np.zeros((DECODER_LAYERS, 1, DECODER_HIDDEN_SIZE), dtype=np.float32)
    encoder_step = onnx_encoder[:, :, :1].astype(np.float32)

    onnx_joint, _, onnx_hidden, onnx_cell = decoder_joint_ort.run(
        ["outputs", "prednet_lengths", "output_states_1", "output_states_2"],
        {
            "encoder_outputs": encoder_step,
            "targets": targets_onnx,
            "target_length": target_length_onnx,
            "input_states_1": hidden_state,
            "input_states_2": cell_state,
        },
    )
    onnx_joint = np.asarray(onnx_joint, dtype=np.float32).reshape(-1)
    onnx_token_logits = onnx_joint[:vocab_size]
    onnx_duration_logits = onnx_joint[vocab_size:]
    onnx_token_id = int(np.argmax(onnx_token_logits))
    onnx_duration = int(np.argmax(onnx_duration_logits))
    onnx_token_prob = softmax_max(onnx_token_logits)

    decoder_outputs = decoder_coreml.predict(
        {
            "targets": targets_coreml,
            "target_length": target_length_coreml,
            "h_in": hidden_state,
            "c_in": cell_state,
        }
    )
    decoder_step = np.asarray(decoder_outputs["decoder"], dtype=np.float32)
    coreml_hidden = np.asarray(decoder_outputs["h_out"], dtype=np.float32)
    coreml_cell = np.asarray(decoder_outputs["c_out"], dtype=np.float32)
    joint_outputs = joint_coreml.predict(
        {
            "encoder_step": encoder_step,
            "decoder_step": decoder_step,
        }
    )
    coreml_token_id = int(np.asarray(joint_outputs["token_id"], dtype=np.int32).reshape(-1)[0])
    coreml_duration = int(np.asarray(joint_outputs["duration"], dtype=np.int32).reshape(-1)[0])
    coreml_token_prob = float(
        np.asarray(joint_outputs["token_prob"], dtype=np.float32).reshape(-1)[0]
    )

    hidden_max_abs = max_abs(np.asarray(onnx_hidden, dtype=np.float32), coreml_hidden)
    cell_max_abs = max_abs(np.asarray(onnx_cell, dtype=np.float32), coreml_cell)
    prob_abs = abs(onnx_token_prob - coreml_token_prob)

    print(
        "decision"
        f" token_id={onnx_token_id}/{coreml_token_id}"
        f" duration={onnx_duration}/{coreml_duration}"
        f" token_prob_abs={prob_abs:.6e}"
        f" h_max_abs={hidden_max_abs:.6e}"
        f" c_max_abs={cell_max_abs:.6e}"
    )

    if onnx_token_id != coreml_token_id:
        raise SystemExit(
            f"token id mismatch: onnx={onnx_token_id} coreml={coreml_token_id}"
        )
    if onnx_duration != coreml_duration:
        raise SystemExit(
            f"duration mismatch: onnx={onnx_duration} coreml={coreml_duration}"
        )
    if prob_abs > args.prob_atol:
        raise SystemExit(
            f"token probability mismatch: {prob_abs:.6e} > {args.prob_atol:.6e}"
        )
    if hidden_max_abs > args.state_atol:
        raise SystemExit(
            f"hidden state mismatch: {hidden_max_abs:.6e} > {args.state_atol:.6e}"
        )
    if cell_max_abs > args.state_atol:
        raise SystemExit(
            f"cell state mismatch: {cell_max_abs:.6e} > {args.state_atol:.6e}"
        )

    compare_greedy_decode(
        onnx_encoder,
        coreml_encoder,
        decoder_joint_ort,
        decoder_coreml,
        joint_coreml,
        args.greedy_steps,
        vocab_size,
    )
    print("CoreML parity checks passed")


def compare_greedy_decode(
    onnx_encoder: np.ndarray,
    coreml_encoder: np.ndarray,
    decoder_joint_ort: ort.InferenceSession,
    decoder_coreml: ct.models.CompiledMLModel,
    joint_coreml: ct.models.CompiledMLModel,
    greedy_steps: int,
    vocab_size: int,
) -> None:
    blank_id = vocab_size - 1
    onnx_hidden = np.zeros((DECODER_LAYERS, 1, DECODER_HIDDEN_SIZE), dtype=np.float32)
    onnx_cell = np.zeros((DECODER_LAYERS, 1, DECODER_HIDDEN_SIZE), dtype=np.float32)
    coreml_hidden = np.zeros((DECODER_LAYERS, 1, DECODER_HIDDEN_SIZE), dtype=np.float32)
    coreml_cell = np.zeros((DECODER_LAYERS, 1, DECODER_HIDDEN_SIZE), dtype=np.float32)
    coreml_decoder_step = None
    coreml_next_hidden = None
    coreml_next_cell = None
    onnx_last_token = blank_id
    coreml_last_token = blank_id
    onnx_frame = 0
    coreml_frame = 0
    onnx_emitted = 0
    coreml_emitted = 0

    for step in range(greedy_steps):
        if onnx_frame >= onnx_encoder.shape[2] or coreml_frame >= coreml_encoder.shape[2]:
            print(
                "greedy"
                f" ended early at step={step}"
                f" onnx_frame={onnx_frame}"
                f" coreml_frame={coreml_frame}"
            )
            return

        onnx_frame_input = onnx_encoder[:, :, onnx_frame : onnx_frame + 1].astype(np.float32)
        onnx_outputs = decoder_joint_ort.run(
            ["outputs", "output_states_1", "output_states_2"],
            {
                "encoder_outputs": onnx_frame_input,
                "targets": np.array([[onnx_last_token]], dtype=np.int32),
                "target_length": np.array([1], dtype=np.int32),
                "input_states_1": onnx_hidden,
                "input_states_2": onnx_cell,
            },
        )
        onnx_logits = np.asarray(onnx_outputs[0], dtype=np.float32).reshape(-1)
        onnx_token_id = int(np.argmax(onnx_logits[:vocab_size]))
        onnx_duration = int(np.argmax(onnx_logits[vocab_size:]))
        if onnx_token_id != blank_id:
            onnx_hidden = np.asarray(onnx_outputs[1], dtype=np.float32)
            onnx_cell = np.asarray(onnx_outputs[2], dtype=np.float32)
            onnx_last_token = onnx_token_id
            onnx_emitted += 1
        onnx_frame, onnx_emitted = advance_state(
            onnx_frame,
            onnx_duration,
            onnx_token_id,
            blank_id,
            onnx_emitted,
        )

        coreml_frame_input = coreml_encoder[:, :, coreml_frame : coreml_frame + 1].astype(
            np.float32
        )
        if coreml_decoder_step is None:
            decoder_outputs = decoder_coreml.predict(
                {
                    "targets": np.array([[coreml_last_token]], dtype=np.int32),
                    "target_length": np.array([1], dtype=np.int32),
                    "h_in": coreml_hidden,
                    "c_in": coreml_cell,
                }
            )
            coreml_decoder_step = np.asarray(decoder_outputs["decoder"], dtype=np.float32)
            coreml_next_hidden = np.asarray(decoder_outputs["h_out"], dtype=np.float32)
            coreml_next_cell = np.asarray(decoder_outputs["c_out"], dtype=np.float32)
        joint_outputs = joint_coreml.predict(
            {
                "encoder_step": coreml_frame_input,
                "decoder_step": coreml_decoder_step,
            }
        )
        coreml_token_id = int(np.asarray(joint_outputs["token_id"], dtype=np.int32).reshape(-1)[0])
        coreml_duration = int(np.asarray(joint_outputs["duration"], dtype=np.int32).reshape(-1)[0])
        if coreml_token_id != blank_id:
            coreml_hidden = coreml_next_hidden
            coreml_cell = coreml_next_cell
            coreml_last_token = coreml_token_id
            coreml_emitted += 1
            coreml_decoder_step = None
            coreml_next_hidden = None
            coreml_next_cell = None
        coreml_frame, coreml_emitted = advance_state(
            coreml_frame,
            coreml_duration,
            coreml_token_id,
            blank_id,
            coreml_emitted,
        )

        if onnx_token_id != coreml_token_id or onnx_duration != coreml_duration:
            raise SystemExit(
                "greedy mismatch"
                f" step={step}"
                f" token_id={onnx_token_id}/{coreml_token_id}"
                f" duration={onnx_duration}/{coreml_duration}"
            )


def advance_state(
    frame_idx: int,
    duration: int,
    token_id: int,
    blank_id: int,
    emitted_tokens: int,
) -> tuple[int, int]:
    if duration > 0:
        return frame_idx + duration, 0

    if token_id == blank_id or emitted_tokens >= 10:
        return frame_idx + 1, 0

    return frame_idx, emitted_tokens


def count_vocab_entries(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right)))


def softmax_max(logits: np.ndarray) -> float:
    shifted = logits - np.max(logits)
    probs = np.exp(shifted)
    probs /= np.sum(probs)
    return float(np.max(probs))


if __name__ == "__main__":
    main()
