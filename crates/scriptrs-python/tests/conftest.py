"""Shared fixtures: the test clip and one loaded pipeline per model."""

from __future__ import annotations

import ctypes
import os
import re
import wave
from pathlib import Path

import numpy as np
import pytest

import scriptrs

DATA = Path(__file__).resolve().parent / "data"
# LibriSpeech test-clean 1089-134686-0007, CC BY 4.0
CLIP = DATA / "librispeech-1089-134686-0007.wav"
CLIP_TEXT = "a cold lucid indifference reigned in his soul"


def read_wav(path: Path) -> np.ndarray:
    """16-bit mono 16 kHz WAV as float32."""
    with wave.open(str(path)) as wav:
        assert (wav.getnchannels(), wav.getframerate(), wav.getsampwidth()) == (1, 16_000, 2)
        frames = wav.readframes(wav.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def normalize(text: str) -> str:
    """Lowercase words without punctuation, to compare with LibriSpeech references."""
    return " ".join(re.findall(r"[a-z']+", text.lower()))


def backend_available() -> bool:
    if scriptrs.BACKEND == "cuda":
        try:
            ctypes.CDLL("libcuda.so.1")
        except OSError:
            return False
    return scriptrs.BACKEND is not None


# SCRIPTRS_TEST_MODELS=parakeet/v2 limits the model tests to one download
TEST_MODELS = os.environ.get("SCRIPTRS_TEST_MODELS", "parakeet/v2,parakeet/v3").split(",")


@pytest.fixture(scope="session", params=TEST_MODELS)
def pipeline(request: pytest.FixtureRequest) -> scriptrs.Pipeline:
    if not backend_available():
        pytest.skip(f"no scriptrs backend on this machine (BACKEND={scriptrs.BACKEND})")
    return scriptrs.Pipeline.from_pretrained(request.param)


@pytest.fixture(scope="session")
def clip() -> np.ndarray:
    return read_wav(CLIP)
