"""Integration tests for the scriptrs Python package."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import scriptrs
from conftest import CLIP, CLIP_TEXT, backend_available, normalize


def test_model_names_resolve_to_one_enum():
    for name in ("parakeet/v3", "Parakeet/V3", " parakeet/v3 ", "nvidia/parakeet-tdt-0.6b-v3"):
        assert scriptrs._model(name) is scriptrs.Model.PARAKEET_V3
    assert scriptrs._model(scriptrs.Model.PARAKEET_V2) is scriptrs.Model.PARAKEET_V2
    # a bare version would be ambiguous once there is more than one model family
    for name in ("v3", "parakeet/v4"):
        with pytest.raises(ValueError, match="'parakeet/v2', 'parakeet/v3'"):
            scriptrs._model(name)


def test_model_languages():
    assert scriptrs.Model.PARAKEET_V2.languages == ["en"]
    v3 = scriptrs.Model.PARAKEET_V3.languages
    assert len(v3) == 25 and {"en", "de", "fr", "uk"} <= set(v3)
    assert sorted(v3) == sorted(scriptrs.ALL_LANGUAGES)


def test_backend_matches_platform():
    expected = {"darwin": "coreml", "linux": "cuda"}.get(sys.platform)
    if sys.platform == "linux" and not _is_x86_64():
        expected = None
    assert scriptrs.BACKEND == expected


def _is_x86_64() -> bool:
    import platform

    return platform.machine() in ("x86_64", "AMD64")


@pytest.mark.skipif(
    scriptrs.BACKEND != "cuda" or backend_available(),
    reason="checks the error on a Linux machine without an NVIDIA driver",
)
def test_missing_driver_fails_before_download(tmp_path):
    with pytest.raises(scriptrs.ScriptrsError) as raised:
        scriptrs.Pipeline.from_pretrained("parakeet/v2", cache_dir=tmp_path)
    assert raised.value.code == "cuda"
    assert not any(tmp_path.iterdir())


def test_bad_precision_is_rejected_before_download(tmp_path):
    with pytest.raises(ValueError, match="precision"):
        scriptrs.Pipeline.from_pretrained("parakeet/v2", precision="int8", cache_dir=tmp_path)  # type: ignore[arg-type]
    assert not any(tmp_path.iterdir())


def test_hugging_face_cache_variables_pick_the_download_dir(monkeypatch, tmp_path):
    for name in ("SCRIPTRS_MODELS_DIR", "HF_HUB_CACHE", "HF_HOME"):
        monkeypatch.delenv(name, raising=False)
    assert scriptrs._cache_dir(None) is None

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert scriptrs._cache_dir(None) == tmp_path / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "cache"))
    assert scriptrs._cache_dir(None) == tmp_path / "cache"
    assert scriptrs._cache_dir(tmp_path / "explicit") == tmp_path / "explicit"
    # a local model directory replaces the download entirely
    monkeypatch.setenv("SCRIPTRS_MODELS_DIR", str(tmp_path / "models"))
    assert scriptrs._cache_dir(None) is None


@pytest.mark.model
def test_transcribes_file_and_samples(pipeline, clip):
    from_file = pipeline.transcribe_file(CLIP)
    assert normalize(from_file.text) == CLIP_TEXT
    assert str(from_file) == from_file.text
    assert from_file.duration_seconds == pytest.approx(len(clip) / 16_000)
    assert from_file.tokens and from_file.chunks
    assert all(0.0 <= token.start <= token.end <= from_file.duration_seconds + 0.1 for token in from_file.tokens)

    assert pipeline.transcribe(clip) == from_file
    assert pipeline.transcribe(clip.tolist()) == from_file
    # strided and big-endian buffers are copied into native float32
    assert pipeline.transcribe(np.repeat(clip, 2)[::2]) == from_file
    assert pipeline.transcribe(clip.astype(">f4")) == from_file


@pytest.mark.model
def test_rejects_bad_samples(pipeline, clip):
    with pytest.raises(TypeError, match="float32"):
        pipeline.transcribe(clip.astype(np.float64))
    with pytest.raises(ValueError, match="1-D"):
        pipeline.transcribe(clip.reshape(1, -1))
    with pytest.raises(TypeError):
        pipeline.transcribe("not audio")
    with pytest.raises(scriptrs.ScriptrsError) as raised:
        pipeline.transcribe(np.zeros(0, dtype=np.float32))
    assert raised.value.code == "empty_audio"


@pytest.mark.model
def test_language_selection(pipeline, clip):
    assert pipeline.languages == pipeline.model.languages
    assert normalize(pipeline.transcribe(clip, language="EN").text) == CLIP_TEXT
    unsupported = "de" if pipeline.model is scriptrs.Model.PARAKEET_V2 else "ja"
    with pytest.raises(scriptrs.ScriptrsError) as raised:
        pipeline.transcribe(clip, language=unsupported)
    assert raised.value.code == "unsupported_language"
    assert "en" in raised.value.message


@pytest.mark.model
def test_long_audio_is_split_and_merged(pipeline, clip):
    gap = np.zeros(16_000 * 2, dtype=np.float32)
    repeats = int(pipeline.max_chunk_seconds // 6) + 3
    long = np.concatenate([np.concatenate([clip, gap]) for _ in range(repeats)])
    assert len(long) / 16_000 > pipeline.max_chunk_seconds

    result = pipeline.transcribe(long)
    assert normalize(result.text) == " ".join([CLIP_TEXT] * repeats)


@pytest.mark.model
def test_transcribe_many_keeps_order(pipeline, clip):
    gap = np.zeros(16_000 * 2, dtype=np.float32)
    long = np.concatenate([np.concatenate([clip, gap]) for _ in range(int(pipeline.max_chunk_seconds // 6) + 3)])
    inputs = [clip, long, clip[: len(clip) // 2], clip]

    batch = pipeline.transcribe_many(inputs, max_batch=4)
    singles = [pipeline.transcribe(samples) for samples in inputs]
    assert [normalize(result.text) for result in batch] == [normalize(result.text) for result in singles]

    with pytest.raises(scriptrs.ScriptrsError, match="input 1") as raised:
        pipeline.transcribe_many([clip, np.zeros(0, dtype=np.float32)])
    assert raised.value.code == "empty_audio"

    with_nan = clip.copy()
    with_nan[100] = np.nan
    with pytest.raises(scriptrs.ScriptrsError, match="input 0") as raised:
        pipeline.transcribe_many([with_nan, clip])
    assert raised.value.code == "invalid_audio"
    with pytest.raises(scriptrs.ScriptrsError) as raised:
        pipeline.transcribe(with_nan)
    assert raised.value.code == "invalid_audio"

    for max_batch in (0, 65):
        with pytest.raises(scriptrs.ScriptrsError) as raised:
            pipeline.transcribe_many([clip], max_batch=max_batch)
        assert raised.value.code == "configuration"


@pytest.mark.model
def test_cli_prints_transcript(pipeline):
    def run(*files):
        return subprocess.run(
            [sys.executable, "-m", "scriptrs.cli", "transcribe", "--model", pipeline.model.value, *files],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    assert normalize(run(str(CLIP))) == CLIP_TEXT

    lines = run(str(CLIP), str(CLIP)).splitlines()
    assert lines[0] == lines[3] == f"==> {CLIP} <=="
    assert lines[2] == ""
    assert normalize(lines[1]) == normalize(lines[4]) == CLIP_TEXT


@pytest.mark.model
def test_vad_pipeline_skips_silence(pipeline, clip):
    vad = scriptrs.Pipeline.from_pretrained(pipeline.model, vad=True)
    assert vad.vad and not pipeline.vad

    silence = np.zeros(16_000 * 20, dtype=np.float32)
    sparse = np.concatenate([silence, clip, silence, clip, silence])
    # VAD regions can clip the first word of a burst, which turns "a cold" into
    # "cold" or "and cold", so match the rest of the sentence
    text = normalize(vad.transcribe(sparse).text)
    assert text.count("cold lucid indifference reigned in his soul") == 2
