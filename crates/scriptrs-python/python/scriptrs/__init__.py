"""Fast NVIDIA Parakeet TDT speech-to-text, on Core ML (macOS) and CUDA (Linux).

Pick a model when you load the pipeline, then transcribe files or 16 kHz mono
samples::

    import scriptrs

    pipeline = scriptrs.Pipeline.from_pretrained("parakeet/v2")  # or "parakeet/v3" for 25 languages
    result = pipeline.transcribe_file("meeting.wav")
    print(result.text)

Inference runs in Rust with the GIL released, so other Python threads keep
running while a call is in progress.
"""

from __future__ import annotations

from enum import Enum
from importlib import metadata
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Sequence, Union

from . import _cuda, _native
from ._errors import ScriptrsError
from ._native import (
    ALL_LANGUAGES,
    BACKEND,
    TimedToken,
    TranscriptChunk,
    TranscriptionResult,
)

if TYPE_CHECKING:
    from typing import Any, Protocol

    from typing_extensions import Buffer

    class _ArrayInterface(Protocol):
        # NumPy only declares __buffer__ for Python 3.12+, so match its array
        # interface too
        @property
        def __array_interface__(self) -> dict[str, Any]: ...

    Samples = Union[Buffer, _ArrayInterface, Sequence[float]]

try:
    __version__ = metadata.version("scriptrs")
except metadata.PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0"


class Model(str, Enum):
    """Which model to run.

    Values are ``"<family>/<version>"``, so other model families can be added
    beside Parakeet. Anywhere a model is taken, its value, such as
    ``"parakeet/v3"``, or the upstream model id, such as
    ``"nvidia/parakeet-tdt-0.6b-v3"``, works too.
    """

    PARAKEET_V2 = "parakeet/v2"
    """NVIDIA Parakeet TDT 0.6B v2: English only, slightly more accurate on English."""
    PARAKEET_V3 = "parakeet/v3"
    """NVIDIA Parakeet TDT 0.6B v3: 25 European languages, detected automatically."""

    @property
    def languages(self) -> list[str]:
        """ISO 639-1 codes of the languages this model transcribes."""
        return _native.languages(self.value)


ModelName = Union[Model, Literal["parakeet/v2", "parakeet/v3"], str]
"""A :class:`Model`, its value, or its upstream model id."""

Precision = Literal["fp16", "tf32", "fp32"]
"""Precision of the CUDA encoder's matrix multiplies; Core ML ignores it.

``"fp16"`` is the default and the fastest. ``"tf32"`` keeps the weights in FP32,
and ``"fp32"`` matches NVIDIA NeMo's FP32 output token for token.
"""

StrPath = Union[str, "PathLike[str]"]

# upstream model ids, so a name copied from Hugging Face or NeMo works as is
_UPSTREAM_IDS = {
    "nvidia/parakeet-tdt-0.6b-v2": "parakeet/v2",
    "nvidia/parakeet-tdt-0.6b-v3": "parakeet/v3",
}
_PRECISIONS = ("fp16", "tf32", "fp32")


def _model(model: ModelName) -> Model:
    if isinstance(model, Model):
        return model
    name = str(model).strip().lower()
    try:
        return Model(_UPSTREAM_IDS.get(name, name))
    except ValueError:
        known = ", ".join(repr(known.value) for known in Model)
        raise ValueError(f"unknown model {model!r}; use one of {known}") from None


def _precision(precision: str) -> str:
    # checked before the driver check and any download, on every platform
    if precision not in _PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}; use one of {', '.join(_PRECISIONS)}")
    return precision


class Pipeline:
    """A loaded Parakeet model.

    Audio of any length works: a recording longer than one model window is split
    into overlapping windows and the transcripts are merged. Calls on one
    pipeline run one at a time.
    """

    def __init__(self, native: _native._Pipeline) -> None:
        """Use :meth:`from_pretrained` or :meth:`from_dir` instead."""
        self._native = native

    @classmethod
    def from_pretrained(
        cls,
        model: ModelName = Model.PARAKEET_V2,
        *,
        vad: bool = False,
        precision: Precision = "fp16",
        cache_dir: Optional[StrPath] = None,
    ) -> Pipeline:
        """Download ``model`` from Hugging Face if needed, then load it.

        The first call downloads the model, about 0.6 GB of Core ML models on
        macOS or 2.5 GB of weights on Linux, into ``cache_dir`` when given, else
        the Hugging Face hub cache that ``HF_HUB_CACHE`` or ``HF_HOME`` sets, else
        ``~/.cache/huggingface/hub``; later calls load from the cache. Setting
        ``SCRIPTRS_MODELS_DIR`` loads a local directory instead, as
        :meth:`from_dir` does.
        With ``vad``, the Silero VAD model is downloaded too, and long recordings
        are split at detected speech instead of at fixed intervals, which helps
        when most of a recording is silence or noise.

        Raises :class:`ScriptrsError` with code ``model_download`` when a
        download fails, or ``cuda`` when Linux has no NVIDIA driver.
        """
        model = _model(model)
        precision = _precision(precision)
        _cuda.prepare()
        cache = None if cache_dir is None else Path(cache_dir)
        return cls(_native._from_pretrained(model.value, vad, precision, cache))

    @classmethod
    def from_dir(
        cls,
        path: StrPath,
        model: ModelName = Model.PARAKEET_V2,
        *,
        vad: bool = False,
        precision: Precision = "fp16",
    ) -> Pipeline:
        """Load ``model`` from a local directory instead of downloading it.

        On macOS the directory holds Core ML models, and on Linux native weights,
        in the layouts the scriptrs README describes.
        """
        model = _model(model)
        precision = _precision(precision)
        _cuda.prepare()
        return cls(_native._from_dir(Path(path), model.value, vad, precision))

    @property
    def model(self) -> Model:
        """The loaded model."""
        return Model(self._native.model)

    @property
    def languages(self) -> list[str]:
        """ISO 639-1 codes of the languages the loaded model transcribes."""
        return self._native.languages

    @property
    def vad(self) -> bool:
        """Whether long recordings are split at speech detected by VAD."""
        return self._native.vad

    @property
    def max_chunk_seconds(self) -> float:
        """Longest input transcribed as one window.

        Longer audio still works through :meth:`transcribe`, which splits it into
        windows of this length: 15 s on Core ML, and 30 s on CUDA, where the
        model accepts 40 s but shorter windows are more accurate.
        """
        return self._native.max_chunk_seconds

    def transcribe(
        self, samples: Samples, *, language: Optional[str] = None
    ) -> TranscriptionResult:
        """Transcribe 16 kHz mono audio.

        ``samples`` is a 1-D float32 buffer, such as a NumPy array, or a sequence
        of floats; it is copied before the GIL is released. ``language`` is an
        ISO 639-1 code from :attr:`languages`, such as ``"de"``; without it,
        Parakeet v3 detects the language and v2 transcribes English.
        """
        return self._native.transcribe(samples, language)

    def transcribe_file(
        self, path: StrPath, *, language: Optional[str] = None
    ) -> TranscriptionResult:
        """Decode an audio file, resample it to 16 kHz mono, and transcribe it.

        WAV, FLAC, MP3, Ogg Vorbis, and AAC in MP4/M4A are supported. The whole
        file is decoded into memory.
        """
        return self._native.transcribe_file(Path(path), language)

    def transcribe_many(
        self,
        inputs: Sequence[Samples],
        *,
        language: Optional[str] = None,
        max_batch: int = 16,
    ) -> list[TranscriptionResult]:
        """Transcribe many 16 kHz mono clips faster than calling :meth:`transcribe` in a loop.

        Clips up to :attr:`max_chunk_seconds` go through a queue that prepares
        the next clips while the model runs, and on CUDA runs up to
        ``max_batch`` (1 to 64) of them in one pass. Longer clips are transcribed
        one at a time afterwards. Results keep input order. If any clip fails, the
        whole call raises :class:`ScriptrsError` naming that clip's index.
        """
        return self._native.transcribe_many(list(inputs), language, max_batch)

    def __repr__(self) -> str:
        return f"Pipeline(model={self.model.value!r}, backend={BACKEND!r}, vad={self.vad})"


__all__ = [
    "ALL_LANGUAGES",
    "BACKEND",
    "Model",
    "ModelName",
    "Pipeline",
    "Precision",
    "ScriptrsError",
    "TimedToken",
    "TranscriptChunk",
    "TranscriptionResult",
]
