from pathlib import Path
from typing import Any, Optional, Sequence, final

BACKEND: Optional[str]
"""Backend compiled into this build: ``"coreml"``, ``"cuda"``, or ``None``."""
ALL_LANGUAGES: list[str]
"""ISO 639-1 codes of every language a model transcribes."""

def languages(model: str) -> list[str]: ...

@final
class TimedToken:
    """One token with its timing and confidence."""

    @property
    def token_id(self) -> int:
        """Token id in the model vocabulary."""
    @property
    def text(self) -> str:
        """Token text, with a leading space where a word starts."""
    @property
    def start(self) -> float:
        """Start time in seconds."""
    @property
    def end(self) -> float:
        """End time in seconds."""
    @property
    def confidence(self) -> float:
        """Model confidence between 0 and 1."""

@final
class TranscriptChunk:
    """A timed span of the transcript."""

    @property
    def start(self) -> float:
        """Start time in seconds."""
    @property
    def end(self) -> float:
        """End time in seconds."""
    @property
    def text(self) -> str:
        """Text of this span."""

@final
class TranscriptionResult:
    """A finished transcription; ``str(result)`` is its text."""

    @property
    def text(self) -> str:
        """Full transcript text."""
    @property
    def chunks(self) -> list[TranscriptChunk]:
        """Timed spans of the transcript."""
    @property
    def tokens(self) -> list[TimedToken]:
        """Every token with its timing and confidence."""
    @property
    def duration_seconds(self) -> float:
        """Length of the input audio in seconds."""

@final
class _Pipeline:
    @property
    def model(self) -> str: ...
    @property
    def vad(self) -> bool: ...
    @property
    def languages(self) -> list[str]: ...
    @property
    def max_chunk_seconds(self) -> float: ...
    def transcribe(self, samples: Any, language: Optional[str] = None) -> TranscriptionResult: ...
    def transcribe_file(self, path: Path, language: Optional[str] = None) -> TranscriptionResult: ...
    def transcribe_many(
        self, inputs: Sequence[Any], language: Optional[str] = None, max_batch: int = 32
    ) -> list[TranscriptionResult]: ...

def _from_pretrained(
    model: str, vad: bool, precision: str, cache_dir: Optional[Path] = None
) -> _Pipeline: ...
def _from_dir(path: Path, model: str, vad: bool, precision: str) -> _Pipeline: ...
