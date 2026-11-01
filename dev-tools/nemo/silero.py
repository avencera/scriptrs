"""Silero VAD TorchScript models scriptrs ships, fetched from Silero's release tags

Both versions ship, so neither can come from the `silero-vad` pip package, which
installs one. Each file is pinned by SHA-256, the hash the exported weights record
as `source_sha256`
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

# version: (release tag, SHA-256 of `src/silero_vad/data/silero_vad.jit` at that tag)
VERSIONS = {
    "v5": ("v5.1.2", "85c48e1f0ecb604e5d2a268f3ccfb912d4f7e935acdc86af5a3fc5b0aea7b29a"),
    # 6.2.1 to 6.2.3 ship the same weights; 6.0 and 6.1 ship older v6 weights
    "v6": ("v6.2.3", "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720"),
}

CACHE = Path(__file__).resolve().parent / ".cache" / "silero"


def jit_path(version: str) -> Path:
    """The version's TorchScript file, downloaded once and checked against its pin"""
    tag, sha256 = VERSIONS[version]
    path = CACHE / tag / "silero_vad.jit"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://github.com/snakers4/silero-vad/raw/{tag}/src/silero_vad/data/silero_vad.jit"
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(url, partial)
        partial.rename(path)
    found = hashlib.sha256(path.read_bytes()).hexdigest()
    if found != sha256:
        raise SystemExit(f"{path} has SHA-256 {found}, expected {sha256} for Silero {tag}")
    return path


def source(version: str) -> str:
    """`source` metadata of the exported weights, the pip release the file matches"""
    return f"silero-vad {VERSIONS[version][0].removeprefix('v')} (16 kHz branch)"
