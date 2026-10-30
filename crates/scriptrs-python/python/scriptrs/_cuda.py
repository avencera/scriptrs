"""Find the NVIDIA driver and make pip-installed cuBLAS visible to the CUDA backend.

The native CUDA backend looks cuBLAS up by name, ``libcublas.so`` and then
``libcublas.so.12``. The ``nvidia-cublas-cu12`` wheel installs it inside
site-packages, where the dynamic loader does not look, so it is loaded here
first: glibc then resolves the ``libcublas.so.12`` lookup to the already-loaded
copy. A ``libcublas.so`` on the loader path, from a CUDA toolkit, still comes
first.
"""

from __future__ import annotations

import ctypes
import threading
from pathlib import Path

from ._errors import ScriptrsError
from ._native import BACKEND

# dependencies before dependents, so each load finds what it links against
_CUBLAS = ("libcublasLt.so.12", "libcublas.so.12")
_prepared = False
# a second thread must wait for the first to finish loading, not skip ahead
_lock = threading.Lock()


def prepare() -> None:
    """Check for the driver and load cuBLAS once, before the first model loads.

    The driver check runs before any download, so a machine without an NVIDIA
    GPU fails fast instead of after fetching a model it cannot run.
    """
    global _prepared
    if BACKEND != "cuda":
        return
    with _lock:
        if not _prepared:
            _check_driver()
            _load_cublas()
            _prepared = True


def _check_driver() -> None:
    try:
        ctypes.CDLL("libcuda.so.1")
    except OSError:
        raise ScriptrsError(
            "cuda",
            "no NVIDIA driver found (libcuda.so.1); on Linux scriptrs runs on NVIDIA GPUs only",
        ) from None


def _load_cublas() -> None:
    try:
        import nvidia  # namespace package shared by the nvidia-* wheels
    except ImportError:
        return

    for root in getattr(nvidia, "__path__", []):
        lib_dir = Path(root) / "cublas" / "lib"
        for name in _CUBLAS:
            path = lib_dir / name
            if path.is_file():
                try:
                    ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    # a broken install falls back to the system search path
                    pass
