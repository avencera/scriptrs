"""Find the NVIDIA driver and make pip-installed cuBLAS visible to the CUDA backend.

The native CUDA backend opens the driver and cuBLAS 12 at run time by name:
``libcuda.so.1``, then ``libcublas.so`` and ``libcublas.so.12`` on Linux, and
``nvcuda.dll``, ``cublas64_12.dll`` and ``cublasLt64_12.dll`` on Windows. The
``nvidia-cublas-cu12`` wheel installs cuBLAS inside site-packages, where neither
loader looks, so it is loaded here first. On Linux ``RTLD_GLOBAL`` makes glibc
resolve the ``libcublas.so.12`` lookup to the already-loaded copy, though a
``libcublas.so`` on the loader path, from a CUDA toolkit, still comes first.
Windows reuses a loaded DLL with the requested name, from whatever directory it
came from, so the wheel's copy wins there even over a toolkit on ``PATH``, which
ships the same names. A ``cublas.dll`` or ``cublas64.dll`` on the search path
would still load first, since the backend tries those names before
``cublas64_12.dll``.
"""

from __future__ import annotations

import ctypes
import sys
import threading
from pathlib import Path

from ._errors import ScriptrsError
from ._native import BACKEND

_WINDOWS = sys.platform == "win32"
# the driver's library, which the NVIDIA driver installs on the loader's own path
DRIVER = "nvcuda.dll" if _WINDOWS else "libcuda.so.1"
# the library directory inside the wheel's ``nvidia`` namespace package, and its
# libraries with dependencies before dependents, since libcublas links against
# libcublasLt by name and each load finds what it links against already loaded
_CUBLAS: tuple[tuple[str, ...], tuple[str, ...]] = (
    (("cublas", "bin"), ("cublasLt64_12.dll", "cublas64_12.dll"))
    if _WINDOWS
    else (("cublas", "lib"), ("libcublasLt.so.12", "libcublas.so.12"))
)
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
        # on Windows a bare name searches the application directory and System32,
        # where the driver puts nvcuda.dll, but not the current directory
        ctypes.CDLL(DRIVER)
    except OSError:
        raise ScriptrsError(
            "cuda",
            f"no NVIDIA driver found ({DRIVER}); on Linux and Windows scriptrs runs on NVIDIA GPUs only",
        ) from None


def _load_cublas() -> None:
    try:
        import nvidia  # namespace package shared by the nvidia-* wheels
    except ImportError:
        return

    subdir, names = _CUBLAS
    for root in getattr(nvidia, "__path__", []):
        lib_dir = Path(root).joinpath(*subdir)
        for name in names:
            path = lib_dir / name
            if path.is_file():
                try:
                    # Windows ignores the mode; an absolute path there also searches the
                    # DLL's own directory for its imports
                    ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    # a broken install falls back to the system search path
                    pass
