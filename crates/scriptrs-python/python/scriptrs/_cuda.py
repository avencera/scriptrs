"""Find the NVIDIA driver and make pip-installed cuBLAS visible to the CUDA backend.

The native CUDA backend opens the driver and cuBLAS at run time by name:
``libcuda.so.1`` and ``libcublas.so.12`` on Linux, and ``nvcuda.dll`` and
``cublas64_12.dll`` on Windows, then the CUDA 13 names. The
``nvidia-cublas-cu12`` wheel installs cuBLAS inside site-packages, where neither
loader looks, so it is loaded here first. Both loaders reuse a library the
process already holds under the name they ask for: glibc matches the
``libcublas.so.12`` lookup to the loaded copy's soname, and on Windows the
backend checks for a loaded module of that name before it searches. The wheel's
copy therefore wins over a CUDA toolkit's, which ships the same names.
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


def _driver_path() -> str:
    """The driver to load: System32's copy on Windows, the loader's soname elsewhere."""
    if sys.platform != "win32":
        return DRIVER
    # a full System32 path, as the backend loads the driver, since a bare name would
    # also search the interpreter's directory for a planted nvcuda.dll
    buffer = ctypes.create_unicode_buffer(260)
    length = ctypes.windll.kernel32.GetSystemDirectoryW(buffer, len(buffer))
    if not 0 < length < len(buffer):
        raise OSError("could not find the System32 directory")
    return str(Path(buffer.value) / DRIVER)


def _check_driver() -> None:
    try:
        ctypes.CDLL(_driver_path())
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
