"""Acceleration backend detection and resource caps.

The generator pulls a numpy-like module from `get_xp()`. When a CUDA backend is
chosen, we still surface results as plain numpy arrays at the boundary so the
rest of the pipeline (PLY writer, point-cloud renderer, sim integration) does
not need to care about the backend.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import psutil

_log = logging.getLogger(__name__)

BACKENDS = ("auto", "cuda_cupy", "cuda_torch", "taichi", "cpu_numpy")


@dataclass
class BackendReport:
    cuda_cupy: bool
    cuda_torch: bool
    taichi: bool
    has_nvidia: bool
    chosen: str  # one of BACKENDS minus "auto"
    notes: list[str]


def _have_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _cupy_cuda_available() -> bool:
    if not _have_module("cupy"):
        return False
    try:
        import cupy as cp  # type: ignore
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


def _torch_cuda_available() -> bool:
    if not _have_module("torch"):
        return False
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def detect_backends(prefer_gpu: bool = True) -> BackendReport:
    """Detect available acceleration backends and pick a default.

    Priority when prefer_gpu: cuda_cupy → cuda_torch → taichi → cpu_numpy.
    Taichi is the explicit cross-vendor fallback when no CUDA is available.
    """
    cupy_ok = _cupy_cuda_available()
    torch_ok = _torch_cuda_available()
    taichi_ok = _have_module("taichi")
    nvidia_ok = _have_module("pynvml")

    notes: list[str] = []
    if not prefer_gpu:
        chosen = "cpu_numpy"
        notes.append("prefer_gpu=False → cpu_numpy")
    elif cupy_ok:
        chosen = "cuda_cupy"
    elif torch_ok:
        chosen = "cuda_torch"
    elif taichi_ok:
        chosen = "taichi"
    else:
        chosen = "cpu_numpy"
        notes.append("no GPU backend available; using NumPy on CPU")

    return BackendReport(
        cuda_cupy=cupy_ok,
        cuda_torch=torch_ok,
        taichi=taichi_ok,
        has_nvidia=nvidia_ok,
        chosen=chosen,
        notes=notes,
    )


def resolve_backend(name: str, prefer_gpu: bool = True) -> str:
    if name == "auto":
        return detect_backends(prefer_gpu).chosen
    if name not in BACKENDS:
        _log.warning("unknown backend %r; falling back to auto", name)
        return detect_backends(prefer_gpu).chosen
    # If the user picked an unavailable backend, downgrade silently.
    rep = detect_backends(prefer_gpu)
    available = {
        "cuda_cupy": rep.cuda_cupy,
        "cuda_torch": rep.cuda_torch,
        "taichi": rep.taichi,
        "cpu_numpy": True,
    }
    if not available.get(name, False):
        _log.warning("backend %s not available; using %s", name, rep.chosen)
        return rep.chosen
    return name


def get_xp(backend: str) -> Any:
    """Return a numpy-compatible module for the chosen backend.

    For taichi we currently use numpy at the boundary and rely on taichi
    kernels inside specific hot paths (carving, sampling). The generator can
    therefore treat all backends uniformly via this `xp` handle.
    """
    if backend == "cuda_cupy":
        try:
            import cupy as cp  # type: ignore
            return cp
        except Exception:
            return np
    if backend == "cuda_torch":
        # Surface numpy for ergonomics; specific ops can opt into torch tensors.
        return np
    return np


# Module-level "current" backend so generation hot paths can opt into the GPU
# without every caller having to thread the backend through their signatures.
_active_backend: str = "cpu_numpy"


def set_active_backend(name: str, prefer_gpu: bool = True) -> str:
    """Resolve and remember the active backend. Returns the resolved name."""
    global _active_backend
    _active_backend = resolve_backend(name, prefer_gpu)
    _log.info("active acceleration backend: %s", _active_backend)
    return _active_backend


def active_backend() -> str:
    return _active_backend


def is_gpu(backend: str | None = None) -> bool:
    return (backend or _active_backend) in ("cuda_cupy", "cuda_torch")


def to_numpy(arr: Any) -> np.ndarray:
    """Convert backend-native arrays to numpy for I/O and rendering."""
    if isinstance(arr, np.ndarray):
        return arr
    # CuPy
    if hasattr(arr, "get") and arr.__class__.__module__.startswith("cupy"):
        return arr.get()
    # Torch
    if arr.__class__.__module__.startswith("torch"):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


# ---- live RAM / VRAM monitoring -------------------------------------------------


def process_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def system_ram_mb() -> tuple[float, float]:
    vm = psutil.virtual_memory()
    return vm.used / 1024 / 1024, vm.total / 1024 / 1024


def vram_mb() -> tuple[float, float] | None:
    """Return (used_mb, total_mb) for the first NVIDIA GPU, or None."""
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.used / 1024 / 1024, info.total / 1024 / 1024
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


def estimate_generation_ram_mb(n_points: int) -> float:
    """Rough upper-bound estimate for a generation pass.

    Each point: 3 floats positions + 3 floats colors + working copies.
    Use a 4x safety multiplier for primitive scratchpad and feature buffers.
    """
    return n_points * 24 * 4 / 1024 / 1024
