"""Device-detection and memory-management helpers that work across CUDA, MPS, and CPU.

This module is the single source of truth for "what device do we run on" so the rest
of the codebase never hard-codes `"cuda"` again.

Resolution order:
1. KHALA_DEVICE env var (explicit override: "cuda" | "mps" | "cpu")
2. CUDA, if available
3. MPS (Apple Silicon), if available
4. CPU
"""
from __future__ import annotations

import os

import torch


def get_device() -> torch.device:
    """Return the preferred torch.device for this host.

    Honors KHALA_DEVICE if set. Falls back to CUDA, then MPS, then CPU.
    """
    override = os.environ.get("KHALA_DEVICE", "").strip().lower()
    if override:
        if override == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("KHALA_DEVICE=cuda but torch.cuda.is_available() is False")
        if override == "mps" and not getattr(torch.backends, "mps", None):
            raise RuntimeError("KHALA_DEVICE=mps but torch.backends.mps is not present")
        if override == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("KHALA_DEVICE=mps but torch.backends.mps.is_available() is False")
        return torch.device(override)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_str() -> str:
    """Convenience for legacy code that wants a string."""
    return str(get_device())


def is_cuda(device: torch.device | str | None = None) -> bool:
    d = torch.device(device) if device is not None else get_device()
    return d.type == "cuda"


def is_mps(device: torch.device | str | None = None) -> bool:
    d = torch.device(device) if device is not None else get_device()
    return d.type == "mps"


def empty_cache(device: torch.device | str | None = None) -> None:
    """Device-agnostic equivalent of torch.cuda.empty_cache().

    Safe to call regardless of device; degrades to no-op on CPU.
    """
    d = torch.device(device) if device is not None else get_device()
    if d.type == "cuda":
        torch.cuda.empty_cache()
    elif d.type == "mps":
        # torch.mps.empty_cache exists on torch >= 2.0
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "empty_cache"):
            mps.empty_cache()


def synchronize(device: torch.device | str | None = None) -> None:
    """Device-agnostic equivalent of torch.cuda.synchronize()."""
    d = torch.device(device) if device is not None else get_device()
    if d.type == "cuda":
        torch.cuda.synchronize()
    elif d.type == "mps":
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "synchronize"):
            mps.synchronize()


def clear_memory(device: torch.device | str | None = None) -> None:
    """gc + empty_cache + synchronize. The portable replacement for clear_cuda_memory()."""
    import gc

    gc.collect()
    empty_cache(device)
    synchronize(device)
    # ipc_collect is CUDA-only and only meaningful for multi-process IPC; skip elsewhere.
    if (torch.device(device) if device is not None else get_device()).type == "cuda":
        torch.cuda.ipc_collect()


def recommended_dtype(device: torch.device | str | None = None) -> torch.dtype:
    """Pick a sensible default dtype per device.

    - CUDA: bfloat16 (matches the upstream --bf16 launcher flag)
    - MPS:  float16  (MPS bf16 support is improving but numerically less stable)
    - CPU:  float32  (bf16 on CPU is slow without AVX-512 BF16)
    """
    d = torch.device(device) if device is not None else get_device()
    if d.type == "cuda":
        return torch.bfloat16
    if d.type == "mps":
        return torch.float16
    return torch.float32
