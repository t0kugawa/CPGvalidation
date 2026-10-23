"""Simulator-independent runtime configuration.

The CLI validates ``--threads`` here and ``configure_kernel`` applies it to an
injected simulator kernel, so both paths are testable without NEST installed.
"""
from __future__ import annotations

from typing import Any


def validate_threads(threads: int) -> int:
    """Return ``threads`` as a positive int, or raise ``ValueError``."""
    value = int(threads)
    if value < 1:
        raise ValueError(f"threads must be >= 1, got {threads!r}")
    return value


def kernel_status(*, resolution_ms: float, seed: int,
                  threads: int) -> dict[str, Any]:
    """The NEST kernel status used by every run (one source of truth)."""
    return {
        "resolution": float(resolution_ms),
        "print_time": False,
        "rng_seed": 1 + int(seed) % (2**32 - 2),
        "local_num_threads": validate_threads(threads),
    }


def configure_kernel(nest_module: Any, *, resolution_ms: float, seed: int,
                     threads: int) -> dict[str, Any]:
    """Reset ``nest_module`` and apply the run's kernel status.

    Validation happens before the kernel is touched, so an invalid thread
    count cannot leave a caller with a half-reset kernel.
    """
    status = kernel_status(
        resolution_ms=resolution_ms, seed=seed, threads=threads)
    nest_module.ResetKernel()
    nest_module.SetKernelStatus(status)
    return status
