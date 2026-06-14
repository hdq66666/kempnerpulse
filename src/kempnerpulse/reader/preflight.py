"""Layer 1 (Read) — preflight checks for the direct DCGM backend.

Verifies that ``dcgmi`` is present and the DCGM host engine answers before a
stream is opened, and returns the discovery output so GPU IDs can be resolved.
Failures raise typed ``ReaderError``s carrying an actionable remediation string
rather than leaving a cryptic subprocess error to surface later.
"""
from __future__ import annotations

import subprocess

from .base import DcgmStreamError, HostEngineUnavailableError


def probe_dcgmi(timeout: float = 10.0) -> str:
    """Run ``dcgmi discovery -l`` and return its stdout.

    Raises:
        HostEngineUnavailableError: ``dcgmi`` is missing, timed out, or the host
            engine returned a non-zero status.
    """
    try:
        result = subprocess.run(
            ["dcgmi", "discovery", "-l"],
            capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise HostEngineUnavailableError(
            "dcgmi command not found.",
            remediation="Install NVIDIA DCGM, or use the prometheus backend.",
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise HostEngineUnavailableError(
            "dcgmi timed out.",
            remediation="Check that the DCGM host engine (nv-hostengine) is running.",
        ) from exc

    if result.returncode != 0:
        raise HostEngineUnavailableError(
            f"dcgmi discovery failed (exit {result.returncode}): "
            f"{result.stderr.strip()}",
            remediation="Check that the DCGM host engine is running and reachable.",
        )
    return result.stdout


__all__ = ["probe_dcgmi", "HostEngineUnavailableError", "DcgmStreamError"]
