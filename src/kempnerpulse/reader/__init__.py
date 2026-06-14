"""KempnerPulse Layer 1 — Read.

Backends acquire raw data from a single source and emit a stream of opaque
``RawRecord``s in that source's own vocabulary. Interpreting those records
(canonical names, units, missing-value policy) belongs to Layer 2, not here.
``make_backend`` selects a backend from a ``ReaderConfig``.
"""
from .base import (
    Backend,
    BackendCaps,
    BackendKind,
    DcgmStreamError,
    ExporterCollisionDetected,
    HostEngineUnavailableError,
    RawRecord,
    ReaderConfig,
    ReaderError,
    ReservationDeniedError,
)


def make_backend(config: ReaderConfig) -> Backend:
    """Construct the backend named by ``config.backend`` (imported lazily)."""
    if config.backend is BackendKind.DCGMI:
        from .dcgmi import DcgmiBackend
        return DcgmiBackend()
    if config.backend is BackendKind.PROMETHEUS:
        from .prometheus import PrometheusBackend
        return PrometheusBackend()
    if config.backend is BackendKind.REPLAY:
        from .replay import ReplayBackend
        return ReplayBackend()
    raise ValueError(f"unsupported backend: {config.backend!r}")


__all__ = [
    "Backend",
    "BackendCaps",
    "BackendKind",
    "RawRecord",
    "ReaderConfig",
    "ReaderError",
    "HostEngineUnavailableError",
    "ExporterCollisionDetected",
    "ReservationDeniedError",
    "DcgmStreamError",
    "make_backend",
]
