"""Per-process source context for translation (resolved once at startup).

``SourceContext`` pins the operating context — which backend produced the
records, how to tag aggregation/provenance, the host/cluster metadata, and the
per-GPU identity (UUID, model) — so that the per-record translator is a pure
mapping that simply stamps these constants onto every ``CanonicalRecord``.
"""
from __future__ import annotations

import socket
from dataclasses import dataclass, field
from typing import Dict, Optional

from ..reader.base import BackendKind
from .schema import AggregationMode, Provenance

# One ~100 ms multiplexer cycle for a POINT-mode (dcgmi) record.
DCGMI_POINT_WINDOW_MICROSECONDS = 100_000
# dcgm-exporter's default profiling collect interval, in microseconds.
PROMETHEUS_WINDOW_MICROSECONDS = 30_000_000

_BACKEND_PROVENANCE = {
    BackendKind.DCGMI: Provenance.DCGMI,
    BackendKind.PROMETHEUS: Provenance.PROMETHEUS,
    BackendKind.REPLAY: Provenance.REPLAY,
}
_BACKEND_AGGREGATION = {
    BackendKind.DCGMI: AggregationMode.POINT,
    BackendKind.PROMETHEUS: AggregationMode.WINDOW,
    BackendKind.REPLAY: AggregationMode.POINT,
}
_BACKEND_WINDOW = {
    BackendKind.DCGMI: DCGMI_POINT_WINDOW_MICROSECONDS,
    BackendKind.PROMETHEUS: PROMETHEUS_WINDOW_MICROSECONDS,
    BackendKind.REPLAY: DCGMI_POINT_WINDOW_MICROSECONDS,
}


@dataclass(frozen=True)
class SourceContext:
    """Static context resolved at startup; frozen for the process lifetime."""
    backend: BackendKind
    provenance: Provenance
    aggregation_mode: AggregationMode
    window_microseconds: int
    hostname: str
    gpu_uuid_by_index: Dict[int, str] = field(default_factory=dict)
    gpu_model_by_index: Dict[int, str] = field(default_factory=dict)
    slurm_metadata: Dict[str, object] = field(default_factory=dict)


def make_source_context(
    backend: BackendKind,
    *,
    hostname: Optional[str] = None,
    gpu_uuid_by_index: Optional[Dict[int, str]] = None,
    gpu_model_by_index: Optional[Dict[int, str]] = None,
    slurm_metadata: Optional[Dict[str, object]] = None,
) -> SourceContext:
    """Build a ``SourceContext`` with backend-derived provenance/aggregation."""
    return SourceContext(
        backend=backend,
        provenance=_BACKEND_PROVENANCE.get(backend, Provenance.DCGMI),
        aggregation_mode=_BACKEND_AGGREGATION.get(backend, AggregationMode.POINT),
        window_microseconds=_BACKEND_WINDOW.get(backend, DCGMI_POINT_WINDOW_MICROSECONDS),
        hostname=hostname or socket.gethostname(),
        gpu_uuid_by_index=dict(gpu_uuid_by_index or {}),
        gpu_model_by_index=dict(gpu_model_by_index or {}),
        slurm_metadata=dict(slurm_metadata or {}),
    )
