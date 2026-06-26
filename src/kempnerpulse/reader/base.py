"""Layer 1 (Read) — backend contract and raw records.

This layer acquires raw data from one source and emits a stream of opaque
``RawRecord`` objects keyed by *the source's own* field names. By design, this
layer:

  * never coerces an ``N/A`` reading to ``0.0`` (it uses ``None``);
  * never looks up field *meanings* — naming, units, and missing-value policy
    are Layer 2's responsibility;
  * never blocks on user input or installs signal handlers (that belongs to the
    cross-cutting tier).

Its runtime dependencies are the standard library only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Mapping, Optional, Protocol, runtime_checkable


class BackendKind(Enum):
    """Which Layer-1 source produced a record."""
    DCGMI = "dcgmi"
    PROMETHEUS = "prometheus"
    REPLAY = "replay"
    NVML_DIRECT = "nvml_direct"   # reserved for v0.6.0+


@dataclass(frozen=True)
class RawRecord:
    """One reading for one entity (a GPU or MIG slice), in the source's vocabulary.

    ``fields`` carries the source's raw key/values for this entity — metric values
    (typically ``float``) and any source labels (``str``). A value is ``None`` when
    the source reported ``N/A``; it is *never* silently coerced to ``0``. Assigning
    meaning to these keys (canonical names, units, identity) is Layer 2's job.
    """
    timestamp: float                       # monotonic seconds since reader start
    wallclock: float                       # unix seconds (for export / display)
    entity_id: str                         # source's own entity key: "0", "3", "mig:0:0", ...
    fields: Mapping[str, Optional[Any]]    # source field name -> raw value (None = N/A)
    source: str                            # "dcgmi", "prometheus", "replay"
    source_version: str                    # e.g. "DCGM 4.5.2", "exporter", "replay-v1"
    error: Optional[str] = None            # surfaced, never swallowed


@dataclass(frozen=True)
class BackendCaps:
    """What a backend can produce.

    ``fields`` is the set of source field names this backend can emit. Layer 2
    uses it to decide which canonical metrics are available for a given source.
    """
    kind: BackendKind
    fields: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class ReaderConfig:
    """Minimal Layer-1 configuration.

    A deliberately small, self-contained slice of configuration so that Layer 1
    can be constructed and tested on its own. The cross-cutting configuration
    tier supplies these values from parsed CLI arguments.
    """
    backend: BackendKind = BackendKind.DCGMI
    poll_seconds: float = 0.1
    source: str = "http://localhost:9400/metrics"   # prometheus endpoint/file, or replay CSV path
    gpu_ids: Optional[tuple[str, ...]] = None        # explicit physical IDs to monitor; None = discover
    all_gpus: bool = False                           # ignore CUDA_VISIBLE_DEVICES / SLURM_JOB_GPUS
    timeout: float = 5.0
    dcgm_field_ids: Optional[str] = None             # dcgmi-only: comma-separated field ids
    dcgm_metric_names: Optional[tuple[str, ...]] = None  # dcgmi-only: parser names matching field ids


@runtime_checkable
class Backend(Protocol):
    """Layer 1 contract. Implementations: dcgmi, prometheus, replay."""

    def open(self, config: ReaderConfig) -> None:
        """Acquire the source (spawn dcgmi, open the HTTP scrape, open the replay file).

        Backends that contend for a shared resource (dcgmi profiling watch) run their
        preflight here and raise a typed ``ReaderError`` with remediation on failure.
        """
        ...

    def stream(self) -> Iterator[RawRecord]:
        """Yield ``RawRecord`` objects until the source is exhausted or ``close()`` is called."""
        ...

    def close(self) -> None:
        """Release the source. Must be safe to call after a failed ``open()``."""
        ...

    @property
    def caps(self) -> BackendCaps:
        """Static capabilities (kind + producible source fields)."""
        ...


# ── Typed errors ──────────────────────────────────────────────────────────────

class ReaderError(RuntimeError):
    """Base for Layer-1 read failures. Carries an actionable remediation string."""

    def __init__(self, message: str, remediation: str = "") -> None:
        super().__init__(message)
        self.remediation = remediation

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base}\n{self.remediation}" if self.remediation else base


class HostEngineUnavailableError(ReaderError):
    """``nv-hostengine`` is not reachable on the local socket."""


class ExporterCollisionDetected(ReaderError):
    """``dcgm-exporter`` already holds the profiling watch; recommend ``--backend prometheus``."""


class ReservationDeniedError(ReaderError):
    """Could not acquire the requested DCGM watch fields without dropping another consumer."""


class DcgmStreamError(ReaderError):
    """The ``dcgmi dmon`` streaming subprocess failed or exited unexpectedly."""
