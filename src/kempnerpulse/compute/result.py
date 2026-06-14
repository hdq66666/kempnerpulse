"""Compute-layer output types — the Compute → Present contract.

A ``ComputedRecord`` wraps the per-sample ``CanonicalRecord`` (so the presenter
still has every metric value) together with the derived signals the Compute
layer produces: the Real Utilization score, the workload classification, health,
and a few convenience derivations. Presenters and the CSV writer consume
``ComputedRecord``; they convert canonical fractions to display units (×100,
bytes/s → GB/s, …) themselves.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from ..translate.schema import CanonicalRecord


class BottleneckCategory(Enum):
    """The coarse five-way rollup used for summary colouring."""
    IDLE = "idle"
    COMPUTE = "compute"
    IO = "io"
    MEMORY = "memory"
    MIXED = "mixed"


class WorkloadClass(Enum):
    """The twelve mutually-exclusive workload categories.

    The value is the exact human-readable status label. ``bottleneck`` gives the
    coarse rollup; ``label`` is the display string.
    """
    IDLE = "idle"
    TENSOR_HEAVY_COMPUTE = "tensor-heavy compute"
    TENSOR_COMPUTE = "tensor compute"
    FP64_HPC_COMPUTE = "FP64 / HPC compute"
    IO_OR_DATA_LOADING = "I/O or data-loading"
    MEMORY_BOUND = "memory-bound"
    COMPUTE_HEAVY = "compute-heavy"
    COMPUTE_ACTIVE = "compute-active"
    MEMORY_ACTIVE = "memory-active"
    BUSY_LOW_SM_USE = "busy, low SM use"
    LOW_UTILIZATION = "low utilization"
    MIXED_OR_MODERATE = "mixed / moderate"

    @property
    def label(self) -> str:
        return self.value

    @property
    def bottleneck(self) -> BottleneckCategory:
        return _BOTTLENECK_OF[self]


_BOTTLENECK_OF = {
    WorkloadClass.IDLE: BottleneckCategory.IDLE,
    WorkloadClass.TENSOR_HEAVY_COMPUTE: BottleneckCategory.COMPUTE,
    WorkloadClass.TENSOR_COMPUTE: BottleneckCategory.COMPUTE,
    WorkloadClass.FP64_HPC_COMPUTE: BottleneckCategory.COMPUTE,
    WorkloadClass.IO_OR_DATA_LOADING: BottleneckCategory.IO,
    WorkloadClass.MEMORY_BOUND: BottleneckCategory.MEMORY,
    WorkloadClass.COMPUTE_HEAVY: BottleneckCategory.COMPUTE,
    WorkloadClass.COMPUTE_ACTIVE: BottleneckCategory.COMPUTE,
    WorkloadClass.MEMORY_ACTIVE: BottleneckCategory.MEMORY,
    WorkloadClass.BUSY_LOW_SM_USE: BottleneckCategory.MIXED,
    WorkloadClass.LOW_UTILIZATION: BottleneckCategory.MIXED,
    WorkloadClass.MIXED_OR_MODERATE: BottleneckCategory.MIXED,
}

# The 12 labels in classification (cascade) order — for fixed-width UI columns.
WORKLOAD_STATUS_LABELS = tuple(wc.label for wc in WorkloadClass)

# Health states, worst-first; styles are Rich style strings.
HEALTH_OK = "OK"
HEALTH_WARN = "WARN"
HEALTH_HOT = "HOT"
HEALTH_CRIT = "CRIT"
HEALTH_LABELS = (HEALTH_OK, HEALTH_WARN, HEALTH_HOT, HEALTH_CRIT)


@dataclass(frozen=True)
class ComputedRecord:
    """One fully-computed per-GPU sample: canonical metrics + derived signals."""
    record: CanonicalRecord

    # Identity (resolved; model_name has no canonical field, so it rides here).
    gpu_index: int
    gpu_uuid: str
    model_name: Optional[str]

    # Real Utilization composite (0..100) + the preset that produced it.
    real_util: float
    preset_name: str
    weights: Tuple[float, float, float, float]   # (sm, tensor, dram, gr)

    # Classification.
    workload_class: WorkloadClass
    bottleneck: BottleneckCategory

    # Health.
    health: str          # one of HEALTH_LABELS
    health_style: str    # Rich style string

    # Convenience derivations (None when inputs are unavailable).
    memory_total_mebibytes: Optional[float] = None
    memory_used_fraction: Optional[float] = None        # [0,1]
    pcie_replay_rate_per_second: Optional[float] = None  # differenced counter

    @property
    def gpu_id(self) -> str:
        return str(self.gpu_index)

    @property
    def status_line(self) -> str:
        return self.workload_class.label
