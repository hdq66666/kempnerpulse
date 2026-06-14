"""The twelve-category workload classification cascade.

Each sample is labeled with exactly one of twelve mutually-exclusive workload
categories. The rules are evaluated top-down and the first match wins, so rule
order is part of the definition: a tensor-dominated sample is "tensor-heavy
compute" and never "compute-heavy", and heavy I/O with busy SMs is not labeled
"I/O" because the I/O rule requires idle SMs. ``MIXED_OR_MODERATE`` is the
explicit fallthrough — it means no rule matched, not a distinct workload.
"""
from __future__ import annotations

from ..translate.schema import CanonicalRecord
from . import thresholds as T
from .real_util import _percent, graphics_engine_percent
from .result import WorkloadClass


def _bytes_per_second(value) -> float:
    """A throughput reading (bytes/s or ``None``) with missing → 0.0."""
    return 0.0 if value is None else value


def classify(record: CanonicalRecord, real_util: float) -> WorkloadClass:
    """Classify one record into a :class:`WorkloadClass` (first match wins).

    ``real_util`` is the composite score for the same record, used only by the
    idle rule. ``gr`` uses the graphics-engine fraction, falling back to the NVML
    busy fraction — the same resolution the composite uses.
    """
    sm = _percent(record.gpu_streaming_multiprocessor_active_cycle_fraction)
    tensor = _percent(record.gpu_tensor_core_pipe_active_cycle_fraction)
    dram = _percent(record.gpu_dram_controller_active_cycle_fraction)
    gr = graphics_engine_percent(record)
    fp64 = _percent(record.gpu_cuda_core_floating_point_64bit_pipe_active_cycle_fraction)
    memcpy = _percent(record.gpu_memory_copy_engine_busy_time_fraction)
    pcie_rx = _bytes_per_second(record.gpu_pcie_receive_throughput_bytes_per_second)
    pcie_tx = _bytes_per_second(record.gpu_pcie_transmit_throughput_bytes_per_second)

    io_heavy = (memcpy >= T.IO_MEMCPY) or (
        max(pcie_rx, pcie_tx) >= T.IO_PCIE_BYTES_PER_SECOND
    )

    # 1 — idle: nothing meaningfully running on the GPU.
    if (real_util < T.IDLE_REAL_UTIL and gr < T.IDLE_ENGINE
            and dram < T.IDLE_ENGINE and not io_heavy):
        return WorkloadClass.IDLE

    # 2 — tensor-heavy compute: dominant tensor pipe with well-loaded SMs.
    if tensor >= T.TENSOR_HEAVY_TENSOR and sm >= T.TENSOR_HEAVY_SM:
        return WorkloadClass.TENSOR_HEAVY_COMPUTE

    # 3 — tensor compute: meaningful tensor activity, moderate SM load.
    if tensor >= T.TENSOR_COMPUTE_TENSOR and sm >= T.TENSOR_COMPUTE_SM:
        return WorkloadClass.TENSOR_COMPUTE

    # 4 — FP64 / HPC compute: appreciable double-precision pipe with loaded SMs.
    if fp64 >= T.FP64_HPC_FP64 and sm >= T.FP64_HPC_SM:
        return WorkloadClass.FP64_HPC_COMPUTE

    # 5 — I/O or data-loading: heavy transfer while SMs are idle.
    if io_heavy and sm < T.IO_SM_IDLE:
        return WorkloadClass.IO_OR_DATA_LOADING

    # 6 — memory-bound: bandwidth-limited, SMs below the effective threshold.
    if dram >= T.MEMORY_BOUND_DRAM and sm < T.MEMORY_BOUND_SM:
        return WorkloadClass.MEMORY_BOUND

    # 7 — compute-heavy: SMs well-utilized, no tensor dominance.
    if sm >= T.SM_EFFECTIVE_HIGH:
        return WorkloadClass.COMPUTE_HEAVY

    # 8 — compute-active: moderate SM use.
    if sm >= T.COMPUTE_ACTIVE_SM:
        return WorkloadClass.COMPUTE_ACTIVE

    # 9 — memory-active: significant DRAM traffic with some SM activity.
    if dram >= T.MEMORY_ACTIVE_DRAM:
        return WorkloadClass.MEMORY_ACTIVE

    # 10 — busy, low SM use: engine active but SMs underutilized.
    if gr >= T.BUSY_LOW_SM_GR and sm < T.BUSY_LOW_SM_SM:
        return WorkloadClass.BUSY_LOW_SM_USE

    # 11 — low utilization: barely any measurable activity.
    if (gr < T.LOW_UTILIZATION_GR and sm < T.LOW_UTILIZATION_SM
            and dram < T.LOW_UTILIZATION_DRAM):
        return WorkloadClass.LOW_UTILIZATION

    # 12 — mixed / moderate: fallthrough, no single dominant pattern.
    return WorkloadClass.MIXED_OR_MODERATE
