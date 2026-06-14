"""The Real Utilization composite.

Real Utilization is one scalar per sample: a weighted combination of four
profiling fractions — streaming-multiprocessor active, tensor-pipe active, DRAM
controller active, and graphics/compute engine active — each scaled to a 0..100
percent, then clamped to ``[0, 100]``. It packages four counters into a single
accessible number; it deliberately does not claim that 100 means the hardware is
fully saturated.
"""
from __future__ import annotations

from typing import Optional

from ..translate.schema import CanonicalRecord
from .presets import Weights


def _percent(fraction: Optional[float]) -> float:
    """A canonical fraction (``[0,1]`` or ``None``) as a 0..100 percent.

    A missing reading contributes nothing to the composite, so ``None`` → 0.0.
    """
    return 0.0 if fraction is None else fraction * 100.0


def graphics_engine_percent(record: CanonicalRecord) -> float:
    """Graphics/compute-engine activity as a percent, with the NVML fallback.

    Prefers the graphics/compute engine active fraction; when that reading is
    absent, falls back to the NVML busy-time fraction (the same time-fraction
    signal ``nvidia-smi`` reports). Both missing → 0.0.
    """
    gr = record.gpu_graphics_compute_engine_active_cycle_fraction
    if gr is None:
        gr = record.gpu_nvml_busy_time_fraction
    return _percent(gr)


def real_util(record: CanonicalRecord, weights: Weights) -> float:
    """Weighted Real Utilization composite for one record, clamped to ``[0,100]``.

    ``weights`` is ``(w_sm, w_tensor, w_dram, w_gr)``. Each input is the
    canonical fraction scaled to a percent; a missing input contributes 0.
    """
    w_sm, w_tensor, w_dram, w_gr = weights

    sm = _percent(record.gpu_streaming_multiprocessor_active_cycle_fraction)
    tensor = _percent(record.gpu_tensor_core_pipe_active_cycle_fraction)
    dram = _percent(record.gpu_dram_controller_active_cycle_fraction)
    gr = graphics_engine_percent(record)

    score = w_sm * sm + w_tensor * tensor + w_dram * dram + w_gr * gr
    return max(0.0, min(100.0, score))
