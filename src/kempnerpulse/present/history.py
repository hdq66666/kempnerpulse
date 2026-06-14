"""Per-GPU rolling time-series store and the record→series adapter.

The plot and focus views need short rolling histories of *display-unit* series
(percent, GB/s, W, °C, …) per GPU. :class:`HistoryStore` is a fixed-capacity
ring buffer keyed by ``(gpu_id, series_key)``; :func:`update_history` reads one
batch of :class:`ComputedRecord`s, converts canonical values to display units,
and pushes them under the keys the views read.

Series keys (the contract the views depend on)::

    real_util sm_active tensor dram gpu_util gr_active sm_occupancy
    fp16 fp32 fp64
    tc_hmma tc_imma tc_dfma tc_dmma tc_qmma
    memcpy mem_used_pct power gpu_temp
    pcie_rx pcie_tx pcie_rxtx nvlink_gbps
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, Iterable, Optional

from ..compute.result import ComputedRecord
from .format import (
    bytes_per_second_to_gigabytes,
    fraction_to_percent,
)

# Default ring-buffer depth: enough samples to fill the widest sparkline/plot.
DEFAULT_HISTORY_MAXLEN = 120


class HistoryStore:
    """Fixed-capacity per-(gpu, key) ring buffers of floats."""

    def __init__(self, maxlen: int = DEFAULT_HISTORY_MAXLEN):
        self.maxlen = maxlen
        self.data: Dict[str, Dict[str, Deque[float]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=maxlen))
        )

    def push(self, gpu_id: str, key: str, value: float) -> None:
        """Append ``value`` to the ``(gpu_id, key)`` series (oldest drops out)."""
        self.data[gpu_id][key].append(value)

    def get(self, gpu_id: str, key: str) -> Deque[float]:
        """Return the ``(gpu_id, key)`` series, or an empty deque if absent."""
        gpu_data = self.data.get(gpu_id)
        if gpu_data is None:
            return deque(maxlen=self.maxlen)
        return gpu_data.get(key, deque(maxlen=self.maxlen))


# ── Canonical-field → display-series mapping ──────────────────────────────────
#
# Each entry is (series_key, canonical_field_name). The value is read from the
# record's CanonicalRecord, converted from a fraction to a percent, and pushed
# only when present (None readings are skipped, never coerced to 0).
_PERCENT_SERIES = (
    ("gpu_util", "gpu_nvml_busy_time_fraction"),
    ("gr_active", "gpu_graphics_compute_engine_active_cycle_fraction"),
    ("sm_active", "gpu_streaming_multiprocessor_active_cycle_fraction"),
    ("sm_occupancy", "gpu_streaming_multiprocessor_warp_occupancy_fraction"),
    ("tensor", "gpu_tensor_core_pipe_active_cycle_fraction"),
    ("dram", "gpu_dram_controller_active_cycle_fraction"),
    ("memcpy", "gpu_memory_copy_engine_busy_time_fraction"),
    ("fp64", "gpu_cuda_core_floating_point_64bit_pipe_active_cycle_fraction"),
    ("fp32", "gpu_cuda_core_floating_point_32bit_pipe_active_cycle_fraction"),
    ("fp16", "gpu_cuda_core_floating_point_16bit_pipe_active_cycle_fraction"),
    ("tc_hmma", "gpu_tensor_core_half_precision_mma_active_cycle_fraction"),
    ("tc_imma", "gpu_tensor_core_integer_mma_active_cycle_fraction"),
    ("tc_dfma", "gpu_tensor_core_double_precision_fma_active_cycle_fraction"),
    ("tc_dmma", "gpu_tensor_core_double_mma_active_cycle_fraction"),
    ("tc_qmma", "gpu_tensor_core_quarter_mma_active_cycle_fraction"),
)

# Raw (already display-unit) canonical fields pushed verbatim.
_RAW_SERIES = (
    ("power", "gpu_board_power_draw_watts"),
    ("gpu_temp", "gpu_die_temperature_celsius"),
    ("pcie_rx", "gpu_pcie_receive_throughput_bytes_per_second"),
    ("pcie_tx", "gpu_pcie_transmit_throughput_bytes_per_second"),
)


def update_history(history: HistoryStore, records: Iterable[ComputedRecord]) -> None:
    """Push one batch of records' display-unit series into ``history``.

    All conversions to display units happen here: percents come from canonical
    fractions ×100, NVLink GB/s from canonical bytes/second ÷1e9. A series is
    skipped (not zero-filled) whenever its canonical source is ``None``.
    """
    for rec in records:
        gpu_id = rec.gpu_id
        canon = rec.record

        # Real Utilization is already a 0..100 composite from the Compute layer.
        history.push(gpu_id, "real_util", rec.real_util)

        # Memory used % comes from the convenience fraction on the record.
        mem_used_pct = fraction_to_percent(rec.memory_used_fraction)
        if mem_used_pct is not None:
            history.push(gpu_id, "mem_used_pct", mem_used_pct)

        for key, field_name in _PERCENT_SERIES:
            pct = fraction_to_percent(getattr(canon, field_name))
            if pct is not None:
                history.push(gpu_id, key, pct)

        for key, field_name in _RAW_SERIES:
            val = getattr(canon, field_name)
            if val is not None:
                history.push(gpu_id, key, val)

        # PCIe RX+TX combined (only when both legs are present).
        pcie_rx = canon.gpu_pcie_receive_throughput_bytes_per_second
        pcie_tx = canon.gpu_pcie_transmit_throughput_bytes_per_second
        if pcie_rx is not None and pcie_tx is not None:
            history.push(gpu_id, "pcie_rxtx", pcie_rx + pcie_tx)

        nvlink_gbps = bytes_per_second_to_gigabytes(
            canon.gpu_nvlink_aggregate_throughput_bytes_per_second
        )
        if nvlink_gbps is not None:
            history.push(gpu_id, "nvlink_gbps", nvlink_gbps)
