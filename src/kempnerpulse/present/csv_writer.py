"""CSV export — the column registry keyed to ``ComputedRecord``/canonical.

The export schema (column names, units, precision, and ordering) is preserved
exactly from the single-file implementation so downstream tooling reads the same
files. The difference is the *source*: each column extracts from a
:class:`ComputedRecord` and its :class:`CanonicalRecord` (canonical fractions/SI)
and converts to the documented display unit at write time.

Audit of units/precision (unchanged from legacy):

* ``*_pct`` columns: ``fraction × 100`` formatted ``.2f``;
* ``pcie_*_bytes_s``: raw bytes/second formatted ``.4f``;
* ``nvlink_gbps``: ``bytes/s ÷ 1e9`` formatted ``.4f`` (== legacy ``MB/s ÷ 1e3``);
* ``power_w`` / ``*_temp_c`` / ``*_clock_mhz`` / ``mem_used_mib``: raw value ``.4f``;
* ``mem_total_mib``: ``.1f``; ``mem_used_pct``: ``.2f``;
* ``energy_j``: cumulative joules ``.1f``;
* ``pcie_replay_rate_s``: differenced rate ``.2f``;
* ``real_util_pct``: ``.2f``; ``status`` / ``health``: strings from the record.

An unavailable reading (``None``) is emitted as an empty field, never ``0``.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional, Sequence, Tuple

from ..compute.result import ComputedRecord
from .format import (
    bytes_per_second_to_gigabytes,
    fraction_to_percent,
)

# A column extractor turns (record, timestamp) into the already-formatted cell.
Extractor = Callable[[ComputedRecord, float], str]


def _fmt(value: Optional[float], digits: int) -> str:
    """Format a float to ``digits`` places, or empty string when unavailable."""
    return f"{value:.{digits}f}" if value is not None else ""


def _short_model_name(name: str) -> str:
    """Shorten a GPU model for CSV: ``'NVIDIA H100 80GB HBM3'`` → ``'H100'``."""
    s = re.sub(r"^NVIDIA\s+", "", name).strip()
    m = re.match(r"(RTX)\s*(\d+)", s)
    if m:
        return m.group(1) + m.group(2)
    parts = s.split()
    return parts[0].split("-")[0] if parts else "GPU"


def _percent_column(canonical_field: str, digits: int = 2) -> Extractor:
    """Extractor for a ``*_pct`` column sourced from a canonical fraction."""
    def extract(rec: ComputedRecord, _ts: float) -> str:
        return _fmt(fraction_to_percent(getattr(rec.record, canonical_field)), digits)
    return extract


def _raw_column(canonical_field: str, digits: int = 4) -> Extractor:
    """Extractor for a raw display-unit column (W, °C, MHz, MiB, bytes/s)."""
    def extract(rec: ComputedRecord, _ts: float) -> str:
        return _fmt(getattr(rec.record, canonical_field), digits)
    return extract


def _timestamp_column(rec: ComputedRecord, ts: float) -> str:
    return f"{ts:.2f}"


def _gpu_id_column(rec: ComputedRecord, _ts: float) -> str:
    return rec.gpu_id


def _model_column(rec: ComputedRecord, _ts: float) -> str:
    return _short_model_name(rec.model_name or "")


def _real_util_column(rec: ComputedRecord, _ts: float) -> str:
    return f"{rec.real_util:.2f}"


def _status_column(rec: ComputedRecord, _ts: float) -> str:
    return rec.status_line


def _health_column(rec: ComputedRecord, _ts: float) -> str:
    return rec.health


def _mem_total_column(rec: ComputedRecord, _ts: float) -> str:
    return _fmt(rec.memory_total_mebibytes, 1)


def _mem_used_pct_column(rec: ComputedRecord, _ts: float) -> str:
    return _fmt(fraction_to_percent(rec.memory_used_fraction), 2)


def _nvlink_gbps_column(rec: ComputedRecord, _ts: float) -> str:
    gbps = bytes_per_second_to_gigabytes(
        rec.record.gpu_nvlink_aggregate_throughput_bytes_per_second
    )
    return _fmt(gbps, 4)


def _pcie_replay_rate_column(rec: ComputedRecord, _ts: float) -> str:
    return _fmt(rec.pcie_replay_rate_per_second, 2)


def _energy_column(rec: ComputedRecord, _ts: float) -> str:
    return _fmt(rec.record.gpu_board_total_energy_joules, 1)


# Ordered column registry: (column_name, extractor). Order and names match legacy.
CSV_COLUMNS: Tuple[Tuple[str, Extractor], ...] = (
    # Tier A: identity + key derived
    ("timestamp", _timestamp_column),
    ("gpu_id", _gpu_id_column),
    ("model", _model_column),
    ("real_util_pct", _real_util_column),
    ("status", _status_column),
    ("health", _health_column),
    # Tier B: core profiling (Real Util components)
    ("sm_active_pct", _percent_column("gpu_streaming_multiprocessor_active_cycle_fraction")),
    ("tensor_active_pct", _percent_column("gpu_tensor_core_pipe_active_cycle_fraction")),
    ("dram_active_pct", _percent_column("gpu_dram_controller_active_cycle_fraction")),
    ("gr_engine_active_pct", _percent_column("gpu_graphics_compute_engine_active_cycle_fraction")),
    ("gpu_util_pct", _percent_column("gpu_nvml_busy_time_fraction")),
    # Tier C: memory/power/thermal
    ("mem_used_mib", _raw_column("gpu_framebuffer_used_mebibytes")),
    ("mem_total_mib", _mem_total_column),
    ("mem_used_pct", _mem_used_pct_column),
    ("power_w", _raw_column("gpu_board_power_draw_watts")),
    ("gpu_temp_c", _raw_column("gpu_die_temperature_celsius")),
    ("mem_temp_c", _raw_column("gpu_memory_die_temperature_celsius")),
    # Tier D: detailed/secondary
    ("sm_occupancy_pct", _percent_column("gpu_streaming_multiprocessor_warp_occupancy_fraction")),
    ("fp16_pipe_pct", _percent_column("gpu_cuda_core_floating_point_16bit_pipe_active_cycle_fraction")),
    ("fp32_pipe_pct", _percent_column("gpu_cuda_core_floating_point_32bit_pipe_active_cycle_fraction")),
    ("fp64_pipe_pct", _percent_column("gpu_cuda_core_floating_point_64bit_pipe_active_cycle_fraction")),
    ("memcpy_util_pct", _percent_column("gpu_memory_copy_engine_busy_time_fraction")),
    ("pcie_rx_bytes_s", _raw_column("gpu_pcie_receive_throughput_bytes_per_second")),
    ("pcie_tx_bytes_s", _raw_column("gpu_pcie_transmit_throughput_bytes_per_second")),
    ("nvlink_gbps", _nvlink_gbps_column),
    ("sm_clock_mhz", _raw_column("gpu_streaming_multiprocessor_clock_frequency_megahertz")),
    ("mem_clock_mhz", _raw_column("gpu_memory_clock_frequency_megahertz")),
    ("pcie_replay_rate_s", _pcie_replay_rate_column),
    ("energy_j", _energy_column),
    ("tc_hmma_pct", _percent_column("gpu_tensor_core_half_precision_mma_active_cycle_fraction")),
    ("tc_imma_pct", _percent_column("gpu_tensor_core_integer_mma_active_cycle_fraction")),
    ("tc_dfma_pct", _percent_column("gpu_tensor_core_double_precision_fma_active_cycle_fraction")),
    ("tc_dmma_pct", _percent_column("gpu_tensor_core_double_mma_active_cycle_fraction")),
    ("tc_qmma_pct", _percent_column("gpu_tensor_core_quarter_mma_active_cycle_fraction")),
)

# Column name → extractor, for spec resolution.
_COLUMN_MAP = {name: extractor for name, extractor in CSV_COLUMNS}

# All column names, in registry order.
CSV_ALL_COLUMN_NAMES: Tuple[str, ...] = tuple(name for name, _ in CSV_COLUMNS)

# The default ("default" spec) subset, in display order.
CSV_DEFAULT_COLUMN_NAMES: Tuple[str, ...] = (
    "timestamp", "gpu_id", "model", "gpu_util_pct", "mem_used_mib",
    "real_util_pct", "sm_active_pct", "tensor_active_pct", "dram_active_pct",
)


class UnknownExportColumns(ValueError):
    """One or more requested export columns are not in the registry."""

    def __init__(self, bad: Sequence[str]):
        self.bad = list(bad)
        super().__init__(
            "unknown export column(s): " + ", ".join(self.bad)
            + "\nAvailable: " + ", ".join(CSV_ALL_COLUMN_NAMES)
        )


def resolve_columns(spec: str) -> List[Tuple[str, Extractor]]:
    """Resolve an export spec to an ordered list of ``(name, extractor)``.

    ``spec`` is ``"default"``, ``"all"``, or a comma-separated list of column
    names. Unknown names raise :class:`UnknownExportColumns` (the caller decides
    how to surface it — this layer does not exit the process).
    """
    if spec == "all":
        return list(CSV_COLUMNS)
    if spec == "default":
        names = list(CSV_DEFAULT_COLUMN_NAMES)
    else:
        names = [c.strip() for c in spec.split(",") if c.strip()]
    bad = [n for n in names if n not in _COLUMN_MAP]
    if bad:
        raise UnknownExportColumns(bad)
    return [(n, _COLUMN_MAP[n]) for n in names]


def csv_header(columns: Sequence[Tuple[str, Extractor]]) -> List[str]:
    """The header row (column names) for a resolved column list."""
    return [name for name, _ in columns]


def csv_row(
    record: ComputedRecord,
    timestamp: float,
    columns: Sequence[Tuple[str, Extractor]],
) -> List[str]:
    """One CSV row for ``record`` at ``timestamp`` over the resolved columns."""
    return [extractor(record, timestamp) for _name, extractor in columns]
