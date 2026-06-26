"""Source-vocabulary → canonical-vocabulary mapping and unit normalization.

The mapping table is keyed by the source field name (DCGM field identifiers,
which both the ``dcgmi`` backend and ``dcgm-exporter`` use) and gives the
canonical ``CanonicalRecord`` field plus a unit kind. Unit kinds are applied by
``convert`` to bring source units into canonical units (fractions in [0,1],
bytes/second, joules, …). A ``None`` reading stays ``None`` — never coerced.
"""
from __future__ import annotations

from typing import Optional, Tuple

# Unit kinds:
#   ratio        : already a [0,1] fraction; clamp defensively to [0,1]
#   percent      : a [0,100] percentage; divide by 100 -> [0,1] fraction
#   megabytes_ps : MB/s gauge; ×1e6 -> bytes/second (NVLink stays a gauge)
#   millijoules  : mJ cumulative; ÷1000 -> joules
#   number       : a non-negative float magnitude, passed through
#   count        : a non-negative integer counter, passed through as int
#   flag         : truthy-if-positive boolean

# source field -> (canonical field name, unit kind)
SOURCE_FIELD_MAP = {
    # Profiling activity counters (already [0,1] ratios)
    "DCGM_FI_PROF_SM_ACTIVE":
        ("gpu_streaming_multiprocessor_active_cycle_fraction", "ratio"),
    "DCGM_FI_PROF_SM_OCCUPANCY":
        ("gpu_streaming_multiprocessor_warp_occupancy_fraction", "ratio"),
    "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE":
        ("gpu_tensor_core_pipe_active_cycle_fraction", "ratio"),
    "DCGM_FI_PROF_PIPE_TENSOR_HMMA_ACTIVE":
        ("gpu_tensor_core_half_precision_mma_active_cycle_fraction", "ratio"),
    "DCGM_FI_PROF_PIPE_TENSOR_IMMA_ACTIVE":
        ("gpu_tensor_core_integer_mma_active_cycle_fraction", "ratio"),
    "DCGM_FI_PROF_PIPE_TENSOR_DFMA_ACTIVE":
        ("gpu_tensor_core_double_precision_fma_active_cycle_fraction", "ratio"),
    "DCGM_FI_PROF_PIPE_TENSOR_DMMA_ACTIVE":
        ("gpu_tensor_core_double_mma_active_cycle_fraction", "ratio"),
    "DCGM_FI_PROF_PIPE_TENSOR_QMMA_ACTIVE":
        ("gpu_tensor_core_quarter_mma_active_cycle_fraction", "ratio"),
    "DCGM_FI_PROF_PIPE_FP64_ACTIVE":
        ("gpu_cuda_core_floating_point_64bit_pipe_active_cycle_fraction", "ratio"),
    "DCGM_FI_PROF_PIPE_FP32_ACTIVE":
        ("gpu_cuda_core_floating_point_32bit_pipe_active_cycle_fraction", "ratio"),
    "DCGM_FI_PROF_PIPE_FP16_ACTIVE":
        ("gpu_cuda_core_floating_point_16bit_pipe_active_cycle_fraction", "ratio"),
    "DCGM_FI_PROF_GR_ENGINE_ACTIVE":
        ("gpu_graphics_compute_engine_active_cycle_fraction", "ratio"),
    "DCGM_FI_PROF_DRAM_ACTIVE":
        ("gpu_dram_controller_active_cycle_fraction", "ratio"),
    # NVML time-fraction utilizations (source reports 0..100 percent)
    "DCGM_FI_DEV_GPU_UTIL":
        ("gpu_nvml_busy_time_fraction", "percent"),
    "DCGM_FI_DEV_MEM_COPY_UTIL":
        ("gpu_memory_copy_engine_busy_time_fraction", "percent"),
    # PCIe throughput (source already reports bytes/second)
    "DCGM_FI_PROF_PCIE_TX_BYTES":
        ("gpu_pcie_transmit_throughput_bytes_per_second", "number"),
    "DCGM_FI_PROF_PCIE_RX_BYTES":
        ("gpu_pcie_receive_throughput_bytes_per_second", "number"),
    # NVLink aggregate (source reports an MB/s gauge)
    "DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL":
        ("gpu_nvlink_aggregate_throughput_bytes_per_second", "megabytes_ps"),
    "DCGM_FI_PROF_NVLINK_TX_BYTES":
        ("gpu_nvlink_transmit_throughput_bytes_per_second", "number"),
    "DCGM_FI_PROF_NVLINK_RX_BYTES":
        ("gpu_nvlink_receive_throughput_bytes_per_second", "number"),
    # Power and energy
    "DCGM_FI_DEV_POWER_USAGE":
        ("gpu_board_power_draw_watts", "number"),
    "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION":
        ("gpu_board_total_energy_joules", "millijoules"),
    # Thermal
    "DCGM_FI_DEV_GPU_TEMP":
        ("gpu_die_temperature_celsius", "number"),
    "DCGM_FI_DEV_MEMORY_TEMP":
        ("gpu_memory_die_temperature_celsius", "number"),
    # Clocks
    "DCGM_FI_DEV_SM_CLOCK":
        ("gpu_streaming_multiprocessor_clock_frequency_megahertz", "number"),
    "DCGM_FI_DEV_MEM_CLOCK":
        ("gpu_memory_clock_frequency_megahertz", "number"),
    # Framebuffer
    "DCGM_FI_DEV_FB_USED":
        ("gpu_framebuffer_used_mebibytes", "number"),
    "DCGM_FI_DEV_FB_FREE":
        ("gpu_framebuffer_free_mebibytes", "number"),
    "DCGM_FI_DEV_FB_RESERVED":
        ("gpu_framebuffer_reserved_mebibytes", "number"),
    # Health / error counters
    "DCGM_FI_DEV_PCIE_REPLAY_COUNTER":
        ("gpu_pcie_replay_count", "count"),
    "DCGM_FI_DEV_XID_ERRORS":
        ("gpu_xid_error_count", "count"),
    "DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS":
        ("gpu_uncorrectable_remapped_row_count", "count"),
    "DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS":
        ("gpu_correctable_remapped_row_count", "count"),
    "DCGM_FI_DEV_ROW_REMAP_FAILURE":
        ("gpu_row_remap_failure_flag", "flag"),
}

# Source label keys that identify the entity (not metrics).
GPU_UUID_LABELS = ("UUID", "uuid")
MODEL_LABELS = ("modelName", "model")


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def convert(unit_kind: str, value):
    """Apply ``unit_kind`` to one source value, returning the canonical value.

    ``None`` passes through as ``None`` (the source did not provide a reading).
    """
    if value is None:
        return None
    if unit_kind == "ratio":
        return _clamp01(float(value))
    if unit_kind == "percent":
        return _clamp01(float(value) / 100.0)
    if unit_kind == "megabytes_ps":
        return max(0.0, float(value) * 1.0e6)
    if unit_kind == "millijoules":
        return float(value) / 1000.0
    if unit_kind == "number":
        return float(value)
    if unit_kind == "count":
        return int(value)
    if unit_kind == "flag":
        return bool(value) and float(value) > 0.0
    raise ValueError(f"unknown unit kind: {unit_kind!r}")


def map_field(source_name: str) -> Optional[Tuple[str, str]]:
    """Return ``(canonical_name, unit_kind)`` for a source field, or ``None``."""
    return SOURCE_FIELD_MAP.get(source_name)
