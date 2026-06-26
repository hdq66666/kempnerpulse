"""The canonical schema — the inter-layer contract.

``CanonicalRecord`` is the single internal vocabulary that Layers 3 (Compute)
and 4 (Present) depend on. Layer 2 (Translate) is the only layer that knows
about source vocabularies (DCGM field IDs, Prometheus names), units, and
backend quirks; it emits ``CanonicalRecord`` objects and nothing above it ever sees a
vendor identifier again.

Field-naming convention (every field follows ``<scope>_<subsystem>_<aspect>_<unit>``):

* ``record_*`` — record-level metadata; ``entity_*`` — GPU / MIG identity;
  ``gpu_*`` — per-GPU hardware readings.
* Ratios are ``..._fraction`` in ``[0.0, 1.0]`` — never ``_pct``. A presenter
  that wants 0–100 multiplies by 100 itself.
* Throughputs carry ``_bytes_per_second``; cumulative counters use the bare
  unit (``_joules``); event counts use ``_count``.
* ``None`` means "the source did not provide this reading" — never coerced to 0.

The names are long and explicit on purpose: a reader of
``gpu_streaming_multiprocessor_active_cycle_fraction`` needs no glossary.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Optional

# Bumped in lockstep with a documented change log. Adding a field is an N+1
# minor bump (old readers tolerate extras); removing one is a major bump.
SCHEMA_VERSION = 1

# One multiplexer cycle on Hopper is ~100 ms; POINT records integrate one (or at
# most two) such cycles. The boundary between POINT and WINDOW aggregation.
POINT_WINDOW_MAX_MICROSECONDS = 200_000


class TranslateError(ValueError):
    """A canonical record violated a schema invariant (see ``validate``)."""


class AggregationMode(Enum):
    """How a record's metric values are integrated over time."""
    POINT = "point"     # one ~100 ms multiplexer-cycle snapshot
    WINDOW = "window"   # time-average over record_window_microseconds


class Provenance(Enum):
    """Where a record came from."""
    DCGMI = "dcgmi"
    PROMETHEUS = "prometheus"
    NVML_FALLBACK = "nvml_fallback"   # a counter was substituted from NVML
    REPLAY = "replay"


@dataclass(frozen=True)
class CanonicalRecord:
    """One fully-translated reading for one entity, in canonical vocabulary.

    All ``Optional[float]`` metric fields are ``None`` unless the source
    provided them. The required block (no defaults) is the metadata every record
    must carry; the optional block is the per-subsystem readings plus cluster
    and reserved metadata that tolerate absence.
    """

    # ── Required record metadata ────────────────────────────────────────────
    record_schema_version: int
    record_timestamp_monotonic_seconds: float       # time.monotonic() at receipt
    record_timestamp_wallclock_unix_seconds: float  # time.time() at receipt
    record_aggregation_mode: AggregationMode
    record_window_microseconds: int                 # the integration window
    record_freshness_microseconds: int              # staleness at delivery
    record_provenance: Provenance
    record_hostname: str                            # always populated

    # ── Required entity identity ─────────────────────────────────────────────
    entity_gpu_index: int                           # logical index as reader sees it
    entity_gpu_uuid: str                            # hardware-stable identifier

    # ── Optional entity identity (some reserved for future use) ──────────────
    entity_mig_instance_index: Optional[int] = None
    entity_process_id: Optional[int] = None                       # reserved
    entity_process_command_line_truncated: Optional[str] = None   # reserved

    # ── Cluster / Slurm / MPI metadata ──────────────────────────────────────
    record_slurm_job_id: Optional[str] = None
    record_slurm_step_id: Optional[str] = None
    record_slurm_array_job_id: Optional[str] = None
    record_slurm_array_task_id: Optional[str] = None
    record_slurm_restart_count: Optional[int] = None
    record_node_index_in_job: Optional[int] = None
    record_mpi_rank: Optional[int] = None
    record_capture_clock_offset_microseconds: Optional[int] = None

    # ── Reserved user-annotation metadata (emitted None in v0.5.0) ───────────
    record_user_annotation_iteration_index: Optional[int] = None
    record_user_annotation_phase_label: Optional[str] = None
    record_user_annotation_step_count: Optional[int] = None
    record_user_annotation_request_id: Optional[str] = None
    record_user_annotation_token_count: Optional[int] = None

    # ── Streaming-multiprocessor counters ───────────────────────────────────
    gpu_streaming_multiprocessor_active_cycle_fraction: Optional[float] = None
    gpu_streaming_multiprocessor_warp_occupancy_fraction: Optional[float] = None

    # ── Tensor-core counters: umbrella + per-precision sub-pipes ─────────────
    gpu_tensor_core_pipe_active_cycle_fraction: Optional[float] = None
    gpu_tensor_core_half_precision_mma_active_cycle_fraction: Optional[float] = None    # HMMA: FP16/BF16
    gpu_tensor_core_integer_mma_active_cycle_fraction: Optional[float] = None           # IMMA: INT8
    gpu_tensor_core_double_precision_fma_active_cycle_fraction: Optional[float] = None  # DFMA: FP64
    # Reserved — not surfaced by DCGM 4.x on Hopper; names exist for forward stability.
    gpu_tensor_core_double_mma_active_cycle_fraction: Optional[float] = None            # DMMA: TF32/FP32
    gpu_tensor_core_quarter_mma_active_cycle_fraction: Optional[float] = None           # QMMA: FP8

    # ── CUDA-core precision pipes ────────────────────────────────────────────
    gpu_cuda_core_floating_point_64bit_pipe_active_cycle_fraction: Optional[float] = None
    gpu_cuda_core_floating_point_32bit_pipe_active_cycle_fraction: Optional[float] = None
    gpu_cuda_core_floating_point_16bit_pipe_active_cycle_fraction: Optional[float] = None

    # ── Graphics / compute engine ───────────────────────────────────────────
    gpu_graphics_compute_engine_active_cycle_fraction: Optional[float] = None

    # ── DRAM / HBM subsystem ─────────────────────────────────────────────────
    gpu_dram_controller_active_cycle_fraction: Optional[float] = None
    gpu_memory_copy_engine_busy_time_fraction: Optional[float] = None

    # ── PCIe — throughputs are differenced rates ─────────────────────────────
    gpu_pcie_transmit_throughput_bytes_per_second: Optional[float] = None
    gpu_pcie_receive_throughput_bytes_per_second: Optional[float] = None
    gpu_pcie_replay_count: Optional[int] = None

    # ── NVLink — aggregate plus real TX/RX rates when profiling fields are present
    gpu_nvlink_aggregate_throughput_bytes_per_second: Optional[float] = None
    gpu_nvlink_transmit_throughput_bytes_per_second: Optional[float] = None
    gpu_nvlink_receive_throughput_bytes_per_second: Optional[float] = None

    # ── Power and energy ─────────────────────────────────────────────────────
    gpu_board_power_draw_watts: Optional[float] = None
    gpu_board_total_energy_joules: Optional[float] = None          # cumulative; not differenced
    gpu_board_enforced_power_limit_watts: Optional[float] = None
    gpu_board_default_power_limit_watts: Optional[float] = None

    # ── Thermal ──────────────────────────────────────────────────────────────
    gpu_die_temperature_celsius: Optional[float] = None
    gpu_memory_die_temperature_celsius: Optional[float] = None

    # ── Clocks ───────────────────────────────────────────────────────────────
    gpu_streaming_multiprocessor_clock_frequency_megahertz: Optional[float] = None
    gpu_memory_clock_frequency_megahertz: Optional[float] = None

    # ── Framebuffer occupancy ────────────────────────────────────────────────
    gpu_framebuffer_used_mebibytes: Optional[float] = None
    gpu_framebuffer_free_mebibytes: Optional[float] = None
    gpu_framebuffer_reserved_mebibytes: Optional[float] = None
    gpu_framebuffer_total_mebibytes: Optional[float] = None        # derived: used+free+reserved

    # ── NVML legacy utilization — time-fraction, exposed for parity ──────────
    gpu_nvml_busy_time_fraction: Optional[float] = None

    # ── Health and error counters ────────────────────────────────────────────
    gpu_xid_error_count: Optional[int] = None
    gpu_uncorrectable_remapped_row_count: Optional[int] = None
    gpu_correctable_remapped_row_count: Optional[int] = None
    gpu_row_remap_failure_flag: Optional[bool] = None

    def validate(self) -> None:
        """Raise ``TranslateError`` if any single-record invariant is violated.

        Single-record invariants only. The cross-record invariant — energy is
        monotonically non-decreasing per entity — is enforced upstream by the
        Translate differencer, which is the only component that sees the
        sequence; it cannot be checked from one record in isolation.
        """
        if self.record_schema_version < 1:
            raise TranslateError(
                f"record_schema_version must be >= 1, got {self.record_schema_version}"
            )

        for name in RATIO_FIELDS:
            val = getattr(self, name)
            if val is not None and not (0.0 <= val <= 1.0):
                raise TranslateError(f"{name} must be in [0.0, 1.0], got {val!r}")

        for name in NONNEGATIVE_COUNT_FIELDS:
            val = getattr(self, name)
            if val is not None and val < 0:
                raise TranslateError(f"{name} must be >= 0, got {val!r}")

        for name in NONNEGATIVE_MAGNITUDE_FIELDS:
            val = getattr(self, name)
            if val is not None and val < 0:
                raise TranslateError(f"{name} must be >= 0, got {val!r}")

        if self.record_window_microseconds < 0:
            raise TranslateError(
                f"record_window_microseconds must be >= 0, got {self.record_window_microseconds}"
            )
        if self.record_freshness_microseconds < 0:
            raise TranslateError(
                f"record_freshness_microseconds must be >= 0, got {self.record_freshness_microseconds}"
            )

        mode, window = self.record_aggregation_mode, self.record_window_microseconds
        if mode is AggregationMode.POINT and window > POINT_WINDOW_MAX_MICROSECONDS:
            raise TranslateError(
                f"POINT records require window <= {POINT_WINDOW_MAX_MICROSECONDS} us, got {window}"
            )
        if mode is AggregationMode.WINDOW and window <= POINT_WINDOW_MAX_MICROSECONDS:
            raise TranslateError(
                f"WINDOW records require window > {POINT_WINDOW_MAX_MICROSECONDS} us, got {window}"
            )


# ── Field groups used by validate() and by Layer 2/4 (single source of truth) ──

# All ratio fields, constrained to [0.0, 1.0].
RATIO_FIELDS = (
    "gpu_streaming_multiprocessor_active_cycle_fraction",
    "gpu_streaming_multiprocessor_warp_occupancy_fraction",
    "gpu_tensor_core_pipe_active_cycle_fraction",
    "gpu_tensor_core_half_precision_mma_active_cycle_fraction",
    "gpu_tensor_core_integer_mma_active_cycle_fraction",
    "gpu_tensor_core_double_precision_fma_active_cycle_fraction",
    "gpu_tensor_core_double_mma_active_cycle_fraction",
    "gpu_tensor_core_quarter_mma_active_cycle_fraction",
    "gpu_cuda_core_floating_point_64bit_pipe_active_cycle_fraction",
    "gpu_cuda_core_floating_point_32bit_pipe_active_cycle_fraction",
    "gpu_cuda_core_floating_point_16bit_pipe_active_cycle_fraction",
    "gpu_graphics_compute_engine_active_cycle_fraction",
    "gpu_dram_controller_active_cycle_fraction",
    "gpu_memory_copy_engine_busy_time_fraction",
    "gpu_nvml_busy_time_fraction",
)

# Event counters / non-negative integers.
NONNEGATIVE_COUNT_FIELDS = (
    "gpu_pcie_replay_count",
    "gpu_xid_error_count",
    "gpu_uncorrectable_remapped_row_count",
    "gpu_correctable_remapped_row_count",
    "record_slurm_restart_count",
)

# Physical magnitudes that cannot be negative.
NONNEGATIVE_MAGNITUDE_FIELDS = (
    "gpu_pcie_transmit_throughput_bytes_per_second",
    "gpu_pcie_receive_throughput_bytes_per_second",
    "gpu_nvlink_aggregate_throughput_bytes_per_second",
    "gpu_nvlink_transmit_throughput_bytes_per_second",
    "gpu_nvlink_receive_throughput_bytes_per_second",
    "gpu_board_power_draw_watts",
    "gpu_board_total_energy_joules",
    "gpu_board_enforced_power_limit_watts",
    "gpu_board_default_power_limit_watts",
    "gpu_streaming_multiprocessor_clock_frequency_megahertz",
    "gpu_memory_clock_frequency_megahertz",
    "gpu_framebuffer_used_mebibytes",
    "gpu_framebuffer_free_mebibytes",
    "gpu_framebuffer_reserved_mebibytes",
    "gpu_framebuffer_total_mebibytes",
)


def canonical_field_names() -> tuple:
    """Every ``CanonicalRecord`` field name, in declaration order."""
    return tuple(f.name for f in fields(CanonicalRecord))
