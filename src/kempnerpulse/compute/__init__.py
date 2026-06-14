"""KempnerPulse Layer 3 — Compute.

Pure-functional domain logic over ``CanonicalRecord``s: the Real Utilization
composite, the workload classification cascade, and health. Outputs are
``ComputedRecord``s for the Present layer. No I/O, no source vocabulary.

``compute_record`` (single sample) and ``compute_tick`` (one tick, threading the
previous record per GPU) are the entry points; the building blocks
(``real_util``, ``classify``, ``health``, presets) are exported for direct use
and testing.
"""
from .classify import classify
from .health import health, warning_temperature_for_model
from .pipeline import compute_record, compute_tick
from .presets import (
    DEFAULT_PRESET_NAME,
    PRESETS,
    preset_name_for_weights,
    resolve_preset,
)
from .real_util import graphics_engine_percent, real_util
from .result import (
    HEALTH_CRIT,
    HEALTH_HOT,
    HEALTH_LABELS,
    HEALTH_OK,
    HEALTH_WARN,
    WORKLOAD_STATUS_LABELS,
    BottleneckCategory,
    ComputedRecord,
    WorkloadClass,
)

__all__ = [
    # Result types (the Compute -> Present contract).
    "ComputedRecord",
    "WorkloadClass",
    "BottleneckCategory",
    "WORKLOAD_STATUS_LABELS",
    "HEALTH_LABELS",
    "HEALTH_OK",
    "HEALTH_WARN",
    "HEALTH_HOT",
    "HEALTH_CRIT",
    # Entry points.
    "compute_record",
    "compute_tick",
    # Building blocks.
    "real_util",
    "graphics_engine_percent",
    "classify",
    "health",
    "warning_temperature_for_model",
    # Presets.
    "PRESETS",
    "DEFAULT_PRESET_NAME",
    "resolve_preset",
    "preset_name_for_weights",
]
