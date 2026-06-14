"""Per-sample health classification.

Health is a single worst-first label derived from error counters, the PCIe
replay rate, and die/memory temperatures against a per-model warning threshold.
The cascade returns the most severe matching state: an unrecoverable memory
remapping is CRIT, a nonzero PCIe replay rate is WARN, an over-threshold
temperature is HOT, and everything else is OK.
"""
from __future__ import annotations

from typing import Optional, Tuple

from ..translate.schema import CanonicalRecord
from .result import HEALTH_CRIT, HEALTH_HOT, HEALTH_OK, HEALTH_WARN

# Rich style strings paired with each health label.
_STYLE_CRIT = "bold red"
_STYLE_WARN = "yellow"
_STYLE_HOT = "yellow"
_STYLE_OK = "green"

# Per-model GPU-die warning temperature (Celsius). Matched by substring on the
# uppercased model name, in this order; the first match wins.
_MODEL_TEMP_WARNINGS: Tuple[Tuple[str, float], ...] = (
    ("A100", 93.0),
    ("H100", 95.0),
    ("H200", 95.0),
    ("RTX 6000", 92.0),
)
_DEFAULT_TEMP_WARNING = 93.0


def warning_temperature_for_model(model_name: Optional[str]) -> float:
    """The GPU-die warning temperature for a model name (default when unknown)."""
    if model_name:
        upper = model_name.upper()
        for token, threshold in _MODEL_TEMP_WARNINGS:
            if token in upper:
                return threshold
    return _DEFAULT_TEMP_WARNING


def health(
    record: CanonicalRecord,
    *,
    pcie_replay_rate_per_second: Optional[float],
    model_name: Optional[str],
) -> Tuple[str, str]:
    """Return ``(health_label, rich_style)`` for one record.

    Cascade, first match wins:

    * CRIT — a row-remap failure flag, or any uncorrectable remapped rows.
    * WARN — a PCIe replay rate that is present and positive.
    * HOT  — a die or memory temperature at/above the model warning threshold.
    * OK   — none of the above.
    """
    warning = warning_temperature_for_model(model_name)

    remap_failed = record.gpu_row_remap_failure_flag is True
    uncorrectable = record.gpu_uncorrectable_remapped_row_count
    if remap_failed or (uncorrectable is not None and uncorrectable > 0):
        return HEALTH_CRIT, _STYLE_CRIT

    if pcie_replay_rate_per_second is not None and pcie_replay_rate_per_second > 0:
        return HEALTH_WARN, _STYLE_WARN

    die_temp = record.gpu_die_temperature_celsius
    mem_temp = record.gpu_memory_die_temperature_celsius
    if (die_temp is not None and die_temp >= warning) or (
        mem_temp is not None and mem_temp >= warning
    ):
        return HEALTH_HOT, _STYLE_HOT

    return HEALTH_OK, _STYLE_OK
