"""Real Utilization weight presets.

A preset is a 4-tuple of weights ``(sm, tensor, dram, gr)`` applied to the four
Real Utilization composite inputs. The weights of a preset sum to 1.0; a custom
weight tuple is auto-normalized by the caller before use. Presets let the same
composite emphasize different subsystems for AI, HPC, or memory-bound workloads.
"""
from __future__ import annotations

from typing import Dict, Tuple

Weights = Tuple[float, float, float, float]

# Built-in presets, keyed by name. Order of weights is (sm, tensor, dram, gr).
PRESETS: Dict[str, Weights] = {
    "ai": (0.35, 0.35, 0.20, 0.10),
    "hpc": (0.45, 0.15, 0.25, 0.15),
    "mem": (0.35, 0.10, 0.40, 0.15),
}

# The preset used when the caller does not request one.
DEFAULT_PRESET_NAME = "ai"

# Floating-point tolerance for matching a weight tuple back to a named preset.
_MATCH_TOLERANCE = 1e-9


def resolve_preset(name: str) -> Weights:
    """Return the weight tuple for a preset name.

    Raises ``KeyError`` (with the known names) if the name is not a built-in
    preset.
    """
    try:
        return PRESETS[name]
    except KeyError:
        known = ", ".join(sorted(PRESETS))
        raise KeyError(f"unknown preset {name!r}; known presets: {known}") from None


def preset_name_for_weights(weights: Weights) -> str:
    """Name a weight tuple: ``"ai"`` / ``"hpc"`` / ``"mem"``, else ``"custom"``.

    A tuple matches a built-in preset only when every weight is within a small
    floating-point tolerance of that preset's weights.
    """
    for name, preset in PRESETS.items():
        if all(abs(w - p) <= _MATCH_TOLERANCE for w, p in zip(weights, preset)):
            return name
    return "custom"
