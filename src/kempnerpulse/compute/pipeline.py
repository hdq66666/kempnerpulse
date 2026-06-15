"""The Compute layer entry point — ``CanonicalRecord`` to ``ComputedRecord``.

``compute_record`` runs the full per-sample pipeline: the Real Utilization
composite, the workload classification, health, and the convenience derivations
(memory totals/fraction and the differenced PCIe replay rate). ``compute_tick``
applies it across one tick's records, threading the per-GPU previous record so
the replay-rate differencing works across ticks.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..translate.schema import CanonicalRecord
from .classify import classify
from .health import health
from .presets import PRESETS, Weights, preset_name_for_weights
from .real_util import real_util
from .result import ComputedRecord

# The default weight preset for the composite when the caller does not pass one.
DEFAULT_WEIGHTS: Weights = PRESETS["ai"]


def _pcie_replay_rate(
    record: CanonicalRecord, prev: Optional[CanonicalRecord]
) -> Optional[float]:
    """Differenced PCIe replay rate (events/s) between ``prev`` and ``record``.

    Returns ``None`` unless there is a previous record, both replay counts are
    present, the monotonic time delta is positive, and the count did not go
    backwards (a counter reset).
    """
    if prev is None:
        return None
    cur_count = record.gpu_pcie_replay_count
    prev_count = prev.gpu_pcie_replay_count
    if cur_count is None or prev_count is None:
        return None
    dt = (record.record_timestamp_monotonic_seconds
          - prev.record_timestamp_monotonic_seconds)
    if dt <= 0:
        return None
    delta = cur_count - prev_count
    if delta < 0:
        return None
    return delta / dt


def _memory_derivations(record: CanonicalRecord):
    """Return ``(total_mebibytes, used_fraction)`` from the framebuffer readings.

    Total is used + free + reserved when all three are present; the used fraction
    is used / total when total is positive. Either may be ``None``.
    """
    used = record.gpu_framebuffer_used_mebibytes
    free = record.gpu_framebuffer_free_mebibytes
    reserved = record.gpu_framebuffer_reserved_mebibytes

    total: Optional[float] = None
    if used is not None and free is not None and reserved is not None:
        total = used + free + reserved

    used_fraction: Optional[float] = None
    if total is not None and total > 0 and used is not None:
        used_fraction = used / total

    return total, used_fraction


def compute_record(
    record: CanonicalRecord,
    *,
    prev: Optional[CanonicalRecord] = None,
    weights: Weights = DEFAULT_WEIGHTS,
    preset_name: Optional[str] = None,
    model_name: Optional[str] = None,
) -> ComputedRecord:
    """Compute every derived signal for one canonical record.

    ``prev`` is the same GPU's previous record (used only for the replay-rate
    difference). ``weights`` is the composite weight tuple; ``preset_name`` is
    resolved from the weights when not supplied. ``model_name`` rides through to
    the result and selects the per-model temperature warning.
    """
    score = real_util(record, weights)
    workload = classify(record, score)

    replay_rate = _pcie_replay_rate(record, prev)
    health_label, health_style = health(
        record,
        pcie_replay_rate_per_second=replay_rate,
        model_name=model_name,
    )

    total_mebibytes, used_fraction = _memory_derivations(record)

    resolved_preset = (preset_name if preset_name is not None
                       else preset_name_for_weights(weights))

    return ComputedRecord(
        record=record,
        gpu_index=record.entity_gpu_index,
        gpu_uuid=record.entity_gpu_uuid,
        model_name=model_name,
        real_util=score,
        preset_name=resolved_preset,
        weights=tuple(weights),
        workload_class=workload,
        bottleneck=workload.bottleneck,
        health=health_label,
        health_style=health_style,
        memory_total_mebibytes=total_mebibytes,
        memory_used_fraction=used_fraction,
        pcie_replay_rate_per_second=replay_rate,
    )


def compute_tick(
    records,
    prev_by_index: Optional[Dict[int, CanonicalRecord]] = None,
    **opts,
) -> List[ComputedRecord]:
    """Compute one tick's records, threading the previous record per GPU.

    ``prev_by_index`` maps ``entity_gpu_index`` to that GPU's previous canonical
    record; it is updated in place so a caller can reuse the same dict across
    ticks. Any ``prev`` in ``opts`` is ignored in favour of the per-GPU lookup.
    """
    if prev_by_index is None:
        prev_by_index = {}
    opts.pop("prev", None)

    out: List[ComputedRecord] = []
    for record in records:
        index = record.entity_gpu_index
        prev = prev_by_index.get(index)
        out.append(compute_record(record, prev=prev, **opts))
        prev_by_index[index] = record
    return out
