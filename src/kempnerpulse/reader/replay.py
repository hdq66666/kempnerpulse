"""Layer 1 (Read) — replay backend.

Replays a saved ``dcgmi dmon`` capture as a deterministic ``RawRecord`` stream,
so the higher layers can be exercised in tests and CI without GPU hardware.
Ticks are split on a repeated ``GPU <id>`` row (the same boundary the live
reader uses) and each is stamped with a synthetic, monotonically increasing
timestamp so replays are reproducible.
"""
from __future__ import annotations

from typing import Iterator, List, Optional

from .base import Backend, BackendCaps, BackendKind, RawRecord, ReaderConfig
from .dcgmi import DCGM_DMON_METRIC_NAMES, parse_dmon_block


def split_dmon_ticks(text: str) -> List[str]:
    """Split a multi-tick ``dcgmi dmon`` capture into per-tick text blocks."""
    blocks: List[str] = []
    current: dict = {}
    order: List[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("ID"):
            continue
        parts = stripped.split()
        if len(parts) < 2 or parts[0] != "GPU":
            continue
        gpu_id = parts[1]
        if gpu_id in current:
            blocks.append("\n".join(current[g] for g in order))
            current = {}
            order = []
        current[gpu_id] = stripped
        order.append(gpu_id)
    if current:
        blocks.append("\n".join(current[g] for g in order))
    return blocks


class ReplayBackend:
    """Replays a captured ``dcgmi dmon`` file as a ``RawRecord`` stream."""

    def __init__(self) -> None:
        self._ticks: List[str] = []
        self._poll_seconds: float = 0.1
        self._closed: bool = False

    def open(self, config: ReaderConfig) -> None:
        with open(config.source, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        self._ticks = split_dmon_ticks(text)
        self._poll_seconds = config.poll_seconds
        self._closed = False

    def stream(self) -> Iterator[RawRecord]:
        for idx, block in enumerate(self._ticks):
            if self._closed:
                break
            ts = idx * self._poll_seconds
            for rec in parse_dmon_block(
                block, source_version="replay", timestamp=ts, wallclock=ts,
            ):
                # Re-tag the source so downstream provenance reads "replay".
                yield RawRecord(
                    timestamp=rec.timestamp,
                    wallclock=rec.wallclock,
                    entity_id=rec.entity_id,
                    fields=rec.fields,
                    source="replay",
                    source_version="replay",
                )

    def close(self) -> None:
        self._closed = True

    @property
    def caps(self) -> BackendCaps:
        return BackendCaps(
            kind=BackendKind.REPLAY,
            fields=frozenset(DCGM_DMON_METRIC_NAMES),
        )


# Fail fast at import time if the class drifts from the Backend contract.
assert isinstance(ReplayBackend(), Backend)
