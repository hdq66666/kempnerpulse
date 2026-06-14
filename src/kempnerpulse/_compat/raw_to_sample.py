"""Bridge between Layer-1 ``RawRecord``s and the legacy ``Sample`` structure.

The legacy single-file module groups one sampling tick as a ``Sample`` —
``{ts, metrics[gpu][name] -> float, labels[gpu][key] -> str}``. The layered
readers instead emit a flat stream of ``RawRecord``s. This module converts
between the two and provides drop-in replacements for the legacy read
functions, so the legacy module can consume the new readers without changing
its downstream code. It is transitional and not part of the public API.

Conversion rules (these preserve the legacy module's exact output):
  * A ``None`` field is a value the source did not provide; it is dropped, so a
    missing reading is *absent* from ``Sample.metrics`` (never ``0``).
  * Within one entity, later non-``None`` values win over earlier ones, and a
    later ``None`` never erases an earlier value — matching how the legacy
    parser kept the last valid reading across a two-tick collection.
  * String fields become ``labels``; numeric fields become ``metrics``.
"""
from __future__ import annotations

import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from ..reader.base import BackendKind, DcgmStreamError, RawRecord, ReaderConfig
from ..reader.dcgmi import (
    DCGM_STREAM_MIN_INTERVAL_MS,
    DcgmiBackend,
    parse_dmon_block,
    run_dmon_once,
)
from ..reader.prometheus import parse_prometheus_records


@dataclass
class Sample:
    """One sampling tick: metrics and labels per GPU, with a wall-clock time."""
    ts: float
    metrics: Dict[str, Dict[str, float]]
    labels: Dict[str, Dict[str, str]]


# ── RawRecord <-> Sample ──────────────────────────────────────────────────────

def records_to_sample(records: Iterable[RawRecord]) -> Sample:
    """Fold a tick's ``RawRecord``s into a ``Sample`` (drops ``None`` fields)."""
    metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
    labels: Dict[str, Dict[str, str]] = defaultdict(dict)
    ts: Optional[float] = None
    for rec in records:
        if ts is None:
            ts = rec.wallclock
        entity = rec.entity_id
        for key, value in rec.fields.items():
            if value is None:
                continue
            if isinstance(value, str):
                labels[entity][key] = value
            else:
                metrics[entity][key] = float(value)
    return Sample(
        ts=ts if ts is not None else time.time(),
        metrics=dict(metrics),
        labels=dict(labels),
    )


def records_to_dcgm_sample(
    records: List[RawRecord],
    gpu_models: Optional[Dict[str, str]] = None,
) -> Sample:
    """Like ``records_to_sample`` but synthesizes the labels dcgmi lacks.

    The direct DCGM source carries no labels, so the legacy parser set a ``gpu``
    label for every GPU row and a ``modelName`` when a model lookup was given.
    """
    sample = records_to_sample(records)
    order: List[str] = []
    for rec in records:
        if rec.entity_id not in order:
            order.append(rec.entity_id)
    for gpu_id in order:
        label = sample.labels.setdefault(gpu_id, {})
        label.setdefault("gpu", gpu_id)
        if gpu_models and gpu_id in gpu_models:
            label.setdefault("modelName", gpu_models[gpu_id])
    return sample


def sample_to_records(sample: Sample, *, source: str = "replay") -> List[RawRecord]:
    """Inverse of ``records_to_sample`` (one ``RawRecord`` per GPU). For tests."""
    records: List[RawRecord] = []
    entities = list(sample.metrics.keys())
    for entity in sample.labels:
        if entity not in entities:
            entities.append(entity)
    for entity in entities:
        fields: Dict[str, object] = {}
        fields.update(sample.metrics.get(entity, {}))
        fields.update(sample.labels.get(entity, {}))
        records.append(RawRecord(
            timestamp=0.0,
            wallclock=sample.ts,
            entity_id=entity,
            fields=fields,
            source=source,
            source_version=source,
        ))
    return records


# ── Legacy parser drop-ins ────────────────────────────────────────────────────

def parse_dcgm_dmon(text: str, gpu_models: Optional[Dict[str, str]] = None) -> Sample:
    """Parse ``dcgmi dmon`` text into a ``Sample`` (legacy-compatible)."""
    return records_to_dcgm_sample(parse_dmon_block(text), gpu_models)


def parse_prometheus_text(text: str) -> Sample:
    """Parse Prometheus exposition text into a ``Sample`` (legacy-compatible)."""
    return records_to_sample(parse_prometheus_records(text))


def load_dcgm_direct(
    gpu_ids: Optional[List[str]] = None,
    interval_ms: int = 100,
) -> str:
    """Run ``dcgmi dmon -c 2`` and return raw stdout text (legacy-compatible).

    Pairs with ``parse_dcgm_dmon`` for the one-shot path, which expects text.
    """
    return run_dmon_once(gpu_ids, interval_ms)


# ── Threaded streaming bridge ─────────────────────────────────────────────────

class DcgmStreamReader:
    """Legacy-compatible threaded wrapper around ``DcgmiBackend``.

    Presents the same constructor and consumer API the legacy module expects —
    ``start``/``stop``, ``get_pair``, ``last_counter``, ``wait_for_new``,
    ``wait_first_sample`` — while delegating subprocess handling and parsing to
    the layered dcgmi reader. A reader thread converts each tick to a ``Sample``
    and publishes the latest pair under a condition variable; a second thread
    drains the subprocess's stderr.
    """

    def __init__(
        self,
        gpu_ids: Optional[List[str]],
        poll_ms: int,
        gpu_models: Optional[Dict[str, str]] = None,
    ) -> None:
        self._backend = DcgmiBackend()
        self._config = ReaderConfig(
            backend=BackendKind.DCGMI,
            poll_seconds=max(DCGM_STREAM_MIN_INTERVAL_MS, int(poll_ms)) / 1000.0,
            gpu_ids=tuple(gpu_ids) if gpu_ids else None,
        )
        self._gpu_models = gpu_models or {}
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._latest: Optional[Sample] = None
        self._prev: Optional[Sample] = None
        self._counter: int = 0
        self._error: Optional[BaseException] = None
        self._started = False
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None

    # ── lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        if self._started:
            return
        try:
            self._backend.open(self._config)
        except DcgmStreamError:
            raise
        except Exception as exc:  # normalize any open failure
            raise DcgmStreamError(str(exc)) from exc
        self._reader_thread = threading.Thread(
            target=self._run, name="dcgm-stream", daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, name="dcgm-stream-err", daemon=True,
        )
        self._reader_thread.start()
        self._stderr_thread.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._stop.set()
        self._backend.close()
        with self._cond:
            self._cond.notify_all()
        for t in (self._reader_thread, self._stderr_thread):
            if t is not None and t.is_alive():
                t.join(timeout=1.0)
        self._started = False

    # ── consumer APIs ───────────────────────────────────────────────────

    def get_pair(self) -> Tuple[Optional[Sample], Optional[Sample]]:
        """Non-blocking snapshot of (latest, prev)."""
        with self._cond:
            if self._error is not None and self._latest is None:
                raise DcgmStreamError(str(self._error))
            return self._latest, self._prev

    def last_counter(self) -> int:
        with self._cond:
            return self._counter

    def wait_for_new(
        self,
        last_counter: int,
        timeout: float = 2.0,
    ) -> Tuple[Optional[Sample], Optional[Sample], int]:
        """Block until the sample counter advances past ``last_counter``.

        Returns ``(latest, prev, new_counter)``; ``(None, None, last_counter)``
        if the reader stops before a new sample arrives.
        """
        deadline = time.monotonic() + timeout
        with self._cond:
            while (self._counter <= last_counter
                   and not self._stop.is_set()
                   and self._error is None):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(timeout=remaining)
            if self._error is not None and self._latest is None:
                raise DcgmStreamError(str(self._error))
            if self._stop.is_set() and self._counter <= last_counter:
                return None, None, last_counter
            return self._latest, self._prev, self._counter

    def wait_first_sample(self, timeout: float = 5.0) -> bool:
        """Block until the first valid sample is published. ``True`` on success."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while (self._counter == 0
                   and not self._stop.is_set()
                   and self._error is None):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(timeout=remaining)
            if self._error is not None:
                raise DcgmStreamError(str(self._error))
            return self._counter > 0

    # ── internal ────────────────────────────────────────────────────────

    def _run(self) -> None:
        try:
            for tick in self._backend.stream_ticks():
                if self._stop.is_set():
                    break
                sample = records_to_dcgm_sample(tick, self._gpu_models)
                with self._cond:
                    self._prev = self._latest
                    self._latest = sample
                    self._counter += 1
                    self._cond.notify_all()
        except BaseException as exc:  # surface, never swallow
            with self._cond:
                if self._error is None:
                    self._error = exc
                self._cond.notify_all()
        finally:
            with self._cond:
                self._cond.notify_all()

    def _drain_stderr(self) -> None:
        stderr = self._backend.stderr
        if stderr is None:
            return
        try:
            for line in stderr:
                if self._stop.is_set():
                    break
                if line.strip():
                    sys.stderr.write(f"[dcgmi] {line}")
                    sys.stderr.flush()
        except Exception:
            pass
