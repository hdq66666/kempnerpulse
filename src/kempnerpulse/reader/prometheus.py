"""Layer 1 (Read) — Prometheus backend (``dcgm-exporter`` ``/metrics``).

Scrapes a ``dcgm-exporter`` HTTP endpoint (or reads a saved exposition-format
file) and emits one ``RawRecord`` per GPU, keyed by the metric names the
exporter publishes. Source labels (``gpu``, ``UUID``, ``modelName``, …) ride
along in the same ``fields`` mapping as string values; Layer 2 separates metric
values from identity labels.

Entry points:
  * ``parse_prometheus_records`` — pure text -> ``RawRecord`` parser.
  * ``load_source`` — fetch the exposition text (HTTP URL or local file).
  * ``PrometheusBackend`` — a scrape-per-poll ``Backend`` implementation.
"""
from __future__ import annotations

import math
import re
import time
import urllib.request
from typing import Dict, Iterator, List, Optional

from .base import (
    Backend,
    BackendCaps,
    BackendKind,
    RawRecord,
    ReaderConfig,
)

# Prometheus exposition-format line grammar.
_RE_METRIC_LINE = re.compile(
    r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\{([^}]*)\}\s+'
    r'([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)$'
)
_RE_BARE_LINE = re.compile(
    r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+'
    r'([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)$'
)
_RE_LABEL = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')


def load_source(source: str, timeout: float = 5.0) -> str:
    """Return exposition-format text from an HTTP(S) URL or a local file path."""
    if source.startswith("http://") or source.startswith("https://"):
        with urllib.request.urlopen(source, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    with open(source, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def parse_prometheus_records(
    text: str,
    *,
    source_version: str = "exporter",
    timestamp: Optional[float] = None,
    wallclock: Optional[float] = None,
) -> List[RawRecord]:
    """Parse exposition text into ``RawRecord``s, one per labelled entity.

    A metric line's entity key is its ``gpu`` label (falling back to ``UUID``
    then ``device``); lines without one are skipped. Bare (unlabelled) metric
    lines are grouped under the ``"global"`` entity. The entity's labels are
    carried in ``fields`` as string values alongside the numeric metrics. All
    records share one ``timestamp``/``wallclock``.
    """
    ts = time.monotonic() if timestamp is None else timestamp
    wc = time.time() if wallclock is None else wallclock
    fields_by_entity: Dict[str, Dict[str, object]] = {}
    order: List[str] = []

    def _entity(key: str) -> Dict[str, object]:
        if key not in fields_by_entity:
            fields_by_entity[key] = {}
            order.append(key)
        return fields_by_entity[key]

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _RE_METRIC_LINE.match(line)
        if m:
            metric, label_blob, value_s = m.groups()
            pairs = {k: v.replace('\\"', '"') for k, v in _RE_LABEL.findall(label_blob)}
            gpu_id = pairs.get("gpu") or pairs.get("UUID") or pairs.get("device")
            if gpu_id is None:
                continue
            try:
                value = float(value_s)
            except ValueError:
                continue
            if math.isinf(value) or math.isnan(value):
                continue
            entity = _entity(gpu_id)
            entity[metric] = value
            # Labels ride along as string fields; Layer 2 separates them out.
            for k, v in pairs.items():
                entity[k] = v
            continue

        m2 = _RE_BARE_LINE.match(line)
        if m2:
            metric, value_s = m2.groups()
            try:
                value = float(value_s)
            except ValueError:
                continue
            if math.isinf(value) or math.isnan(value):
                continue
            _entity("global")[metric] = value

    return [
        RawRecord(
            timestamp=ts,
            wallclock=wc,
            entity_id=key,
            fields=fields_by_entity[key],
            source="prometheus",
            source_version=source_version,
        )
        for key in order
    ]


class PrometheusBackend:
    """``Backend`` that scrapes the exporter once per poll interval.

    Unlike the dcgmi backend there is no long-lived subprocess: each pass
    through ``stream`` performs one scrape and yields that scrape's records,
    then sleeps ``poll_seconds`` before the next. ``dcgm-exporter`` refreshes
    profiling fields on its own (~30 s) scrape cycle, which sets the true
    ceiling on useful poll rates.
    """

    def __init__(self) -> None:
        self._config: Optional[ReaderConfig] = None
        self._closed: bool = False

    def open(self, config: ReaderConfig) -> None:
        self._config = config
        self._closed = False

    def stream(self) -> Iterator[RawRecord]:
        cfg = self._config
        if cfg is None:
            return
        first = True
        while not self._closed:
            if not first:
                time.sleep(cfg.poll_seconds)
            first = False
            text = load_source(cfg.source, timeout=cfg.timeout)
            yield from parse_prometheus_records(text)

    def close(self) -> None:
        self._closed = True

    @property
    def caps(self) -> BackendCaps:
        # The exporter's field set is discovered from the scrape, not known
        # ahead of time; advertise an empty set until the first scrape.
        return BackendCaps(kind=BackendKind.PROMETHEUS, fields=frozenset())


# Fail fast at import time if the class drifts from the Backend contract.
assert isinstance(PrometheusBackend(), Backend)
