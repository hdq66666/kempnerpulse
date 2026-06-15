"""Parser tests for the Prometheus (dcgm-exporter) reader."""
import os

from kempnerpulse.reader.prometheus import parse_prometheus_records

_FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "prometheus_sample.txt"
)


def _load():
    with open(_FIXTURE, encoding="utf-8") as f:
        return f.read()


def test_records_keyed_by_gpu_label_plus_global():
    ids = [r.entity_id for r in parse_prometheus_records(_load())]
    assert "0" in ids and "1" in ids
    assert "global" in ids  # the bare (unlabelled) metric line


def test_lines_without_entity_label_are_skipped():
    # No gpu/UUID/device label -> no entity to key on -> dropped.
    text = 'DCGM_FI_PROF_SM_ACTIVE{foo="bar"} 0.5\n'
    assert parse_prometheus_records(text) == []
