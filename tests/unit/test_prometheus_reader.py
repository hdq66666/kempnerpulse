"""Parser tests for the Prometheus (dcgm-exporter) reader."""
import os

from kempnerpulse.reader.prometheus import parse_prometheus_records
from kempnerpulse._compat import parse_prometheus_text

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


def test_sample_partitions_metrics_from_labels():
    sample = parse_prometheus_text(_load())
    # numeric metric stays a float in metrics
    assert sample.metrics["0"]["DCGM_FI_PROF_SM_ACTIVE"] == 0.95
    assert sample.metrics["0"]["DCGM_FI_PROF_PIPE_TENSOR_ACTIVE"] == 0.48
    # string labels are separated out of metrics
    assert sample.labels["0"]["gpu"] == "0"
    assert sample.labels["0"]["modelName"] == "NVIDIA H100 80GB HBM3"
    assert sample.labels["0"]["Hostname"] == "node01"
    assert "modelName" not in sample.metrics["0"]


def test_bare_metric_lands_in_global():
    sample = parse_prometheus_text(_load())
    assert sample.metrics["global"]["some_bare_metric"] == 42.0


def test_lines_without_entity_label_are_skipped():
    # No gpu/UUID/device label -> no entity to key on -> dropped.
    text = 'DCGM_FI_PROF_SM_ACTIVE{foo="bar"} 0.5\n'
    assert parse_prometheus_records(text) == []
