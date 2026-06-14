"""Parser tests for the dcgmi (direct DCGM) reader."""
import os

from kempnerpulse.reader.dcgmi import parse_dmon_block
from kempnerpulse._compat import parse_dcgm_dmon

_FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "dcgmi_dmon_2tick.txt"
)


def _load():
    with open(_FIXTURE, encoding="utf-8") as f:
        return f.read()


def test_one_record_per_gpu_row_in_order():
    records = parse_dmon_block(_load())
    # 2 ticks x 2 GPUs = 4 rows, in file order.
    assert [r.entity_id for r in records] == ["0", "1", "0", "1"]
    assert all(r.source == "dcgmi" for r in records)


def test_na_becomes_none_never_zero():
    records = parse_dmon_block(_load())
    cold_gpu0 = records[0]  # first tick: profiling fields are N/A
    assert cold_gpu0.fields["DCGM_FI_PROF_SM_ACTIVE"] is None
    assert cold_gpu0.fields["DCGM_FI_PROF_PIPE_TENSOR_ACTIVE"] is None
    # device fields in the same row are real values
    assert cold_gpu0.fields["DCGM_FI_DEV_POWER_USAGE"] == 350.5
    assert cold_gpu0.fields["DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL"] == 25000.0


def test_sample_drops_none_and_synthesizes_labels():
    sample = parse_dcgm_dmon(_load(), gpu_models={"0": "NVIDIA H100", "1": "NVIDIA H100"})
    # dcgmi has no labels of its own; gpu is always set, modelName from lookup
    assert sample.labels["0"]["gpu"] == "0"
    assert sample.labels["0"]["modelName"] == "NVIDIA H100"
    # no None ever leaks into metrics
    assert all(v is not None for v in sample.metrics["0"].values())


def test_two_tick_merge_keeps_last_non_none():
    sample = parse_dcgm_dmon(_load())
    g0 = sample.metrics["0"]
    # tick-2 values win where present
    assert g0["DCGM_FI_PROF_SM_ACTIVE"] == 0.95
    assert g0["DCGM_FI_DEV_GPU_TEMP"] == 39.0          # was 38 in tick 1
    # tick-2 N/A must NOT erase the tick-1 reading
    assert g0["DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL"] == 25000.0


def test_skips_headers_and_blank_lines():
    text = "#Entity ...\nID\n\n   \nGPU 0 1\n"
    # Only one data row, one metric column present.
    records = parse_dmon_block(text)
    assert len(records) == 1
    assert records[0].entity_id == "0"
    assert records[0].fields["DCGM_FI_DEV_SM_CLOCK"] == 1.0
