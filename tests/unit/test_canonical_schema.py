"""Contract tests for the canonical schema (the inter-layer contract)."""
import pytest

from kempnerpulse.translate import (
    SCHEMA_VERSION,
    AggregationMode,
    CanonicalRecord,
    Provenance,
    TranslateError,
    canonical_field_names,
)
from kempnerpulse.translate.schema import (
    NONNEGATIVE_COUNT_FIELDS,
    NONNEGATIVE_MAGNITUDE_FIELDS,
    RATIO_FIELDS,
)


def _valid(**overrides):
    """A minimal schema-valid POINT record; override any field by keyword."""
    base = dict(
        record_schema_version=SCHEMA_VERSION,
        record_timestamp_monotonic_seconds=1.0,
        record_timestamp_wallclock_unix_seconds=1_700_000_000.0,
        record_aggregation_mode=AggregationMode.POINT,
        record_window_microseconds=100_000,
        record_freshness_microseconds=0,
        record_provenance=Provenance.DCGMI,
        record_hostname="node01",
        entity_gpu_index=0,
        entity_gpu_uuid="GPU-aaaa",
    )
    base.update(overrides)
    return CanonicalRecord(**base)


def test_minimal_point_record_validates():
    _valid().validate()  # must not raise


def test_window_record_validates():
    _valid(
        record_aggregation_mode=AggregationMode.WINDOW,
        record_window_microseconds=30_000_000,
    ).validate()


def test_populated_record_validates():
    _valid(
        gpu_streaming_multiprocessor_active_cycle_fraction=0.95,
        gpu_tensor_core_pipe_active_cycle_fraction=0.48,
        gpu_dram_controller_active_cycle_fraction=0.30,
        gpu_nvml_busy_time_fraction=1.0,
        gpu_pcie_transmit_throughput_bytes_per_second=1.5e9,
        gpu_board_power_draw_watts=351.0,
        gpu_pcie_replay_count=0,
        gpu_framebuffer_used_mebibytes=11050.0,
    ).validate()


def test_unobserved_fields_default_to_none():
    rec = _valid()
    assert rec.gpu_streaming_multiprocessor_active_cycle_fraction is None
    assert rec.gpu_tensor_core_quarter_mma_active_cycle_fraction is None  # reserved
    assert rec.record_slurm_job_id is None


@pytest.mark.parametrize("value", [1.01, -0.01, 5.0])
def test_ratio_out_of_range_rejected(value):
    with pytest.raises(TranslateError):
        _valid(gpu_streaming_multiprocessor_active_cycle_fraction=value).validate()


def test_point_with_oversized_window_rejected():
    with pytest.raises(TranslateError):
        _valid(record_window_microseconds=300_000).validate()  # POINT must be <= 200_000


def test_window_with_undersized_window_rejected():
    with pytest.raises(TranslateError):
        _valid(
            record_aggregation_mode=AggregationMode.WINDOW,
            record_window_microseconds=100_000,  # WINDOW must be > 200_000
        ).validate()


def test_negative_count_rejected():
    with pytest.raises(TranslateError):
        _valid(gpu_pcie_replay_count=-1).validate()


def test_negative_magnitude_rejected():
    with pytest.raises(TranslateError):
        _valid(gpu_board_power_draw_watts=-10.0).validate()


def test_bad_schema_version_rejected():
    with pytest.raises(TranslateError):
        _valid(record_schema_version=0).validate()


def test_schema_version_is_one():
    assert SCHEMA_VERSION == 1
    assert _valid().record_schema_version == 1


def test_validator_field_groups_reference_real_fields():
    # Guards against typos in the validate() field-group tuples: every name
    # they reference must be an actual CanonicalRecord field.
    names = set(canonical_field_names())
    for group in (RATIO_FIELDS, NONNEGATIVE_COUNT_FIELDS, NONNEGATIVE_MAGNITUDE_FIELDS):
        for field_name in group:
            assert field_name in names, f"{field_name} is not a CanonicalRecord field"
