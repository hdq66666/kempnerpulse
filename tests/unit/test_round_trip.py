"""Round-trip between the legacy ``Sample`` and ``RawRecord``s."""
from kempnerpulse._compat import Sample, records_to_sample, sample_to_records


def test_sample_records_sample_is_identity():
    sample = Sample(
        ts=123.0,
        metrics={
            "0": {"DCGM_FI_PROF_SM_ACTIVE": 0.9, "DCGM_FI_DEV_POWER_USAGE": 350.0},
            "1": {"DCGM_FI_PROF_SM_ACTIVE": 0.7},
        },
        labels={
            "0": {"gpu": "0", "modelName": "NVIDIA H100"},
            "1": {"gpu": "1"},
        },
    )
    rebuilt = records_to_sample(sample_to_records(sample))
    assert rebuilt.metrics == sample.metrics
    assert rebuilt.labels == sample.labels
    assert rebuilt.ts == sample.ts
