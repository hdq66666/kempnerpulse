"""Replay backend: deterministic re-emission of a captured dcgmi stream."""
import os

from kempnerpulse.reader import BackendKind, ReaderConfig, make_backend

_FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "dcgmi_dmon_2tick.txt"
)


def _cfg():
    return ReaderConfig(backend=BackendKind.REPLAY, source=_FIXTURE, poll_seconds=0.1)


def test_replay_emits_all_rows_tagged_replay():
    backend = make_backend(_cfg())
    backend.open(_cfg())
    try:
        records = list(backend.stream())
    finally:
        backend.close()
    assert [r.entity_id for r in records] == ["0", "1", "0", "1"]
    assert all(r.source == "replay" for r in records)


def test_replay_timestamps_are_deterministic_per_tick():
    backend = make_backend(_cfg())
    backend.open(_cfg())
    try:
        records = list(backend.stream())
    finally:
        backend.close()
    # tick 0 -> 0.0s, tick 1 -> 0.1s
    assert sorted({round(r.timestamp, 6) for r in records}) == [0.0, 0.1]
    # N/A handling is preserved through replay
    assert records[0].fields["DCGM_FI_PROF_SM_ACTIVE"] is None
