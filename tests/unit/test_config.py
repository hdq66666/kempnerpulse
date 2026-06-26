"""Unit tests for the cross-cutting configuration tier.

Covers ``build_config`` (parsing, defaults, backend-aware poll, preset
resolution, weight normalization, export const, history floor) and
``validate_poll`` (the poll-validation rules as data). Also asserts that the
best-effort ``identification`` and ``system_queries`` modules import and degrade
to typed empties without raising when their external commands are absent.
"""
import math

import pytest

from kempnerpulse.compute.presets import PRESETS
from kempnerpulse.config import (
    DCGM_STREAM_MIN_INTERVAL_MS,
    Config,
    build_config,
    build_parser,
    parse_nvlink_fit,
    parse_weights,
    validate_poll,
)
from kempnerpulse.reader.base import BackendKind


# ── parsing & defaults ────────────────────────────────────────────────────────

def test_defaults():
    cfg = build_config([])
    assert isinstance(cfg, Config)
    assert cfg.backend is BackendKind.DCGMI
    assert cfg.source == "http://localhost:9400/metrics"
    assert cfg.gpu_ids is None
    assert cfg.show_all is False
    assert cfg.once is False
    assert cfg.focus_gpu is None
    assert cfg.export_spec is None
    assert cfg.sp_fast is False
    assert cfg.nvlink_fit is None
    # Default preset is AI.
    assert cfg.preset_name == "ai"
    assert cfg.weights == PRESETS["ai"]
    assert cfg.history_length == 120


def test_backend_aware_poll_default_dcgm():
    cfg = build_config(["--backend", "dcgm"])
    assert cfg.backend is BackendKind.DCGMI
    assert cfg.poll_seconds == pytest.approx(0.1)


def test_backend_aware_poll_default_prometheus():
    cfg = build_config(["--backend", "prometheus"])
    assert cfg.backend is BackendKind.PROMETHEUS
    assert cfg.poll_seconds == pytest.approx(1.0)


def test_explicit_poll_overrides_default():
    cfg = build_config(["--backend", "dcgm", "--poll", "0.5"])
    assert cfg.poll_seconds == pytest.approx(0.5)


def test_backend_token_mapping():
    assert build_config(["--backend", "dcgm"]).backend is BackendKind.DCGMI
    assert build_config(["--backend", "prometheus"]).backend is BackendKind.PROMETHEUS


def test_source_override():
    cfg = build_config(["--source", "/tmp/metrics.txt"])
    assert cfg.source == "/tmp/metrics.txt"


# ── preset selection ──────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "flag,name",
    [("--ai-weights", "ai"), ("--hpc-weights", "hpc"), ("--mem-weights", "mem")],
)
def test_preset_flags(flag, name):
    cfg = build_config([flag])
    assert cfg.preset_name == name
    assert cfg.weights == PRESETS[name]


def test_custom_weights_named_custom_and_normalized():
    # Sums to 2.0 -> normalized to sum 1.0; not a built-in preset -> "custom".
    cfg = build_config(["--weights", "0.8,0.6,0.4,0.2"])
    assert cfg.preset_name == "custom"
    assert sum(cfg.weights) == pytest.approx(1.0)
    assert cfg.weights == pytest.approx((0.4, 0.3, 0.2, 0.1))


def test_custom_weights_matching_preset_named():
    # Explicit values equal to the HPC preset are recognized as "hpc".
    cfg = build_config(["--weights", "0.45,0.15,0.25,0.15"])
    assert cfg.preset_name == "hpc"


def test_parse_weights_validator_errors():
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        parse_weights("1,2,3")          # not four values
    with pytest.raises(argparse.ArgumentTypeError):
        parse_weights("a,b,c,d")        # non-numeric
    with pytest.raises(argparse.ArgumentTypeError):
        parse_weights("0,0,0,0")        # sums to zero


def test_parse_weights_normalizes():
    vals = parse_weights("2,2,2,2")
    assert math.isclose(sum(vals), 1.0)
    assert vals == pytest.approx((0.25, 0.25, 0.25, 0.25))


def test_parse_nvlink_fit():
    assert parse_nvlink_fit("1.37") == pytest.approx((1.37, 0.0))
    assert parse_nvlink_fit("1.37,2.5") == pytest.approx((1.37, 2.5))


def test_parse_nvlink_fit_errors():
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        parse_nvlink_fit("0")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_nvlink_fit("a")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_nvlink_fit("1,2,3")


# ── export, history, gpus ─────────────────────────────────────────────────────

def test_export_const_default():
    cfg = build_config(["--export"])
    assert cfg.export_spec == "default"


def test_export_all_and_custom():
    assert build_config(["--export", "all"]).export_spec == "all"
    assert build_config(["--export", "gpu_id,real_util_pct"]).export_spec == (
        "gpu_id,real_util_pct"
    )


def test_export_absent_is_none():
    assert build_config([]).export_spec is None


def test_sp_fast_and_nvlink_fit_config():
    cfg = build_config(["--sp-fast", "--nvlink-fit", "1.37,2.0"])
    assert cfg.sp_fast is True
    assert cfg.nvlink_fit == pytest.approx((1.37, 2.0))


def test_sp_fast_requires_dcgm_backend():
    with pytest.raises(SystemExit):
        build_config(["--backend", "prometheus", "--sp-fast"])


def test_history_floor():
    assert build_config(["--history", "5"]).history_length == 10   # floored
    assert build_config(["--history", "10"]).history_length == 10
    assert build_config(["--history", "500"]).history_length == 500


def test_explicit_gpus_tuple_and_showall():
    cfg = build_config(["--gpus", "0,1,3", "--show-all"])
    assert cfg.gpu_ids == ("0", "1", "3")
    assert cfg.show_all is True


def test_once_flag():
    assert build_config(["--once"]).once is True


def test_config_is_frozen():
    cfg = build_config([])
    with pytest.raises(Exception):
        cfg.poll_seconds = 2.0  # type: ignore[misc]


def test_build_parser_introspectable():
    parser = build_parser()
    # Parsing without args yields argparse defaults (poll unset -> None here).
    ns = parser.parse_args([])
    assert ns.backend == "dcgm"
    assert ns.poll is None
    assert ns.history == 120
    assert ns.sp_fast is False


# ── validate_poll ─────────────────────────────────────────────────────────────

def _cfg(backend: BackendKind, poll: float) -> Config:
    return Config(
        backend=backend,
        poll_seconds=poll,
        source="x",
        gpu_ids=None,
        show_all=False,
        weights=PRESETS["ai"],
        preset_name="ai",
        export_spec=None,
        once=False,
        focus_gpu=None,
        history_length=120,
    )


def test_validate_poll_nonpositive_is_error():
    res = validate_poll(_cfg(BackendKind.DCGMI, 0.0))
    assert res.error is not None
    assert res.clamped is False
    res2 = validate_poll(_cfg(BackendKind.PROMETHEUS, -1.0))
    assert res2.error is not None


def test_validate_poll_prometheus_subsecond_is_error():
    res = validate_poll(_cfg(BackendKind.PROMETHEUS, 0.5))
    assert res.error is not None
    assert "1.0" in res.error or ">= 1.0" in res.error


def test_validate_poll_prometheus_ok():
    res = validate_poll(_cfg(BackendKind.PROMETHEUS, 1.0))
    assert res.error is None
    assert res.clamped is False
    assert res.note is None
    assert res.effective_poll_seconds == pytest.approx(1.0)


def test_validate_poll_dcgm_subfloor_clamps_with_note():
    res = validate_poll(_cfg(BackendKind.DCGMI, 0.05))   # 50ms < 100ms floor
    assert res.error is None
    assert res.clamped is True
    assert res.note is not None
    assert "100ms" in res.note
    assert res.effective_poll_seconds == pytest.approx(
        DCGM_STREAM_MIN_INTERVAL_MS / 1000.0
    )


def test_validate_poll_dcgm_at_floor_not_clamped():
    res = validate_poll(_cfg(BackendKind.DCGMI, 0.1))
    assert res.error is None
    assert res.clamped is False
    assert res.note is None


def test_validate_poll_dcgm_above_floor_ok():
    res = validate_poll(_cfg(BackendKind.DCGMI, 1.0))
    assert res.error is None
    assert res.clamped is False
    assert res.effective_poll_seconds == pytest.approx(1.0)


# ── best-effort modules import and degrade cleanly ────────────────────────────

def test_system_queries_importable_and_safe():
    from kempnerpulse import system_queries as sq

    # Empty bus map -> empty result, no raise.
    assert sq.query_gpu_processes({}) == {}
    # RAM query returns a 2-tuple (values may be None off a /proc host).
    used, total = sq.query_system_ram()
    assert (used is None) or isinstance(used, float)
    assert (total is None) or isinstance(total, float)
    # CpuSampler holds per-instance state; first sample has no percentage.
    sampler = sq.CpuSampler()
    threads, cores, pct, busy = sampler.sample()
    assert pct is None and busy is None


def test_identification_importable_and_safe():
    from kempnerpulse import identification as ident

    # gather_slurm_metadata never raises and returns a dict.
    meta = ident.gather_slurm_metadata()
    assert isinstance(meta, dict)
    # identify() degrades to a populated Identity even with no GPUs present.
    cfg = build_config([])
    identity = ident.identify(cfg)
    assert isinstance(identity, ident.Identity)
    assert isinstance(identity.hostname, str) and identity.hostname
    assert isinstance(identity.bus_id_to_index, dict)
    assert isinstance(identity.slurm_metadata, dict)
