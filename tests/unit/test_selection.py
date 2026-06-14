"""Unit tests for the cross-cutting GPU selection tier.

Covers ``GPUSelector`` precedence (explicit ``--gpus`` -> ``--show-all`` ->
visibility env vars in order -> accessible fallback), clamping to the accessible
set, and the ``_parse_gpu_list`` / ``_expand_ranges`` id-spec grammar.
"""
import pytest

from kempnerpulse.selection import GPUSelector


@pytest.fixture(autouse=True)
def _clear_visibility_env(monkeypatch):
    """Ensure visibility env vars don't leak between tests."""
    for key in (
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "SLURM_STEP_GPUS",
        "SLURM_JOB_GPUS",
    ):
        monkeypatch.delenv(key, raising=False)


# ── precedence ────────────────────────────────────────────────────────────────

def test_explicit_gpus_wins(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,6")
    sel = GPUSelector(explicit="0,1", accessible={"0", "1", "2", "5", "6"})
    allowed, reason, source = sel.resolve()
    assert allowed == {"0", "1"}
    assert reason == "--gpus"
    assert source == "0,1"


def test_show_all_uses_accessible_and_ignores_env(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    sel = GPUSelector(explicit=None, disable_auto=True, accessible={"0", "1", "2"})
    allowed, reason, source = sel.resolve()
    assert allowed == {"0", "1", "2"}
    assert reason == "all"
    assert source is None


def test_env_precedence_order(monkeypatch):
    # All four set; CUDA_VISIBLE_DEVICES is first in the order and should win.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("SLURM_STEP_GPUS", "2")
    monkeypatch.setenv("SLURM_JOB_GPUS", "3")
    sel = GPUSelector(explicit=None, accessible={"0", "1", "2", "3"})
    allowed, reason, source = sel.resolve()
    assert allowed == {"0"}
    assert reason == "CUDA_VISIBLE_DEVICES"
    assert source == "0"


def test_env_falls_through_to_next_when_empty(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "1,2")
    sel = GPUSelector(explicit=None, accessible={"0", "1", "2", "3"})
    allowed, reason, _ = sel.resolve()
    assert allowed == {"1", "2"}
    assert reason == "NVIDIA_VISIBLE_DEVICES"


def test_slurm_step_before_job(monkeypatch):
    monkeypatch.setenv("SLURM_STEP_GPUS", "1")
    monkeypatch.setenv("SLURM_JOB_GPUS", "2,3")
    sel = GPUSelector(explicit=None, accessible={"0", "1", "2", "3"})
    allowed, reason, _ = sel.resolve()
    assert allowed == {"1"}
    assert reason == "SLURM_STEP_GPUS"


def test_fallback_to_accessible_when_no_env():
    sel = GPUSelector(explicit=None, accessible={"0", "1"})
    allowed, reason, source = sel.resolve()
    assert allowed == {"0", "1"}
    assert reason == "all"
    assert source is None


def test_env_with_no_accessible_overlap_falls_through(monkeypatch):
    # Env parses to ids that don't intersect accessible -> empty after clamp ->
    # treated as "not usable", falls through to the accessible fallback.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "9")
    sel = GPUSelector(explicit=None, accessible={"0", "1"})
    allowed, reason, _ = sel.resolve()
    assert allowed == {"0", "1"}
    assert reason == "all"


# ── clamping ──────────────────────────────────────────────────────────────────

def test_explicit_clamped_to_accessible():
    sel = GPUSelector(explicit="0,1,2,3", accessible={"1", "2"})
    allowed, _, _ = sel.resolve()
    assert allowed == {"1", "2"}


def test_no_accessible_means_no_clamp():
    sel = GPUSelector(explicit="0,1,9", accessible=None)
    allowed, _, _ = sel.resolve()
    assert allowed == {"0", "1", "9"}


# ── _parse_gpu_list grammar ───────────────────────────────────────────────────

def test_parse_comma_list():
    assert GPUSelector._parse_gpu_list("0,1,2") == {"0", "1", "2"}


def test_parse_range():
    assert GPUSelector._parse_gpu_list("0-3") == {"0", "1", "2", "3"}


def test_parse_mixed_list_and_range():
    assert GPUSelector._parse_gpu_list("0,2-4,7") == {"0", "2", "3", "4", "7"}


def test_parse_reversed_range():
    assert GPUSelector._parse_gpu_list("3-1") == {"1", "2", "3"}


def test_parse_bracketed_hostlist_range():
    assert GPUSelector._parse_gpu_list("node[0-2]") == {"0", "1", "2"}


def test_parse_bracketed_with_comma_not_supported():
    # The top-level comma split runs before the bracket regex, so a comma
    # *inside* brackets is not supported (ported legacy behavior): each fragment
    # fails the bracket/range/suffix patterns and contributes nothing.
    assert GPUSelector._parse_gpu_list("node[0,2-3]") == set()


def test_parse_gpu_suffix_forms():
    assert GPUSelector._parse_gpu_list("gpu2") == {"2"}
    assert GPUSelector._parse_gpu_list("0000:17:00.0/gpu3") == {"3"}
    assert GPUSelector._parse_gpu_list("mig:4") == {"4"}


def test_parse_sentinels_empty():
    for sentinel in ("all", "none", "void", "", "  "):
        assert GPUSelector._parse_gpu_list(sentinel) == set()


def test_expand_ranges_helper():
    assert GPUSelector._expand_ranges("0-2,5") == {"0", "1", "2", "5"}
    assert GPUSelector._expand_ranges("4") == {"4"}
