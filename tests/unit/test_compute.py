"""Unit tests for Layer 3 (Compute): composite, classification, health, pipeline.

Records are hand-built ``CanonicalRecord``s — the minimal required metadata plus
whatever metric fractions a case needs. All metric inputs are canonical
fractions in ``[0, 1]`` (the Compute layer scales them to percents itself);
PCIe throughputs are bytes/s.
"""
import pytest

from kempnerpulse.compute import (
    HEALTH_CRIT,
    HEALTH_HOT,
    HEALTH_OK,
    HEALTH_WARN,
    PRESETS,
    BottleneckCategory,
    ComputedRecord,
    WorkloadClass,
    classify,
    compute_record,
    compute_tick,
    graphics_engine_percent,
    health,
    preset_name_for_weights,
    real_util,
    resolve_preset,
)
from kempnerpulse.translate.schema import (
    SCHEMA_VERSION,
    AggregationMode,
    CanonicalRecord,
    Provenance,
)


def _rec(**overrides):
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
    rec = CanonicalRecord(**base)
    rec.validate()  # keep test fixtures schema-valid
    return rec


# ── presets ──────────────────────────────────────────────────────────────────

def test_presets_table():
    assert PRESETS["ai"] == (0.35, 0.35, 0.20, 0.10)
    assert PRESETS["hpc"] == (0.45, 0.15, 0.25, 0.15)
    assert PRESETS["mem"] == (0.35, 0.10, 0.40, 0.15)


def test_resolve_preset_round_trips():
    for name in ("ai", "hpc", "mem"):
        assert resolve_preset(name) == PRESETS[name]


def test_resolve_preset_unknown_raises():
    with pytest.raises(KeyError):
        resolve_preset("nope")


def test_preset_name_for_weights():
    assert preset_name_for_weights((0.35, 0.35, 0.20, 0.10)) == "ai"
    assert preset_name_for_weights((0.45, 0.15, 0.25, 0.15)) == "hpc"
    assert preset_name_for_weights((0.35, 0.10, 0.40, 0.15)) == "mem"
    assert preset_name_for_weights((0.25, 0.25, 0.25, 0.25)) == "custom"


# ── real_util arithmetic ─────────────────────────────────────────────────────

def _sample_for_arithmetic():
    # sm=90%, tensor=50%, dram=30%, gr=40% after the ×100 scaling.
    return _rec(
        gpu_streaming_multiprocessor_active_cycle_fraction=0.90,
        gpu_tensor_core_pipe_active_cycle_fraction=0.50,
        gpu_dram_controller_active_cycle_fraction=0.30,
        gpu_graphics_compute_engine_active_cycle_fraction=0.40,
    )


def test_real_util_ai_preset():
    # 0.35*90 + 0.35*50 + 0.20*30 + 0.10*40 = 59.0
    assert real_util(_sample_for_arithmetic(), PRESETS["ai"]) == pytest.approx(59.0)


def test_real_util_hpc_preset():
    # 0.45*90 + 0.15*50 + 0.25*30 + 0.15*40 = 61.5
    assert real_util(_sample_for_arithmetic(), PRESETS["hpc"]) == pytest.approx(61.5)


def test_real_util_mem_preset():
    # 0.35*90 + 0.10*50 + 0.40*30 + 0.15*40 = 54.5
    assert real_util(_sample_for_arithmetic(), PRESETS["mem"]) == pytest.approx(54.5)


def test_real_util_missing_inputs_are_zero():
    # Only DRAM present; everything else contributes 0.
    rec = _rec(gpu_dram_controller_active_cycle_fraction=0.50)
    assert real_util(rec, PRESETS["ai"]) == pytest.approx(0.20 * 50.0)


def test_real_util_clamped_to_100():
    rec = _rec(
        gpu_streaming_multiprocessor_active_cycle_fraction=1.0,
        gpu_tensor_core_pipe_active_cycle_fraction=1.0,
        gpu_dram_controller_active_cycle_fraction=1.0,
        gpu_graphics_compute_engine_active_cycle_fraction=1.0,
    )
    assert real_util(rec, PRESETS["ai"]) == 100.0


def test_real_util_empty_record_is_zero():
    assert real_util(_rec(), PRESETS["ai"]) == 0.0


# ── GR → NVML fallback ───────────────────────────────────────────────────────

def test_gr_falls_back_to_nvml_when_engine_missing():
    rec = _rec(gpu_nvml_busy_time_fraction=1.0)  # no gr-engine reading
    assert graphics_engine_percent(rec) == pytest.approx(100.0)
    # ai weight on gr is 0.10, all other inputs 0 → 10.0
    assert real_util(rec, PRESETS["ai"]) == pytest.approx(10.0)


def test_gr_engine_preferred_over_nvml():
    rec = _rec(
        gpu_graphics_compute_engine_active_cycle_fraction=0.20,
        gpu_nvml_busy_time_fraction=1.0,
    )
    assert graphics_engine_percent(rec) == pytest.approx(20.0)


def test_gr_both_missing_is_zero():
    assert graphics_engine_percent(_rec()) == 0.0


# ── classification (>= 6 distinct rules, first-match-wins) ────────────────────

def _classify(rec, weights=PRESETS["ai"]):
    return classify(rec, real_util(rec, weights))


def test_classify_idle():
    rec = _rec()  # nothing running
    assert _classify(rec) is WorkloadClass.IDLE
    assert WorkloadClass.IDLE.bottleneck is BottleneckCategory.IDLE


def test_classify_tensor_heavy():
    rec = _rec(
        gpu_tensor_core_pipe_active_cycle_fraction=0.60,
        gpu_streaming_multiprocessor_active_cycle_fraction=0.70,
    )
    assert _classify(rec) is WorkloadClass.TENSOR_HEAVY_COMPUTE


def test_classify_tensor_compute():
    rec = _rec(
        gpu_tensor_core_pipe_active_cycle_fraction=0.20,
        gpu_streaming_multiprocessor_active_cycle_fraction=0.45,
    )
    assert _classify(rec) is WorkloadClass.TENSOR_COMPUTE


def test_classify_fp64_hpc():
    rec = _rec(
        gpu_cuda_core_floating_point_64bit_pipe_active_cycle_fraction=0.30,
        gpu_streaming_multiprocessor_active_cycle_fraction=0.60,
    )
    assert _classify(rec) is WorkloadClass.FP64_HPC_COMPUTE


def test_classify_io_via_pcie():
    rec = _rec(
        gpu_streaming_multiprocessor_active_cycle_fraction=0.10,
        gpu_pcie_receive_throughput_bytes_per_second=2e9,  # >= 1 GB/s
    )
    assert _classify(rec) is WorkloadClass.IO_OR_DATA_LOADING


def test_classify_io_via_memcpy():
    rec = _rec(
        gpu_streaming_multiprocessor_active_cycle_fraction=0.10,
        gpu_memory_copy_engine_busy_time_fraction=0.50,  # >= 40%
    )
    assert _classify(rec) is WorkloadClass.IO_OR_DATA_LOADING


def test_classify_memory_bound():
    rec = _rec(
        gpu_dram_controller_active_cycle_fraction=0.60,
        gpu_streaming_multiprocessor_active_cycle_fraction=0.30,
    )
    assert _classify(rec) is WorkloadClass.MEMORY_BOUND


def test_classify_compute_heavy():
    rec = _rec(gpu_streaming_multiprocessor_active_cycle_fraction=0.85)
    assert _classify(rec) is WorkloadClass.COMPUTE_HEAVY


def test_classify_compute_active():
    rec = _rec(gpu_streaming_multiprocessor_active_cycle_fraction=0.55)
    assert _classify(rec) is WorkloadClass.COMPUTE_ACTIVE


def test_classify_memory_active():
    rec = _rec(
        gpu_dram_controller_active_cycle_fraction=0.45,
        gpu_streaming_multiprocessor_active_cycle_fraction=0.20,
    )
    assert _classify(rec) is WorkloadClass.MEMORY_ACTIVE


def test_classify_busy_low_sm_use():
    rec = _rec(
        gpu_graphics_compute_engine_active_cycle_fraction=0.50,
        gpu_streaming_multiprocessor_active_cycle_fraction=0.10,
    )
    assert _classify(rec) is WorkloadClass.BUSY_LOW_SM_USE


def test_classify_low_utilization():
    rec = _rec(
        gpu_graphics_compute_engine_active_cycle_fraction=0.10,
        gpu_streaming_multiprocessor_active_cycle_fraction=0.10,
        gpu_dram_controller_active_cycle_fraction=0.10,
    )
    assert _classify(rec) is WorkloadClass.LOW_UTILIZATION


def test_classify_mixed_fallthrough():
    # No rule matches: moderate SM/DRAM/GR, no tensor, no FP64, no I/O.
    rec = _rec(
        gpu_streaming_multiprocessor_active_cycle_fraction=0.30,
        gpu_dram_controller_active_cycle_fraction=0.20,
        gpu_graphics_compute_engine_active_cycle_fraction=0.30,
    )
    cls = _classify(rec)
    assert cls is WorkloadClass.MIXED_OR_MODERATE
    assert cls.bottleneck is BottleneckCategory.MIXED


def test_classify_first_match_wins_tensor_over_compute_heavy():
    # SM>=80 (compute-heavy) AND tensor>=50 -> tensor-heavy wins (rule 2 first).
    rec = _rec(
        gpu_streaming_multiprocessor_active_cycle_fraction=0.90,
        gpu_tensor_core_pipe_active_cycle_fraction=0.60,
    )
    assert _classify(rec) is WorkloadClass.TENSOR_HEAVY_COMPUTE


def test_classify_io_needs_idle_sm():
    # Heavy PCIe but busy SMs -> NOT I/O (rule 5 requires sm<30); SM>=80 -> heavy.
    rec = _rec(
        gpu_streaming_multiprocessor_active_cycle_fraction=0.85,
        gpu_pcie_receive_throughput_bytes_per_second=5e9,
    )
    assert _classify(rec) is WorkloadClass.COMPUTE_HEAVY


# ── health cascade ───────────────────────────────────────────────────────────

def test_health_ok():
    rec = _rec(gpu_die_temperature_celsius=60.0)
    label, style = health(rec, pcie_replay_rate_per_second=None, model_name="H100")
    assert label == HEALTH_OK
    assert style == "green"


def test_health_crit_on_remap_failure():
    rec = _rec(gpu_row_remap_failure_flag=True)
    label, style = health(rec, pcie_replay_rate_per_second=None, model_name="H100")
    assert label == HEALTH_CRIT
    assert style == "bold red"


def test_health_crit_on_uncorrectable_rows():
    rec = _rec(gpu_uncorrectable_remapped_row_count=1)
    label, _ = health(rec, pcie_replay_rate_per_second=None, model_name="H100")
    assert label == HEALTH_CRIT


def test_health_crit_outranks_temperature_and_replay():
    rec = _rec(
        gpu_row_remap_failure_flag=True,
        gpu_die_temperature_celsius=99.0,
    )
    label, _ = health(rec, pcie_replay_rate_per_second=5.0, model_name="H100")
    assert label == HEALTH_CRIT


def test_health_warn_on_replay_rate():
    rec = _rec(gpu_die_temperature_celsius=60.0)
    label, style = health(rec, pcie_replay_rate_per_second=2.5, model_name="H100")
    assert label == HEALTH_WARN
    assert style == "yellow"


def test_health_replay_rate_zero_is_not_warn():
    rec = _rec(gpu_die_temperature_celsius=60.0)
    label, _ = health(rec, pcie_replay_rate_per_second=0.0, model_name="H100")
    assert label == HEALTH_OK


def test_health_hot_on_die_temp():
    # H100 warning is 95; 96 >= 95 -> HOT.
    rec = _rec(gpu_die_temperature_celsius=96.0)
    label, style = health(rec, pcie_replay_rate_per_second=None, model_name="H100")
    assert label == HEALTH_HOT
    assert style == "yellow"


def test_health_hot_on_memory_temp():
    rec = _rec(gpu_memory_die_temperature_celsius=96.0)
    label, _ = health(rec, pcie_replay_rate_per_second=None, model_name="H200")
    assert label == HEALTH_HOT


def test_health_model_specific_threshold():
    # A100 warning is 93. 94 is HOT on A100 but OK on H100 (95).
    hot = _rec(gpu_die_temperature_celsius=94.0)
    label_a100, _ = health(hot, pcie_replay_rate_per_second=None, model_name="A100-SXM4")
    label_h100, _ = health(hot, pcie_replay_rate_per_second=None, model_name="H100-SXM5")
    assert label_a100 == HEALTH_HOT
    assert label_h100 == HEALTH_OK


def test_health_default_threshold_when_model_unknown():
    # Default warning is 93. 93 >= 93 -> HOT even for an unknown model / None.
    rec = _rec(gpu_die_temperature_celsius=93.0)
    label_unknown, _ = health(rec, pcie_replay_rate_per_second=None, model_name="Tesla V100")
    label_none, _ = health(rec, pcie_replay_rate_per_second=None, model_name=None)
    assert label_unknown == HEALTH_HOT
    assert label_none == HEALTH_HOT


def test_health_replay_outranks_hot():
    rec = _rec(gpu_die_temperature_celsius=99.0)
    label, _ = health(rec, pcie_replay_rate_per_second=1.0, model_name="H100")
    assert label == HEALTH_WARN


# ── pipeline: compute_record ─────────────────────────────────────────────────

def test_compute_record_basic_fields():
    rec = _rec(
        entity_gpu_index=3,
        entity_gpu_uuid="GPU-zzz",
        gpu_streaming_multiprocessor_active_cycle_fraction=0.90,
        gpu_tensor_core_pipe_active_cycle_fraction=0.60,
    )
    out = compute_record(rec, weights=PRESETS["ai"], model_name="H100")
    assert isinstance(out, ComputedRecord)
    assert out.gpu_index == 3
    assert out.gpu_uuid == "GPU-zzz"
    assert out.model_name == "H100"
    assert out.workload_class is WorkloadClass.TENSOR_HEAVY_COMPUTE
    assert out.bottleneck is BottleneckCategory.COMPUTE
    assert out.real_util == pytest.approx(real_util(rec, PRESETS["ai"]))
    assert out.weights == PRESETS["ai"]
    assert out.record is rec


def test_compute_record_resolves_preset_name_from_weights():
    rec = _rec()
    assert compute_record(rec, weights=PRESETS["hpc"]).preset_name == "hpc"
    assert compute_record(rec, weights=(0.4, 0.3, 0.2, 0.1)).preset_name == "custom"


def test_compute_record_explicit_preset_name_overrides():
    rec = _rec()
    out = compute_record(rec, weights=PRESETS["ai"], preset_name="my-custom")
    assert out.preset_name == "my-custom"


def test_compute_record_memory_derivation():
    rec = _rec(
        gpu_framebuffer_used_mebibytes=30000.0,
        gpu_framebuffer_free_mebibytes=9000.0,
        gpu_framebuffer_reserved_mebibytes=1000.0,
    )
    out = compute_record(rec)
    assert out.memory_total_mebibytes == pytest.approx(40000.0)
    assert out.memory_used_fraction == pytest.approx(0.75)


def test_compute_record_memory_none_when_incomplete():
    rec = _rec(gpu_framebuffer_used_mebibytes=30000.0)  # free/reserved missing
    out = compute_record(rec)
    assert out.memory_total_mebibytes is None
    assert out.memory_used_fraction is None


def test_compute_record_no_prev_means_no_replay_rate():
    out = compute_record(_rec(gpu_pcie_replay_count=5))
    assert out.pcie_replay_rate_per_second is None


# ── pipeline: PCIe replay differencing ───────────────────────────────────────

def test_pcie_replay_rate_differenced():
    prev = _rec(record_timestamp_monotonic_seconds=10.0, gpu_pcie_replay_count=4)
    cur = _rec(record_timestamp_monotonic_seconds=12.0, gpu_pcie_replay_count=10)
    out = compute_record(cur, prev=prev)
    # (10 - 4) / (12 - 10) = 3.0 events/s
    assert out.pcie_replay_rate_per_second == pytest.approx(3.0)
    # nonzero replay rate -> WARN health
    assert out.health == HEALTH_WARN


def test_pcie_replay_rate_none_on_counter_reset():
    prev = _rec(record_timestamp_monotonic_seconds=10.0, gpu_pcie_replay_count=10)
    cur = _rec(record_timestamp_monotonic_seconds=12.0, gpu_pcie_replay_count=2)
    out = compute_record(cur, prev=prev)
    assert out.pcie_replay_rate_per_second is None


def test_pcie_replay_rate_none_on_nonpositive_dt():
    prev = _rec(record_timestamp_monotonic_seconds=12.0, gpu_pcie_replay_count=4)
    cur = _rec(record_timestamp_monotonic_seconds=12.0, gpu_pcie_replay_count=10)
    out = compute_record(cur, prev=prev)
    assert out.pcie_replay_rate_per_second is None


def test_pcie_replay_rate_none_when_count_missing():
    prev = _rec(record_timestamp_monotonic_seconds=10.0)  # no replay count
    cur = _rec(record_timestamp_monotonic_seconds=12.0, gpu_pcie_replay_count=10)
    out = compute_record(cur, prev=prev)
    assert out.pcie_replay_rate_per_second is None


# ── pipeline: compute_tick ───────────────────────────────────────────────────

def test_compute_tick_threads_prev_per_index():
    prev_by_index = {}

    tick1 = [
        _rec(entity_gpu_index=0, record_timestamp_monotonic_seconds=10.0,
             gpu_pcie_replay_count=0),
        _rec(entity_gpu_index=1, record_timestamp_monotonic_seconds=10.0,
             gpu_pcie_replay_count=100),
    ]
    out1 = compute_tick(tick1, prev_by_index, weights=PRESETS["ai"])
    assert len(out1) == 2
    assert all(o.pcie_replay_rate_per_second is None for o in out1)  # no prev yet
    assert set(prev_by_index) == {0, 1}

    tick2 = [
        _rec(entity_gpu_index=0, record_timestamp_monotonic_seconds=12.0,
             gpu_pcie_replay_count=6),
        _rec(entity_gpu_index=1, record_timestamp_monotonic_seconds=12.0,
             gpu_pcie_replay_count=100),
    ]
    out2 = compute_tick(tick2, prev_by_index, weights=PRESETS["ai"])
    by_index = {o.gpu_index: o for o in out2}
    assert by_index[0].pcie_replay_rate_per_second == pytest.approx(3.0)  # (6-0)/2
    assert by_index[1].pcie_replay_rate_per_second == pytest.approx(0.0)  # (100-100)/2


def test_compute_tick_creates_prev_dict_when_omitted():
    out = compute_tick([_rec()], weights=PRESETS["ai"])
    assert len(out) == 1
    assert isinstance(out[0], ComputedRecord)
