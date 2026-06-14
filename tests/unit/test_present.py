"""Unit tests for the Present layer (Layer 4): CSV export, render, history."""
import io

import pytest
from rich.console import Console

from kempnerpulse.compute.result import (
    BottleneckCategory,
    ComputedRecord,
    WorkloadClass,
)
from kempnerpulse.translate import (
    SCHEMA_VERSION,
    AggregationMode,
    CanonicalRecord,
    Provenance,
)
from kempnerpulse.present import (
    CommandController,
    HistoryStore,
    SummaryContext,
    csv_header,
    csv_row,
    render_dashboard,
    resolve_columns,
    update_history,
)
from kempnerpulse.present import UnknownExportColumns


# ── Builders ──────────────────────────────────────────────────────────────────

def _canonical(**overrides) -> CanonicalRecord:
    """A minimal schema-valid POINT CanonicalRecord; override fields by keyword."""
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


def _computed(
    *,
    gpu_index: int = 0,
    real_util: float = 42.0,
    workload_class: WorkloadClass = WorkloadClass.TENSOR_HEAVY_COMPUTE,
    health: str = "OK",
    health_style: str = "green",
    model_name: str = "NVIDIA H100 80GB HBM3",
    memory_total_mebibytes=81559.0,
    memory_used_fraction=0.25,
    pcie_replay_rate_per_second=0.0,
    canonical=None,
    **canonical_overrides,
) -> ComputedRecord:
    """A hand-built ComputedRecord wrapping a CanonicalRecord."""
    if canonical is None:
        canonical = _canonical(entity_gpu_index=gpu_index, **canonical_overrides)
    return ComputedRecord(
        record=canonical,
        gpu_index=gpu_index,
        gpu_uuid=canonical.entity_gpu_uuid,
        model_name=model_name,
        real_util=real_util,
        preset_name="ai",
        weights=(0.35, 0.35, 0.20, 0.10),
        workload_class=workload_class,
        bottleneck=workload_class.bottleneck,
        health=health,
        health_style=health_style,
        memory_total_mebibytes=memory_total_mebibytes,
        memory_used_fraction=memory_used_fraction,
        pcie_replay_rate_per_second=pcie_replay_rate_per_second,
    )


def _fully_populated() -> ComputedRecord:
    """A ComputedRecord with every CSV-relevant canonical field set."""
    canon = _canonical(
        gpu_streaming_multiprocessor_active_cycle_fraction=0.90,
        gpu_streaming_multiprocessor_warp_occupancy_fraction=0.55,
        gpu_tensor_core_pipe_active_cycle_fraction=0.4830,
        gpu_tensor_core_half_precision_mma_active_cycle_fraction=0.45,
        gpu_tensor_core_integer_mma_active_cycle_fraction=0.0,
        gpu_tensor_core_double_precision_fma_active_cycle_fraction=0.0,
        gpu_tensor_core_double_mma_active_cycle_fraction=0.0,
        gpu_tensor_core_quarter_mma_active_cycle_fraction=0.0,
        gpu_dram_controller_active_cycle_fraction=0.30,
        gpu_graphics_compute_engine_active_cycle_fraction=0.88,
        gpu_nvml_busy_time_fraction=1.0,
        gpu_cuda_core_floating_point_16bit_pipe_active_cycle_fraction=0.10,
        gpu_cuda_core_floating_point_32bit_pipe_active_cycle_fraction=0.20,
        gpu_cuda_core_floating_point_64bit_pipe_active_cycle_fraction=0.0,
        gpu_memory_copy_engine_busy_time_fraction=0.12,
        gpu_pcie_receive_throughput_bytes_per_second=1.5e9,
        gpu_pcie_transmit_throughput_bytes_per_second=2.5e9,
        gpu_nvlink_aggregate_throughput_bytes_per_second=900e9,  # -> 900 GB/s
        gpu_board_power_draw_watts=351.0,
        gpu_board_total_energy_joules=12345.6,
        gpu_die_temperature_celsius=62.0,
        gpu_memory_die_temperature_celsius=70.0,
        gpu_streaming_multiprocessor_clock_frequency_megahertz=1980.0,
        gpu_memory_clock_frequency_megahertz=1593.0,
        gpu_framebuffer_used_mebibytes=11050.0,
    )
    return _computed(
        canonical=canon,
        real_util=73.25,
        workload_class=WorkloadClass.TENSOR_HEAVY_COMPUTE,
        memory_total_mebibytes=81559.0,
        memory_used_fraction=11050.0 / 81559.0,
        pcie_replay_rate_per_second=0.0,
    )


# ── (a) CSV column registry ────────────────────────────────────────────────────

def test_csv_default_columns_values_and_units():
    rec = _fully_populated()
    cols = resolve_columns("default")
    header = csv_header(cols)
    assert header == [
        "timestamp", "gpu_id", "model", "gpu_util_pct", "mem_used_mib",
        "real_util_pct", "sm_active_pct", "tensor_active_pct", "dram_active_pct",
    ]
    row = dict(zip(header, csv_row(rec, 1234.5, cols)))
    assert row["timestamp"] == "1234.50"           # .2f
    assert row["gpu_id"] == "0"
    assert row["model"] == "H100"                   # shortened
    assert row["gpu_util_pct"] == "100.00"          # 1.0 fraction -> 100.00
    assert row["mem_used_mib"] == "11050.0000"      # raw .4f
    assert row["real_util_pct"] == "73.25"          # from ComputedRecord .2f
    assert row["sm_active_pct"] == "90.00"          # 0.90 -> 90.00
    assert row["tensor_active_pct"] == "48.30"      # 0.4830 -> 48.30
    assert row["dram_active_pct"] == "30.00"


def test_csv_all_columns_values_units_and_precision():
    rec = _fully_populated()
    cols = resolve_columns("all")
    header = csv_header(cols)
    row = dict(zip(header, csv_row(rec, 10.0, cols)))

    # Identity / derived
    assert row["status"] == "tensor-heavy compute"
    assert row["health"] == "OK"
    # Percent columns: fraction * 100, .2f
    assert row["gr_engine_active_pct"] == "88.00"
    assert row["sm_occupancy_pct"] == "55.00"
    assert row["fp16_pipe_pct"] == "10.00"
    assert row["fp32_pipe_pct"] == "20.00"
    assert row["memcpy_util_pct"] == "12.00"
    assert row["tc_hmma_pct"] == "45.00"
    # PCIe bytes/s raw .4f
    assert row["pcie_rx_bytes_s"] == "1500000000.0000"
    assert row["pcie_tx_bytes_s"] == "2500000000.0000"
    # NVLink GB/s = bytes/s / 1e9, .4f
    assert row["nvlink_gbps"] == "900.0000"
    # Power / temp / clock raw .4f
    assert row["power_w"] == "351.0000"
    assert row["gpu_temp_c"] == "62.0000"
    assert row["mem_temp_c"] == "70.0000"
    assert row["sm_clock_mhz"] == "1980.0000"
    assert row["mem_clock_mhz"] == "1593.0000"
    # Memory totals / pct
    assert row["mem_total_mib"] == "81559.0"        # .1f
    assert row["mem_used_pct"] == "13.55"           # (11050/81559)*100, .2f
    # Energy cumulative .1f
    assert row["energy_j"] == "12345.6"
    # Replay rate .2f
    assert row["pcie_replay_rate_s"] == "0.00"


def test_csv_na_stays_empty_not_zero():
    """A record missing optional readings emits empty fields, never '0'."""
    rec = _computed(
        memory_total_mebibytes=None,
        memory_used_fraction=None,
        pcie_replay_rate_per_second=None,
    )  # canonical has all metric fields None by default
    cols = resolve_columns("all")
    header = csv_header(cols)
    row = dict(zip(header, csv_row(rec, 0.0, cols)))

    for empty_col in [
        "sm_active_pct", "tensor_active_pct", "dram_active_pct", "gr_engine_active_pct",
        "gpu_util_pct", "mem_used_mib", "power_w", "gpu_temp_c", "mem_temp_c",
        "sm_occupancy_pct", "fp16_pipe_pct", "fp32_pipe_pct", "fp64_pipe_pct",
        "memcpy_util_pct", "pcie_rx_bytes_s", "pcie_tx_bytes_s", "nvlink_gbps",
        "sm_clock_mhz", "mem_clock_mhz", "pcie_replay_rate_s", "energy_j",
        "mem_total_mib", "mem_used_pct", "tc_hmma_pct", "tc_imma_pct",
        "tc_dfma_pct", "tc_dmma_pct", "tc_qmma_pct",
    ]:
        assert row[empty_col] == "", f"{empty_col} should be empty for an absent reading"

    # Non-metric columns are still populated.
    assert row["gpu_id"] == "0"
    assert row["real_util_pct"] == "42.00"
    assert row["status"] == "tensor-heavy compute"
    assert row["health"] == "OK"


def test_csv_custom_column_spec_and_unknown():
    cols = resolve_columns("gpu_id,real_util_pct")
    assert csv_header(cols) == ["gpu_id", "real_util_pct"]
    with pytest.raises(UnknownExportColumns) as exc:
        resolve_columns("gpu_id,not_a_column")
    assert "not_a_column" in str(exc.value)


# ── (b) Render smoke ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n_records", [1, 2, 4])
@pytest.mark.parametrize("size", [(120, 40), (200, 50), (90, 30), (300, 60)])
def test_render_dashboard_smoke(n_records, size):
    width, height = size
    records = []
    for i in range(n_records):
        records.append(_fully_populated() if i == 0 else _computed(
            gpu_index=i,
            entity_gpu_uuid=f"GPU-{i}",
            real_util=float(10 * i),
            gpu_streaming_multiprocessor_active_cycle_fraction=0.5,
        ))
    history = HistoryStore()
    update_history(history, records)
    controller = CommandController()
    ctx = SummaryContext(source="http://localhost:9400/metrics", poll=1.0,
                         selection_desc=f"{n_records} GPU(s)", app_version="0.5.0")

    out = render_dashboard(
        records, history,
        console_width=width, console_height=height,
        controller=controller, summary_context=ctx,
    )
    # Render to a throwaway console to actually exercise the layout engine.
    console = Console(file=io.StringIO(), width=width, height=height)
    console.print(out)
    text = console.file.getvalue()
    assert text  # produced output without raising


@pytest.mark.parametrize("mode", ["fleet", "plot", "jobs", "focus"])
def test_render_dashboard_all_view_modes(mode):
    records = [_fully_populated(), _computed(gpu_index=1, entity_gpu_uuid="GPU-1")]
    history = HistoryStore()
    update_history(history, records)
    controller = CommandController()
    if mode == "plot":
        controller.line_mode = True
    elif mode == "jobs":
        controller.jobs_mode = True
    elif mode == "focus":
        controller.focus_gpu = "0"
    ctx = SummaryContext()
    out = render_dashboard(
        records, history,
        console_width=200, console_height=50,
        controller=controller, summary_context=ctx,
    )
    console = Console(file=io.StringIO(), width=200, height=50)
    console.print(out)
    assert console.file.getvalue()


def test_render_dashboard_too_small_gate():
    """Below the per-card minimum a placeholder renders (no raise)."""
    records = [_fully_populated()]
    history = HistoryStore()
    controller = CommandController()
    out = render_dashboard(
        records, history,
        console_width=10, console_height=5,
        controller=controller, summary_context=SummaryContext(),
    )
    console = Console(file=io.StringIO(), width=10, height=5)
    console.print(out)
    assert "Too" in console.file.getvalue()


def test_render_dashboard_empty_records():
    history = HistoryStore()
    controller = CommandController()
    out = render_dashboard(
        [], history,
        console_width=120, console_height=40,
        controller=controller, summary_context=SummaryContext(selection_desc="none"),
    )
    console = Console(file=io.StringIO(), width=120, height=40)
    console.print(out)
    assert "No GPU data" in console.file.getvalue()


# ── (c) HistoryStore / update_history round-trip ────────────────────────────────

def test_history_store_push_get_roundtrip_and_maxlen():
    hist = HistoryStore(maxlen=3)
    for v in (1.0, 2.0, 3.0, 4.0):
        hist.push("0", "real_util", v)
    series = list(hist.get("0", "real_util"))
    assert series == [2.0, 3.0, 4.0]              # oldest dropped at maxlen
    # Unknown gpu / key returns an empty deque, not an error.
    assert list(hist.get("9", "real_util")) == []
    assert list(hist.get("0", "nope")) == []


def test_update_history_pushes_display_units():
    rec = _fully_populated()
    hist = HistoryStore()
    update_history(hist, [rec])

    # Percent series come from canonical fractions * 100 (float-multiply, so approx).
    assert list(hist.get("0", "sm_active"))[-1] == pytest.approx(90.0)
    assert list(hist.get("0", "tensor"))[-1] == pytest.approx(48.30)
    assert list(hist.get("0", "dram"))[-1] == pytest.approx(30.0)
    assert list(hist.get("0", "gpu_util"))[-1] == pytest.approx(100.0)
    assert list(hist.get("0", "gr_active"))[-1] == pytest.approx(88.0)
    assert list(hist.get("0", "sm_occupancy"))[-1] == pytest.approx(55.0)
    assert list(hist.get("0", "fp16"))[-1] == pytest.approx(10.0)
    assert list(hist.get("0", "memcpy"))[-1] == pytest.approx(12.0)
    assert list(hist.get("0", "tc_hmma"))[-1] == pytest.approx(45.0)

    # Real util is the composite as-is (already 0..100).
    assert list(hist.get("0", "real_util")) == [73.25]

    # Memory used % from the convenience fraction.
    assert list(hist.get("0", "mem_used_pct"))[-1] == pytest.approx(13.55, abs=0.01)

    # Raw display-unit series.
    assert list(hist.get("0", "power")) == [351.0]
    assert list(hist.get("0", "gpu_temp")) == [62.0]
    assert list(hist.get("0", "pcie_rx")) == [1.5e9]
    assert list(hist.get("0", "pcie_tx")) == [2.5e9]
    assert list(hist.get("0", "pcie_rxtx")) == [4.0e9]

    # NVLink GB/s = bytes/s / 1e9.
    assert list(hist.get("0", "nvlink_gbps")) == [900.0]


def test_update_history_skips_absent_series():
    """Absent canonical readings are not pushed (no zero-fill)."""
    rec = _computed()  # canonical metric fields all None
    hist = HistoryStore()
    update_history(hist, [rec])
    # real_util is always pushed; mem_used_pct present via fraction.
    assert list(hist.get("0", "real_util")) == [42.0]
    assert list(hist.get("0", "mem_used_pct"))[-1] == pytest.approx(25.0)
    # Everything sourced from an absent canonical field stays empty.
    for absent in ("sm_active", "tensor", "dram", "gpu_util", "power",
                   "gpu_temp", "pcie_rx", "pcie_rxtx", "nvlink_gbps"):
        assert list(hist.get("0", absent)) == []
