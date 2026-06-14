"""Rich panels and the line-plot renderable.

Every widget consumes :class:`ComputedRecord`s and reads metric values from the
wrapped :class:`CanonicalRecord` (canonical fractions/SI), converting to display
units at render time via :mod:`.format`. The responsive layout — fixed-width
bars, stacked-vs-two-column card detail, fixed-width health/status badges,
no-wrap headers, summary/footer field-drop ordering, fleet vertical scrolling,
and the focus-view reflowing info grid — is reproduced from the single-file
implementation, with the magic numbers kept as named constants.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..compute.result import ComputedRecord, HEALTH_LABELS, WORKLOAD_STATUS_LABELS
from .controller import CommandController, SCROLL_PAGE  # noqa: F401  (SCROLL_PAGE re-exported)
from .format import (
    bytes_per_second_to_gigabytes,
    fmt_bytes_per_s,
    fmt_gbps,
    fmt_joules,
    fmt_mhz,
    fmt_mib,
    fmt_num,
    fmt_pct,
    fmt_temp,
    fmt_watts,
    fmt_duration,
    fraction_to_percent,
    make_bar,
    nvlink_util_style,
    power_style,
    sparkline,
    temp_style,
    usage_style,
)
from .history import HistoryStore

APP_NAME = "KempnerPulse GPU Dashboard"

# ── Fixed-width badge widths (no jitter as values change) ─────────────────────
HEALTH_LABEL_WIDTH = max(len(h) for h in HEALTH_LABELS)
STATUS_LABEL_MARGIN = 2
STATUS_DISPLAY_WIDTH = max(len(s) for s in WORKLOAD_STATUS_LABELS) + STATUS_LABEL_MARGIN

# ── Line-chart glyphs & palette ───────────────────────────────────────────────
_CH_HLINE = "─"
_CH_VLINE = "│"
_CH_ULCORNER = "┌"
_CH_URCORNER = "┐"
_CH_LLCORNER = "└"
_CH_LRCORNER = "┘"
LINE_PLOT_COLORS = [
    "green", "cyan", "yellow", "magenta", "red", "blue", "white", "bright_green",
]

# ── Default Real Utilization weight presets (name lookup for the footer) ──────
WEIGHT_PRESETS = {
    (0.35, 0.35, 0.20, 0.10): "AI/ML Workflow",
    (0.45, 0.15, 0.25, 0.15): "HPC Workflow",
    (0.35, 0.10, 0.40, 0.15): "Memory-bound Workflow",
}


def workflow_label(weights: Tuple[float, float, float, float]) -> str:
    rounded = tuple(round(w, 2) for w in weights)
    return WEIGHT_PRESETS.get(rounded, "Custom Workflow")


# ── Canonical metric accessors (return display units; None when unavailable) ──

def _pct_of(rec: ComputedRecord, field_name: str) -> Optional[float]:
    """Read a canonical fraction field and return it as a percent (or None)."""
    return fraction_to_percent(getattr(rec.record, field_name))


def _raw(rec: ComputedRecord, field_name: str):
    return getattr(rec.record, field_name)


def _nvlink_gbps(rec: ComputedRecord) -> Optional[float]:
    return bytes_per_second_to_gigabytes(
        rec.record.gpu_nvlink_aggregate_throughput_bytes_per_second
    )


def _energy_j(rec: ComputedRecord) -> Optional[float]:
    return rec.record.gpu_board_total_energy_joules


def _mem_used_pct(rec: ComputedRecord) -> Optional[float]:
    return fraction_to_percent(rec.memory_used_fraction)


# ── Job rows ──────────────────────────────────────────────────────────────────

@dataclass
class GpuProcess:
    """A single compute process running on a GPU (for the Job View)."""
    pid: int
    user: str
    gid: str
    gpu_id: str
    gpu_mem_mib: Optional[float]
    command: str


# ══════════════════════════════════════════════════════════════════════════════
# Summary panel
# ══════════════════════════════════════════════════════════════════════════════

# Each summary field needs ~SUMMARY_FIELD_MIN_WIDTH columns to stay readable; as
# the bar narrows, the fields in SUMMARY_DROP_ORDER are dropped one by one.
SUMMARY_FIELD_MIN_WIDTH = 16
SUMMARY_DROP_ORDER = ("CPU", "RAM", "FB used", "Health", "Power")


def summary_panel(
    records: List[ComputedRecord],
    *,
    app_version: str = "",
    cpu_info: Tuple[Optional[int], Optional[int], Optional[float], Optional[int]] = (None, None, None, None),
    ram_info: Tuple[Optional[float], Optional[float]] = (None, None),
    console_width: int = 200,
) -> Panel:
    n = len(records)
    avg_real = sum(r.real_util for r in records) / n if n else 0.0
    avg_power = sum((_raw(r, "gpu_board_power_draw_watts") or 0.0) for r in records) / n if n else 0.0
    total_power = sum((_raw(r, "gpu_board_power_draw_watts") or 0.0) for r in records)
    total_fb_used = sum((_raw(r, "gpu_framebuffer_used_mebibytes") or 0.0) for r in records)
    total_fb = sum((r.memory_total_mebibytes or 0.0) for r in records)
    mem_pct = 100.0 * total_fb_used / total_fb if total_fb > 0 else 0.0
    active = sum(
        1 for r in records
        if r.real_util >= 20 or ((_mem_used_pct(r) or 0) >= 20)
    )
    critical = sum(1 for r in records if r.health != "OK")
    cpu_threads, cpu_cores, cpu_pct, cpu_busy = cpu_info
    ram_used_gb, ram_total_gb = ram_info

    # CPU text: " 32 / 64 (50.0%)" - fixed width.
    if cpu_cores is not None and cpu_busy is not None and cpu_pct is not None:
        core_w = len(str(cpu_cores))
        cpu_text = f"{cpu_busy:>{core_w}} / {cpu_cores} ({cpu_pct:5.1f}%)"
    elif cpu_cores is not None:
        core_w = len(str(cpu_cores))
        cpu_text = f"{'--':>{core_w}} / {cpu_cores} ( -- %)"
    else:
        cpu_text = "--"

    def _fmt_ram(gb: Optional[float]) -> str:
        if gb is None:
            return "  -- "
        if gb >= 1024:
            return f"{gb / 1024:5.1f}TB"
        return f"{gb:5.1f}GB"

    if ram_used_gb is not None and ram_total_gb is not None and ram_total_gb > 0:
        ram_pct = 100.0 * ram_used_gb / ram_total_gb
        ram_text = f"{_fmt_ram(ram_used_gb)} / {_fmt_ram(ram_total_gb)} ({ram_pct:5.1f}%)"
    else:
        ram_text = "--"
        ram_pct = 0.0

    def _fmt_fb(mib: Optional[float]) -> str:
        if mib is None:
            return "   --  "
        if mib >= 1024:
            return f"{mib / 1024:6.1f}GiB"
        return f"{mib:6.0f}MiB"

    fb_text = f"{_fmt_fb(total_fb_used)} / {_fmt_fb(total_fb)} ({mem_pct:5.1f}%)"

    fields = [
        ("GPUs", Text(f"GPUs\n{n}", style="bold cyan", justify="center")),
        ("Active", Text(f"Active\n{active}", style="bold green" if active else "dim", justify="center")),
        ("Avg real util", Text(f"Avg real util\n{avg_real:.1f}%", style=usage_style(avg_real), justify="center")),
        ("Power", Text(f"Power (tot/avg)\n{total_power:.0f}W / {avg_power:.0f}W", style=power_style(avg_power) if n else "dim", justify="center")),
        ("FB used", Text(f"FB used\n{fb_text}", style=usage_style(mem_pct), justify="center")),
        ("CPU", Text(f"CPU\n{cpu_text}", style=usage_style(cpu_pct) if cpu_pct is not None else "dim", justify="center")),
        ("RAM", Text(f"RAM\n{ram_text}", style=usage_style(ram_pct), justify="center")),
        ("Health", Text(f"Health\n{critical} warn/crit", style="bold red" if critical else "green", justify="center")),
    ]
    to_drop = list(SUMMARY_DROP_ORDER)
    while len(fields) * SUMMARY_FIELD_MIN_WIDTH > max(1, console_width) and to_drop:
        drop_name = to_drop.pop(0)
        fields = [f for f in fields if f[0] != drop_name]
    grid = Table.grid(expand=True)
    for _ in fields:
        grid.add_column(justify="center")
    grid.add_row(*(cell for _, cell in fields))
    title = f"{APP_NAME} (v{app_version})" if app_version else APP_NAME
    return Panel(grid, title=title, border_style="cyan", box=box.ROUNDED)


# ══════════════════════════════════════════════════════════════════════════════
# Fleet View (GPU cards)
# ══════════════════════════════════════════════════════════════════════════════

# Fleet-card bar-block layout.
FLEET_BAR_WIDTH = 12        # filled/empty cells in each real/mem/pwr bar
FLEET_BAR_LABEL_WIDTH = 4   # width of the "real"/"mem "/"pwr " label column
FLEET_BAR_GAP = 2           # spacer between the three bar groups

# Fleet-view responsive layout breakpoints (console columns).
CARD_DETAIL_LABEL_MIN = 11        # detail label column min width (left & right grids)
CARD_DETAIL_LEFT_VALUE_MIN = 18   # left detail value column min width
CARD_DETAIL_RIGHT_VALUE_MIN = 22  # right detail value column min width
CARD_DETAIL_GAP = 2               # horizontal padding between the two detail columns
CARD_2COL_MIN_WIDTH = (CARD_DETAIL_LABEL_MIN + CARD_DETAIL_LEFT_VALUE_MIN + CARD_DETAIL_GAP
                       + CARD_DETAIL_LABEL_MIN + CARD_DETAIL_RIGHT_VALUE_MIN)
CARD_BORDER_PAD = 4               # panel border + internal padding overhead per card
CARD_FULL_WIDTH = CARD_2COL_MIN_WIDTH + CARD_BORDER_PAD

W_MIN_1COL = CARD_DETAIL_LABEL_MIN + CARD_DETAIL_RIGHT_VALUE_MIN + CARD_BORDER_PAD
CARD_MIN_HEIGHT = 13                       # title + ~9 detail rows + 3 bars (2-col card)
CARD_1COL_HEIGHT = 24                      # stacked card: title + 18 detail rows + 3 vertical bars + border
SUMMARY_PANEL_ROWS = 5
FOOTER_PANEL_ROWS = 3
FLEET_PANEL_BORDER = 2
MIN_DASH_WIDTH = W_MIN_1COL + 4            # one narrow card + panel slack
MIN_DASH_HEIGHT = CARD_MIN_HEIGHT + SUMMARY_PANEL_ROWS + FOOTER_PANEL_ROWS + FLEET_PANEL_BORDER
A_CARD = CARD_FULL_WIDTH / CARD_MIN_HEIGHT  # ~5.23: a full card is ~5x wider than tall
RAGGED_WEIGHT = 0.05                         # score penalty per empty trailing grid cell
UTIL_WEIGHT = 0.15                           # bounded reward for using affordable columns
MAX_GPUS = 8


def _card_header(rec: ComputedRecord) -> Text:
    """No-wrap "GPU <id>  <model>  [health]" header (truncates, never wraps)."""
    gpu = rec.gpu_id
    name = rec.model_name or "GPU"
    header_parts = [(f"GPU {gpu}  ", "bold"), (f"{name}  ", "dim"),
                    (f"[{rec.health:^{HEALTH_LABEL_WIDTH}}]", rec.health_style)]
    header = Text.assemble(*header_parts)
    header.no_wrap = True
    header.overflow = "ellipsis"
    return header


def gpu_card(
    rec: ComputedRecord,
    history: HistoryStore,
    power_limit: Optional[float] = None,
    nvlink_limit: Optional[float] = None,
    detail_columns: int = 2,
) -> Panel:
    gpu = rec.gpu_id

    gpu_util = _pct_of(rec, "gpu_nvml_busy_time_fraction")
    gr_active = _pct_of(rec, "gpu_graphics_compute_engine_active_cycle_fraction")
    sm_active = _pct_of(rec, "gpu_streaming_multiprocessor_active_cycle_fraction")
    sm_occ = _pct_of(rec, "gpu_streaming_multiprocessor_warp_occupancy_fraction")
    tensor = _pct_of(rec, "gpu_tensor_core_pipe_active_cycle_fraction")
    dram = _pct_of(rec, "gpu_dram_controller_active_cycle_fraction")

    sm_combo_style = usage_style(sm_active)
    _power_w = _raw(rec, "gpu_board_power_draw_watts")
    _power_text = (f"{fmt_watts(_power_w)} / {fmt_watts(power_limit)}"
                   if power_limit else fmt_watts(_power_w))
    _gpu_t = _raw(rec, "gpu_die_temperature_celsius")
    _mem_t = _raw(rec, "gpu_memory_die_temperature_celsius")
    _max_t = max(t for t in (_gpu_t, _mem_t) if t is not None) if (_gpu_t is not None or _mem_t is not None) else None
    memcpy_pct = _pct_of(rec, "gpu_memory_copy_engine_busy_time_fraction")
    nvlink_gbps = _nvlink_gbps(rec)
    nvlink_text = "N/A" if nvlink_limit is None and nvlink_gbps is None else fmt_gbps(nvlink_gbps)
    _replay = rec.pcie_replay_rate_per_second

    left_rows = [
        ("Real util", Text(fmt_pct(rec.real_util), style=usage_style(rec.real_util))),
        ("GPU util", Text(fmt_pct(gpu_util), style=usage_style(gpu_util))),
        ("GR active", Text(fmt_pct(gr_active), style=usage_style(gr_active))),
        ("SM actv/occ", Text(f"{fmt_pct(sm_active)} / {fmt_pct(sm_occ)}", style=sm_combo_style)),
        ("Tensor", Text(fmt_pct(tensor), style=usage_style(tensor))),
        ("DRAM", Text(fmt_pct(dram), style=usage_style(dram))),
        ("Memory", Text(
            f"{fmt_mib(_raw(rec, 'gpu_framebuffer_used_mebibytes'))} / "
            f"{fmt_mib(rec.memory_total_mebibytes)} ({fmt_pct(_mem_used_pct(rec))})",
            style=usage_style(_mem_used_pct(rec)))),
        ("Power", Text(_power_text, style=power_style(_power_w))),
        ("Temps", Text(f"GPU {fmt_temp(_gpu_t)} | MEM {fmt_temp(_mem_t)}",
                       style=temp_style(_max_t, rec.model_name))),
    ]
    right_rows = [
        ("Memcpy", Text(fmt_pct(memcpy_pct), style=usage_style(memcpy_pct))),
        ("PCIe RX", Text(fmt_bytes_per_s(_raw(rec, "gpu_pcie_receive_throughput_bytes_per_second")), style="cyan")),
        ("PCIe TX", Text(fmt_bytes_per_s(_raw(rec, "gpu_pcie_transmit_throughput_bytes_per_second")), style="cyan")),
        ("NVLink Δ", Text(nvlink_text, style=nvlink_util_style(nvlink_gbps, nvlink_limit))),
        ("PCIe replay", Text(fmt_num(_replay, 2) + "/s", style="yellow" if (_replay or 0) > 0 else "dim")),
        ("SM clock", Text(fmt_mhz(_raw(rec, "gpu_streaming_multiprocessor_clock_frequency_megahertz")), style="green")),
        ("MEM clock", Text(fmt_mhz(_raw(rec, "gpu_memory_clock_frequency_megahertz")), style="green")),
        ("Energy", Text(fmt_joules(_energy_j(rec)), style="magenta")),
        ("Status", Text(f"{rec.status_line:<{STATUS_DISPLAY_WIDTH}}", style=rec.health_style)),
    ]

    def _detail_grid(rows, value_min):
        g = Table.grid(padding=(0, 1))
        g.add_column(justify="left", min_width=CARD_DETAIL_LABEL_MIN, no_wrap=True)
        g.add_column(justify="right", min_width=value_min, no_wrap=True)
        for _lbl, _val in rows:
            g.add_row(_lbl, _val)
        return g

    if detail_columns >= 2:
        table = Table.grid(expand=True, padding=(0, CARD_DETAIL_GAP))
        table.add_column(ratio=1)
        table.add_column(ratio=1)
        table.add_row(_detail_grid(left_rows, CARD_DETAIL_LEFT_VALUE_MIN),
                      _detail_grid(right_rows, CARD_DETAIL_RIGHT_VALUE_MIN))
        detail_block = table
    else:
        detail_block = _detail_grid(left_rows + right_rows, CARD_DETAIL_RIGHT_VALUE_MIN)

    power_w = (_raw(rec, "gpu_board_power_draw_watts") or 0.0)
    power_cap = power_limit if power_limit and power_limit > 0 else 700.0
    power_pct = min(100.0, power_w / power_cap * 100.0)
    mem_used_pct = _mem_used_pct(rec)
    bw = FLEET_BAR_WIDTH
    bars = Table.grid(expand=False)
    if detail_columns >= 2:
        bars.add_column(width=FLEET_BAR_LABEL_WIDTH, no_wrap=True)
        bars.add_column(width=FLEET_BAR_WIDTH, no_wrap=True)
        bars.add_column(width=FLEET_BAR_GAP)
        bars.add_column(width=FLEET_BAR_LABEL_WIDTH, no_wrap=True)
        bars.add_column(width=FLEET_BAR_WIDTH, no_wrap=True)
        bars.add_column(width=FLEET_BAR_GAP)
        bars.add_column(width=FLEET_BAR_LABEL_WIDTH, no_wrap=True)
        bars.add_column(width=FLEET_BAR_WIDTH, no_wrap=True)
        bars.add_row(
            Text("real", style="dim"), make_bar(rec.real_util, width=bw),
            Text(""),
            Text("mem ", style="dim"), make_bar(mem_used_pct, width=bw),
            Text(""),
            Text("pwr ", style="dim"), make_bar(power_pct, width=bw, style_override=power_style(power_w)),
        )
    else:
        bars.add_column(width=FLEET_BAR_LABEL_WIDTH, no_wrap=True)
        bars.add_column(width=FLEET_BAR_WIDTH, no_wrap=True)
        bars.add_row(Text("real", style="dim"), make_bar(rec.real_util, width=bw))
        bars.add_row(Text("mem ", style="dim"), make_bar(mem_used_pct, width=bw))
        bars.add_row(Text("pwr ", style="dim"), make_bar(power_pct, width=bw, style_override=power_style(power_w)))

    body = Group(_card_header(rec), detail_block, bars)
    border = "red" if rec.health == "CRIT" else "yellow" if rec.health != "OK" else "blue"
    return Panel(body, box=box.ROUNDED, border_style=border)


def candidate_cols(n: int) -> List[int]:
    """Sensible column counts for n cards: full row (n), full column (1), exact
    divisors, and balanced ragged grids; pruned so every row but the last is full."""
    cols = {1, n}
    for c in range(1, n + 1):
        if n % c == 0:
            cols.add(c)
    for r in range(1, n + 1):
        cols.add(-(-n // r))
    out = []
    for c in sorted(cols):
        rows = -(-n // c)
        if (rows - 1) * c < n:
            out.append(c)
    return out


def choose_grid(n: int, W: int, H: int, w_min: Optional[int] = None) -> Tuple[int, int]:
    """Choose (cols, rows) for n cards in a W×H area, matching the grid aspect ratio
    to the window's. Never more columns than fit at the card minimum width. Deterministic."""
    n = max(1, min(MAX_GPUS, n))
    if n == 1:
        return (1, 1)
    w_min = W_MIN_1COL if w_min is None else w_min
    fit_cols = max(1, W // w_min)
    feasible = [c for c in candidate_cols(n) if c <= fit_cols] or [1]
    a_win = max(0.01, W / max(1, H))

    def score(c: int):
        rows = -(-n // c)
        a_grid = (c / rows) * A_CARD
        mismatch = abs(math.log(a_grid) - math.log(a_win))
        ragged = RAGGED_WEIGHT * (c * rows - n)
        util = UTIL_WEIGHT * min(c, fit_cols) / max(1, fit_cols)
        return (mismatch + ragged - util, rows, -c, c)

    cols = min(feasible, key=score)
    return (cols, -(-n // cols))


def fleet_panel(
    records: List[ComputedRecord],
    history: HistoryStore,
    cards_per_row: int = 2,
    detail_columns: int = 2,
    power_limits: Optional[Dict[str, float]] = None,
    nvlink_bw_limits: Optional[Dict[str, float]] = None,
    avail_height: Optional[int] = None,
    controller: Optional[CommandController] = None,
) -> Panel:
    cards_per_row = max(1, cards_per_row)
    rows: List[List[Panel]] = []
    for idx in range(0, len(records), cards_per_row):
        rows.append([
            gpu_card(r, history, (power_limits or {}).get(r.gpu_id),
                     (nvlink_bw_limits or {}).get(r.gpu_id), detail_columns=detail_columns)
            for r in records[idx: idx + cards_per_row]
        ])

    # Vertical scroll: show the window of card-rows that fits the height.
    total_rows = len(rows)
    card_h = CARD_MIN_HEIGHT if detail_columns >= 2 else CARD_1COL_HEIGHT
    visible = total_rows if avail_height is None else max(1, avail_height // card_h)
    offset = controller.fleet_scroll_offset if controller is not None else 0
    offset = max(0, min(offset, max(0, total_rows - visible)))
    if controller is not None:
        controller.fleet_scroll_offset = offset
    shown = rows[offset: offset + visible]

    grid = Table.grid(expand=True)
    for _ in range(cards_per_row):
        grid.add_column(ratio=1)
    for row in shown:
        padded = row + [Text("")] * (cards_per_row - len(row))
        grid.add_row(*padded)

    title = "Fleet overview"
    if total_rows > visible:
        up = "▲" if offset > 0 else " "
        down = "▼" if offset + visible < total_rows else " "
        title = f"Fleet overview  {up}{down} {offset + 1}-{min(offset + visible, total_rows)}/{total_rows}"
    return Panel(grid, title=title, border_style="blue", box=box.ROUNDED)


def build_fleet_panel(
    records: List[ComputedRecord],
    history: HistoryStore,
    avail_width: int,
    avail_height: int,
    power_limits: Optional[Dict[str, float]] = None,
    nvlink_bw_limits: Optional[Dict[str, float]] = None,
    controller: Optional[CommandController] = None,
) -> Panel:
    """Lay out the fleet for an available width×height. Shared by the main fleet
    view and the focus-mode mini-fleet so both behave identically."""
    cols, _rows = choose_grid(len(records), avail_width, avail_height)
    detail_columns = 2 if avail_width // max(1, cols) >= CARD_FULL_WIDTH else 1
    return fleet_panel(records, history, cards_per_row=cols, detail_columns=detail_columns,
                       power_limits=power_limits, nvlink_bw_limits=nvlink_bw_limits,
                       avail_height=avail_height, controller=controller)


# ══════════════════════════════════════════════════════════════════════════════
# Line-chart renderer (Plot View)
# ══════════════════════════════════════════════════════════════════════════════

def _data_level(rows: int, value: float) -> int:
    """0-100 percentage → screen row (0 = top = 100%, rows-1 = bottom = 0%)."""
    if rows <= 1:
        return 0
    level = rows - 1 - round(value * (rows - 1) / 100.0)
    return max(0, min(rows - 1, int(level)))


def _render_line_chart(
    gpu_data: List[Tuple[str, List[float]]],
    chart_rows: int,
    chart_cols: int,
    vmax: float = 100.0,
) -> List[List[Tuple[str, int]]]:
    """Render a line chart into a 2D character grid: grid[row][col] = (char, color_index)."""
    _PRI = {' ': 0, _CH_HLINE: 1, _CH_VLINE: 2,
            _CH_ULCORNER: 2, _CH_URCORNER: 2, _CH_LLCORNER: 2, _CH_LRCORNER: 2}

    grid: List[List[Tuple[str, int]]] = [[(' ', -1)] * chart_cols for _ in range(chart_rows)]
    pri: List[List[int]] = [[0] * chart_cols for _ in range(chart_rows)]
    if chart_rows < 2 or chart_cols < 1 or not gpu_data:
        return grid

    def _put(r: int, c: int, ch: str, cidx: int) -> None:
        p = _PRI.get(ch, 1)
        if p >= pri[r][c]:
            grid[r][c] = (ch, cidx)
            pri[r][c] = p

    for line_idx, (_gpu_id, values) in enumerate(gpu_data):
        if not values:
            continue
        if vmax > 0 and vmax != 100.0:
            norm = [max(0.0, min(100.0, v / vmax * 100.0)) for v in values]
        else:
            norm = [max(0.0, min(100.0, v)) for v in values]
        if len(norm) < chart_cols:
            norm = [0.0] * (chart_cols - len(norm)) + norm
        elif len(norm) > chart_cols:
            norm = norm[-chart_cols:]

        prev_row: Optional[int] = None
        for col in range(chart_cols):
            cur_row = _data_level(chart_rows, norm[col])
            if prev_row is None or cur_row == prev_row:
                _put(cur_row, col, _CH_HLINE, line_idx)
            elif cur_row < prev_row:
                _put(cur_row, col, _CH_ULCORNER, line_idx)
                _put(prev_row, col, _CH_LRCORNER, line_idx)
                for r in range(cur_row + 1, prev_row):
                    _put(r, col, _CH_VLINE, line_idx)
            else:
                _put(prev_row, col, _CH_URCORNER, line_idx)
                _put(cur_row, col, _CH_LLCORNER, line_idx)
                for r in range(prev_row + 1, cur_row):
                    _put(r, col, _CH_VLINE, line_idx)
            prev_row = cur_row
    return grid


class LinePlotRenderable:
    """Rich renderable that draws a line chart, adapting to available width."""

    def __init__(
        self,
        gpu_data: List[Tuple[str, List[float]]],
        chart_rows: int = 10,
        vmax: float = 100.0,
        poll: float = 1.0,
    ):
        self.gpu_data = gpu_data
        self.chart_rows = chart_rows
        self.vmax = vmax
        self.poll = poll

    def __rich_console__(self, console, options):
        width = options.max_width
        y_label_w = 4                         # "100 " is 4 chars
        chart_cols = max(1, width - y_label_w)

        grid = _render_line_chart(self.gpu_data, self.chart_rows, chart_cols, self.vmax)

        label_rows: Dict[int, int] = {}
        for pct in (100, 75, 50, 25, 0):
            r = _data_level(self.chart_rows, float(pct))
            if r not in label_rows:
                label_rows[r] = pct

        for row_idx in range(self.chart_rows):
            line = Text()
            if row_idx in label_rows:
                line.append(f"{label_rows[row_idx]:>3} ", style="dim")
            else:
                line.append("    ", style="dim")
            for char, cidx in grid[row_idx]:
                if cidx >= 0:
                    line.append(char, style=LINE_PLOT_COLORS[cidx % len(LINE_PLOT_COLORS)])
                else:
                    line.append(char)
            yield line

        if self.poll > 0:
            x_line = Text()
            x_line.append(" " * y_label_w)
            ruler = [" "] * chart_cols
            total_s = chart_cols * self.poll
            n_ticks = min(5, max(2, chart_cols // 20))
            for i in range(n_ticks + 1):
                frac = i / n_ticks
                col = int(frac * (chart_cols - 1))
                secs = total_s * (1.0 - frac)
                label = fmt_duration(-secs, signed=True) if secs > 0 else "0s"
                start = max(0, min(col, chart_cols - len(label)))
                for j, ch in enumerate(label):
                    if start + j < chart_cols:
                        ruler[start + j] = ch
            x_line.append("".join(ruler), style="dim")
            yield x_line

    def __rich_measure__(self, console, options):
        from rich.measure import Measurement
        return Measurement(10, options.max_width)


def _line_plot_legend(gpu_ids: List[str], records: List[ComputedRecord]) -> Text:
    """Shared legend mapping GPU colour → GPU id/model, shown once above the charts."""
    legend = Text()
    model_of = {r.gpu_id: (r.model_name or "") for r in records}
    for idx, gid in enumerate(gpu_ids):
        if idx > 0:
            legend.append("   ")
        color = LINE_PLOT_COLORS[idx % len(LINE_PLOT_COLORS)]
        model = model_of.get(gid, "")
        legend.append("━━", style=color)
        legend.append(f" GPU{gid}", style=f"bold {color}")
        if model:
            legend.append(f" {model}", style="dim")
    return legend


def line_plot_view_panel(
    records: List[ComputedRecord],
    history: HistoryStore,
    pcie_bw_limits: Optional[Dict[str, float]] = None,
    pcie_info: str = "",
    poll: float = 1.0,
    power_limits: Optional[Dict[str, float]] = None,
    console_height: int = 50,
) -> Panel:
    """Build the full Plot View: shared legend + 3×3 grid of line charts."""
    gpu_ids = sorted({r.gpu_id for r in records}, key=lambda x: int(x) if x.isdigit() else x)

    # chart_rows = (console_height - 23) / 3 (the vertical-budget derivation:
    # summary(5)+footer(3)+outer borders(2)+legend(1)+blank(1)+2 spacers + 3*(rows+3)).
    chart_rows = max(3, (console_height - 23) // 3)
    pcie_vmax = 0.0
    if pcie_bw_limits:
        for gid in gpu_ids:
            if gid in pcie_bw_limits:
                pcie_vmax = max(pcie_vmax, pcie_bw_limits[gid])
    if pcie_vmax <= 0:
        for gid in gpu_ids:
            hist = list(history.get(gid, "pcie_rxtx"))
            if hist:
                pcie_vmax = max(pcie_vmax, max(hist))
        pcie_vmax = max(pcie_vmax, 1.0)

    def _chart_panel(title: str, hist_key: str, vmax: float = 100.0) -> Panel:
        gpu_data = [(gid, list(history.get(gid, hist_key))) for gid in gpu_ids]
        return Panel(
            LinePlotRenderable(gpu_data, chart_rows=chart_rows, vmax=vmax, poll=poll),
            title=title,
            border_style="blue",
            box=box.ROUNDED,
        )

    pcie_title = (f"PCIe RX+TX %  ({pcie_info})" if pcie_info
                  else f"PCIe RX+TX %  (max {fmt_bytes_per_s(pcie_vmax)})")

    panels = [
        _chart_panel("Real util %", "real_util"),
        _chart_panel("GPU util %", "gpu_util"),
        _chart_panel("GR active %", "gr_active"),
        _chart_panel("SM active %", "sm_active"),
        _chart_panel("SM occupancy %", "sm_occupancy"),
        _chart_panel("Tensor active %", "tensor"),
        _chart_panel("DRAM active %", "dram"),
        _chart_panel("Memcpy %", "memcpy"),
        _chart_panel(pcie_title, "pcie_rxtx", vmax=pcie_vmax),
    ]

    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    grid.add_column(width=1)
    grid.add_column(ratio=1)
    grid.add_column(width=1)
    grid.add_column(ratio=1)
    grid.add_row(panels[0], Text(""), panels[1], Text(""), panels[2])
    grid.add_row(Text(""), Text(""), Text(""), Text(""), Text(""))
    grid.add_row(panels[3], Text(""), panels[4], Text(""), panels[5])
    grid.add_row(Text(""), Text(""), Text(""), Text(""), Text(""))
    grid.add_row(panels[6], Text(""), panels[7], Text(""), panels[8])

    legend = _line_plot_legend(gpu_ids, records)
    return Panel(Group(legend, Text(""), grid), title="Plot View", border_style="cyan", box=box.ROUNDED)


# ══════════════════════════════════════════════════════════════════════════════
# Focus View
# ══════════════════════════════════════════════════════════════════════════════

FOCUS_INFO_FIELD_W = len("Status: ") + STATUS_DISPLAY_WIDTH
FOCUS_INFO_MAX_COLS = 4
FOCUS_METRIC_NOW_W = 12
FOCUS_SPLIT_LEFT_RATIO = 3
FOCUS_SPLIT_RIGHT_RATIO = 4
FOCUS_PANEL_MIN_WIDTH = 80
FOCUS_SPLIT_MIN_WIDTH = FOCUS_PANEL_MIN_WIDTH * (FOCUS_SPLIT_LEFT_RATIO + FOCUS_SPLIT_RIGHT_RATIO) // FOCUS_SPLIT_RIGHT_RATIO


def selected_gpu_panel(
    rec: ComputedRecord,
    history: HistoryStore,
    power_limit: Optional[float] = None,
    nvlink_limit: Optional[float] = None,
    console_width: int = 200,
) -> Panel:
    gpu = rec.gpu_id
    title = f"Focused GPU {gpu}"

    nvlink_gbps = _nvlink_gbps(rec)
    nvlink_max = nvlink_limit or 400.0

    metric_rows: list = [
        "Utilization",
        ("Real util", rec.real_util, "real_util", 100),
        ("GPU util", _pct_of(rec, "gpu_nvml_busy_time_fraction"), "gpu_util", 100),
        ("GR active", _pct_of(rec, "gpu_graphics_compute_engine_active_cycle_fraction"), "gr_active", 100),
        "Streaming Multiprocessors",
        ("SM active", _pct_of(rec, "gpu_streaming_multiprocessor_active_cycle_fraction"), "sm_active", 100),
        ("SM occupancy", _pct_of(rec, "gpu_streaming_multiprocessor_warp_occupancy_fraction"), "sm_occupancy", 100),
        "Compute Pipelines",
        ("Tensor", _pct_of(rec, "gpu_tensor_core_pipe_active_cycle_fraction"), "tensor", 100),
        ("FP16 pipe", _pct_of(rec, "gpu_cuda_core_floating_point_16bit_pipe_active_cycle_fraction"), "fp16", 100),
        ("FP32 pipe", _pct_of(rec, "gpu_cuda_core_floating_point_32bit_pipe_active_cycle_fraction"), "fp32", 100),
        ("FP64 pipe", _pct_of(rec, "gpu_cuda_core_floating_point_64bit_pipe_active_cycle_fraction"), "fp64", 100),
    ]

    tc_metrics = [
        ("TC FP16/BF16", "gpu_tensor_core_half_precision_mma_active_cycle_fraction", "tc_hmma"),
        ("TC INT8", "gpu_tensor_core_integer_mma_active_cycle_fraction", "tc_imma"),
        ("TC FP64", "gpu_tensor_core_double_precision_fma_active_cycle_fraction", "tc_dfma"),
        ("TC TF32/FP32", "gpu_tensor_core_double_mma_active_cycle_fraction", "tc_dmma"),
        ("TC FP8", "gpu_tensor_core_quarter_mma_active_cycle_fraction", "tc_qmma"),
    ]
    tc_rows = [(lbl, _pct_of(rec, field), hk, 100) for lbl, field, hk in tc_metrics
               if _pct_of(rec, field) is not None]
    if tc_rows:
        metric_rows.append("Tensor Core Detail")
        metric_rows.extend(tc_rows)

    metric_rows.extend([
        "Memory",
        ("DRAM", _pct_of(rec, "gpu_dram_controller_active_cycle_fraction"), "dram", 100),
        ("Memory used", _mem_used_pct(rec), "mem_used_pct", 100),
        "Interconnect & Power",
        ("NVLink Δ", nvlink_gbps, None, nvlink_max),
        ("Power", _raw(rec, "gpu_board_power_draw_watts"), "power", None),
        ("GPU temp", _raw(rec, "gpu_die_temperature_celsius"), "gpu_temp", None),
    ])

    table = Table(box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("Metric", style="bold")
    table.add_column("Now", justify="right", width=FOCUS_METRIC_NOW_W, no_wrap=True)
    table.add_column("Bar", justify="left")
    table.add_column("Trend", justify="left")

    for entry in metric_rows:
        if isinstance(entry, str):
            table.add_row(Text(""), Text(""), Text(""), Text(""))
            table.add_row(Text(f"── {entry}", style="dim italic"), Text(""), Text(""), Text(""))
            continue
        label, value, hist_key, vmax = entry
        if label == "Power":
            now = Text(fmt_watts(value), style=power_style(value))
            max_w = power_limit or 700.0
            bar = make_bar(min(100.0, (value or 0.0) / max_w * 100.0), 22, style_override=power_style(value))
            trend = Text(sparkline(history.get(gpu, hist_key or "power"), 28), style=power_style(value))
        elif label == "NVLink Δ":
            nv_style = nvlink_util_style(value, nvlink_max)
            if nvlink_limit is None and value is None:
                now = Text("N/A", style="dim")
                bar = Text("")
                trend = Text("")
            else:
                now = Text(fmt_gbps(value), style=nv_style)
                nv_cap = nvlink_max if nvlink_max and nvlink_max > 0 else 400.0
                pct_for_bar = 0.0 if value is None else min(100.0, value / nv_cap * 100.0)
                bar = make_bar(pct_for_bar, 22, style_override=nv_style)
                trend = Text(sparkline(history.get(gpu, "nvlink_gbps"), 28, vmax), style=nv_style)
        elif "temp" in label.lower():
            now = Text(fmt_temp(value), style=temp_style(value, rec.model_name))
            bar = make_bar(min(100.0, (value or 0.0)), 22, style_override=temp_style(value, rec.model_name))
            trend = Text(sparkline(history.get(gpu, hist_key or "gpu_temp"), 28), style=temp_style(value, rec.model_name))
        else:
            now = Text(fmt_pct(value), style=usage_style(value))
            bar = make_bar(value, 22)
            trend = Text(sparkline(history.get(gpu, hist_key or "real_util"), 28, vmax), style=usage_style(value))
        table.add_row(label, now, bar, trend)

    _replay = rec.pcie_replay_rate_per_second
    info_fields = [
        Text(f"Status: {rec.status_line}", style=rec.health_style),
        Text(f"PCIe RX: {fmt_bytes_per_s(_raw(rec, 'gpu_pcie_receive_throughput_bytes_per_second'))}", style="cyan"),
        Text(f"PCIe TX: {fmt_bytes_per_s(_raw(rec, 'gpu_pcie_transmit_throughput_bytes_per_second'))}", style="cyan"),
        Text(f"NVLink Δ: {'N/A' if nvlink_max is None and nvlink_gbps is None else fmt_gbps(nvlink_gbps)}",
             style=nvlink_util_style(nvlink_gbps, nvlink_max)),
        Text(f"Energy: {fmt_joules(_energy_j(rec))}", style="magenta"),
        Text(f"Power: {fmt_watts(_raw(rec, 'gpu_board_power_draw_watts'))}",
             style=power_style(_raw(rec, "gpu_board_power_draw_watts"))),
        Text(f"SM clk: {fmt_mhz(_raw(rec, 'gpu_streaming_multiprocessor_clock_frequency_megahertz'))}", style="green"),
        Text(f"MEM clk: {fmt_mhz(_raw(rec, 'gpu_memory_clock_frequency_megahertz'))}", style="green"),
        Text(f"Replay rate: {fmt_num(_replay, 2)}/s", style="yellow" if (_replay or 0) > 0 else "dim"),
    ]
    info_cols = max(1, min(FOCUS_INFO_MAX_COLS, console_width // FOCUS_INFO_FIELD_W))
    info = Table.grid(expand=True)
    for _ in range(info_cols):
        info.add_column(width=FOCUS_INFO_FIELD_W, no_wrap=True)
    for i in range(0, len(info_fields), info_cols):
        chunk = info_fields[i:i + info_cols]
        chunk += [Text("")] * (info_cols - len(chunk))
        info.add_row(*chunk)

    return Panel(Group(info, table), title=title, title_align="center", border_style="cyan", box=box.ROUNDED)


# ══════════════════════════════════════════════════════════════════════════════
# Job View
# ══════════════════════════════════════════════════════════════════════════════

def jobs_view_panel(
    records: List[ComputedRecord],
    gpu_processes: Dict[str, List[GpuProcess]],
) -> Panel:
    """Render a table of all running GPU compute processes with per-GPU metrics."""
    record_map = {r.gpu_id: r for r in records}

    jtable = Table(
        title="Running GPU Processes",
        box=box.SIMPLE_HEAVY,
        expand=True,
        show_lines=False,
        padding=(0, 1),
    )
    jtable.add_column("PID", justify="right", style="bold", no_wrap=True, width=8)
    jtable.add_column("User", justify="left", style="green", no_wrap=True, width=12)
    jtable.add_column("GPU", justify="right", style="cyan", no_wrap=True, width=4)
    jtable.add_column("GID", justify="left", style="yellow", no_wrap=True, width=14)
    jtable.add_column("*Status", justify="left", no_wrap=True, width=22)
    jtable.add_column("GPU Mem", justify="right", no_wrap=True, width=8)
    jtable.add_column("*GPU Util", justify="right", no_wrap=True, width=9)
    jtable.add_column("*Real Util", justify="right", no_wrap=True, width=10)
    jtable.add_column("*Tensor", justify="right", no_wrap=True, width=8)
    jtable.add_column("Command", justify="left", ratio=1, no_wrap=True)

    all_procs: List[Tuple[GpuProcess, ComputedRecord]] = []
    for gpu_id in sorted(gpu_processes.keys(), key=lambda x: int(x) if x.isdigit() else x):
        rec = record_map.get(gpu_id)
        if rec is None:
            continue
        for p in gpu_processes[gpu_id]:
            all_procs.append((p, rec))

    if not all_procs:
        return Panel(
            Text("No compute processes running on visible GPUs.", style="dim"),
            title="Job View",
            border_style="cyan",
            box=box.ROUNDED,
        )

    for proc, rec in all_procs:
        gpu_util = _pct_of(rec, "gpu_nvml_busy_time_fraction")
        tensor = _pct_of(rec, "gpu_tensor_core_pipe_active_cycle_fraction")
        if proc.gpu_mem_mib is not None:
            mem_text = f"{proc.gpu_mem_mib / 1024:.1f}G" if proc.gpu_mem_mib >= 1024 else f"{int(proc.gpu_mem_mib)}M"
        else:
            mem_text = "—"
        jtable.add_row(
            str(proc.pid),
            proc.user[:12],
            proc.gpu_id,
            proc.gid[:14],
            Text(rec.status_line, style=rec.health_style),
            Text(mem_text, style="magenta"),
            Text(fmt_pct(gpu_util), style=usage_style(gpu_util)),
            Text(fmt_pct(rec.real_util), style=usage_style(rec.real_util)),
            Text(fmt_pct(tensor), style=usage_style(tensor)),
            Text(proc.command, overflow="ellipsis", no_wrap=True, style="dim"),
        )

    footnote = Text("  * Per-GPU metric (shared across all processes on the same GPU)", style="dim italic")
    return Panel(Group(jtable, footnote), title="Job View", border_style="cyan", box=box.ROUNDED)
