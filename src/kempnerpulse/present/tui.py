"""Dashboard assembly: the footer, the render-context, and ``render_dashboard``.

``render_dashboard`` is the single entry point the lifecycle layer calls each
frame. It takes the per-frame data (the computed records, the rolling history,
the input controller, and the console dimensions) plus a
:class:`SummaryContext` carrying everything else the summary/footer and the
view panels need (system stats, the source/poll labels, the active weight
preset, hardware limits, and any job-process listing). The min-size gate,
focus-mode split, and view dispatch reproduce the single-file behaviour exactly.
"""
from __future__ import annotations

import re
import socket
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from rich import box
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..compute.result import ComputedRecord
from .controller import CommandController
from .format import fmt_duration
from .history import HistoryStore
from .widgets import (
    CARD_FULL_WIDTH,
    FLEET_PANEL_BORDER,
    FOCUS_SPLIT_LEFT_RATIO,
    FOCUS_SPLIT_MIN_WIDTH,
    FOCUS_SPLIT_RIGHT_RATIO,
    FOOTER_PANEL_ROWS,
    GpuProcess,
    MIN_DASH_HEIGHT,
    MIN_DASH_WIDTH,
    SUMMARY_PANEL_ROWS,
    build_fleet_panel,
    fleet_panel,
    jobs_view_panel,
    line_plot_view_panel,
    selected_gpu_panel,
    summary_panel,
    workflow_label,
)

# The default Real Utilization preset weights (AI/ML), used when a context omits them.
DEFAULT_WEIGHTS: Tuple[float, float, float, float] = (0.35, 0.35, 0.20, 0.10)


@dataclass
class SummaryContext:
    """Everything the summary bar, footer, and view panels need besides the
    per-frame records/history/controller/dimensions.

    The lifecycle layer builds one of these each frame (or reuses a mostly-static
    one and refreshes the volatile system fields). All fields have sensible
    defaults so a minimal caller can pass ``SummaryContext()``.
    """
    # Footer / summary labels.
    source: str = ""                 # data source URL or backend label
    poll: float = 1.0                # poll interval in seconds
    selection_desc: str = "all"      # human-readable GPU selection description
    weights: Tuple[float, float, float, float] = DEFAULT_WEIGHTS
    app_version: str = ""            # shown in the summary panel title

    # System stats (volatile; refreshed per frame by the caller).
    #   cpu_info = (num_threads, num_cores, cpu_percent, busy_cores)
    cpu_info: Tuple[Optional[int], Optional[int], Optional[float], Optional[int]] = (None, None, None, None)
    #   ram_info = (used_gb, total_gb)
    ram_info: Tuple[Optional[float], Optional[float]] = (None, None)

    # Hardware limits, keyed by GPU id (for bar scaling / colouring).
    power_limits: Dict[str, float] = field(default_factory=dict)
    nvlink_bw_limits: Dict[str, float] = field(default_factory=dict)
    pcie_bw_limits: Dict[str, float] = field(default_factory=dict)
    pcie_info: str = ""              # PCIe-bandwidth label for the plot title
    nvlink_fit: Optional[Tuple[float, float]] = None

    # Per-GPU process listing for the Job View.
    gpu_processes: Dict[str, List[GpuProcess]] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════════════

# Right-side status fields drop one by one as the footer narrows (host → src →
# poll → date), keeping the left side (Visible / workflow / Commands) readable.
FOOTER_LEFT_MIN_WIDTH = 30   # columns reserved for the left status before dropping right fields
FOOTER_CHROME_WIDTH = 4      # panel border (2) + spacer column (2)


def footer_panel(
    selection_desc: str,
    controller: CommandController,
    source: str = "",
    poll: float = 1.0,
    weights: Tuple[float, float, float, float] = DEFAULT_WEIGHTS,
    console_width: int = 200,
) -> Panel:
    selection_text = selection_desc
    msg = controller.last_message or controller.hint()
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")  # fixed 19 chars
    hostname = socket.gethostname().split(".")[0]
    display_source = re.sub(r"^https?://", "", source)
    wf_label = workflow_label(weights)

    right_parts = [
        f"host={hostname}",
        f"src={display_source}",
        f"poll={fmt_duration(poll)}",
        now_str,
    ]
    right_budget = max(0, console_width - FOOTER_LEFT_MIN_WIDTH - FOOTER_CHROME_WIDTH)
    while right_parts and len("  ".join(right_parts)) > right_budget:
        right_parts.pop(0)            # drop host first, then src, then poll, then date
    right_plain = "  ".join(right_parts)
    right = Text(right_plain, style="dim", no_wrap=True)
    right_w = len(right_plain)

    left = Text.assemble(
        ("Visible ", "bold cyan"), (selection_text, "dim"),
        ("   ", ""), (wf_label, "bold magenta"),
        ("   Commands ", "bold"), (msg, "green" if controller.command_mode else "dim"),
    )
    left.no_wrap = True
    left.overflow = "ellipsis"
    line = Table.grid(expand=True)
    line.add_column(ratio=1, no_wrap=True)
    line.add_column(width=2)
    line.add_column(width=right_w, justify="right", no_wrap=True)
    line.add_row(left, Text(""), right)
    return Panel(line, border_style="dim", box=box.ROUNDED)


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard
# ══════════════════════════════════════════════════════════════════════════════

def _too_small_gate(console_width: int, console_height: int) -> Layout:
    """The 'widen me' placeholder shown below the per-card minimum size."""
    w_ok = console_width >= MIN_DASH_WIDTH
    h_ok = console_height >= MIN_DASH_HEIGHT
    bar = "bold yellow"
    w_style = "green" if w_ok else "bold red"
    h_style = "green" if h_ok else "bold red"
    title = "Terminal Too Small"
    w_txt = "Width Is OK" if w_ok else "Width Too Narrow"
    h_txt = "Height Is OK" if h_ok else "Height Too Short"
    inner = len(title) + 6                        # width between the side '*' borders
    placeholder = Text(justify="center", no_wrap=True)
    placeholder.append("\n" * max(0, (console_height - 6) // 2))   # rough vertical centering
    if console_width >= inner + 2:
        placeholder.append(f"*** {title} ***\n", bar)
        placeholder.append("*", bar)
        placeholder.append(w_txt.center(inner), w_style)
        placeholder.append("*\n", bar)
        placeholder.append("*", bar)
        placeholder.append(h_txt.center(inner), h_style)
        placeholder.append("*\n", bar)
        placeholder.append("*" * (inner + 2), bar)
    else:
        placeholder.append(f"{title}\n", bar)
        placeholder.append(f"{w_txt}\n", w_style)
        placeholder.append(h_txt, h_style)
    gate = Layout()
    gate.update(placeholder)
    return gate


def render_dashboard(
    records: List[ComputedRecord],
    history: HistoryStore,
    *,
    console_width: int,
    console_height: int,
    controller: CommandController,
    summary_context: SummaryContext,
) -> Layout:
    """Assemble the full dashboard Layout for one frame.

    Returns a Rich renderable (a :class:`~rich.layout.Layout`). Below the
    per-card minimum size it returns a 'widen me' gate instead of squeezed cards.
    View selection follows ``controller`` (plot / job / focus / fleet).
    """
    ctx = summary_context

    # Hard minimum: below this the cards would be squeezed — show a placeholder.
    if console_width < MIN_DASH_WIDTH or console_height < MIN_DASH_HEIGHT:
        return _too_small_gate(console_width, console_height)

    layout = Layout()
    layout.split_column(
        Layout(name="summary", size=SUMMARY_PANEL_ROWS),
        Layout(name="middle", ratio=1),
        Layout(name="footer", size=FOOTER_PANEL_ROWS),
    )
    layout["summary"].update(summary_panel(
        records,
        app_version=ctx.app_version,
        cpu_info=ctx.cpu_info,
        ram_info=ctx.ram_info,
        console_width=console_width,
    ))
    layout["footer"].update(footer_panel(
        ctx.selection_desc, controller, ctx.source, ctx.poll, ctx.weights,
        console_width=console_width,
    ))

    if not records:
        layout["middle"].update(Panel(
            f"No GPU data matched the current selection ({ctx.selection_desc}).\n"
            "Try --show-all or override with --gpus 0,1",
            border_style="red",
            box=box.ROUNDED,
        ))
        return layout

    if controller.line_mode:
        layout["middle"].update(line_plot_view_panel(
            records, history, ctx.pcie_bw_limits, ctx.pcie_info, poll=ctx.poll,
            power_limits=ctx.power_limits, console_height=console_height,
        ))
    elif controller.jobs_mode:
        layout["middle"].update(jobs_view_panel(records, ctx.gpu_processes))
    elif controller.focus_gpu is not None:
        selected = next((r for r in records if r.gpu_id == controller.focus_gpu), records[0])
        gpu_power_limit = ctx.power_limits.get(selected.gpu_id)
        gpu_nvlink_limit = ctx.nvlink_bw_limits.get(selected.gpu_id)
        if console_width >= FOCUS_SPLIT_MIN_WIDTH:
            layout["middle"].split_row(
                Layout(name="left", ratio=FOCUS_SPLIT_LEFT_RATIO),
                Layout(name="right", ratio=FOCUS_SPLIT_RIGHT_RATIO),
            )
            fleet_h = console_height - SUMMARY_PANEL_ROWS - FOOTER_PANEL_ROWS - FLEET_PANEL_BORDER
            total_ratio = FOCUS_SPLIT_LEFT_RATIO + FOCUS_SPLIT_RIGHT_RATIO
            left_w = console_width * FOCUS_SPLIT_LEFT_RATIO // total_ratio
            panel_w = console_width * FOCUS_SPLIT_RIGHT_RATIO // total_ratio
            layout["middle"]["left"].update(build_fleet_panel(
                records, history, left_w, fleet_h, ctx.power_limits, ctx.nvlink_bw_limits,
                nvlink_fit=ctx.nvlink_fit,
                controller=controller,
                force_single_column=True,
            ))
            layout["middle"]["right"].update(selected_gpu_panel(
                selected, history, gpu_power_limit, gpu_nvlink_limit,
                console_width=panel_w, nvlink_fit=ctx.nvlink_fit,
            ))
        else:
            layout["middle"].update(selected_gpu_panel(
                selected, history, gpu_power_limit, gpu_nvlink_limit,
                console_width=console_width, nvlink_fit=ctx.nvlink_fit,
            ))
    else:
        cards_per_row = 1 if len(records) <= 1 else 2
        detail_columns = 2 if console_width // cards_per_row >= CARD_FULL_WIDTH else 1
        layout["middle"].update(fleet_panel(
            records, history,
            cards_per_row=cards_per_row,
            detail_columns=detail_columns,
            power_limits=ctx.power_limits,
            nvlink_bw_limits=ctx.nvlink_bw_limits,
            nvlink_fit=ctx.nvlink_fit,
            controller=controller,
        ))
    return layout
