"""KempnerPulse Layer 4 — Present.

The terminal UI and CSV export. This layer consumes :class:`ComputedRecord` objects
(canonical metric values plus the derived Real Utilization / classification /
health signals) and renders them — it converts canonical fractions/SI to display
units (percent, GB/s, …) itself and never reaches back into source vocabulary.

Public API
----------

Rendering / interaction:

* ``render_dashboard(records, history, *, console_width, console_height,
  controller, summary_context) -> Layout`` — assemble one dashboard frame.
* ``SummaryContext`` — the per-frame context (system stats, source/poll labels,
  weight preset, hardware limits, job listing) the dashboard needs beyond the
  records. The lifecycle layer builds this.
* ``CommandController`` / ``cbreak_stdin(enabled)`` — keyboard command state and
  the raw-terminal context manager.
* ``HistoryStore`` / ``update_history(history, records)`` — the rolling
  per-GPU display-unit time-series store and its record→series adapter.
* ``GpuProcess`` — a Job-View process row (built by the lifecycle layer).

CSV export:

* ``resolve_columns(spec) -> [(name, extractor), ...]`` — resolve ``"default"``
  / ``"all"`` / a comma-list to an ordered column set.
* ``csv_header(columns) -> [str]`` and
  ``csv_row(record, timestamp, columns, nvlink_fit=None) -> [str]`` — emit the header and one
  row, with units/precision matching the documented export schema.
* ``UnknownExportColumns`` — raised by ``resolve_columns`` on a bad spec.
"""
from .controller import SCROLL_PAGE, CommandController, cbreak_stdin
from .csv_writer import (
    CSV_ALL_COLUMN_NAMES,
    CSV_COLUMNS,
    CSV_DEFAULT_COLUMN_NAMES,
    UnknownExportColumns,
    csv_header,
    csv_row,
    resolve_columns,
)
from .history import DEFAULT_HISTORY_MAXLEN, HistoryStore, update_history, update_nvlink_history
from .tui import DEFAULT_WEIGHTS, SummaryContext, footer_panel, render_dashboard
from .widgets import GpuProcess

__all__ = [
    # Rendering / interaction
    "render_dashboard",
    "SummaryContext",
    "CommandController",
    "cbreak_stdin",
    "HistoryStore",
    "update_history",
    "update_nvlink_history",
    "GpuProcess",
    # CSV export
    "resolve_columns",
    "csv_header",
    "csv_row",
    "UnknownExportColumns",
    # Constants / extras (stable, useful to callers)
    "SCROLL_PAGE",
    "DEFAULT_WEIGHTS",
    "DEFAULT_HISTORY_MAXLEN",
    "CSV_COLUMNS",
    "CSV_ALL_COLUMN_NAMES",
    "CSV_DEFAULT_COLUMN_NAMES",
    "footer_panel",
]
