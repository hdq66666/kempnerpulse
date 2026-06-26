"""Display formatting, styling, and canonical→display unit conversion.

The Present layer renders ``ComputedRecord`` objects whose metric values are canonical:
ratios in ``[0, 1]``, throughputs in bytes/second, energy in joules, and so on.
Display, however, wants percents, GB/s, and watts. The converters here are the
single place that bridge the two: a presenter never multiplies by 100 inline,
it calls :func:`fraction_to_percent`. ``None`` (an unavailable reading) always
maps to ``None`` and renders as ``--`` — never coerced to ``0``.

The ``fmt_*`` formatters and ``*_style`` colour helpers are ported verbatim from
the single-file implementation so the rendered output is byte-for-byte familiar.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

from rich.text import Text

# ── Spark / bar glyphs ────────────────────────────────────────────────────────
SPARK_BLOCKS = "▁▂▃▄▅▆▇█"
BAR_FILLED_GLYPH = "▓"
BAR_EMPTY_GLYPH = "░"
BAR_EMPTY_STYLE = "bright_black"

# Per-GPU temperature thresholds (°C), keyed by a substring of the model name.
GPU_TEMP_THRESHOLDS: Dict[str, Dict[str, int]] = {
    "A100": {"normal": 85, "warning": 93, "critical": 95},
    "H100": {"normal": 85, "warning": 95, "critical": 105},
    "H200": {"normal": 80, "warning": 95, "critical": 105},
    "RTX 6000": {"normal": 85, "warning": 92, "critical": 105},
}
_DEFAULT_TEMP_THRESHOLDS = {"normal": 85, "warning": 93, "critical": 105}


def get_temp_thresholds(model_name: Optional[str] = None) -> Dict[str, int]:
    """Return the warning/critical temperature thresholds for a GPU model."""
    if model_name:
        upper = model_name.upper()
        for key in GPU_TEMP_THRESHOLDS:
            if key.upper() in upper:
                return GPU_TEMP_THRESHOLDS[key]
    return _DEFAULT_TEMP_THRESHOLDS


# ── Canonical → display unit converters ───────────────────────────────────────
#
# Canonical values are fractions/SI; display wants percent/GB·s⁻¹. ``None`` (the
# source did not provide a reading) passes through as ``None`` so the formatters
# render it as "--" rather than a misleading 0.

def fraction_to_percent(value: Optional[float]) -> Optional[float]:
    """A ``[0, 1]`` fraction → a ``[0, 100]`` percent, clamped defensively."""
    if value is None:
        return None
    return max(0.0, min(100.0, float(value) * 100.0))


def bytes_per_second_to_gigabytes(value: Optional[float]) -> Optional[float]:
    """A bytes/second throughput → GB/s (decimal, ÷1e9).

    Applied to NVLink and PCIe alike. NVLink GB/s computed this way equals the
    legacy ``MB/s ÷ 1e3`` figure, since the canonical NVLink rate is the legacy
    MB/s gauge × 1e6.
    """
    if value is None:
        return None
    return float(value) / 1e9


def millijoules_to_joules(value: Optional[float]) -> Optional[float]:
    """Cumulative millijoules → joules. (Canonical energy is already joules.)"""
    if value is None:
        return None
    return float(value) / 1000.0


# ── Scalar formatters ─────────────────────────────────────────────────────────

def fmt_pct(v: Optional[float], digits: int = 0) -> str:
    if v is None or math.isnan(v):
        return "--"
    return f"{v:.{digits}f}%"


def fmt_num(v: Optional[float], digits: int = 1) -> str:
    if v is None or math.isnan(v):
        return "--"
    return f"{v:.{digits}f}"


def fmt_temp(v: Optional[float]) -> str:
    if v is None:
        return "--"
    return f"{v:.0f}°C"


def fmt_watts(v: Optional[float]) -> str:
    if v is None:
        return "--"
    return f"{v:.0f}W"


def fmt_mhz(v: Optional[float]) -> str:
    if v is None:
        return "--"
    return f"{v:.0f}MHz"


def fmt_mib(v: Optional[float]) -> str:
    if v is None:
        return "--"
    if v >= 1024:
        return f"{v / 1024:.1f}GiB"
    return f"{v:.0f}MiB"


def fmt_bytes_per_s(v: Optional[float]) -> str:
    if v is None:
        return "--"
    units = ["B/s", "KiB/s", "MiB/s", "GiB/s", "TiB/s"]
    x = float(v)
    idx = 0
    while abs(x) >= 1024 and idx < len(units) - 1:
        x /= 1024
        idx += 1
    return f"{x:.1f}{units[idx]}"


def fmt_gbps(v: Optional[float], digits: int = 2) -> str:
    if v is None or math.isnan(v):
        return "--"
    return f"{v:.{digits}f}GB/s"


def apply_nvlink_fit(
    gbps: Optional[float],
    fit: Optional[Tuple[float, float]],
) -> Optional[float]:
    if gbps is None:
        return None
    if fit is None:
        return gbps
    scale, offset = fit
    return gbps * scale + offset


def fmt_nvlink_gbps(
    gbps: Optional[float],
    fit: Optional[Tuple[float, float]] = None,
) -> str:
    if gbps is None or math.isnan(gbps):
        return "--"
    est = apply_nvlink_fit(gbps, fit)
    if fit is None or est is None or math.isnan(est):
        return fmt_gbps(gbps)
    return f"{gbps:.1f} {est:.1f}GB/s↑"


def fmt_joules(v: Optional[float]) -> str:
    if v is None:
        return "--"
    if v > 1000:
        return f"{v / 1000:.1f}kJ"
    return f"{v:.0f}J"


def fmt_duration(seconds: float, *, signed: bool = False) -> str:
    """Compact duration label that picks units to keep the number readable.

    Used by the footer's ``poll=`` indicator (always positive) and by the
    line-plot x-axis tick labels (which pass ``signed=True`` for offsets like
    ``-50ms`` relative to "now").

    Examples::

        fmt_duration(1.0)    -> "1s"
        fmt_duration(0.5)    -> "500ms"
        fmt_duration(0.05)   -> "50ms"
        fmt_duration(0.001)  -> "1ms"
        fmt_duration(0.0)    -> "0s"
        fmt_duration(-0.05, signed=True) -> "-50ms"
    """
    if seconds == 0:
        return "0s"
    sign = "-" if (signed and seconds < 0) else ""
    val = abs(seconds)
    if val >= 1.0:
        if val >= 10:
            return f"{sign}{val:.0f}s"
        return f"{sign}{val:.1f}s".replace(".0s", "s")
    ms = val * 1000.0
    if ms >= 1.0:
        if ms >= 10:
            return f"{sign}{ms:.0f}ms"
        return f"{sign}{ms:.1f}ms".replace(".0ms", "ms")
    # Sub-millisecond: round up to 1ms rather than printing "0ms".
    return f"{sign}1ms"


# ── Colour-style selectors (input is in display units: percent / W / °C / GB·s⁻¹) ─

def usage_style(p: Optional[float]) -> str:
    if p is None:
        return "dim"
    if p >= 90:
        return "bold red"
    if p >= 75:
        return "bold yellow"
    if p >= 40:
        return "bold green"
    if p >= 10:
        return "cyan"
    return "dim"


def temp_style(t: Optional[float], model_name: Optional[str] = None) -> str:
    if t is None:
        return "dim"
    th = get_temp_thresholds(model_name)
    if t >= th["critical"]:
        return "bold red"
    if t >= th["warning"]:
        return "bold yellow"
    if t >= th["normal"]:
        return "green"
    return "cyan"


def power_style(w: Optional[float]) -> str:
    if w is None:
        return "dim"
    if w >= 600:
        return "bold red"
    if w >= 450:
        return "bold yellow"
    if w >= 200:
        return "green"
    return "cyan"


def io_rate_style_gbps(v: Optional[float]) -> str:
    if v is None:
        return "dim"
    if v >= 100:
        return "bold red"
    if v >= 50:
        return "bold yellow"
    if v >= 10:
        return "bold green"
    if v > 0:
        return "cyan"
    return "dim"


def nvlink_util_style(gbps: Optional[float], limit_gbps: Optional[float]) -> str:
    """Colour an NVLink value by absolute GB/s."""
    return io_rate_style_gbps(gbps)


# ── Sparkline & bar ───────────────────────────────────────────────────────────

def sparkline(values: Iterable[float], width: int = 24, vmax: Optional[float] = None) -> str:
    """A unicode-block sparkline of ``values`` rendered into ``width`` columns."""
    seq = list(values)
    if not seq:
        return " " * width
    if len(seq) > width:
        out: List[float] = []
        for i in range(width):
            start = int(i * len(seq) / width)
            end = max(start + 1, int((i + 1) * len(seq) / width))
            chunk = seq[start:end]
            out.append(sum(chunk) / len(chunk))
        seq = out
    if len(seq) < width:
        seq = [seq[0]] * (width - len(seq)) + seq
    local_max = max(seq) if seq else 1.0
    if vmax is None:
        vmax = local_max if local_max > 0 else 1.0
    vmax = max(vmax, 1e-9)
    chars = []
    for v in seq:
        idx = int(round((len(SPARK_BLOCKS) - 1) * max(0.0, min(1.0, v / vmax))))
        chars.append(SPARK_BLOCKS[idx])
    return "".join(chars)


def make_bar(pct: Optional[float], width: int = 18, style_override: Optional[str] = None) -> Text:
    """A fixed-``width`` horizontal bar for a ``[0, 100]`` percent value."""
    pct = 0.0 if pct is None else max(0.0, min(100.0, pct))
    filled = int(round(width * pct / 100.0))
    style = style_override if style_override is not None else usage_style(pct)
    t = Text()
    t.append(BAR_FILLED_GLYPH * filled, style=style)
    t.append(BAR_EMPTY_GLYPH * (width - filled), style=BAR_EMPTY_STYLE)
    return t
