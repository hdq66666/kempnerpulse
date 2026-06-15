"""Cross-cutting tier — parsed configuration.

Turns command-line arguments into a single frozen ``Config`` value that the
lifecycle, reader, translate, compute, and present layers all read from. This
module owns *parsing and validation only*: it builds the argument parser,
resolves the weight preset, applies the backend-aware ``--poll`` default, and
reports poll-validation problems as data. It never prints, never exits, and
never installs signal handlers — those are the lifecycle's responsibility.

Runtime dependencies are the standard library only.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

from .compute.presets import (
    DEFAULT_PRESET_NAME,
    PRESETS,
    Weights,
    preset_name_for_weights,
)
from .reader.base import BackendKind

# DCGM profiling counters refresh at ~10 Hz through the shared hardware-counter
# multiplexer; below this floor a direct ``dcgmi`` stream returns mostly-blank
# profiling rows, so a sub-floor ``--poll`` is clamped up to it. Kept in sync
# with the reader's ``DCGM_STREAM_MIN_INTERVAL_MS``.
DCGM_STREAM_MIN_INTERVAL_MS = 100

# Backend-aware ``--poll`` defaults (seconds), applied when ``--poll`` is unset.
DEFAULT_POLL_DCGM_SECONDS = 0.1
DEFAULT_POLL_PROMETHEUS_SECONDS = 1.0

# Minimum samples retained for sparkline history regardless of ``--history``.
MIN_HISTORY_LENGTH = 10

_DEFAULT_SOURCE = "http://localhost:9400/metrics"

# CLI backend token -> reader BackendKind.
_BACKEND_BY_NAME = {
    "dcgm": BackendKind.DCGMI,
    "prometheus": BackendKind.PROMETHEUS,
    "replay": BackendKind.REPLAY,   # replay a saved capture (--source FILE); no GPU needed
}


@dataclass(frozen=True)
class Config:
    """Fully-resolved run configuration, frozen for the process lifetime.

    Field semantics:
      * ``gpu_ids`` is the *explicit* ``--gpus`` selection (already a tuple of
        string ids), or ``None`` when the flag was not supplied. Environment /
        accessibility resolution is the selection layer's job, not config's.
      * ``weights`` is normalized to sum to 1; ``preset_name`` is the matching
        preset name (``"ai"`` / ``"hpc"`` / ``"mem"``) or ``"custom"``.
      * ``export_spec`` is ``None`` (no export), ``"default"`` (``--export`` with
        no argument), ``"all"``, or a comma-separated column list.
      * ``focus_gpu`` is the id to start focused on, or ``None``.
    """
    backend: BackendKind
    poll_seconds: float
    source: str
    gpu_ids: Optional[Tuple[str, ...]]
    show_all: bool
    weights: Weights
    preset_name: str
    export_spec: Optional[str]
    once: bool
    focus_gpu: Optional[str]
    history_length: int


def _pkg_version() -> str:
    """Best-effort installed version of ``kempnerpulse`` for ``--version``."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - importlib.metadata is stdlib >=3.8
        return "unknown"
    try:
        return version("kempnerpulse")
    except PackageNotFoundError:
        return "unknown"
    except Exception:
        return "unknown"


def parse_weights(raw: str) -> Weights:
    """Validate a ``--weights`` string into a normalized 4-tuple.

    Requires exactly four comma-separated numeric values in
    ``SM,TENSOR,DRAM,GR`` order summing to a positive value; the tuple is
    normalized to sum to 1. Raises ``argparse.ArgumentTypeError`` on any
    malformed input so argparse reports it cleanly.
    """
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--weights requires four comma-separated values: SM,TENSOR,DRAM,GR"
        )
    try:
        vals = tuple(float(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--weights values must be numeric") from exc
    total = sum(vals)
    if total <= 0:
        raise argparse.ArgumentTypeError("--weights must sum to a positive value")
    if abs(total - 1.0) > 1e-6:
        vals = tuple(v / total for v in vals)
    return vals  # type: ignore[return-value]


_HELP_EPILOG = """
Application:
  KempnerPulse is a terminal dashboard for NVIDIA DCGM hardware-counter metrics.
  It is designed to help distinguish idle GPUs, real compute, memory pressure,
  transfer/copy pressure, and hardware-health issues at a glance.

Real util equation:
  RealUtil = clamp(0, 100,
              Wsm * SM_ACTIVE
            + Wtensor * TENSOR_ACTIVE
            + Wdram * DRAM_ACTIVE
            + Wgr * GR_ENGINE_ACTIVE)

  GR_ENGINE_ACTIVE is a profiling-level hardware counter (DCGM field 1001).
  If it is unavailable the dashboard falls back to GPU_UTIL (field 203).

Weight presets (convenience flags):
  --ai-weights             AI / LLM training and inference  (0.35,0.35,0.20,0.10) [default]
  --hpc-weights            General mixed CUDA / HPC         (0.45,0.15,0.25,0.15)
  --mem-weights            Memory-bound / bandwidth-heavy   (0.35,0.10,0.40,0.15)

  Or supply custom weights with --weights W_SM,W_TENSOR,W_DRAM,W_GR (normalized to sum to 1).

GPU visibility selection:
  The dashboard uses the first matching source in this order:
    1. --gpus
    2. CUDA_VISIBLE_DEVICES
    3. NVIDIA_VISIBLE_DEVICES
    4. SLURM_STEP_GPUS
    5. SLURM_JOB_GPUS
  If none are usable, all GPUs accessible to the process are shown. Use
  --show-all to ignore the environment and show every accessible GPU, or --gpus
  to force an explicit list. All selections are filtered against GPUs accessible
  to the current process (as reported by nvidia-smi), respecting cgroup and
  container restrictions.

Backend selection:
  --backend dcgm           (default) Query dcgmi dmon directly for true per-sample
                           resolution (down to a 100ms floor). Best for single-node
                           workload profiling. Requires the DCGM host engine.
  --backend prometheus     Read metrics from the dcgm-exporter Prometheus HTTP
                           endpoint. Profiling fields update at the exporter's
                           configured interval (typically ~30s). Best for
                           fleet-level monitoring; requires --poll >= 1.0.

Examples:
  kempnerpulse
  kempnerpulse --poll 1.0
  kempnerpulse --focus-gpu 0
  kempnerpulse --hpc-weights
  kempnerpulse --weights 0.40,0.30,0.20,0.10
  kempnerpulse --gpus 2,3
  kempnerpulse --show-all
  kempnerpulse --source http://otherhost:9400/metrics
  kempnerpulse --backend dcgm --poll 0.5
  kempnerpulse --backend dcgm --export all --poll 0.1
"""


def build_parser() -> argparse.ArgumentParser:
    """Construct the KempnerPulse command-line parser.

    Returned standalone so callers (and tests) can introspect or reuse it
    without triggering a parse. Defaults mirror the legacy CLI surface; the
    weight presets and custom-weight default are sourced from the compute layer's
    ``PRESETS`` so the two never drift.
    """
    default_weights = PRESETS[DEFAULT_PRESET_NAME]
    parser = argparse.ArgumentParser(
        prog="kempnerpulse",
        description=(
            "KempnerPulse: CLI dashboard for NVIDIA DCGM hardware-counter metrics "
            "with SLURM/CUDA GPU visibility awareness"
        ),
        epilog=_HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {_pkg_version()}"
    )
    parser.add_argument(
        "--source",
        default=_DEFAULT_SOURCE,
        help=(
            "Path to a dcgm-exporter text file or an http(s) /metrics endpoint. "
            f"Default: {_DEFAULT_SOURCE}"
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["prometheus", "dcgm", "replay"],
        default="dcgm",
        help=(
            "Metric collection backend. 'prometheus' reads from the dcgm-exporter "
            "HTTP endpoint (~30s resolution for profiling fields). 'dcgm' queries "
            "dcgmi dmon directly for true high-resolution sampling (down to 100ms). "
            "Default: dcgm"
        ),
    )
    parser.add_argument(
        "--poll",
        type=float,
        default=None,
        help=(
            "Sampling/refresh interval in seconds. With --backend dcgm, drives a "
            "persistent dcgmi stream honored down to a 100ms floor (DCGM profiling "
            "counters refresh at ~10Hz internally; smaller values would yield blank "
            "profiling rows). With --backend prometheus, must be >= 1.0 "
            "(dcgm-exporter scrapes profiling fields at ~30s, so sub-second values "
            "just duplicate samples). Default: 0.1 (dcgm) / 1.0 (prometheus)."
        ),
    )
    parser.add_argument(
        "--history",
        type=int,
        default=120,
        help="Number of samples kept for sparkline history. Default: 120",
    )
    parser.add_argument(
        "--focus-gpu",
        default=None,
        help="Start in focused view for one GPU id, for example 0",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Render one snapshot and exit instead of running live",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help=(
            "Explicit GPU ids or ranges to show, for example 0,1 or 0-3. "
            "Overrides SLURM/CUDA visibility env vars."
        ),
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Ignore SLURM/CUDA visibility env vars and show every accessible GPU",
    )
    parser.add_argument(
        "--weights",
        type=parse_weights,
        default=default_weights,
        help=(
            "Comma-separated real-util weights in SM,TENSOR,DRAM,GR order. Values "
            "are normalized to sum to 1. Example: --weights 0.40,0.30,0.20,0.10"
        ),
    )
    parser.add_argument(
        "--ai-weights",
        dest="weights",
        action="store_const",
        const=PRESETS["ai"],
        help="Use AI/LLM training weight preset (0.35,0.35,0.20,0.10) — the default",
    )
    parser.add_argument(
        "--hpc-weights",
        dest="weights",
        action="store_const",
        const=PRESETS["hpc"],
        help="Use general HPC weight preset (0.45,0.15,0.25,0.15)",
    )
    parser.add_argument(
        "--mem-weights",
        dest="weights",
        action="store_const",
        const=PRESETS["mem"],
        help="Use memory-bound weight preset (0.35,0.10,0.40,0.15)",
    )
    parser.add_argument(
        "--export",
        nargs="?",
        const="default",
        default=None,
        metavar="COLS",
        help=(
            "Output CSV to stdout. Use --export for default columns, --export all "
            "for every column, or --export col1,col2,... for a custom set."
        ),
    )
    return parser


def build_config(argv: Optional[list[str]] = None) -> Config:
    """Parse ``argv`` (or ``sys.argv``) into a frozen ``Config``.

    Resolves the backend enum, applies the backend-aware ``--poll`` default when
    unset, names the weight preset, and floors the history length. Poll *values*
    are not validated here — call :func:`validate_poll` on the returned config.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    backend = _BACKEND_BY_NAME[args.backend]

    poll_seconds = args.poll
    if poll_seconds is None:
        poll_seconds = (
            DEFAULT_POLL_DCGM_SECONDS
            if backend is BackendKind.DCGMI
            else DEFAULT_POLL_PROMETHEUS_SECONDS
        )

    weights: Weights = tuple(args.weights)  # type: ignore[assignment]
    preset_name = preset_name_for_weights(weights)

    gpu_ids: Optional[Tuple[str, ...]] = None
    if args.gpus is not None:
        gpu_ids = tuple(args.gpus.split(",")) if isinstance(args.gpus, str) else None

    return Config(
        backend=backend,
        poll_seconds=poll_seconds,
        source=args.source,
        gpu_ids=gpu_ids,
        show_all=args.show_all,
        weights=weights,
        preset_name=preset_name,
        export_spec=args.export,
        once=args.once,
        focus_gpu=args.focus_gpu,
        history_length=max(MIN_HISTORY_LENGTH, args.history),
    )


@dataclass(frozen=True)
class PollValidation:
    """Outcome of poll validation, returned as data for the lifecycle to act on.

    ``error`` is a user-facing message when the configured poll is invalid (the
    lifecycle should print it and exit non-zero), else ``None``. When the dcgm
    backend is asked for a sub-floor interval, ``clamped`` is ``True`` and
    ``note`` carries an advisory the lifecycle may print to stderr;
    ``effective_poll_seconds`` is the interval that will actually be used.
    """
    error: Optional[str]
    clamped: bool
    note: Optional[str]
    effective_poll_seconds: float


def validate_poll(config: Config) -> PollValidation:
    """Validate ``config.poll_seconds`` against the backend, returning data only.

    Rules (ported from the legacy CLI):
      * ``poll <= 0`` is an error for any backend.
      * prometheus requires ``poll >= 1.0`` (the exporter's scrape interval is
        the true ceiling; sub-second values just duplicate samples).
      * dcgm allows a sub-100ms request but clamps it up to the 100ms profiling
        floor, reported via ``clamped`` / ``note`` rather than printed here.

    No printing or exiting happens in this function; the lifecycle owns that.
    """
    poll = config.poll_seconds

    if poll <= 0:
        return PollValidation(
            error=(
                f"--poll must be a positive number of seconds (got {poll}). "
                "Use e.g. --poll 0.1 for 100ms or --poll 2 for 2s."
            ),
            clamped=False,
            note=None,
            effective_poll_seconds=poll,
        )

    if config.backend is BackendKind.PROMETHEUS:
        if poll < 1.0:
            return PollValidation(
                error=(
                    f"--poll {poll}s is below the Prometheus backend's effective "
                    "sampling rate. dcgm-exporter scrapes DCGM at ~30s for "
                    "profiling fields, so sub-second --poll values produce "
                    "duplicate samples with no new data. Use --backend dcgm for "
                    "true high-resolution sampling, or raise --poll to >= 1.0."
                ),
                clamped=False,
                note=None,
                effective_poll_seconds=poll,
            )
        return PollValidation(
            error=None, clamped=False, note=None, effective_poll_seconds=poll
        )

    # dcgm backend: clamp sub-floor requests up to the profiling floor.
    requested_ms = int(round(poll * 1000))
    if requested_ms < DCGM_STREAM_MIN_INTERVAL_MS:
        effective = DCGM_STREAM_MIN_INTERVAL_MS / 1000.0
        note = (
            "DCGM profiling counters (SM/Tensor/DRAM Active, etc.) refresh at "
            f"~10Hz internally; --poll {poll}s would yield mostly-blank profiling "
            f"rows. Clamping to {DCGM_STREAM_MIN_INTERVAL_MS}ms."
        )
        return PollValidation(
            error=None, clamped=True, note=note, effective_poll_seconds=effective
        )

    return PollValidation(
        error=None, clamped=False, note=None, effective_poll_seconds=poll
    )
