"""Cross-cutting lifecycle — run the Read → Translate → Compute → Present pipeline.

This is the only module that owns the process lifecycle: it builds the backend,
translator, and per-GPU compute state; drives the live TUI / one-shot / CSV
export modes; and centralizes signal handling and teardown. Each sampling tick
flows Read (RawRecord) → Translate (CanonicalRecord) → Compute (ComputedRecord)
→ Present (render / CSV).
"""
from __future__ import annotations

import atexit
import csv
import signal
import sys
import threading
import time
from dataclasses import replace
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from rich.console import Console
from rich.live import Live

from .compute import ComputedRecord, compute_record
from .config import Config, validate_poll
from .identification import Identity, identify
from .present import (
    CommandController,
    HistoryStore,
    SummaryContext,
    cbreak_stdin,
    csv_header,
    csv_row,
    render_dashboard,
    resolve_columns,
    update_history,
    update_nvlink_history,
    UnknownExportColumns,
)
from .reader import BackendKind, DcgmStreamError, ReaderConfig, make_backend
from .reader.base import RawRecord
from .reader.dcgmi import (
    DCGM_DMON_NO_NVLINK_FIELDS,
    dcgm_dashboard_fields_for_nvlink_source,
    dcgm_field_ids,
    dcgm_metric_names,
    dcgm_nvlink_fields_for_source,
    probe_dcgm_nvlink_source,
)
from .selection import GPUSelector
from .system_queries import CpuSampler, query_gpu_processes, query_system_ram
from .translate import make_translator

RENDER_MIN_INTERVAL_SECONDS = 0.25
INPUT_TICK_SECONDS = 0.02
FIRST_SAMPLE_TIMEOUT_SECONDS = 5.0


def _version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version("kempnerpulse")
        except PackageNotFoundError:
            return "unknown"
    except Exception:
        return "unknown"


# ── tick sourcing ─────────────────────────────────────────────────────────────

def _group_by_timestamp(records: Iterable[RawRecord]) -> Iterator[List[RawRecord]]:
    """Group a per-record stream into ticks (records sharing one timestamp)."""
    bucket: List[RawRecord] = []
    cur_ts: Optional[float] = None
    for rec in records:
        if cur_ts is not None and rec.timestamp != cur_ts and bucket:
            yield bucket
            bucket = []
        cur_ts = rec.timestamp
        bucket.append(rec)
    if bucket:
        yield bucket


def _tick_iterator(backend) -> Iterator[List[RawRecord]]:
    """Yield one list of RawRecords per tick, for any backend."""
    if hasattr(backend, "stream_ticks"):
        return backend.stream_ticks()
    return _group_by_timestamp(backend.stream())


class ThreadedTickReader:
    """Runs a backend's tick stream in a daemon thread; publishes the latest tick.

    The live TUI needs to read input and render while waiting for the next
    sample, so the blocking tick stream runs off-thread and the newest tick is
    published under a condition variable with a monotonically increasing counter.
    """

    def __init__(self, backend) -> None:
        self._backend = backend
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._latest: Optional[List[RawRecord]] = None
        self._counter = 0
        self._error: Optional[BaseException] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._reader_thread = threading.Thread(target=self._run, name="tick-reader", daemon=True)
        self._reader_thread.start()
        if getattr(self._backend, "stderr", None) is not None:
            self._stderr_thread = threading.Thread(target=self._drain_stderr, name="tick-stderr", daemon=True)
            self._stderr_thread.start()

    def _run(self) -> None:
        try:
            for tick in _tick_iterator(self._backend):
                if self._stop.is_set():
                    break
                with self._cond:
                    self._latest = tick
                    self._counter += 1
                    self._cond.notify_all()
        except BaseException as exc:  # surface, never swallow
            with self._cond:
                if self._error is None:
                    self._error = exc
                self._cond.notify_all()
        finally:
            with self._cond:
                self._cond.notify_all()

    def _drain_stderr(self) -> None:
        stderr = getattr(self._backend, "stderr", None)
        if stderr is None:
            return
        try:
            for line in stderr:
                if self._stop.is_set():
                    break
                if line.strip():
                    sys.stderr.write(f"[dcgmi] {line}")
                    sys.stderr.flush()
        except Exception:
            pass

    def last_counter(self) -> int:
        with self._cond:
            return self._counter

    def get_latest(self) -> Tuple[Optional[List[RawRecord]], int]:
        with self._cond:
            if self._error is not None and self._latest is None:
                raise DcgmStreamError(str(self._error))
            return self._latest, self._counter

    def wait_first_sample(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._counter == 0 and not self._stop.is_set() and self._error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(timeout=remaining)
            if self._error is not None:
                raise DcgmStreamError(str(self._error))
            return self._counter > 0

    def wait_for_new(
        self,
        last_counter: int,
        timeout: Optional[float] = None,
    ) -> Tuple[Optional[List[RawRecord]], int]:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while self._counter == last_counter and not self._stop.is_set() and self._error is None:
                if deadline is None:
                    self._cond.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return self._latest, self._counter
                self._cond.wait(timeout=remaining)
            if self._error is not None:
                raise DcgmStreamError(str(self._error))
            return self._latest, self._counter

    def stop(self) -> None:
        self._stop.set()
        try:
            self._backend.close()
        except Exception:
            pass
        with self._cond:
            self._cond.notify_all()
        for t in (self._reader_thread, self._stderr_thread):
            if t is not None and t.is_alive():
                t.join(timeout=1.0)


# ── pipeline ──────────────────────────────────────────────────────────────────

class Pipeline:
    """Translate + Compute a tick of RawRecords into sorted ComputedRecords."""

    def __init__(self, config: Config, identity: Identity, allowed: Optional[set]) -> None:
        self._config = config
        self._identity = identity
        self._allowed = allowed
        self._translator = make_translator(
            config.backend,
            hostname=identity.hostname,
            gpu_uuid_by_index=identity.gpu_uuid_by_index,
            gpu_model_by_index=identity.gpu_model_by_index,
            slurm_metadata=identity.slurm_metadata,
        )
        self._prev_by_index: Dict[int, object] = {}

    def process(self, raw_tick: Iterable[RawRecord]) -> List[ComputedRecord]:
        out: List[ComputedRecord] = []
        for record in self._translator.translate_tick(raw_tick):
            if self._allowed is not None and str(record.entity_gpu_index) not in self._allowed:
                continue
            computed = compute_record(
                record,
                prev=self._prev_by_index.get(record.entity_gpu_index),
                weights=self._config.weights,
                preset_name=self._config.preset_name,
                model_name=self._identity.gpu_model_by_index.get(record.entity_gpu_index),
            )
            self._prev_by_index[record.entity_gpu_index] = record
            out.append(computed)
        out.sort(key=lambda c: c.gpu_index)
        return out

    def overlay_nvlink(
        self,
        records: List[ComputedRecord],
        raw_tick: Iterable[RawRecord],
    ) -> Tuple[List[ComputedRecord], List[ComputedRecord]]:
        """Overlay NVLink-only readings onto cached computed records.

        This intentionally does not update ``_prev_by_index``: fast NVLink ticks
        should not perturb full-sample differencing such as PCIe replay rate.
        """
        nvlink_by_index = {}
        for record in self._translator.translate_tick(raw_tick):
            if self._allowed is not None and str(record.entity_gpu_index) not in self._allowed:
                continue
            nvlink = record.gpu_nvlink_aggregate_throughput_bytes_per_second
            if nvlink is None:
                continue
            nvlink_by_index[record.entity_gpu_index] = record
        if not nvlink_by_index:
            return records, []

        updated: List[ComputedRecord] = []
        changed: List[ComputedRecord] = []
        for computed in records:
            overlay = nvlink_by_index.get(computed.gpu_index)
            if overlay is None:
                updated.append(computed)
                continue
            current_values = (
                computed.record.gpu_nvlink_aggregate_throughput_bytes_per_second,
                computed.record.gpu_nvlink_transmit_throughput_bytes_per_second,
                computed.record.gpu_nvlink_receive_throughput_bytes_per_second,
            )
            overlay_values = (
                overlay.gpu_nvlink_aggregate_throughput_bytes_per_second,
                overlay.gpu_nvlink_transmit_throughput_bytes_per_second,
                overlay.gpu_nvlink_receive_throughput_bytes_per_second,
            )
            if overlay_values == current_values:
                updated.append(computed)
                continue
            new_record = replace(
                computed.record,
                record_timestamp_monotonic_seconds=overlay.record_timestamp_monotonic_seconds,
                record_timestamp_wallclock_unix_seconds=overlay.record_timestamp_wallclock_unix_seconds,
                gpu_nvlink_aggregate_throughput_bytes_per_second=overlay.gpu_nvlink_aggregate_throughput_bytes_per_second,
                gpu_nvlink_transmit_throughput_bytes_per_second=overlay.gpu_nvlink_transmit_throughput_bytes_per_second,
                gpu_nvlink_receive_throughput_bytes_per_second=overlay.gpu_nvlink_receive_throughput_bytes_per_second,
            )
            new_computed = replace(computed, record=new_record)
            updated.append(new_computed)
            changed.append(new_computed)
        return updated, changed


def _make_reader_config(
    config: Config,
    identity: Identity,
    poll_seconds: float,
    dcgm_fields: Optional[Tuple[Tuple[int, str], ...]] = None,
) -> ReaderConfig:
    gpu_ids = (tuple(identity.dcgm_physical_gpu_ids)
               if identity.dcgm_physical_gpu_ids else None)
    return ReaderConfig(
        backend=config.backend,
        poll_seconds=poll_seconds,
        source=config.source,
        gpu_ids=gpu_ids,
        all_gpus=config.show_all,
        dcgm_field_ids=dcgm_field_ids(dcgm_fields) if dcgm_fields is not None else None,
        dcgm_metric_names=dcgm_metric_names(dcgm_fields) if dcgm_fields is not None else None,
    )


def _summary_context(
    config: Config,
    identity: Identity,
    poll_seconds: float,
    selection_desc: str,
    cpu_info,
    ram_info,
    gpu_processes,
) -> SummaryContext:
    return SummaryContext(
        source=(config.source if config.backend is BackendKind.PROMETHEUS else config.backend.value),
        poll=poll_seconds,
        selection_desc=selection_desc,
        weights=config.weights,
        app_version=_version(),
        cpu_info=cpu_info,
        ram_info=ram_info,
        power_limits=identity.power_limit_watts_by_id,
        nvlink_bw_limits=identity.nvlink_bw_limit_gbps_by_id,
        nvlink_fit=config.nvlink_fit,
        pcie_bw_limits=identity.pcie_bw_limit_bytes_per_second_by_id,
        pcie_info=identity.pcie_info,
        gpu_processes=gpu_processes,
    )


# ── run modes ─────────────────────────────────────────────────────────────────

def _run_export(
    config: Config,
    pipeline: Pipeline,
    backend,
    poll_seconds: float,
    fast_backend=None,
) -> int:
    try:
        columns = resolve_columns(config.export_spec or "default")
    except UnknownExportColumns as exc:
        print(f"kempnerpulse: {exc}", file=sys.stderr)
        return 1
    writer = csv.writer(sys.stdout)
    writer.writerow(csv_header(columns))
    sys.stdout.flush()

    def emit(records: List[ComputedRecord]) -> None:
        for rec in records:
            ts = rec.record.record_timestamp_wallclock_unix_seconds
            writer.writerow(csv_row(rec, ts, columns, config.nvlink_fit))
        sys.stdout.flush()

    try:
        if fast_backend is not None:
            base_reader = ThreadedTickReader(backend)
            fast_reader = ThreadedTickReader(fast_backend)
            base_reader.start()
            fast_reader.start()
            try:
                if not base_reader.wait_first_sample(FIRST_SAMPLE_TIMEOUT_SECONDS):
                    return 1
                fast_reader.wait_first_sample(FIRST_SAMPLE_TIMEOUT_SECONDS)
                records: List[ComputedRecord] = []
                base_seen = -1
                fast_seen = -1
                while True:
                    base_tick, base_counter = base_reader.get_latest()
                    if base_tick is not None and base_counter != base_seen:
                        records = pipeline.process(base_tick)
                        base_seen = base_counter
                    fast_tick, fast_seen_new = fast_reader.wait_for_new(
                        fast_seen, timeout=max(poll_seconds * 5.0, 2.0)
                    )
                    if fast_tick is None:
                        break
                    if fast_seen_new == fast_seen:
                        continue
                    fast_seen = fast_seen_new
                    records, _changed = pipeline.overlay_nvlink(records, fast_tick)
                    emit(records)
            finally:
                fast_reader.stop()
                base_reader.stop()
            return 0

        ticks = _tick_iterator(backend)
        for raw_tick in ticks:
            emit(pipeline.process(raw_tick))
            if config.once:
                break
    except KeyboardInterrupt:
        pass
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except BrokenPipeError:
            pass
    return 0


def _run_once(config: Config, pipeline: Pipeline, backend, console: Console,
              summary_ctx_builder) -> int:
    raw_tick = next(_tick_iterator(backend), None)
    records = pipeline.process(raw_tick) if raw_tick else []
    history = HistoryStore(maxlen=config.history_length)
    update_history(history, records)
    controller = CommandController(config.focus_gpu)
    console.print(render_dashboard(
        records, history,
        console_width=console.width, console_height=console.height,
        controller=controller, summary_context=summary_ctx_builder(records),
    ))
    return 0


def _run_live(config: Config, identity: Identity, pipeline: Pipeline, backend,
              console: Console, poll_seconds: float, selection_desc: str,
              fast_backend=None) -> int:
    reader = ThreadedTickReader(backend)
    fast_reader = ThreadedTickReader(fast_backend) if fast_backend is not None else None
    reader.start()
    if fast_reader is not None:
        fast_reader.start()
    atexit.register(reader.stop)
    if fast_reader is not None:
        atexit.register(fast_reader.stop)
    cpu_sampler = CpuSampler()
    history = HistoryStore(maxlen=config.history_length)
    controller = CommandController(config.focus_gpu)

    try:
        if not reader.wait_first_sample(FIRST_SAMPLE_TIMEOUT_SECONDS):
            console.print("[bold red]Timed out waiting for the first sample.[/]")
            if fast_reader is not None:
                fast_reader.stop()
            reader.stop()
            return 1
        if fast_reader is not None:
            fast_reader.wait_first_sample(FIRST_SAMPLE_TIMEOUT_SECONDS)
    except DcgmStreamError as exc:
        console.print(f"[bold red]Reader error: {exc}[/]")
        if fast_reader is not None:
            fast_reader.stop()
        reader.stop()
        return 1

    records: List[ComputedRecord] = []
    visible_ids: set = set()
    cpu_info = (None, None, None, None)
    ram_info = (None, None)
    gpu_processes: Dict[str, list] = {}
    seen_counter = 0
    seen_fast_counter = 0
    last_ram_query = 0.0

    def fetch() -> None:
        nonlocal records, visible_ids, cpu_info, ram_info, gpu_processes
        nonlocal seen_counter, seen_fast_counter, last_ram_query
        changed = False
        tick, counter = reader.get_latest()
        if tick is not None and counter != seen_counter:
            seen_counter = counter
            records = pipeline.process(tick)
            update_history(history, records)
            changed = True

        if fast_reader is not None:
            fast_tick, fast_counter = fast_reader.get_latest()
            if fast_tick is not None and fast_counter != seen_fast_counter:
                seen_fast_counter = fast_counter
                records, changed_records = pipeline.overlay_nvlink(records, fast_tick)
                if changed_records:
                    update_nvlink_history(history, changed_records)
                    changed = True

        if not changed:
            return
        visible_ids = {r.gpu_id for r in records}
        cpu_info = cpu_sampler.sample()
        now = time.monotonic()
        if now - last_ram_query >= 1.0:
            ram_info = query_system_ram()
            last_ram_query = now
        gpu_processes = query_gpu_processes(identity.bus_id_to_index) if controller.jobs_mode else {}
        if controller.focus_gpu is not None and controller.focus_gpu not in visible_ids and visible_ids:
            controller.focus_gpu = sorted(visible_ids, key=lambda x: int(x) if x.isdigit() else x)[0]

    def layout():
        ctx = _summary_context(config, identity, poll_seconds, selection_desc,
                               cpu_info, ram_info, gpu_processes)
        return render_dashboard(
            records, history,
            console_width=console.width, console_height=console.height,
            controller=controller, summary_context=ctx,
        )

    fetch()
    last_state: Optional[tuple] = None
    cached = None
    last_render = 0.0
    with cbreak_stdin(enabled=True):
        with Live(layout(), console=console, screen=True, auto_refresh=False) as live:
            while True:
                controller.handle_input(visible_ids)
                if controller.should_exit:
                    break
                try:
                    fetch()
                except DcgmStreamError as exc:
                    console.print(f"[bold red]Reader stopped: {exc}[/]")
                    break
                now = time.monotonic()
                state = (controller.command_mode, controller.buffer, controller.last_message,
                         controller.focus_gpu, controller.line_mode, controller.jobs_mode,
                         controller.fleet_scroll_offset, seen_counter)
                if state != last_state:
                    cached = None
                    last_state = state
                if cached is None or now - last_render >= RENDER_MIN_INTERVAL_SECONDS:
                    cached = layout()
                    live.update(cached, refresh=True)
                    last_render = now
                time.sleep(INPUT_TICK_SECONDS)
    if fast_reader is not None:
        fast_reader.stop()
    reader.stop()
    return 0


def run(config: Config) -> int:
    """Validate, identify hardware, build the pipeline, and dispatch a run mode."""
    console = Console()

    poll = validate_poll(config)
    if poll.error:
        console.print(f"[bold red]Error:[/] {poll.error}")
        return 1
    if poll.note:
        print(f"kempnerpulse: {poll.note}", file=sys.stderr)
    poll_seconds = poll.effective_poll_seconds

    identity = identify(config)
    accessible = identity.accessible_gpu_ids
    if (config.backend is not BackendKind.REPLAY
            and accessible is not None and not accessible):
        console.print("[bold red]KempnerPulse requires a node with accessible NVIDIA GPUs.[/]")
        return 1

    selector = GPUSelector(
        explicit=list(config.gpu_ids) if config.gpu_ids else None,
        disable_auto=config.show_all,
        accessible=accessible,
    )
    allowed, _reason, _src = selector.resolve()
    selection_desc = ("all" if allowed is None
                      else ",".join(sorted(allowed, key=lambda x: int(x) if x.isdigit() else x)) or "none")

    pipeline = Pipeline(config, identity, allowed)

    dcgm_fields: Optional[Tuple[Tuple[int, str], ...]] = None
    fast_dcgm_fields: Optional[Tuple[Tuple[int, str], ...]] = None
    base_poll_seconds = poll_seconds
    if config.backend is BackendKind.DCGMI:
        if allowed:
            probe_gpu_ids = sorted(allowed, key=lambda x: int(x) if x.isdigit() else x)
        else:
            probe_gpu_ids = list(identity.dcgm_physical_gpu_ids) if identity.dcgm_physical_gpu_ids else None
        nvlink_source = probe_dcgm_nvlink_source(probe_gpu_ids)
        if config.sp_fast and not config.once:
            base_poll_seconds = 1.0
            dcgm_fields = DCGM_DMON_NO_NVLINK_FIELDS
            fast_dcgm_fields = dcgm_nvlink_fields_for_source(nvlink_source)
        else:
            dcgm_fields = dcgm_dashboard_fields_for_nvlink_source(nvlink_source)

    backend_config = _make_reader_config(config, identity, base_poll_seconds, dcgm_fields)
    backend = make_backend(backend_config)
    try:
        backend.open(backend_config)
    except DcgmStreamError as exc:
        console.print(f"[bold red]Backend failed to start: {exc}[/]")
        return 1
    atexit.register(backend.close)
    fast_backend = None
    if fast_dcgm_fields:
        fast_config = _make_reader_config(config, identity, poll_seconds, fast_dcgm_fields)
        fast_backend = make_backend(fast_config)
        try:
            fast_backend.open(fast_config)
        except DcgmStreamError as exc:
            console.print(f"[bold red]Fast NVLink backend failed to start: {exc}[/]")
            try:
                backend.close()
            except Exception:
                pass
            return 1
        atexit.register(fast_backend.close)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    def summary_ctx_builder(records):
        return _summary_context(config, identity, poll_seconds, selection_desc,
                                CpuSampler().sample(), query_system_ram(), {})

    try:
        if config.export_spec is not None:
            return _run_export(config, pipeline, backend, poll_seconds, fast_backend=fast_backend)
        if config.once:
            return _run_once(config, pipeline, backend, console, summary_ctx_builder)
        return _run_live(
            config, identity, pipeline, backend, console, poll_seconds, selection_desc,
            fast_backend=fast_backend,
        )
    finally:
        if fast_backend is not None:
            try:
                fast_backend.close()
            except Exception:
                pass
        try:
            backend.close()
        except Exception:
            pass
