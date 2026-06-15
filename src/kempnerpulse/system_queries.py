"""Cross-cutting tier — per-sample host/process queries (best-effort).

These run on every sampling tick to enrich the display with host CPU/RAM load
and the GPU compute processes. Unlike the reader layer (which raises typed
errors so the lifecycle can surface remediation), everything here is
*best-effort*: any missing command, permission error, timeout, or non-zero exit
degrades to an empty / ``None`` result and is never raised. A monitoring tool
must keep rendering GPU metrics even when host introspection is unavailable.

Runtime dependencies are the standard library only.
"""
from __future__ import annotations

import grp
import os
import pwd
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# nvidia-smi calls are bounded so a wedged driver can't stall the render loop.
_NVIDIA_SMI_TIMEOUT = 5.0
# nproc is queried once and cached; a short timeout is plenty.
_NPROC_TIMEOUT = 2.0
# Per-core utilization above this fraction counts a core as "busy".
_BUSY_CORE_THRESHOLD = 0.05


@dataclass
class GpuProcess:
    """A single compute process running on a GPU."""
    pid: int
    user: str
    gid: str
    gpu_id: str
    gpu_mem_mib: Optional[float]
    command: str


class CpuSampler:
    """Stateful sampler for host CPU load from ``/proc/stat``.

    Each :meth:`sample` reads ``/proc/stat`` and diffs against the previous
    snapshot to compute utilization, so the first call (no prior snapshot)
    returns ``None`` for the percentage and busy-core count. The logical-CPU
    count and the (Slurm-aware) physical core count are cached on the instance;
    no module- or function-level state is used.
    """

    def __init__(self) -> None:
        # (aggregate_idle, aggregate_total, [(core_idle, core_total), ...]).
        self._prev: Optional[Tuple[int, int, List[Tuple[int, int]]]] = None
        self._nproc_cache: Optional[int] = None

    def _num_cores(self, fallback: int) -> int:
        """Total cores via ``nproc --all`` (Slurm-aware), cached; best-effort."""
        if self._nproc_cache is not None:
            return self._nproc_cache
        try:
            result = subprocess.run(
                ["nproc", "--all"],
                capture_output=True,
                text=True,
                timeout=_NPROC_TIMEOUT,
            )
            num_cores = int(result.stdout.strip()) if result.returncode == 0 else fallback
        except (FileNotFoundError, OSError, ValueError, subprocess.TimeoutExpired):
            num_cores = fallback
        self._nproc_cache = num_cores
        return num_cores

    def sample(
        self,
    ) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[int]]:
        """Return ``(num_threads, num_cores, cpu_percent, busy_cores)``.

        ``num_threads`` is ``os.cpu_count()`` (logical CPUs); ``num_cores`` is
        the ``nproc --all`` total; ``cpu_percent`` is overall utilization over
        the interval since the previous call; ``busy_cores`` counts cores above
        the busy threshold. ``cpu_percent`` and ``busy_cores`` are ``None`` on
        the first call and whenever ``/proc/stat`` is unreadable.
        """
        try:
            aggregate_idle = aggregate_total = 0
            per_core: List[Tuple[int, int]] = []  # (idle, total) per core
            with open("/proc/stat", "r") as f:
                for line in f:
                    if line.startswith("cpu "):
                        parts = line.split()
                        aggregate_idle = int(parts[4]) + int(parts[5])
                        aggregate_total = sum(int(x) for x in parts[1:])
                    elif line.startswith("cpu"):
                        parts = line.split()
                        c_idle = int(parts[4]) + int(parts[5])
                        c_total = sum(int(x) for x in parts[1:])
                        per_core.append((c_idle, c_total))
                    else:
                        break  # past all cpu lines
            if aggregate_total == 0:
                return None, None, None, None
            num_threads = os.cpu_count() or len(per_core)
            num_cores = self._num_cores(num_threads)

            prev = self._prev
            self._prev = (aggregate_idle, aggregate_total, per_core)
            if prev is None:
                return num_threads, num_cores, None, None

            prev_idle, prev_total, prev_per_core = prev
            d_total = aggregate_total - prev_total
            d_idle = aggregate_idle - prev_idle
            if d_total <= 0:
                return num_threads, num_cores, None, None
            pct = 100.0 * (1.0 - d_idle / d_total)

            busy = 0
            for i, (c_idle, c_total) in enumerate(per_core):
                if i < len(prev_per_core):
                    pc_idle, pc_total = prev_per_core[i]
                    dc_total = c_total - pc_total
                    dc_idle = c_idle - pc_idle
                    if dc_total > 0 and (1.0 - dc_idle / dc_total) > _BUSY_CORE_THRESHOLD:
                        busy += 1
            return num_threads, num_cores, max(0.0, min(100.0, pct)), busy
        except (OSError, ValueError):
            return None, None, None, None


def query_system_ram() -> Tuple[Optional[float], Optional[float]]:
    """Return ``(used_gb, total_gb)`` from ``/proc/meminfo`` (best-effort).

    "Used" is ``MemTotal - MemAvailable`` (falling back to ``MemFree`` when
    ``MemAvailable`` is absent). Returns ``(None, None)`` if ``/proc/meminfo``
    cannot be read or parsed.
    """
    try:
        info: Dict[str, int] = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    info[key] = int(parts[1])  # kB
        total_kb = info.get("MemTotal", 0)
        avail_kb = info.get("MemAvailable", info.get("MemFree", 0))
        total_gb = total_kb / (1024 * 1024)
        used_gb = (total_kb - avail_kb) / (1024 * 1024)
        return used_gb, total_gb
    except (OSError, ValueError):
        return None, None


def query_gpu_processes(
    bus_id_to_index: Dict[str, str],
) -> Dict[str, List[GpuProcess]]:
    """List GPU compute processes via nvidia-smi, keyed by GPU index (best-effort).

    Uses ``--query-compute-apps`` (instant, no sampling delay) and requires a
    ``bus_id_to_index`` mapping (uppercased PCI bus id -> GPU index) to attribute
    each process to a GPU. For each PID, the owning user/group are resolved from
    ``/proc/<pid>`` ownership and the full command line from
    ``/proc/<pid>/cmdline``. Returns ``{gpu_index: [GpuProcess, ...]}``; an empty
    dict if the mapping is empty or nvidia-smi is unavailable.
    """
    if not bus_id_to_index:
        return {}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_bus_id,pid,used_gpu_memory,process_name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT,
        )
        if result.returncode != 0:
            return {}
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return {}

    procs: Dict[str, List[GpuProcess]] = defaultdict(list)
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",", 3)]
        if len(parts) < 3:
            continue
        bus_id = parts[0].upper()
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        try:
            mem_mib: Optional[float] = float(parts[2])
        except ValueError:
            mem_mib = None
        proc_name = parts[3].strip() if len(parts) > 3 else "?"

        gpu_id = bus_id_to_index.get(bus_id, "?")

        # Resolve user / group from /proc ownership.
        user, group = "?", "?"
        try:
            st = os.stat(f"/proc/{pid}")
            try:
                user = pwd.getpwuid(st.st_uid).pw_name
            except KeyError:
                user = str(st.st_uid)
            try:
                group = grp.getgrgid(st.st_gid).gr_name
            except KeyError:
                group = str(st.st_gid)
        except (FileNotFoundError, PermissionError, OSError):
            pass

        # Full command line, falling back to the process name.
        cmd = proc_name
        try:
            with open(f"/proc/{pid}/cmdline", "r") as f:
                raw = f.read().replace("\x00", " ").strip()
                if raw:
                    cmd = raw
        except (FileNotFoundError, PermissionError, OSError):
            pass

        procs[gpu_id].append(
            GpuProcess(
                pid=pid,
                user=user,
                gid=group,
                gpu_id=gpu_id,
                gpu_mem_mib=mem_mib,
                command=cmd,
            )
        )
    return dict(procs)
