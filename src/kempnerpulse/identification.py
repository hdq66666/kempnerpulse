"""Cross-cutting tier — startup-once device identity and capability.

Resolves, exactly once at startup, the static facts about the host and its GPUs
that the rest of the pipeline treats as constants for the process lifetime:
hostname, per-GPU UUID / model, power and bandwidth limits, the PCI bus-id ->
index map, the set of accessible GPUs, the dcgmi physical-id mapping, and the
SLURM/MPI job metadata.

Every external-command query here is *best-effort*: a missing command,
permission error, timeout, or non-zero exit degrades to an empty result and is
never raised (in contrast with the reader layer, which raises typed errors). The
top-level :func:`identify` likewise never raises — it degrades to empty maps so
the lifecycle always receives a usable ``Identity``.

The ``slurm_metadata`` produced by :func:`gather_slurm_metadata` uses the
canonical-record metadata keys, so it can be passed straight into the translate
layer's ``SourceContext``.

Runtime dependencies are the standard library only.
"""
from __future__ import annotations

import os
import re
import socket
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .config import Config
from .reader.base import BackendKind
from .reader.dcgmi import resolve_dcgm_gpu_ids
from .reader.preflight import probe_dcgmi

# nvidia-smi queries are bounded so a wedged driver can't stall startup.
_NVIDIA_SMI_TIMEOUT = 5.0
# dcgmi discovery can be slower than nvidia-smi; give it more room.
_DCGMI_DISCOVERY_TIMEOUT = 10.0

# PCIe per-lane rates in bytes/sec for each generation (encoding overhead folded
# in): Gen1/2 use 8b/10b, Gen3-5 use 128b/130b, Gen6 uses 242b/256b.
_PCIE_GEN_LANE_RATE: Dict[int, float] = {
    1: 250e6,       # 2.5 GT/s * 8b/10b
    2: 500e6,       # 5   GT/s * 8b/10b
    3: 984.6e6,     # 8   GT/s * 128b/130b
    4: 1969.2e6,    # 16  GT/s * 128b/130b
    5: 3938.5e6,    # 32  GT/s * 128b/130b
    6: 7563.0e6,    # 64  GT/s * 242b/256b
}

# SLURM/MPI environment variable -> canonical metadata key, split by value type
# so ids stay strings and counts/indices are coerced to int (None if malformed).
_SLURM_STR_ENV: Tuple[Tuple[str, str], ...] = (
    ("SLURM_JOB_ID", "record_slurm_job_id"),
    ("SLURM_STEP_ID", "record_slurm_step_id"),
    ("SLURM_ARRAY_JOB_ID", "record_slurm_array_job_id"),
    ("SLURM_ARRAY_TASK_ID", "record_slurm_array_task_id"),
)
_SLURM_INT_ENV: Tuple[Tuple[str, str], ...] = (
    ("SLURM_RESTART_COUNT", "record_slurm_restart_count"),
    ("SLURM_NODEID", "record_node_index_in_job"),
)
# First set wins for the MPI rank (Open MPI, then PMIx, then SLURM).
_MPI_RANK_ENV: Tuple[str, ...] = (
    "OMPI_COMM_WORLD_RANK",
    "PMIX_RANK",
    "SLURM_PROCID",
)


@dataclass
class Identity:
    """Static host/GPU identity and capability, resolved once at startup.

    Per-GPU maps are provided both string-keyed (matching nvidia-smi / dcgmi id
    strings, for downstream lookups) and int-keyed (convenient for the translate
    layer's ``SourceContext``). For the dcgm backend, the string-keyed
    capability maps are re-keyed onto the physical dcgmi indices so per-GPU
    lookups line up with the ids the reader emits.
    """
    hostname: str
    gpu_uuid_by_index: Dict[int, str] = field(default_factory=dict)
    gpu_model_by_index: Dict[int, str] = field(default_factory=dict)
    power_limit_watts_by_index: Dict[int, float] = field(default_factory=dict)
    pcie_bw_limit_bytes_per_second_by_index: Dict[int, float] = field(default_factory=dict)
    pcie_info: str = ""
    nvlink_bw_limit_gbps_by_index: Dict[int, float] = field(default_factory=dict)
    bus_id_to_index: Dict[str, str] = field(default_factory=dict)
    accessible_gpu_ids: Optional[Set[str]] = None
    dcgm_physical_gpu_ids: Optional[List[str]] = None
    dcgm_phys_to_local: Dict[str, str] = field(default_factory=dict)
    slurm_metadata: Dict[str, object] = field(default_factory=dict)

    # String-keyed views (nvidia-smi / dcgmi id strings) for downstream lookups.
    gpu_uuid_by_id: Dict[str, str] = field(default_factory=dict)
    gpu_model_by_id: Dict[str, str] = field(default_factory=dict)
    power_limit_watts_by_id: Dict[str, float] = field(default_factory=dict)
    pcie_bw_limit_bytes_per_second_by_id: Dict[str, float] = field(default_factory=dict)
    nvlink_bw_limit_gbps_by_id: Dict[str, float] = field(default_factory=dict)


# ── nvidia-smi hardware queries (best-effort, string-keyed by GPU index) ──────

def query_gpu_uuids() -> Dict[str, str]:
    """Map GPU index -> UUID via nvidia-smi (best-effort)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=_NVIDIA_SMI_TIMEOUT,
        )
        if result.returncode != 0:
            return {}
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return {}
    uuids: Dict[str, str] = {}
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",", 1)]
        if len(parts) == 2 and parts[0]:
            uuids[parts[0]] = parts[1]
    return uuids


def query_gpu_models() -> Dict[str, str]:
    """Map GPU index -> model name via ``nvidia-smi -L`` (best-effort)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True, text=True, timeout=_NVIDIA_SMI_TIMEOUT,
        )
        if result.returncode != 0:
            return {}
        models: Dict[str, str] = {}
        for line in result.stdout.strip().splitlines():
            m = re.match(r"^GPU\s+(\d+):\s+(.+?)\s*\(UUID:", line)
            if m:
                models[m.group(1)] = m.group(2)
        return models
    except (OSError, subprocess.TimeoutExpired):
        return {}


def query_power_limits() -> Dict[str, float]:
    """Map GPU index -> max power limit in watts via nvidia-smi (best-effort)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,power.max_limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=_NVIDIA_SMI_TIMEOUT,
        )
        if result.returncode != 0:
            return {}
        limits: Dict[str, float] = {}
        for line in result.stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) == 2:
                gpu_id = parts[0].strip()
                try:
                    limits[gpu_id] = float(parts[1].strip())
                except ValueError:
                    continue
        return limits
    except (OSError, subprocess.TimeoutExpired):
        return {}


def query_pcie_bandwidth() -> Tuple[Dict[str, float], str]:
    """Map GPU index -> max bidirectional PCIe bandwidth (bytes/s) (best-effort).

    Returns ``(limits, info_string)`` where ``info_string`` summarizes the
    fastest link, e.g. ``"Gen5 x16  63.0 GB/s bidir"``. Bandwidth is
    ``lane_rate(gen) * width * 2`` (the ``*2`` accounts for full-duplex).
    """
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,pcie.link.gen.max,pcie.link.width.max",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=_NVIDIA_SMI_TIMEOUT,
        )
        if result.returncode != 0:
            return {}, ""
        limits: Dict[str, float] = {}
        gens: List[int] = []
        widths: List[int] = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            gpu_id = parts[0]
            try:
                gen = int(parts[1])
                width = int(parts[2])
            except ValueError:
                continue
            lane_rate = _PCIE_GEN_LANE_RATE.get(gen, 0.0)
            limits[gpu_id] = lane_rate * width * 2  # bidirectional
            gens.append(gen)
            widths.append(width)
        if gens:
            gen_v = max(gens)
            width_v = max(widths)
            max_bw = max(limits.values()) if limits else 0.0
            info = f"Gen{gen_v} x{width_v}  {max_bw / 1e9:.1f} GB/s bidir"
        else:
            info = ""
        return limits, info
    except (OSError, subprocess.TimeoutExpired):
        return {}, ""


def query_nvlink_bandwidth() -> Dict[str, float]:
    """Map GPU index -> aggregate NVLink bandwidth in GB/s (best-effort).

    Parses ``nvidia-smi nvlink -s`` and sums per-link speeds, doubling the total
    (each link is full-duplex and ``DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL`` counts
    TX+RX). Returns an empty dict if NVLink is unavailable.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "nvlink", "-s"],
            capture_output=True, text=True, timeout=_NVIDIA_SMI_TIMEOUT,
        )
        if result.returncode != 0:
            return {}
        limits: Dict[str, float] = {}
        current_gpu: Optional[str] = None
        total: float = 0.0
        for line in result.stdout.splitlines():
            m = re.match(r"^GPU\s+(\d+):", line)
            if m:
                if current_gpu is not None:
                    limits[current_gpu] = total * 2
                current_gpu = m.group(1)
                total = 0.0
                continue
            m2 = re.match(r"^\s+Link\s+\d+:\s+([\d.]+)\s+GB/s", line)
            if m2 and current_gpu is not None:
                total += float(m2.group(1))
        if current_gpu is not None:
            limits[current_gpu] = total * 2
        return limits
    except (OSError, subprocess.TimeoutExpired):
        return {}


def query_bus_id_mapping() -> Dict[str, str]:
    """Map uppercased PCI bus id -> GPU index via nvidia-smi (best-effort)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=_NVIDIA_SMI_TIMEOUT,
        )
        if result.returncode != 0:
            return {}
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return {}
    mapping: Dict[str, str] = {}
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            mapping[parts[1].upper()] = parts[0]
    return mapping


def query_accessible_gpus() -> Optional[Set[str]]:
    """Set of GPU index strings the current process can access (best-effort).

    nvidia-smi respects cgroup/container restrictions, so this reflects the GPUs
    actually reachable by the process. Returns ``None`` if nvidia-smi is
    unavailable (the caller then applies no accessibility filtering).
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=_NVIDIA_SMI_TIMEOUT,
        )
        if result.returncode != 0:
            return None
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    ids: Set[str] = set()
    for line in result.stdout.strip().splitlines():
        token = line.strip()
        if token.isdigit():
            ids.add(token)
    return ids if ids else None


# ── SLURM / MPI metadata ──────────────────────────────────────────────────────

def gather_slurm_metadata() -> Dict[str, object]:
    """Collect SLURM/MPI job metadata from the environment (best-effort).

    Keys are the canonical-record metadata names (``record_slurm_job_id``, …) so
    the result drops straight into the translate layer's ``SourceContext``. Job /
    step / array ids are kept as strings; restart count, node index, and MPI rank
    are coerced to int. Any variable that is unset or malformed is omitted.
    """
    meta: Dict[str, object] = {}

    for env_name, key in _SLURM_STR_ENV:
        value = os.environ.get(env_name)
        if value is not None and value != "":
            meta[key] = value

    for env_name, key in _SLURM_INT_ENV:
        value = os.environ.get(env_name)
        if value is None or value == "":
            continue
        try:
            meta[key] = int(value)
        except ValueError:
            continue

    for env_name in _MPI_RANK_ENV:
        value = os.environ.get(env_name)
        if value is None or value == "":
            continue
        try:
            meta["record_mpi_rank"] = int(value)
        except ValueError:
            continue
        break  # first valid value wins

    return meta


def _hostname() -> str:
    """Resolve the node name from SLURM, falling back to the socket hostname."""
    return os.environ.get("SLURMD_NODENAME") or socket.gethostname()


def _rekey_to_int(mapping: Dict[str, float]) -> Dict[int, float]:
    """Re-key a numeric string-keyed GPU map to int keys, dropping non-numeric ids."""
    out: Dict[int, float] = {}
    for key, value in mapping.items():
        if key.isdigit():
            out[int(key)] = value
    return out


def _rekey_str_to_int(mapping: Dict[str, str]) -> Dict[int, str]:
    """Re-key a string-valued string-keyed GPU map to int keys."""
    out: Dict[int, str] = {}
    for key, value in mapping.items():
        if key.isdigit():
            out[int(key)] = value
    return out


def _apply_phys_rekey(
    mapping: Dict[str, float], local_to_phys: Dict[str, str]
) -> Dict[str, float]:
    """Re-key a capability map from local cgroup ids to physical dcgmi ids.

    Inside a SLURM cgroup nvidia-smi reports local indices (e.g. ``"0"``) while
    dcgmi reports physical indices (e.g. ``"1"``); this maps the former onto the
    latter so per-GPU lookups match the ids the reader emits. Ids without a
    mapping pass through unchanged.
    """
    return {local_to_phys.get(k, k): v for k, v in mapping.items()}


def _apply_phys_rekey_str(
    mapping: Dict[str, str], local_to_phys: Dict[str, str]
) -> Dict[str, str]:
    """String-valued variant of :func:`_apply_phys_rekey`."""
    return {local_to_phys.get(k, k): v for k, v in mapping.items()}


def identify(config: Config) -> Identity:
    """Resolve the full startup ``Identity`` for the configured backend.

    For the dcgm backend: probe dcgmi discovery, resolve the physical-id mapping,
    take the accessible set as those physical ids (falling back to nvidia-smi),
    and re-key the per-GPU capability maps from local cgroup ids onto physical
    dcgmi ids. For prometheus: the accessible set comes from nvidia-smi and the
    dcgmi mapping is left unset (the prometheus bus-id bridge lives elsewhere).

    Never raises — any failed query degrades to an empty result.
    """
    hostname = _hostname()
    slurm_metadata = gather_slurm_metadata()

    # String-keyed (nvidia-smi index) capability maps.
    gpu_uuids = query_gpu_uuids()
    gpu_models = query_gpu_models()
    power_limits = query_power_limits()
    pcie_bw_limits, pcie_info = query_pcie_bandwidth()
    nvlink_bw_limits = query_nvlink_bandwidth()
    bus_id_map = query_bus_id_mapping()

    dcgm_physical_gpu_ids: Optional[List[str]] = None
    dcgm_phys_to_local: Dict[str, str] = {}
    accessible: Optional[Set[str]] = None

    if config.backend is BackendKind.DCGMI:
        # Discover physical dcgmi GPU ids; tolerate any failure (degrade to None).
        try:
            discovery_stdout = probe_dcgmi(timeout=_DCGMI_DISCOVERY_TIMEOUT)
        except Exception:
            discovery_stdout = ""
        if discovery_stdout:
            try:
                physical_ids, phys_to_local = resolve_dcgm_gpu_ids(discovery_stdout)
            except Exception:
                physical_ids, phys_to_local = [], {}
            if physical_ids:
                dcgm_physical_gpu_ids = physical_ids
                dcgm_phys_to_local = phys_to_local

        if dcgm_physical_gpu_ids:
            # dcgmi operates outside the cgroup; its physical ids are the
            # accessible set, and capability maps must be re-keyed onto them.
            accessible = set(dcgm_physical_gpu_ids)
            local_to_phys = {v: k for k, v in dcgm_phys_to_local.items()}
            gpu_uuids = _apply_phys_rekey_str(gpu_uuids, local_to_phys)
            gpu_models = _apply_phys_rekey_str(gpu_models, local_to_phys)
            power_limits = _apply_phys_rekey(power_limits, local_to_phys)
            pcie_bw_limits = _apply_phys_rekey(pcie_bw_limits, local_to_phys)
            nvlink_bw_limits = _apply_phys_rekey(nvlink_bw_limits, local_to_phys)
        else:
            accessible = query_accessible_gpus()
    else:
        # Prometheus backend: accessibility from nvidia-smi. The dcgm-exporter
        # bus-id bridge that maps local ids to physical exporter ids is resolved
        # by the prometheus reader/lifecycle, not here.
        accessible = query_accessible_gpus()

    return Identity(
        hostname=hostname,
        gpu_uuid_by_index=_rekey_str_to_int(gpu_uuids),
        gpu_model_by_index=_rekey_str_to_int(gpu_models),
        power_limit_watts_by_index=_rekey_to_int(power_limits),
        pcie_bw_limit_bytes_per_second_by_index=_rekey_to_int(pcie_bw_limits),
        pcie_info=pcie_info,
        nvlink_bw_limit_gbps_by_index=_rekey_to_int(nvlink_bw_limits),
        bus_id_to_index=bus_id_map,
        accessible_gpu_ids=accessible,
        dcgm_physical_gpu_ids=dcgm_physical_gpu_ids,
        dcgm_phys_to_local=dcgm_phys_to_local,
        slurm_metadata=slurm_metadata,
        gpu_uuid_by_id=gpu_uuids,
        gpu_model_by_id=gpu_models,
        power_limit_watts_by_id=power_limits,
        pcie_bw_limit_bytes_per_second_by_id=pcie_bw_limits,
        nvlink_bw_limit_gbps_by_id=nvlink_bw_limits,
    )
