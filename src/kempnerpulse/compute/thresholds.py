"""Numeric cutoffs for the Real Utilization composite and the classification.

Every threshold lives here as a named constant so the classification cascade and
the composite read as prose. Each constant is tagged as either:

* NVIDIA-anchored — stated in NVIDIA's published DCGM profiling guidance, and
* pragmatic — a KempnerPulse partition of the NVIDIA-anchored space, chosen to
  separate workload regimes that the anchored cutoffs alone do not name.

All percent thresholds are on the 0..100 scale (canonical fractions ×100).
"""
from __future__ import annotations

# ── NVIDIA-anchored cutoffs ──────────────────────────────────────────────────
# "A value of 0.8 or greater is necessary, but not sufficient, for effective use
# of the GPU." Used to flag a well-saturated streaming-multiprocessor array.
SM_EFFECTIVE_HIGH = 80.0          # NVIDIA-anchored (SM Active >= 0.80)

# "A value less than 0.5 likely indicates ineffective GPU usage." Used as the
# boundary below which SM occupancy is treated as not effectively driving compute.
SM_INEFFECTIVE = 50.0             # NVIDIA-anchored (SM Active < 0.50)

# "In practice a peak of ~0.8 (80%) is the maximum achievable." The practical
# ceiling for the DRAM controller active fraction.
DRAM_PRACTICAL_PEAK = 80.0        # NVIDIA-anchored (DRAM Active practical peak ~0.80)

# Tensor-core activity saturates near ~93% in NVIDIA's own dcgmproftester run.
TENSOR_SATURATION = 93.0          # NVIDIA-anchored (Tensor ~0.93 saturation)

# ── Pragmatic KempnerPulse partitions ────────────────────────────────────────
# Real Utilization below this, with no graphics/DRAM activity, reads as idle.
IDLE_REAL_UTIL = 5.0              # pragmatic
IDLE_ENGINE = 5.0                # pragmatic (gr / dram "nothing running" floor)

# Tensor-heavy compute: dominant tensor pipe with well-loaded SMs.
TENSOR_HEAVY_TENSOR = 50.0       # pragmatic
TENSOR_HEAVY_SM = 60.0           # pragmatic

# Tensor compute: meaningful tensor activity with moderate SM load.
TENSOR_COMPUTE_TENSOR = 15.0     # pragmatic
TENSOR_COMPUTE_SM = 40.0         # pragmatic

# FP64 / HPC compute: appreciable double-precision pipe with loaded SMs.
FP64_HPC_FP64 = 20.0             # pragmatic
FP64_HPC_SM = 50.0              # pragmatic

# I/O or data-loading: heavy transfer with idle SMs.
IO_MEMCPY = 40.0                # pragmatic (memory-copy engine busy fraction)
IO_PCIE_BYTES_PER_SECOND = 1e9   # pragmatic (>= 1 GB/s on either PCIe direction)
IO_SM_IDLE = 30.0              # pragmatic (SM below this counts as idle for I/O)

# Memory-bound: bandwidth-limited with SMs below the effective threshold.
MEMORY_BOUND_DRAM = 50.0         # pragmatic
MEMORY_BOUND_SM = 50.0          # pragmatic

# Memory-active: significant DRAM traffic alongside some SM activity.
MEMORY_ACTIVE_DRAM = 40.0        # pragmatic

# Compute-active: moderate SM use, no tensor dominance.
COMPUTE_ACTIVE_SM = 50.0         # pragmatic

# Busy, low SM use: graphics engine active but SMs underutilized.
BUSY_LOW_SM_GR = 40.0           # pragmatic
BUSY_LOW_SM_SM = 25.0          # pragmatic

# Low utilization: barely any measurable activity on any tracked axis.
LOW_UTILIZATION_GR = 15.0        # pragmatic
LOW_UTILIZATION_SM = 15.0        # pragmatic
LOW_UTILIZATION_DRAM = 15.0      # pragmatic
