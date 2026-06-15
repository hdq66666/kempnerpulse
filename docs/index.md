# KempnerPulse

A terminal dashboard for **NVIDIA DCGM hardware-counter metrics**, with
SLURM/CUDA GPU-visibility awareness.

KempnerPulse reads DCGM profiling counters — SM Active, Tensor Active, DRAM
Active, GR Engine Active, the precision pipes, PCIe/NVLink throughput, power,
thermals, and clocks — and renders them live in the terminal. It synthesizes a
weighted **Real Utilization** score and a **12-category workload
classification** so you can tell idle GPUs from real compute, memory pressure,
transfer/copy pressure, and hardware-health issues at a glance.

Where `nvidia-smi` reads NVML and reports a single high-level `GPU-Util`
time-fraction ("was a kernel running?"), KempnerPulse reads DCGM and exposes the
*composition* of active GPU time ("which functional units are busy, and how
hard?"). The two are complementary; KempnerPulse focuses on the fine-grained
hardware-counter view.

## New here?

- **{doc}`getting-started/install`** — install with `uv` or `pip`, prerequisites
  (the DCGM host engine for the `dcgm` backend), and SLURM notes.
- **{doc}`getting-started/quickstart`** — launch the live dashboard, take a
  one-shot snapshot, export CSV, switch backends, and pick a weight preset.

## How it works

- **{doc}`architecture`** — the four-layer pipeline (Read → Translate → Compute
  → Present) and the cross-cutting tier.
- **{doc}`classification`** — the Real Utilization composite and the 12-category
  workload taxonomy with their thresholds.
- **{doc}`schema`** — the canonical record: the internal, vendor-neutral
  vocabulary every layer above Read depends on.

## Using it

- **{doc}`guide/cli`** — the full command-line surface (`kempnerpulse` / `kp`),
  every flag, and the interactive key commands.
- **{doc}`guide/backends`** — `dcgm` (direct `dcgmi dmon`), `prometheus`
  (`dcgm-exporter`), and `replay` (a saved capture, no GPU needed).
- **{doc}`guide/running-on-slurm`** — launch the dashboard on an allocated
  compute node from the login node.
- **{doc}`export`** — the CSV export schema and column reference.

## Reference

- **{doc}`metrics`** — every DCGM field KempnerPulse consumes, with units and
  practical peaks.
- **{doc}`api/index`** — API reference, auto-generated from the package source.

```{toctree}
:hidden:
:maxdepth: 2
:caption: Getting Started

getting-started/install
getting-started/quickstart
getting-started/views
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Concepts

architecture
classification
schema
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Usage

guide/cli
guide/backends
guide/running-on-slurm
export
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Reference

metrics
api/index
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Project

contributing
```
