# KempnerPulse

[![PyPI](https://img.shields.io/pypi/v/kempnerpulse)](https://pypi.org/project/kempnerpulse/)

> `nvidia-smi` says 100% GPU utilization — but are your tensor cores even active? KempnerPulse shows what's *actually* happening.

A terminal dashboard for **NVIDIA DCGM hardware-counter metrics**, with
SLURM/CUDA GPU-visibility awareness. It reads DCGM profiling counters (SM Active,
Tensor Active, DRAM Active, …) directly from `dcgmi dmon` (~100 ms) or from a
[dcgm-exporter](https://github.com/NVIDIA/dcgm-exporter) endpoint, synthesizes a
weighted **Real Utilization** score and a **12-category workload
classification**, and renders four interactive views in the terminal.

![KempnerPulse demo](https://raw.githubusercontent.com/KempnerInstitute/kempnerpulse/main/docs/images/kempner_pulse_screen_record.gif)

## Highlights

- **Four views** — Fleet, Focus (per-GPU sparkline history), Plot (line charts), Job (running processes).
- **Real Utilization** — a weighted composite of SM / Tensor / DRAM / GR-engine counters, with AI, HPC, and memory-bound presets.
- **12-category workload classification** plus health monitoring (thermal, PCIe replay, ECC), from NVIDIA's DCGM profiling guidance.
- **Three backends** — `dcgm` (direct `dcgmi dmon`), `prometheus` (`dcgm-exporter`), and `replay` (a saved capture, no GPU needed).
- **SLURM / CUDA aware** — shows only your allocated GPUs.
- **Lightweight** — standard library + `rich`; ~8% of one CPU core.

## Install

```bash
uv tool install kempnerpulse     # isolated `kempnerpulse` + `kp` commands
# or: pip install kempnerpulse
```

Requires Linux, Python ≥ 3.9, and NVIDIA data-center GPUs; the `dcgm` backend
needs the DCGM host engine. See the
[installation guide](https://kempnerinstitute.github.io/kempnerpulse/getting-started/install.html)
for uv specifics, the supported-GPU list, and running on a SLURM compute node.

## Quick start

```bash
kempnerpulse                      # live dashboard (or `kp`)
kp --focus-gpu 0                  # start focused on GPU 0
kp --hpc-weights                  # HPC weight preset
kp --export all > metrics.csv     # CSV capture
kp --backend prometheus --source http://host:9400/metrics
```

Press `Ctrl-C` to quit; type `:focus <id>`, `:plot`, `:job`, or `:q` to switch views.

## Documentation

**Full documentation: <https://kempnerinstitute.github.io/kempnerpulse/>**

- [Quickstart](https://kempnerinstitute.github.io/kempnerpulse/getting-started/quickstart.html) and [Views](https://kempnerinstitute.github.io/kempnerpulse/getting-started/views.html)
- [CLI reference](https://kempnerinstitute.github.io/kempnerpulse/guide/cli.html) — every flag and interactive command
- [Architecture](https://kempnerinstitute.github.io/kempnerpulse/architecture.html) — the four-layer pipeline
- [Workload classification](https://kempnerinstitute.github.io/kempnerpulse/classification.html) and [canonical schema](https://kempnerinstitute.github.io/kempnerpulse/schema.html)
- [CSV export](https://kempnerinstitute.github.io/kempnerpulse/export.html) and [DCGM metrics](https://kempnerinstitute.github.io/kempnerpulse/metrics.html)

## License

MIT. See [LICENSE](LICENSE).
