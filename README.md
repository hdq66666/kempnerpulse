# KempnerPulse

[![PyPI](https://img.shields.io/pypi/v/kempnerpulse)](https://pypi.org/project/kempnerpulse/) [![Tests](https://github.com/KempnerInstitute/kempnerpulse/actions/workflows/tests.yml/badge.svg)](https://github.com/KempnerInstitute/kempnerpulse/actions/workflows/tests.yml)

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

## V100 Custom Improvements

The `v100-custom` branch keeps the upstream behavior by default and adds a set
of V100/NVLink-focused improvements:

- **NVLink source selection** — direct DCGM mode now probes NVLink counters at
  startup, prefers `DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL` (field `449`), and falls
  back to `DCGM_FI_PROF_NVLINK_TX_BYTES` / `DCGM_FI_PROF_NVLINK_RX_BYTES`
  (fields `1011` / `1012`) only when field `449` has no usable data.
- **Fast NVLink polling** — `--sp-fast` keeps the full dashboard metric stream at
  a stable 1 second cadence while polling a lightweight NVLink-only stream at
  `--poll`, so NVLink can refresh quickly without disturbing slower profiling
  counters.
- **NVLink fit display/export** — `--nvlink-fit SCALE[,OFFSET]` displays a fitted
  estimate as `raw fittedGB/s` while preserving the raw `nvlink_gbps` value; CSV
  export adds `nvlink_est_gbps` for the fitted value.
- **Focus view left pane** — `:focus` now keeps the mini fleet in a vertical
  single-column layout, making each GPU card readable on narrower terminals.
- **Focus summary NVLink RX/TX** — real `NVLink RX` and `NVLink TX` readings are
  shown in the focus summary after `Replay rate` when fields `1011` / `1012` are
  available. If the selected source is field `449`, these entries stay absent
  instead of showing synthetic values.
- **Default NVLink Delta preserved** — the existing `NVLink Δ` aggregate remains
  in the fleet cards and focus summary, still backed by field `449` when it is
  readable.
- **Focus table cleanup** — the old `NVLink Δ` metric row was removed from the
  focus table to avoid width jitter with fitted values, and the
  `Interconnect & Power` section was renamed to `Others`.
- **Fast export overlay** — CSV export can overlay fast NVLink samples onto the
  latest full metric records when `--sp-fast` is enabled.
- **Test coverage** — unit tests cover custom DCGM field parsing, NVLink fallback
  aggregation, fitted CSV output, vertical focus layout, and focus summary
  placement of `NVLink RX` / `NVLink TX`.

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
kp --backend dcgm --poll 0.1 --sp-fast --nvlink-fit 1.37
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
