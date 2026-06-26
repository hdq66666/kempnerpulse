# Command-line reference

The dashboard is launched as `kempnerpulse` or the alias `kp` — they are
identical.

```bash
kempnerpulse [options]
kp [options]
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--source SOURCE` | `http://localhost:9400/metrics` | A `dcgm-exporter` HTTP `/metrics` endpoint or a local file. For `--backend replay`, the path to a saved `dcgmi dmon` capture. |
| `--backend {dcgm,prometheus,replay}` | `dcgm` | Metric source. See {doc}`backends`. |
| `--poll POLL` | `0.1` (dcgm) / `1.0` (prometheus) | Sampling interval in seconds. The `dcgm` backend is honored down to a 100 ms floor; `prometheus` requires `>= 1.0`. |
| `--sp-fast` | off | With `--backend dcgm`, sample NVLink at `--poll` in a lightweight stream while normal dashboard metrics use a 1 s stream. |
| `--nvlink-fit SCALE[,OFFSET]` | — | Display/export `nvlink_est_gbps` using `raw * SCALE + OFFSET`; raw `nvlink_gbps` is preserved. |
| `--history HISTORY` | `120` | Samples kept for sparkline history. |
| `--focus-gpu ID` | — | Start in the focused single-GPU view. |
| `--once` | off | Render one snapshot and exit. |
| `--gpus GPUS` | — | Explicit GPU ids or ranges (`0,1` or `0-3`). Overrides the visibility environment. |
| `--show-all` | off | Ignore `CUDA_VISIBLE_DEVICES` / `SLURM_*_GPUS` and show every accessible GPU. |
| `--weights W_SM,W_TENSOR,W_DRAM,W_GR` | AI preset | Custom Real-Utilization weights (normalized to sum to 1). |
| `--ai-weights` | (default) | Preset `0.35,0.35,0.20,0.10`. |
| `--hpc-weights` | — | Preset `0.45,0.15,0.25,0.15`. |
| `--mem-weights` | — | Preset `0.35,0.10,0.40,0.15`. |
| `--export [COLS]` | — | Write CSV to stdout. Bare for the default columns, `all` for every column, or a comma-separated list. See {doc}`../export`. |
| `--version` | — | Print the version and exit. |

## Interactive commands

While the live dashboard runs, type a `:`-prefixed command and press Enter:

| Command | Action |
|---|---|
| `:focus <id>` | focused single-GPU detail view |
| `:plot` | line-chart (history) view |
| `:job` | running-process table |
| `:q` | step back one view level (fleet → exit) |
| `:exit` | quit (also `Ctrl-C`) |
| `Esc` | cancel an unfinished `:` command |

When the fleet has more GPU cards than fit the window, scroll the rows with
`↑`/`↓`, `PgUp`/`PgDn`, or `j`/`k`. The layout is responsive: it drops summary
and footer fields as the terminal narrows, switches GPU cards between one- and
two-column detail, and shows a "terminal too small" notice below a minimum size.

## GPU visibility selection

KempnerPulse shows only the GPUs visible to your process, picking the first
available source in this order:

1. `--gpus`
2. `CUDA_VISIBLE_DEVICES`
3. `NVIDIA_VISIBLE_DEVICES`
4. `SLURM_STEP_GPUS`
5. `SLURM_JOB_GPUS`

If none are set, all GPUs on the node are shown. `--show-all` ignores the
environment entirely. Every selection is filtered against the GPUs actually
accessible to the process (as reported by `nvidia-smi`), so cgroup and container
restrictions are always respected.

## Examples

```bash
kempnerpulse                                   # live, dcgm backend, 100 ms
kp --poll 1.0                                  # slower refresh
kp --focus-gpu 0                               # start focused on GPU 0
kp --hpc-weights                               # HPC weight preset
kp --gpus 2,3                                  # only GPUs 2 and 3
kp --once                                      # single snapshot
kp --backend dcgm --export all --poll 0.1      # high-res CSV capture
kp --backend dcgm --poll 0.1 --sp-fast         # fast NVLink refresh
kp --backend dcgm --poll 0.1 --sp-fast --nvlink-fit 1.37
kp --backend prometheus --source http://host:9400/metrics --poll 2
kp --backend replay --source capture.txt --once
```
