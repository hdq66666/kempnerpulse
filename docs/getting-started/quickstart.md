# Quickstart

Every command works as either `kempnerpulse` or the shorter `kp`.

## Launch the live dashboard

```bash
kempnerpulse
```

This is equivalent to `kempnerpulse --backend dcgm --poll 0.1` — it streams
directly from `dcgmi dmon` at ~100 ms resolution and shows every GPU visible to
your process (respecting `CUDA_VISIBLE_DEVICES` / `SLURM_JOB_GPUS`). Press
`Ctrl-C` to quit.

## One-shot snapshot

```bash
kp --once          # render a single frame and exit
```

## Export CSV

```bash
kp --export                 # default columns, to stdout
kp --export all > run.csv   # every column
kp --export sm_active_pct,tensor_active_pct,real_util_pct   # a custom set
```

See {doc}`../export` for the full column reference.

## Pick a weight preset

The Real Utilization composite is a weighted blend of SM / Tensor / DRAM / GR
counters. Choose a preset for your workload, or supply custom weights:

```bash
kp --ai-weights      # AI / LLM training & inference (default): 0.35,0.35,0.20,0.10
kp --hpc-weights     # general HPC:                              0.45,0.15,0.25,0.15
kp --mem-weights     # memory-bound / bandwidth-heavy:           0.35,0.10,0.40,0.15
kp --weights 0.40,0.30,0.20,0.10   # custom (normalized to sum to 1)
```

## Choose which GPUs to show

```bash
kp --focus-gpu 0     # start focused on GPU 0
kp --gpus 0,1        # only GPUs 0 and 1 (also accepts ranges like 0-3)
kp --show-all        # ignore SLURM/CUDA env and show every accessible GPU
```

## Switch backends

```bash
kp --backend prometheus --source http://localhost:9400/metrics --poll 1.0
kp --backend replay --source capture.txt --once     # no GPU needed
```

## Interactive commands

While the live dashboard is running, type `:`-prefixed commands:

| Command | Action |
|---|---|
| `:focus <id>` | focus a single GPU's detailed view |
| `:plot` | line-chart view (history sparklines) |
| `:job` | running-process table |
| `:q` | step back one view (focus/plot/job → fleet) |
| `:exit` | quit |

In the fleet view, scroll card rows with `↑`/`↓`, `PgUp`/`PgDn`, or `j`/`k` when
there are more GPUs than fit the window.
