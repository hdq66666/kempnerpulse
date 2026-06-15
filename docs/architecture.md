# Architecture

KempnerPulse is organized as a four-layer data-flow pipeline over a
cross-cutting tier. Data flows strictly top to bottom; each layer depends only
on the ones above it, and each has a single responsibility.

```
   source
      │
      ▼
   ┌─────────────────────────────────────────────┐
   │ Read       reader/     →  RawRecord         │
   │ Translate  translate/  →  CanonicalRecord   │
   │ Compute    compute/    →  ComputedRecord    │
   │ Present    present/    →  terminal UI / CSV │
   └─────────────────────────────────────────────┘

   cross-cutting: config · identification · selection · system_queries · lifecycle
```

## Layer 1 — Read (`kempnerpulse.reader`)

Acquires raw data from one source and emits a stream of opaque `RawRecord`s
keyed by *the source's own* field names. This layer:

- never coerces an `N/A` reading to `0.0` — it uses `None`;
- never interprets field *meanings* (naming, units, missing-value policy are
  Layer 2's job);
- has no source-vocabulary leakage above it.

Backends implement a common `Backend` protocol: `dcgmi` (a persistent
`dcgmi dmon` subprocess), `prometheus` (a `dcgm-exporter` scrape), and `replay`
(a saved capture). See {doc}`guide/backends`.

## Layer 2 — Translate (`kempnerpulse.translate`)

Maps each `RawRecord` to a `CanonicalRecord` — a single, vendor-neutral internal
vocabulary. It owns everything backend-, version-, and unit-specific: source
field names, unit normalization (percentages → fractions in `[0, 1]`,
MB/s → bytes/second, millijoules → joules), and missing-value policy. Nothing
above this layer ever sees a DCGM field identifier again. See {doc}`schema` for
the canonical record contract.

## Layer 3 — Compute (`kempnerpulse.compute`)

Pure-functional domain logic over canonical records, producing a
`ComputedRecord`: the weighted **Real Utilization** score, the 12-category
**workload classification**, and **health**. No I/O, no source vocabulary, no
UI. This is the layer that's fully testable without a GPU. See
{doc}`classification` for the composite formula and the taxonomy.

## Layer 4 — Present (`kempnerpulse.present`)

Consumes `ComputedRecord`s and renders them: the Rich terminal UI (fleet, focus,
plot, and job views) and the CSV writer. It converts canonical fractions/SI to
display units (percent, GB/s, …) at render time and never reaches back into
source vocabulary.

## Cross-cutting tier

- **`config`** — parses the command line into an immutable `Config`.
- **`identification`** — resolves device identity and capabilities at startup
  (GPU model/UUID, power and bandwidth limits, GPU-id resolution, SLURM
  metadata) via `nvidia-smi` / `dcgmi discovery`.
- **`selection`** — resolves which GPUs to show, honoring `--gpus`, `--show-all`,
  and the `CUDA_VISIBLE_DEVICES` / `SLURM_*_GPUS` environment.
- **`system_queries`** — per-sample host stats (CPU, RAM, GPU processes).
- **`lifecycle`** — the run loop (live TUI, one-shot, CSV export), a threaded
  tick reader, and centralized signal handling / teardown.

## Why the strict boundaries

A change to a DCGM field name stops at Layer 2; a change to the classification
cascade stops at Layer 3; a vendor or backend addition is a drop-in at Layer 1.
Layer 3 stays testable without hardware, and Layer 4 stays stable across backend
or driver upgrades.
