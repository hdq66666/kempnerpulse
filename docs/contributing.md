# Contributing

## Architecture boundaries

KempnerPulse is a strict four-layer pipeline (see {doc}`architecture`). Keep
changes within their layer:

- **Read** emits source-vocabulary `RawRecord`s and never interprets meaning.
- **Translate** owns all source/version/unit knowledge and emits canonical records.
- **Compute** is pure functions over canonical records — no I/O, no UI.
- **Present** only renders; it converts to display units itself.

Runtime dependencies are the **standard library plus `rich`** only. Public code
should not reference internal planning documents.

## Development setup

```bash
git clone https://github.com/KempnerInstitute/kempnerpulse
cd kempnerpulse
uv tool install -e .     # editable install of the `kempnerpulse` / `kp` commands
```

## Tests

`pytest` is provided as a dev dependency group; `uv run` installs the project
(which brings in `rich`) plus the group, so no flags are needed:

```bash
uv run pytest tests/ -q
```

Tests are organized as `tests/unit/` (per-module) and `tests/integration/`
(the replay backend exercising the full pipeline without a GPU).

## Documentation

The docs are built with Sphinx + the Furo theme + MyST (Markdown). The `docs`
dependency group provides the toolchain:

```bash
uv run --group docs make -C docs html      # one-shot build → docs/_build/html
uv run --group docs make -C docs live       # live-reload preview on http://127.0.0.1:8000
uv run --group docs make -C docs strict     # warnings as errors (what CI runs)
uv run --group docs make -C docs clean      # remove build output
```

Documentation pages live under `docs/` as Markdown. New pages must be added to a
`{toctree}` (in `docs/index.md` or a section page) or the strict build fails. CI
(`.github/workflows/docs.yml`) builds the docs with warnings-as-errors on every
PR and deploys to GitHub Pages on push to `main`.
