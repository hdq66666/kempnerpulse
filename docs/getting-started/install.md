# Installation

## Prerequisites

- **Linux** with **Python 3.9+** and at least one **NVIDIA GPU**.
- For the default **`dcgm` backend**: a working **NVIDIA DCGM** install — the
  `dcgmi` command on `PATH`, the DCGM host engine (`nv-hostengine`) reachable,
  and permission to read profiling counters (DCGM profiling has required admin
  privileges since driver 418.43).
- For the **`prometheus` backend**: a reachable
  [`dcgm-exporter`](https://github.com/NVIDIA/dcgm-exporter) `/metrics` endpoint.
- The only runtime dependency is [`rich`](https://github.com/Textualize/rich);
  everything else is the Python standard library.
- **Supported GPUs:** NVIDIA data-center GPUs — V100, A100, H100, H200, B200,
  B300. Grace-Hopper (GH200), Grace-Blackwell (GB200), and RTX support is planned
  but not yet tested; AMD GPUs are not supported.

You can also explore KempnerPulse with no GPU at all using the **`replay`
backend** on a saved capture — see {doc}`../guide/backends`.

## Install

With [uv](https://docs.astral.sh/uv/) (recommended — installs `kempnerpulse` and
`kp` as isolated commands available from any directory):

```bash
uv tool install kempnerpulse
uv tool update-shell           # one-time: add uv's tool bin dir (e.g. ~/.local/bin) to PATH
```

Run it once without installing (ephemeral, cached):

```bash
uvx kempnerpulse --help
```

Or with `pip`:

```bash
pip install kempnerpulse
```

### From source

From a local checkout, pass the path — a bare `uv tool install kempnerpulse`
always resolves from PyPI regardless of the current directory:

```bash
git clone https://github.com/KempnerInstitute/kempnerpulse
cd kempnerpulse
uv tool install -e .            # editable: code changes are picked up live
```

If the command already exists from an earlier install, uv refuses with
`error: Executable already exists`; add `--force` to overwrite:

```bash
uv tool install --force -e .
```

## Verify

```bash
kempnerpulse --version
kp --help                  # `kp` is an alias for `kempnerpulse`
```

If `dcgmi` is unavailable, the `dcgm` backend exits with an actionable message;
use `--backend prometheus` (pointing at a `dcgm-exporter` endpoint) or
`--backend replay` instead.

## Overhead

KempnerPulse is lightweight: roughly 8% of a single CPU core (measured on an AMD
EPYC 9374F) and negligible memory. It samples DCGM counters off-GPU, so it does
not perturb the workload under observation.

## Upgrade / uninstall

```bash
uv tool upgrade kempnerpulse
uv tool uninstall kempnerpulse
```
