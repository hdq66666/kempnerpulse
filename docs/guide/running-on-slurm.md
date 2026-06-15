# Running on a SLURM compute node

On an HPC cluster the GPUs live on compute nodes, not the login node. If you
have an active allocation you can launch the dashboard on that node from the
login node with the wrapper script, without installing anything on the compute
node yourself.

```{note}
`scripts/kempnerpulse-on-cnode.sh` currently targets the **FASRC cluster**,
where a shared KempnerPulse virtual environment is available on the compute
nodes. It exits gracefully if that environment is not found. Point it at your
own environment with `KEMPNERPULSE_VENVPATH` (below).
```

## Find your compute node

```bash
squeue -u $USER
```

## Launch

Using `holygpu8a11101` as an example node:

```bash
./scripts/kempnerpulse-on-cnode.sh holygpu8a11101
```

The wrapper SSHes to the node, verifies it has GPUs, activates the KempnerPulse
environment, and launches the dashboard. Any extra arguments are passed straight
through to `kempnerpulse`:

```bash
./scripts/kempnerpulse-on-cnode.sh holygpu8a11101 --focus-gpu 0 --hpc-weights
```

To exit, press `Ctrl-C`.

## Pointing the wrapper at a different environment

If your install lives somewhere other than the shared path, set
`KEMPNERPULSE_VENVPATH` to the virtual-environment root (the script appends
`bin/activate`):

```bash
KEMPNERPULSE_VENVPATH=$HOME/path/to/your/venv ./scripts/kempnerpulse-on-cnode.sh holygpu8a11101
```

After a successful run with the override, the script offers to save it as the
new default (rewriting the hardcoded path inside the script and leaving a
`.bak` alongside it for recovery). Answering anything other than yes leaves the
script unchanged.
