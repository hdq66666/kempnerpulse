#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <compute-node> [kempnerpulse args...]" >&2
  exit 1
}

if [[ $# -lt 1 ]]; then
  usage
fi

node="$1"
shift

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

remote_cmd=$'if ! command -v nvidia-smi >/dev/null 2>&1; then\n'
remote_cmd+=$'  echo "Target node does not have nvidia-smi available; refusing to launch kempnerpulse." >&2\n'
remote_cmd+=$'  exit 1\n'
remote_cmd+=$'fi\n'
remote_cmd+=$'if ! nvidia-smi -L >/dev/null 2>&1; then\n'
remote_cmd+=$'  echo "Target node does not appear to have GPUs; refusing to launch kempnerpulse." >&2\n'
remote_cmd+=$'  exit 1\n'
remote_cmd+=$'fi\n'
remote_cmd+="cd $(printf '%q' "$script_dir") && source /n/holylfs06/LABS/kempner_shared/Everyone/common_envs/kempnerpulse/.venv/bin/activate && exec kempnerpulse"
for arg in "$@"; do
  remote_cmd+=" $(printf '%q' "$arg")"
done

exec ssh -tt -o StrictHostKeyChecking=no "$node" bash -lc "$(printf '%q' "$remote_cmd")"
