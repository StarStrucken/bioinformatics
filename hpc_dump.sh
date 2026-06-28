#!/usr/bin/env bash
set -euo pipefail

target="${1:-cpu}"

if [[ "$#" -gt 1 ]]; then
  echo "usage: bash hpc_dump.sh [cpu|luna|both]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

mkdir -p logs

case "$target" in
  cpu)
    sbatch hpc_dump_cpu.sbatch
    ;;
  luna)
    sbatch hpc_luna_gpu.sbatch
    ;;
  both)
    cpu_job="$(sbatch --parsable hpc_dump_cpu.sbatch)"
    sbatch --dependency="afterany:$cpu_job" hpc_luna_gpu.sbatch
    ;;
  *)
    echo "usage: bash hpc_dump.sh [cpu|luna|both]" >&2
    exit 2
    ;;
esac
