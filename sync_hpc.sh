#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 0 ]]; then
  echo "usage: bash sync_hpc.sh" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_HOST="${XENUM_HPC_HOST:-hi202241@login23-g-1.hpc.itc.rwth-aachen.de}"
REMOTE_DIR="${XENUM_HPC_PROJECT_DIR:-/work/hi202241/bioinformatics}"

# Intentionally no --delete: datasets, outputs, logs, and remote-only
# submodule checkouts must not be removed from the HPC copy.
rsync -avh --progress \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='.venv-luna/' \
  --exclude='__pycache__/' \
  --exclude='*.py[cod]' \
  --exclude='data/' \
  --exclude='downloads/' \
  --exclude='outputs/' \
  --exclude='logs/' \
  "$REPO_ROOT/" \
  "$REMOTE_HOST:$REMOTE_DIR/"
