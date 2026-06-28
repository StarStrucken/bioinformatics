#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
cd "$REPO_ROOT"

if [[ "$#" -ne 0 ]]; then
  echo "usage: bash scripts/bench_all.sh" >&2
  exit 2
fi

run_python -m xenum.cli.bench_all_xy
