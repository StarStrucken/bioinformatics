#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
cd "$REPO_ROOT"

if [[ "$#" -lt 1 ]]; then
  echo "usage: bash scripts/spagcn_baseline.sh DATASET_ID [adapter args...]" >&2
  echo "set XENUM_SPAGCN_PYTHON=/path/to/spagcn-env/bin/python to use a separate environment" >&2
  exit 2
fi

py="${XENUM_SPAGCN_PYTHON:-python}"

"$py" -m xenum.baselines.spagcn "$@"
