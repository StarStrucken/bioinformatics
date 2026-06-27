#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/precalc.sh"
cd "$REPO_ROOT"

if [[ "$#" -ne 0 ]]; then
  echo "usage: bash scripts/precalc_all.sh" >&2
  exit 2
fi

mkdir -p "$XENUM_OUTPUT_DIR"

for dataset_id in $(dataset_ids); do
  precalc_dataset "$dataset_id"
done
