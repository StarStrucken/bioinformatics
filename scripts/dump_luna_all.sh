#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
cd "$REPO_ROOT"

if [[ "$#" -ne 0 ]]; then
  echo "usage: bash scripts/dump_luna_all.sh" >&2
  exit 2
fi

mkdir -p "$XENUM_OUTPUT_DIR"
CURRENT_DATASET=""

rebuild_global_outputs() {
  run_python -m xenum.cli.bench_all_xy || true
}

dataset_loop_ids() {
  if [[ -n "${XENUM_DATASET_IDS:-}" ]]; then
    printf '%s\n' $XENUM_DATASET_IDS
  else
    dataset_ids
  fi
}

mark_interrupted() {
  if [[ -n "$CURRENT_DATASET" ]]; then
    local out_dir="$XENUM_OUTPUT_DIR/$CURRENT_DATASET"
    local tmp="$out_dir/luna_pipeline_status.json.tmp"
    mkdir -p "$out_dir"
    printf '{"dataset":"%s","status":"interrupted"}\n' "$CURRENT_DATASET" > "$tmp"
    mv "$tmp" "$out_dir/luna_pipeline_status.json"
  fi

  rebuild_global_outputs
  exit 130
}

trap mark_interrupted INT TERM

for dataset_id in $(dataset_loop_ids); do
  CURRENT_DATASET="$dataset_id"
  echo "=== luna $dataset_id ==="
  run_python -m xenum.cli.luna_dataset "$dataset_id"
  run_python -m xenum.reports.render "$dataset_id" || true
  CURRENT_DATASET=""
  rebuild_global_outputs
done

rebuild_global_outputs
