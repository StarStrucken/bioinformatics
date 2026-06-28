#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/precalc.sh"
cd "$REPO_ROOT"

if [[ "$#" -ne 0 ]]; then
  echo "usage: bash scripts/dump_all.sh" >&2
  exit 2
fi

mkdir -p "$XENUM_OUTPUT_DIR"
CURRENT_DATASET=""

write_pipeline_status() {
  local dataset_id="$1"
  local status="$2"
  local out_dir="$XENUM_OUTPUT_DIR/$dataset_id"
  local tmp="$out_dir/pipeline_status.json.tmp"

  mkdir -p "$out_dir"
  printf '{"dataset":"%s","status":"%s"}\n' "$dataset_id" "$status" > "$tmp"
  mv "$tmp" "$out_dir/pipeline_status.json"
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
    write_pipeline_status "$CURRENT_DATASET" "interrupted"
  fi

  exit 130
}

trap mark_interrupted INT TERM

pipeline_was_interrupted() {
  local dataset_id="$1"
  local status_file="$XENUM_OUTPUT_DIR/$dataset_id/pipeline_status.json"

  [[ -s "$status_file" ]] || return 1
  grep -Eq '"status"[[:space:]]*:[[:space:]]*"(interrupted|failed)"' "$status_file"
}

dump_outputs_current() {
  local dataset_id="$1"
  local out_dir="$XENUM_OUTPUT_DIR/$dataset_id"
  local stamp="$out_dir/summary.json"

  [[ -d "$out_dir" ]] || return 1
  [[ -s "$stamp" ]] || return 1

  local required=(
    "$out_dir/nodes.csv"
    "$out_dir/bench_xy.csv"
    "$out_dir/bench_xy_summary.csv"
    "$out_dir/bench_xy_best.csv"
    "$out_dir/best_k_by_measurement.csv"
  )

  local path
  for path in "${required[@]}"; do
    [[ -s "$path" ]] || return 1
  done

  pipeline_was_interrupted "$dataset_id" && return 1

  if newer_than_stamp "$dataset_id" "$stamp" \
    "$XENUM_DATASETS" \
    external/LUNA \
    xenum/dump \
    xenum/reports \
    xenum_measurements.py \
    xenum_common.py \
    xenum_paths.py; then
    return 1
  fi

  return 0
}

reports_current() {
  local dataset_id="$1"
  local out_dir="$XENUM_OUTPUT_DIR/$dataset_id"

  local required=(
    "$out_dir/reports/summary.json"
    "$out_dir/reports/tables/best_k_by_measurement.csv"
    "$out_dir/reports/tables/bench_xy_summary.csv"
    "$out_dir/reports/tables/bench_xy_best.csv"
    "$out_dir/reports/tables/report_overview.csv"
    "$out_dir/reports/figures/benchmark_best_k.png"
  )

  local path
  for path in "${required[@]}"; do
    [[ -s "$path" ]] || return 1
  done

  return 0
}

reset_report_outputs() {
  local dataset_id="$1"
  local out_dir="$XENUM_OUTPUT_DIR/$dataset_id"

  if [[ -d "$out_dir/reports" ]]; then
    rm -rf "$out_dir/reports"
  fi
}

for dataset_id in $(dataset_loop_ids); do
  CURRENT_DATASET="$dataset_id"
  precalc_dataset "$dataset_id"

  echo "=== dump learned=True $dataset_id ==="

  if dump_outputs_current "$dataset_id"; then
    echo "dump outputs are current"
  else
    run_python -m xenum.cli.clean_outputs "$dataset_id"
    run_python -m xenum.dump.cli "$dataset_id"
  fi

  if reports_current "$dataset_id"; then
    echo "reports are current"
  else
    reset_report_outputs "$dataset_id"
    run_python -m xenum.reports.render "$dataset_id"
  fi

  write_pipeline_status "$dataset_id" "completed"
  CURRENT_DATASET=""
done

bash scripts/bench_all.sh
