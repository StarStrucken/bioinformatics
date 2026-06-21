#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/precalc.sh"
cd "$REPO_ROOT"

if [[ "$#" -ne 0 ]]; then
  echo "usage: bash scripts/dump_all.sh" >&2
  exit 2
fi

mkdir -p outputs

dump_needs_clean() {
  local dataset_id="$1"
  local out_dir="outputs/$dataset_id"
  local stamp="$out_dir/summary.json"

  [[ ! -d "$out_dir" ]] && return 1
  [[ ! -s "$stamp" ]] && return 0

  local required=(
    "$out_dir/nodes.csv"
    "$out_dir/bench_xy.csv"
    "$out_dir/bench_xy_summary.csv"
    "$out_dir/bench_xy_best.csv"
    "$out_dir/best_k_by_measurement.csv"
    "$out_dir/reports/summary.json"
    "$out_dir/reports/tables/best_k_by_measurement.csv"
    "$out_dir/reports/tables/bench_xy_summary.csv"
    "$out_dir/reports/tables/bench_xy_best.csv"
    "$out_dir/reports/tables/report_overview.csv"
    "$out_dir/reports/figures/benchmark_best_k.png"
  )

  local path
  for path in "${required[@]}"; do
    [[ -s "$path" ]] || return 0
  done

  newer_than_stamp "$dataset_id" "$stamp" \
    datasets.tsv \
    xenum/dump \
    xenum/reports \
    xenum/cli/bench_all_xy.py \
    xenum_measurements.py \
    xenum_common.py \
    xenum_paths.py
}

reset_report_outputs() {
  local dataset_id="$1"
  local out_dir="outputs/$dataset_id"

  if [[ -d "$out_dir/reports" ]]; then
    rm -rf "$out_dir/reports"
  fi
}

spagcn_needs_run() {
  local dataset_id="$1"
  local out_dir="outputs/$dataset_id"
  local stamp="$out_dir/baselines/spagcn/summary.json"

  [[ "${XENUM_SPAGCN_FORCE:-0}" == "1" ]] && return 0
  [[ ! -s "$stamp" ]] && return 0
  [[ ! -s "$out_dir/baselines/spagcn/bench_xy.csv" ]] && return 0

  newer_than_stamp "$dataset_id" "$stamp" \
    scripts/spagcn_baseline.sh \
    xenum/baselines \
    xenum_paths.py
}

run_spagcn_baseline() {
  local dataset_id="$1"

  if [[ "${XENUM_SKIP_SPAGCN:-0}" == "1" ]]; then
    echo "=== baseline spagcn $dataset_id skipped ==="
    return
  fi

  if ! spagcn_needs_run "$dataset_id"; then
    echo "=== baseline spagcn $dataset_id is current ==="
    return
  fi

  echo "=== baseline spagcn $dataset_id ==="

  if [[ "${XENUM_SPAGCN_REQUIRED:-0}" == "1" ]]; then
    bash scripts/spagcn_baseline.sh "$dataset_id"
  else
    bash scripts/spagcn_baseline.sh "$dataset_id" \
      || echo "spagcn baseline unavailable for $dataset_id; see outputs/$dataset_id/baselines/spagcn/summary.json" >&2
  fi
}

for dataset_id in $(dataset_ids); do
  precalc_dataset "$dataset_id"

  echo "=== dump learned=True $dataset_id ==="

  if dump_needs_clean "$dataset_id"; then
    python -m xenum.cli.clean_outputs "$dataset_id"
  fi

  reset_report_outputs "$dataset_id"
  python -m xenum.dump.cli "$dataset_id"
  run_spagcn_baseline "$dataset_id"
  python -m xenum.reports.render "$dataset_id"
done

python -m xenum.cli.bench_all_xy
