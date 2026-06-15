#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
cd "$REPO_ROOT"

if [[ "$#" -ne 0 ]]; then
  echo "usage: bash scripts/dump_all.sh" >&2
  exit 2
fi

bash scripts/precalc_all.sh

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

for dataset_id in $(dataset_ids); do
  echo "=== dump learned=True $dataset_id ==="

  if dump_needs_clean "$dataset_id"; then
    python -m xenum.cli.clean_outputs "$dataset_id"
  fi

  python -m xenum.dump.cli "$dataset_id"
  python -m xenum.reports.render "$dataset_id"
done

python -m xenum.cli.bench_all_xy
