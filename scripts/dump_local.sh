#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
cd "$REPO_ROOT"

if [[ "$#" -ne 0 ]]; then
  echo "usage: bash scripts/dump_local.sh" >&2
  exit 2
fi

LOCAL_ENV="${XENUM_LOCAL_ENV:-.venv}"

if [[ -d "$LOCAL_ENV" ]]; then
  # shellcheck disable=SC1091
  source "$LOCAL_ENV/bin/activate"
fi

export XENUM_MAIN_PYTHON="${XENUM_MAIN_PYTHON:-python}"

selected=()

for dataset_id in $(dataset_ids); do
  n_cells="$(run_python -m xenum.cli.dataset_cell_count "$dataset_id")"

  if [[ "$n_cells" -lt 1000 ]]; then
    selected+=("$dataset_id")
    echo "local include $dataset_id n_cells=$n_cells"
  else
    echo "local skip $dataset_id n_cells=$n_cells"
  fi
done

if [[ "${#selected[@]}" -eq 0 ]]; then
  echo "no local datasets with n_cells < 1000"
  run_python -m xenum.cli.bench_all_xy || true
  exit 0
fi

export XENUM_DATASET_IDS="${selected[*]}"
bash scripts/dump_all.sh
