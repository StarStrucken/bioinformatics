#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
XENUM_MAIN_PYTHON="${XENUM_MAIN_PYTHON:-python}"
XENUM_DATA_DIR="${XENUM_DATA_DIR:-$REPO_ROOT/data}"
XENUM_OUTPUT_DIR="${XENUM_OUTPUT_DIR:-$REPO_ROOT/outputs}"
XENUM_DATASETS="${XENUM_DATASETS:-$REPO_ROOT/datasets.tsv}"

dataset_ids() {
  tail -n +2 "$XENUM_DATASETS" | awk -F'\t' 'NF && $1 !~ /^#/ { print $1 }'
}

run_python() {
  "$XENUM_MAIN_PYTHON" "$@"
}

newer_than_stamp() {
  local dataset_id="$1"
  local stamp="$2"
  shift 2

  [[ ! -s "$stamp" ]] && return 0

  local path
  for path in "$@"; do
    [[ -e "$path" ]] || continue
    if [[ -n "$(find "$path" -type f -newer "$stamp" -not -path '*/__pycache__/*' -print -quit 2>/dev/null)" ]]; then
      return 0
    fi
  done

  if [[ -d "$XENUM_DATA_DIR/$dataset_id" ]] \
    && [[ -n "$(find "$XENUM_DATA_DIR/$dataset_id" -type f -newer "$stamp" -print -quit 2>/dev/null)" ]]; then
    return 0
  fi

  return 1
}
