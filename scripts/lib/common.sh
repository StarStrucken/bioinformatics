#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

dataset_ids() {
  tail -n +2 datasets.tsv | awk -F'\t' 'NF && $1 !~ /^#/ { print $1 }'
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

  if [[ -d "data/$dataset_id" ]] \
    && [[ -n "$(find "data/$dataset_id" -type f -newer "$stamp" -print -quit 2>/dev/null)" ]]; then
    return 0
  fi

  return 1
}
