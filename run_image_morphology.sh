#!/usr/bin/env bash
set -euo pipefail

dataset_id="${1:?dataset_id required}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out_dir="$repo_root/outputs/$dataset_id"
log_dir="$out_dir/diagnostics"

mkdir -p "$log_dir"

python "$repo_root/run_image_morphology.py" "$dataset_id" \
  > "$log_dir/image_morphology.out" \
  2> "$log_dir/image_morphology.err"
