#!/usr/bin/env bash
set -euo pipefail

dataset_id="$1"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_dir="$repo_root/outputs/$dataset_id"
cache_dir="$out_dir/cache"
log_dir="$out_dir/logs"

mkdir -p "$cache_dir" "$log_dir"

python "$repo_root/scripts/run_image_morphology.py" "$dataset_id" \
  > "$log_dir/morphology_image.out" \
  2> "$log_dir/morphology_image.err"
