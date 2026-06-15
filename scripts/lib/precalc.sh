#!/usr/bin/env bash

morphology_files=(
  "cache/morphology_image_features.parquet"
  "cache/morphology_image_summary_features.parquet"
  "cache/morphology_image_histogram_features.parquet"
  "cache/morphology_image_texture_features.parquet"
  "cache/morphology_image_all_features.parquet"
)

pair_measurements=(
  "expression"
  "morphology"
  "spatial"
  "seq_jaccard"
  "seq_jaccard_all"
  "seq_blast"
)

optional_pair_measurements=(
  "morphology_image"
  "morphology_image_summary"
  "morphology_image_histogram"
  "morphology_image_texture"
  "morphology_image_all"
)

morphology_needs_run() {
  local dataset_id="$1"
  local out_dir="outputs/$dataset_id"
  local stamp="$out_dir/cache/morphology_image_meta.json"

  [[ ! -s "$stamp" ]] && return 0

  local rel
  for rel in "${morphology_files[@]}"; do
    [[ -s "$out_dir/$rel" ]] || return 0
  done

  newer_than_stamp "$dataset_id" "$stamp" datasets.tsv xenum/image xenum_paths.py
}

precalc_needs_run() {
  local dataset_id="$1"
  local out_dir="outputs/$dataset_id"
  local stamp="$out_dir/cache/precompute_summary.json"

  [[ ! -s "$stamp" ]] && return 0
  [[ -s "$out_dir/cache/nodes_precomputed.csv" ]] || return 0

  local measurement
  for measurement in "${pair_measurements[@]}"; do
    [[ -s "$out_dir/cache/pairs_${measurement}.parquet" ]] || return 0
  done

  for measurement in "${optional_pair_measurements[@]}"; do
    if [[ -s "$out_dir/cache/${measurement}_features.parquet" ]] \
      && [[ ! -s "$out_dir/cache/pairs_${measurement}.parquet" ]]; then
      return 0
    fi
  done

  if newer_than_stamp "$dataset_id" "$stamp" datasets.tsv xenum/dump xenum/image xenum_common.py xenum_measurements.py xenum_paths.py; then
    return 0
  fi

  if [[ -n "$(find "$out_dir/cache" -maxdepth 1 -type f -name 'morphology_image*features.parquet' -newer "$stamp" -print -quit 2>/dev/null)" ]]; then
    return 0
  fi

  return 1
}

precalc_dataset() {
  local dataset_id="$1"

  echo "=== precalc $dataset_id ==="

  if morphology_needs_run "$dataset_id"; then
    python -m xenum.image.run_morphology "$dataset_id"
  else
    echo "morphology image features are current"
  fi

  if precalc_needs_run "$dataset_id"; then
    python -m xenum.dump.precompute "$dataset_id"
  else
    echo "precompute cache is current"
  fi
}
