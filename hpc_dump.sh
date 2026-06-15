#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs outputs

target="${1:-all}"

if [[ "$target" == "all" ]]; then
  ids="$(tail -n +2 datasets.tsv | awk -F'\t' 'NF && $1 !~ /^#/ { print $1 }')"
else
  ids="$target"
fi

for id in $ids; do
  [[ -z "$id" ]] && continue

  echo "submit $id time=${TIME:-48:00:00} partition=${PARTITION:-c23ms}"

  sbatch \
    --partition="${PARTITION:-c23ms}" \
    --job-name="xenum_${id}" \
    --time="${TIME:-48:00:00}" \
    --cpus-per-task="${CPUS:-16}" \
    --mem="${MEM:-64G}" \
    --output="logs/%x-%j.out" \
    --wrap="cd '$PWD'; . .venv/bin/activate; python -m xenum.dump.cli '$id'; python -m xenum.reports.render '$id'"
done
