#!/usr/bin/env bash
set -e

target="${1:?usage: bash hpc_dump.sh all|dataset_id}"

PARTITION="${PARTITION:-c23ms}"

mkdir -p logs outputs

if [[ "$target" == "all" ]]; then
  ids=$(tail -n +2 datasets.tsv | cut -f1)
else
  ids="$target"
fi

for id in $ids; do
  cells=$(awk -F'\t' -v id="$id" '$1 == id { print $4 }' downloads/meta.tsv)

  if [[ -z "$cells" ]]; then
    echo "miss meta $id"
    continue
  fi

  mins=$((30 + (cells + 99999) / 100000 * 12))
  hours=$((mins / 60))
  rest=$((mins % 60))
  walltime=$(printf "%02d:%02d:00" "$hours" "$rest")

  echo "submit $id cells=$cells time=$walltime partition=$PARTITION"

  sbatch \
    --partition="$PARTITION" \
    --job-name="xenum_${id}" \
    --time="$walltime" \
    --cpus-per-task=8 \
    --mem=32G \
    --output="logs/%x-%j.out" \
    --wrap="
      set -e
      cd '$PWD'
      . .venv/bin/activate
      python dump_xenum.py \
        --xenium-dir data/$id \
        --out-dir outputs/$id
    "
done
