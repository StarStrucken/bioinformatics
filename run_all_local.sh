#!/usr/bin/env bash
set -u

mkdir -p logs outputs

target="${1:-all}"

if [[ "$target" == "all" ]]; then
  ids="$(tail -n +2 datasets.tsv | awk -F'\t' 'NF && $1 !~ /^#/ { print $1 }')"
else
  ids="$target"
fi

for id in $ids; do
  [[ -z "$id" ]] && continue

  echo "=== $id ==="

  bash scripts/dump_xenum.sh "$id" \
    > "logs/dump_${id}.out" \
    2> "logs/dump_${id}.err"

  code=$?

  if [[ "$code" -eq 0 ]]; then
    echo "OK $id"
  else
    echo "FAIL $id code=$code"
  fi
done
