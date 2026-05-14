#!/usr/bin/env bash

mkdir -p logs outputs

while IFS=$'\t' read -r id url; do
  [[ "$id" == "dataset_id" ]] && continue
  [[ -z "$id" ]] && continue

  echo "=== $id ==="

  python dump_xenum.py "$id" \
    > "logs/dump_${id}.out" \
    2> "logs/dump_${id}.err"

  code=$?

  if [[ "$code" -eq 0 ]]; then
    echo "OK $id"
  else
    echo "FAIL $id code=$code"
  fi

done < datasets.tsv
