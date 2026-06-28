#!/usr/bin/env python3
from __future__ import annotations

import argparse

import pandas as pd

from xenum_paths import data_dir

CELL_TABLE_NAMES = ("cells.csv.gz", "cells.csv", "cells.parquet")

def find_cell_table(dataset_id):
    root = data_dir(dataset_id)
    roots = [root] if root.name == "bundle" else [root, root / "bundle"]

    for base in roots:
        for name in CELL_TABLE_NAMES:
            path = base / name
            if path.exists():
                return path

    for base in roots:
        if not base.exists():
            continue
        for name in CELL_TABLE_NAMES:
            found = sorted(base.rglob(name))
            if found:
                return found[0]

    raise FileNotFoundError(f"cell table not found for {dataset_id}")

def count_rows(path):
    if path.suffix == ".parquet":
        return int(len(pd.read_parquet(path, columns=[])))

    return int(sum(len(chunk) for chunk in pd.read_csv(path, usecols=[0], chunksize=100_000)))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_id")
    return p.parse_args()

def main():
    args = parse_args()
    print(count_rows(find_cell_table(args.dataset_id)))

if __name__ == "__main__":
    main()
