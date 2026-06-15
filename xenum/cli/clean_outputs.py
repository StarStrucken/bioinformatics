#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from xenum_paths import existing_out_dir

KEEP_DIRS = {
    "cache",
}

REMOVE_FILE_PATTERNS = [
    "nodes*.csv",
    "edges*.csv",
    "predictions*.csv",
    "checks*.json",
    "pairs*.parquet",
    "representation*.npz",
    "bench*.csv",
    "best_k_by_measurement.csv",
    "learned_mix*.csv",
    "learned_mix*.json",
    "summary.json",
]

REMOVE_EMPTY_DIRS = True
DRY_RUN = False

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_id")
    return p.parse_args()

def remove_path(path):
    if DRY_RUN:
        print(f"would remove: {path}")
        return

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()

    print(f"removed: {path}")

def clean_files(out_dir):
    seen = set()

    for pattern in REMOVE_FILE_PATTERNS:
        for path in out_dir.glob(pattern):
            if not path.is_file():
                continue

            if path.name in seen:
                continue

            seen.add(path.name)
            remove_path(path)

def clean_dirs(out_dir):
    for path in sorted(out_dir.iterdir()):
        if not path.is_dir():
            continue

        if path.name in KEEP_DIRS:
            continue

        remove_path(path)

def clean_empty_dirs(out_dir):
    if not REMOVE_EMPTY_DIRS:
        return

    for path in sorted(out_dir.rglob("*"), reverse=True):
        if not path.is_dir():
            continue

        if path == out_dir:
            continue

        if any(part in KEEP_DIRS for part in path.relative_to(out_dir).parts):
            continue

        try:
            path.rmdir()
            print(f"removed empty dir: {path}")
        except OSError:
            pass

def main():
    args = parse_args()
    out_dir = existing_out_dir(args.dataset_id)

    print(f"cleaning: {out_dir}")
    print(f"keeping dirs: {', '.join(sorted(KEEP_DIRS))}")

    clean_files(out_dir)
    clean_dirs(out_dir)
    clean_empty_dirs(out_dir)

    print(f"done: {out_dir}")

if __name__ == "__main__":
    main()
