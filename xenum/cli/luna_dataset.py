#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from xenum_measurements import LUNA_MEASUREMENTS
from xenum_paths import data_dir, out_dir as make_out_dir

from xenum.dump.benchmarks import add_spatial_reference, best_k_by_measurement, normalize_benchmark_rows, summarize_benchmarks
from xenum.dump.features import make_nodes
from xenum.dump.io import load_xenium, make_output_sections, mirror_outputs
from xenum.dump.luna import run_luna_measurements

def atomic_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_id")
    return p.parse_args()

def main():
    args = parse_args()
    out_dir = make_out_dir(args.dataset_id)
    bench_path = out_dir / "bench_xy.csv"

    if not bench_path.exists():
        print(f"{args.dataset_id}: skip LUNA, missing {bench_path}", flush=True)
        return

    make_output_sections(out_dir)

    adata = load_xenium(data_dir(args.dataset_id))
    nodes = make_nodes(adata)

    luna_rows, _summaries = run_luna_measurements(
        out_dir,
        args.dataset_id,
        adata,
        nodes,
    )

    bench = pd.read_csv(bench_path)
    bench = bench[~bench["measurement"].astype(str).isin(LUNA_MEASUREMENTS)].copy()
    bench = pd.concat([bench, pd.DataFrame(luna_rows)], ignore_index=True, sort=False)
    bench = normalize_benchmark_rows(bench).sort_values(["leaky", "measurement", "k", "seed"], na_position="last")
    bench = add_spatial_reference(bench)

    atomic_csv(bench, out_dir / "bench_xy.csv")
    atomic_csv(bench, out_dir / "bench_xy_by_k.csv")

    bench_summary, bench_best = summarize_benchmarks(bench)
    atomic_csv(bench_summary, out_dir / "bench_xy_summary.csv")
    atomic_csv(bench_best, out_dir / "bench_xy_best.csv")
    atomic_csv(best_k_by_measurement(bench), out_dir / "best_k_by_measurement.csv")

    mirror_outputs(out_dir)
    print(f"{args.dataset_id}: LUNA rows merged into {bench_path}", flush=True)

if __name__ == "__main__":
    main()
