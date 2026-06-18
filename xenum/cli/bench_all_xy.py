#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

def read_bench(path, source, baseline=""):
    df = pd.read_csv(path)

    if "source" not in df.columns:
        df["source"] = source

    if "baseline" not in df.columns:
        df["baseline"] = baseline

    return df

def main():
    rows = []

    for p in sorted(Path("outputs").glob("*/bench_xy.csv")):
        rows.append(read_bench(p, "core"))

    for p in sorted(Path("outputs").glob("*/baselines/*/bench_xy.csv")):
        rows.append(read_bench(p, "baseline", p.parent.name))

    if not rows:
        raise SystemExit("no bench_xy.csv files found")

    df = pd.concat(rows, ignore_index=True)

    if "k" not in df.columns:
        df["k"] = 4

    if "leaky" not in df.columns:
        df["leaky"] = False

    df["source"] = df["source"].fillna("core")
    df["baseline"] = df["baseline"].fillna("")

    clean = df[(~df["leaky"]) | (df["source"].eq("baseline"))].copy()
    clean["rank"] = clean.groupby("dataset")["median_vs_center"].rank(method="min")

    summary = (
        clean
        .groupby(["source", "baseline", "measurement", "k"], dropna=False)
        .agg(
            datasets=("dataset", "nunique"),
            wins=("rank", lambda x: int((x == 1).sum())),
            rank_mean=("rank", "mean"),
            median_vs_center_mean=("median_vs_center", "mean"),
            median_vs_center_median=("median_vs_center", "median"),
            median_xy_error_median=("median_xy_error", "median"),
            coverage_mean=("coverage", "mean"),
        )
        .reset_index()
        .sort_values(["wins", "rank_mean", "median_vs_center_median"], ascending=[False, True, True])
    )

    print(summary.to_string(index=False))

    best = (
        clean
        .sort_values(["dataset", "median_vs_center", "median_xy_error"])
        .groupby("dataset")
        .head(1)
        [["dataset", "source", "baseline", "measurement", "k", "median_vs_center", "median_xy_error", "coverage"]]
    )

    print()
    print(best.to_string(index=False))

    best.to_csv("outputs/bench_xy_best.csv", index=False)
    summary.to_csv("outputs/bench_xy_summary.csv", index=False)
    df.to_csv("outputs/bench_xy_all.csv", index=False)

if __name__ == "__main__":
    main()
