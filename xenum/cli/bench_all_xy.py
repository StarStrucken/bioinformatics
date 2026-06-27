#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xenum.dump.benchmarks import normalize_benchmark_rows, selected_benchmark_rows
from xenum_paths import OUTPUT_ROOT

OUTPUTS = OUTPUT_ROOT
PNG_METADATA = {"Software": "xenum"}

def atomic_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)

def atomic_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.png")
    fig.savefig(tmp, dpi=180, metadata=PNG_METADATA)
    tmp.replace(path)

def read_benchmarks():
    rows = []

    for path in sorted(OUTPUTS.glob("*/bench_xy.csv")):
        df = pd.read_csv(path)

        if "dataset" not in df.columns:
            df["dataset"] = path.parent.name

        rows.append(df)

    if not rows:
        raise SystemExit("no bench_xy.csv files found")

    return normalize_benchmark_rows(pd.concat(rows, ignore_index=True, sort=False))

def selected_rows(df):
    parts = []

    for _dataset, group in df.groupby("dataset", sort=False):
        selected = selected_benchmark_rows(group)

        if not selected.empty:
            parts.append(selected)

    if not parts:
        return pd.DataFrame(columns=df.columns)

    return pd.concat(parts, ignore_index=True, sort=False)

def summary_table(selected):
    clean = selected.dropna(subset=["median_xy_error"]).copy()

    if clean.empty:
        return pd.DataFrame()

    clean["rank"] = clean.groupby("dataset")["median_xy_error"].rank(method="min")

    return (
        clean
        .groupby(["measurement"], dropna=False)
        .agg(
            datasets=("dataset", "nunique"),
            leaky=("leaky", "max"),
            wins=("rank", lambda x: int((x == 1).sum())),
            rank_mean=("rank", "mean"),
            mean_xy_error_median=("mean_xy_error", "median"),
            median_xy_error_median=("median_xy_error", "median"),
            p90_xy_error_median=("p90_xy_error", "median"),
            coverage_mean=("coverage", "mean"),
        )
        .reset_index()
        .sort_values(["wins", "rank_mean", "median_xy_error_median"], ascending=[False, True, True])
    )

def best_table(selected):
    clean = selected.dropna(subset=["median_xy_error"]).copy()

    if clean.empty:
        return pd.DataFrame()

    return (
        clean
        .sort_values(["dataset", "median_xy_error", "mean_xy_error", "p90_xy_error", "k", "seed"], na_position="last")
        .groupby("dataset")
        .head(1)
        [[
            "dataset",
            "measurement",
            "k",
            "seed",
            "leaky",
            "coverage",
            "mean_xy_error",
            "median_xy_error",
            "p90_xy_error",
        ]]
    )

def matrix(selected, metric):
    clean = selected.dropna(subset=[metric]).copy()

    if clean.empty:
        return pd.DataFrame()

    out = clean.pivot_table(index="measurement", columns="dataset", values=metric, aggfunc="first")
    return out.reset_index()

def render_boxplot(selected):
    clean = selected.dropna(subset=["median_xy_error"]).copy()

    if clean.empty:
        return None

    order = (
        clean
        .groupby("measurement")["median_xy_error"]
        .median()
        .sort_values()
        .index
        .tolist()
    )
    data = [
        clean[clean["measurement"].eq(measurement)]["median_xy_error"].to_numpy(dtype=float)
        for measurement in order
    ]
    leaky = {
        measurement: bool(clean[clean["measurement"].eq(measurement)]["leaky"].fillna(False).astype(bool).any())
        for measurement in order
    }

    width = max(10.0, min(24.0, 0.45 * len(order) + 5.0))
    fig, ax = plt.subplots(figsize=(width, 5.5))
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)

    for measurement, box in zip(order, bp["boxes"]):
        box.set_facecolor("#4C78A8")
        box.set_alpha(0.7)

        if leaky[measurement]:
            box.set_hatch("//")
            box.set_edgecolor("#222222")
            box.set_linewidth(0.9)

    rng = np.random.default_rng(0)
    for i, measurement in enumerate(order, start=1):
        vals = clean[clean["measurement"].eq(measurement)]["median_xy_error"].to_numpy(dtype=float)
        jitter = rng.normal(0.0, 0.035, size=len(vals))
        ax.scatter(np.full(len(vals), i, dtype=float) + jitter, vals, s=16, color="#111111", alpha=0.75, zorder=3)

    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_ylabel("median xy error")
    ax.set_title("Global measurement comparison")
    ax.grid(axis="y", alpha=0.2)

    if any(leaky.values()):
        from matplotlib.patches import Patch

        ax.legend(handles=[Patch(facecolor="#4C78A8", edgecolor="#222222", hatch="//", label="leaky")], loc="best")

    fig.tight_layout()
    return fig

def main():
    all_rows = read_benchmarks()
    selected = selected_rows(all_rows)
    summary = summary_table(selected)
    best = best_table(selected)

    atomic_csv(all_rows, OUTPUTS / "bench_xy_all.csv")
    atomic_csv(best, OUTPUTS / "bench_xy_best.csv")
    atomic_csv(summary, OUTPUTS / "bench_xy_summary.csv")
    atomic_csv(matrix(selected, "mean_xy_error"), OUTPUTS / "benchmark_matrix_mean.csv")
    atomic_csv(matrix(selected, "median_xy_error"), OUTPUTS / "benchmark_matrix_median.csv")
    atomic_csv(matrix(selected, "p90_xy_error"), OUTPUTS / "benchmark_matrix_p90.csv")

    fig = render_boxplot(selected)
    if fig is not None:
        atomic_figure(fig, OUTPUTS / "benchmark_boxplot.png")
        plt.close(fig)

    if not summary.empty:
        print(summary.to_string(index=False))
    print()
    if not best.empty:
        print(best.to_string(index=False))

if __name__ == "__main__":
    main()
