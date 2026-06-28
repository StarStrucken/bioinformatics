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

    if not rows and (OUTPUTS / "bench_xy_all.csv").exists():
        rows.append(pd.read_csv(OUTPUTS / "bench_xy_all.csv"))

    if not rows:
        raise SystemExit("no per-dataset bench_xy.csv files or bench_xy_all.csv found")

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

def benchmark_plot_rows(selected):
    clean = selected.copy()

    if clean.empty:
        return clean

    clean["dataset"] = clean["dataset"].astype(str)
    clean["measurement"] = clean["measurement"].astype(str)

    for col in ["median_xy_error", "mean_xy_error", "p90_xy_error"]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    clean = clean.dropna(subset=["median_xy_error", "mean_xy_error", "p90_xy_error"]).copy()

    if clean.empty:
        return clean

    max_error = clean.groupby("dataset")["median_xy_error"].transform("max")
    min_error = clean.groupby("dataset")["median_xy_error"].transform("min")
    error_range = max_error - min_error

    for src, dst in [
        ("median_xy_error", "median_error_percent_max"),
        ("mean_xy_error", "mean_error_percent_max"),
        ("p90_xy_error", "p90_error_percent_max"),
    ]:
        clean[dst] = np.divide(
            clean[src] * 100.0,
            max_error,
            out=np.full(len(clean), np.nan, dtype=float),
            where=max_error.to_numpy(dtype=float) > 0,
        )

    for src, dst in [
        ("median_xy_error", "median_performance_score"),
        ("mean_xy_error", "mean_performance_score"),
        ("p90_xy_error", "p90_performance_score"),
    ]:
        clean[dst] = np.divide(
            (max_error - clean[src]) * 100.0,
            error_range,
            out=np.full(len(clean), np.nan, dtype=float),
            where=error_range.to_numpy(dtype=float) > 0,
        )

    return clean


def measurement_plot_order(clean):
    order = (
        clean
        .groupby("measurement")["median_xy_error"]
        .median()
        .sort_values()
        .index
        .tolist()
    )
    return order


def dataset_color_map(datasets):
    cmap = plt.get_cmap("tab20" if len(datasets) > 10 else "tab10")
    return {
        dataset: cmap(i % cmap.N)
        for i, dataset in enumerate(datasets)
    }


def render_grouped_barplot(
    benchmarks,
    *,
    median_col,
    mean_col,
    p90_col,
    ylabel,
    title,
    order,
    datasets,
    dataset_colors,
    yscale=None,
    reference_line=None,
):
    clean = benchmarks.dropna(subset=[median_col, mean_col, p90_col]).copy()

    if yscale == "log":
        clean = clean[clean[median_col] > 0].copy()

    if clean.empty:
        return None

    order = [measurement for measurement in order if clean["measurement"].eq(measurement).any()]
    if not order or not datasets:
        return None

    leaky = {
        (measurement, dataset): bool(
            clean[
                clean["measurement"].eq(measurement)
                & clean["dataset"].eq(dataset)
            ]["leaky"].fillna(False).astype(bool).any()
        )
        for measurement in order
        for dataset in datasets
    }

    dataset_step = 0.18
    bar_width = 0.11 if len(datasets) > 8 else 0.13
    group_stride = max(1.25, (len(datasets) - 1) * dataset_step + 0.75)
    centers = np.arange(len(order), dtype=float) * group_stride
    offsets = (np.arange(len(datasets), dtype=float) - (len(datasets) - 1) / 2.0) * dataset_step

    positions = []
    row_meta = []
    for measurement_index, measurement in enumerate(order):
        center = centers[measurement_index]

        for dataset_index, dataset in enumerate(datasets):
            group = clean[
                clean["measurement"].eq(measurement)
                & clean["dataset"].eq(dataset)
            ].copy()

            if group.empty:
                continue

            position = center + offsets[dataset_index]
            positions.append(position)
            row_meta.append((measurement, dataset, group.iloc[0]))

    width = max(11.0, min(36.0, 0.62 * len(order) + 0.22 * len(datasets) + 5.0))
    fig, ax = plt.subplots(figsize=(width, 5.5))

    medians = np.asarray([float(row[median_col]) for _measurement, _dataset, row in row_meta], dtype=float)
    means = np.asarray([float(row[mean_col]) for _measurement, _dataset, row in row_meta], dtype=float)
    p90s = np.asarray([float(row[p90_col]) for _measurement, _dataset, row in row_meta], dtype=float)

    bars = ax.bar(
        positions,
        medians,
        width=bar_width,
        color=[dataset_colors[dataset] for _measurement, dataset, _row in row_meta],
        edgecolor="#222222",
        linewidth=0.6,
        alpha=0.78,
        zorder=2,
    )

    for bar, (measurement, dataset, _row) in zip(bars, row_meta):
        if leaky[(measurement, dataset)]:
            bar.set_hatch("//")
            bar.set_linewidth(0.8)

    lower = np.maximum(0.0, medians - p90s)
    upper = np.maximum(0.0, p90s - medians)
    ax.errorbar(
        positions,
        medians,
        yerr=np.vstack([lower, upper]),
        fmt="none",
        color="#222222",
        linewidth=0.8,
        capsize=2,
        zorder=3,
    )
    ax.scatter(
        positions,
        means,
        marker="D",
        s=20,
        color="#111111",
        edgecolor="#111111",
        linewidth=0.25,
        zorder=4,
    )

    for i in range(len(centers) - 1):
        ax.axvline((centers[i] + centers[i + 1]) / 2.0, color="#dddddd", linewidth=0.7, zorder=0)

    if yscale:
        ax.set_yscale(yscale)
    if reference_line is not None:
        y, label = reference_line
        ax.axhline(y, color="#333333", linestyle="--", linewidth=0.9, label=label, zorder=1)

    ax.set_xlim(centers[0] - group_stride * 0.55, centers[-1] + group_stride * 0.55)
    ax.set_xticks(centers)
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=dataset_colors[dataset], edgecolor="#222222", label=dataset)
        for dataset in datasets
    ]
    handles.append(Line2D([0], [0], marker="D", color="none", markerfacecolor="#111111", markeredgecolor="#111111", markersize=5, label="mean"))
    handles.append(Line2D([0], [0], color="#222222", linewidth=0.8, marker="_", markersize=5, label="p90"))
    if any(leaky.values()):
        handles.append(Patch(facecolor="#ffffff", edgecolor="#222222", hatch="//", label="leaky"))
    if reference_line is not None:
        handles.append(Line2D([0], [0], color="#333333", linestyle="--", linewidth=0.9, label=reference_line[1]))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

    fig.tight_layout()
    return fig


def remove_obsolete_figures():
    for name in [
        "benchmark_boxplot_normalized_error.png",
        "benchmark_boxplot_percent_improvement.png",
    ]:
        path = OUTPUTS / name
        if path.exists():
            path.unlink()

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

    remove_obsolete_figures()

    plot_rows = benchmark_plot_rows(selected)
    if not plot_rows.empty:
        order = measurement_plot_order(plot_rows)
        datasets = sorted(plot_rows["dataset"].dropna().unique().tolist())
        dataset_colors = dataset_color_map(datasets)
        figures = [
            (
                "benchmark_boxplot.png",
                "median_xy_error",
                "mean_xy_error",
                "p90_xy_error",
                "xy error",
                "Selected median xy error by dataset",
                None,
                None,
            ),
            (
                "benchmark_boxplot_error_percent_max.png",
                "median_error_percent_max",
                "mean_error_percent_max",
                "p90_error_percent_max",
                "error (% of dataset maximum)",
                "Selected error as percent of dataset maximum",
                None,
                (100.0, "worst median"),
            ),
            (
                "benchmark_boxplot_performance_score.png",
                "median_performance_score",
                "mean_performance_score",
                "p90_performance_score",
                "min-max performance score (%)",
                "Selected min-max performance score by dataset",
                None,
                (0.0, "worst median"),
            ),
        ]

        for filename, median_col, mean_col, p90_col, ylabel, title, yscale, reference_line in figures:
            fig = render_grouped_barplot(
                plot_rows,
                median_col=median_col,
                mean_col=mean_col,
                p90_col=p90_col,
                ylabel=ylabel,
                title=title,
                order=order,
                datasets=datasets,
                dataset_colors=dataset_colors,
                yscale=yscale,
                reference_line=reference_line,
            )
            if fig is not None:
                atomic_figure(fig, OUTPUTS / filename)
                plt.close(fig)

    if not summary.empty:
        print(summary.to_string(index=False))
    print()
    if not best.empty:
        print(best.to_string(index=False))

if __name__ == "__main__":
    main()
