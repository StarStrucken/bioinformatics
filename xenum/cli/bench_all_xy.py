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

def benchmark_plot_rows(benchmarks):
    clean = benchmarks.copy()
    if "status" in clean.columns:
        clean = clean[clean["status"].eq("ok")].copy()
    clean = clean.dropna(subset=["median_xy_error"]).copy()

    if clean.empty:
        return clean

    clean["dataset"] = clean["dataset"].astype(str)
    clean["measurement"] = clean["measurement"].astype(str)
    clean["median_xy_error"] = pd.to_numeric(clean["median_xy_error"], errors="coerce")

    if "spatial_best_median_xy_error" not in clean.columns:
        spatial_best = (
            clean[clean["measurement"].eq("spatial")]
            .groupby("dataset")["median_xy_error"]
            .min()
        )
        clean["spatial_best_median_xy_error"] = clean["dataset"].map(spatial_best)

    clean["spatial_best_median_xy_error"] = pd.to_numeric(
        clean["spatial_best_median_xy_error"],
        errors="coerce",
    )
    clean["normalized_error"] = np.divide(
        clean["median_xy_error"],
        clean["spatial_best_median_xy_error"],
        out=np.full(len(clean), np.nan, dtype=float),
        where=clean["spatial_best_median_xy_error"].to_numpy(dtype=float) > 0,
    )
    clean["improvement_pct"] = 100.0 * (1.0 - clean["normalized_error"])

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


def render_grouped_boxplot(
    benchmarks,
    *,
    metric,
    ylabel,
    title,
    order,
    datasets,
    dataset_colors,
    yscale=None,
    reference_line=None,
):
    clean = benchmarks.dropna(subset=[metric]).copy()

    if yscale == "log":
        clean = clean[clean[metric] > 0].copy()

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
    box_width = 0.11 if len(datasets) > 8 else 0.13
    group_stride = max(1.25, (len(datasets) - 1) * dataset_step + 0.75)
    centers = np.arange(len(order), dtype=float) * group_stride
    offsets = (np.arange(len(datasets), dtype=float) - (len(datasets) - 1) / 2.0) * dataset_step

    data = []
    positions = []
    box_meta = []
    point_meta = []
    for measurement_index, measurement in enumerate(order):
        center = centers[measurement_index]

        for dataset_index, dataset in enumerate(datasets):
            group = clean[
                clean["measurement"].eq(measurement)
                & clean["dataset"].eq(dataset)
            ].copy()

            if "k" in group.columns:
                group["_k_sort"] = pd.to_numeric(group["k"], errors="coerce")
                sort_cols = ["_k_sort"]
                if "seed" in group.columns:
                    group["_seed_sort"] = pd.to_numeric(group["seed"], errors="coerce")
                    sort_cols.append("_seed_sort")
                group = group.sort_values(sort_cols, na_position="last")

            vals = group[metric].to_numpy(dtype=float)

            if vals.size == 0:
                continue

            position = center + offsets[dataset_index]
            point_meta.append((position, measurement, dataset, vals))

            if vals.size > 1:
                data.append(vals)
                positions.append(position)
                box_meta.append((measurement, dataset, vals))

    width = max(11.0, min(36.0, 0.62 * len(order) + 0.22 * len(datasets) + 5.0))
    fig, ax = plt.subplots(figsize=(width, 5.5))

    if data:
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        )

        for i, (measurement, dataset, _vals) in enumerate(box_meta):
            box = bp["boxes"][i]
            box.set_facecolor(dataset_colors[dataset])
            box.set_alpha(0.62)
            box.set_edgecolor("#222222")
            box.set_linewidth(0.8)

            if leaky[(measurement, dataset)]:
                box.set_hatch("//")

            for whisker in bp["whiskers"][2 * i: 2 * i + 2]:
                whisker.set_color("#333333")
                whisker.set_linewidth(0.75)
            for cap in bp["caps"][2 * i: 2 * i + 2]:
                cap.set_color("#333333")
                cap.set_linewidth(0.75)
            bp["medians"][i].set_color("#111111")
            bp["medians"][i].set_linewidth(1.0)

    for position, _measurement, dataset, vals in point_meta:
        if vals.size == 1:
            jitter = np.zeros(1, dtype=float)
            point_size = 20
        else:
            jitter = np.linspace(-box_width * 0.18, box_width * 0.18, vals.size)
            point_size = 14

        ax.scatter(
            np.full(len(vals), position, dtype=float) + jitter,
            vals,
            s=point_size,
            color=dataset_colors[dataset],
            edgecolor="#111111",
            linewidth=0.25,
            alpha=0.9,
            zorder=3,
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

    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=dataset_colors[dataset], edgecolor="#222222", label=dataset)
        for dataset in datasets
    ]
    if any(leaky.values()):
        handles.append(Patch(facecolor="#ffffff", edgecolor="#222222", hatch="//", label="leaky"))
    if reference_line is not None:
        from matplotlib.lines import Line2D

        handles.append(Line2D([0], [0], color="#333333", linestyle="--", linewidth=0.9, label=reference_line[1]))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

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

    plot_rows = benchmark_plot_rows(all_rows)
    if not plot_rows.empty:
        order = measurement_plot_order(plot_rows)
        datasets = sorted(plot_rows["dataset"].dropna().unique().tolist())
        dataset_colors = dataset_color_map(datasets)
        figures = [
            (
                "benchmark_boxplot.png",
                "median_xy_error",
                "median xy error (log scale)",
                "Absolute median xy error by dataset",
                "log",
                None,
            ),
            (
                "benchmark_boxplot_normalized_error.png",
                "normalized_error",
                "median xy error / spatial best median xy error",
                "Normalized median xy error by dataset",
                None,
                (1.0, "spatial best"),
            ),
            (
                "benchmark_boxplot_percent_improvement.png",
                "improvement_pct",
                "improvement vs spatial best (%)",
                "Percentage improvement over spatial best by dataset",
                None,
                (0.0, "spatial best"),
            ),
        ]

        for filename, metric, ylabel, title, yscale, reference_line in figures:
            fig = render_grouped_boxplot(
                plot_rows,
                metric=metric,
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
