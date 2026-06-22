#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from xenum_paths import existing_out_dir

FIG_DPI = 180
MAX_LINE_SEGMENTS = 30_000
MAX_SCATTER_POINTS = 100_000
PREDICTION_FIGSIZE = (8, 8)
BENCHMARK_FIGSIZE = (10, 5)
PNG_METADATA = {
    "Software": "xenum",
}
BASELINE_COLORS = {
    "core": "#4C78A8",
    "baseline": "#F58518",
}
REPORT_EXAMPLE_MEASUREMENTS = (
    "morphology_image",
    "morphology_image_summary",
    "morphology_image_histogram",
    "morphology_image_texture",
    "morphology_image_all",
)

PREDICTION_ALPHA = 0.85
REAL_ALPHA = 0.25
LINE_ALPHA = 0.18

USE_GLOBAL_ERROR_SCALE = True

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_id")
    return p.parse_args()

def report_dirs(out_dir: Path):
    report_dir = out_dir / "reports"
    table_dir = report_dir / "tables"
    figure_dir = report_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return report_dir, table_dir, figure_dir

def read_best_k(out_dir: Path, table_dir: Path):
    path = table_dir / "best_k_by_measurement.csv"

    if not path.exists():
        path = out_dir / "best_k_by_measurement.csv"

    if not path.exists():
        raise FileNotFoundError("best_k_by_measurement.csv not found")

    return pd.read_csv(path)

def prediction_path_exists(out_dir: Path, measurement: str, k: int):
    return (out_dir / f"predictions_{measurement}_k{int(k)}.csv").exists()

def add_report_examples(out_dir: Path, best_df: pd.DataFrame):
    bench_path = out_dir / "bench_xy.csv"

    if not bench_path.exists():
        return best_df

    bench = pd.read_csv(bench_path)

    if bench.empty or "measurement" not in bench.columns or "k" not in bench.columns:
        return best_df

    existing = set(best_df["measurement"].astype(str)) if "measurement" in best_df.columns else set()
    rows = []

    for measurement in REPORT_EXAMPLE_MEASUREMENTS:
        if measurement in existing:
            continue

        cand = bench[bench["measurement"].astype(str).eq(measurement)].copy()

        if cand.empty:
            continue

        if "leaky" in cand.columns:
            cand = cand[~cand["leaky"].astype(bool)]

        cand = cand.dropna(subset=["median_xy_error"])

        if cand.empty:
            continue

        cand = cand.sort_values(["median_xy_error", "p90_xy_error", "k"], na_position="last")

        for row in cand.to_dict("records"):
            if prediction_path_exists(out_dir, measurement, int(row["k"])):
                rows.append(row)
                existing.add(measurement)
                break

    if not rows:
        return best_df

    return pd.concat([best_df, pd.DataFrame(rows)], ignore_index=True, sort=False)

def row_source(row):
    source = getattr(row, "source", None)

    if source is None or pd.isna(source):
        return "core"

    return str(source)

def row_label(row):
    label = getattr(row, "report_label", None)

    if label is not None and not pd.isna(label):
        return str(label)

    measurement = str(row.measurement)

    baseline = getattr(row, "baseline", None)
    if baseline is not None and not pd.isna(baseline) and str(baseline):
        return f"{baseline}:{measurement}"

    return measurement

def prediction_path_for_row(out_dir: Path, row):
    path = getattr(row, "prediction_path", None)

    if path is not None and not pd.isna(path) and str(path):
        return Path(path)

    return out_dir / f"predictions_{row.measurement}_k{int(row.k)}.csv"

def baseline_prediction_path(baseline_dir: Path, measurement: str, k: int):
    return baseline_dir / f"predictions_{measurement}_k{int(k)}.csv"

def read_baseline_rows(out_dir: Path):
    rows = []

    for path in sorted((out_dir / "baselines").glob("*/bench_xy.csv")):
        baseline = path.parent.name
        bench = pd.read_csv(path)

        if bench.empty:
            continue

        for row in bench.to_dict("records"):
            measurement = str(row.get("measurement", baseline))
            k = int(row.get("k", 0))
            pred_path = baseline_prediction_path(path.parent, measurement, k)

            row["source"] = "baseline"
            row["baseline"] = str(row.get("baseline") or baseline)
            row["report_label"] = f"{row['baseline']}:{measurement}"
            row["prediction_path"] = str(pred_path)
            row["figure_name"] = f"prediction_{row['baseline']}_{measurement}.png"

            rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)

def add_baseline_rows(out_dir: Path, best_df: pd.DataFrame):
    baseline_df = read_baseline_rows(out_dir)

    if baseline_df.empty:
        return best_df

    core = best_df.copy()

    if "source" not in core.columns:
        core["source"] = "core"

    if "baseline" not in core.columns:
        core["baseline"] = ""

    if "report_label" not in core.columns:
        core["report_label"] = core["measurement"].astype(str)
    else:
        core["report_label"] = core["report_label"].fillna("")
        core.loc[core["report_label"].astype(str).eq(""), "report_label"] = core["measurement"].astype(str)

    if "prediction_path" not in core.columns:
        core["prediction_path"] = ""

    if "figure_name" not in core.columns:
        core["figure_name"] = ""

    return pd.concat([core, baseline_df], ignore_index=True, sort=False)

def sample_indices(n, max_n, seed=0):
    if n <= max_n:
        return np.arange(n, dtype=int)

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_n, replace=False))

def finite_prediction_mask(df):
    cols = ["x", "y", "pred_x", "pred_y", "error"]
    ok = np.ones(len(df), dtype=bool)

    for c in cols:
        ok &= np.isfinite(df[c].to_numpy(dtype=float))

    return ok

def prediction_title(measurement, k, df, label=None):
    ok = finite_prediction_mask(df)
    err = df.loc[ok, "error"].to_numpy(dtype=float)
    title = str(label or measurement)

    if len(err) == 0:
        return f"{title} k={k}"

    median = float(np.median(err))
    p90 = float(np.quantile(err, 0.90))
    coverage = float(ok.mean())

    return f"{title} k={k}  median={median:.2f}  p90={p90:.2f}  coverage={coverage:.3f}"

def render_prediction(out_dir: Path, figure_dir: Path, measurement: str, k: int, error_limits=None, prediction_path=None, figure_name=None, label=None):
    path = Path(prediction_path) if prediction_path else out_dir / f"predictions_{measurement}_k{k}.csv"

    if not path.exists():
        print(f"skip {measurement}: missing {path}", flush=True)
        return None

    df = pd.read_csv(path)
    ok = finite_prediction_mask(df)
    idx_all = np.flatnonzero(ok)

    if len(idx_all) == 0:
        print(f"skip {measurement}: no finite predictions", flush=True)
        return None

    idx_scatter = idx_all[sample_indices(len(idx_all), MAX_SCATTER_POINTS)]
    idx_lines = idx_all[sample_indices(len(idx_all), MAX_LINE_SEGMENTS)]

    x = df["x"].to_numpy(dtype=float)
    y = -df["y"].to_numpy(dtype=float)
    px = df["pred_x"].to_numpy(dtype=float)
    py = -df["pred_y"].to_numpy(dtype=float)
    err = df["error"].to_numpy(dtype=float)

    segments = np.stack(
        [
            np.column_stack([x[idx_lines], y[idx_lines]]),
            np.column_stack([px[idx_lines], py[idx_lines]]),
        ],
        axis=1,
    )

    fig, ax = plt.subplots(figsize=PREDICTION_FIGSIZE)

    lc = LineCollection(
        segments,
        linewidths=0.35,
        alpha=LINE_ALPHA,
    )
    ax.add_collection(lc)

    ax.scatter(
        x[idx_scatter],
        y[idx_scatter],
        s=2,
        alpha=REAL_ALPHA,
        label="real",
    )

    finite_err = err[idx_scatter]
    if error_limits is None:
        lo = float(np.quantile(finite_err, 0.05))
        hi = float(np.quantile(finite_err, 0.95))
    else:
        lo, hi = error_limits

    if hi <= lo:
        hi = lo + 1.0

    sc = ax.scatter(
        px[idx_scatter],
        py[idx_scatter],
        c=np.clip(err[idx_scatter], lo, hi),
        s=5,
        alpha=PREDICTION_ALPHA,
        label="predicted",
    )

    ax.set_title(prediction_title(measurement, k, df, label=label))
    ax.set_xlabel("x")
    ax.set_ylabel("y * -1")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", markerscale=3)
    fig.colorbar(sc, ax=ax, label="prediction error")

    out_path = figure_dir / (figure_name or f"prediction_{measurement}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, metadata=PNG_METADATA)
    plt.close(fig)

    print(f"rendered: {out_path}", flush=True)
    return out_path

def render_benchmark_best(
    table_dir: Path,
    figure_dir: Path,
    best_df: pd.DataFrame,
    *,
    filename="benchmark_best_k.png",
    title="Best measurement/baseline: median with p90 whisker",
):
    if best_df.empty:
        return None

    df = best_df.copy()
    df = df.dropna(subset=["median_xy_error"])

    if df.empty:
        return None

    df = df.sort_values(["median_xy_error", "measurement"])

    labels = [f"{row_label(r)}\nk={int(r.k)}" for r in df.itertuples(index=False)]
    vals = df["median_xy_error"].to_numpy(dtype=float)
    colors = [
        BASELINE_COLORS["baseline"] if row_source(r) == "baseline" else BASELINE_COLORS["core"]
        for r in df.itertuples(index=False)
    ]

    fig, ax = plt.subplots(figsize=BENCHMARK_FIGSIZE)
    ax.bar(np.arange(len(df)), vals, color=colors)
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("xy error")
    ax.set_title(title)

    if "p90_xy_error" in df.columns:
        p90 = df["p90_xy_error"].to_numpy(dtype=float)
        upper = np.maximum(0.0, p90 - vals)
        lower = np.zeros_like(upper)

        ax.errorbar(
            np.arange(len(df)),
            vals,
            yerr=np.vstack([lower, upper]),
            fmt="none",
            linewidth=0.8,
            capsize=2,
        )

    out_path = figure_dir / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, metadata=PNG_METADATA)
    plt.close(fig)

    print(f"rendered: {out_path}", flush=True)
    return out_path

def write_report_overview(table_dir: Path, best_df: pd.DataFrame):
    cols = [
        "source",
        "baseline",
        "report_label",
        "measurement",
        "k",
        "leaky",
        "median_xy_error",
        "p90_xy_error",
        "coverage",
        "pred_spread_ratio",
        "median_vs_spatial_best",
        "median_vs_spatial_same_k",
        "spatial_best_k",
        "spatial_best_median_xy_error",
    ]

    cols = [c for c in cols if c in best_df.columns]

    if not cols:
        return None

    df = best_df[cols].copy()

    if "median_xy_error" in df.columns:
        df = df.sort_values(["median_xy_error", "measurement"])

    out_path = table_dir / "report_overview.csv"
    round_cols = [
        "median_xy_error",
        "p90_xy_error",
        "coverage",
        "pred_spread_ratio",
        "median_vs_spatial_best",
        "median_vs_spatial_same_k",
        "spatial_best_median_xy_error",
    ]

    for c in round_cols:
        if c in df.columns:
            df[c] = df[c].round(4)
    df.to_csv(out_path, index=False)

    print(f"written: {out_path}", flush=True)
    return out_path

def global_error_limits(out_dir: Path, best_df: pd.DataFrame):
    vals = []

    for r in best_df.itertuples(index=False):
        path = prediction_path_for_row(out_dir, r)

        if not path.exists():
            continue

        df = pd.read_csv(path)
        ok = finite_prediction_mask(df)
        err = df.loc[ok, "error"].to_numpy(dtype=float)

        if len(err):
            vals.append(err)

    if not vals:
        return None

    err = np.concatenate(vals)
    return float(np.quantile(err, 0.05)), float(np.quantile(err, 0.95))

def main():
    args = parse_args()

    out_dir = existing_out_dir(args.dataset_id)
    _, table_dir, figure_dir = report_dirs(out_dir)

    best_df = read_best_k(out_dir, table_dir)
    best_df = add_report_examples(out_dir, best_df)
    best_df = add_baseline_rows(out_dir, best_df)
    write_report_overview(table_dir, best_df)

    rendered = []

    error_limits = global_error_limits(out_dir, best_df) if USE_GLOBAL_ERROR_SCALE else None

    for r in best_df.itertuples(index=False):
        measurement = str(r.measurement)
        k = int(r.k)

        prediction_path = prediction_path_for_row(out_dir, r)
        figure_name = getattr(r, "figure_name", None)
        if figure_name is not None and (pd.isna(figure_name) or not str(figure_name)):
            figure_name = None

        path = render_prediction(
            out_dir,
            figure_dir,
            measurement,
            k,
            error_limits=error_limits,
            prediction_path=prediction_path,
            figure_name=figure_name,
            label=row_label(r),
        )
        if path is not None:
            rendered.append(path)

    bench_path = render_benchmark_best(table_dir, figure_dir, best_df)
    if bench_path is not None:
        rendered.append(bench_path)

    comparison = best_df.copy()
    if not comparison.empty:
        if "leaky" in comparison.columns:
            leaky = comparison["leaky"].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
        else:
            leaky = pd.Series(False, index=comparison.index)

        source = comparison["source"].fillna("core").astype(str) if "source" in comparison.columns else pd.Series("core", index=comparison.index)
        baseline = comparison["baseline"].fillna("").astype(str) if "baseline" in comparison.columns else pd.Series("", index=comparison.index)
        keep = (~leaky) | (source.eq("baseline") & baseline.eq("spagcn"))
        comparison = comparison[keep].copy()

    comparison_path = render_benchmark_best(
        table_dir,
        figure_dir,
        comparison,
        filename="benchmark_spagcn_vs_core.png",
        title="SpaGCN vs best non-leaky methods: median with p90 whisker",
    )
    if comparison_path is not None:
        rendered.append(comparison_path)

    print(f"figures rendered: {len(rendered)}", flush=True)
    print(f"figures dir: {figure_dir}", flush=True)

if __name__ == "__main__":
    main()
