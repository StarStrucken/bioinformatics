#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from xenum_paths import existing_out_dir

LEAKY = {"spatial", "mix", "expr_morph_spatial"}

def measurements(out_dir):
    out = []
    for p in sorted(out_dir.glob("edges_*.csv")):
        m = p.stem.removeprefix("edges_")
        if (out_dir / f"nodes_{m}.csv").exists():
            out.append(m)
    return out

def pred_from_edges(nodes, edges):
    xy = nodes[["x_centroid", "y_centroid"]].to_numpy(float)
    n = len(nodes)

    sx = np.zeros(n, dtype=float)
    sy = np.zeros(n, dtype=float)
    sw = np.zeros(n, dtype=float)
    cnt = np.zeros(n, dtype=np.int64)
    weighted = "neighbor_weight" in edges.columns

    for r in edges.itertuples(index=False):
        a = int(r.source)
        b = int(r.target)
        w = float(r.neighbor_weight) if weighted else 1.0

        if not np.isfinite(w) or w <= 0:
            continue

        sx[a] += w * xy[b, 0]
        sy[a] += w * xy[b, 1]
        sw[a] += w
        cnt[a] += 1

        sx[b] += w * xy[a, 0]
        sy[b] += w * xy[a, 1]
        sw[b] += w
        cnt[b] += 1

    ok = sw > 0
    pred = np.zeros_like(xy)
    pred[ok, 0] = sx[ok] / sw[ok]
    pred[ok, 1] = sy[ok] / sw[ok]

    err = np.sqrt(((pred[ok] - xy[ok]) ** 2).sum(axis=1))
    return ok, err

def pred_from_prediction_file(path):
    pred = pd.read_csv(path)
    err = pred["error"].to_numpy(float)
    ok = np.isfinite(err)
    return ok, err[ok]

def summarize(dataset_id):
    out_dir = existing_out_dir(dataset_id)
    rows = []

    base_nodes = pd.read_csv(out_dir / "nodes.csv")
    xy = base_nodes[["x_centroid", "y_centroid"]].to_numpy(float)
    center = xy.mean(axis=0)
    center_err = np.sqrt(((xy - center) ** 2).sum(axis=1))

    for m in measurements(out_dir):
        nodes = pd.read_csv(out_dir / f"nodes_{m}.csv")
        edges = pd.read_csv(out_dir / f"edges_{m}.csv")

        learned_pred_path = out_dir / "predictions_learned_mix_k0.csv"

        if m == "learned_mix" and learned_pred_path.exists():
            ok, err = pred_from_prediction_file(learned_pred_path)
        else:
            ok, err = pred_from_edges(nodes, edges)

        rows.append({
            "dataset": dataset_id,
            "measurement": m,
            "leaky": m in LEAKY,
            "n_nodes": int(len(nodes)),
            "n_edges": int(len(edges)),
            "coverage": float(ok.mean()),
            "mean_xy_error": float(err.mean()) if len(err) else None,
            "median_xy_error": float(np.median(err)) if len(err) else None,
            "p90_xy_error": float(np.quantile(err, 0.90)) if len(err) else None,
            "center_median_error": float(np.median(center_err)),
            "median_vs_center": float(np.median(err) / np.median(center_err)) if len(err) else None,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["leaky", "median_xy_error"], ascending=[True, True])
    df.to_csv(out_dir / "bench_xy.csv", index=False)
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_id")
    args = p.parse_args()

    df = summarize(args.dataset_id)

    cols = [
        "measurement",
        "leaky",
        "n_edges",
        "coverage",
        "median_xy_error",
        "p90_xy_error",
        "median_vs_center",
    ]

    print(df[cols].to_string(index=False))

if __name__ == "__main__":
    main()
