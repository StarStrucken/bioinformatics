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
    cnt = np.zeros(n, dtype=np.int64)

    for r in edges.itertuples(index=False):
        a = int(r.source)
        b = int(r.target)

        sx[a] += xy[b, 0]
        sy[a] += xy[b, 1]
        cnt[a] += 1

        sx[b] += xy[a, 0]
        sy[b] += xy[a, 1]
        cnt[b] += 1

    ok = cnt > 0
    pred = np.zeros_like(xy)
    pred[ok, 0] = sx[ok] / cnt[ok]
    pred[ok, 1] = sy[ok] / cnt[ok]

    err = np.sqrt(((pred[ok] - xy[ok]) ** 2).sum(axis=1))
    return ok, err

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
