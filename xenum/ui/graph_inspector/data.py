from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from xenum_measurements import DEPRECATED_MEASUREMENTS, MEASUREMENTS, OPTIONAL_MEASUREMENTS, VISIBLE_MEASUREMENTS

from .config import FALLBACK_PREDICTION_K, SHOW_HIDDEN_MEASUREMENTS

def discover_measurements(out_dir: Path):
    found = []

    for edge_path in sorted(out_dir.glob("edges_*.csv")):
        m = edge_path.stem.removeprefix("edges_")

        if "_k" in m:
            continue

        if m in DEPRECATED_MEASUREMENTS:
            continue

        node_path = out_dir / f"nodes_{m}.csv"
        if node_path.exists():
            found.append(m)

    ordered = [m for m in VISIBLE_MEASUREMENTS if m in found]
    ordered.extend(m for m in OPTIONAL_MEASUREMENTS if m in found and m not in ordered)
    ordered.extend(m for m in found if m not in ordered and m not in MEASUREMENTS)

    if SHOW_HIDDEN_MEASUREMENTS:
        ordered.extend(m for m in MEASUREMENTS if m in found and m not in ordered)
        ordered.extend(m for m in found if m not in ordered)

    if not ordered:
        ordered = [m for m in MEASUREMENTS if m in found]

    if not ordered and (out_dir / "nodes_graph.csv").exists() and (out_dir / "edges.csv").exists():
        ordered = ["graph"]

    return ordered

def load_dump(out_dir: Path):
    nodes = {}
    edges = {}
    measurements = discover_measurements(out_dir)

    for m in measurements:
        if m == "graph":
            node_path = out_dir / "nodes_graph.csv"
            edge_path = out_dir / "edges.csv"
        else:
            node_path = out_dir / f"nodes_{m}.csv"
            edge_path = out_dir / f"edges_{m}.csv"

        nodes[m] = pd.read_csv(node_path)
        edges[m] = pd.read_csv(edge_path)

    if not measurements:
        raise FileNotFoundError(f"no nodes_*.csv / edges_*.csv in {out_dir}")

    return measurements, nodes, edges

def load_best_prediction_k(out_dir: Path):
    path = out_dir / "best_k_by_measurement.csv"

    if not path.exists():
        return {}

    df = pd.read_csv(path)
    out = {}

    for r in df.itertuples(index=False):
        out[str(r.measurement)] = int(r.k)

    return out

def load_best_k_table(out_dir: Path):
    path = out_dir / "best_k_by_measurement.csv"

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)

def build_neighbors(edges_by_measurement):
    out = {}

    for m, edges in edges_by_measurement.items():
        for r in edges.itertuples(index=False):
            s = int(r.source)
            t = int(r.target)

            nd = float(r.neighbor_distance)
            xy = float(getattr(r, "distance", getattr(r, "xy_distance", np.nan)))

            out.setdefault(s, {}).setdefault(m, []).append((t, nd, xy))
            out.setdefault(t, {}).setdefault(m, []).append((s, nd, xy))

    for node, by_m in out.items():
        for m, vals in by_m.items():
            vals.sort(key=lambda x: x[1])

    return out

def load_predictions(out_dir: Path, measurements, best_k):
    out = {}

    for m in measurements:
        k = int(best_k.get(m, FALLBACK_PREDICTION_K))
        path = out_dir / f"predictions_{m}_k{k}.csv"

        if path.exists():
            df = pd.read_csv(path)
            df.attrs["k"] = k
            out[m] = df

    return out

def prediction_k_text(measurements, best_k, predictions):
    rows = []

    for m in measurements:
        if m in predictions:
            k = int(predictions[m].attrs.get("k", FALLBACK_PREDICTION_K))
            mark = "" if m in best_k else " fallback"
            rows.append(f"{m:<12} k {k}{mark}")

    if not rows:
        return "no prediction files"

    return "\n".join(rows)
