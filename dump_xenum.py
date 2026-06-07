#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.neighbors import NearestNeighbors

from xenum_common import zscore, parse_seq_ids, sequence_distance
from xenum_measurements import ACTIVE_MEASUREMENTS, HIDDEN_MEASUREMENTS, LEAKY_MEASUREMENTS, MEASUREMENTS, VISIBLE_MEASUREMENTS
from xenum_paths import data_dir, out_dir as make_out_dir

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

K = 4
BENCH_K_VALUES = (1, 2, 3, 4, 5, 8, 12, 16, 24, 32)
EXPRESSION_PCS = 30
LEARNED_MIX_NAME = "learned_mix"
LEARNED_BASE_MEASUREMENTS = ("expression", "morphology", "seq_jaccard")
LEARNED_WEIGHT_VALUES = (0.0, 0.25, 0.5, 1.0, 2.0)
LEARNED_MIN_COVERAGE = 0.95
LEARNED_SCORE_MODE = "median_p90"
LEARNED_P90_WEIGHT = 0.25

USE_NEIGHBOR_CUTOFF = True
CUTOFF_QUANTILE = 0.995
CUTOFF_MAD = 8.0
MIN_EDGES_PER_NODE = 0
TOP_GENES_PER_CELL = 32

CELL_TABLE_NAMES = ("cells.csv.gz", "cells.csv", "cells.parquet")

NODE_BASE_COLS = [
    "cell_id",
    "x_centroid",
    "y_centroid",
    "x_norm",
    "y_norm",
    "log_total_counts",
    "log_detected_genes",
    "cell_area",
    "nucleus_area",
    "nucleus_cell_ratio",
]

EDGE_COLS = [
    "measurement",
    "source",
    "target",
    "neighbor_distance",
    "xy_distance",
    "expression_distance",
    "morphology_distance",
    "source_cell_id",
    "target_cell_id",
    "source_component_id",
    "target_component_id",
    "source_component_size",
    "target_component_size",
    "same_component",
]

def find_file(root: Path, names: tuple[str, ...]) -> Path:
    roots = [root] if root.name == "bundle" else [root, root / "bundle"]
    for base in roots:
        for name in names:
            p = base / name
            if p.exists():
                return p
    for base in roots:
        for name in names:
            for p in base.rglob(name):
                return p
    raise FileNotFoundError(names)

def read_cells(xenium_dir: Path):
    path = find_file(xenium_dir, CELL_TABLE_NAMES)
    cells = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    cells["cell_id"] = cells["cell_id"].astype(str)
    return cells.drop_duplicates("cell_id").set_index("cell_id"), path

def load_xenium(xenium_dir: Path):
    import scanpy as sc
    matrix_path = find_file(xenium_dir, ("cell_feature_matrix.h5",))
    adata = sc.read_10x_h5(matrix_path, gex_only=False)
    adata.var_names_make_unique()
    adata.obs_names = adata.obs_names.astype(str)
    if "feature_types" in adata.var.columns:
        adata = adata[:, adata.var["feature_types"].astype(str).eq("Gene Expression")].copy()
    cells, cells_path = read_cells(xenium_dir)
    ids = adata.obs_names.intersection(cells.index)
    adata = adata[ids].copy()
    cells = cells.loc[adata.obs_names]
    coords = cells[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
    adata.obsm["spatial"] = coords
    adata.obs["cell_id"] = adata.obs_names.astype(str)
    adata.obs["x_centroid"] = coords[:, 0]
    adata.obs["y_centroid"] = coords[:, 1]
    adata.obs["cell_area"] = cells["cell_area"].to_numpy(dtype=np.float32) if "cell_area" in cells else np.zeros(adata.n_obs, dtype=np.float32)
    adata.obs["nucleus_area"] = cells["nucleus_area"].to_numpy(dtype=np.float32) if "nucleus_area" in cells else np.zeros(adata.n_obs, dtype=np.float32)
    adata.uns["xenium_clean"] = {"xenium_dir": str(xenium_dir), "matrix_path": str(matrix_path), "cells_path": str(cells_path)}
    return adata

def clean_float(x):
    x = np.asarray(x, dtype=np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def per_cell_counts(x):
    total = np.asarray(x.sum(axis=1)).ravel().astype(np.float32)
    detected = np.asarray(x.getnnz(axis=1) if sp.issparse(x) else (x > 0).sum(axis=1)).ravel().astype(np.float32)
    return clean_float(total), clean_float(detected)

def top_gene_ids(x, n=32):
    out = []
    x = x.tocsr() if sp.issparse(x) else np.asarray(x)

    for i in range(x.shape[0]):
        row = x.getrow(i) if sp.issparse(x) else x[i]
        if sp.issparse(row):
            idx = row.indices
            val = row.data
        else:
            idx = np.flatnonzero(row)
            val = row[idx]

        if len(idx) == 0:
            out.append("")
            continue

        take = np.argsort(-val)[:n]
        out.append(" ".join(str(int(v)) for v in idx[take]))

    return out

def log_norm_matrix(x, target_sum=10000.0):
    total = np.asarray(x.sum(axis=1)).ravel().astype(np.float32)
    scale = target_sum / np.maximum(total, 1.0)
    if sp.issparse(x):
        out = x.tocsr(copy=True).astype(np.float32)
        out = out.multiply(scale[:, None]).tocsr()
        out.data = np.log1p(out.data)
        return out
    out = np.asarray(x, dtype=np.float32) * scale[:, None]
    return np.log1p(out)

def expression_pcs(adata, n_pcs: int):
    x = log_norm_matrix(adata.X)
    n = min(int(n_pcs), max(1, adata.n_obs - 1), max(1, adata.n_vars - 1))
    if sp.issparse(x):
        emb = TruncatedSVD(n_components=n, random_state=0).fit_transform(x)
    else:
        emb = PCA(n_components=n, random_state=0).fit_transform(x)
    return zscore(emb)

def make_nodes(adata):
    total, detected = per_cell_counts(adata.X)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    low = coords.min(axis=0)
    span = np.ptp(coords, axis=0) + 1e-6
    norm = (coords - low) / span
    cell_area = clean_float(adata.obs["cell_area"].to_numpy(dtype=np.float32))
    nucleus_area = clean_float(adata.obs["nucleus_area"].to_numpy(dtype=np.float32))
    ratio = np.divide(nucleus_area, cell_area, out=np.zeros_like(nucleus_area), where=cell_area > 0)
    return pd.DataFrame({
        "cell_id": adata.obs["cell_id"].to_numpy(dtype=str),
        "x_centroid": coords[:, 0],
        "y_centroid": coords[:, 1],
        "x_norm": norm[:, 0],
        "y_norm": norm[:, 1],
        "log_total_counts": np.log1p(total),
        "log_detected_genes": np.log1p(detected),
        "cell_area": cell_area,
        "nucleus_area": nucleus_area,
        "nucleus_cell_ratio": clean_float(ratio),
    })

def make_pairs_all_pairs(nodes, blocks, measurement):
    n = len(nodes)
    emb = block_embedding(blocks, measurement)
    rows = []

    for i in tqdm(range(n), desc=f"pairs {measurement}"):
        d = np.linalg.norm(emb[i + 1:] - emb[i], axis=1)
        for off, dist in enumerate(d, start=i + 1):
            rows.append((i, off, float(dist)))

    return pd.DataFrame(rows, columns=["source", "target", "distance"])

def make_pairs_sequence_all_pairs(nodes, measurement):
    seqs = [parse_seq_ids(v) for v in nodes["top_gene_ids"].to_numpy()]
    rows = []

    for i in tqdm(range(len(seqs)), desc=f"pairs {measurement}"):
        for j in range(i + 1, len(seqs)):
            d = sequence_distance(seqs[i], seqs[j], measurement)
            rows.append((i, j, float(d)))

    return pd.DataFrame(rows, columns=["source", "target", "distance"])

def load_or_make_pairs(out_dir, nodes, blocks, measurement):
    path = out_dir / f"pairs_{measurement}.parquet"

    if path.exists():
        return pd.read_parquet(path)

    if measurement.startswith("seq_"):
        pairs = make_pairs_sequence_all_pairs(nodes, measurement)
    else:
        pairs = make_pairs_all_pairs(nodes, blocks, measurement)

    pairs.to_parquet(path, index=False)
    return pairs

def neighbor_lists_from_pairs(n, pairs):
    best = [[] for _ in range(n)]

    for r in pairs.itertuples(index=False):
        a = int(r.source)
        b = int(r.target)
        d = float(r.distance)
        best[a].append((b, d))
        best[b].append((a, d))

    for vals in best:
        vals.sort(key=lambda x: x[1])

    return best

def edges_from_neighbor_lists(nodes, blocks, measurement, best, k):
    rows = []

    for i, vals in enumerate(best):
        for j, d in vals[:int(k)]:
            rows.append((min(i, j), max(i, j), d))

    edges = pd.DataFrame(rows, columns=["source", "target", "neighbor_distance"])

    if len(edges):
        edges = edges.sort_values("neighbor_distance").drop_duplicates(["source", "target"])
        edges = edges.sort_values(["source", "target"]).reset_index(drop=True)

    cutoff = auto_neighbor_cutoff(
        edges["neighbor_distance"].to_numpy(),
        CUTOFF_QUANTILE,
        CUTOFF_MAD,
    ) if USE_NEIGHBOR_CUTOFF and len(edges) else np.inf

    raw_edges = len(edges)

    if USE_NEIGHBOR_CUTOFF and np.isfinite(cutoff):
        edges = edges[edges["neighbor_distance"] <= cutoff].copy()

    edges.attrs["neighbor_cutoff"] = float(cutoff)
    edges.attrs["raw_directed_edges"] = raw_edges
    edges.attrs["kept_directed_edges"] = len(edges)
    edges.attrs["pruned_directed_edges"] = raw_edges - len(edges)

    return finish_edges(nodes, blocks, measurement, edges)

def edges_from_pairs(nodes, blocks, measurement, pairs, k):
    best = neighbor_lists_from_pairs(len(nodes), pairs)
    return edges_from_neighbor_lists(nodes, blocks, measurement, best, k)

def build_blocks(adata, nodes, n_pcs):
    spatial = zscore(nodes[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32))
    morphology = zscore(nodes[["cell_area", "nucleus_area", "nucleus_cell_ratio"]].to_numpy(dtype=np.float32))
    expression = expression_pcs(adata, n_pcs)
    return {"spatial": spatial, "expression": expression, "morphology": morphology}

def block_embedding(blocks, measurement):
    parts = []
    for name, weight in MEASUREMENTS[measurement]["blocks"].items():
        parts.append(blocks[name] * float(weight))
    return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]

def pair_distance(block, src, dst):
    return np.linalg.norm(block[dst] - block[src], axis=1).astype(np.float32)

def components(n, edges):
    if n == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    if edges.empty:
        labels = np.arange(n, dtype=np.int32)
        return labels, np.ones(n, dtype=np.int32)
    src = edges["source"].to_numpy(dtype=np.int64)
    dst = edges["target"].to_numpy(dtype=np.int64)
    data = np.ones(src.size * 2, dtype=np.int8)
    graph = sp.csr_matrix((data, (np.r_[src, dst], np.r_[dst, src])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    labels = labels.astype(np.int32)
    sizes = pd.Series(labels).map(pd.Series(labels).value_counts()).to_numpy(dtype=np.int32)
    return labels, sizes

def auto_neighbor_cutoff(values: np.ndarray, quantile=0.995, mad_scale=8.0):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    values = values[values > 0]
    if values.size < 10:
        return np.inf
    q_cut = float(np.quantile(values, float(quantile)))
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    if mad < 1e-6:
        return q_cut
    mad_cut = med + float(mad_scale) * 1.4826 * mad
    out = min(q_cut, mad_cut)
    return float(out) if np.isfinite(out) and out > 0 else np.inf

def soft_prune_knn(src, dst, nd, cutoff, min_edges_per_node=1):
    keep = nd <= cutoff
    if min_edges_per_node <= 0 or src.size == 0:
        return keep
    order = np.lexsort((nd, src))
    seen = {}
    for i in order:
        s = int(src[i])
        c = seen.get(s, 0)
        if c < min_edges_per_node:
            keep[i] = True
            seen[s] = c + 1
    return keep

def finish_edges(nodes, blocks, measurement, edges):
    labels, sizes = components(len(nodes), edges)

    graph_nodes = nodes.copy()
    graph_nodes["component_id"] = labels
    graph_nodes["component_size"] = sizes

    src = edges["source"].to_numpy(dtype=np.int64)
    dst = edges["target"].to_numpy(dtype=np.int64)
    ids = nodes["cell_id"].to_numpy(dtype=str)

    edges["measurement"] = MEASUREMENTS[measurement]["label"]
    edges["xy_distance"] = pair_distance(blocks["spatial"], src, dst)
    edges["expression_distance"] = pair_distance(blocks["expression"], src, dst)
    edges["morphology_distance"] = pair_distance(blocks["morphology"], src, dst)
    edges["source_cell_id"] = ids[src]
    edges["target_cell_id"] = ids[dst]
    edges["source_component_id"] = labels[src]
    edges["target_component_id"] = labels[dst]
    edges["source_component_size"] = sizes[src]
    edges["target_component_size"] = sizes[dst]
    edges["same_component"] = (labels[src] == labels[dst]).astype(np.int8)

    return edges[EDGE_COLS], graph_nodes

def checks(nodes, graph_nodes, edges, measurement):
    return {
        "measurement": measurement,
        "label": MEASUREMENTS[measurement]["label"],
        "n_nodes": int(len(nodes)),
        "n_edges": int(len(edges)),
        "n_components": int(graph_nodes["component_id"].nunique()) if len(graph_nodes) else 0,
        "largest_component": int(graph_nodes["component_size"].max()) if len(graph_nodes) else 0,
        "singletons": int((graph_nodes["component_size"] == 1).sum()) if len(graph_nodes) else 0,
        "neighbor_distance_median": float(edges["neighbor_distance"].median()) if len(edges) else None,
        "neighbor_distance_max": float(edges["neighbor_distance"].max()) if len(edges) else None,
        "neighbor_cutoff": edges.attrs.get("neighbor_cutoff"),
        "raw_directed_edges": edges.attrs.get("raw_directed_edges"),
        "kept_directed_edges": edges.attrs.get("kept_directed_edges"),
        "pruned_directed_edges": edges.attrs.get("pruned_directed_edges"),
    }

def prediction_from_edges(dataset_id, nodes, edges, measurement, k):
    xy = nodes[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
    n = len(nodes)

    sx = np.zeros(n, dtype=np.float64)
    sy = np.zeros(n, dtype=np.float64)
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
    pred = np.full((n, 2), np.nan, dtype=np.float32)
    pred[ok, 0] = sx[ok] / cnt[ok]
    pred[ok, 1] = sy[ok] / cnt[ok]

    dx = pred[:, 0] - xy[:, 0]
    dy = pred[:, 1] - xy[:, 1]
    err = np.sqrt(dx * dx + dy * dy)

    center = xy.mean(axis=0)
    center_err = np.sqrt(((xy - center) ** 2).sum(axis=1))
    ok_err = err[ok]

    row = {
        "dataset": dataset_id,
        "measurement": measurement,
        "label": MEASUREMENTS[measurement]["label"],
        "k": int(k),
        "leaky": measurement in LEAKY_MEASUREMENTS,
        "n_nodes": int(n),
        "n_edges": int(len(edges)),
        "coverage": float(ok.mean()) if n else 0.0,
        "mean_xy_error": float(ok_err.mean()) if len(ok_err) else None,
        "median_xy_error": float(np.median(ok_err)) if len(ok_err) else None,
        "p90_xy_error": float(np.quantile(ok_err, 0.90)) if len(ok_err) else None,
        "center_median_error": float(np.median(center_err)) if len(center_err) else None,
        "median_vs_center": float(np.median(ok_err) / np.median(center_err)) if len(ok_err) and np.median(center_err) > 0 else None,
    }

    pred_df = pd.DataFrame({
        "node": np.arange(n, dtype=np.int64),
        "cell_id": nodes["cell_id"].to_numpy(dtype=str),
        "x": xy[:, 0],
        "y": xy[:, 1],
        "pred_x": pred[:, 0],
        "pred_y": pred[:, 1],
        "dx": dx,
        "dy": dy,
        "error": err,
        "used_neighbors": cnt,
        "measurement": measurement,
        "k": int(k),
    })

    return pred_df, row

def add_spatial_reference(df):
    df = df.copy()

    spatial = (
        df[df["measurement"] == "spatial"]
        [[
            "dataset",
            "k",
            "median_xy_error",
            "mean_xy_error",
            "p90_xy_error",
        ]]
        .rename(columns={
            "median_xy_error": "spatial_median_xy_error_same_k",
            "mean_xy_error": "spatial_mean_xy_error_same_k",
            "p90_xy_error": "spatial_p90_xy_error_same_k",
        })
    )

    df = df.merge(spatial, on=["dataset", "k"], how="left")

    df["median_vs_spatial_same_k"] = np.divide(
        df["median_xy_error"],
        df["spatial_median_xy_error_same_k"],
        out=np.full(len(df), np.nan, dtype=float),
        where=df["spatial_median_xy_error_same_k"].to_numpy(dtype=float) > 0,
    )

    best_spatial = (
        spatial
        .sort_values(["dataset", "spatial_median_xy_error_same_k"])
        .groupby("dataset")
        .head(1)
        [[
            "dataset",
            "k",
            "spatial_median_xy_error_same_k",
        ]]
        .rename(columns={
            "k": "spatial_best_k",
            "spatial_median_xy_error_same_k": "spatial_best_median_xy_error",
        })
    )

    df = df.merge(best_spatial, on="dataset", how="left")

    df["median_vs_spatial_best"] = np.divide(
        df["median_xy_error"],
        df["spatial_best_median_xy_error"],
        out=np.full(len(df), np.nan, dtype=float),
        where=df["spatial_best_median_xy_error"].to_numpy(dtype=float) > 0,
    )

    return df

def summarize_benchmarks(df):
    clean = df[~df["leaky"]].copy()
    clean = clean.dropna(subset=["median_vs_spatial_best"])

    if clean.empty:
        return pd.DataFrame(), pd.DataFrame()

    clean["rank"] = clean.groupby("dataset")["median_vs_spatial_best"].rank(method="min")

    summary = (
        clean
        .groupby(["measurement", "k"])
        .agg(
            datasets=("dataset", "nunique"),
            wins=("rank", lambda x: int((x == 1).sum())),
            rank_mean=("rank", "mean"),
            median_xy_error_median=("median_xy_error", "median"),
            p90_xy_error_median=("p90_xy_error", "median"),
            coverage_mean=("coverage", "mean"),
            median_vs_spatial_best_mean=("median_vs_spatial_best", "mean"),
            median_vs_spatial_best_median=("median_vs_spatial_best", "median"),
            median_vs_spatial_same_k_median=("median_vs_spatial_same_k", "median"),
        )
        .reset_index()
        .sort_values(
            ["wins", "rank_mean", "median_vs_spatial_best_median"],
            ascending=[False, True, True],
        )
    )

    best = (
        clean
        .sort_values(["dataset", "median_vs_spatial_best", "median_xy_error"])
        .groupby("dataset")
        .head(1)
        [[
            "dataset",
            "measurement",
            "k",
            "median_xy_error",
            "p90_xy_error",
            "coverage",
            "median_vs_spatial_best",
            "median_vs_spatial_same_k",
            "spatial_best_k",
            "spatial_best_median_xy_error",
        ]]
    )

    return summary, best

def best_k_by_measurement(df, min_coverage=0.95):
    clean = df[
        (~df["leaky"])
        & (df["coverage"] >= float(min_coverage))
    ].copy()

    clean = clean.dropna(subset=["median_vs_spatial_best", "median_xy_error"])

    if clean.empty:
        return pd.DataFrame()

    best = (
        clean
        .sort_values(["measurement", "median_vs_spatial_best", "median_xy_error"])
        .groupby("measurement")
        .head(1)
        [[
            "measurement",
            "k",
            "median_xy_error",
            "p90_xy_error",
            "coverage",
            "median_vs_spatial_best",
            "median_vs_spatial_same_k",
            "spatial_best_k",
            "spatial_best_median_xy_error",
        ]]
        .reset_index(drop=True)
    )

    return best

def learned_weight_grid(names, values):
    import itertools

    seen = set()

    for vals in itertools.product(values, repeat=len(names)):
        vals = [float(v) for v in vals]

        if all(v == 0.0 for v in vals):
            continue

        mx = max(vals)

        if mx <= 0:
            continue

        norm = tuple(round(v / mx, 8) for v in vals)

        if norm in seen:
            continue

        seen.add(norm)
        yield dict(zip(names, norm))

def learned_label(weights):
    parts = []

    for k, v in weights.items():
        parts.append(f"{k}{v:g}")

    return "learned_" + "_".join(parts)

def normalized_pair_distances(pairs):
    d = pairs["distance"].to_numpy(dtype=np.float32)
    ok = np.isfinite(d) & (d > 0)

    if ok.any():
        scale = float(np.median(d[ok]))
    else:
        scale = 1.0

    if scale <= 0 or not np.isfinite(scale):
        scale = 1.0

    return d / scale

def combine_pair_tables(pair_tables, weights):
    first = pair_tables[next(iter(weights))]
    src = first["source"].to_numpy(dtype=np.int64)
    dst = first["target"].to_numpy(dtype=np.int64)

    acc = np.zeros(len(first), dtype=np.float32)

    for name, w in weights.items():
        pairs = pair_tables[name]

        if not (
            np.array_equal(src, pairs["source"].to_numpy(dtype=np.int64))
            and np.array_equal(dst, pairs["target"].to_numpy(dtype=np.int64))
        ):
            raise RuntimeError(f"pair order mismatch for {name}")

        d = normalized_pair_distances(pairs)
        acc += (float(w) * d) ** 2

    out = pd.DataFrame({
        "source": src,
        "target": dst,
        "distance": np.sqrt(acc).astype(np.float32),
    })

    return out

def run_learned_mix(out_dir, dataset_id, nodes, blocks, node_cols, bench_k_values):
    pair_tables = {}

    for m in LEARNED_BASE_MEASUREMENTS:
        pair_tables[m] = load_or_make_pairs(out_dir, nodes[node_cols], blocks, m)

    rows = []
    best_item = None

    MEASUREMENTS[LEARNED_MIX_NAME] = {
        "label": LEARNED_MIX_NAME,
        "blocks": {},
    }

    for weights in tqdm(
        list(learned_weight_grid(LEARNED_BASE_MEASUREMENTS, LEARNED_WEIGHT_VALUES)),
        desc="learned mix",
    ):
        label = learned_label(weights)
        pairs = combine_pair_tables(pair_tables, weights)
        best = neighbor_lists_from_pairs(len(nodes), pairs)

        for kk in bench_k_values:
            edges, graph_nodes = edges_from_neighbor_lists(
                nodes[node_cols],
                blocks,
                LEARNED_MIX_NAME,
                best,
                kk,
            )

            pred_df, row = prediction_from_edges(
                dataset_id,
                nodes[node_cols],
                edges,
                LEARNED_MIX_NAME,
                kk,
            )

            row["label"] = label

            for name, w in weights.items():
                row[f"weight_{name}"] = float(w)

            rows.append(row)

            if row["coverage"] >= LEARNED_MIN_COVERAGE and row["median_xy_error"] is not None:
                median_err = float(row["median_xy_error"])

                if row["p90_xy_error"] is None or not np.isfinite(row["p90_xy_error"]):
                    p90_err = np.inf
                else:
                    p90_err = float(row["p90_xy_error"])

                main_score = median_err + LEARNED_P90_WEIGHT * p90_err

                score = (
                    main_score,
                    median_err,
                    p90_err,
                    int(kk),
                    label,
                )

                item = {
                    "score": score,
                    "weights": weights.copy(),
                }

                if best_item is None or score < best_item["score"]:
                    best_item = item

    grid = pd.DataFrame(rows)
    grid.to_csv(out_dir / "learned_mix_grid.csv", index=False)

    top = grid.copy()
    top["score"] = top["median_xy_error"] + LEARNED_P90_WEIGHT * top["p90_xy_error"]
    top = top.sort_values(["score", "median_xy_error", "p90_xy_error", "k"]).head(30)
    top.to_csv(out_dir / "learned_mix_top.csv", index=False)

    if best_item is None:
        return [], {}

    best_weights = best_item["weights"]
    best_label = learned_label(best_weights)

    MEASUREMENTS[LEARNED_MIX_NAME] = {
        "label": best_label,
        "blocks": {},
    }

    pairs = combine_pair_tables(pair_tables, best_weights)
    pairs.to_parquet(out_dir / "pairs_learned_mix.parquet", index=False)

    best = neighbor_lists_from_pairs(len(nodes), pairs)

    bench_rows = []
    summaries = {}

    for kk in bench_k_values:
        edges, graph_nodes = edges_from_neighbor_lists(
            nodes[node_cols],
            blocks,
            LEARNED_MIX_NAME,
            best,
            kk,
        )

        chk = checks(nodes, graph_nodes, edges, LEARNED_MIX_NAME)
        pred_df, bench_row = prediction_from_edges(
            dataset_id,
            nodes[node_cols],
            edges,
            LEARNED_MIX_NAME,
            kk,
        )

        bench_row["label"] = best_label

        for name, w in best_weights.items():
            bench_row[f"weight_{name}"] = float(w)

        bench_rows.append(bench_row)

        graph_nodes.to_csv(out_dir / f"nodes_{LEARNED_MIX_NAME}_k{kk}.csv", index=False)
        edges.to_csv(out_dir / f"edges_{LEARNED_MIX_NAME}_k{kk}.csv", index=False)
        pred_df.to_csv(out_dir / f"predictions_{LEARNED_MIX_NAME}_k{kk}.csv", index=False)
        (out_dir / f"checks_{LEARNED_MIX_NAME}_k{kk}.json").write_text(json.dumps(chk, indent=2) + "\n")

        if kk == int(K):
            graph_nodes.to_csv(out_dir / f"nodes_{LEARNED_MIX_NAME}.csv", index=False)
            edges.to_csv(out_dir / f"edges_{LEARNED_MIX_NAME}.csv", index=False)
            (out_dir / f"checks_{LEARNED_MIX_NAME}.json").write_text(json.dumps(chk, indent=2) + "\n")
            write_npz(out_dir, LEARNED_MIX_NAME, graph_nodes, edges)
            summaries[LEARNED_MIX_NAME] = chk

    payload = {
        "measurement": LEARNED_MIX_NAME,
        "label": best_label,
        "weights": best_weights,
        "base_measurements": list(LEARNED_BASE_MEASUREMENTS),
        "weight_values": list(LEARNED_WEIGHT_VALUES),
        "min_coverage": float(LEARNED_MIN_COVERAGE),
        "p90_weight": float(LEARNED_P90_WEIGHT),
    }

    (out_dir / "learned_mix_weights.json").write_text(json.dumps(payload, indent=2) + "\n")

    return bench_rows, summaries

def parse_k_list(s):
    vals = []

    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))

    return vals

def write_npz(out_dir, measurement, graph_nodes, edges):
    node_features = graph_nodes[["log_total_counts", "log_detected_genes", "cell_area", "nucleus_area", "nucleus_cell_ratio", "component_id", "component_size"]].to_numpy(dtype=np.float32)
    pos = graph_nodes[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
    edge_index = edges[["source", "target"]].to_numpy(dtype=np.int64).T if len(edges) else np.empty((2, 0), dtype=np.int64)
    edge_attr = edges[["neighbor_distance", "xy_distance", "expression_distance", "morphology_distance", "same_component"]].to_numpy(dtype=np.float32) if len(edges) else np.empty((0, 5), dtype=np.float32)
    np.savez_compressed(
        out_dir / f"representation_{measurement}.npz",
        node_features=node_features,
        node_positions=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        cell_ids=graph_nodes["cell_id"].to_numpy(dtype=str),
        feature_columns=np.asarray(["log_total_counts", "log_detected_genes", "cell_area", "nucleus_area", "nucleus_cell_ratio", "component_id", "component_size"], dtype=str),
        edge_feature_columns=np.asarray(["neighbor_distance", "xy_distance", "expression_distance", "morphology_distance", "same_component"], dtype=str),
        measurement=np.asarray([MEASUREMENTS[measurement]["label"]], dtype=str),
    )

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_id")
    return p.parse_args()

def main():
    args = parse_args()

    xenium_dir = data_dir(args.dataset_id)
    out_dir = make_out_dir(args.dataset_id)

    adata = load_xenium(xenium_dir)

    nodes = make_nodes(adata)
    nodes["top_gene_ids"] = top_gene_ids(adata.X, TOP_GENES_PER_CELL)
    blocks = build_blocks(adata, nodes, EXPRESSION_PCS)

    expr_cols = []
    for i in range(blocks["expression"].shape[1]):
        col = f"expression_pc{i + 1}"
        nodes[col] = blocks["expression"][:, i]
        expr_cols.append(col)

    node_cols = NODE_BASE_COLS + ["top_gene_ids"] + expr_cols
    nodes[node_cols].to_csv(out_dir / "nodes.csv", index=False)

    summaries = {}
    bench_rows = []
    bench_k_values = sorted(set([int(K), *BENCH_K_VALUES]))
    measurement_names = list(ACTIVE_MEASUREMENTS)

    for m in tqdm(measurement_names, desc="graphs"):
        pairs = load_or_make_pairs(out_dir, nodes[node_cols], blocks, m)
        best = neighbor_lists_from_pairs(len(nodes), pairs)

        for kk in bench_k_values:
            edges, graph_nodes = edges_from_neighbor_lists(
                nodes[node_cols],
                blocks,
                m,
                best,
                kk,
            )

            chk = checks(nodes, graph_nodes, edges, m)
            pred_df, bench_row = prediction_from_edges(args.dataset_id, nodes[node_cols], edges, m, kk)
            bench_rows.append(bench_row)

            graph_nodes.to_csv(out_dir / f"nodes_{m}_k{kk}.csv", index=False)
            edges.to_csv(out_dir / f"edges_{m}_k{kk}.csv", index=False)
            pred_df.to_csv(out_dir / f"predictions_{m}_k{kk}.csv", index=False)
            (out_dir / f"checks_{m}_k{kk}.json").write_text(json.dumps(chk, indent=2) + "\n")

            if kk == int(K):
                graph_nodes.to_csv(out_dir / f"nodes_{m}.csv", index=False)
                edges.to_csv(out_dir / f"edges_{m}.csv", index=False)
                (out_dir / f"checks_{m}.json").write_text(json.dumps(chk, indent=2) + "\n")
                write_npz(out_dir, m, graph_nodes, edges)
                summaries[m] = chk

        base = summaries.get(m)
        if base is not None:
            print(f"{m}: nodes={base['n_nodes']} edges={base['n_edges']} components={base['n_components']}", flush=True)
        else:
            print(f"{m}: bench done", flush=True)

    learned_rows, learned_summaries = run_learned_mix(
        out_dir,
        args.dataset_id,
        nodes,
        blocks,
        node_cols,
        bench_k_values,
    )

    bench_rows.extend(learned_rows)
    summaries.update(learned_summaries)

    bench_df = pd.DataFrame(bench_rows).sort_values(["leaky", "measurement", "k"])
    bench_df = add_spatial_reference(bench_df)

    bench_df.to_csv(out_dir / "bench_xy.csv", index=False)
    bench_df.to_csv(out_dir / "bench_xy_by_k.csv", index=False)

    bench_summary, bench_best = summarize_benchmarks(bench_df)
    bench_summary.to_csv(out_dir / "bench_xy_summary.csv", index=False)
    bench_best.to_csv(out_dir / "bench_xy_best.csv", index=False)

    best_k = best_k_by_measurement(bench_df)
    best_k.to_csv(out_dir / "best_k_by_measurement.csv", index=False)

    best_nonleaky = (
        bench_df[~bench_df["leaky"]]
        .dropna(subset=["median_vs_spatial_best"])
        .sort_values(["median_vs_spatial_best", "median_xy_error"])
        .head(10)
    )

    print()
    print("bench top non-leaky:", flush=True)
    print(
        best_nonleaky[
            [
                "measurement",
                "k",
                "n_edges",
                "coverage",
                "median_xy_error",
                "median_vs_spatial_best",
                "median_vs_spatial_same_k",
            ]
        ].to_string(index=False),
        flush=True,
    )

    print(f"bench saved: {out_dir / 'bench_xy.csv'}", flush=True)
    print(f"predictions saved: {out_dir}", flush=True)


    (out_dir / "summary.json").write_text(json.dumps({
        **adata.uns["xenium_clean"],
        "n_cells_loaded": int(adata.n_obs),
        "n_cells_dumped": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "k": int(K),
        "bench_k_values": bench_k_values,
        "learned_mix_name": LEARNED_MIX_NAME,
        "learned_base_measurements": list(LEARNED_BASE_MEASUREMENTS),
        "learned_weight_values": list(LEARNED_WEIGHT_VALUES),
        "learned_min_coverage": float(LEARNED_MIN_COVERAGE),
        "expression_pcs": int(EXPRESSION_PCS),
        "use_neighbor_cutoff": bool(USE_NEIGHBOR_CUTOFF),
        "cutoff_quantile": float(CUTOFF_QUANTILE),
        "cutoff_mad": float(CUTOFF_MAD),
        "min_edges_per_node": int(MIN_EDGES_PER_NODE),
        "measurements": measurement_names,
        "visible_measurements": list(VISIBLE_MEASUREMENTS),
        "hidden_measurements": list(HIDDEN_MEASUREMENTS),
        "active_measurements": list(ACTIVE_MEASUREMENTS),
        "measurement_defs": {m: MEASUREMENTS[m] for m in measurement_names},
        "node_columns": node_cols,
        "edge_columns": EDGE_COLS,
        "edge_summaries": summaries,
    }, indent=2) + "\n")

    print(f"saved: {out_dir}")

if __name__ == "__main__":
    main()
