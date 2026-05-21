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
from xenum_measurements import MEASUREMENTS
from xenum_paths import data_dir, out_dir as make_out_dir

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

K = 4
EXPRESSION_PCS = 30

USE_NEIGHBOR_CUTOFF = True
CUTOFF_QUANTILE = 0.995
CUTOFF_MAD = 8.0
MIN_EDGES_PER_NODE = 1
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

def edges_from_pairs(nodes, blocks, measurement, pairs, k):
    best = [[] for _ in range(len(nodes))]

    for r in pairs.itertuples(index=False):
        a = int(r.source)
        b = int(r.target)
        d = float(r.distance)
        best[a].append((b, d))
        best[b].append((a, d))

    rows = []
    for i, vals in enumerate(best):
        vals.sort(key=lambda x: x[1])
        for j, d in vals[:k]:
            rows.append((min(i, j), max(i, j), d))

    edges = pd.DataFrame(rows, columns=["source", "target", "neighbor_distance"])
    edges = edges.sort_values("neighbor_distance").drop_duplicates(["source", "target"])
    edges = edges.sort_values(["source", "target"]).reset_index(drop=True)

    return finish_edges(nodes, blocks, measurement, edges)

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
    p.add_argument("--k", type=int, default=K)
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

    for m in tqdm(list(MEASUREMENTS), desc="graphs"):
        pairs = load_or_make_pairs(out_dir, nodes[node_cols], blocks, m)
        edges, graph_nodes = edges_from_pairs(
            nodes[node_cols],
            blocks,
            m,
            pairs,
            args.k,
        )

        chk = checks(nodes, graph_nodes, edges, m)

        graph_nodes.to_csv(out_dir / f"nodes_{m}.csv", index=False)
        edges.to_csv(out_dir / f"edges_{m}.csv", index=False)
        (out_dir / f"checks_{m}.json").write_text(json.dumps(chk, indent=2) + "\n")

        write_npz(out_dir, m, graph_nodes, edges)

        summaries[m] = chk
        print(f"{m}: nodes={chk['n_nodes']} edges={chk['n_edges']} components={chk['n_components']}", flush=True)

    (out_dir / "summary.json").write_text(json.dumps({
        **adata.uns["xenium_clean"],
        "n_cells_loaded": int(adata.n_obs),
        "n_cells_dumped": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "k": int(args.k),
        "expression_pcs": int(EXPRESSION_PCS),
        "use_neighbor_cutoff": bool(USE_NEIGHBOR_CUTOFF),
        "cutoff_quantile": float(CUTOFF_QUANTILE),
        "cutoff_mad": float(CUTOFF_MAD),
        "min_edges_per_node": int(MIN_EDGES_PER_NODE),
        "measurements": list(MEASUREMENTS),
        "measurement_defs": MEASUREMENTS,
        "node_columns": node_cols,
        "edge_columns": EDGE_COLS,
        "edge_summaries": summaries,
    }, indent=2) + "\n")

    print(f"saved: {out_dir}")

if __name__ == "__main__":
    main()
