from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from xenum_common import parse_seq_ids, sequence_distance
from xenum_measurements import LEAKY_MEASUREMENTS, MEASUREMENTS

from .config import CUTOFF_MAD, CUTOFF_QUANTILE, EDGE_COLS, MIN_EDGES_PER_NODE, USE_NEIGHBOR_CUTOFF, tqdm
from .features import block_embedding, jaccard_distance, pair_distance
from .io import pair_cache_path, pair_legacy_path, write_pair_cache

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
    col = "detected_gene_ids" if measurement == "seq_jaccard_all" else "top_gene_ids"
    seqs = [parse_seq_ids(v) for v in nodes[col].to_numpy()]
    rows = []

    for i in tqdm(range(len(seqs)), desc=f"pairs {measurement}"):
        for j in range(i + 1, len(seqs)):
            if measurement in {"seq_jaccard", "seq_jaccard_all"}:
                d = jaccard_distance(seqs[i], seqs[j])
            else:
                d = sequence_distance(seqs[i], seqs[j], measurement)

            rows.append((i, j, float(d)))

    return pd.DataFrame(rows, columns=["source", "target", "distance"])

def load_or_make_pairs(out_dir, nodes, blocks, measurement):
    cache_path = pair_cache_path(out_dir, measurement)
    legacy_path = pair_legacy_path(out_dir, measurement)

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    if legacy_path.exists():
        pairs = pd.read_parquet(legacy_path)
        write_pair_cache(out_dir, measurement, pairs)
        return pairs

    if measurement.startswith("seq_"):
        pairs = make_pairs_sequence_all_pairs(nodes, measurement)
    else:
        pairs = make_pairs_all_pairs(nodes, blocks, measurement)

    write_pair_cache(out_dir, measurement, pairs)
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

    if "neighbor_weight" not in edges.columns:
        edges["neighbor_weight"] = 1.0

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
        "neighbor_weight_median": float(edges["neighbor_weight"].median()) if len(edges) and "neighbor_weight" in edges.columns else None,
        "neighbor_weight_max": float(edges["neighbor_weight"].max()) if len(edges) and "neighbor_weight" in edges.columns else None,
        "neighbor_cutoff": edges.attrs.get("neighbor_cutoff"),
        "raw_directed_edges": edges.attrs.get("raw_directed_edges"),
        "kept_directed_edges": edges.attrs.get("kept_directed_edges"),
        "pruned_directed_edges": edges.attrs.get("pruned_directed_edges"),
    }

def prediction_metrics(dataset_id, nodes, pred, used_neighbors, measurement, k, seed=None, status="ok", n_edges=0):
    xy = nodes[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
    n = len(nodes)

    pred = np.asarray(pred, dtype=np.float32)
    used_neighbors = np.asarray(used_neighbors, dtype=np.int64)
    ok = np.isfinite(pred).all(axis=1)

    dx = pred[:, 0] - xy[:, 0]
    dy = pred[:, 1] - xy[:, 1]
    err = np.sqrt(dx * dx + dy * dy)

    center = xy.mean(axis=0) if n else np.zeros(2, dtype=np.float32)
    center_err = np.sqrt(((xy - center) ** 2).sum(axis=1))
    ok_err = err[ok]

    real_spread = float(np.sqrt(np.var(xy[:, 0]) + np.var(xy[:, 1]))) if n else 0.0

    if ok.any():
        pred_ok = pred[ok]
        pred_spread = float(np.sqrt(np.nanvar(pred_ok[:, 0]) + np.nanvar(pred_ok[:, 1])))
    else:
        pred_spread = 0.0

    pred_spread_ratio = pred_spread / real_spread if real_spread > 0 else None

    row = {
        "dataset": dataset_id,
        "measurement": measurement,
        "label": MEASUREMENTS[measurement]["label"],
        "k": int(k),
        "seed": None if seed is None else int(seed),
        "status": status,
        "leaky": measurement in LEAKY_MEASUREMENTS,
        "n_nodes": int(n),
        "n_edges": int(n_edges),
        "coverage": float(ok.mean()) if n else 0.0,
        "real_spread": real_spread,
        "pred_spread": pred_spread,
        "pred_spread_ratio": pred_spread_ratio,
        "mean_xy_error": float(ok_err.mean()) if len(ok_err) else None,
        "median_xy_error": float(np.median(ok_err)) if len(ok_err) else None,
        "p90_xy_error": float(np.quantile(ok_err, 0.90)) if len(ok_err) else None,
        "center_median_error": float(np.median(center_err)) if len(center_err) else None,
        "median_vs_center": float(np.median(ok_err) / np.median(center_err)) if len(ok_err) and np.median(center_err) > 0 else None,
    }

    pred_data = {
        "node": np.arange(n, dtype=np.int64),
        "cell_id": nodes["cell_id"].to_numpy(dtype=str),
        "x": xy[:, 0],
        "y": xy[:, 1],
        "pred_x": pred[:, 0],
        "pred_y": pred[:, 1],
        "dx": dx,
        "dy": dy,
        "error": err,
        "used_neighbors": used_neighbors,
        "measurement": measurement,
        "k": int(k),
        "seed": None if seed is None else int(seed),
        "status": status,
    }

    pred_df = pd.DataFrame(pred_data)

    return pred_df, row

def prediction_from_edges(dataset_id, nodes, edges, measurement, k, weight_col=None, directed=False, seed=None, status="ok"):
    xy = nodes[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
    n = len(nodes)

    sx = np.zeros(n, dtype=np.float64)
    sy = np.zeros(n, dtype=np.float64)
    sw = np.zeros(n, dtype=np.float64)
    cnt = np.zeros(n, dtype=np.int64)

    for r in edges.itertuples(index=False):
        a = int(r.source)
        b = int(r.target)
        w = float(getattr(r, weight_col)) if weight_col else 1.0

        if not np.isfinite(w) or w <= 0:
            continue

        sx[a] += w * xy[b, 0]
        sy[a] += w * xy[b, 1]
        sw[a] += w
        cnt[a] += 1

        if not directed:
            sx[b] += w * xy[a, 0]
            sy[b] += w * xy[a, 1]
            sw[b] += w
            cnt[b] += 1

    ok = sw > 0
    pred = np.full((n, 2), np.nan, dtype=np.float32)
    pred[ok, 0] = sx[ok] / sw[ok]
    pred[ok, 1] = sy[ok] / sw[ok]

    pred_df, row = prediction_metrics(
        dataset_id,
        nodes,
        pred,
        cnt,
        measurement,
        k,
        seed=seed,
        status=status,
        n_edges=len(edges),
    )

    if weight_col:
        pred_df["neighbor_weight_sum"] = sw

    return pred_df, row

def prediction_from_direct_coordinates(dataset_id, nodes, pred, measurement, seed=None, status="ok"):
    used = np.zeros(len(nodes), dtype=np.int64)
    return prediction_metrics(
        dataset_id,
        nodes,
        pred,
        used,
        measurement,
        0,
        seed=seed,
        status=status,
        n_edges=0,
    )
