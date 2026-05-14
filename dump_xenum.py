#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

CELL_TABLE_NAMES = ("cells.csv.gz", "cells.csv", "cells.parquet")
DATA_ROOT = Path("data")
OUTPUT_ROOT = Path("outputs")

EDGE_MEASUREMENTS = {
    "spatial": {
        "label": "spatial:xy",
        "weights": {"spatial": 1.0, "profile": 0.0, "morphology": 0.0},
        "cutoff_quantile": 0.995,
        "cutoff_mad": 8.0,
    },
    "profile": {
        "label": "rna:summary",
        "weights": {"spatial": 0.0, "profile": 1.0, "morphology": 0.0},
        "cutoff_quantile": 0.98,
        "cutoff_mad": 6.0,
    },
    "morphology": {
        "label": "morphology:area",
        "weights": {"spatial": 0.0, "profile": 0.0, "morphology": 1.0},
        "cutoff_quantile": 0.98,
        "cutoff_mad": 6.0,
    },
    "mix": {
        "label": "mix:quick",
        "weights": {"spatial": 1.0, "profile": 0.35, "morphology": 0.50},
        "cutoff_quantile": 0.99,
        "cutoff_mad": 8.0,
    },
}


def find_file(root: Path, names: tuple[str, ...]) -> Path:
    roots = [root] if root.name == "bundle" else [root, root / "bundle"]

    for base in roots:
        for name in names:
            path = base / name
            if path.exists():
                return path

    for base in roots:
        for name in names:
            for path in base.rglob(name):
                return path

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
        mask = adata.var["feature_types"].astype(str).eq("Gene Expression")
        adata = adata[:, mask].copy()

    cells, cells_path = read_cells(xenium_dir)
    ids = adata.obs_names.intersection(cells.index)

    adata = adata[ids].copy()
    cells = cells.loc[adata.obs_names]

    coords = cells[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)

    adata.obsm["spatial"] = coords
    adata.obs["cell_id"] = adata.obs_names.astype(str)
    adata.obs["x_centroid"] = coords[:, 0]
    adata.obs["y_centroid"] = coords[:, 1]
    adata.obs["cell_area"] = cells["cell_area"].to_numpy() if "cell_area" in cells else 0.0
    adata.obs["nucleus_area"] = cells["nucleus_area"].to_numpy() if "nucleus_area" in cells else 0.0

    adata.uns["xenium_dump"] = {
        "xenium_dir": str(xenium_dir),
        "matrix_path": str(matrix_path),
        "cells_path": str(cells_path),
    }
    return adata

def zscore(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)

def knn_labels(coords: np.ndarray, k: int) -> np.ndarray:
    if coords.shape[0] < 2:
        return np.zeros(coords.shape[0], dtype=np.int32)

    kk = min(k, coords.shape[0] - 1)
    graph = NearestNeighbors(n_neighbors=kk + 1).fit(coords).kneighbors_graph(coords)
    graph = graph.maximum(graph.T)

    _, labels = connected_components(graph, directed=False)
    return labels.astype(np.int32)


def rough_grid(coords: np.ndarray, max_cells: int):
    tile_target = max(max_cells // 64, 1)
    n_tiles = int(np.ceil(coords.shape[0] / tile_target))

    span = np.ptp(coords, axis=0)
    aspect = max(float(span[0] / (span[1] + 1e-6)), 1e-6)

    nx = max(1, int(np.ceil(np.sqrt(n_tiles * aspect))))
    ny = max(1, int(np.ceil(n_tiles / nx)))

    low = coords.min(axis=0)

    x = np.clip(
        ((coords[:, 0] - low[0]) / (span[0] + 1e-6) * nx).astype(int),
        0,
        nx - 1,
    )
    y = np.clip(
        ((coords[:, 1] - low[1]) / (span[1] + 1e-6) * ny).astype(int),
        0,
        ny - 1,
    )

    code = y * nx + x
    counts = np.bincount(code, minlength=nx * ny)

    seed = int(counts.argmax())
    sx, sy = seed % nx, seed // nx

    nonempty = np.flatnonzero(counts)
    tx, ty = nonempty % nx, nonempty // nx

    order = nonempty[np.argsort((tx - sx) ** 2 + (ty - sy) ** 2)]

    picked = []
    total = 0

    for c in order:
        picked.append(int(c))
        total += int(counts[c])
        if total >= max_cells:
            break

    picked = set(picked)
    padded = set(picked)

    for c in picked:
        cx, cy = c % nx, c // nx
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                xx, yy = cx + dx, cy + dy
                if 0 <= xx < nx and 0 <= yy < ny:
                    padded.add(yy * nx + xx)

    core = np.flatnonzero(np.isin(code, list(picked)))
    candidate = np.flatnonzero(np.isin(code, list(padded)))

    return core, candidate


def select_cells(adata, max_cells: int, k: int):
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)

    if adata.n_obs <= max_cells:
        out = adata.copy()
        out.obs["island_id"] = knn_labels(coords, k)
        out.uns["crop_mode"] = "all"
        return out

    core, candidate = rough_grid(coords, max_cells)

    sub_coords = coords[candidate]
    labels = knn_labels(sub_coords, k)

    core_mask = np.zeros(adata.n_obs, dtype=bool)
    core_mask[core] = True

    core_labels = labels[core_mask[candidate]]
    seed = coords[core].mean(axis=0)

    selected_local = []
    total = 0

    for label in pd.Series(core_labels).value_counts().index:
        local = np.flatnonzero(labels == label)

        if total and total + local.size > max_cells:
            continue

        selected_local.append(local)
        total += local.size

        if total >= max_cells * 0.85:
            break

    selected_local = (
        np.concatenate(selected_local)
        if selected_local
        else np.array([], dtype=np.int64)
    )

    if selected_local.size == 0 or selected_local.size > max_cells:
        d = np.linalg.norm(sub_coords - seed, axis=1)
        selected_local = np.argsort(d)[:max_cells]

    selected = candidate[selected_local]

    out = adata[selected].copy()
    _, island_id = np.unique(labels[selected_local], return_inverse=True)

    out.obs["island_id"] = island_id.astype(np.int32)
    out.uns["crop_mode"] = "grid_knn"
    out.uns["max_cells"] = int(max_cells)

    return out


def per_cell_counts(x):
    total = np.asarray(x.sum(axis=1)).ravel().astype(np.float32)
    detected = np.asarray(
        x.getnnz(axis=1) if sp.issparse(x) else (x > 0).sum(axis=1)
    ).ravel()
    return total, detected.astype(np.int32)


def make_nodes(adata) -> pd.DataFrame:
    total, detected = per_cell_counts(adata.X)

    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    low = coords.min(axis=0)
    high = coords.max(axis=0)
    span = high - low + 1e-6

    centered = coords - coords.mean(axis=0)
    aligned = (coords - low) / span

    island = adata.obs["island_id"].to_numpy(dtype=np.int32)
    island_size = pd.Series(island).map(pd.Series(island).value_counts()).to_numpy(dtype=np.int32)

    cell_area = adata.obs["cell_area"].to_numpy(dtype=np.float32)
    nucleus_area = adata.obs["nucleus_area"].to_numpy(dtype=np.float32)

    return pd.DataFrame(
        {
            "node_index": np.arange(adata.n_obs, dtype=np.int64),
            "cell_id": adata.obs["cell_id"].to_numpy(dtype=str),

            "x_centroid": coords[:, 0],
            "y_centroid": coords[:, 1],
            "x_centered": centered[:, 0],
            "y_centered": centered[:, 1],
            "x_aligned": aligned[:, 0],
            "y_aligned": aligned[:, 1],

            "spatial_island_id": island,
            "spatial_island_size": island_size,
            "island_id": island,
            "island_size": island_size,

            "total_counts": total,
            "log_total_counts": np.log1p(total),
            "n_detected_genes": detected,
            "log_detected_genes": np.log1p(detected),

            "cell_area": cell_area,
            "nucleus_area": nucleus_area,
            "nucleus_cell_ratio": nucleus_area / (cell_area + 1e-6),
        }
    )

def edge_blocks(nodes: pd.DataFrame):
    spatial = zscore(nodes[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32))

    profile = zscore(
        nodes[
            [
                "log_total_counts",
                "log_detected_genes",
            ]
        ].to_numpy(dtype=np.float32)
    )

    morphology = zscore(
        nodes[
            [
                "cell_area",
                "nucleus_area",
                "nucleus_cell_ratio",
            ]
        ].to_numpy(dtype=np.float32)
    )

    return {
        "spatial": spatial,
        "profile": profile,
        "morphology": morphology,
    }


def edge_embedding(blocks: dict[str, np.ndarray], measurement: str) -> np.ndarray:
    weights = EDGE_MEASUREMENTS[measurement]["weights"]
    parts = [blocks[name] * weight for name, weight in weights.items() if weight > 0]
    return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]


def pair_block_distance(block: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    return np.linalg.norm(block[dst] - block[src], axis=1).astype(np.float32)

def graph_components(n: int, edges: pd.DataFrame):
    if n == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    if edges.empty:
        labels = np.arange(n, dtype=np.int32)
        sizes = np.ones(n, dtype=np.int32)
        return labels, sizes

    row = edges["source"].to_numpy(dtype=np.int64)
    col = edges["target"].to_numpy(dtype=np.int64)

    data = np.ones(row.size * 2, dtype=np.int8)
    graph = sp.csr_matrix(
        (
            data,
            (
                np.r_[row, col],
                np.r_[col, row],
            ),
        ),
        shape=(n, n),
    )

    _, labels = connected_components(graph, directed=False)
    labels = labels.astype(np.int32)

    sizes = pd.Series(labels).map(pd.Series(labels).value_counts()).to_numpy(dtype=np.int32)

    return labels, sizes


def nodes_for_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    out = nodes.copy()

    labels, sizes = graph_components(nodes.shape[0], edges)

    out["component_id"] = labels
    out["component_size"] = sizes

    out["graph_island_id"] = labels
    out["graph_island_size"] = sizes

    return out


def add_graph_components_to_edges(edges: pd.DataFrame, graph_nodes: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        edges = edges.copy()
        edges["source_component_id"] = []
        edges["target_component_id"] = []
        edges["source_component_size"] = []
        edges["target_component_size"] = []
        edges["same_component"] = []
        return edges

    edges = edges.copy()

    src = edges["source"].to_numpy(dtype=np.int64)
    dst = edges["target"].to_numpy(dtype=np.int64)

    component_id = graph_nodes["component_id"].to_numpy(dtype=np.int32)
    component_size = graph_nodes["component_size"].to_numpy(dtype=np.int32)

    edges["source_component_id"] = component_id[src]
    edges["target_component_id"] = component_id[dst]
    edges["source_component_size"] = component_size[src]
    edges["target_component_size"] = component_size[dst]
    edges["same_component"] = (
        edges["source_component_id"].to_numpy()
        == edges["target_component_id"].to_numpy()
    ).astype(np.int8)

    return edges

def graph_checks(
    nodes: pd.DataFrame,
    graph_nodes: pd.DataFrame,
    edges: pd.DataFrame,
    measurement: str,
) -> dict:
    n = int(nodes.shape[0])
    m = int(edges.shape[0])

    out = {
        "measurement": measurement,
        "label": EDGE_MEASUREMENTS[measurement]["label"],

        "n_nodes": n,
        "n_edges": m,

        "n_components": int(graph_nodes["component_id"].nunique()) if n else 0,
        "largest_component": int(graph_nodes["component_size"].max()) if n else 0,
        "singletons": int((graph_nodes["component_size"] == 1).sum()) if n else 0,

        "neighbor_cutoff": edges.attrs.get("neighbor_cutoff"),
        "candidate_directed_edges": edges.attrs.get("candidate_directed_edges"),
        "kept_directed_edges": edges.attrs.get("kept_directed_edges"),
        "pruned_directed_edges": edges.attrs.get("pruned_directed_edges"),
        "duplicated_undirected_edges": edges.attrs.get("duplicated_undirected_edges", 0),
    }

    if m == 0:
        out.update(
            {
                "bad_self_loops": 0,
                "bad_node_index": 0,
                "duplicated_edges_after_write": 0,
                "same_component_mean": None,
                "same_spatial_island_mean": None,
                "distance_median": None,
                "distance_max": None,
                "neighbor_distance_median": None,
                "neighbor_distance_max": None,
            }
        )
        return out

    src = edges["source"].to_numpy(dtype=np.int64)
    dst = edges["target"].to_numpy(dtype=np.int64)

    out["bad_self_loops"] = int((src == dst).sum())
    out["bad_node_index"] = int(((src < 0) | (dst < 0) | (src >= n) | (dst >= n)).sum())
    out["duplicated_edges_after_write"] = int(edges.duplicated(["source", "target"]).sum())

    if "same_component" in edges:
        out["same_component_mean"] = float(edges["same_component"].mean())
    else:
        out["same_component_mean"] = None

    if "same_spatial_island" in edges:
        out["same_spatial_island_mean"] = float(edges["same_spatial_island"].mean())
    else:
        out["same_spatial_island_mean"] = None

    out["distance_median"] = float(edges["distance"].median())
    out["distance_max"] = float(edges["distance"].max())
    out["neighbor_distance_median"] = float(edges["neighbor_distance"].median())
    out["neighbor_distance_max"] = float(edges["neighbor_distance"].max())

    return out


def print_graph_checks(checks: dict):
    bad = []

    if checks["bad_self_loops"]:
        bad.append(f"self_loops={checks['bad_self_loops']}")

    if checks["bad_node_index"]:
        bad.append(f"bad_node_index={checks['bad_node_index']}")

    if checks["duplicated_edges_after_write"]:
        bad.append(f"duplicated_edges={checks['duplicated_edges_after_write']}")

    if checks["same_component_mean"] is not None and checks["same_component_mean"] < 1.0:
        bad.append(f"same_component_mean={checks['same_component_mean']:.4f}")

    msg = (
        f"{checks['measurement']}: "
        f"nodes={checks['n_nodes']} "
        f"edges={checks['n_edges']} "
        f"components={checks['n_components']} "
        f"largest={checks['largest_component']} "
        f"singletons={checks['singletons']} "
        f"pruned={checks.get('pruned_directed_edges')} "
        f"self_candidates={checks.get('self_loop_candidates')}"
    )

    if bad:
        print("CHECK WARNING:", msg, "|", ", ".join(bad))
    else:
        print("CHECK OK:", msg)

def auto_neighbor_cutoff(values: np.ndarray, measurement: str) -> float:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    values = values[values > 0]

    if values.size < 10:
        return np.inf

    cfg = EDGE_MEASUREMENTS[measurement]

    if cfg.get("max_neighbor_distance") is not None:
        return float(cfg["max_neighbor_distance"])

    q = float(cfg.get("cutoff_quantile", 0.98))
    mad_scale = float(cfg.get("cutoff_mad", 6.0))

    quantile_cutoff = float(np.quantile(values, q))

    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))

    if mad < 1e-6:
        return quantile_cutoff

    mad_cutoff = med + mad_scale * 1.4826 * mad

    cutoff = min(quantile_cutoff, mad_cutoff)

    if not np.isfinite(cutoff) or cutoff <= 0:
        return np.inf

    return float(cutoff)

def make_edges(
    nodes: pd.DataFrame,
    blocks: dict[str, np.ndarray],
    k: int,
    measurement: str,
) -> pd.DataFrame:
    n = nodes.shape[0]

    cols = [
        "measurement",
        "source",
        "target",
        "source_cell_id",
        "target_cell_id",

        "distance",
        "neighbor_distance",
        "spatial_scaled_distance",
        "profile_distance",
        "morphology_distance",
        "combo_distance",

        "dx",
        "dy",
        "abs_dx",
        "abs_dy",
        "angle",
        "direction_x",
        "direction_y",

        "same_spatial_island",

        "total_counts_diff",
        "detected_genes_diff",
        "cell_area_diff",
        "nucleus_area_diff",
        "nucleus_cell_ratio_diff",

        "spatial_weight",
        "profile_weight",
        "morphology_weight",
        "combo_weight",
    ]

    if n < 2:
        return pd.DataFrame(columns=cols)

    embed = edge_embedding(blocks, measurement)

    kk = min(k, n - 1)
    neighbor_distance, idx = (
        NearestNeighbors(n_neighbors=kk + 1, algorithm="kd_tree")
        .fit(embed)
        .kneighbors(embed)
    )

    source = np.repeat(np.arange(n), kk)
    target = idx[:, 1:].ravel()
    neighbor_distance = neighbor_distance[:, 1:].ravel()

    raw_candidate_directed_edges = int(neighbor_distance.size)

    not_self = source != target
    self_loop_candidates = int((~not_self).sum())

    source = source[not_self]
    target = target[not_self]
    neighbor_distance = neighbor_distance[not_self]

    candidate_directed_edges = int(neighbor_distance.size)
    neighbor_cutoff = auto_neighbor_cutoff(neighbor_distance, measurement)

    keep = neighbor_distance <= neighbor_cutoff

    source = source[keep]
    target = target[keep]
    neighbor_distance = neighbor_distance[keep]

    kept_directed_edges = int(neighbor_distance.size)
    pruned_directed_edges = candidate_directed_edges - kept_directed_edges

    edges = pd.DataFrame(
        {
            "source": np.minimum(source, target),
            "target": np.maximum(source, target),
            "neighbor_distance": neighbor_distance,
        }
    )

    edges = edges.sort_values("neighbor_distance")
    duplicated_undirected_edges = int(edges.duplicated(["source", "target"]).sum())
    edges = edges.drop_duplicates(["source", "target"])
    edges = edges.sort_values(["source", "target"]).reset_index(drop=True)

    edges.attrs["neighbor_cutoff"] = float(neighbor_cutoff)
    edges.attrs["raw_candidate_directed_edges"] = raw_candidate_directed_edges
    edges.attrs["candidate_directed_edges"] = candidate_directed_edges
    edges.attrs["kept_directed_edges"] = kept_directed_edges
    edges.attrs["pruned_directed_edges"] = pruned_directed_edges
    edges.attrs["self_loop_candidates"] = self_loop_candidates
    edges.attrs["duplicated_undirected_edges"] = duplicated_undirected_edges

    src = edges["source"].to_numpy(dtype=np.int64)
    dst = edges["target"].to_numpy(dtype=np.int64)

    x = nodes["x_centroid"].to_numpy(dtype=np.float32)
    y = nodes["y_centroid"].to_numpy(dtype=np.float32)

    dx = x[dst] - x[src]
    dy = y[dst] - y[src]
    distance = np.sqrt(dx * dx + dy * dy).astype(np.float32)

    spatial_scaled_distance = pair_block_distance(blocks["spatial"], src, dst)
    profile_distance = pair_block_distance(blocks["profile"], src, dst)
    morphology_distance = pair_block_distance(blocks["morphology"], src, dst)

    weights = EDGE_MEASUREMENTS[measurement]["weights"]

    combo_distance = np.sqrt(
        (weights["spatial"] * spatial_scaled_distance) ** 2
        + (weights["profile"] * profile_distance) ** 2
        + (weights["morphology"] * morphology_distance) ** 2
    ).astype(np.float32)

    ids = nodes["cell_id"].to_numpy(dtype=str)
    island = nodes["island_id"].to_numpy(dtype=np.int32)

    log_total = nodes["log_total_counts"].to_numpy(dtype=np.float32)
    log_detected = nodes["log_detected_genes"].to_numpy(dtype=np.float32)
    cell_area = nodes["cell_area"].to_numpy(dtype=np.float32)
    nucleus_area = nodes["nucleus_area"].to_numpy(dtype=np.float32)
    nucleus_ratio = nodes["nucleus_cell_ratio"].to_numpy(dtype=np.float32)

    edges["measurement"] = EDGE_MEASUREMENTS[measurement]["label"]

    edges["distance"] = distance
    edges["spatial_scaled_distance"] = spatial_scaled_distance
    edges["profile_distance"] = profile_distance
    edges["morphology_distance"] = morphology_distance
    edges["combo_distance"] = combo_distance

    edges["dx"] = dx
    edges["dy"] = dy
    edges["abs_dx"] = np.abs(dx)
    edges["abs_dy"] = np.abs(dy)
    edges["angle"] = np.arctan2(dy, dx)
    edges["direction_x"] = dx / (distance + 1e-6)
    edges["direction_y"] = dy / (distance + 1e-6)

    edges["same_spatial_island"] = (island[src] == island[dst]).astype(np.int8)

    edges["total_counts_diff"] = np.abs(log_total[dst] - log_total[src])
    edges["detected_genes_diff"] = np.abs(log_detected[dst] - log_detected[src])
    edges["cell_area_diff"] = np.abs(cell_area[dst] - cell_area[src])
    edges["nucleus_area_diff"] = np.abs(nucleus_area[dst] - nucleus_area[src])
    edges["nucleus_cell_ratio_diff"] = np.abs(nucleus_ratio[dst] - nucleus_ratio[src])

    edges["spatial_weight"] = 1.0 / (1.0 + spatial_scaled_distance)
    edges["profile_weight"] = 1.0 / (1.0 + profile_distance)
    edges["morphology_weight"] = 1.0 / (1.0 + morphology_distance)
    edges["combo_weight"] = 1.0 / (1.0 + combo_distance)

    edges["source_cell_id"] = ids[src]
    edges["target_cell_id"] = ids[dst]

    return edges[cols]


def representation(nodes: pd.DataFrame, edges: pd.DataFrame):
    features = [
        "x_aligned",
        "y_aligned",

        "total_counts",
        "log_total_counts",
        "n_detected_genes",
        "log_detected_genes",

        "cell_area",
        "nucleus_area",
        "nucleus_cell_ratio",

        "spatial_island_id",
        "spatial_island_size",

        "component_id",
        "component_size",
    ]

    edge_features = [
        "distance",
        "neighbor_distance",
        "spatial_scaled_distance",
        "profile_distance",
        "morphology_distance",
        "combo_distance",

        "dx",
        "dy",
        "abs_dx",
        "abs_dy",
        "angle",
        "direction_x",
        "direction_y",

        "same_spatial_island",
        "same_component",

        "total_counts_diff",
        "detected_genes_diff",
        "cell_area_diff",
        "nucleus_area_diff",
        "nucleus_cell_ratio_diff",

        "spatial_weight",
        "profile_weight",
        "morphology_weight",
        "combo_weight",

        "source_component_id",
        "target_component_id",
        "source_component_size",
        "target_component_size",
    ]

    x = nodes[features].to_numpy(dtype=np.float32)
    pos = nodes[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
    ids = nodes["cell_id"].to_numpy(dtype=str)

    undirected = pd.concat(
        [
            edges[["source", "target"]],
            edges.rename(columns={"source": "target", "target": "source"})[
                ["source", "target"]
            ],
        ],
        ignore_index=True,
    )

    edge_index = undirected.to_numpy(dtype=np.int64).T

    attr = edges[edge_features].to_numpy(dtype=np.float32)

    edge_attr = np.vstack(
        [
            attr,
            attr,
        ]
    )

    return x, pos, edge_index, edge_attr, ids, features, edge_features

def plot_cells(nodes: pd.DataFrame, out_path: Path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 7))
    plt.scatter(
        nodes["x_centroid"],
        nodes["y_centroid"],
        c=nodes["island_id"],
        s=6,
        alpha=0.85,
    )
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_graph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    out_path: Path,
    metric: str | None = None,
    node_color: str = "component_id",
):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    pos = nodes[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
    pairs = edges[["source", "target"]].to_numpy(dtype=np.int64)

    fig, ax = plt.subplots(figsize=(7, 7))

    if pairs.size:
        lines = LineCollection(pos[pairs], linewidths=0.25, alpha=0.25)

        if metric is not None and metric in edges:
            lines.set_array(edges[metric].to_numpy(dtype=np.float32))
            fig.colorbar(lines, ax=ax, label=metric)

        ax.add_collection(lines)

    color = nodes[node_color] if node_color in nodes else nodes["spatial_island_id"]

    ax.scatter(
        nodes["x_centroid"],
        nodes["y_centroid"],
        c=color,
        s=6,
        alpha=0.85,
    )

    ax.invert_yaxis()
    ax.axis("equal")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_nodes_by_metric(nodes: pd.DataFrame, out_path: Path, metric: str):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 7))
    plt.scatter(
        nodes["x_centroid"],
        nodes["y_centroid"],
        c=nodes[metric],
        s=6,
        alpha=0.85,
    )
    plt.colorbar(label=metric)
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_edge_hist(edges: pd.DataFrame, out_path: Path):
    import matplotlib.pyplot as plt

    cols = [
        "distance",
        "neighbor_distance",
        "spatial_scaled_distance",
        "profile_distance",
        "morphology_distance",
        "combo_distance",
    ]

    cols = [c for c in cols if c in edges]

    fig, axes = plt.subplots(len(cols), 1, figsize=(8, 2.2 * len(cols)))

    if len(cols) == 1:
        axes = [axes]

    for ax, c in zip(axes, cols):
        values = edges[c].to_numpy(dtype=np.float32)
        values = values[np.isfinite(values)]

        ax.hist(values, bins=60, alpha=0.8)
        ax.set_title(c)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def write_summary(
    out_dir: Path,
    adata,
    selected,
    nodes: pd.DataFrame,
    features: list[str],
    edge_features: list[str],
    k: int,
    measurements: list[str],
    edge_summaries: dict,
):
    summary = {
        **adata.uns["xenium_dump"],
        "crop_mode": selected.uns["crop_mode"],
        "n_cells_loaded": int(adata.n_obs),
        "n_cells_dumped": int(selected.n_obs),
        "n_features": int(adata.n_vars),
        "k": int(k),
        "measurements": measurements,
        "edge_measurements": EDGE_MEASUREMENTS,
        "edge_summaries": edge_summaries,
        "n_spatial_islands": int(nodes["spatial_island_id"].nunique()),
        "node_feature_columns": features,
        "edge_feature_columns": edge_features,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_id")
    p.add_argument("--max-cells", type=int, default=50_000)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--only", choices=["all", *EDGE_MEASUREMENTS.keys()], default="all")
    return p.parse_args()

def dump_one_graph(
    out_dir: Path,
    nodes: pd.DataFrame,
    blocks: dict[str, np.ndarray],
    k: int,
    measurement: str,
):
    edges = make_edges(nodes, blocks, k, measurement)

    graph_nodes = nodes_for_graph(nodes, edges)
    edges = add_graph_components_to_edges(edges, graph_nodes)

    checks = graph_checks(nodes, graph_nodes, edges, measurement)
    print_graph_checks(checks)

    x, pos, edge_index, edge_attr, ids, features, edge_features = representation(
        graph_nodes,
        edges,
    )

    graph_nodes.to_csv(out_dir / f"nodes_{measurement}.csv", index=False)
    edges.to_csv(out_dir / f"edges_{measurement}.csv", index=False)
    (out_dir / f"checks_{measurement}.json").write_text(
        json.dumps(checks, indent=2) + "\n"
    )

    np.savez_compressed(
        out_dir / f"representation_{measurement}.npz",
        node_features=x,
        node_positions=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        cell_ids=ids,
        feature_columns=np.asarray(features, dtype=str),
        edge_feature_columns=np.asarray(edge_features, dtype=str),
        measurement=np.asarray([EDGE_MEASUREMENTS[measurement]["label"]], dtype=str),
    )

    plot_graph(
        graph_nodes,
        edges,
        out_dir / f"knn_graph_{measurement}.png",
        node_color="component_id",
    )

    plot_graph(
        graph_nodes,
        edges,
        out_dir / f"weighted_graph_{measurement}.png",
        "combo_weight",
        node_color="component_id",
    )

    plot_edge_hist(edges, out_dir / f"edge_metrics_{measurement}.png")

    if measurement == "mix":
        graph_nodes.to_csv(out_dir / "nodes_graph.csv", index=False)
        edges.to_csv(out_dir / "edges.csv", index=False)

        np.savez_compressed(
            out_dir / "representation.npz",
            node_features=x,
            node_positions=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            cell_ids=ids,
            feature_columns=np.asarray(features, dtype=str),
            edge_feature_columns=np.asarray(edge_features, dtype=str),
            measurement=np.asarray([EDGE_MEASUREMENTS[measurement]["label"]], dtype=str),
        )

        plot_graph(
            graph_nodes,
            edges,
            out_dir / "knn_graph.png",
            node_color="component_id",
        )

    return edges, graph_nodes, features, edge_features, checks

def main():
    args = parse_args()

    xenium_dir = DATA_ROOT / args.dataset_id
    out_dir = OUTPUT_ROOT / args.dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)

    adata = load_xenium(xenium_dir)
    selected = select_cells(adata, args.max_cells, args.k)

    nodes = make_nodes(selected)
    nodes.to_csv(out_dir / "nodes.csv", index=False)
    blocks = edge_blocks(nodes)

    plot_cells(nodes, out_dir / "cells.png")
    plot_nodes_by_metric(nodes, out_dir / "counts.png", "log_total_counts")
    plot_nodes_by_metric(nodes, out_dir / "nucleus_ratio.png", "nucleus_cell_ratio")

    measurements = (
        list(EDGE_MEASUREMENTS.keys())
        if args.only == "all"
        else [args.only]
    )

    edge_summaries = {}
    features = []
    edge_features = []

    for measurement in measurements:
        edges, graph_nodes, features, edge_features, checks = dump_one_graph(
            out_dir,
            nodes,
            blocks,
            args.k,
            measurement,
        )

        edge_summaries[measurement] = {
            "label": EDGE_MEASUREMENTS[measurement]["label"],
            "n_edges": int(edges.shape[0]),
            "n_components": int(graph_nodes["component_id"].nunique()),
            "largest_component": int(graph_nodes["component_size"].max()),
            "singletons": int((graph_nodes["component_size"] == 1).sum()),
            "weights": EDGE_MEASUREMENTS[measurement]["weights"],
            "neighbor_cutoff": edges.attrs.get("neighbor_cutoff"),
            "raw_candidate_directed_edges": edges.attrs.get("raw_candidate_directed_edges"),
            "candidate_directed_edges": edges.attrs.get("candidate_directed_edges"),
            "kept_directed_edges": edges.attrs.get("kept_directed_edges"),
            "pruned_directed_edges": edges.attrs.get("pruned_directed_edges"),
            "self_loop_candidates": edges.attrs.get("self_loop_candidates", 0),
            "duplicated_undirected_edges": edges.attrs.get("duplicated_undirected_edges", 0),
            "checks": checks,
            "node_csv": f"nodes_{measurement}.csv",
            "edge_csv": f"edges_{measurement}.csv",
            "checks_json": f"checks_{measurement}.json",
            "representation_npz": f"representation_{measurement}.npz",
        }

        print(f"{measurement}: edges={edges.shape[0]}")

    write_summary(
        out_dir,
        adata,
        selected,
        nodes,
        features,
        edge_features,
        args.k,
        measurements,
        edge_summaries,
    )

    print(f"saved: {out_dir}")
    print(f"cells: {nodes.shape[0]}")
    print(f"spatial islands: {nodes['spatial_island_id'].nunique()}")


if __name__ == "__main__":
    main()
