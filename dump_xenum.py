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

    return pd.DataFrame(
        {
            "node_index": np.arange(adata.n_obs, dtype=np.int64),
            "cell_id": adata.obs["cell_id"].to_numpy(dtype=str),
            "x_centroid": adata.obsm["spatial"][:, 0],
            "y_centroid": adata.obsm["spatial"][:, 1],
            "island_id": adata.obs["island_id"].to_numpy(dtype=np.int32),
            "total_counts": total,
            "n_detected_genes": detected,
            "cell_area": adata.obs["cell_area"].to_numpy(),
            "nucleus_area": adata.obs["nucleus_area"].to_numpy(),
        }
    )


def make_edges(nodes: pd.DataFrame, k: int) -> pd.DataFrame:
    n = nodes.shape[0]

    if n < 2:
        return pd.DataFrame(
            columns=[
                "source",
                "target",
                "source_cell_id",
                "target_cell_id",
                "distance",
            ]
        )

    coords = nodes[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
    kk = min(k, n - 1)

    dist, idx = NearestNeighbors(n_neighbors=kk + 1).fit(coords).kneighbors(coords)

    source = np.repeat(np.arange(n), kk)
    target = idx[:, 1:].ravel()
    distance = dist[:, 1:].ravel()

    edges = pd.DataFrame(
        {
            "source": np.minimum(source, target),
            "target": np.maximum(source, target),
            "distance": distance,
        }
    )

    edges = edges.sort_values("distance")
    edges = edges.drop_duplicates(["source", "target"])
    edges = edges.sort_values(["source", "target"])

    ids = nodes["cell_id"].to_numpy(dtype=str)

    edges["source_cell_id"] = ids[edges["source"].to_numpy()]
    edges["target_cell_id"] = ids[edges["target"].to_numpy()]

    return edges[
        [
            "source",
            "target",
            "source_cell_id",
            "target_cell_id",
            "distance",
        ]
    ]


def representation(nodes: pd.DataFrame, edges: pd.DataFrame):
    features = [
        "total_counts",
        "n_detected_genes",
        "cell_area",
        "nucleus_area",
        "island_id",
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

    return x, pos, edge_index, ids, features


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


def plot_graph(nodes: pd.DataFrame, edges: pd.DataFrame, out_path: Path):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    pos = nodes[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
    pairs = edges[["source", "target"]].to_numpy(dtype=np.int64)

    fig, ax = plt.subplots(figsize=(7, 7))

    if pairs.size:
        ax.add_collection(LineCollection(pos[pairs], linewidths=0.25, alpha=0.2))

    ax.scatter(
        nodes["x_centroid"],
        nodes["y_centroid"],
        c=nodes["island_id"],
        s=6,
        alpha=0.85,
    )

    ax.invert_yaxis()
    ax.axis("equal")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_summary(
    out_dir: Path,
    adata,
    selected,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    features: list[str],
    k: int,
):
    summary = {
        **adata.uns["xenium_dump"],
        "crop_mode": selected.uns["crop_mode"],
        "n_cells_loaded": int(adata.n_obs),
        "n_cells_dumped": int(selected.n_obs),
        "n_features": int(adata.n_vars),
        "k": int(k),
        "n_edges": int(edges.shape[0]),
        "n_islands": int(nodes["island_id"].nunique()),
        "node_feature_columns": features,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--xenium-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-cells", type=int, default=50_000)
    p.add_argument("--k", type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    adata = load_xenium(args.xenium_dir)
    selected = select_cells(adata, args.max_cells, args.k)

    nodes = make_nodes(selected)
    edges = make_edges(nodes, args.k)

    x, pos, edge_index, ids, features = representation(nodes, edges)

    nodes.to_csv(args.out_dir / "nodes.csv", index=False)
    edges.to_csv(args.out_dir / "edges.csv", index=False)

    np.savez_compressed(
        args.out_dir / "representation.npz",
        node_features=x,
        node_positions=pos,
        edge_index=edge_index,
        cell_ids=ids,
        feature_columns=np.asarray(features, dtype=str),
    )

    plot_cells(nodes, args.out_dir / "cells.png")
    plot_graph(nodes, edges, args.out_dir / "knn_graph.png")

    write_summary(args.out_dir, adata, selected, nodes, edges, features, args.k)

    print(f"saved: {args.out_dir}")
    print(f"cells: {nodes.shape[0]}")
    print(f"edges: {edges.shape[0]}")
    print(f"islands: {nodes['island_id'].nunique()}")


if __name__ == "__main__":
    main()
