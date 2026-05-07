#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

CELL_TABLE_NAMES = ("cells.csv.gz", "cells.csv", "cells.parquet")


def find_file(root: Path, names: tuple[str, ...] | list[str]) -> Path:
    search_roots = [root]
    if root.name != "bundle":
        search_roots.append(root / "bundle")

    for search_root in search_roots:
        for name in names:
            candidate = search_root / name
            if candidate.exists():
                return candidate

    for search_root in search_roots:
        for name in names:
            matches = sorted(search_root.rglob(name))
            if matches:
                return matches[0]

    joined = ", ".join(names)
    raise FileNotFoundError(f"Could not find any of [{joined}] under {root}")


def read_cells_table(xenium_dir: Path) -> tuple[pd.DataFrame, Path]:
    cells_path = find_file(xenium_dir, CELL_TABLE_NAMES)

    if cells_path.suffixes[-2:] == [".csv", ".gz"] or cells_path.suffix == ".csv":
        cells = pd.read_csv(cells_path)
    elif cells_path.suffix == ".parquet":
        cells = pd.read_parquet(cells_path)
    else:
        raise ValueError(f"Unsupported cells table format: {cells_path}")

    if "cell_id" not in cells.columns:
        raise ValueError("Cells table must contain a `cell_id` column.")

    required = {"x_centroid", "y_centroid"}
    missing = sorted(required - set(cells.columns))
    if missing:
        raise ValueError(f"Cells table is missing required columns: {missing}")

    cells = cells.copy()
    cells["cell_id"] = cells["cell_id"].astype(str)
    cells = cells.drop_duplicates(subset="cell_id").set_index("cell_id")
    return cells, cells_path


def load_xenium_dataset(xenium_dir: Path):
    import scanpy as sc

    matrix_path = find_file(xenium_dir, ("cell_feature_matrix.h5",))
    adata = sc.read_10x_h5(matrix_path, gex_only=False)
    adata.var_names_make_unique()
    adata.obs_names = adata.obs_names.astype(str)

    feature_source = "all_features"
    if "feature_types" in adata.var.columns:
        gene_mask = adata.var["feature_types"].astype(str) == "Gene Expression"
        if bool(gene_mask.any()):
            adata = adata[:, gene_mask].copy()
            feature_source = "Gene Expression"

    cells, cells_path = read_cells_table(xenium_dir)
    shared_ids = adata.obs_names.intersection(cells.index)
    if len(shared_ids) == 0:
        raise ValueError("No matching cell IDs between the matrix and the cells table.")

    adata = adata[shared_ids].copy()
    cells = cells.loc[adata.obs_names].copy()

    adata.obs["cell_id"] = adata.obs_names.astype(str)
    adata.obs["x_centroid"] = pd.to_numeric(cells["x_centroid"], errors="coerce")
    adata.obs["y_centroid"] = pd.to_numeric(cells["y_centroid"], errors="coerce")

    if "cell_area" in cells.columns:
        adata.obs["cell_area"] = pd.to_numeric(cells["cell_area"], errors="coerce")
    else:
        adata.obs["cell_area"] = np.nan

    if "nucleus_area" in cells.columns:
        adata.obs["nucleus_area"] = pd.to_numeric(cells["nucleus_area"], errors="coerce")
    else:
        adata.obs["nucleus_area"] = np.nan

    spatial = adata.obs[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
    valid = np.isfinite(spatial).all(axis=1)
    if not bool(valid.all()):
        adata = adata[valid].copy()
        spatial = spatial[valid]

    adata.obsm["spatial"] = spatial
    adata.uns["xenium_dump"] = {
        "xenium_dir": str(xenium_dir),
        "matrix_path": str(matrix_path),
        "cells_path": str(cells_path),
        "feature_source": feature_source,
    }
    return adata


def select_spatial_subset(adata, n_cells: int):
    if n_cells <= 0:
        raise ValueError("n_cells must be positive.")

    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    center = np.nanmedian(coords, axis=0)
    distances = np.linalg.norm(coords - center, axis=1)
    selected_idx = np.argsort(distances)[: min(n_cells, adata.n_obs)]

    selected = adata[selected_idx].copy()
    selected.obsm["spatial"] = coords[selected_idx].copy()
    selected.obs["distance_to_crop_center"] = distances[selected_idx]
    selected.uns["crop_center"] = [float(center[0]), float(center[1])]
    return selected


def _per_cell_counts(x_matrix) -> tuple[np.ndarray, np.ndarray]:
    total_counts = np.asarray(x_matrix.sum(axis=1)).ravel()

    if sp.issparse(x_matrix):
        n_detected = np.asarray(x_matrix.getnnz(axis=1)).ravel()
    else:
        n_detected = np.asarray((x_matrix > 0).sum(axis=1)).ravel()

    return total_counts.astype(np.float32), n_detected.astype(np.int32)


def make_nodes_table(adata) -> pd.DataFrame:
    total_counts, n_detected_genes = _per_cell_counts(adata.X)

    nodes = pd.DataFrame(
        {
            "node_index": np.arange(adata.n_obs, dtype=np.int64),
            "cell_id": adata.obs["cell_id"].to_numpy(dtype=str),
            "x_centroid": adata.obsm["spatial"][:, 0],
            "y_centroid": adata.obsm["spatial"][:, 1],
            "total_counts": total_counts,
            "n_detected_genes": n_detected_genes,
            "cell_area": pd.to_numeric(adata.obs["cell_area"], errors="coerce").to_numpy(),
            "nucleus_area": pd.to_numeric(
                adata.obs["nucleus_area"], errors="coerce"
            ).to_numpy(),
        }
    )

    nodes["cell_area"] = nodes["cell_area"].fillna(0.0)
    nodes["nucleus_area"] = nodes["nucleus_area"].fillna(0.0)

    safe_cell_area = nodes["cell_area"].replace(0.0, np.nan)
    ratio = nodes["nucleus_area"].divide(safe_cell_area)
    nodes["nucleus_cell_area_ratio"] = (
        ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    return nodes


def make_knn_edges(nodes: pd.DataFrame, k: int) -> pd.DataFrame:
    if nodes.shape[0] < 2:
        return pd.DataFrame(
            columns=[
                "source",
                "target",
                "source_cell_id",
                "target_cell_id",
                "distance",
            ]
        )

    positions = nodes[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
    effective_k = min(max(1, k), positions.shape[0] - 1)

    knn = NearestNeighbors(n_neighbors=effective_k + 1)
    knn.fit(positions)
    distances, indices = knn.kneighbors(positions)

    edge_distance: dict[tuple[int, int], float] = {}
    for source in range(positions.shape[0]):
        for neighbor_offset in range(1, effective_k + 1):
            target = int(indices[source, neighbor_offset])
            pair = tuple(sorted((source, target)))
            distance = float(distances[source, neighbor_offset])

            previous = edge_distance.get(pair)
            if previous is None or distance < previous:
                edge_distance[pair] = distance

    rows = []
    cell_ids = nodes["cell_id"].to_numpy(dtype=str)
    for (source, target), distance in sorted(edge_distance.items()):
        rows.append(
            {
                "source": source,
                "target": target,
                "source_cell_id": cell_ids[source],
                "target_cell_id": cell_ids[target],
                "distance": distance,
            }
        )

    return pd.DataFrame(rows)


def build_representation_arrays(nodes: pd.DataFrame, edges: pd.DataFrame):
    feature_columns = [
        "total_counts",
        "n_detected_genes",
        "cell_area",
        "nucleus_area",
        "nucleus_cell_area_ratio",
    ]
    node_features = nodes[feature_columns].to_numpy(dtype=np.float32)
    node_positions = nodes[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
    cell_ids = nodes["cell_id"].to_numpy(dtype=str)

    if edges.empty:
        edge_index = np.empty((2, 0), dtype=np.int64)
    else:
        undirected_edges = pd.concat(
            [
                edges[["source", "target"]],
                edges.rename(columns={"source": "target", "target": "source"})[
                    ["source", "target"]
                ],
            ],
            ignore_index=True,
        )
        edge_index = undirected_edges.to_numpy(dtype=np.int64).T

    return node_features, node_positions, edge_index, cell_ids, feature_columns


def plot_selected_cells(nodes: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 7))
    plt.scatter(nodes["x_centroid"], nodes["y_centroid"], s=6, alpha=0.8)
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Selected Xenium cells")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_knn_graph(nodes: pd.DataFrame, edges: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    positions = nodes[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)

    plt.figure(figsize=(7, 7))
    for source, target in edges[["source", "target"]].to_numpy(dtype=np.int64):
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        plt.plot([x1, x2], [y1, y2], linewidth=0.4, alpha=0.2, color="tab:gray")

    plt.scatter(nodes["x_centroid"], nodes["y_centroid"], s=6, alpha=0.8, color="tab:blue")
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("kNN graph over Xenium cell coordinates")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_summary(
    out_dir: Path,
    adata,
    selected,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    feature_columns: list[str],
    k_requested: int,
) -> None:
    x_values = nodes["x_centroid"].to_numpy(dtype=np.float32)
    y_values = nodes["y_centroid"].to_numpy(dtype=np.float32)

    summary = {
        "xenium_dir": adata.uns["xenium_dump"]["xenium_dir"],
        "matrix_path": adata.uns["xenium_dump"]["matrix_path"],
        "cells_path": adata.uns["xenium_dump"]["cells_path"],
        "expression_source": adata.uns["xenium_dump"]["feature_source"],
        "n_cells_loaded": int(adata.n_obs),
        "n_cells_selected": int(selected.n_obs),
        "n_expression_features": int(adata.n_vars),
        "n_edges_undirected": int(edges.shape[0]),
        "k_neighbors_requested": int(k_requested),
        "k_neighbors_effective": int(min(max(1, k_requested), max(selected.n_obs - 1, 1))),
        "node_feature_columns": feature_columns,
        "crop_center": selected.uns["crop_center"],
        "x_range": [float(x_values.min()), float(x_values.max())],
        "y_range": [float(y_values.min()), float(y_values.max())],
        "files": {
            "nodes_csv": "nodes.csv",
            "edges_csv": "edges.csv",
            "summary_json": "summary.json",
            "representation_npz": "representation.npz",
            "selected_cells_plot": "selected_cells.png",
            "knn_graph_plot": "knn_graph.png",
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a Xenium bundle, dump a small graph-like cell representation, and plot it.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--xenium-dir", type=Path, required=True, help="Xenium output directory.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--n-cells", type=int, default=2000, help="Cells in the spatial subset.")
    parser.add_argument("--k", type=int, default=6, help="Neighbors per node for the kNN graph.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    adata = load_xenium_dataset(args.xenium_dir)
    selected = select_spatial_subset(adata, n_cells=args.n_cells)

    nodes = make_nodes_table(selected)
    edges = make_knn_edges(nodes, k=args.k)
    node_features, node_positions, edge_index, cell_ids, feature_columns = (
        build_representation_arrays(nodes, edges)
    )

    nodes.to_csv(args.out_dir / "nodes.csv", index=False)
    edges.to_csv(args.out_dir / "edges.csv", index=False)
    np.savez_compressed(
        args.out_dir / "representation.npz",
        node_features=node_features,
        node_positions=node_positions,
        edge_index=edge_index,
        cell_ids=cell_ids,
        feature_columns=np.asarray(feature_columns, dtype=str),
    )

    plot_selected_cells(nodes, args.out_dir / "selected_cells.png")
    plot_knn_graph(nodes, edges, args.out_dir / "knn_graph.png")
    write_summary(
        args.out_dir,
        adata=adata,
        selected=selected,
        nodes=nodes,
        edges=edges,
        feature_columns=feature_columns,
        k_requested=args.k,
    )

    print(f"Saved representation to {args.out_dir}")
    print(f"Selected cells: {nodes.shape[0]}")
    print(f"Undirected edges: {edges.shape[0]}")
    print(f"Node feature shape: {node_features.shape}")
    print(f"Edge index shape: {edge_index.shape}")


if __name__ == "__main__":
    main()
