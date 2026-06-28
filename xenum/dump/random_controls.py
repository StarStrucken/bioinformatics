from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .benchmarks import selected_benchmark_rows
from .config import K, RANDOM_SEEDS, tqdm
from .graph import checks, finish_edges, prediction_from_direct_coordinates, prediction_from_edges
from .npz import write_npz

RANDOM_PERMUTATION = "random_permutation"
RANDOM_NEIGHBORS = "random_neighbors"
RANDOM_ARTIFACT_PATTERNS = (
    f"predictions_{RANDOM_PERMUTATION}_seed*_k*.csv",
    f"predictions_{RANDOM_PERMUTATION}_k*.csv",
    f"nodes_{RANDOM_NEIGHBORS}_seed*_k*.csv",
    f"edges_{RANDOM_NEIGHBORS}_seed*_k*.csv",
    f"predictions_{RANDOM_NEIGHBORS}_seed*_k*.csv",
    f"checks_{RANDOM_NEIGHBORS}_seed*_k*.json",
    f"nodes_{RANDOM_NEIGHBORS}_k*.csv",
    f"edges_{RANDOM_NEIGHBORS}_k*.csv",
    f"predictions_{RANDOM_NEIGHBORS}_k*.csv",
    f"checks_{RANDOM_NEIGHBORS}_k*.json",
    f"nodes_{RANDOM_NEIGHBORS}.csv",
    f"edges_{RANDOM_NEIGHBORS}.csv",
    f"checks_{RANDOM_NEIGHBORS}.json",
    f"representation_{RANDOM_NEIGHBORS}.npz",
)

def cleanup_random_artifacts(out_dir):
    for pattern in RANDOM_ARTIFACT_PATTERNS:
        for path in out_dir.glob(pattern):
            if path.is_file():
                path.unlink()

def selected_row(rows):
    selected = selected_benchmark_rows(pd.DataFrame(rows))

    if selected.empty:
        return None

    return selected.iloc[0].to_dict()

def run_random_permutation(out_dir, dataset_id, nodes):
    xy = nodes[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
    rows = []

    for seed in tqdm(RANDOM_SEEDS, desc=RANDOM_PERMUTATION):
        rng = np.random.default_rng(int(seed))
        pred = xy[rng.permutation(len(xy))].copy()
        _pred_df, row = prediction_from_direct_coordinates(
            dataset_id,
            nodes,
            pred,
            RANDOM_PERMUTATION,
            seed=int(seed),
        )
        rows.append(row)

    best = selected_row(rows)

    if best is not None:
        seed = int(best["seed"])
        rng = np.random.default_rng(seed)
        pred = xy[rng.permutation(len(xy))].copy()
        pred_df, _row = prediction_from_direct_coordinates(
            dataset_id,
            nodes,
            pred,
            RANDOM_PERMUTATION,
            seed=seed,
        )
        pred_df.to_csv(out_dir / f"predictions_{RANDOM_PERMUTATION}_k0.csv", index=False)

    return rows, {}

def random_neighbor_edges(nodes, blocks, k, seed):
    n = len(nodes)
    rows = []
    k = max(0, min(int(k), max(0, n - 1)))
    rng = np.random.default_rng(int(seed))

    if k == 0 or n <= 1:
        base = pd.DataFrame(columns=["source", "target", "neighbor_distance"])
        base.attrs["neighbor_cutoff"] = np.inf
        base.attrs["raw_directed_edges"] = 0
        base.attrs["kept_directed_edges"] = 0
        base.attrs["pruned_directed_edges"] = 0
        return finish_edges(nodes, blocks, RANDOM_NEIGHBORS, base)

    all_nodes = np.arange(n, dtype=np.int64)

    for source in range(n):
        candidates = all_nodes[all_nodes != source]
        targets = rng.choice(candidates, size=k, replace=False)

        for rank, target in enumerate(targets, start=1):
            rows.append((min(source, int(target)), max(source, int(target)), float(rank)))

    base = pd.DataFrame(rows, columns=["source", "target", "neighbor_distance"])

    if len(base):
        base = base.sort_values(["neighbor_distance", "source", "target"])
        base = base.drop_duplicates(["source", "target"]).reset_index(drop=True)

    base.attrs["neighbor_cutoff"] = np.inf
    base.attrs["raw_directed_edges"] = int(n * k)
    base.attrs["kept_directed_edges"] = int(len(base))
    base.attrs["pruned_directed_edges"] = int(n * k - len(base))

    return finish_edges(nodes, blocks, RANDOM_NEIGHBORS, base)

def run_random_neighbors(out_dir, dataset_id, nodes, blocks, bench_k_values):
    rows = []
    summaries = {}

    for seed in tqdm(RANDOM_SEEDS, desc=RANDOM_NEIGHBORS):
        for kk in bench_k_values:
            edges, graph_nodes = random_neighbor_edges(nodes, blocks, kk, seed)
            pred_df, row = prediction_from_edges(
                dataset_id,
                nodes,
                edges,
                RANDOM_NEIGHBORS,
                kk,
                seed=int(seed),
            )
            rows.append(row)

    best = selected_row(rows)

    if best is not None:
        best_seed = int(best["seed"])
        best_k = int(best["k"])
        edges, graph_nodes = random_neighbor_edges(nodes, blocks, best_k, best_seed)
        chk = checks(nodes, graph_nodes, edges, RANDOM_NEIGHBORS)
        pred_df, _row = prediction_from_edges(
            dataset_id,
            nodes,
            edges,
            RANDOM_NEIGHBORS,
            best_k,
            seed=best_seed,
        )

        graph_nodes.to_csv(out_dir / f"nodes_{RANDOM_NEIGHBORS}_k{best_k}.csv", index=False)
        edges.to_csv(out_dir / f"edges_{RANDOM_NEIGHBORS}_k{best_k}.csv", index=False)
        pred_df.to_csv(out_dir / f"predictions_{RANDOM_NEIGHBORS}_k{best_k}.csv", index=False)
        (out_dir / f"checks_{RANDOM_NEIGHBORS}_k{best_k}.json").write_text(json.dumps(chk, indent=2) + "\n")
        write_npz(out_dir, RANDOM_NEIGHBORS, graph_nodes, edges)
        summaries[RANDOM_NEIGHBORS] = chk

    return rows, summaries

def run_random_controls(out_dir, dataset_id, nodes, blocks, bench_k_values):
    rows = []
    summaries = {}
    cleanup_random_artifacts(out_dir)

    permutation_rows, permutation_summaries = run_random_permutation(out_dir, dataset_id, nodes)
    neighbor_rows, neighbor_summaries = run_random_neighbors(out_dir, dataset_id, nodes, blocks, bench_k_values)

    rows.extend(permutation_rows)
    rows.extend(neighbor_rows)
    summaries.update(permutation_summaries)
    summaries.update(neighbor_summaries)

    return rows, summaries
