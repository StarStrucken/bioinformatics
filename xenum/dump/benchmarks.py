from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import multiprocessing as mp

import numpy as np
import pandas as pd

from xenum_measurements import MEASUREMENTS

from .config import BEST_K_MIN_PRED_SPREAD_RATIO, K, LEARNED_MIN_COVERAGE, LEARNED_MIN_PRED_SPREAD_RATIO, LEARNED_MIX_NAME, LEARNED_MIX_WORKERS, LEARNED_P90_WEIGHT, LEARNED_WEIGHT_VALUES, tqdm
from .graph import checks, edges_from_neighbor_lists, load_or_make_pairs, neighbor_lists_from_pairs, prediction_from_edges
from .io import write_pair_cache
from .npz import write_npz

_LEARNED_CONTEXT = {}

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

    if "pred_spread_ratio" in clean.columns:
        strict = clean[clean["pred_spread_ratio"] >= BEST_K_MIN_PRED_SPREAD_RATIO].copy()
        if not strict.empty:
            clean = strict

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

def learned_mix_candidate(row, weights, kk, label):
    spread_ok = (
        row.get("pred_spread_ratio") is None
        or row.get("pred_spread_ratio") >= LEARNED_MIN_PRED_SPREAD_RATIO
    )

    if row["coverage"] < LEARNED_MIN_COVERAGE or not spread_ok or row["median_xy_error"] is None:
        return None

    median_err = float(row["median_xy_error"])

    if row["p90_xy_error"] is None or not np.isfinite(row["p90_xy_error"]):
        p90_err = np.inf
    else:
        p90_err = float(row["p90_xy_error"])

    main_score = median_err + LEARNED_P90_WEIGHT * p90_err

    return {
        "score": (
            main_score,
            median_err,
            p90_err,
            int(kk),
            label,
        ),
        "weights": weights.copy(),
    }

def evaluate_learned_weights(index, weights, pair_tables, dataset_id, nodes, blocks, node_cols, bench_k_values):
    label = learned_label(weights)
    pairs = combine_pair_tables(pair_tables, weights)
    best = neighbor_lists_from_pairs(len(nodes), pairs)
    rows = []
    best_item = None

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

        item = learned_mix_candidate(row, weights, kk, label)
        if item is not None and (best_item is None or item["score"] < best_item["score"]):
            best_item = item

    return index, rows, best_item

def evaluate_learned_weights_from_context(index, weights):
    ctx = _LEARNED_CONTEXT
    return evaluate_learned_weights(
        index,
        weights,
        ctx["pair_tables"],
        ctx["dataset_id"],
        ctx["nodes"],
        ctx["blocks"],
        ctx["node_cols"],
        ctx["bench_k_values"],
    )

def learned_mix_process_context():
    try:
        return mp.get_context("fork")
    except ValueError:
        return None

def learned_mix_grid_valid(grid, base_measurements):
    required = {
        "k",
        "coverage",
        "median_xy_error",
        "p90_xy_error",
    }

    if grid.empty or not required.issubset(grid.columns):
        return False

    for name in base_measurements:
        if f"weight_{name}" not in grid.columns:
            return False

    return True

def learned_mix_weights_from_row(row, base_measurements):
    return {
        name: float(row[f"weight_{name}"])
        for name in base_measurements
    }

def learned_mix_best_from_grid(grid, base_measurements):
    best_item = None

    for row in grid.to_dict("records"):
        weights = learned_mix_weights_from_row(row, base_measurements)
        label = str(row.get("label") or learned_label(weights))
        item = learned_mix_candidate(row, weights, int(row["k"]), label)

        if item is not None and (best_item is None or item["score"] < best_item["score"]):
            best_item = item

    return best_item

def write_learned_mix_top(out_dir, grid):
    top = grid.copy()
    top["score"] = top["median_xy_error"] + LEARNED_P90_WEIGHT * top["p90_xy_error"]

    if "pred_spread_ratio" in top.columns:
        top = top.sort_values(
            ["score", "median_xy_error", "p90_xy_error", "pred_spread_ratio", "k"],
            ascending=[True, True, True, False, True],
        ).head(30)
    else:
        top = top.sort_values(["score", "median_xy_error", "p90_xy_error", "k"]).head(30)

    top.to_csv(out_dir / "learned_mix_top.csv", index=False)
    return top

def run_learned_mix(out_dir, dataset_id, nodes, blocks, node_cols, bench_k_values, base_measurements):
    pair_tables = {}

    for m in base_measurements:
        pair_tables[m] = load_or_make_pairs(out_dir, nodes[node_cols], blocks, m)

    rows = []
    best_item = None

    MEASUREMENTS[LEARNED_MIX_NAME] = {
        "label": LEARNED_MIX_NAME,
        "blocks": {},
    }

    grid_path = out_dir / "learned_mix_grid.csv"

    if grid_path.exists():
        grid = pd.read_csv(grid_path)

        if learned_mix_grid_valid(grid, base_measurements):
            print(f"learned mix grid reused: {grid_path}", flush=True)
            best_item = learned_mix_best_from_grid(grid, base_measurements)
            write_learned_mix_top(out_dir, grid)
        else:
            print(f"learned mix grid ignored: incompatible {grid_path}", flush=True)
            grid = None
    else:
        grid = None

    if grid is None:
        weight_sets = list(learned_weight_grid(base_measurements, LEARNED_WEIGHT_VALUES))
        workers = min(int(LEARNED_MIX_WORKERS), len(weight_sets))

        print(f"learned mix weight sets={len(weight_sets)} workers={workers}", flush=True)

        if workers <= 1:
            results = [
                evaluate_learned_weights(
                    i,
                    weights,
                    pair_tables,
                    dataset_id,
                    nodes,
                    blocks,
                    node_cols,
                    bench_k_values,
                )
                for i, weights in enumerate(tqdm(weight_sets, desc="learned mix"))
            ]
        else:
            results = []
            mp_context = learned_mix_process_context()

            if mp_context is not None:
                executor_cls = ProcessPoolExecutor
                executor_kwargs = {
                    "max_workers": workers,
                    "mp_context": mp_context,
                }
                submit_args = [
                    (evaluate_learned_weights_from_context, i, weights)
                    for i, weights in enumerate(weight_sets)
                ]
                _LEARNED_CONTEXT.update({
                    "pair_tables": pair_tables,
                    "dataset_id": dataset_id,
                    "nodes": nodes,
                    "blocks": blocks,
                    "node_cols": node_cols,
                    "bench_k_values": bench_k_values,
                })
                worker_mode = "processes"
            else:
                executor_cls = ThreadPoolExecutor
                executor_kwargs = {
                    "max_workers": workers,
                }
                submit_args = [
                    (
                        evaluate_learned_weights,
                        i,
                        weights,
                        pair_tables,
                        dataset_id,
                        nodes,
                        blocks,
                        node_cols,
                        bench_k_values,
                    )
                    for i, weights in enumerate(weight_sets)
                ]
                worker_mode = "threads"

            print(f"learned mix worker mode={worker_mode}", flush=True)

            try:
                with executor_cls(**executor_kwargs) as pool:
                    futures = [
                        pool.submit(fn, *args)
                        for fn, *args in submit_args
                    ]

                    for future in tqdm(as_completed(futures), total=len(futures), desc="learned mix"):
                        results.append(future.result())
            finally:
                _LEARNED_CONTEXT.clear()

            results.sort(key=lambda x: x[0])

        for _, part_rows, item in results:
            rows.extend(part_rows)

            if item is not None and (best_item is None or item["score"] < best_item["score"]):
                best_item = item

        grid = pd.DataFrame(rows)
        grid.to_csv(grid_path, index=False)
        write_learned_mix_top(out_dir, grid)

    if best_item is None:
        return [], {}

    best_weights = best_item["weights"]
    best_label = learned_label(best_weights)

    MEASUREMENTS[LEARNED_MIX_NAME] = {
        "label": best_label,
        "blocks": {},
    }

    pairs = combine_pair_tables(pair_tables, best_weights)
    write_pair_cache(out_dir, LEARNED_MIX_NAME, pairs)

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
        "base_measurements": list(base_measurements),
        "weight_values": list(LEARNED_WEIGHT_VALUES),
        "min_coverage": float(LEARNED_MIN_COVERAGE),
        "p90_weight": float(LEARNED_P90_WEIGHT),
    }

    (out_dir / "learned_mix_weights.json").write_text(json.dumps(payload, indent=2) + "\n")

    return bench_rows, summaries
