from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import multiprocessing as mp

import numpy as np
import pandas as pd

from xenum_measurements import MEASUREMENTS

from .config import BEST_K_MIN_PRED_SPREAD_RATIO, K, LEARNED_MIN_COVERAGE, LEARNED_MIN_PRED_SPREAD_RATIO, LEARNED_MIX_MODE, LEARNED_MIX_NAME, LEARNED_MIX_OUTPUT_K, LEARNED_MIX_WORKERS, LEARNED_P90_WEIGHT, LEARNED_WEIGHT_VALUES, tqdm
from .graph import checks, edges_from_neighbor_lists, finish_edges, load_or_make_pairs, neighbor_lists_from_pairs, prediction_from_edges
from .io import write_pair_cache
from .npz import write_npz

_LEARNED_CONTEXT = {}

RANDOM_SELECTION_MEASUREMENTS = {
    "random_permutation",
    "random_neighbors",
}

BENCH_REQUIRED_COLUMNS = [
    "dataset",
    "measurement",
    "k",
    "seed",
    "leaky",
    "coverage",
    "mean_xy_error",
    "median_xy_error",
    "p90_xy_error",
    "status",
]

def normalize_benchmark_rows(df):
    df = df.copy()

    for col in BENCH_REQUIRED_COLUMNS:
        if col not in df.columns:
            if col == "seed":
                df[col] = None
            elif col == "status":
                df[col] = "ok"
            elif col == "leaky":
                df[col] = False
            else:
                df[col] = np.nan

    df["status"] = df["status"].fillna("ok").astype(str)
    if df["leaky"].dtype == object:
        df["leaky"] = df["leaky"].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
    else:
        df["leaky"] = df["leaky"].fillna(False).astype(bool)

    return df

def benchmark_sort_columns(df):
    df = df.copy()
    for col in ["median_xy_error", "mean_xy_error", "p90_xy_error", "k", "seed"]:
        if col not in df.columns:
            df[col] = np.nan

    df["_median_sort"] = df["median_xy_error"].fillna(np.inf)
    df["_mean_sort"] = df["mean_xy_error"].fillna(np.inf)
    df["_p90_sort"] = df["p90_xy_error"].fillna(np.inf)
    df["_k_sort"] = pd.to_numeric(df["k"], errors="coerce").fillna(0).astype(int)
    df["_seed_sort"] = pd.to_numeric(df["seed"], errors="coerce").fillna(np.inf)
    return df

def selected_benchmark_rows(df, min_coverage=0.95):
    df = normalize_benchmark_rows(df)
    ok = df[df["status"].eq("ok")].copy()
    ok = ok.dropna(subset=["median_xy_error", "mean_xy_error", "p90_xy_error"])

    if ok.empty:
        return pd.DataFrame(columns=df.columns)

    selected = []

    for measurement, group in ok.groupby("measurement", sort=False):
        group = group.copy()

        if measurement not in RANDOM_SELECTION_MEASUREMENTS:
            covered = group[group["coverage"] >= float(min_coverage)].copy()
            if not covered.empty:
                group = covered

            if "pred_spread_ratio" in group.columns:
                strict = group[group["pred_spread_ratio"] >= BEST_K_MIN_PRED_SPREAD_RATIO].copy()
                if not strict.empty:
                    group = strict

            if "median_vs_spatial_best" in group.columns:
                with_ref = group.dropna(subset=["median_vs_spatial_best"]).copy()
                if not with_ref.empty:
                    group = with_ref

        work = benchmark_sort_columns(group)

        if measurement not in RANDOM_SELECTION_MEASUREMENTS and "median_vs_spatial_best" in work.columns:
            work["_spatial_sort"] = work["median_vs_spatial_best"].fillna(np.inf)
            sort_cols = ["_spatial_sort", "_median_sort", "_mean_sort", "_p90_sort", "_k_sort", "_seed_sort"]
        else:
            sort_cols = ["_median_sort", "_mean_sort", "_p90_sort", "_k_sort", "_seed_sort"]

        row = work.sort_values(sort_cols).drop(columns=[c for c in work.columns if c.startswith("_")]).head(1)
        selected.append(row)

    return pd.concat(selected, ignore_index=True, sort=False) if selected else pd.DataFrame(columns=df.columns)

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
    clean = normalize_benchmark_rows(df)
    clean = selected_benchmark_rows(clean)
    clean = clean.dropna(subset=["median_vs_spatial_best"])

    if clean.empty:
        return pd.DataFrame(), pd.DataFrame()

    clean["rank"] = clean.groupby("dataset")["median_vs_spatial_best"].rank(method="min")

    summary = (
        clean
        .groupby(["measurement", "k"])
        .agg(
            datasets=("dataset", "nunique"),
            leaky=("leaky", "max"),
            wins=("rank", lambda x: int((x == 1).sum())),
            rank_mean=("rank", "mean"),
            mean_xy_error_median=("mean_xy_error", "median"),
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
            "seed",
            "leaky",
            "mean_xy_error",
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
    best = selected_benchmark_rows(df, min_coverage=min_coverage)

    if best.empty:
        return pd.DataFrame()

    cols = [
        "measurement",
        "k",
        "seed",
        "leaky",
        "status",
        "mean_xy_error",
        "median_xy_error",
        "p90_xy_error",
        "coverage",
        "median_vs_spatial_best",
        "median_vs_spatial_same_k",
        "spatial_best_k",
        "spatial_best_median_xy_error",
    ]
    cols = [c for c in cols if c in best.columns]
    best = best[cols].reset_index(drop=True)

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

def pair_distance_scale(pairs):
    d = pairs["distance"].to_numpy(dtype=np.float32)
    ok = np.isfinite(d) & (d > 0)

    if ok.any():
        scale = float(np.median(d[ok]))
    else:
        scale = 1.0

    if scale <= 0 or not np.isfinite(scale):
        scale = 1.0

    return scale

def learned_base_k_map(base_measurements, base_k_by_measurement):
    out = {}

    for name in base_measurements:
        raw = None if base_k_by_measurement is None else base_k_by_measurement.get(name)

        try:
            kk = int(raw)
        except (TypeError, ValueError):
            kk = int(K)

        out[name] = max(1, kk)

    return out

def base_k_candidate(row, kk):
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

    return (
        median_err + LEARNED_P90_WEIGHT * p90_err,
        median_err,
        p90_err,
        int(kk),
    )

def infer_learned_base_k(dataset_id, nodes, blocks, node_cols, measurement, pairs, bench_k_values):
    best = neighbor_lists_from_pairs(len(nodes), pairs)
    best_score = None
    best_k = int(K)

    for kk in bench_k_values:
        edges, _graph_nodes = edges_from_neighbor_lists(
            nodes[node_cols],
            blocks,
            measurement,
            best,
            kk,
        )
        _pred_df, row = prediction_from_edges(
            dataset_id,
            nodes[node_cols],
            edges,
            measurement,
            kk,
        )
        score = base_k_candidate(row, kk)

        if score is not None and (best_score is None or score < best_score):
            best_score = score
            best_k = int(kk)

    return best_k

def fill_missing_learned_base_k(dataset_id, nodes, blocks, node_cols, pair_tables, bench_k_values, base_measurements, base_k_by_measurement):
    out = learned_base_k_map(base_measurements, base_k_by_measurement)
    supplied = set(base_k_by_measurement or {})

    for name in base_measurements:
        if name in supplied:
            continue

        out[name] = infer_learned_base_k(
            dataset_id,
            nodes,
            blocks,
            node_cols,
            name,
            pair_tables[name],
            bench_k_values,
        )
        print(f"learned mix inferred base k: {name} k={out[name]}", flush=True)

    return out

def learned_neighbor_candidates(n, pair_tables, base_k_by_measurement):
    parts = []
    base_k = learned_base_k_map(pair_tables.keys(), base_k_by_measurement)

    for name, pairs in pair_tables.items():
        kk = base_k[name]
        scale = pair_distance_scale(pairs)
        best = neighbor_lists_from_pairs(n, pairs)
        rows = []

        for source, vals in enumerate(best):
            for rank, (target, dist) in enumerate(vals[:kk], start=1):
                normalized_distance = float(dist) / scale

                if not np.isfinite(normalized_distance):
                    continue

                rank_weight = 1.0 / float(rank)
                distance_weight = 1.0 / (1.0 + max(0.0, normalized_distance))
                base_score = rank_weight * distance_weight

                if base_score <= 0 or not np.isfinite(base_score):
                    continue

                rows.append((
                    int(source),
                    int(target),
                    name,
                    int(rank),
                    float(normalized_distance),
                    float(base_score),
                ))

        if rows:
            parts.append(pd.DataFrame(
                rows,
                columns=[
                    "source",
                    "target",
                    "measurement",
                    "rank",
                    "normalized_distance",
                    "base_score",
                ],
            ))

    if not parts:
        return pd.DataFrame(columns=[
            "source",
            "target",
            "measurement",
            "rank",
            "normalized_distance",
            "base_score",
        ]), base_k

    return pd.concat(parts, ignore_index=True), base_k

def learned_directed_neighbors(candidates, weights):
    if candidates.empty:
        return pd.DataFrame(columns=[
            "source",
            "target",
            "neighbor_distance",
            "neighbor_weight",
            "support_measurements",
            "best_rank",
            "min_normalized_distance",
        ])

    work = candidates.copy()
    work["measurement_weight"] = work["measurement"].map(weights).astype(float)
    work = work[work["measurement_weight"] > 0].copy()

    if work.empty:
        return pd.DataFrame(columns=[
            "source",
            "target",
            "neighbor_distance",
            "neighbor_weight",
            "support_measurements",
            "best_rank",
            "min_normalized_distance",
        ])

    work["neighbor_weight"] = work["measurement_weight"] * work["base_score"]
    work = work[np.isfinite(work["neighbor_weight"]) & (work["neighbor_weight"] > 0)].copy()

    if work.empty:
        return pd.DataFrame(columns=[
            "source",
            "target",
            "neighbor_distance",
            "neighbor_weight",
            "support_measurements",
            "best_rank",
            "min_normalized_distance",
        ])

    directed = (
        work
        .groupby(["source", "target"], sort=False)
        .agg(
            neighbor_weight=("neighbor_weight", "sum"),
            support_measurements=("measurement", "nunique"),
            best_rank=("rank", "min"),
            min_normalized_distance=("normalized_distance", "min"),
        )
        .reset_index()
    )
    directed["neighbor_distance"] = 1.0 / directed["neighbor_weight"]

    return directed.sort_values(["source", "neighbor_distance", "target"]).reset_index(drop=True)

def learned_edges_from_directed(nodes, blocks, directed):
    if directed.empty:
        base = pd.DataFrame(columns=["source", "target", "neighbor_distance", "neighbor_weight"])
    else:
        source = directed["source"].to_numpy(dtype=np.int64)
        target = directed["target"].to_numpy(dtype=np.int64)

        base = pd.DataFrame({
            "source": np.minimum(source, target),
            "target": np.maximum(source, target),
            "neighbor_weight": directed["neighbor_weight"].to_numpy(dtype=np.float64),
        })
        base = (
            base
            .groupby(["source", "target"], sort=False)
            .agg(neighbor_weight=("neighbor_weight", "sum"))
            .reset_index()
        )
        base["neighbor_distance"] = 1.0 / base["neighbor_weight"]
        base = base.sort_values(["source", "target"]).reset_index(drop=True)

    base.attrs["neighbor_cutoff"] = np.inf
    base.attrs["raw_directed_edges"] = int(len(directed))
    base.attrs["kept_directed_edges"] = int(len(directed))
    base.attrs["pruned_directed_edges"] = 0

    return finish_edges(nodes, blocks, LEARNED_MIX_NAME, base)

def learned_undirected_edge_count(directed):
    if directed.empty:
        return 0

    source = directed["source"].to_numpy(dtype=np.int64)
    target = directed["target"].to_numpy(dtype=np.int64)

    pairs = pd.DataFrame({
        "source": np.minimum(source, target),
        "target": np.maximum(source, target),
    })

    return int(pairs.drop_duplicates(["source", "target"]).shape[0])

def learned_pair_cache_from_edges(edges):
    pairs = edges[["source", "target", "neighbor_distance", "neighbor_weight"]].copy()
    pairs = pairs.rename(columns={"neighbor_distance": "distance"})
    return pairs

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

def annotate_learned_row(row, weights, base_k_by_measurement, label):
    row["label"] = label
    row["mix_mode"] = LEARNED_MIX_MODE

    for name, kk in base_k_by_measurement.items():
        row[f"k_{name}"] = int(kk)

    for name, w in weights.items():
        row[f"weight_{name}"] = float(w)

    return row

def evaluate_learned_weights(index, weights, candidates, dataset_id, nodes, blocks, node_cols, base_k_by_measurement):
    label = learned_label(weights)
    directed = learned_directed_neighbors(candidates, weights)

    _pred_df, row = prediction_from_edges(
        dataset_id,
        nodes[node_cols],
        directed,
        LEARNED_MIX_NAME,
        LEARNED_MIX_OUTPUT_K,
        weight_col="neighbor_weight",
        directed=True,
    )

    row["n_union_edges"] = learned_undirected_edge_count(directed)
    row["n_directed_neighbors"] = int(len(directed))
    annotate_learned_row(row, weights, base_k_by_measurement, label)

    best_item = learned_mix_candidate(row, weights, LEARNED_MIX_OUTPUT_K, label)

    return index, [row], best_item

def evaluate_learned_weights_from_context(index, weights):
    ctx = _LEARNED_CONTEXT
    return evaluate_learned_weights(
        index,
        weights,
        ctx["candidates"],
        ctx["dataset_id"],
        ctx["nodes"],
        ctx["blocks"],
        ctx["node_cols"],
        ctx["base_k_by_measurement"],
    )

def learned_mix_process_context():
    try:
        return mp.get_context("fork")
    except ValueError:
        return None

def learned_mix_grid_valid(grid, base_measurements, base_k_by_measurement):
    required = {
        "k",
        "mix_mode",
        "coverage",
        "median_xy_error",
        "p90_xy_error",
    }

    if grid.empty or not required.issubset(grid.columns):
        return False

    if not grid["mix_mode"].astype(str).eq(LEARNED_MIX_MODE).all():
        return False

    for name in base_measurements:
        if f"weight_{name}" not in grid.columns or f"k_{name}" not in grid.columns:
            return False

        try:
            if not grid[f"k_{name}"].astype(int).eq(int(base_k_by_measurement[name])).all():
                return False
        except (TypeError, ValueError):
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

def run_learned_mix(out_dir, dataset_id, nodes, blocks, node_cols, bench_k_values, base_measurements, base_k_by_measurement=None):
    pair_tables = {}

    for m in base_measurements:
        pair_tables[m] = load_or_make_pairs(out_dir, nodes[node_cols], blocks, m)

    learned_base_k = fill_missing_learned_base_k(
        dataset_id,
        nodes,
        blocks,
        node_cols,
        pair_tables,
        bench_k_values,
        base_measurements,
        base_k_by_measurement,
    )
    candidates, learned_base_k = learned_neighbor_candidates(
        len(nodes),
        pair_tables,
        learned_base_k,
    )

    print(f"learned mix mode={LEARNED_MIX_MODE} base_k={learned_base_k}", flush=True)

    if candidates.empty:
        print("learned mix skipped: no neighbor candidates", flush=True)
        return [], {}

    rows = []
    best_item = None

    MEASUREMENTS[LEARNED_MIX_NAME] = {
        "label": LEARNED_MIX_NAME,
        "blocks": {},
    }

    grid_path = out_dir / "learned_mix_grid.csv"

    if grid_path.exists():
        grid = pd.read_csv(grid_path)

        if learned_mix_grid_valid(grid, base_measurements, learned_base_k):
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
                    candidates,
                    dataset_id,
                    nodes,
                    blocks,
                    node_cols,
                    learned_base_k,
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
                    "candidates": candidates,
                    "dataset_id": dataset_id,
                    "nodes": nodes,
                    "blocks": blocks,
                    "node_cols": node_cols,
                    "base_k_by_measurement": learned_base_k,
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
                        candidates,
                        dataset_id,
                        nodes,
                        blocks,
                        node_cols,
                        learned_base_k,
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

    directed = learned_directed_neighbors(candidates, best_weights)
    edges, graph_nodes = learned_edges_from_directed(
        nodes[node_cols],
        blocks,
        directed,
    )
    write_pair_cache(out_dir, LEARNED_MIX_NAME, learned_pair_cache_from_edges(edges))

    chk = checks(nodes, graph_nodes, edges, LEARNED_MIX_NAME)
    pred_df, bench_row = prediction_from_edges(
        dataset_id,
        nodes[node_cols],
        directed,
        LEARNED_MIX_NAME,
        LEARNED_MIX_OUTPUT_K,
        weight_col="neighbor_weight",
        directed=True,
    )
    bench_row["n_union_edges"] = int(len(edges))
    bench_row["n_directed_neighbors"] = int(len(directed))
    annotate_learned_row(bench_row, best_weights, learned_base_k, best_label)

    graph_nodes.to_csv(out_dir / f"nodes_{LEARNED_MIX_NAME}_k{LEARNED_MIX_OUTPUT_K}.csv", index=False)
    edges.to_csv(out_dir / f"edges_{LEARNED_MIX_NAME}_k{LEARNED_MIX_OUTPUT_K}.csv", index=False)
    pred_df.to_csv(out_dir / f"predictions_{LEARNED_MIX_NAME}_k{LEARNED_MIX_OUTPUT_K}.csv", index=False)
    (out_dir / f"checks_{LEARNED_MIX_NAME}_k{LEARNED_MIX_OUTPUT_K}.json").write_text(json.dumps(chk, indent=2) + "\n")

    graph_nodes.to_csv(out_dir / f"nodes_{LEARNED_MIX_NAME}.csv", index=False)
    edges.to_csv(out_dir / f"edges_{LEARNED_MIX_NAME}.csv", index=False)
    (out_dir / f"checks_{LEARNED_MIX_NAME}.json").write_text(json.dumps(chk, indent=2) + "\n")
    write_npz(out_dir, LEARNED_MIX_NAME, graph_nodes, edges)

    bench_rows = [bench_row]
    summaries = {
        LEARNED_MIX_NAME: chk,
    }

    payload = {
        "measurement": LEARNED_MIX_NAME,
        "mix_mode": LEARNED_MIX_MODE,
        "label": best_label,
        "k": int(LEARNED_MIX_OUTPUT_K),
        "weights": best_weights,
        "base_k_by_measurement": learned_base_k,
        "base_measurements": list(base_measurements),
        "weight_values": list(LEARNED_WEIGHT_VALUES),
        "min_coverage": float(LEARNED_MIN_COVERAGE),
        "p90_weight": float(LEARNED_P90_WEIGHT),
        "neighbor_weight_formula": "measurement_weight * (1 / rank) * (1 / (1 + normalized_distance))",
    }

    (out_dir / "learned_mix_weights.json").write_text(json.dumps(payload, indent=2) + "\n")

    return bench_rows, summaries
