from __future__ import annotations

import argparse
import json

import pandas as pd

from xenum_measurements import ACTIVE_MEASUREMENTS, CONTROL_MEASUREMENTS, HIDDEN_MEASUREMENTS, MEASUREMENTS, OPTIONAL_MEASUREMENTS, SPAGCN_MEASUREMENTS, VISIBLE_MEASUREMENTS
from xenum_paths import data_dir, out_dir as make_out_dir

from .benchmarks import add_spatial_reference, best_k_by_measurement, normalize_benchmark_rows, run_learned_mix, summarize_benchmarks
from .config import BENCH_K_VALUES, CUTOFF_MAD, CUTOFF_QUANTILE, DIAGNOSTICS_DIR, EDGE_COLS, EXPRESSION_PCS, K, LEARNED_BASE_MEASUREMENTS, LEARNED_MIN_COVERAGE, LEARNED_MIX_MODE, LEARNED_MIX_NAME, LEARNED_MIX_OUTPUT_K, LEARNED_WEIGHT_VALUES, MIN_EDGES_PER_NODE, NODE_BASE_COLS, REPORT_DIR, RUN_LEARNED_MIX, TOP_GENES_PER_CELL, USE_NEIGHBOR_CUTOFF, tqdm
from .features import available_measurements, build_blocks, detected_gene_ids, load_morphology_image_blocks, make_nodes, measurement_available, top_gene_ids
from .graph import checks, edges_from_neighbor_lists, load_or_make_pairs, neighbor_lists_from_pairs, prediction_from_edges
from .io import load_xenium, make_output_sections, mirror_outputs
from .npz import write_npz
from .random_controls import run_random_controls
from .spagcn import run_spagcn_measurements

def parse_k_list(s):
    vals = []

    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))

    return vals

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_id")
    return p.parse_args()

def main():
    args = parse_args()

    xenium_dir = data_dir(args.dataset_id)
    out_dir = make_out_dir(args.dataset_id)
    make_output_sections(out_dir)

    adata = load_xenium(xenium_dir)

    nodes = make_nodes(adata)
    nodes["top_gene_ids"] = top_gene_ids(adata.X, TOP_GENES_PER_CELL)
    nodes["detected_gene_ids"] = detected_gene_ids(adata.X)
    blocks = build_blocks(adata, nodes, EXPRESSION_PCS)

    morph_img_blocks = load_morphology_image_blocks(out_dir, nodes)
    morph_img_meta = {}

    for name, item in morph_img_blocks.items():
        block, cols, path = item
        blocks[name] = block
        morph_img_meta[name] = {
            "path": path,
            "feature_count": len(cols),
        }
        print(f"{name} loaded: {path} features={len(cols)}", flush=True)

    if not morph_img_blocks:
        print("morphology_image missing: skipped", flush=True)

    morph_img_path = morph_img_meta.get("morphology_image", {}).get("path")
    morph_img_cols = [None] * int(morph_img_meta.get("morphology_image", {}).get("feature_count", 0))

    expr_cols = []
    for i in range(blocks["expression"].shape[1]):
        col = f"expression_pc{i + 1}"
        nodes[col] = blocks["expression"][:, i]
        expr_cols.append(col)

    node_cols = NODE_BASE_COLS + ["top_gene_ids", "detected_gene_ids"] + expr_cols
    nodes[node_cols].to_csv(out_dir / "nodes.csv", index=False)

    summaries = {}
    bench_rows = []
    bench_k_values = sorted(set([int(K), *BENCH_K_VALUES]))
    measurement_names = available_measurements(blocks)

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

    random_rows, random_summaries = run_random_controls(
        out_dir,
        args.dataset_id,
        nodes[node_cols],
        blocks,
        bench_k_values,
    )
    bench_rows.extend(random_rows)
    summaries.update(random_summaries)

    spagcn_rows, spagcn_summaries = run_spagcn_measurements(
        out_dir,
        args.dataset_id,
        adata,
        nodes[node_cols],
        node_cols,
        blocks,
        bench_k_values,
    )
    bench_rows.extend(spagcn_rows)
    summaries.update(spagcn_summaries)

    learned_base_measurements = [
        m for m in LEARNED_BASE_MEASUREMENTS
        if measurement_available(m, blocks)
    ]
    learned_base_k = {}

    if bench_rows:
        preliminary_bench = add_spatial_reference(pd.DataFrame(bench_rows))
        preliminary_best_k = best_k_by_measurement(preliminary_bench, LEARNED_MIN_COVERAGE)
        learned_base_k = {
            str(r.measurement): int(r.k)
            for r in preliminary_best_k.itertuples(index=False)
            if str(r.measurement) in learned_base_measurements
        }

    if RUN_LEARNED_MIX:
        learned_rows, learned_summaries = run_learned_mix(
            out_dir,
            args.dataset_id,
            nodes,
            blocks,
            node_cols,
            bench_k_values,
            learned_base_measurements,
            learned_base_k,
        )
    else:
        learned_rows = []
        learned_summaries = {}
        print("learned mix skipped", flush=True)

    bench_rows.extend(learned_rows)
    summaries.update(learned_summaries)

    bench_df = normalize_benchmark_rows(pd.DataFrame(bench_rows)).sort_values(["leaky", "measurement", "k", "seed"], na_position="last")
    bench_df = add_spatial_reference(bench_df)

    bench_df.to_csv(out_dir / "bench_xy.csv", index=False)
    bench_df.to_csv(out_dir / "bench_xy_by_k.csv", index=False)

    bench_summary, bench_best = summarize_benchmarks(bench_df)
    bench_summary.to_csv(out_dir / "bench_xy_summary.csv", index=False)
    bench_best.to_csv(out_dir / "bench_xy_best.csv", index=False)

    best_k = best_k_by_measurement(bench_df)
    best_k.to_csv(out_dir / "best_k_by_measurement.csv", index=False)

    best_preview = (
        best_k
        .dropna(subset=["median_vs_spatial_best"])
        .sort_values(["median_vs_spatial_best", "median_xy_error"])
        .head(10)
    )
    summary_measurements = list(dict.fromkeys([
        *measurement_names,
        *CONTROL_MEASUREMENTS,
        *SPAGCN_MEASUREMENTS,
        *learned_base_measurements,
    ]))

    print()
    print("bench top selected:", flush=True)
    if best_preview.empty:
        print("(none)", flush=True)
    else:
        print(
            best_preview[
                [
                    "measurement",
                    "k",
                    "seed",
                    "leaky",
                    "coverage",
                    "mean_xy_error",
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
        "learned_mix_mode": LEARNED_MIX_MODE,
        "learned_mix_output_k": int(LEARNED_MIX_OUTPUT_K),
        "learned_base_measurements": list(learned_base_measurements),
        "learned_base_k_by_measurement": learned_base_k,
        "morphology_image_features_path": morph_img_path,
        "morphology_image_feature_count": len(morph_img_cols),
        "morphology_image_blocks": morph_img_meta,
        "learned_weight_values": list(LEARNED_WEIGHT_VALUES),
        "learned_min_coverage": float(LEARNED_MIN_COVERAGE),
        "expression_pcs": int(EXPRESSION_PCS),
        "use_neighbor_cutoff": bool(USE_NEIGHBOR_CUTOFF),
        "cutoff_quantile": float(CUTOFF_QUANTILE),
        "cutoff_mad": float(CUTOFF_MAD),
        "min_edges_per_node": int(MIN_EDGES_PER_NODE),
        "measurements": measurement_names,
        "control_measurements": list(CONTROL_MEASUREMENTS),
        "spagcn_measurements": list(SPAGCN_MEASUREMENTS),
        "visible_measurements": list(VISIBLE_MEASUREMENTS),
        "hidden_measurements": list(HIDDEN_MEASUREMENTS),
        "active_measurements": measurement_names,
        "measurement_defs": {m: MEASUREMENTS[m] for m in summary_measurements},
        "node_columns": node_cols,
        "edge_columns": EDGE_COLS,
        "edge_summaries": summaries,
    }, indent=2) + "\n")
    mirror_outputs(out_dir)

    print(f"saved: {out_dir}", flush=True)
    print(f"reports: {out_dir / REPORT_DIR}", flush=True)
    print(f"diagnostics: {out_dir / DIAGNOSTICS_DIR}", flush=True)

if __name__ == "__main__":
    main()
