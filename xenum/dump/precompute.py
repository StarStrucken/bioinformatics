#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from xenum_measurements import HIDDEN_MEASUREMENTS, OPTIONAL_MEASUREMENTS, VISIBLE_MEASUREMENTS
from xenum_paths import out_dir as make_out_dir
from xenum_paths import data_dir

from .config import CACHE_DIR, EXPRESSION_PCS, NODE_BASE_COLS, TOP_GENES_PER_CELL
from .features import (
    available_measurements,
    build_blocks,
    detected_gene_ids,
    load_morphology_image_blocks,
    make_nodes,
    measurement_available,
    top_gene_ids,
)
from .graph import load_or_make_pairs
from .io import load_xenium, make_output_sections

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_id")
    return p.parse_args()

def main():
    args = parse_args()

    xenium_dir = data_dir(args.dataset_id)
    out_dir = make_out_dir(args.dataset_id)
    dirs = make_output_sections(out_dir)
    cache_dir = dirs["cache"]

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
    nodes[node_cols].to_csv(cache_dir / "nodes_precomputed.csv", index=False)

    measurements = available_measurements(blocks)

    for m in measurements:
        print(f"precompute pairs: {m}", flush=True)
        load_or_make_pairs(out_dir, nodes[node_cols], blocks, m)

    summary = {
        **adata.uns["xenium_clean"],
        "dataset_id": args.dataset_id,
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "expression_pcs": int(EXPRESSION_PCS),
        "measurements": measurements,
        "visible_measurements": list(VISIBLE_MEASUREMENTS),
        "optional_measurements": list(OPTIONAL_MEASUREMENTS),
        "hidden_measurements": list(HIDDEN_MEASUREMENTS),
        "morphology_image_features_path": morph_img_path,
        "morphology_image_feature_count": len(morph_img_cols),
        "morphology_image_blocks": morph_img_meta,
        "node_columns": node_cols,
    }

    (cache_dir / "precompute_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"precompute saved: {cache_dir}", flush=True)

if __name__ == "__main__":
    main()
