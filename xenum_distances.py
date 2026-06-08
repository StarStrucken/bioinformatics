import numpy as np

from xenum_common import parse_seq_ids, sequence_distance, zscore
from xenum_measurements import MEASUREMENTS

def distance_blocks(nodes):
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

    expr_cols = [c for c in nodes.columns if c.startswith("expression_pc") or c.startswith("expr_pc")]
    expression = zscore(nodes[expr_cols].to_numpy(dtype=np.float32)) if expr_cols else profile

    if "top_gene_ids" in nodes:
        align = [parse_seq_ids(v) for v in nodes["top_gene_ids"].to_numpy()]
    else:
        align = [() for _ in range(len(nodes))]

    return {
        "spatial": spatial,
        "profile": profile,
        "expression": expression,
        "morphology": morphology,
        "align": align,
    }


def pair_distance(blocks, measurement, a, b):
    if measurement in {"align", "seq_local", "seq_jaccard", "seq_blast"}:
        return sequence_distance(blocks["align"][a], blocks["align"][b], measurement)

    if measurement in {"mix_nonspatial", "mix"}:
        ds = blocks["spatial"][a] - blocks["spatial"][b]
        de = blocks["expression"][a] - blocks["expression"][b]
        dm = blocks["morphology"][a] - blocks["morphology"][b]
        w_spatial = 0.0 if measurement == "mix_nonspatial" else 1.0
        return float(np.sqrt((w_spatial * np.linalg.norm(ds)) ** 2 + np.linalg.norm(de) ** 2 + np.linalg.norm(dm) ** 2))

    if measurement == "expr_morph":
        de = blocks["expression"][a] - blocks["expression"][b]
        dm = blocks["morphology"][a] - blocks["morphology"][b]
        return float(np.sqrt(np.sum(de * de) + np.sum(dm * dm)))

    if measurement == "expr_morph_spatial":
        ds = blocks["spatial"][a] - blocks["spatial"][b]
        de = blocks["expression"][a] - blocks["expression"][b]
        dm = blocks["morphology"][a] - blocks["morphology"][b]
        return float(np.sqrt(np.sum(ds * ds) + np.sum(de * de) + np.sum(dm * dm)))

    if measurement not in blocks:
        return float("nan")

    v = blocks[measurement][a] - blocks[measurement][b]
    return float(np.sqrt(np.sum(v * v)))
