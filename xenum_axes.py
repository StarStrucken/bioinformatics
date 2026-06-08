from pathlib import Path

import numpy as np
import pandas as pd

from xenum_common import clean_vec, parse_seq_ids, zscore

def build_axis_space(nodes, out_dir: Path):
    axis = {}
    presets = {}

    add_axis(axis, "spatial.x", nodes["x_centroid"].to_numpy(dtype=np.float32))
    add_axis(axis, "spatial.y", nodes["y_centroid"].to_numpy(dtype=np.float32))
    presets["xy"] = ("spatial.x", "spatial.y", True)

    if have(nodes, ["x_aligned", "y_aligned"]):
        add_axis(axis, "spatial.norm.x", nodes["x_aligned"].to_numpy(dtype=np.float32))
        add_axis(axis, "spatial.norm.y", nodes["y_aligned"].to_numpy(dtype=np.float32))
        presets["xy norm"] = ("spatial.norm.x", "spatial.norm.y", True)

    if have(nodes, ["x_centered", "y_centered"]):
        add_axis(axis, "spatial.centered.x", nodes["x_centered"].to_numpy(dtype=np.float32))
        add_axis(axis, "spatial.centered.y", nodes["y_centered"].to_numpy(dtype=np.float32))
        presets["xy centered"] = ("spatial.centered.x", "spatial.centered.y", True)
    else:
        add_axis(axis, "spatial.centered.x", axis["spatial.x"] - axis["spatial.x"].mean())
        add_axis(axis, "spatial.centered.y", axis["spatial.y"] - axis["spatial.y"].mean())
        presets["xy centered"] = ("spatial.centered.x", "spatial.centered.y", True)

    profile_cols = ["log_total_counts", "log_detected_genes"]
    morph_cols = ["cell_area", "nucleus_area", "nucleus_cell_ratio"]

    profile_raw = nodes[profile_cols].to_numpy(dtype=np.float32)
    profile = zscore(profile_raw)
    morphology_raw = nodes[morph_cols].to_numpy(dtype=np.float32)
    morphology = zscore(morphology_raw)

    add_axis(axis, "profile.log_total_counts_z", profile[:, 0])
    add_axis(axis, "profile.log_detected_genes_z", profile[:, 1])

    for col, vals in zip(morph_cols, morphology.T):
        add_axis(axis, f"morphology.{col}_z", vals)

    expr_cols = [c for c in nodes.columns if c.startswith("expression_pc") or c.startswith("expr_pc")]
    if expr_cols:
        expression = zscore(nodes[expr_cols].to_numpy(dtype=np.float32))
        for col, vals in zip(expr_cols, expression.T):
            add_axis(axis, f"expression.{col}", vals)
        expression_pc = pca2(expression)
        add_pair(axis, presets, "expression", expression_pc, False)

    profile_pc = pca2(profile)
    morphology_pc = pca2(morphology)
    pm_pc = pca2(np.c_[profile * 0.35, morphology * 0.50])

    add_pair(axis, presets, "profile", profile_pc, False)
    add_pair(axis, presets, "morphology", morphology_pc, False)
    add_pair(axis, presets, "profile+morphology", pm_pc, False)

    presets["profile x morphology"] = ("profile.x", "morphology.x", False)

    if "top_gene_ids" in nodes:
        seqs = [parse_seq_ids(v) for v in nodes["top_gene_ids"].to_numpy()]
        add_axis(axis, "seq.length", np.asarray([len(s) for s in seqs], dtype=np.float32))
        add_axis(axis, "seq.unique_genes", np.asarray([len(set(s)) for s in seqs], dtype=np.float32))
        add_axis(axis, "seq.first_gene_id", np.asarray([s[0] if s else -1 for s in seqs], dtype=np.float32))

    theta = np.arctan2(axis["spatial.centered.y"], axis["spatial.centered.x"])
    radius = np.sqrt(axis["spatial.centered.x"] ** 2 + axis["spatial.centered.y"] ** 2)
    add_axis(axis, "polar.theta", theta)
    add_axis(axis, "polar.radius_z", zscore(radius)[:, 0])
    presets["polar xy"] = ("polar.theta", "polar.radius_z", False)

    for col in [
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
    ]:
        if col in nodes and pd.api.types.is_numeric_dtype(nodes[col]):
            add_axis(axis, f"raw.{col}", nodes[col].to_numpy(dtype=np.float32))

    extra_axis, extra_presets = load_extra_axes(out_dir, len(nodes))
    for k, v in extra_axis.items():
        add_axis(axis, k, v)
    for k, v in extra_presets.items():
        if v[0] in axis and v[1] in axis:
            presets[f"hpc {k}"] = v

    return axis, presets

def load_extra_axes(out_dir: Path, n: int):
    axis = {}
    presets = {}

    paths = []
    for name in ["coord_axes.npz", "coords.npz", "projections.npz"]:
        p = out_dir / name
        if p.exists():
            paths.append(p)

    coord_dir = out_dir / "coords"
    if coord_dir.exists():
        paths.extend(sorted(coord_dir.glob("*.npz")))

    for path in paths:
        try:
            data = np.load(path, allow_pickle=False)
        except Exception:
            continue

        for key in data.files:
            arr = np.asarray(data[key])
            base = str(key).replace(" ", "_")

            if arr.ndim == 1 and arr.shape[0] == n:
                add_axis(axis, base, arr)
            elif arr.ndim == 2 and arr.shape[0] == n and arr.shape[1] >= 2:
                add_pair(axis, presets, base, arr[:, :2], False)

    for name in ["coord_axes.csv", "coords.csv", "projections.csv"]:
        path = out_dir / name
        if not path.exists():
            continue

        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        if len(df) != n:
            continue

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                add_axis(axis, col, df[col].to_numpy(dtype=np.float32))

        for a, b in [("umap1", "umap2"), ("umap_1", "umap_2"), ("layout_x", "layout_y"), ("fa_x", "fa_y")]:
            if a in axis and b in axis:
                presets[a.rsplit("_", 1)[0] if "_" in a else a[:-1]] = (a, b, False)

    return axis, presets

def pca2(x):
    x = zscore(x)

    if x.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)

    if x.shape[1] == 1:
        return np.c_[x[:, 0], np.zeros(x.shape[0], dtype=np.float32)]

    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    n = min(2, vt.shape[0])
    out = x @ vt[:n].T

    if n == 1:
        out = np.c_[out[:, 0], np.zeros(out.shape[0], dtype=np.float32)]

    return zscore(out).astype(np.float32)

def have(nodes, cols):
    return set(cols).issubset(nodes.columns)

def add_axis(axis, name, values):
    if name in axis:
        name = f"extra.{name}"
    axis[name] = clean_vec(values)

def add_pair(axis, presets, name, xy, invert_y=False):
    xy = np.asarray(xy, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] < 2:
        return

    xk = f"{name}.x"
    yk = f"{name}.y"
    add_axis(axis, xk, xy[:, 0])
    add_axis(axis, yk, xy[:, 1])
    presets[name] = (xk, yk, bool(invert_y))

