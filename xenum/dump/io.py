from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CACHE_DIR, CELL_TABLE_NAMES, DIAGNOSTICS_DIR, REPORT_DIR

def find_file(root: Path, names: tuple[str, ...]) -> Path:
    roots = [root] if root.name == "bundle" else [root, root / "bundle"]
    for base in roots:
        for name in names:
            p = base / name
            if p.exists():
                return p
    for base in roots:
        for name in names:
            for p in base.rglob(name):
                return p
    raise FileNotFoundError(names)

def make_output_sections(out_dir):
    report_dir = out_dir / REPORT_DIR
    report_tables_dir = report_dir / "tables"
    report_figures_dir = report_dir / "figures"
    cache_dir = out_dir / CACHE_DIR
    diagnostics_dir = out_dir / DIAGNOSTICS_DIR
    diagnostics_checks_dir = diagnostics_dir / "checks"

    for p in [
        report_dir,
        report_tables_dir,
        report_figures_dir,
        cache_dir,
        diagnostics_dir,
        diagnostics_checks_dir,
    ]:
        p.mkdir(parents=True, exist_ok=True)

    return {
        "reports": report_dir,
        "report_tables": report_tables_dir,
        "report_figures": report_figures_dir,
        "cache": cache_dir,
        "diagnostics": diagnostics_dir,
        "diagnostics_checks": diagnostics_checks_dir,
    }

def copy_if_exists(src, dst):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

def pair_cache_path(out_dir, measurement):
    return out_dir / CACHE_DIR / f"pairs_{measurement}.parquet"

def pair_legacy_path(out_dir, measurement):
    return out_dir / f"pairs_{measurement}.parquet"

def write_pair_cache(out_dir, measurement, pairs):
    path = pair_cache_path(out_dir, measurement)
    path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(path, index=False)
    return path

def mirror_outputs(out_dir):
    dirs = make_output_sections(out_dir)

    for name in [
        "summary.json",
        "learned_mix_weights.json",
    ]:
        copy_if_exists(out_dir / name, dirs["reports"] / name)

    for name in [
        "best_k_by_measurement.csv",
        "bench_xy_summary.csv",
        "bench_xy_best.csv",
        "learned_mix_top.csv",
    ]:
        copy_if_exists(out_dir / name, dirs["report_tables"] / name)

    for name in [
        "bench_xy.csv",
        "bench_xy_by_k.csv",
        "learned_mix_grid.csv",
    ]:
        copy_if_exists(out_dir / name, dirs["diagnostics"] / name)

    for path in out_dir.glob("checks_*.json"):
        copy_if_exists(path, dirs["diagnostics_checks"] / path.name)

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
        adata = adata[:, adata.var["feature_types"].astype(str).eq("Gene Expression")].copy()
    cells, cells_path = read_cells(xenium_dir)
    ids = adata.obs_names.intersection(cells.index)
    adata = adata[ids].copy()
    cells = cells.loc[adata.obs_names]
    coords = cells[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
    adata.obsm["spatial"] = coords
    adata.obs["cell_id"] = adata.obs_names.astype(str)
    adata.obs["x_centroid"] = coords[:, 0]
    adata.obs["y_centroid"] = coords[:, 1]
    adata.obs["cell_area"] = cells["cell_area"].to_numpy(dtype=np.float32) if "cell_area" in cells else np.zeros(adata.n_obs, dtype=np.float32)
    adata.obs["nucleus_area"] = cells["nucleus_area"].to_numpy(dtype=np.float32) if "nucleus_area" in cells else np.zeros(adata.n_obs, dtype=np.float32)
    adata.uns["xenium_clean"] = {"xenium_dir": str(xenium_dir), "matrix_path": str(matrix_path), "cells_path": str(cells_path)}
    return adata
