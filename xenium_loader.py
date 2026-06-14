#!/usr/bin/env python3
"""
xenium loader for spagcn - converts xenium data to anndata objects
compatible with spatial transcriptomics pipelines
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

import scanpy as sc
from anndata import AnnData

from xenum_common import clean_vec


def read_xenium_cells(xenium_dir: Path) -> Tuple[pd.DataFrame, Path]:
    """read cell metadata from xenium output"""
    # look for cell table in standard locations
    cell_table_names = ("cells.csv.gz", "cells.csv", "cells.parquet")
    
    roots = [xenium_dir] if xenium_dir.name == "bundle" else [xenium_dir, xenium_dir / "bundle"]
    for base in roots:
        for name in cell_table_names:
            p = base / name
            if p.exists():
                cells = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
                cells["cell_id"] = cells["cell_id"].astype(str)
                return cells.drop_duplicates("cell_id").set_index("cell_id"), p
    
    # fallback to recursive search
    for base in roots:
        for name in cell_table_names:
            for p in base.rglob(name):
                cells = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
                cells["cell_id"] = cells["cell_id"].astype(str)
                return cells.drop_duplicates("cell_id").set_index("cell_id"), p
    
    raise FileNotFoundError(f"no cell table found in {xenium_dir}")


def find_xenium_file(xenium_dir: Path, names: tuple[str, ...]) -> Path:
    """locate xenium output files"""
    roots = [xenium_dir] if xenium_dir.name == "bundle" else [xenium_dir, xenium_dir / "bundle"]
    
    # check direct paths first
    for base in roots:
        for name in names:
            p = base / name
            if p.exists():
                return p
    
    # recursive search if not found
    for base in roots:
        for name in names:
            for p in base.rglob(name):
                return p
    
    raise FileNotFoundError(f"files {names} not found in {xenium_dir}")


def load_xenium_raw(xenium_dir: Path) -> AnnData:
    """
    load raw xenium data from directory
    reads gene expression matrix and cell metadata
    """
    xenium_dir = Path(xenium_dir)
    
    # read gene expression h5 matrix
    matrix_path = find_xenium_file(xenium_dir, ("cell_feature_matrix.h5",))
    adata = sc.read_10x_h5(matrix_path, gex_only=False)
    adata.var_names_make_unique()
    adata.obs_names = adata.obs_names.astype(str)
    
    # filter to gene expression only if there are mixed feature types
    if "feature_types" in adata.var.columns:
        adata = adata[:, adata.var["feature_types"].astype(str).eq("Gene Expression")].copy()
    
    # read cell metadata
    cells, cells_path = read_xenium_cells(xenium_dir)
    
    # intersect with available cells
    ids = adata.obs_names.intersection(cells.index)
    adata = adata[ids].copy()
    cells = cells.loc[adata.obs_names]
    
    # add spatial coordinates
    coords = cells[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
    adata.obsm["spatial"] = coords
    
    # populate observations with spatial and morphological features
    adata.obs["cell_id"] = adata.obs_names.astype(str)
    adata.obs["x_centroid"] = coords[:, 0]
    adata.obs["y_centroid"] = coords[:, 1]
    adata.obs["cell_area"] = cells["cell_area"].to_numpy(dtype=np.float32) if "cell_area" in cells else np.zeros(adata.n_obs, dtype=np.float32)
    adata.obs["nucleus_area"] = cells["nucleus_area"].to_numpy(dtype=np.float32) if "nucleus_area" in cells else np.zeros(adata.n_obs, dtype=np.float32)
    
    # store metadata
    adata.uns["xenium_source"] = {
        "xenium_dir": str(xenium_dir),
        "matrix_path": str(matrix_path),
        "cells_path": str(cells_path)
    }
    
    return adata


def preprocess_for_spagcn(adata: AnnData, min_genes: int = 200, min_cells: int = 10) -> AnnData:
    """
    prepare xenium anndata for spagcn analysis
    filters genes and cells, normalizes expression
    """
    adata = adata.copy()
    
    # filter cells by detected genes
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    adata = adata[adata.obs["n_genes_by_counts"] >= min_genes].copy()
    
    # filter genes by number of cells with expression
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    # log normalize
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # store raw for later
    adata.raw = adata.copy()
    
    return adata


def prepare_spagcn_input(
    xenium_dir: Path,
    min_genes: int = 200,
    min_cells: int = 10,
    use_raw: bool = False
) -> AnnData:
    """
    complete pipeline: load xenium data and prepare for spagcn
    
    Args:
        xenium_dir: path to xenium output directory
        min_genes: minimum genes per cell for filtering
        min_cells: minimum cells per gene for filtering
        use_raw: whether to use raw counts or normalized
    
    Returns:
        anndata object ready for spagcn
    """
    print(f"loading xenium data from {xenium_dir}...")
    adata = load_xenium_raw(xenium_dir)
    print(f"loaded {adata.n_obs} cells, {adata.n_vars} genes")
    
    print(f"preprocessing for spagcn (min_genes={min_genes}, min_cells={min_cells})...")
    adata = preprocess_for_spagcn(adata, min_genes=min_genes, min_cells=min_cells)
    print(f"after filtering: {adata.n_obs} cells, {adata.n_vars} genes")
    
    # set up coordinate columns for spagcn compatibility
    if "x_array" not in adata.obs:
        # use normalized coordinates if original array coordinates not available
        x = adata.obs["x_centroid"].values
        y = adata.obs["y_centroid"].values
        
        x_norm = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.zeros_like(x)
        y_norm = (y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else np.zeros_like(y)
        
        adata.obs["x_array"] = x_norm
        adata.obs["y_array"] = y_norm
        adata.obs["x_pixel"] = x
        adata.obs["y_pixel"] = y
    
    print(f"ready for spagcn analysis")
    return adata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="load and prepare xenium data for spagcn")
    parser.add_argument("xenium_dir", type=Path, help="path to xenium output directory")
    parser.add_argument("--output", "-o", type=Path, help="save h5ad file", default=None)
    parser.add_argument("--min-genes", type=int, default=200, help="minimum genes per cell")
    parser.add_argument("--min-cells", type=int, default=10, help="minimum cells per gene")
    
    args = parser.parse_args()
    
    # prepare data
    adata = prepare_spagcn_input(
        args.xenium_dir,
        min_genes=args.min_genes,
        min_cells=args.min_cells
    )
    
    # save if requested
    if args.output:
        adata.write_h5ad(args.output)
        print(f"saved to {args.output}")
    else:
        print(adata)
