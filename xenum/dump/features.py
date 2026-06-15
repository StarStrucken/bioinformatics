from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD

from xenum_common import zscore
from xenum_measurements import HIDDEN_MEASUREMENTS, MEASUREMENTS, OPTIONAL_MEASUREMENTS, VISIBLE_MEASUREMENTS

from .config import MORPHOLOGY_IMAGE_FEATURE_FILES

def clean_float(x):
    x = np.asarray(x, dtype=np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def per_cell_counts(x):
    total = np.asarray(x.sum(axis=1)).ravel().astype(np.float32)
    detected = np.asarray(x.getnnz(axis=1) if sp.issparse(x) else (x > 0).sum(axis=1)).ravel().astype(np.float32)
    return clean_float(total), clean_float(detected)

def top_gene_ids(x, n=32):
    out = []
    x = x.tocsr() if sp.issparse(x) else np.asarray(x)

    for i in range(x.shape[0]):
        row = x.getrow(i) if sp.issparse(x) else x[i]
        if sp.issparse(row):
            idx = row.indices
            val = row.data
        else:
            idx = np.flatnonzero(row)
            val = row[idx]

        if len(idx) == 0:
            out.append("")
            continue

        take = np.argsort(-val)[:n]
        out.append(" ".join(str(int(v)) for v in idx[take]))

    return out

def detected_gene_ids(x):
    out = []
    x = x.tocsr() if sp.issparse(x) else np.asarray(x)

    for i in range(x.shape[0]):
        row = x.getrow(i) if sp.issparse(x) else x[i]

        if sp.issparse(row):
            idx = row.indices
        else:
            idx = np.flatnonzero(row)

        if len(idx) == 0:
            out.append("")
            continue

        out.append(" ".join(str(int(v)) for v in idx))

    return out

def jaccard_distance(a, b):
    a = set(a)
    b = set(b)

    u = len(a | b)

    if u == 0:
        return 0.0

    return 1.0 - len(a & b) / u

def load_morphology_feature_table(out_dir, nodes, measurement, rel_paths):
    path = None

    for rel in rel_paths:
        p = out_dir / rel
        if p.exists():
            path = p
            break

    if path is None:
        return None

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if "cell_id" in df.columns:
        df["cell_id"] = df["cell_id"].astype(str)
        base = nodes[["cell_id"]].copy()
        aligned = base.merge(df, on="cell_id", how="left")
    elif len(df) == len(nodes):
        aligned = df.reset_index(drop=True).copy()
    else:
        print(f"{measurement} skipped: cannot align {path}", flush=True)
        return None

    skip = {
        "cell_id",
        "x",
        "y",
        "x_centroid",
        "y_centroid",
        "label",
        "labels",
        "node",
        "index",
    }

    feature_cols = [
        c for c in aligned.columns
        if c not in skip and pd.api.types.is_numeric_dtype(aligned[c])
    ]

    if not feature_cols:
        print(f"{measurement} skipped: no numeric features in {path}", flush=True)
        return None

    mat = aligned[feature_cols].to_numpy(dtype=np.float32)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return zscore(mat), feature_cols, str(path)

def load_morphology_image_block(out_dir, nodes):
    return load_morphology_feature_table(
        out_dir,
        nodes,
        "morphology_image",
        MORPHOLOGY_IMAGE_FEATURE_FILES["morphology_image"],
    )

def load_morphology_image_blocks(out_dir, nodes):
    out = {}

    for measurement, rel_paths in MORPHOLOGY_IMAGE_FEATURE_FILES.items():
        item = load_morphology_feature_table(out_dir, nodes, measurement, rel_paths)

        if item is not None:
            out[measurement] = item

    return out

def measurement_available(measurement, blocks):
    if measurement not in MEASUREMENTS:
        return False

    if measurement.startswith("seq_"):
        return True

    for name in MEASUREMENTS[measurement]["blocks"]:
        if name not in blocks:
            return False

    return True

def available_measurements(blocks):
    out = []

    for m in VISIBLE_MEASUREMENTS + OPTIONAL_MEASUREMENTS + HIDDEN_MEASUREMENTS:
        if measurement_available(m, blocks):
            out.append(m)

    return out

def log_norm_matrix(x, target_sum=10000.0):
    total = np.asarray(x.sum(axis=1)).ravel().astype(np.float32)
    scale = target_sum / np.maximum(total, 1.0)
    if sp.issparse(x):
        out = x.tocsr(copy=True).astype(np.float32)
        out = out.multiply(scale[:, None]).tocsr()
        out.data = np.log1p(out.data)
        return out
    out = np.asarray(x, dtype=np.float32) * scale[:, None]
    return np.log1p(out)

def expression_pcs(adata, n_pcs: int):
    x = log_norm_matrix(adata.X)
    n = min(int(n_pcs), max(1, adata.n_obs - 1), max(1, adata.n_vars - 1))
    if sp.issparse(x):
        emb = TruncatedSVD(n_components=n, random_state=0).fit_transform(x)
    else:
        emb = PCA(n_components=n, random_state=0).fit_transform(x)
    return zscore(emb)

def make_nodes(adata):
    total, detected = per_cell_counts(adata.X)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    low = coords.min(axis=0)
    span = np.ptp(coords, axis=0) + 1e-6
    norm = (coords - low) / span
    cell_area = clean_float(adata.obs["cell_area"].to_numpy(dtype=np.float32))
    nucleus_area = clean_float(adata.obs["nucleus_area"].to_numpy(dtype=np.float32))
    ratio = np.divide(nucleus_area, cell_area, out=np.zeros_like(nucleus_area), where=cell_area > 0)
    return pd.DataFrame({
        "cell_id": adata.obs["cell_id"].to_numpy(dtype=str),
        "x_centroid": coords[:, 0],
        "y_centroid": coords[:, 1],
        "x_norm": norm[:, 0],
        "y_norm": norm[:, 1],
        "log_total_counts": np.log1p(total),
        "log_detected_genes": np.log1p(detected),
        "cell_area": cell_area,
        "nucleus_area": nucleus_area,
        "nucleus_cell_ratio": clean_float(ratio),
    })

def build_blocks(adata, nodes, n_pcs):
    spatial = zscore(nodes[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32))
    morphology = zscore(nodes[["cell_area", "nucleus_area", "nucleus_cell_ratio"]].to_numpy(dtype=np.float32))
    expression = expression_pcs(adata, n_pcs)
    return {"spatial": spatial, "expression": expression, "morphology": morphology}

def block_embedding(blocks, measurement):
    parts = []
    for name, weight in MEASUREMENTS[measurement]["blocks"].items():
        parts.append(blocks[name] * float(weight))
    return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]

def pair_distance(block, src, dst):
    return np.linalg.norm(block[dst] - block[src], axis=1).astype(np.float32)
