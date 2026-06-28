from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from xenum_common import zscore
from xenum_measurements import SPAGCN_MEASUREMENTS

from .config import (
    K,
    SPAGCN_ALPHA,
    SPAGCN_BETA,
    SPAGCN_LR,
    SPAGCN_MAX_EPOCHS,
    SPAGCN_MIN_CELLS,
    SPAGCN_NUM_PCS,
    SPAGCN_P,
    SPAGCN_RESOLUTION,
    SPAGCN_SEED,
    SPAGCN_TOL,
)
from .graph import checks, edges_from_neighbor_lists, load_or_make_pairs, neighbor_lists_from_pairs, prediction_from_edges
from .npz import write_npz
from xenum_paths import data_dir

XY_VARIANT = "spagcn_xy"
HISTOLOGY_VARIANT = "spagcn_histology"
ADJACENCY_MEASUREMENT = "spagcn_adjacency"
VARIANT_MEASUREMENTS = {
    XY_VARIANT: ("spagcn_xy_embedding", "spagcn_xy_probabilities"),
    HISTOLOGY_VARIANT: ("spagcn_histology_embedding", "spagcn_histology_probabilities"),
}

class SpaGCNSkip(RuntimeError):
    pass

def ensure_scipy_sparse_compat():
    import scipy.sparse as sparse

    if not hasattr(sparse.spmatrix, "A"):
        sparse.spmatrix.A = property(lambda matrix: matrix.toarray())

def import_spagcn():
    ensure_scipy_sparse_compat()

    import SpaGCN as spg
    return spg

def normalize_for_spagcn(adata, spg):
    import scanpy as sc

    adata = adata.copy()
    adata.var_names = [str(x).upper() for x in adata.var_names]
    adata.var_names_make_unique()
    adata.var["genename"] = adata.var_names.astype(str)

    spg.prefilter_genes(adata, min_cells=int(SPAGCN_MIN_CELLS))
    spg.prefilter_specialgenes(adata)

    if adata.n_vars == 0:
        raise RuntimeError("SpaGCN preprocessing removed all genes")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata

def find_default_image(dataset_id):
    root = data_dir(dataset_id)
    candidates = []

    if root.exists():
        for name in [
            "morphology.ome.tif",
            "morphology.tif",
            "histology.tif",
            "histology.png",
            "histology.jpg",
        ]:
            p = root / name
            if p.exists():
                candidates.append(p)

        candidates.extend(sorted((root / "morphology_focus").glob("*.ome.tif")))
        candidates.extend(sorted((root / "morphology_focus").glob("*.tif")))
        candidates.extend(sorted((root / "morphology_focus").glob("*.tiff")))
        candidates.extend(sorted((root / "morphology_focus").glob("*.png")))
        candidates.extend(sorted((root / "morphology_focus").glob("*.jpg")))
        candidates.extend(sorted((root / "morphology_focus").glob("*.jpeg")))

    return candidates[0] if candidates else None

def normalize_image_array(arr, axes=None):
    img = np.asarray(arr)

    if axes and len(axes) == img.ndim:
        axes = list(axes)
        squeeze = [i for i, size in enumerate(img.shape) if size == 1]

        if squeeze:
            img = np.squeeze(img, axis=tuple(squeeze))
            axes = [axis for i, axis in enumerate(axes) if i not in squeeze]

        lower = [str(axis).lower() for axis in axes]

        if "y" in lower and "x" in lower:
            y_axis = lower.index("y")
            x_axis = lower.index("x")
            rest = [i for i in range(img.ndim) if i not in {y_axis, x_axis}]
            img = np.transpose(img, [y_axis, x_axis, *rest])

            if rest:
                img = img.reshape(img.shape[0], img.shape[1], int(np.prod(img.shape[2:])))
            else:
                img = img[:, :, None]
    else:
        img = np.squeeze(img)

        if img.ndim == 3 and img.shape[-1] not in {1, 2, 3, 4}:
            img = np.moveaxis(img, 0, -1)

    if img.ndim == 2:
        img = img[:, :, None]
    elif img.ndim != 3:
        raise SpaGCNSkip("unsupported image shape {}".format(img.shape))

    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.shape[2] == 2:
        third = img.mean(axis=2, keepdims=True)
        img = np.concatenate([img, third], axis=2)
    elif img.shape[2] > 3:
        img = img[:, :, :3]

    return np.asarray(img)

def read_tiff_image(path):
    try:
        import tifffile
    except Exception as e:
        raise SpaGCNSkip("tifffile import failed: {}: {}".format(type(e).__name__, e)) from e

    try:
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            return normalize_image_array(series.asarray(), axes=getattr(series, "axes", None))
    except Exception as e:
        raise SpaGCNSkip("tifffile read failed for {}: {}: {}".format(path, type(e).__name__, e)) from e

def read_image(path):
    if path is None:
        raise SpaGCNSkip("no image path")

    suffixes = [s.lower() for s in Path(path).suffixes]
    if any(s in {".tif", ".tiff"} for s in suffixes):
        return read_tiff_image(path)

    try:
        import cv2
    except Exception as e:
        raise SpaGCNSkip("cv2 import failed: {}: {}".format(type(e).__name__, e)) from e

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise SpaGCNSkip("cv2.imread returned None")

    return normalize_image_array(img)

def image_pixel_coords(coords, image):
    x = coords[:, 0]
    y = coords[:, 1]
    h, w = image.shape[:2]

    yx_ok = (
        np.nanmin(x) >= 0
        and np.nanmin(y) >= 0
        and np.nanmax(x) < w
        and np.nanmax(y) < h
    )

    if yx_ok:
        return np.rint(y).astype(int).tolist(), np.rint(x).astype(int).tolist(), "row=y,column=x"

    xy_ok = (
        np.nanmin(x) >= 0
        and np.nanmin(y) >= 0
        and np.nanmax(x) < h
        and np.nanmax(y) < w
    )

    if xy_ok:
        return np.rint(x).astype(int).tolist(), np.rint(y).astype(int).tolist(), "row=x,column=y"

    raise SpaGCNSkip(
        "centroids outside image bounds image_shape={} x_range=({:.3f},{:.3f}) y_range=({:.3f},{:.3f})".format(
            image.shape,
            float(np.nanmin(x)),
            float(np.nanmax(x)),
            float(np.nanmin(y)),
            float(np.nanmax(y)),
        )
    )

def calculate_adj(spg, dataset_id, coords, use_histology):
    x = coords[:, 0].astype(float).tolist()
    y = coords[:, 1].astype(float).tolist()

    meta = {
        "histology_used": False,
        "image_path": None,
        "pixel_orientation": None,
    }

    if not use_histology:
        return spg.calculate_adj_matrix(x=x, y=y, histology=False), meta

    image_path = find_default_image(dataset_id)
    image = read_image(image_path)
    x_pixel, y_pixel, orientation = image_pixel_coords(coords, image)

    meta.update({
        "histology_used": True,
        "image_path": str(image_path),
        "pixel_orientation": orientation,
        "image_shape": list(image.shape),
    })

    adj = spg.calculate_adj_matrix(
        x=x,
        y=y,
        x_pixel=x_pixel,
        y_pixel=y_pixel,
        image=image,
        beta=int(SPAGCN_BETA),
        alpha=float(SPAGCN_ALPHA),
        histology=True,
    )

    return adj, meta

def positive_adj_stats(adj):
    positive = adj[np.isfinite(adj) & (adj > 0)]
    return {
        "n": int(adj.shape[0]),
        "min_positive": float(positive.min()) if len(positive) else None,
        "median_positive": float(np.median(positive)) if len(positive) else None,
        "p90_positive": float(np.quantile(positive, 0.90)) if len(positive) else None,
        "max": float(np.nanmax(adj)) if adj.size else None,
    }

def domain_summary(domains):
    rows = []

    for domain, group in domains.groupby("domain"):
        xy = group[["x", "y"]].to_numpy(dtype=np.float64)
        center = xy.mean(axis=0)
        radius = np.sqrt(((xy - center) ** 2).sum(axis=1))
        rows.append({
            "domain": domain,
            "n_cells": int(len(group)),
            "center_x": float(center[0]),
            "center_y": float(center[1]),
            "mean_xy_radius": float(radius.mean()) if len(radius) else None,
            "median_xy_radius": float(np.median(radius)) if len(radius) else None,
            "p90_xy_radius": float(np.quantile(radius, 0.90)) if len(radius) else None,
        })

    return pd.DataFrame(rows).sort_values(["domain"]).reset_index(drop=True)

def write_matrix_with_ids(path, nodes, matrix, prefix):
    df = pd.DataFrame(
        np.asarray(matrix, dtype=np.float32),
        columns=[f"{prefix}_{i + 1}" for i in range(np.asarray(matrix).shape[1])],
    )
    df.insert(0, "cell_id", nodes["cell_id"].to_numpy(dtype=str))
    df.insert(0, "node", np.arange(len(nodes), dtype=np.int64))
    df.to_csv(path, index=False)

def neighbor_lists_from_adj(adj):
    adj = np.asarray(adj, dtype=np.float32)

    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("SpaGCN adjacency must be square, got {}".format(adj.shape))

    best = []

    for i in range(adj.shape[0]):
        row = adj[i]
        order = np.argsort(row, kind="stable")
        vals = []

        for j in order:
            if int(j) == i:
                continue

            distance = float(row[j])

            if np.isfinite(distance):
                vals.append((int(j), distance))

        best.append(vals)

    return best

def skipped_rows(dataset_id, nodes, measurements, bench_k_values, status, reason, seed=None):
    rows = []

    for measurement in measurements:
        for kk in bench_k_values:
            rows.append({
                "dataset": dataset_id,
                "measurement": measurement,
                "label": measurement,
                "k": int(kk),
                "seed": None if seed is None else int(seed),
                "status": status,
                "status_reason": reason,
                "leaky": True,
                "n_nodes": int(len(nodes)),
                "n_edges": 0,
                "coverage": 0.0,
                "mean_xy_error": None,
                "median_xy_error": None,
                "p90_xy_error": None,
            })

    return rows

def variant_status_measurements(variant, measurements):
    if variant == XY_VARIANT:
        return (ADJACENCY_MEASUREMENT, *measurements)

    return measurements

def train_variant(spg, dataset_id, out_dir, variant, adata, nodes, coords, use_histology):
    started = time.perf_counter()
    adj, image_meta = calculate_adj(spg, dataset_id, coords, use_histology)
    adj_stats = positive_adj_stats(adj)

    num_pcs = min(int(SPAGCN_NUM_PCS), max(1, adata.n_obs - 1), max(1, adata.n_vars - 1))
    l_value = spg.search_l(float(SPAGCN_P), adj, start=0.01, end=1000, tol=0.01, max_run=100)

    if l_value is None:
        l_value = adj_stats["median_positive"] or 1.0

    clf = spg.SpaGCN()
    clf.set_l(float(l_value))
    clf.train(
        adata,
        adj,
        num_pcs=int(num_pcs),
        init_spa=True,
        init="louvain",
        res=float(SPAGCN_RESOLUTION),
        tol=float(SPAGCN_TOL),
        lr=float(SPAGCN_LR),
        max_epochs=int(SPAGCN_MAX_EPOCHS),
    )

    import torch

    with torch.no_grad():
        z, q = clf.model.predict(clf.embed, clf.adj_exp)

    embedding = z.detach().cpu().numpy()
    probabilities = q.detach().cpu().numpy()
    domains = np.argmax(probabilities, axis=1).astype(int)

    domains_df = pd.DataFrame({
        "node": np.arange(len(nodes), dtype=np.int64),
        "cell_id": nodes["cell_id"].to_numpy(dtype=str),
        "x": nodes["x_centroid"].to_numpy(dtype=np.float64),
        "y": nodes["y_centroid"].to_numpy(dtype=np.float64),
        "domain": domains,
        "max_probability": np.max(probabilities, axis=1),
    })

    domains_df.to_csv(out_dir / f"{variant}_domains.csv", index=False)
    domain_summary(domains_df).to_csv(out_dir / f"{variant}_domains_summary.csv", index=False)
    write_matrix_with_ids(out_dir / f"{variant}_embedding.csv", nodes, embedding, "z")
    write_matrix_with_ids(out_dir / f"{variant}_probabilities.csv", nodes, probabilities, "q")

    meta = {
        "dataset": dataset_id,
        "variant": variant,
        "status": "ok",
        "seed": int(SPAGCN_SEED),
        "num_pcs": int(num_pcs),
        "l": float(l_value),
        "p": float(SPAGCN_P),
        "resolution": float(SPAGCN_RESOLUTION),
        "max_epochs": int(SPAGCN_MAX_EPOCHS),
        "histology": image_meta,
        "adjacency_stats": adj_stats,
        "runtime_sec": time.perf_counter() - started,
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "pid": os.getpid(),
            "spagcn_version": getattr(spg, "__version__", None),
        },
    }
    (out_dir / f"{variant}_runtime.json").write_text(json.dumps(meta, indent=2) + "\n")

    return {
        f"{variant}_embedding": zscore(embedding),
        f"{variant}_probabilities": zscore(probabilities),
    }, meta, adj

def benchmark_spagcn_adjacency(out_dir, dataset_id, nodes, node_cols, blocks, adj, bench_k_values):
    rows = []
    summaries = {}
    measurement = ADJACENCY_MEASUREMENT
    best = neighbor_lists_from_adj(adj)

    for kk in bench_k_values:
        edges, graph_nodes = edges_from_neighbor_lists(
            nodes[node_cols],
            blocks,
            measurement,
            best,
            kk,
        )
        chk = checks(nodes, graph_nodes, edges, measurement)
        pred_df, row = prediction_from_edges(
            dataset_id,
            nodes[node_cols],
            edges,
            measurement,
            kk,
            seed=int(SPAGCN_SEED),
        )
        rows.append(row)

        graph_nodes.to_csv(out_dir / f"nodes_{measurement}_k{kk}.csv", index=False)
        edges.to_csv(out_dir / f"edges_{measurement}_k{kk}.csv", index=False)
        pred_df.to_csv(out_dir / f"predictions_{measurement}_k{kk}.csv", index=False)
        (out_dir / f"checks_{measurement}_k{kk}.json").write_text(json.dumps(chk, indent=2) + "\n")

        if kk == int(K):
            graph_nodes.to_csv(out_dir / f"nodes_{measurement}.csv", index=False)
            edges.to_csv(out_dir / f"edges_{measurement}.csv", index=False)
            (out_dir / f"checks_{measurement}.json").write_text(json.dumps(chk, indent=2) + "\n")
            write_npz(out_dir, measurement, graph_nodes, edges)
            summaries[measurement] = chk

    return rows, summaries

def benchmark_spagcn_blocks(out_dir, dataset_id, nodes, node_cols, blocks, bench_k_values, measurements):
    rows = []
    summaries = {}

    for measurement in measurements:
        pairs = load_or_make_pairs(out_dir, nodes[node_cols], blocks, measurement)
        best = neighbor_lists_from_pairs(len(nodes), pairs)

        for kk in bench_k_values:
            edges, graph_nodes = edges_from_neighbor_lists(
                nodes[node_cols],
                blocks,
                measurement,
                best,
                kk,
            )
            chk = checks(nodes, graph_nodes, edges, measurement)
            pred_df, row = prediction_from_edges(
                dataset_id,
                nodes[node_cols],
                edges,
                measurement,
                kk,
                seed=int(SPAGCN_SEED),
            )
            rows.append(row)

            graph_nodes.to_csv(out_dir / f"nodes_{measurement}_k{kk}.csv", index=False)
            edges.to_csv(out_dir / f"edges_{measurement}_k{kk}.csv", index=False)
            pred_df.to_csv(out_dir / f"predictions_{measurement}_k{kk}.csv", index=False)
            (out_dir / f"checks_{measurement}_k{kk}.json").write_text(json.dumps(chk, indent=2) + "\n")

            if kk == int(K):
                graph_nodes.to_csv(out_dir / f"nodes_{measurement}.csv", index=False)
                edges.to_csv(out_dir / f"edges_{measurement}.csv", index=False)
                (out_dir / f"checks_{measurement}.json").write_text(json.dumps(chk, indent=2) + "\n")
                write_npz(out_dir, measurement, graph_nodes, edges)
                summaries[measurement] = chk

    return rows, summaries

def run_spagcn_measurements(out_dir, dataset_id, adata, nodes, node_cols, blocks, bench_k_values):
    started = time.perf_counter()

    try:
        spg = import_spagcn()
    except Exception as e:
        reason = "{}: {}".format(type(e).__name__, e)
        (out_dir / "spagcn_status.json").write_text(json.dumps({
            "dataset": dataset_id,
            "status": "missing_dependency",
            "reason": reason,
            "runtime_sec": time.perf_counter() - started,
        }, indent=2) + "\n")
        print(f"SpaGCN skipped: {reason}", flush=True)
        return skipped_rows(dataset_id, nodes, SPAGCN_MEASUREMENTS, bench_k_values, "missing_dependency", reason), {}

    try:
        import torch

        random.seed(int(SPAGCN_SEED))
        np.random.seed(int(SPAGCN_SEED))
        torch.manual_seed(int(SPAGCN_SEED))

        spagcn_adata = normalize_for_spagcn(adata, spg)
    except Exception as e:
        reason = "{}: {}".format(type(e).__name__, e)
        (out_dir / "spagcn_status.json").write_text(json.dumps({
            "dataset": dataset_id,
            "status": "failed",
            "reason": reason,
            "runtime_sec": time.perf_counter() - started,
        }, indent=2) + "\n")
        print(f"SpaGCN failed: {reason}", flush=True)
        return skipped_rows(dataset_id, nodes, SPAGCN_MEASUREMENTS, bench_k_values, "failed", reason, seed=SPAGCN_SEED), {}

    rows = []
    summaries = {}
    status = {
        "dataset": dataset_id,
        "status": "started",
        "variants": {},
    }
    coords = nodes[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float64)

    for variant, measurements in VARIANT_MEASUREMENTS.items():
        use_histology = variant == HISTOLOGY_VARIANT

        try:
            variant_blocks, meta, adj = train_variant(
                spg,
                dataset_id,
                out_dir,
                variant,
                spagcn_adata,
                nodes,
                coords,
                use_histology,
            )
            blocks.update(variant_blocks)

            if variant == XY_VARIANT:
                adj_rows, adj_summaries = benchmark_spagcn_adjacency(
                    out_dir,
                    dataset_id,
                    nodes,
                    node_cols,
                    blocks,
                    adj,
                    bench_k_values,
                )
                rows.extend(adj_rows)
                summaries.update(adj_summaries)
                meta["adjacency_measurement"] = ADJACENCY_MEASUREMENT

            part_rows, part_summaries = benchmark_spagcn_blocks(
                out_dir,
                dataset_id,
                nodes,
                node_cols,
                blocks,
                bench_k_values,
                measurements,
            )
            rows.extend(part_rows)
            summaries.update(part_summaries)
            status["variants"][variant] = meta
        except SpaGCNSkip as e:
            reason = str(e)
            rows.extend(skipped_rows(
                dataset_id,
                nodes,
                variant_status_measurements(variant, measurements),
                bench_k_values,
                "skipped",
                reason,
                seed=SPAGCN_SEED,
            ))
            status["variants"][variant] = {
                "status": "skipped",
                "reason": reason,
            }
            print(f"{variant} skipped: {reason}", flush=True)
        except Exception as e:
            reason = "{}: {}".format(type(e).__name__, e)
            rows.extend(skipped_rows(
                dataset_id,
                nodes,
                variant_status_measurements(variant, measurements),
                bench_k_values,
                "failed",
                reason,
                seed=SPAGCN_SEED,
            ))
            status["variants"][variant] = {
                "status": "failed",
                "reason": reason,
            }
            print(f"{variant} failed: {reason}", flush=True)

    status["status"] = "ok"
    status["runtime_sec"] = time.perf_counter() - started
    (out_dir / "spagcn_status.json").write_text(json.dumps(status, indent=2) + "\n")

    return rows, summaries
