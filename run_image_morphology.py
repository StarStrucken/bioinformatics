#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from xenum_image_morphology_config import DATASET_IMAGE_MORPHOLOGY
from xenum_paths import out_dir as make_out_dir

DEFAULT_LABELS_KEY = "cell_labels"
DEFAULT_IMAGE_KEY = "morphology_focus"
DEFAULT_ADATA_KEY = "morphology"
DEFAULT_FEATURES = [
    "summary",
    "histogram",
]
DEFAULT_LIBRARY_ID = "xenium"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_id")
    return p.parse_args()

def require_config(dataset_id):
    cfg = DATASET_IMAGE_MORPHOLOGY.get(dataset_id)

    if cfg is None:
        raise KeyError(f"no image morphology config for {dataset_id}")

    if not cfg.get("xenium_dir") and not cfg.get("sdata_path"):
        raise ValueError(f"xenium_dir or sdata_path is not set for {dataset_id}")

    return cfg

def apply_crop(sdata, crop_cfg):
    if not crop_cfg:
        return sdata

    from spatialdata import bounding_box_query

    return bounding_box_query(
        sdata,
        min_coordinate=crop_cfg["min_coordinate"],
        max_coordinate=crop_cfg["max_coordinate"],
        axes=tuple(crop_cfg.get("axes", ["x", "y"])),
        target_coordinate_system=crop_cfg.get("target_coordinate_system", "global"),
    )

def load_sdata(cfg):
    if cfg.get("sdata_path"):
        import spatialdata as sd
        print(f"read spatialdata: {cfg['sdata_path']}", flush=True)
        return sd.read_zarr(cfg["sdata_path"])

    from spatialdata_io import xenium

    xenium_dir = Path(cfg["xenium_dir"])
    print(f"read xenium: {xenium_dir}", flush=True)

    kwargs = {
        "cells_boundaries": True,
        "nucleus_boundaries": True,
        "cells_as_circles": False,
        "cells_labels": True,
        "nucleus_labels": True,
        "transcripts": False,
        "morphology_mip": True,
        "morphology_focus": True,
        "aligned_images": False,
        "cells_table": True,
        "gex_only": True,
    }

    kwargs.update(cfg.get("xenium_kwargs", {}))

    return xenium(xenium_dir, **kwargs)

def pick_key(keys, preferred, words):
    keys = list(keys)

    if preferred in keys:
        return preferred

    for word in words:
        hits = [k for k in keys if word in str(k)]
        if hits:
            return hits[0]

    if len(keys) == 1:
        return keys[0]

    raise KeyError(f"cannot pick key, preferred={preferred}, available={keys}")

def dataset_to_dataarray(ds):
    import xarray as xr

    if isinstance(ds, xr.DataArray):
        return ds

    if isinstance(ds, xr.Dataset):
        candidates = []
        for name, arr in ds.data_vars.items():
            if arr.ndim >= 2:
                candidates.append((name, arr))

        if not candidates:
            raise ValueError(f"no image-like arrays in dataset: {list(ds.data_vars)}")

        return candidates[0][1]

    raise TypeError(type(ds).__name__)

def image_tree_children(x):
    if hasattr(x, "children"):
        return list(x.children.items())

    try:
        keys = list(x.keys())
    except Exception:
        return []

    out = []
    for k in keys:
        try:
            child = x[k]
        except Exception:
            continue
        out.append((str(k), child))

    return out

def find_image_dataarray(x, preferred_scale=None):
    import xarray as xr

    if isinstance(x, xr.DataArray):
        return x

    if isinstance(x, xr.Dataset):
        return dataset_to_dataarray(x)

    if hasattr(x, "ds"):
        try:
            ds = x.ds
            if ds is not None and len(ds.data_vars):
                return dataset_to_dataarray(ds)
        except Exception:
            pass

    children = image_tree_children(x)

    if not children:
        raise TypeError(f"cannot extract image DataArray from {type(x).__name__}")

    names = [name for name, _ in children]
    ordered = []

    for name in [preferred_scale, "scale0", "0", "image"]:
        if name and name in names and name not in ordered:
            ordered.append(name)

    for name in names:
        if name not in ordered:
            ordered.append(name)

    by_name = dict(children)

    errors = []
    for name in ordered:
        try:
            return find_image_dataarray(by_name[name], preferred_scale=None)
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")

    raise ValueError("cannot find image array; " + " | ".join(errors))

def normalize_image_for_squidpy(arr):
    arr = arr.squeeze(drop=True)

    rename = {}
    for d in arr.dims:
        if d in {"c", "channel"}:
            rename[d] = "channels"

    if rename:
        arr = arr.rename(rename)

    if "y" not in arr.dims or "x" not in arr.dims:
        raise ValueError(f"image dims must contain x/y, got {arr.dims}")

    dims = list(arr.dims)

    order = ["y", "x"]

    if "z" not in dims:
        arr = arr.expand_dims({"z": [0]}, axis=2)

    order.append("z")

    if "channels" in arr.dims:
        order.append("channels")

    for d in arr.dims:
        if d not in order:
            order.append(d)

    arr = arr.transpose(*order)

    if "channels" not in arr.dims:
        arr = arr.expand_dims({"channels": [0]}, axis=3)
        arr = arr.transpose("y", "x", "z", "channels")

    dims_arg = tuple(arr.dims)

    if dims_arg != ("y", "x", "z", "channels"):
        print(f"weird image dims after normalize: {dims_arg}", flush=True)

    return arr, dims_arg

def make_image_container(sdata_image, image_key, scale):
    import squidpy as sq

    arr = find_image_dataarray(sdata_image, preferred_scale=scale)
    arr, dims_arg = normalize_image_for_squidpy(arr)

    print(f"image array dims: {arr.dims}", flush=True)
    print(f"image array shape: {arr.shape}", flush=True)
    print(f"squidpy dims: {dims_arg}", flush=True)

    return sq.im.ImageContainer(
        arr,
        layer=image_key,
        dims=dims_arg,
        lazy=True,
    )

def ensure_spatial_coords(adata):
    if "spatial" in adata.obsm:
        return

    for x_col, y_col in [
        ("x_centroid", "y_centroid"),
        ("x", "y"),
        ("center_x", "center_y"),
    ]:
        if x_col in adata.obs and y_col in adata.obs:
            adata.obsm["spatial"] = adata.obs[[x_col, y_col]].to_numpy(dtype=float)
            print(f"spatial coords from obs: {x_col}, {y_col}", flush=True)
            return

    raise ValueError("adata.obsm['spatial'] is missing")

def ensure_squidpy_spatial(adata, library_id, crop_size):
    ensure_spatial_coords(adata)

    adata.obs["library_id"] = str(library_id)

    spatial = adata.uns.setdefault("spatial", {})
    block = spatial.setdefault(str(library_id), {})

    block.setdefault("images", {})
    block.setdefault("metadata", {})
    block.setdefault(
        "scalefactors",
        {
            "spot_diameter_fullres": float(crop_size),
            "tissue_hires_scalef": 1.0,
            "tissue_lowres_scalef": 1.0,
        },
    )

def apply_spatial_pixel_transform(adata, cfg):
    scale = cfg.get("spatial_scale")
    offset = cfg.get("spatial_offset", [0.0, 0.0])

    if scale is None:
        return

    coords = np.asarray(adata.obsm["spatial"], dtype=float).copy()

    if isinstance(scale, (list, tuple)):
        sx = float(scale[0])
        sy = float(scale[1])
    else:
        sx = float(scale)
        sy = float(scale)

    ox = float(offset[0])
    oy = float(offset[1])

    coords[:, 0] = coords[:, 0] * sx + ox
    coords[:, 1] = coords[:, 1] * sy + oy

    adata.obsm["spatial"] = coords

    print(f"spatial pixel transform: sx={sx} sy={sy} ox={ox} oy={oy}", flush=True)
    print("pixel x range:", float(np.nanmin(coords[:, 0])), float(np.nanmax(coords[:, 0])), flush=True)
    print("pixel y range:", float(np.nanmin(coords[:, 1])), float(np.nanmax(coords[:, 1])), flush=True)

def numeric_df(x, index, prefix):
    if x is None:
        return None

    if isinstance(x, pd.DataFrame):
        df = x.copy()
    else:
        arr = np.asarray(x)
        if arr.ndim == 1:
            arr = arr[:, None]
        df = pd.DataFrame(
            arr,
            index=index,
            columns=[f"{prefix}_{i + 1}" for i in range(arr.shape[1])],
        )

    keep = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            keep.append(c)

    if not keep:
        return None

    df = df[keep].copy()
    df.columns = [str(c) if str(c).startswith(prefix) else f"{prefix}_{c}" for c in df.columns]
    return df.reset_index(drop=True)

def flatten_morphology_table(adata, key_added):
    obs = adata.obs.copy()

    if "cell_id" in obs.columns:
        cell_id = obs["cell_id"].astype(str).to_numpy()
    else:
        cell_id = obs.index.astype(str).to_numpy()

    out = pd.DataFrame({"cell_id": cell_id})
    parts = []

    for key in [key_added, "image_features", "img_features", "features"]:
        if key in adata.obsm:
            part = numeric_df(adata.obsm[key], obs.index, key)
            if part is not None:
                parts.append(part)

        if key in adata.uns:
            part = numeric_df(adata.uns[key], obs.index, key)
            if part is not None and len(part) == len(out):
                parts.append(part)

    if parts:
        out = pd.concat([out] + parts, axis=1)

    seen = set()
    cols = []

    for c in out.columns:
        if c in seen:
            continue
        seen.add(c)
        cols.append(c)

    out = out[cols]

    feature_cols = [c for c in out.columns if c != "cell_id"]
    if not feature_cols:
        raise ValueError("no numeric morphology image features found after Squidpy run")

    return out

def get_result_table(sdata, adata_key):
    if adata_key in sdata.tables:
        return sdata.tables[adata_key]

    if "table" in sdata.tables:
        print(f"use fallback table: table", flush=True)
        return sdata.tables["table"]

    keys = list(sdata.tables.keys())
    if len(keys) == 1:
        print(f"use fallback table: {keys[0]}", flush=True)
        return sdata.tables[keys[0]]

    raise KeyError(f"cannot pick result table, available={keys}")

def main():
    args = parse_args()
    cfg = require_config(args.dataset_id)

    out_dir = make_out_dir(args.dataset_id)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    import squidpy as sq

    sdata = load_sdata(cfg)
    sdata = apply_crop(sdata, cfg.get("crop"))

    print("images:", list(sdata.images.keys()), flush=True)
    print("labels:", list(sdata.labels.keys()), flush=True)
    print("tables:", list(sdata.tables.keys()), flush=True)

    labels_key = pick_key(
        sdata.labels.keys(),
        cfg.get("labels_key", DEFAULT_LABELS_KEY),
        ["cell", "label"],
    )
    image_key = pick_key(
        sdata.images.keys(),
        cfg.get("image_key", DEFAULT_IMAGE_KEY),
        ["morphology_focus", "focus", "morphology"],
    )
    adata_key = cfg.get("adata_key_added", DEFAULT_ADATA_KEY)

    print(f"labels_key: {labels_key}", flush=True)
    print(f"image_key: {image_key}", flush=True)
    print(f"adata_key_added: {adata_key}", flush=True)

    adata = get_result_table(sdata, "table")

    crop_size = int(cfg.get("crop_size", 80))
    library_id = str(cfg.get("library_id", DEFAULT_LIBRARY_ID))

    ensure_squidpy_spatial(adata, library_id, crop_size)
    apply_spatial_pixel_transform(adata, cfg)

    img = make_image_container(
        sdata.images[image_key],
        image_key=image_key,
        scale=cfg.get("scale", "scale0"),
    )
    coords = np.asarray(adata.obsm["spatial"], dtype=float)

    arr_dbg = find_image_dataarray(
        sdata.images[image_key],
        preferred_scale=cfg.get("scale", "scale0"),
    )
    arr_dbg, _ = normalize_image_for_squidpy(arr_dbg)
    vals = arr_dbg.to_numpy()

    print("spatial x range:", float(np.nanmin(coords[:, 0])), float(np.nanmax(coords[:, 0])), flush=True)
    print("spatial y range:", float(np.nanmin(coords[:, 1])), float(np.nanmax(coords[:, 1])), flush=True)
    print("image y/x:", vals.shape[0], vals.shape[1], flush=True)
    print("raw image min/max:", float(np.nanmin(vals)), float(np.nanmax(vals)), flush=True)

    for idx in [0, len(coords) // 2, len(coords) - 1]:
        cx = int(round(coords[idx, 0]))
        cy = int(round(coords[idx, 1]))
        patch = vals[
            max(0, cy - 40):cy + 41,
            max(0, cx - 40):cx + 41,
            :,
            :,
        ]

        print("debug cell:", idx, "xy:", cx, cy, "patch shape:", patch.shape, flush=True)

        if patch.size:
            print("debug patch min/max:", float(np.nanmin(patch)), float(np.nanmax(patch)), flush=True)
    # coords = np.asarray(adata.obsm["spatial"], dtype=float)
    # print("spatial x range:", float(np.nanmin(coords[:, 0])), float(np.nanmax(coords[:, 0])), flush=True)
    # print("spatial y range:", float(np.nanmin(coords[:, 1])), float(np.nanmax(coords[:, 1])), flush=True)
    #
    # arr = find_image_dataarray(sdata.images[image_key], preferred_scale=cfg.get("scale", "scale0"))
    # arr, _ = normalize_image_for_squidpy(arr)
    # vals = arr.to_numpy()
    # print("raw image min/max:", float(np.nanmin(vals)), float(np.nanmax(vals)), flush=True)
    #
    # cy = int(round(coords[0, 1]))
    # cx = int(round(coords[0, 0]))
    # patch = vals[max(0, cy - 40):cy + 41, max(0, cx - 40):cx + 41, :, :]
    # print("first centroid:", cx, cy, flush=True)
    # print("manual patch shape:", patch.shape, flush=True)
    # print("manual patch min/max:", float(np.nanmin(patch)), float(np.nanmax(patch)), flush=True)

    feature_sets = cfg.get("feature_sets")

    if not feature_sets:
        feature_sets = {
            "morphology_image": cfg.get("features", DEFAULT_FEATURES),
        }

    all_meta = {}

    for feature_name, features in feature_sets.items():
        print(f"feature set: {feature_name} {features}", flush=True)

        sq.im.calculate_image_features(
            adata=adata,
            img=img,
            layer=image_key,
            features=features,
            key_added=feature_name,
            library_id=library_id,
            spatial_key="spatial",
            spot_scale=float(cfg.get("spot_scale", 1.0)),
            n_jobs=int(cfg.get("n_jobs", 4)),
            show_progress_bar=True,
        )

        df = flatten_morphology_table(adata, feature_name)

        out_path = cache_dir / f"{feature_name}_features.parquet"
        df.to_parquet(out_path, index=False)

        meta = {
            "dataset_id": args.dataset_id,
            "feature_name": feature_name,
            "xenium_dir": cfg.get("xenium_dir"),
            "sdata_path": cfg.get("sdata_path"),
            "labels_key": labels_key,
            "image_key": image_key,
            "scale": cfg.get("scale"),
            "features": list(features),
            "images": list(sdata.images.keys()),
            "labels": list(sdata.labels.keys()),
            "tables": list(sdata.tables.keys()),
            "n_rows": int(len(df)),
            "n_features": int(len(df.columns) - 1),
            "output": str(out_path),
        }

        (cache_dir / f"{feature_name}_meta.json").write_text(
            json.dumps(meta, indent=2) + "\n"
        )

        all_meta[feature_name] = meta

        print(f"saved: {out_path}", flush=True)
        print(f"rows: {len(df)}", flush=True)
        print(f"features: {len(df.columns) - 1}", flush=True)

        if feature_name == "morphology_image_all":
            legacy_path = cache_dir / "morphology_image_features.parquet"
            df.to_parquet(legacy_path, index=False)
            print(f"saved legacy: {legacy_path}", flush=True)

    (cache_dir / "morphology_image_meta.json").write_text(
        json.dumps(all_meta, indent=2) + "\n"
    )

if __name__ == "__main__":
    main()
