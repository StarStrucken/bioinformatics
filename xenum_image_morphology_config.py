from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"

DEFAULT_PIXEL_SIZE = 0.2125
DEFAULT_CROP_SIZE = 80

DATASET_IDS = (
    "tiny_human_kidney_protein_v4",
    "tiny_human_ovary_multimodal_v4",
    "tiny_human_ovary_nucexp_v4",
    "tiny_mouse_ileum_multimodal_v3",
    "tiny_mouse_ileum_nucexp_v3",
    "breast_2fov_v2",
    "lung_2fov_v2",
    "prostate_prime_5k_v3",
    "ovarian_prime_5k_v3",
    "lymph_node_prime_5k_v3",
)

FEATURE_SETS = {
    "morphology_image_summary": ["summary"],
    "morphology_image_histogram": ["histogram"],
    "morphology_image_texture": ["texture"],
    "morphology_image_all": ["summary", "histogram", "texture"],
}

def pixel_size_for(dataset_id):
    path = DATA_ROOT / str(dataset_id) / "experiment.xenium"

    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        pixel_size = float(meta.get("pixel_size", DEFAULT_PIXEL_SIZE))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pixel_size = DEFAULT_PIXEL_SIZE

    if pixel_size <= 0:
        return DEFAULT_PIXEL_SIZE

    return pixel_size

def image_morphology_config(dataset_id):
    pixel_size = pixel_size_for(dataset_id)

    return {
        "xenium_dir": f"data/{dataset_id}",
        "image_glob": "morphology_focus/*.ome.tif",
        "cell_boundaries": "cell_boundaries.parquet",
        "nucleus_boundaries": "nucleus_boundaries.parquet",
        "cells": "cells.parquet",
        "labels_key": "cell_labels",
        "image_key": "morphology_focus",
        "adata_key_added": "morphology",
        "library_id": "xenium",
        "scale": "scale0",
        "crop_size": DEFAULT_CROP_SIZE,
        "spot_scale": 1.0,
        "spatial_scale": 1.0 / pixel_size,
        "spatial_offset": [0.0, 0.0],
        "feature_sets": {name: list(features) for name, features in FEATURE_SETS.items()},
    }

DATASET_IMAGE_MORPHOLOGY = {
    dataset_id: image_morphology_config(dataset_id)
    for dataset_id in DATASET_IDS
}
