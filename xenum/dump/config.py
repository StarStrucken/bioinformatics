from __future__ import annotations

import os

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

def int_env(name, default):
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return max(1, int(default))

def learned_mix_default_workers():
    cpus = int_env("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)
    return max(1, min(8, cpus))

K = 4
BENCH_K_VALUES = (1, 2, 3, 4, 5, 8, 12, 16, 24, 32)
RANDOM_SEEDS = tuple(range(20))
EXPRESSION_PCS = 30
SPAGCN_SEED = int_env("XENUM_SPAGCN_SEED", 100)
SPAGCN_NUM_PCS = int_env("XENUM_SPAGCN_NUM_PCS", 50)
SPAGCN_MIN_CELLS = int_env("XENUM_SPAGCN_MIN_CELLS", 3)
SPAGCN_MAX_EPOCHS = int_env("XENUM_SPAGCN_MAX_EPOCHS", 200)
SPAGCN_P = float(os.environ.get("XENUM_SPAGCN_P", "0.5"))
SPAGCN_RESOLUTION = float(os.environ.get("XENUM_SPAGCN_RESOLUTION", "0.4"))
SPAGCN_LR = float(os.environ.get("XENUM_SPAGCN_LR", "0.005"))
SPAGCN_TOL = float(os.environ.get("XENUM_SPAGCN_TOL", "0.005"))
SPAGCN_ALPHA = float(os.environ.get("XENUM_SPAGCN_ALPHA", "1.0"))
SPAGCN_BETA = int_env("XENUM_SPAGCN_BETA", 49)
LUNA_PYTHON = os.environ.get("XENUM_LUNA_PYTHON", "").strip()
LUNA_SEED = int_env("XENUM_LUNA_SEED", 0)
LUNA_EPOCHS = int_env("XENUM_LUNA_EPOCHS", 1)
LUNA_BATCH_SIZE = int_env("XENUM_LUNA_BATCH_SIZE", 1)
LUNA_GPUS_PER_NODE = int(os.environ.get("XENUM_LUNA_GPUS_PER_NODE", "1"))
LUNA_TIMEOUT_SEC = int(os.environ.get("XENUM_LUNA_TIMEOUT_SEC", "0") or "0")
LUNA_ALLOW_REFLECTION = os.environ.get("XENUM_LUNA_ALLOW_REFLECTION", "1").strip().lower() not in {"0", "false", "no"}
LEARNED_MIX_NAME = "learned_mix"
LEARNED_MIX_MODE = "neighbor_union_v1"
LEARNED_MIX_OUTPUT_K = 0
RUN_LEARNED_MIX = True
LEARNED_BASE_MEASUREMENTS = (
    "expression",
    "morphology",
    "morphology_image",
    "seq_jaccard_all",
)
LEARNED_WEIGHT_VALUES = (0.0, 0.25, 0.5, 1.0, 2.0)
LEARNED_MIN_COVERAGE = 0.95
LEARNED_SCORE_MODE = "median_p90"
LEARNED_P90_WEIGHT = 0.5
LEARNED_MIX_WORKERS = int_env("XENUM_LEARNED_MIX_WORKERS", learned_mix_default_workers())

BEST_K_MIN_PRED_SPREAD_RATIO = 0.35
LEARNED_MIN_PRED_SPREAD_RATIO = 0.35

USE_NEIGHBOR_CUTOFF = True
CUTOFF_QUANTILE = 0.995
CUTOFF_MAD = 8.0
MIN_EDGES_PER_NODE = 0
TOP_GENES_PER_CELL = 32
MORPHOLOGY_IMAGE_FEATURE_FILES = {
    "morphology_image": (
        "cache/morphology_image_features.parquet",
        "cache/morphology_image_features.csv",
        "morphology_image_features.parquet",
        "morphology_image_features.csv",
    ),
    "morphology_image_summary": (
        "cache/morphology_image_summary_features.parquet",
        "cache/morphology_image_summary_features.csv",
    ),
    "morphology_image_histogram": (
        "cache/morphology_image_histogram_features.parquet",
        "cache/morphology_image_histogram_features.csv",
    ),
    "morphology_image_texture": (
        "cache/morphology_image_texture_features.parquet",
        "cache/morphology_image_texture_features.csv",
    ),
    "morphology_image_all": (
        "cache/morphology_image_all_features.parquet",
        "cache/morphology_image_all_features.csv",
    ),
}
REPORT_DIR = "reports"
CACHE_DIR = "cache"
DIAGNOSTICS_DIR = "diagnostics"

CELL_TABLE_NAMES = ("cells.csv.gz", "cells.csv", "cells.parquet")

NODE_BASE_COLS = [
    "cell_id",
    "x_centroid",
    "y_centroid",
    "x_norm",
    "y_norm",
    "log_total_counts",
    "log_detected_genes",
    "cell_area",
    "nucleus_area",
    "nucleus_cell_ratio",
]

EDGE_COLS = [
    "measurement",
    "source",
    "target",
    "neighbor_distance",
    "neighbor_weight",
    "xy_distance",
    "expression_distance",
    "morphology_distance",
    "source_cell_id",
    "target_cell_id",
    "source_component_id",
    "target_component_id",
    "source_component_size",
    "target_component_size",
    "same_component",
]
