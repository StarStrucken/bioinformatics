from __future__ import annotations

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

K = 4
BENCH_K_VALUES = (1, 2, 3, 4, 5, 8, 12, 16, 24, 32)
EXPRESSION_PCS = 30
LEARNED_MIX_NAME = "learned_mix"
RUN_LEARNED_MIX = False
LEARNED_BASE_MEASUREMENTS = (
    "expression",
    "morphology",
    "morphology_image",
    "seq_jaccard",
    "seq_jaccard_all",
    "seq_blast",
)
LEARNED_WEIGHT_VALUES = (0.0, 0.25, 0.5, 1.0, 2.0)
LEARNED_MIN_COVERAGE = 0.95
LEARNED_SCORE_MODE = "median_p90"
LEARNED_P90_WEIGHT = 0.5

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
