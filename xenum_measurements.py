MEASUREMENTS = {
    # adata.X -> log1p CPM -> sklearn.decomposition.TruncatedSVD/PCA
    "expression": {
        "label": "expression",
        "blocks": {"expression": 1.0},
    },

    # cells["cell_area"], cells["nucleus_area"]; nucleus_area / cell_area
    "morphology": {
        "label": "morphology",
        "blocks": {"morphology": 1.0},
    },

    # cache/morphology_image_features.parquet; squidpy/cpmeasure image features
    "morphology_image": {
        "label": "morphology_image",
        "blocks": {"morphology_image": 1.0},
    },

    "morphology_image_summary": {
        "label": "morphology_image_summary",
        "blocks": {"morphology_image_summary": 1.0},
    },

    "morphology_image_histogram": {
        "label": "morphology_image_histogram",
        "blocks": {"morphology_image_histogram": 1.0},
    },

    "morphology_image_texture": {
        "label": "morphology_image_texture",
        "blocks": {"morphology_image_texture": 1.0},
    },

    "morphology_image_all": {
        "label": "morphology_image_all",
        "blocks": {"morphology_image_all": 1.0},
    },

    # adata.X row nonzero values -> top 32 gene ids; Jaccard distance
    "seq_jaccard": {
        "label": "seq_jaccard_top32",
        "blocks": {},
    },

    # adata.X row nonzero gene ids; Jaccard distance
    "seq_jaccard_all": {
        "label": "seq_jaccard_all",
        "blocks": {},
    },

    # cells["x_centroid"], cells["y_centroid"]; leaky oracle
    "spatial": {
        "label": "spatial",
        "blocks": {"spatial": 1.0},
    },

    # expression + morphology; old manual mix
    # "expr_morph": {
    #     "label": "expression+morphology",
    #     "blocks": {"expression": 1.0, "morphology": 1.0},
    # },

    # expression + morphology; old manual mix
    # "morph_heavy": {
    #     "label": "morph_heavy",
    #     "blocks": {"expression": 1.0, "morphology": 2.0},
    # },

    # expression + morphology; old duplicate when spatial weight is 0
    # "mix_nonspatial": {
    #     "label": "mix_nonspatial",
    #     "blocks": {"spatial": 0.0, "expression": 1.0, "morphology": 1.0},
    # },

    # expression + morphology + cells["x_centroid"], cells["y_centroid"]; leaky
    # "expr_morph_spatial": {
    #     "label": "expression+morphology+spatial",
    #     "blocks": {"expression": 1.0, "morphology": 1.0, "spatial": 1.0},
    # },

    # expression + morphology + cells["x_centroid"], cells["y_centroid"]; leaky
    # "mix": {
    #     "label": "mix",
    #     "blocks": {"spatial": 1.0, "expression": 1.0, "morphology": 1.0},
    # },

    # adata.X row top ids; old sequence experiment
    # "seq_local": {"label": "seq_local", "blocks": {}},

    # adata.X row nonzero values -> top 32 gene ids; alignment-like sequence distance
    "seq_blast": {
        "label": "seq_blast_top32",
        "blocks": {},
    },
}

VISIBLE_MEASUREMENTS = [
    "expression",
    "morphology",
    "seq_jaccard",
    "seq_jaccard_all",
    "seq_blast",
]

OPTIONAL_MEASUREMENTS = [
    "morphology_image",
    "morphology_image_summary",
    "morphology_image_histogram",
    "morphology_image_texture",
    "morphology_image_all",
]

HIDDEN_MEASUREMENTS = [
    "spatial",
]

ACTIVE_MEASUREMENTS = VISIBLE_MEASUREMENTS + HIDDEN_MEASUREMENTS

DEPRECATED_MEASUREMENTS = {
    "expr_morph",
    "morph_heavy",
    "mix_nonspatial",
    "expr_morph_spatial",
    "mix",
    "seq_local",
}

LEAKY_MEASUREMENTS = {
    "spatial",
    "mix",
    "expr_morph_spatial",
}

MEASUREMENT_ORDER = VISIBLE_MEASUREMENTS + OPTIONAL_MEASUREMENTS + HIDDEN_MEASUREMENTS
