MEASUREMENTS = {
    "expression": {"label": "expression", "blocks": {"expression": 1.0}},
    "morphology": {"label": "morphology", "blocks": {"morphology": 1.0}},

    "expr_morph": {
        "label": "expression+morphology",
        "blocks": {"expression": 1.0, "morphology": 1.0},
    },

    "morph_heavy": {
        "label": "morph_heavy",
        "blocks": {"spatial": 0.2, "expression": 1.0, "morphology": 2.0},
    },

    "seq_jaccard": {"label": "seq_jaccard", "blocks": {}},

    "spatial": {"label": "spatial", "blocks": {"spatial": 1.0}},

    # "mix_nonspatial": {
    #     "label": "mix_nonspatial",
    #     "blocks": {"spatial": 0.0, "expression": 1.0, "morphology": 1.0},
    # },
    # "expr_morph_spatial": {
    #     "label": "expression+morphology+spatial",
    #     "blocks": {"expression": 1.0, "morphology": 1.0, "spatial": 1.0},
    # },
    # "mix": {
    #     "label": "mix",
    #     "blocks": {"spatial": 1.0, "expression": 1.0, "morphology": 1.0},
    # },
    # "seq_local": {"label": "seq_local", "blocks": {}},
    # "seq_blast": {"label": "seq_blast", "blocks": {}},
}

VISIBLE_MEASUREMENTS = [
    "expression",
    "morphology",
    "expr_morph",
    "morph_heavy",
    "seq_jaccard",
]

HIDDEN_MEASUREMENTS = [
    "spatial",
]

ACTIVE_MEASUREMENTS = VISIBLE_MEASUREMENTS + HIDDEN_MEASUREMENTS

LEAKY_MEASUREMENTS = {
    "spatial",
    "mix",
    "expr_morph_spatial",
}

MEASUREMENT_ORDER = ACTIVE_MEASUREMENTS
