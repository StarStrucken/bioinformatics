MEASUREMENTS = {
    # Expression PCs from adata.X: library-size normalize to CPM, log1p, then
    # TruncatedSVD/PCA and z-score. Exported back to nodes.csv as expression_pc*.
    # Distance: Euclidean norm on the weighted expression block embedding.
    # Refs: xenum/dump/features.py:169-187, xenum/dump/cli.py:66-73
    #       xenum/dump/features.py:217-224, xenum/dump/graph.py:15-25
    "expression": {
        "label": "expression",
        "blocks": {"expression": 1.0},
    },

    # Cell-shape block from the Xenium cells table: cell_area, nucleus_area,
    # and nucleus_area / cell_area, then z-scored for distance calculations.
    # Distance: Euclidean norm on the weighted morphology block embedding.
    # Refs: xenum/dump/features.py:189-215, xenum/dump/features.py:217-224
    #       xenum/dump/graph.py:15-25
    "morphology": {
        "label": "morphology",
        "blocks": {"morphology": 1.0},
    },

    # Full morphology-image feature table. Produced by Squidpy image features
    # and written as cache/morphology_image_features.parquet. In normal runs
    # this is the legacy copy of morphology_image_all_features.parquet.
    # Distance: Euclidean norm on the z-scored morphology-image feature block.
    # Refs: xenum/image/run_morphology.py:488-538, xenum/dump/config.py:35-41
    #       xenum/dump/features.py:75-145, xenum/dump/cli.py:48-58
    #       xenum/dump/features.py:217-224, xenum/dump/graph.py:15-25
    "morphology_image": {
        "label": "morphology_image",
        "blocks": {"morphology_image": 1.0},
    },

    # Squidpy "summary" image features from per-cell morphology crops.
    # Distance: Euclidean norm on the z-scored summary feature block.
    # Refs: xenum/image/run_morphology.py:488-538, xenum/dump/config.py:42-45
    #       xenum/dump/features.py:75-145, xenum/dump/cli.py:48-58
    #       xenum/dump/features.py:217-224, xenum/dump/graph.py:15-25
    "morphology_image_summary": {
        "label": "morphology_image_summary",
        "blocks": {"morphology_image_summary": 1.0},
    },

    # Squidpy "histogram" image features from per-cell morphology crops.
    # Distance: Euclidean norm on the z-scored histogram feature block.
    # Refs: xenum/image/run_morphology.py:488-538, xenum/dump/config.py:46-49
    #       xenum/dump/features.py:75-145, xenum/dump/cli.py:48-58
    #       xenum/dump/features.py:217-224, xenum/dump/graph.py:15-25
    "morphology_image_histogram": {
        "label": "morphology_image_histogram",
        "blocks": {"morphology_image_histogram": 1.0},
    },

    # Squidpy "texture" image features from per-cell morphology crops.
    # Distance: Euclidean norm on the z-scored texture feature block.
    # Refs: xenum/image/run_morphology.py:488-538, xenum/dump/config.py:50-53
    #       xenum/dump/features.py:75-145, xenum/dump/cli.py:48-58
    #       xenum/dump/features.py:217-224, xenum/dump/graph.py:15-25
    "morphology_image_texture": {
        "label": "morphology_image_texture",
        "blocks": {"morphology_image_texture": 1.0},
    },

    # Combined morphology-image feature table: summary + histogram + texture.
    # Distance: Euclidean norm on the z-scored combined feature block.
    # Refs: xenum/image/run_morphology.py:488-538, xenum/dump/config.py:54-57
    #       xenum/dump/features.py:75-145, xenum/dump/cli.py:48-58
    #       xenum/dump/features.py:217-224, xenum/dump/graph.py:15-25
    "morphology_image_all": {
        "label": "morphology_image_all",
        "blocks": {"morphology_image_all": 1.0},
    },

    # Sequence-like top-gene baseline: top 32 nonzero genes per adata.X row,
    # compared with Jaccard distance over gene-id sets.
    # Distance: 1 - |intersection| / |union| on top_gene_ids.
    # Refs: xenum/dump/config.py:34, xenum/dump/features.py:22-42
    #       xenum/dump/graph.py:27-41, xenum/dump/features.py:64-73
    "seq_jaccard": {
        "label": "seq_jaccard_top32",
        "blocks": {},
    },

    # Sequence-like all-detected-gene baseline: every nonzero gene id per
    # adata.X row, compared with Jaccard distance over gene-id sets.
    # Distance: 1 - |intersection| / |union| on detected_gene_ids.
    # Refs: xenum/dump/features.py:44-73, xenum/dump/graph.py:27-41
    "seq_jaccard_all": {
        "label": "seq_jaccard_all",
        "blocks": {},
    },

    # Spatial oracle from Xenium centroids. This uses the target x/y coordinates
    # directly, so it is hidden and marked leaky.
    # Distance: Euclidean norm on z-scored x_centroid/y_centroid.
    # Refs: xenum/dump/features.py:189-215, xenum/dump/features.py:217-224
    #       xenum/dump/graph.py:15-25
    "spatial": {
        "label": "spatial",
        "blocks": {"spatial": 1.0},
    },

    # Random coordinate-permutation control. It copies the real centroid array,
    # permutes coordinate rows once per seed, and uses those coordinates as the
    # direct prediction without building a graph.
    # Distance: Euclidean norm between real x/y and permuted predicted x/y.
    # Refs: xenum/dump/random_controls.py:61-82, xenum/dump/graph.py:204-334
    "random_permutation": {
        "label": "random_permutation",
        "blocks": {},
    },

    # Random-neighbor graph control. For each seed and k, every cell samples k
    # distinct other cells; the sampled graph uses the common reconstruction.
    # Distance: Euclidean prediction error after averaging sampled neighbors.
    # Refs: xenum/dump/random_controls.py:86-154, xenum/dump/graph.py:272-334
    "random_neighbors": {
        "label": "random_neighbors",
        "blocks": {},
    },

    # Raw SpaGCN XY adjacency graph. SpaGCN calculates pairwise distances from
    # Xenium centroids; those distances are used directly as kNN neighbor
    # distances instead of deriving a separate feature embedding.
    # Distance: SpaGCN.calculate_adj_matrix(histology=False)[source, target].
    # Refs: xenum/dump/spagcn.py:212-223, xenum/dump/spagcn.py:287-311
    #       xenum/dump/spagcn.py:418-455, xenum/dump/graph.py:78-107
    "spagcn_adjacency": {
        "label": "spagcn_adjacency",
        "blocks": {},
    },

    # SpaGCN run with expression matrix and XY-only SpaGCN adjacency. The latent
    # graph-convolution embedding z is exported as a feature block.
    # Distance: Euclidean norm on the z-scored SpaGCN latent embedding.
    # Refs: xenum/dump/spagcn.py:159-370, xenum/dump/graph.py:15-25
    "spagcn_xy_embedding": {
        "label": "spagcn_xy_embedding",
        "blocks": {"spagcn_xy_embedding": 1.0},
    },

    # SpaGCN run with expression matrix and XY-only SpaGCN adjacency. The
    # assignment probability matrix q is exported as a feature block.
    # Distance: Euclidean norm on the z-scored SpaGCN probability matrix.
    # Refs: xenum/dump/spagcn.py:159-370, xenum/dump/graph.py:15-25
    "spagcn_xy_probabilities": {
        "label": "spagcn_xy_probabilities",
        "blocks": {"spagcn_xy_probabilities": 1.0},
    },

    # SpaGCN run with expression matrix and XY + histology SpaGCN adjacency. The
    # latent graph-convolution embedding z is exported as a feature block.
    # Distance: Euclidean norm on the z-scored SpaGCN latent embedding.
    # Refs: xenum/dump/spagcn.py:144-175, xenum/dump/spagcn.py:212-247
    #       xenum/dump/spagcn.py:343-416, xenum/dump/graph.py:15-25
    "spagcn_histology_embedding": {
        "label": "spagcn_histology_embedding",
        "blocks": {"spagcn_histology_embedding": 1.0},
    },

    # SpaGCN run with expression matrix and XY + histology SpaGCN adjacency. The
    # assignment probability matrix q is exported as a feature block.
    # Distance: Euclidean norm on the z-scored SpaGCN probability matrix.
    # Refs: xenum/dump/spagcn.py:144-175, xenum/dump/spagcn.py:212-247
    #       xenum/dump/spagcn.py:343-416, xenum/dump/graph.py:15-25
    "spagcn_histology_probabilities": {
        "label": "spagcn_histology_probabilities",
        "blocks": {"spagcn_histology_probabilities": 1.0},
    },

    # LUNA self-trained expression-only coordinate generator. The wrapper writes
    # LUNA train/test CSVs from the same Xenium expression matrix, trains with
    # real target coordinates, infers zero-coordinate test rows, then aligns the
    # generated layout back to real centroids.
    # Distance: Euclidean XY error after similarity Procrustes alignment; no kNN.
    # Refs: xenum/dump/luna.py:140-199, xenum/dump/luna.py:258-409
    #       xenum/dump/graph.py:204-334
    "luna_expression_self": {
        "label": "luna_expression_self",
        "blocks": {},
    },

    # LUNA transfer expression-only coordinate generator. A compatible external
    # training dataset/checkpoint must be configured; otherwise the measurement
    # is recorded as skipped and no self-trained fallback is used.
    # Distance: Euclidean XY error after similarity Procrustes alignment; no kNN.
    # Refs: xenum/dump/luna.py:413-429, xenum/dump/graph.py:204-334
    "luna_expression_transfer": {
        "label": "luna_expression_transfer",
        "blocks": {},
    },

    # Deprecated manual expression + morphology mix. Kept only so old cached
    # outputs can still be recognized.
    # Distance: Euclidean norm after concatenating expression and morphology.
    # Refs: xenum/dump/features.py:217-224, xenum/dump/graph.py:15-25,
    #       xenum_distances.py:56-59
    # "expr_morph": {
    #     "label": "expression+morphology",
    #     "blocks": {"expression": 1.0, "morphology": 1.0},
    # },

    # Deprecated manual mix with morphology weighted 2x.
    # Distance: Euclidean norm after concatenating expression and 2x morphology.
    # Refs: xenum/dump/features.py:217-224, xenum/dump/graph.py:15-25
    # "morph_heavy": {
    #     "label": "morph_heavy",
    #     "blocks": {"expression": 1.0, "morphology": 2.0},
    # },

    # Deprecated expression + morphology + spatial mixer with spatial weight 0.
    # Distance: sqrt((0 * ||spatial||)^2 + ||expression||^2 + ||morphology||^2).
    # Refs: xenum/dump/features.py:217-224, xenum/dump/graph.py:15-25,
    #       xenum_distances.py:49-54
    # "mix_nonspatial": {
    #     "label": "mix_nonspatial",
    #     "blocks": {"spatial": 0.0, "expression": 1.0, "morphology": 1.0},
    # },

    # Deprecated leaky expression + morphology + spatial-centroid mix.
    # Distance: Euclidean norm after concatenating spatial, expression, morphology.
    # Refs: xenum/dump/features.py:217-224, xenum/dump/graph.py:15-25,
    #       xenum_distances.py:61-65
    # "expr_morph_spatial": {
    #     "label": "expression+morphology+spatial",
    #     "blocks": {"expression": 1.0, "morphology": 1.0, "spatial": 1.0},
    # },

    # Deprecated leaky expression + morphology + spatial mixer.
    # Distance: sqrt(||spatial||^2 + ||expression||^2 + ||morphology||^2).
    # Refs: xenum/dump/features.py:217-224, xenum/dump/graph.py:15-25,
    #       xenum_distances.py:49-54
    # "mix": {
    #     "label": "mix",
    #     "blocks": {"spatial": 1.0, "expression": 1.0, "morphology": 1.0},
    # },

    # Deprecated local-alignment sequence baseline over top-gene ids.
    # Distance: 1 - Smith-Waterman-style local-alignment score / max score.
    # Refs: xenum_common.py:103-106, xenum_common.py:121-123,
    #       xenum_distances.py:45-48
    # "seq_local": {"label": "seq_local", "blocks": {}},

}

VISIBLE_MEASUREMENTS = [
    "expression",
    "morphology",
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

CONTROL_MEASUREMENTS = [
    "random_permutation",
    "random_neighbors",
]

SPAGCN_MEASUREMENTS = [
    "spagcn_adjacency",
    "spagcn_xy_embedding",
    "spagcn_xy_probabilities",
    "spagcn_histology_embedding",
    "spagcn_histology_probabilities",
]

LUNA_MEASUREMENTS = [
    "luna_expression_self",
    "luna_expression_transfer",
]

ACTIVE_MEASUREMENTS = VISIBLE_MEASUREMENTS + HIDDEN_MEASUREMENTS

DEPRECATED_MEASUREMENTS = {
    "expr_morph",
    "morph_heavy",
    "mix_nonspatial",
    "expr_morph_spatial",
    "mix",
    "seq_jaccard",
    "seq_jaccard_all",
    "seq_local",
}

LEAKY_MEASUREMENTS = {
    "spatial",
    "mix",
    "expr_morph_spatial",
    "spagcn_adjacency",
    "spagcn_xy_embedding",
    "spagcn_xy_probabilities",
    "spagcn_histology_embedding",
    "spagcn_histology_probabilities",
    "luna_expression_self",
}

MEASUREMENT_ORDER = VISIBLE_MEASUREMENTS + OPTIONAL_MEASUREMENTS + HIDDEN_MEASUREMENTS + CONTROL_MEASUREMENTS + SPAGCN_MEASUREMENTS + LUNA_MEASUREMENTS
