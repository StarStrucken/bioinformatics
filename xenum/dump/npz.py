from __future__ import annotations

import numpy as np

from xenum_measurements import MEASUREMENTS

def write_npz(out_dir, measurement, graph_nodes, edges):
    node_features = graph_nodes[["log_total_counts", "log_detected_genes", "cell_area", "nucleus_area", "nucleus_cell_ratio", "component_id", "component_size"]].to_numpy(dtype=np.float32)
    pos = graph_nodes[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
    edge_index = edges[["source", "target"]].to_numpy(dtype=np.int64).T if len(edges) else np.empty((2, 0), dtype=np.int64)
    edge_attr = edges[["neighbor_distance", "xy_distance", "expression_distance", "morphology_distance", "same_component"]].to_numpy(dtype=np.float32) if len(edges) else np.empty((0, 5), dtype=np.float32)
    np.savez_compressed(
        out_dir / f"representation_{measurement}.npz",
        node_features=node_features,
        node_positions=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        cell_ids=graph_nodes["cell_id"].to_numpy(dtype=str),
        feature_columns=np.asarray(["log_total_counts", "log_detected_genes", "cell_area", "nucleus_area", "nucleus_cell_ratio", "component_id", "component_size"], dtype=str),
        edge_feature_columns=np.asarray(["neighbor_distance", "xy_distance", "expression_distance", "morphology_distance", "same_component"], dtype=str),
        measurement=np.asarray([MEASUREMENTS[measurement]["label"]], dtype=str),
    )
