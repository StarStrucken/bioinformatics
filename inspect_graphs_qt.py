#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg


MEASUREMENTS = ["spatial", "profile", "morphology", "align", "mix_nonspatial", "mix"]
MAX_VISIBLE_EDGES = 12_000
MAX_COMPONENT_EDGES = 2_000
COMPONENT_PALETTE = [
    (230, 25, 75),    # red
    (0, 130, 200),    # blue
    (60, 180, 75),    # green
    (245, 130, 48),   # orange
    (145, 30, 180),   # purple
    (70, 240, 240),   # cyan
    (240, 50, 230),   # magenta
    (210, 245, 60),   # lime
    (250, 190, 190),  # pink
    (0, 128, 128),    # teal
    (230, 190, 255),  # lavender
    (170, 110, 40),   # brown
]


def load_dump(out_dir: Path):
    nodes = {}
    edges = {}

    for m in MEASUREMENTS:
        node_path = out_dir / f"nodes_{m}.csv"
        edge_path = out_dir / f"edges_{m}.csv"

        nodes[m] = pd.read_csv(node_path)
        edges[m] = pd.read_csv(edge_path)

    return nodes, edges


def build_neighbors(edges_by_measurement):
    out = {}

    for m, edges in edges_by_measurement.items():
        for r in edges.itertuples(index=False):
            s = int(r.source)
            t = int(r.target)

            nd = float(r.neighbor_distance)
            xy = float(r.distance)

            out.setdefault(s, {}).setdefault(m, []).append((t, nd, xy))
            out.setdefault(t, {}).setdefault(m, []).append((s, nd, xy))

    for node, by_m in out.items():
        for m, vals in by_m.items():
            vals.sort(key=lambda x: x[1])

    return out

def zscore(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]

    ok = np.isfinite(x)
    counts = ok.sum(axis=0)
    mean = np.where(ok, x, 0.0).sum(axis=0) / np.maximum(counts, 1)

    x = np.where(ok, x, mean)
    z = (x - mean) / (x.std(axis=0) + 1e-6)

    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def distance_blocks(nodes):
    spatial = zscore(nodes[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32))

    profile = zscore(
        nodes[
            [
                "log_total_counts",
                "log_detected_genes",
            ]
        ].to_numpy(dtype=np.float32)
    )

    morphology = zscore(
        nodes[
            [
                "cell_area",
                "nucleus_area",
                "nucleus_cell_ratio",
            ]
        ].to_numpy(dtype=np.float32)
    )

    align = [parse_seq_ids(v) for v in nodes["top_gene_ids"].to_numpy()]

    return {
        "spatial": spatial,
        "profile": profile,
        "morphology": morphology,
        "align": align,
    }


def pair_distance(blocks, measurement, a, b):
    if measurement == "align":
        return normalized_edit_distance(blocks["align"][a], blocks["align"][b])

    if measurement in {"mix_nonspatial", "mix"}:
        spatial = blocks["spatial"][a] - blocks["spatial"][b]
        profile = blocks["profile"][a] - blocks["profile"][b]
        morphology = blocks["morphology"][a] - blocks["morphology"][b]

        d_spatial = float(np.sqrt(np.sum(spatial * spatial)))
        d_profile = float(np.sqrt(np.sum(profile * profile)))
        d_morphology = float(np.sqrt(np.sum(morphology * morphology)))
        d_align = normalized_edit_distance(blocks["align"][a], blocks["align"][b])

        w_spatial = 0.0 if measurement == "mix_nonspatial" else 1.0

        return float(
            np.sqrt(
                (w_spatial * d_spatial) ** 2
                + (0.35 * d_profile) ** 2
                + (0.50 * d_morphology) ** 2
                + (0.75 * d_align) ** 2
            )
        )

    v = blocks[measurement][a] - blocks[measurement][b]
    return float(np.sqrt(np.sum(v * v)))

def parse_seq_ids(x):
    if x is None or pd.isna(x):
        return ()

    s = str(x).strip()

    if not s:
        return ()

    return tuple(int(v) for v in s.split())


def edit_distance(a, b, insert_cost=1.0, delete_cost=1.0, substitute_cost=2.0):
    if a == b:
        return 0.0

    if not a:
        return len(b) * insert_cost

    if not b:
        return len(a) * delete_cost

    prev = np.arange(len(b) + 1, dtype=np.float32) * insert_cost

    for i, ca in enumerate(a, start=1):
        curr = np.empty(len(b) + 1, dtype=np.float32)
        curr[0] = i * delete_cost

        for j, cb in enumerate(b, start=1):
            sub = 0.0 if ca == cb else substitute_cost
            curr[j] = min(
                prev[j] + delete_cost,
                curr[j - 1] + insert_cost,
                prev[j - 1] + sub,
            )

        prev = curr

    return float(prev[-1])


def normalized_edit_distance(a, b):
    return edit_distance(a, b, 1.0, 1.0, 2.0) / max(len(a), len(b), 1)

class GraphPane(QtWidgets.QWidget):
    nodeClicked = QtCore.Signal(int)

    def __init__(self, measurement, nodes, edges):
        super().__init__()

        self.measurement = measurement
        self.nodes = nodes
        self.edges = edges

        self.x = nodes["x_centroid"].to_numpy(dtype=float)
        self.y = nodes["y_centroid"].to_numpy(dtype=float)

        self.component_id = (
            nodes["component_id"].to_numpy(dtype=int)
            if "component_id" in nodes
            else np.zeros(len(nodes), dtype=int)
        )
        self.component_size = (
            nodes["component_size"].to_numpy(dtype=int)
            if "component_size" in nodes
            else np.ones(len(nodes), dtype=int)
        )

        self.visible_components = set(np.unique(self.component_id).astype(int))
        self.show_singletons = True
        self.visible_node_mask = np.ones(len(nodes), dtype=bool)

        self.edge_pairs = (
            edges[["source", "target"]].to_numpy(dtype=int)
            if len(edges)
            else np.empty((0, 2), dtype=int)
        )
        self.rng = np.random.default_rng(0)
        self.visible_edge_pairs = self._sample_pairs(self.edge_pairs, MAX_VISIBLE_EDGES)
        self.component_pair_cache = {}

        self.plot = pg.PlotWidget()
        self.plot.setTitle(measurement)
        self.plot.invertY(True)
        self.plot.showGrid(x=True, y=True, alpha=0.15)

        self.all_edges = pg.PlotDataItem(
            pen=pg.mkPen(120, 120, 120, 70, width=0.8),
            connect="finite",
        )

        self.component_edges = pg.PlotDataItem(
            pen=pg.mkPen(0, 120, 255, 170, width=1.8),
            connect="finite",
        )

        self.selected_edges = pg.PlotDataItem(
            pen=pg.mkPen(220, 30, 30, 255, width=3.0),
            connect="finite",
        )

        self.scatter = pg.ScatterPlotItem(
            size=6,
            pen=pg.mkPen(40, 40, 40, 80),
        )
        self.scatter.sigClicked.connect(self._clicked)

        self.component_nodes = pg.ScatterPlotItem(
            size=9,
            brush=pg.mkBrush(0, 120, 255, 150),
            pen=pg.mkPen(0, 60, 120, 180),
        )

        self.neighbor_nodes = pg.ScatterPlotItem(
            size=11,
            brush=pg.mkBrush(230, 40, 40, 240),
            pen=pg.mkPen(80, 0, 0, 220),
        )

        self.selected_node = pg.ScatterPlotItem(
            size=16,
            brush=pg.mkBrush(255, 210, 0, 255),
            pen=pg.mkPen(0, 0, 0, 255, width=2),
        )

        for item in [
            self.all_edges,
            self.component_edges,
            self.selected_edges,
            self.component_nodes,
            self.neighbor_nodes,
            self.selected_node,
        ]:
            item.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)

        self.plot.addItem(self.all_edges)
        self.plot.addItem(self.component_edges)
        self.plot.addItem(self.selected_edges)
        self.plot.addItem(self.scatter)
        self.plot.addItem(self.component_nodes)
        self.plot.addItem(self.neighbor_nodes)
        self.plot.addItem(self.selected_node)

        self._draw_all_edges()
        self._refresh_nodes()
        self._refresh_edges()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.plot)

    def _sample_pairs(self, pairs, limit):
        if pairs is None or len(pairs) <= limit:
            return pairs

        idx = self.rng.choice(len(pairs), limit, replace=False)
        return pairs[idx]


    def _component_pairs(self, comp):
        comp = int(comp)

        if comp in self.component_pair_cache:
            return self.component_pair_cache[comp]

        if len(self.edge_pairs) == 0:
            pairs = self.edge_pairs
        else:
            src = self.edge_pairs[:, 0]
            dst = self.edge_pairs[:, 1]

            src_comp = self.component_id[src] == comp
            dst_comp = self.component_id[dst] == comp
            visible = self.visible_node_mask[src] & self.visible_node_mask[dst]

            pairs = self.edge_pairs[src_comp & dst_comp & visible]
            pairs = self._sample_pairs(pairs, MAX_COMPONENT_EDGES)

        self.component_pair_cache[comp] = pairs
        return pairs

    def _component_brushes(self, idx=None):
        if idx is None:
            comps = self.component_id
        else:
            comps = self.component_id[idx]

        brushes = []

        for c in comps:
            c = int(c)

            if c < 0:
                brushes.append(pg.mkBrush(120, 120, 120, 190))
                continue

            if c < len(COMPONENT_PALETTE):
                r, g, b = COMPONENT_PALETTE[c]
                brushes.append(pg.mkBrush(r, g, b, 220))
                continue

            hue = (c * 0.61803398875) % 1.0
            color = QtGui.QColor.fromHsvF(hue, 0.85, 0.95, 0.85)
            brushes.append(pg.mkBrush(color))

        return brushes

    def _edge_lines(self, pairs):
        if pairs is None or len(pairs) == 0:
            return [], []

        xs = np.empty(len(pairs) * 3, dtype=float)
        ys = np.empty(len(pairs) * 3, dtype=float)

        xs[0::3] = self.x[pairs[:, 0]]
        xs[1::3] = self.x[pairs[:, 1]]
        xs[2::3] = np.nan

        ys[0::3] = self.y[pairs[:, 0]]
        ys[1::3] = self.y[pairs[:, 1]]
        ys[2::3] = np.nan

        return xs, ys

    def _draw_all_edges(self):
        self._refresh_edges()

    def _clicked(self, item, points):
        if points is None or len(points) == 0:
            return

        data = points[0].data()

        if data is None:
            return

        self.nodeClicked.emit(int(data))

    def show_selection(self, node, neighbors):
        if node is None:
            return

        if node >= len(self.visible_node_mask) or not self.visible_node_mask[node]:
            self.component_nodes.setData([], [])
            self.neighbor_nodes.setData([], [])
            self.selected_node.setData([], [])
            self.component_edges.setData([], [])
            self.selected_edges.setData([], [])
            return

        comp = self.component_id[node]
        comp_nodes = np.flatnonzero((self.component_id == comp) & self.visible_node_mask)

        self.component_nodes.setData(
            x=self.x[comp_nodes],
            y=self.y[comp_nodes],
        )

        comp_pairs = self._component_pairs(comp)
        xs, ys = self._edge_lines(comp_pairs)
        self.component_edges.setData(xs, ys)

        neigh = np.asarray(
            [n for n, _, _ in neighbors if n < len(self.visible_node_mask) and self.visible_node_mask[n]],
            dtype=int,
        )

        if len(neigh):
            self.neighbor_nodes.setData(
                x=self.x[neigh],
                y=self.y[neigh],
            )

            selected_pairs = np.c_[
                np.full(len(neigh), node, dtype=int),
                neigh,
            ]

            xs, ys = self._edge_lines(selected_pairs)
            self.selected_edges.setData(xs, ys)
        else:
            self.neighbor_nodes.setData([], [])
            self.selected_edges.setData([], [])

        self.selected_node.setData(
            x=[self.x[node]],
            y=[self.y[node]],
        )

    def set_component_visible(self, component_id, visible):
        component_id = int(component_id)

        if visible:
            self.visible_components.add(component_id)
        else:
            self.visible_components.discard(component_id)

        self._refresh_visibility()


    def set_singletons_visible(self, visible):
        self.show_singletons = bool(visible)
        self._refresh_visibility()


    def _refresh_visibility(self):
        comp_ok = np.isin(self.component_id, list(self.visible_components))

        if self.show_singletons:
            single_ok = np.ones_like(comp_ok, dtype=bool)
        else:
            single_ok = self.component_size > 1

        self.visible_node_mask = comp_ok & single_ok

        self._refresh_nodes()
        self._refresh_edges()

        self.component_nodes.setData([], [])
        self.neighbor_nodes.setData([], [])
        self.selected_node.setData([], [])
        self.component_edges.setData([], [])
        self.selected_edges.setData([], [])


    def _refresh_nodes(self):
        idx = np.flatnonzero(self.visible_node_mask)

        self.scatter.setData(
            x=self.x[idx],
            y=self.y[idx],
            brush=self._component_brushes(idx),
            data=idx,
        )


    def _refresh_edges(self):
        if len(self.edge_pairs) == 0:
            self.all_edges.setData([], [])
            return

        keep = (
            self.visible_node_mask[self.edge_pairs[:, 0]]
            & self.visible_node_mask[self.edge_pairs[:, 1]]
        )

        pairs = self.edge_pairs[keep]
        pairs = self._sample_pairs(pairs, MAX_VISIBLE_EDGES)

        xs, ys = self._edge_lines(pairs)
        self.all_edges.setData(xs, ys)

        self.component_pair_cache = {}


class NeighborTable(QtWidgets.QTableWidget):
    def __init__(self):
        super().__init__()
        self.setColumnCount(3 + len(MEASUREMENTS))
        self.setHorizontalHeaderLabels(["neighbor", "cell_id", "overlap", *MEASUREMENTS])

        self.horizontalHeader().setStretchLastSection(True)
        self.setSortingEnabled(True)

    def show_node(self, node, nodes, blocks, neighbors_by_measurement):
        union = {}

        for m in MEASUREMENTS:
            for rank, (n, nd, xy) in enumerate(neighbors_by_measurement.get(m, []), start=1):
                union.setdefault(n, {})[m] = (rank, nd, xy)

        rows = []

        for n, hits in union.items():
            overlap = len(hits)
            best_rank = min(v[0] for v in hits.values())
            rows.append((n, overlap, best_rank, hits))

        rows.sort(key=lambda x: (-x[1], x[2], x[0]))

        self.setSortingEnabled(False)
        self.setRowCount(len(rows))

        base_nodes = nodes["spatial"]

        for row_i, (n, overlap, _, hits) in enumerate(rows):
            cell_id = str(base_nodes.iloc[n].get("cell_id", n))

            base_vals = [
                str(n),
                cell_id,
                str(overlap),
            ]

            for col_i, val in enumerate(base_vals):
                item = QtWidgets.QTableWidgetItem(val)
                if overlap > 1:
                    item.setBackground(QtGui.QColor(255, 245, 180))
                self.setItem(row_i, col_i, item)

            for j, m in enumerate(MEASUREMENTS, start=3):
                d = pair_distance(blocks, m, node, n)

                if m in hits:
                    rank, nd, xy = hits[m]
                    txt = f"EDGE #{rank} d={d:.3g} xy={xy:.1f}"
                    item = QtWidgets.QTableWidgetItem(txt)
                    item.setBackground(QtGui.QColor(180, 255, 190))
                    item.setForeground(QtGui.QColor(0, 80, 0))
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                else:
                    txt = f"d={d:.3g}"
                    item = QtWidgets.QTableWidgetItem(txt)
                    item.setForeground(QtGui.QColor(120, 120, 120))

                self.setItem(row_i, j, item)

        self.setSortingEnabled(True)
        self.resizeColumnsToContents()

class CollapsibleSection(QtWidgets.QWidget):
    def __init__(self, title, expanded=False):
        super().__init__()

        self.button = QtWidgets.QToolButton()
        self.button.setText(title)
        self.button.setCheckable(True)
        self.button.setChecked(expanded)
        self.button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

        self.content = QtWidgets.QWidget()
        self.content_layout = QtWidgets.QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(18, 2, 2, 6)
        self.content_layout.setSpacing(2)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.button)
        layout.addWidget(self.content)

        self.button.toggled.connect(self._toggle)
        self._toggle(expanded)

    def _toggle(self, checked):
        self.content.setVisible(checked)
        arrow = QtCore.Qt.ArrowType.DownArrow if checked else QtCore.Qt.ArrowType.RightArrow
        self.button.setArrowType(arrow)

    def addWidget(self, widget):
        self.content_layout.addWidget(widget)

class ComponentFilterPanel(QtWidgets.QWidget):
    def __init__(self, panes):
        super().__init__()

        self.panes = panes

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        title = QtWidgets.QLabel("components")
        layout.addWidget(title)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)

        body = QtWidgets.QWidget()
        body_layout = QtWidgets.QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)

        for m in MEASUREMENTS:
            pane = panes[m]

            n_singletons = int((pane.component_size == 1).sum())

            comps = pd.DataFrame(
                {
                    "component_id": pane.component_id,
                    "component_size": pane.component_size,
                }
            ).drop_duplicates()

            comps = comps[comps["component_size"] > 1]
            comps = comps.sort_values(["component_size", "component_id"], ascending=[False, True])

            section = CollapsibleSection(
                f"{m}  components={len(comps)}  singletons={n_singletons}",
                expanded=(m == "spatial"),
            )

            single = QtWidgets.QCheckBox(f"singletons ({n_singletons})")
            single.setChecked(True)
            single.toggled.connect(
                lambda checked, mm=m: self.panes[mm].set_singletons_visible(checked)
            )
            section.addWidget(single)

            for r in comps.itertuples(index=False):
                cid = int(r.component_id)
                size = int(r.component_size)

                cb = QtWidgets.QCheckBox(f"C{cid} ({size})")
                cb.setChecked(True)
                cb.toggled.connect(
                    lambda checked, mm=m, cc=cid: self.panes[mm].set_component_visible(cc, checked)
                )
                section.addWidget(cb)

            body_layout.addWidget(section)

        body_layout.addStretch(1)

        scroll.setWidget(body)
        layout.addWidget(scroll)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, out_dir: Path):
        super().__init__()

        self.out_dir = out_dir
        self.nodes, self.edges = load_dump(out_dir)
        self.neighbors = build_neighbors(self.edges)
        self.blocks = distance_blocks(self.nodes["spatial"])

        self.setWindowTitle(f"Xenium graph inspector: {out_dir}")

        root = QtWidgets.QWidget()
        self.setCentralWidget(root)

        self.panes = {}

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(2)

        for i, m in enumerate(MEASUREMENTS):
            pane = GraphPane(m, self.nodes[m], self.edges[m])
            pane.nodeClicked.connect(self.select_node)
            self.panes[m] = pane
            grid.addWidget(pane, i // 2, i % 2)

        master = self.panes["spatial"].plot
        for m, pane in self.panes.items():
            if m != "spatial":
                pane.plot.setXLink(master)
                pane.plot.setYLink(master)

        self.info = QtWidgets.QLabel("click a node")
        self.table = NeighborTable()
        self.filters = ComponentFilterPanel(self.panes)

        side = QtWidgets.QVBoxLayout()
        side.addWidget(self.info)
        side.addWidget(self.table, stretch=3)
        side.addWidget(self.filters, stretch=2)

        layout = QtWidgets.QHBoxLayout(root)
        layout.addLayout(grid, stretch=4)
        layout.addLayout(side, stretch=2)

        self.resize(1600, 900)

    def select_node(self, node):
        cell_id = self.nodes["spatial"].iloc[node].get("cell_id", node)

        self.info.setText(f"node={node} cell_id={cell_id}")

        by_m = self.neighbors.get(node, {})

        for m, pane in self.panes.items():
            pane.show_selection(node, by_m.get(m, []))

        self.table.show_node(node, self.nodes, self.blocks, by_m)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_id")
    p.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = p.parse_args()

    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")
    pg.setConfigOptions(antialias=False)

    out_dir = args.outputs / args.dataset_id

    app = QtWidgets.QApplication([])
    app.setStyleSheet("""
    QWidget {
        background: white;
        color: black;
    }
    QTableWidget {
        background: white;
        color: black;
        gridline-color: #cccccc;
    }
    QHeaderView::section {
        background: #eeeeee;
        color: black;
    }
    QLabel {
        color: black;
    }
    """)
    win = MainWindow(out_dir)
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
