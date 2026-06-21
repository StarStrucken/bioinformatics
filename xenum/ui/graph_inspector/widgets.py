from __future__ import annotations

import numpy as np
import pandas as pd

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from xenum_distances import pair_distance
from xenum_measurements import OPTIONAL_MEASUREMENTS, VISIBLE_MEASUREMENTS

from .config import COMPONENT_PALETTE, FALLBACK_PREDICTION_K, MAX_COMPONENT_EDGES, MAX_VISIBLE_EDGES, MAX_VISIBLE_NODES

def make_side_label(text):
    label = QtWidgets.QLabel(text)
    label.setWordWrap(True)
    label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
    label.setSizePolicy(
        QtWidgets.QSizePolicy.Policy.Preferred,
        QtWidgets.QSizePolicy.Policy.Maximum,
    )
    return label

class GraphPane(QtWidgets.QWidget):
    nodeClicked = QtCore.Signal(int)

    def __init__(self, measurement, nodes, edges, axis_values):
        super().__init__()

        self.measurement = measurement
        self.nodes = nodes
        self.edges = edges
        self.axis_values = axis_values
        self.x_axis = "spatial.x"
        self.y_axis = "spatial.y"
        self.y_scale = -1.0

        self.x = self.axis_values[self.x_axis].astype(float)
        self.y = self.axis_values[self.y_axis].astype(float) * self.y_scale

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
        self.component_pair_cache = {}

        self.plot = pg.PlotWidget()
        self.plot.setTitle(measurement)
        self.plot.invertY(False)
        self.plot.setLabel("bottom", self.x_axis)
        self.plot.setLabel("left", self._y_label())
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

        self._refresh_nodes()
        self._refresh_edges()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.plot)

    def _sample_indices(self, idx, limit):
        if idx is None or len(idx) <= limit:
            return idx

        take = self.rng.choice(len(idx), limit, replace=False)
        return idx[take]

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

    def _y_label(self):
        if abs(self.y_scale - 1.0) < 1e-9:
            return self.y_axis
        return f"{self.y_axis} * {self.y_scale:g}"

    def set_axes(self, x_axis, y_axis, y_scale):
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.y_scale = float(y_scale)
        self.x = self.axis_values[x_axis].astype(float)
        self.y = self.axis_values[y_axis].astype(float) * self.y_scale
        self.plot.setLabel("bottom", x_axis)
        self.plot.setLabel("left", self._y_label())
        self.plot.invertY(False)
        self._refresh_nodes()
        self._refresh_edges()
        self.component_nodes.setData([], [])
        self.neighbor_nodes.setData([], [])
        self.selected_node.setData([], [])
        self.component_edges.setData([], [])
        self.selected_edges.setData([], [])

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
        idx = self._sample_indices(idx, MAX_VISIBLE_NODES)

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

class PredictionPane(QtWidgets.QWidget):
    nodeClicked = QtCore.Signal(int)

    def __init__(self, measurement, predictions):
        super().__init__()

        self.measurement = measurement
        self.predictions = predictions
        self.k = int(predictions.attrs.get("k", FALLBACK_PREDICTION_K))

        self.node_ids = predictions["node"].to_numpy(dtype=int)
        self.node_to_row = {int(v): i for i, v in enumerate(self.node_ids)}

        self.x = predictions["x"].to_numpy(dtype=float)
        self.y = -predictions["y"].to_numpy(dtype=float)
        self.pred_x = predictions["pred_x"].to_numpy(dtype=float)
        self.pred_y = -predictions["pred_y"].to_numpy(dtype=float)
        self.error = predictions["error"].to_numpy(dtype=float)
        self.used_neighbors = predictions["used_neighbors"].to_numpy(dtype=int)

        self.ok = (
            np.isfinite(self.x)
            & np.isfinite(self.y)
            & np.isfinite(self.pred_x)
            & np.isfinite(self.pred_y)
        )

        self.plot = pg.PlotWidget()
        self.plot.setTitle(f"{measurement} prediction k={self.k}")
        self.plot.setLabel("bottom", "x")
        self.plot.setLabel("left", "y * -1")
        self.plot.showGrid(x=True, y=True, alpha=0.15)

        self.lines = pg.PlotDataItem(
            pen=pg.mkPen(80, 80, 80, 45, width=0.7),
            connect="finite",
        )

        self.real_scatter = pg.ScatterPlotItem(
            size=4,
            brush=pg.mkBrush(150, 150, 150, 45),
            pen=pg.mkPen(120, 120, 120, 35),
        )
        self.real_scatter.sigClicked.connect(self._clicked)

        self.pred_scatter = pg.ScatterPlotItem(
            size=7,
            pen=pg.mkPen(20, 20, 20, 80),
        )
        self.pred_scatter.sigClicked.connect(self._clicked)

        self.selected_line = pg.PlotDataItem(
            pen=pg.mkPen(220, 30, 30, 255, width=3.0),
            connect="finite",
        )

        self.selected_node = pg.ScatterPlotItem(
            size=15,
            brush=pg.mkBrush(255, 210, 0, 255),
            pen=pg.mkPen(0, 0, 0, 255, width=2),
        )

        self.predicted_node = pg.ScatterPlotItem(
            size=15,
            brush=pg.mkBrush(230, 40, 40, 255),
            pen=pg.mkPen(80, 0, 0, 255, width=2),
        )

        self.lines.setZValue(1)
        self.real_scatter.setZValue(2)
        self.pred_scatter.setZValue(3)
        self.selected_line.setZValue(10)
        self.selected_node.setZValue(11)
        self.predicted_node.setZValue(12)

        for item in [
            self.lines,
            self.selected_line,
            self.selected_node,
            self.predicted_node,
        ]:
            item.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)

        self.plot.addItem(self.lines)
        self.plot.addItem(self.real_scatter)
        self.plot.addItem(self.pred_scatter)
        self.plot.addItem(self.selected_line)
        self.plot.addItem(self.selected_node)
        self.plot.addItem(self.predicted_node)

        self._refresh()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.plot)

    def _line_data(self, idx):
        if idx is None or len(idx) == 0:
            return [], []

        xs = np.empty(len(idx) * 3, dtype=float)
        ys = np.empty(len(idx) * 3, dtype=float)

        xs[0::3] = self.x[idx]
        xs[1::3] = self.pred_x[idx]
        xs[2::3] = np.nan

        ys[0::3] = self.y[idx]
        ys[1::3] = self.pred_y[idx]
        ys[2::3] = np.nan

        return xs, ys

    def _prediction_brushes(self, idx):
        vals = self.error[idx]
        finite = vals[np.isfinite(vals)]

        if len(finite) == 0:
            return [pg.mkBrush(120, 120, 120, 100) for _ in idx]

        lo = float(np.quantile(finite, 0.05))
        hi = float(np.quantile(finite, 0.95))

        if hi <= lo:
            hi = lo + 1.0

        brushes = []

        for v in vals:
            if not np.isfinite(v):
                brushes.append(pg.mkBrush(120, 120, 120, 100))
                continue

            t = (float(v) - lo) / (hi - lo)
            t = max(0.0, min(1.0, t))

            if t < 0.25:
                color = QtGui.QColor(30, 120, 255, 230)
            elif t < 0.50:
                color = QtGui.QColor(40, 190, 80, 230)
            elif t < 0.75:
                color = QtGui.QColor(245, 190, 40, 230)
            else:
                color = QtGui.QColor(230, 60, 40, 230)

            brushes.append(pg.mkBrush(color))

        return brushes

    def _refresh(self):
        idx = np.flatnonzero(self.ok)

        xs, ys = self._line_data(idx)
        self.lines.setData(xs, ys)

        self.real_scatter.setData(
            x=self.x[idx],
            y=self.y[idx],
            data=self.node_ids[idx],
        )

        self.pred_scatter.setData(
            x=self.pred_x[idx],
            y=self.pred_y[idx],
            brush=self._prediction_brushes(idx),
            data=self.node_ids[idx],
        )

    def _clicked(self, item, points):
        if points is None or len(points) == 0:
            return

        data = points[0].data()

        if data is None:
            return

        self.nodeClicked.emit(int(data))

    def show_selection(self, node):
        row = self.node_to_row.get(int(node))

        if row is None or not self.ok[row]:
            self.selected_line.setData([], [])
            self.selected_node.setData([], [])
            self.predicted_node.setData([], [])
            return None

        xs, ys = self._line_data(np.asarray([row], dtype=int))
        self.selected_line.setData(xs, ys)

        self.selected_node.setData(
            x=[self.x[row]],
            y=[self.y[row]],
        )

        self.predicted_node.setData(
            x=[self.pred_x[row]],
            y=[self.pred_y[row]],
        )

        return {
            "measurement": self.measurement,
            "k": int(self.k),
            "error": float(self.error[row]),
            "used_neighbors": int(self.used_neighbors[row]),
            "dx": float(self.pred_x[row] - self.x[row]),
            "dy": float(-(self.pred_y[row] - self.y[row])),
        }

class BenchmarkTable(QtWidgets.QTableWidget):
    def __init__(self, df):
        super().__init__()

        self.df = df.copy()
        self.setSortingEnabled(True)
        self.horizontalHeader().setStretchLastSection(True)
        self.setAlternatingRowColors(True)
        self._fill()

    def _fmt(self, value, digits=3):
        if pd.isna(value):
            return ""
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}g}"
        return str(value)

    def _fill(self):
        if self.df.empty:
            self.setColumnCount(1)
            self.setRowCount(1)
            self.setHorizontalHeaderLabels(["status"])
            self.setItem(0, 0, QtWidgets.QTableWidgetItem("no best_k_by_measurement.csv"))
            return

        cols = [
            "measurement",
            "k",
            "median_xy_error",
            "p90_xy_error",
            "coverage",
            "pred_spread_ratio",
            "median_vs_spatial_best",
            "median_vs_spatial_same_k",
            "spatial_best_k",
            "spatial_best_median_xy_error",
        ]

        cols = [c for c in cols if c in self.df.columns]
        df = self.df[cols].copy()

        if "median_vs_spatial_best" in df.columns:
            df = df.sort_values(["median_vs_spatial_best", "median_xy_error"])

        labels = {
            "measurement": "measurement",
            "k": "k",
            "median_xy_error": "median err",
            "p90_xy_error": "p90 err",
            "coverage": "coverage",
            "pred_spread_ratio": "pred spread",
            "median_vs_spatial_best": "vs spatial best",
            "median_vs_spatial_same_k": "vs spatial same k",
            "spatial_best_k": "spatial best k",
            "spatial_best_median_xy_error": "spatial best err",
        }

        self.setColumnCount(len(cols))
        self.setRowCount(len(df))
        self.setHorizontalHeaderLabels([labels.get(c, c) for c in cols])

        for row_i, r in enumerate(df.itertuples(index=False)):
            vals = list(r)

            for col_i, value in enumerate(vals):
                col = cols[col_i]
                digits = 4 if col == "coverage" else 3
                item = QtWidgets.QTableWidgetItem(self._fmt(value, digits=digits))

                if col in {"k", "spatial_best_k"} and not pd.isna(value):
                    item.setData(QtCore.Qt.ItemDataRole.DisplayRole, int(value))
                elif isinstance(value, (float, np.floating)) and not pd.isna(value):
                    item.setData(QtCore.Qt.ItemDataRole.DisplayRole, float(value))

                if col == "median_vs_spatial_best" and not pd.isna(value):
                    v = float(value)
                    if v < 5:
                        item.setBackground(QtGui.QColor(190, 255, 190))
                    elif v < 15:
                        item.setBackground(QtGui.QColor(255, 245, 180))
                    else:
                        item.setBackground(QtGui.QColor(255, 210, 210))

                self.setItem(row_i, col_i, item)

        self.resizeColumnsToContents()

class NeighborTable(QtWidgets.QTableWidget):
    def __init__(self, measurements):
        super().__init__()
        self.measurements = []
        self.set_measurements(measurements)
        self.horizontalHeader().setStretchLastSection(True)
        self.setSortingEnabled(True)

    def set_measurements(self, measurements):
        self.measurements = list(measurements)
        self.setColumnCount(3 + len(self.measurements))
        self.setHorizontalHeaderLabels(["neighbor", "cell_id", "overlap", *self.measurements])

    def show_node(self, node, nodes, blocks, neighbors_by_measurement):
        union = {}

        for m in self.measurements:
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

        base_nodes = nodes["spatial"] if "spatial" in nodes else next(iter(nodes.values()))

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

            for j, m in enumerate(self.measurements, start=3):
                try:
                    d = pair_distance(blocks, m, node, n)
                except Exception:
                    d = np.nan

                if m in hits:
                    rank, nd, xy = hits[m]
                    if not np.isfinite(d):
                        d = nd
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
    def __init__(self, panes, measurements):
        super().__init__()

        self.panes = panes
        self.measurements = list(measurements)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        title = QtWidgets.QLabel("components")
        layout.addWidget(title)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)

        body = QtWidgets.QWidget()
        body_layout = QtWidgets.QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)

        for m in self.measurements:
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

class CoordinatePanel(QtWidgets.QWidget):
    changed = QtCore.Signal(str, str, float)

    def __init__(self, axis_names, presets):
        super().__init__()

        self.presets = presets
        self.axis_names = list(axis_names)
        self._syncing = False

        self.mode = QtWidgets.QComboBox()
        self.mode.addItems([*presets.keys(), "manual"])

        self.x_axis = QtWidgets.QComboBox()
        self.y_axis = QtWidgets.QComboBox()
        self.x_axis.addItems(self.axis_names)
        self.y_axis.addItems(self.axis_names)

        self.y_scale = QtWidgets.QDoubleSpinBox()
        self.y_scale.setRange(-1000000.0, 1000000.0)
        self.y_scale.setDecimals(6)
        self.y_scale.setSingleStep(0.1)
        self.y_scale.setValue(-1.0)
        self.y_scale.setKeyboardTracking(False)

        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel("coords"), 0, 0)
        layout.addWidget(self.mode, 0, 1)
        layout.addWidget(QtWidgets.QLabel("x"), 1, 0)
        layout.addWidget(self.x_axis, 1, 1)
        layout.addWidget(QtWidgets.QLabel("y"), 2, 0)
        layout.addWidget(self.y_axis, 2, 1)
        layout.addWidget(QtWidgets.QLabel("left coef"), 3, 0)
        layout.addWidget(self.y_scale, 3, 1)

        self.mode.currentTextChanged.connect(self._mode_changed)
        self.x_axis.currentTextChanged.connect(self._manual_changed)
        self.y_axis.currentTextChanged.connect(self._manual_changed)
        self.y_scale.valueChanged.connect(self._manual_changed)

        self._mode_changed(self.mode.currentText())

    def _set_combo(self, box, value):
        idx = box.findText(value)
        if idx >= 0:
            box.setCurrentIndex(idx)

    def _mode_changed(self, mode):
        if self._syncing:
            return

        self._syncing = True

        if mode in self.presets:
            x, y, inv = self.presets[mode]
            self._set_combo(self.x_axis, x)
            self._set_combo(self.y_axis, y)
            self.y_scale.setValue(-1.0 if inv else 1.0)
            self.x_axis.setEnabled(False)
            self.y_axis.setEnabled(False)
            self.y_scale.setEnabled(True)
        else:
            self.x_axis.setEnabled(True)
            self.y_axis.setEnabled(True)
            self.y_scale.setEnabled(True)

        self._syncing = False
        self._emit()

    def _manual_changed(self, *args):
        if self._syncing:
            return

        if self.mode.currentText() != "manual":
            self._syncing = True
            self._set_combo(self.mode, "manual")
            self.x_axis.setEnabled(True)
            self.y_axis.setEnabled(True)
            self.y_scale.setEnabled(True)
            self._syncing = False

        self._emit()

    def _emit(self):
        self.changed.emit(
            self.x_axis.currentText(),
            self.y_axis.currentText(),
            float(self.y_scale.value()),
        )

class MeasurementPanel(QtWidgets.QWidget):
    changed = QtCore.Signal(list)

    def __init__(self, measurements):
        super().__init__()
        self.boxes = {}
        self.body_layout = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel("measurements"))

        buttons = QtWidgets.QHBoxLayout()
        self.all_btn = QtWidgets.QPushButton("all")
        self.core_btn = QtWidgets.QPushButton("core")
        buttons.addWidget(self.all_btn)
        buttons.addWidget(self.core_btn)
        layout.addLayout(buttons)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        self.body_layout = QtWidgets.QVBoxLayout(body)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(body)
        layout.addWidget(scroll)

        for m in measurements:
            self.add_measurement(m, checked=True)

        self.body_layout.addStretch(1)
        self.all_btn.clicked.connect(self._all)
        self.core_btn.clicked.connect(self._core)

    def add_measurement(self, measurement, checked=True):
        if measurement in self.boxes:
            self.boxes[measurement].setChecked(checked)
            return
        cb = QtWidgets.QCheckBox(measurement)
        cb.setChecked(bool(checked))
        cb.toggled.connect(lambda _: self._emit())
        self.boxes[measurement] = cb
        stretch = self.body_layout.takeAt(self.body_layout.count() - 1) if self.body_layout.count() else None
        self.body_layout.addWidget(cb)
        if stretch is not None:
            self.body_layout.addItem(stretch)

    def visible(self):
        return [m for m, cb in self.boxes.items() if cb.isChecked()]

    def _emit(self):
        self.changed.emit(self.visible())

    def _all(self):
        for cb in self.boxes.values():
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.blockSignals(False)
        self._emit()

    def _core(self):
        keep = set(VISIBLE_MEASUREMENTS)
        keep.update(m for m in OPTIONAL_MEASUREMENTS if m in self.boxes)
        keep.update(m for m in self.boxes if m.startswith("learned"))
        for m, cb in self.boxes.items():
            cb.blockSignals(True)
            cb.setChecked(m in keep)
            cb.blockSignals(False)
        self._emit()

