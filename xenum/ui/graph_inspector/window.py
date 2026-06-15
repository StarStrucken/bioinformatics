from __future__ import annotations

from pathlib import Path

from PySide6 import QtWidgets

from xenum_axes import build_axis_space
from xenum_distances import distance_blocks

from .data import build_neighbors, load_best_k_table, load_best_prediction_k, load_dump, load_predictions, prediction_k_text
from .widgets import BenchmarkTable, ComponentFilterPanel, CoordinatePanel, GraphPane, MeasurementPanel, NeighborTable, PredictionPane, make_side_label

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, out_dir: Path):
        super().__init__()

        self.out_dir = out_dir
        self.measurements, self.nodes, self.edges = load_dump(out_dir)
        self.best_prediction_k = load_best_prediction_k(out_dir)
        self.predictions = load_predictions(out_dir, self.measurements, self.best_prediction_k)
        self.best_k_table = load_best_k_table(out_dir)
        self.base_measurement = "spatial" if "spatial" in self.nodes else self.measurements[0]
        self.neighbors = build_neighbors(self.edges)
        self.blocks = distance_blocks(self.nodes[self.base_measurement])
        self.axis_values, self.presets = build_axis_space(self.nodes[self.base_measurement], out_dir)
        self.selected_node_id = None
        self.visible_measurements = list(self.measurements)

        self.setWindowTitle(f"Xenium graph inspector: {out_dir}")

        root = QtWidgets.QWidget()
        self.setCentralWidget(root)

        self.panes = {}
        self.grid = QtWidgets.QGridLayout()
        self.grid.setSpacing(2)

        for m in self.measurements:
            self._create_pane(m)

        self.prediction_panes = {}
        self.prediction_grid = QtWidgets.QGridLayout()
        self.prediction_grid.setSpacing(2)

        for m in self.measurements:
            if m in self.predictions:
                self._create_prediction_pane(m)

        self.coords = CoordinatePanel(self.axis_values.keys(), self.presets)
        self.coords.changed.connect(self.set_axes)

        self.measure_panel = MeasurementPanel(self.measurements)
        self.measure_panel.changed.connect(self.set_visible_measurements)

        self.info = QtWidgets.QLabel("click a node")
        self.table = NeighborTable(self.visible_measurements)
        self.filters = ComponentFilterPanel(self.panes, self.measurements)

        side = QtWidgets.QVBoxLayout()
        side.addWidget(self.coords)
        side.addWidget(self.measure_panel, stretch=1)
        side.addWidget(self.info)
        side.addWidget(self.table, stretch=3)
        side.addWidget(self.filters, stretch=2)

        self.tabs = QtWidgets.QTabWidget()

        graph_tab = QtWidgets.QWidget()
        graph_layout = QtWidgets.QHBoxLayout(graph_tab)
        graph_layout.addLayout(self.grid, stretch=4)
        graph_layout.addLayout(side, stretch=2)

        prediction_tab = QtWidgets.QWidget()
        prediction_side = QtWidgets.QVBoxLayout()
        prediction_side.setContentsMargins(6, 6, 6, 6)
        prediction_side.setSpacing(6)

        self.prediction_info = make_side_label("click a node")

        self.prediction_k_box = QtWidgets.QPlainTextEdit()
        self.prediction_k_box.setReadOnly(True)
        self.prediction_k_box.setPlainText(
            prediction_k_text(self.measurements, self.best_prediction_k, self.predictions)
        )
        self.prediction_k_box.setMaximumHeight(120)
        self.prediction_k_box.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)

        prediction_side.addWidget(make_side_label("prediction: best k per measurement"))
        prediction_side.addWidget(self.prediction_k_box)
        prediction_side.addWidget(make_side_label("color: displacement error"))
        prediction_side.addWidget(make_side_label("blue/green small\nyellow/red large"))
        prediction_side.addWidget(self.prediction_info)
        prediction_side.addStretch(1)

        prediction_layout = QtWidgets.QHBoxLayout(prediction_tab)
        prediction_layout.addLayout(self.prediction_grid, stretch=4)
        prediction_layout.addLayout(prediction_side, stretch=1)

        benchmark_tab = QtWidgets.QWidget()
        benchmark_layout = QtWidgets.QVBoxLayout(benchmark_tab)
        benchmark_layout.setContentsMargins(6, 6, 6, 6)

        self.benchmark_table = BenchmarkTable(self.best_k_table)

        benchmark_title = QtWidgets.QLabel("best k by measurement")
        benchmark_title.setWordWrap(True)

        benchmark_note = QtWidgets.QLabel("lower vs spatial is better")
        benchmark_note.setWordWrap(True)

        benchmark_layout.addWidget(benchmark_title)
        benchmark_layout.addWidget(benchmark_note)
        benchmark_layout.addWidget(self.benchmark_table)

        self.tabs.addTab(graph_tab, "Graphs")
        self.tabs.addTab(prediction_tab, "Prediction")
        self.tabs.addTab(benchmark_tab, "Benchmark")

        layout = QtWidgets.QVBoxLayout(root)
        layout.addWidget(self.tabs)

        self._link_panes()
        self._link_prediction_panes()
        self._rebuild_grid()
        self.resize(1750, 980)

    def _create_pane(self, measurement):
        pane = GraphPane(measurement, self.nodes[measurement], self.edges[measurement], self.axis_values)
        pane.nodeClicked.connect(self.select_node)
        self.panes[measurement] = pane
        return pane

    def _create_prediction_pane(self, measurement):
        pane = PredictionPane(measurement, self.predictions[measurement])
        pane.nodeClicked.connect(self.select_node)
        self.prediction_panes[measurement] = pane
        return pane

    def _link_panes(self):
        master = self.panes[self.base_measurement].plot
        for m, pane in self.panes.items():
            if m != self.base_measurement:
                pane.plot.setXLink(master)
                pane.plot.setYLink(master)

    def _link_prediction_panes(self):
        if not self.prediction_panes:
            return

        first = next(iter(self.prediction_panes.values())).plot

        for pane in self.prediction_panes.values():
            if pane.plot is not first:
                pane.plot.setXLink(first)
                pane.plot.setYLink(first)

    def _rebuild_grid(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

        visible = [m for m in self.visible_measurements if m in self.panes]
        for i, m in enumerate(visible):
            pane = self.panes[m]
            pane.setVisible(True)
            self.grid.addWidget(pane, i // 2, i % 2)

        for m, pane in self.panes.items():
            if m not in visible:
                pane.setVisible(False)

        while self.prediction_grid.count():
            item = self.prediction_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

        pred_visible = [m for m in visible if m in self.prediction_panes]

        for i, m in enumerate(pred_visible):
            pane = self.prediction_panes[m]
            pane.setVisible(True)
            self.prediction_grid.addWidget(pane, i // 2, i % 2)

        for m, pane in self.prediction_panes.items():
            if m not in pred_visible:
                pane.setVisible(False)

        self.table.set_measurements(visible)
        if self.selected_node_id is not None:
            self.select_node(self.selected_node_id)

    def set_visible_measurements(self, measurements):
        measurements = [m for m in measurements if m in self.panes]
        if not measurements:
            measurements = [self.base_measurement]
        self.visible_measurements = measurements
        self._rebuild_grid()

    def set_axes(self, x_axis, y_axis, y_scale):
        if not x_axis or not y_axis:
            return

        for pane in self.panes.values():
            pane.set_axes(x_axis, y_axis, y_scale)

        if self.base_measurement in self.panes:
            self.panes[self.base_measurement].plot.enableAutoRange()

        if self.selected_node_id is not None:
            self.select_node(self.selected_node_id)

    def select_node(self, node):
        self.selected_node_id = int(node)
        cell_id = self.nodes[self.base_measurement].iloc[node].get("cell_id", node)

        self.info.setText(f"node={node} cell_id={cell_id}")

        by_m = self.neighbors.get(node, {})

        for m, pane in self.panes.items():
            pane.show_selection(node, by_m.get(m, []))

        self.table.show_node(node, self.nodes, self.blocks, by_m)

        pred_rows = []

        for m, pane in self.prediction_panes.items():
            info = pane.show_selection(node)
            if info is not None and m in self.visible_measurements:
                pred_rows.append(info)

        if pred_rows:
            best = sorted(pred_rows, key=lambda x: x["error"])[0]
            self.prediction_info.setText(
                f"node {node}\n"
                f"best {best['measurement']}\n"
                f"k {best['k']}\n"
                f"error {best['error']:.3g}\n"
                f"neighbors {best['used_neighbors']}\n"
                f"dx {best['dx']:.3g}\n"
                f"dy {best['dy']:.3g}"
            )
        else:
            self.prediction_info.setText(f"node {node}\nno prediction")
