from __future__ import annotations

import argparse
from pathlib import Path

from PySide6 import QtWidgets
import pyqtgraph as pg

from xenum_paths import existing_out_dir

from .window import MainWindow

def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_id")
    args = p.parse_args()

    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")
    pg.setConfigOptions(antialias=False)

    out_dir = existing_out_dir(args.dataset_id)

    app = QtWidgets.QApplication([])
    app.setStyleSheet((Path(__file__).with_name("styles") / "app.qss").read_text())
    win = MainWindow(out_dir)
    win.show()
    app.exec()

if __name__ == "__main__":
    main()
