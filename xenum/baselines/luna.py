#!/usr/bin/env python3
from __future__ import annotations

# XENUM_LUNA_ADAPTER_V1

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xenum.dump.io import load_xenium
from xenum_paths import ROOT, data_dir, out_dir as make_out_dir


BASELINE_NAME = "luna"
MEASUREMENT = "luna_coordinates"
DEFAULT_LUNA_ROOT = ROOT / "external" / "LUNA"
DEFAULT_CONFIG_ROOT = ROOT / "data" / "luna"
METADATA_COLUMNS = {"coord_X", "coord_Y", "cell_section", "cell_class"}
ALIASES = {
    "x": "coord_X",
    "y": "coord_Y",
    "region": "cell_section",
    "subclass": "cell_class",
}


class LunaSkip(RuntimeError):
    pass


class LunaFailure(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_id")
    return parser.parse_args()


def baseline_dir(dataset_id: str) -> Path:
    path = make_out_dir(dataset_id) / "baselines" / BASELINE_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def resolve_path(value: str | None, base: Path, default: Path) -> Path:
    if value is None:
        return default
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_settings(dataset_id: str) -> tuple[dict[str, Any], Path]:
    config_dir = DEFAULT_CONFIG_ROOT / dataset_id
    config_path = config_dir / "config.json"
    raw: dict[str, Any] = {}
    if config_path.is_file():
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise LunaFailure(f"configuration must be a JSON object: {config_path}")

    luna_root = Path(os.environ.get("XENUM_LUNA_ROOT", raw.get("luna_root", DEFAULT_LUNA_ROOT))).expanduser()
    if not luna_root.is_absolute():
        luna_root = ROOT / luna_root

    run_dir = resolve_path(raw.get("run_dir"), config_dir, config_dir / "run")
    train_csv = resolve_path(raw.get("train_csv"), config_dir, config_dir / "train.csv")

    settings = {
        "config_path": config_path,
        "config_dir": config_dir,
        "luna_root": luna_root.resolve(),
        "luna_python": os.environ.get("XENUM_LUNA_PYTHON", raw.get("luna_python", sys.executable)),
        "run_dir": run_dir,
        "train_csv": train_csv,
        "checkpoint": raw.get("checkpoint"),
        "gene_columns": raw.get("gene_columns"),
        "expression_transform": raw.get("expression_transform", "log2p"),
        "allow_missing_genes": bool(raw.get("allow_missing_genes", False)),
        "cell_class_obs": raw.get("cell_class_obs"),
        "non_leaky": bool(raw.get("non_leaky", False)),
        "max_cells": int(raw.get("max_cells", 50_000)),
    }
    return settings, config_path


def select_checkpoint(settings: dict[str, Any]) -> Path:
    run_dir = Path(settings["run_dir"])
    checkpoint_dir = run_dir / "checkpoints"
    configured = settings.get("checkpoint")

    if configured:
        checkpoint = Path(str(configured)).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = checkpoint_dir / checkpoint
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_file():
            raise LunaSkip(f"configured checkpoint is missing: {checkpoint}")
        return checkpoint

    def epoch(path: Path) -> int:
        match = re.search(r"epoch=(\d+)", path.name)
        return int(match.group(1)) if match else -1

    candidates = sorted(checkpoint_dir.glob("epoch=*.ckpt"), key=lambda path: (epoch(path), path.name))
    if not candidates:
        raise LunaSkip(f"no epoch checkpoint found in {checkpoint_dir}")
    return candidates[-1].resolve()


def standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {name: target for name, target in ALIASES.items() if name in frame.columns and target not in frame.columns}
    return frame.rename(columns=rename)


def infer_gene_columns(frame: pd.DataFrame, configured: Any) -> list[str]:
    if configured is not None:
        if not isinstance(configured, list) or not configured or not all(isinstance(value, str) for value in configured):
            raise LunaFailure("gene_columns must be a non-empty JSON string list")
        genes = [value.strip() for value in configured]
    else:
        genes = [
            str(column)
            for column in frame.columns
            if str(column) not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
        ]

    if not genes:
        raise LunaFailure("no numeric gene columns found in the LUNA training CSV")
    if len(set(genes)) != len(genes):
        raise LunaFailure("duplicate gene columns in the LUNA training CSV")
    missing = [gene for gene in genes if gene not in frame.columns]
    if missing:
        raise LunaFailure(f"configured gene columns missing from training CSV: {missing[:10]}")
    return genes


def prepare_training_csv(settings: dict[str, Any], cache_dir: Path) -> tuple[Path, list[str], set[str]]:
    source = Path(settings["train_csv"])
    if not source.is_file():
        raise LunaSkip(f"training CSV is missing: {source}")

    frame = pd.read_csv(source, index_col=0)
    frame = standardize_columns(frame)
    genes = infer_gene_columns(frame, settings.get("gene_columns"))

    missing_metadata = METADATA_COLUMNS - set(frame.columns)
    if missing_metadata:
        raise LunaFailure(
            "training CSV is missing LUNA metadata columns: " + ", ".join(sorted(missing_metadata))
        )

    normalized = frame[genes + ["coord_X", "coord_Y", "cell_section", "cell_class"]].copy()
    normalized[genes] = normalized[genes].apply(pd.to_numeric, errors="raise")
    normalized.index = np.arange(len(normalized), dtype=np.int64)
    normalized.index.name = "cell_ID"

    target = cache_dir / "train.csv"
    normalized.to_csv(target)
    classes = set(normalized["cell_class"].astype(str))
    return target, genes, classes


def expression_matrix(adata, genes: list[str], allow_missing: bool) -> tuple[np.ndarray, list[str]]:
    names = [str(value) for value in adata.var_names]
    exact = {name: index for index, name in enumerate(names)}
    folded: dict[str, int | None] = {}
    for index, name in enumerate(names):
        key = name.casefold()
        folded[key] = index if key not in folded else None

    indices: list[int | None] = []
    missing: list[str] = []
    for gene in genes:
        index = exact.get(gene)
        if index is None:
            index = folded.get(gene.casefold())
        if index is None:
            missing.append(gene)
        indices.append(index)

    if missing and not allow_missing:
        preview = ", ".join(missing[:15])
        suffix = "" if len(missing) <= 15 else f" ... (+{len(missing) - 15})"
        raise LunaSkip(f"Xenium data is missing {len(missing)} checkpoint genes: {preview}{suffix}")

    present_positions = [position for position, index in enumerate(indices) if index is not None]
    present_indices = [int(index) for index in indices if index is not None]
    matrix = np.zeros((adata.n_obs, len(genes)), dtype=np.float32)
    if present_indices:
        values = adata.X[:, present_indices]
        if hasattr(values, "toarray"):
            values = values.toarray()
        matrix[:, present_positions] = np.asarray(values, dtype=np.float32)
    return matrix, missing


def transform_expression(matrix: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return matrix
    if mode == "log2p":
        if np.nanmin(matrix) < 0:
            raise LunaFailure("log2p expression transform received negative values")
        return np.log2(1.0 + matrix).astype(np.float32)
    if mode == "log1p":
        if np.nanmin(matrix) < 0:
            raise LunaFailure("log1p expression transform received negative values")
        return np.log1p(matrix).astype(np.float32)
    raise LunaFailure(f"unsupported expression_transform: {mode}")


def dummy_coordinates(n: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 3:
        raise LunaSkip("LUNA requires at least three cells")
    values = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return values, np.roll(values, max(1, n // 3))


def choose_cell_classes(adata, settings: dict[str, Any], training_classes: set[str]) -> tuple[np.ndarray, str]:
    requested = settings.get("cell_class_obs")
    candidates = [requested] if requested else ["cell_class", "cell_type", "subclass"]
    for column in candidates:
        if column and column in adata.obs.columns:
            values = adata.obs[column].astype(str).to_numpy()
            return values, str(column)

    fallback = sorted(training_classes)[0] if training_classes else "unknown"
    return np.full(adata.n_obs, fallback, dtype=object), "fallback_training_class"


def prepare_test_csv(
    dataset_id: str,
    settings: dict[str, Any],
    cache_dir: Path,
    genes: list[str],
    training_classes: set[str],
):
    adata = load_xenium(data_dir(dataset_id))
    if adata.n_obs > int(settings["max_cells"]):
        raise LunaSkip(
            f"dataset has {adata.n_obs} cells, above LUNA max_cells={settings['max_cells']}; "
            "raise max_cells in data/luna/<dataset>/config.json after checking GPU memory"
        )

    matrix, missing = expression_matrix(adata, genes, bool(settings["allow_missing_genes"]))
    matrix = transform_expression(matrix, str(settings["expression_transform"]))
    classes, class_source = choose_cell_classes(adata, settings, training_classes)
    coord_x, coord_y = dummy_coordinates(adata.n_obs)

    frame = pd.DataFrame(matrix, columns=genes)
    frame["coord_X"] = coord_x
    frame["coord_Y"] = coord_y
    frame["cell_section"] = dataset_id
    frame["cell_class"] = classes
    frame.index = np.arange(adata.n_obs, dtype=np.int64)
    frame.index.name = "cell_ID"

    target = cache_dir / "test.csv"
    frame.to_csv(target)

    nodes = pd.DataFrame(
        {
            "node": np.arange(adata.n_obs, dtype=np.int64),
            "cell_id": adata.obs["cell_id"].astype(str).to_numpy(),
            "x": adata.obs["x_centroid"].to_numpy(dtype=np.float64),
            "y": adata.obs["y_centroid"].to_numpy(dtype=np.float64),
        }
    )
    return target, nodes, missing, class_source


def run_command(command: list[str], cwd: Path, log_path: Path) -> int:
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("HYDRA_FULL_ERROR", "1")

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return process.wait()


def similarity_align(predicted: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[1] != 2:
        raise LunaFailure(f"coordinate shape mismatch: predicted={predicted.shape}, target={target.shape}")

    pred_mean = predicted.mean(axis=0)
    target_mean = target.mean(axis=0)
    pred_centered = predicted - pred_mean
    target_centered = target - target_mean
    denominator = float(np.sum(pred_centered * pred_centered))
    if denominator <= 0:
        raise LunaFailure("LUNA generated degenerate coordinates")

    u, singular, vt = np.linalg.svd(pred_centered.T @ target_centered)
    rotation = u @ vt
    scale = float(np.sum(singular) / denominator)
    aligned = scale * pred_centered @ rotation + target_mean
    return aligned, {
        "scale": scale,
        "rotation": rotation.tolist(),
        "predicted_center": pred_mean.tolist(),
        "target_center": target_mean.tolist(),
        "reflection_allowed": True,
    }


def read_prediction(raw_dir: Path) -> tuple[Path, pd.DataFrame, int]:
    candidates = sorted(raw_dir.rglob("metadata_pred.csv"))
    if not candidates:
        raise LunaFailure(f"LUNA completed but no metadata_pred.csv was found under {raw_dir}")
    path = candidates[0]
    frame = pd.read_csv(path, index_col=0)
    if not {"coord_X", "coord_Y"}.issubset(frame.columns):
        raise LunaFailure(f"unexpected LUNA prediction columns in {path}")
    try:
        frame.index = pd.to_numeric(frame.index).astype(np.int64)
    except Exception as exc:
        raise LunaFailure(f"LUNA prediction index is not numeric: {path}") from exc
    return path, frame, len(candidates)


def build_benchmark(dataset_id: str, out: Path, nodes: pd.DataFrame, prediction: pd.DataFrame, non_leaky: bool):
    joined = nodes.set_index("node").join(
        prediction[["coord_X", "coord_Y"]].rename(
            columns={"coord_X": "raw_pred_x", "coord_Y": "raw_pred_y"}
        ),
        how="inner",
    )
    joined = joined.replace([np.inf, -np.inf], np.nan).dropna()
    if len(joined) < 3:
        raise LunaFailure("fewer than three finite LUNA predictions matched Xenium cells")

    target = joined[["x", "y"]].to_numpy(dtype=np.float64)
    predicted = joined[["raw_pred_x", "raw_pred_y"]].to_numpy(dtype=np.float64)
    aligned, alignment = similarity_align(predicted, target)
    error = np.sqrt(np.sum((aligned - target) ** 2, axis=1))
    center = target.mean(axis=0)
    center_error = np.sqrt(np.sum((target - center) ** 2, axis=1))

    predictions = pd.DataFrame(
        {
            "node": joined.index.to_numpy(dtype=np.int64),
            "cell_id": joined["cell_id"].astype(str).to_numpy(),
            "x": target[:, 0],
            "y": target[:, 1],
            "pred_x": aligned[:, 0],
            "pred_y": aligned[:, 1],
            "error": error,
            "measurement": MEASUREMENT,
            "k": 0,
        }
    ).sort_values("node")
    predictions.to_csv(out / f"predictions_{MEASUREMENT}_k0.csv", index=False)

    median = float(np.median(error))
    center_median = float(np.median(center_error))
    row = {
        "dataset": dataset_id,
        "baseline": BASELINE_NAME,
        "measurement": MEASUREMENT,
        "k": 0,
        "leaky": not non_leaky,
        "n_nodes": int(len(nodes)),
        "n_edges": 0,
        "coverage": float(len(predictions) / len(nodes)),
        "mean_xy_error": float(np.mean(error)),
        "median_xy_error": median,
        "p90_xy_error": float(np.quantile(error, 0.90)),
        "center_median_error": center_median,
        "median_vs_center": median / center_median if center_median > 0 else None,
    }
    pd.DataFrame([row]).to_csv(out / "bench_xy.csv", index=False)
    return predictions, row, alignment


def render_reconstruction(out: Path, predictions: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(predictions["x"], -predictions["y"], s=4, alpha=0.30, label="real")
    scatter = ax.scatter(
        predictions["pred_x"],
        -predictions["pred_y"],
        c=predictions["error"],
        s=7,
        alpha=0.85,
        label="LUNA",
    )
    ax.set_title("LUNA reconstruction after global similarity alignment")
    ax.set_xlabel("x")
    ax.set_ylabel("y * -1")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    fig.colorbar(scatter, ax=ax, label="prediction error")
    fig.tight_layout()
    path = out / "reconstruction.png"
    fig.savefig(path, dpi=180, metadata={"Software": "xenum"})
    plt.close(fig)
    return path.name


def write_note(out: Path, status: str, reason: str | None = None) -> None:
    lines = [
        "# LUNA Baseline",
        "",
        f"Status: `{status}`.",
        "",
        "LUNA is optional. The main pipeline continues when its repository, checkpoint, or compatible training CSV is absent.",
        "",
        "A successful run writes `bench_xy.csv`, `predictions_luna_coordinates_k0.csv`, and `reconstruction.png`.",
    ]
    if reason:
        lines.extend(["", f"Reason: {reason}"])
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(out: Path, summary: dict[str, Any]) -> None:
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_note(out, str(summary.get("status")), summary.get("reason") or summary.get("error"))


def run(dataset_id: str) -> None:
    started = time.perf_counter()
    out = baseline_dir(dataset_id)
    summary_path = out / "summary.json"
    bench_path = out / "bench_xy.csv"

    if bench_path.is_file() and summary_path.is_file() and os.environ.get("XENUM_LUNA_FORCE", "0") != "1":
        print(f"LUNA baseline is current: {out}", flush=True)
        return

    settings, config_path = load_settings(dataset_id)
    summary: dict[str, Any] = {
        "baseline": BASELINE_NAME,
        "dataset": dataset_id,
        "status": "started",
        "config_path": str(config_path),
        "output_dir": str(out),
    }

    try:
        luna_root = Path(settings["luna_root"])
        if not (luna_root / "main.py").is_file():
            raise LunaSkip(f"LUNA repository is missing: {luna_root}")

        checkpoint = select_checkpoint(settings)
        run_dir = Path(settings["run_dir"])
        hydra_config = run_dir / ".hydra" / "config.yaml"
        if not hydra_config.is_file():
            raise LunaSkip(f"checkpoint run is missing .hydra/config.yaml: {hydra_config}")

        cache_dir = out / "cache"
        raw_dir = out / "raw"
        cache_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)

        train_csv, genes, training_classes = prepare_training_csv(settings, cache_dir)
        test_csv, nodes, missing_genes, class_source = prepare_test_csv(
            dataset_id,
            settings,
            cache_dir,
            genes,
            training_classes,
        )

        checkpoint_dir = checkpoint.parent
        experiment_name = f"xenum_{safe_name(dataset_id)}"
        command = [
            str(settings["luna_python"]),
            "main.py",
            f"general.name={experiment_name}",
            "general.mode=test_only",
            "general.wandb=disabled",
            f"dataset.dataset_name={experiment_name}",
            f"dataset.train_data_path={train_csv}",
            f"dataset.test_data_path={test_csv}",
            "dataset.gene_columns_start=0",
            f"dataset.gene_columns_end={len(genes)}",
            "dataset.validation_data_path=null",
            "dataset.maximum_graph_size.train=null",
            "dataset.maximum_graph_size.test=null",
            "dataset.maximum_graph_size.validation=null",
            f"test.save_dir={raw_dir}",
            f"test.checkpoints_parent_dir={checkpoint_dir}",
            f'test.checkpoints_name_list=["{checkpoint.name}"]',
            "validation.if_validate=false",
        ]

        summary.update(
            {
                "luna_root": str(luna_root),
                "luna_python": str(settings["luna_python"]),
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint),
                "train_csv": str(settings["train_csv"]),
                "prepared_train_csv": str(train_csv),
                "prepared_test_csv": str(test_csv),
                "n_cells": int(len(nodes)),
                "n_genes": int(len(genes)),
                "missing_genes_filled_with_zero": missing_genes,
                "expression_transform": settings["expression_transform"],
                "cell_class_source": class_source,
                "non_leaky_declared": bool(settings["non_leaky"]),
                "command": command,
            }
        )

        returncode = run_command(command, luna_root, out / "luna.log")
        if returncode != 0:
            raise LunaFailure(f"LUNA exited with status {returncode}; see {out / 'luna.log'}")

        prediction_path, prediction, prediction_count = read_prediction(raw_dir)
        predictions, bench, alignment = build_benchmark(
            dataset_id,
            out,
            nodes,
            prediction,
            bool(settings["non_leaky"]),
        )
        reconstruction = render_reconstruction(out, predictions)

        summary.update(
            {
                "status": "ok",
                "prediction_source": str(prediction_path),
                "prediction_files_found": prediction_count,
                "selected_prediction_policy": "first_sorted_metadata_pred",
                "alignment": alignment,
                "benchmark": bench,
                "outputs": {
                    "bench_xy": "bench_xy.csv",
                    "predictions": f"predictions_{MEASUREMENT}_k0.csv",
                    "reconstruction": reconstruction,
                    "log": "luna.log",
                },
                "runtime_sec": time.perf_counter() - started,
            }
        )
        write_summary(out, summary)
        print(f"LUNA baseline saved: {out}", flush=True)

    except LunaSkip as exc:
        summary.update(
            {
                "status": "skipped",
                "reason": str(exc),
                "runtime_sec": time.perf_counter() - started,
            }
        )
        write_summary(out, summary)
        print(f"LUNA skipped for {dataset_id}: {exc}", flush=True)
        if os.environ.get("XENUM_LUNA_REQUIRED", "0") == "1":
            raise SystemExit(1)

    except Exception as exc:
        summary.update(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "runtime_sec": time.perf_counter() - started,
            }
        )
        write_summary(out, summary)
        print(f"LUNA failed for {dataset_id}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        if os.environ.get("XENUM_LUNA_REQUIRED", "0") == "1":
            raise


def main():
    args = parse_args()
    run(args.dataset_id)


if __name__ == "__main__":
    main()
