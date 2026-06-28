from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from xenum_measurements import LUNA_MEASUREMENTS, MEASUREMENTS
from xenum_paths import ROOT

from .config import (
    CACHE_DIR,
    LUNA_ALLOW_REFLECTION,
    LUNA_BATCH_SIZE,
    LUNA_EPOCHS,
    LUNA_GPUS_PER_NODE,
    LUNA_PYTHON,
    LUNA_SEED,
    LUNA_TIMEOUT_SEC,
)
from .features import log_norm_matrix
from .graph import prediction_from_direct_coordinates

LUNA_SOURCE = ROOT / "external" / "LUNA"
LUNA_EXPRESSION_SELF = "luna_expression_self"
LUNA_EXPRESSION_TRANSFER = "luna_expression_transfer"
LUNA_GENE_COLUMNS_START = 4

class LunaSkip(RuntimeError):
    pass

class LunaFailure(RuntimeError):
    pass

def luna_commit():
    try:
        proc = subprocess.run(
            ["git", "-C", str(LUNA_SOURCE), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None

    return proc.stdout.strip() or None

def luna_env():
    env = os.environ.copy()
    env["WANDB_DISABLED"] = "true"
    env["WANDB_MODE"] = "disabled"
    env["PYTHONHASHSEED"] = str(int(LUNA_SEED))
    env["PYTHONPATH"] = "{}{}{}".format(
        LUNA_SOURCE,
        os.pathsep,
        env.get("PYTHONPATH", ""),
    )
    return env

def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)

def atomic_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)

def hash_strings(values):
    h = hashlib.sha256()

    for value in values:
        h.update(str(value).encode("utf-8"))
        h.update(b"\0")

    return h.hexdigest()

def luna_stage_config(dataset_id, adata, nodes):
    cell_ids = nodes["cell_id"].to_numpy(dtype=str)
    genes = [str(v) for v in adata.var_names]

    config = {
        "version": 1,
        "dataset": dataset_id,
        "measurement": LUNA_EXPRESSION_SELF,
        "luna_commit": luna_commit(),
        "seed": int(LUNA_SEED),
        "epochs": int(LUNA_EPOCHS),
        "batch_size": int(LUNA_BATCH_SIZE),
        "allow_reflection": bool(LUNA_ALLOW_REFLECTION),
        "n_cells": int(len(nodes)),
        "n_genes": int(len(genes)),
        "cell_ids_sha256": hash_strings(cell_ids),
        "gene_names_sha256": hash_strings(genes),
    }
    payload = json.dumps(config, sort_keys=True)
    config["config_hash"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return config

def status_row(dataset_id, nodes, measurement, status, reason, *, seed=None, runtime_sec=None):
    return {
        "dataset": dataset_id,
        "measurement": measurement,
        "label": MEASUREMENTS[measurement]["label"],
        "k": 0,
        "seed": None if seed is None else int(seed),
        "status": status,
        "status_reason": reason,
        "leaky": measurement == LUNA_EXPRESSION_SELF,
        "n_nodes": int(len(nodes)),
        "n_edges": 0,
        "coverage": 0.0,
        "mean_xy_error": None,
        "median_xy_error": None,
        "p90_xy_error": None,
        "runtime_sec": runtime_sec,
    }

def probe_luna_environment(luna_python):
    script = (
        "import json, sys\n"
        "out = {'python': sys.version, 'executable': sys.executable}\n"
        "try:\n"
        "    import torch\n"
        "    out.update({\n"
        "        'torch': getattr(torch, '__version__', None),\n"
        "        'cuda': getattr(torch.version, 'cuda', None),\n"
        "        'cuda_available': bool(torch.cuda.is_available()),\n"
        "    })\n"
        "except Exception as e:\n"
        "    out.update({'torch_error': type(e).__name__ + ': ' + str(e)})\n"
        "print(json.dumps(out))\n"
    )
    proc = subprocess.run(
        [luna_python, "-c", script],
        cwd=LUNA_SOURCE,
        env=luna_env(),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise LunaSkip(f"LUNA environment probe failed: {detail}")

    try:
        info = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as e:
        raise LunaSkip(f"LUNA environment probe returned invalid JSON: {e}") from e

    if info.get("torch_error"):
        raise LunaSkip(info["torch_error"])

    if not info.get("cuda_available"):
        raise LunaSkip("LUNA environment cannot access CUDA")

    return info

def section_values(adata, dataset_id):
    for col in ("cell_section", "section", "fov", "fov_name", "sample"):
        if col in adata.obs:
            return adata.obs[col].astype(str).to_numpy()

    return np.full(adata.n_obs, str(dataset_id), dtype=object)

def class_values(adata):
    for col in ("cell_class", "cell_type", "annotation", "cluster"):
        if col in adata.obs:
            return adata.obs[col].astype(str).to_numpy()

    return np.full(adata.n_obs, "unknown", dtype=object)

def expression_table(adata, nodes, dataset_id, *, zero_coordinates):
    x = log_norm_matrix(adata.X)
    if sp.issparse(x):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    gene_names = [str(v) for v in adata.var_names]
    if len(set(gene_names)) != len(gene_names):
        raise LunaFailure("LUNA input gene names are not unique")

    node_ids = np.arange(len(nodes), dtype=np.int64)
    df = pd.DataFrame(x, index=node_ids, columns=gene_names)
    df.index.name = "node"

    coords = nodes[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
    if zero_coordinates:
        coords = np.zeros_like(coords)

    df.insert(0, "cell_class", class_values(adata))
    df.insert(0, "cell_section", section_values(adata, dataset_id))
    df.insert(0, "coord_Y", coords[:, 1])
    df.insert(0, "coord_X", coords[:, 0])

    return df, gene_names

def write_luna_inputs(work_dir, dataset_id, adata, nodes):
    input_dir = work_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    train_df, gene_names = expression_table(adata, nodes, dataset_id, zero_coordinates=False)
    test_df, test_gene_names = expression_table(adata, nodes, dataset_id, zero_coordinates=True)

    if gene_names != test_gene_names:
        raise LunaFailure("LUNA train/test gene order mismatch")

    gene_columns_end = LUNA_GENE_COLUMNS_START + len(gene_names)
    train_path = input_dir / "train.csv"
    test_path = input_dir / "test.csv"
    map_path = input_dir / "cell_id_map.csv"

    train_df.to_csv(train_path)
    test_df.to_csv(test_path)
    pd.DataFrame({
        "node": np.arange(len(nodes), dtype=np.int64),
        "cell_id": nodes["cell_id"].to_numpy(dtype=str),
    }).to_csv(map_path, index=False)

    meta = {
        "train_data_path": str(train_path),
        "test_data_path": str(test_path),
        "cell_id_map_path": str(map_path),
        "gene_columns_start": int(LUNA_GENE_COLUMNS_START),
        "gene_columns_end": int(gene_columns_end),
        "gene_columns_end_semantics": "exclusive",
        "n_genes": int(len(gene_names)),
        "n_cells": int(len(nodes)),
    }
    atomic_text(input_dir / "input_meta.json", json.dumps(meta, indent=2) + "\n")
    return meta

def run_command(cmd, cwd, env, stdout_path, stderr_path, timeout_sec):
    timeout = int(timeout_sec) if int(timeout_sec) > 0 else None

    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            check=False,
            stdout=stdout,
            stderr=stderr,
            text=True,
            timeout=timeout,
        )

    if proc.returncode != 0:
        tail = ""
        if stderr_path.exists():
            tail = "\n".join(stderr_path.read_text(errors="replace").splitlines()[-20:])
        raise LunaFailure(f"LUNA command failed with exit code {proc.returncode}: {tail}")

def find_metadata_predictions(results_dir):
    paths = sorted(results_dir.rglob("metadata_pred.csv"))
    if not paths:
        raise LunaFailure(f"no LUNA metadata_pred.csv found under {results_dir}")

    frames = []
    for path in paths:
        df = pd.read_csv(path, index_col=0)
        if not {"coord_X", "coord_Y"}.issubset(df.columns):
            raise LunaFailure(f"LUNA prediction file lacks coord_X/coord_Y: {path}")
        df["_luna_prediction_path"] = str(path)
        frames.append(df)

    pred = pd.concat(frames, axis=0, sort=False)
    pred.index = pred.index.astype(str)

    duplicated = pred.index[pred.index.duplicated()].unique().tolist()
    if duplicated:
        raise LunaFailure(f"LUNA returned duplicated cell identifiers: {duplicated[:5]}")

    return pred

def ordered_raw_predictions(pred, nodes):
    expected = [str(i) for i in range(len(nodes))]
    got = set(pred.index.astype(str))
    missing = [v for v in expected if v not in got]

    if missing:
        raise LunaFailure(f"LUNA predictions missing {len(missing)} cell identifiers; first={missing[:5]}")

    extra = sorted(got - set(expected))
    if extra:
        raise LunaFailure(f"LUNA predictions contain unexpected cell identifiers; first={extra[:5]}")

    return pred.loc[expected, ["coord_X", "coord_Y"]].to_numpy(dtype=np.float64)

def align_coordinates(raw_pred, real_xy, *, allow_reflection):
    raw_pred = np.asarray(raw_pred, dtype=np.float64)
    real_xy = np.asarray(real_xy, dtype=np.float64)
    ok = np.isfinite(raw_pred).all(axis=1) & np.isfinite(real_xy).all(axis=1)

    if ok.sum() < 2:
        raise LunaFailure("not enough finite LUNA coordinates for alignment")

    src = raw_pred[ok]
    dst = real_xy[ok]
    src_center = src.mean(axis=0)
    dst_center = dst.mean(axis=0)
    src_centered = src - src_center
    dst_centered = dst - dst_center

    u, _s, vt = np.linalg.svd(src_centered.T @ dst_centered, full_matrices=False)
    rotation = u @ vt
    reflected = bool(np.linalg.det(rotation) < 0)

    if reflected and not allow_reflection:
        u[:, -1] *= -1
        rotation = u @ vt
        reflected = False

    rotated = src_centered @ rotation
    denom = float((rotated * rotated).sum())
    scale = float((rotated * dst_centered).sum() / denom) if denom > 0 else 1.0

    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0

    aligned = (raw_pred - src_center) @ rotation * scale + dst_center
    translation = dst_center - src_center @ rotation * scale

    meta = {
        "alignment_method": "similarity_procrustes",
        "allow_reflection": bool(allow_reflection),
        "reflection_used": reflected,
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "n_aligned": int(ok.sum()),
    }
    return aligned.astype(np.float32), meta

def checkpoint_is_current(persistent_dir, stage_config, checkpoint_name):
    config_path = persistent_dir / "stage_config.json"
    checkpoint_path = persistent_dir / "run" / "checkpoints" / checkpoint_name

    if not config_path.exists() or not checkpoint_path.exists():
        return False

    try:
        old = json.loads(config_path.read_text())
    except Exception:
        return False

    return old.get("config_hash") == stage_config["config_hash"]

def luna_overrides(dataset_id, input_meta, run_dir, results_dir, checkpoint_name, *, mode):
    run_name = f"{dataset_id}_{LUNA_EXPRESSION_SELF}"
    overrides = [
        f"hydra.run.dir={run_dir}",
        f"general.mode={mode}",
        f"general.name={run_name}",
        "general.wandb=disabled",
        f"general.seed={int(LUNA_SEED)}",
        "general.enable_progress_bar=false",
        f"distribute.gpus_per_node={int(LUNA_GPUS_PER_NODE)}",
        f"dataset.dataset_name={run_name}",
        f"dataset.train_data_path={input_meta['train_data_path']}",
        f"dataset.test_data_path={input_meta['test_data_path']}",
        "dataset.validation_data_path=null",
        f"dataset.gene_columns_start={int(input_meta['gene_columns_start'])}",
        f"dataset.gene_columns_end={int(input_meta['gene_columns_end'])}",
        "dataset.maximum_graph_size.train=null",
        "dataset.maximum_graph_size.test=null",
        "dataset.maximum_graph_size.validation=null",
        f"train.n_epochs={int(LUNA_EPOCHS)}",
        f"train.batch_size={int(LUNA_BATCH_SIZE)}",
        f"test.batch_size={int(LUNA_BATCH_SIZE)}",
        f"test.save_dir={results_dir}",
        f"test.checkpoints_name_list=[\"{checkpoint_name}\"]",
        "validation.if_validate=false",
        "validation.save_model_every_n_epochs=1",
        "validation.save_top_k_models=1",
        "validation.check_val_every_n_epochs=1",
    ]

    if mode == "test_only":
        overrides.append(f"test.checkpoints_parent_dir={run_dir / 'checkpoints'}")

    return overrides

def luna_scratch_dir(dataset_id, measurement):
    root = os.environ.get("XENUM_LUNA_WORK_DIR") or os.environ.get("TMPDIR")

    if root:
        return Path(root) / "xenum_luna" / str(dataset_id) / measurement

    return None

def reuse_luna_expression_self(out_dir, dataset_id, nodes, stage_config):
    measurement = LUNA_EXPRESSION_SELF
    pred_path = out_dir / f"predictions_{measurement}_k0.csv"
    runtime_path = out_dir / f"{measurement}_runtime.json"

    if not pred_path.exists() or not runtime_path.exists():
        return None

    try:
        runtime = json.loads(runtime_path.read_text())
    except Exception:
        return None

    if runtime.get("status") != "ok":
        return None

    if runtime.get("config_hash") != stage_config["config_hash"]:
        return None

    try:
        pred_df = pd.read_csv(pred_path)
    except Exception:
        return None

    expected = nodes["cell_id"].to_numpy(dtype=str)
    if "cell_id" not in pred_df.columns or list(pred_df["cell_id"].astype(str)) != list(expected):
        return None

    if not {"pred_x", "pred_y"}.issubset(pred_df.columns):
        return None

    pred = pred_df[["pred_x", "pred_y"]].to_numpy(dtype=np.float32)
    if not np.isfinite(pred).all():
        return None

    _pred_df, row = prediction_from_direct_coordinates(
        dataset_id,
        nodes,
        pred,
        measurement,
        seed=int(LUNA_SEED),
        status="ok",
    )
    row.update({
        "status_reason": None,
        "alignment_method": runtime.get("alignment", {}).get("alignment_method"),
        "alignment_allow_reflection": runtime.get("alignment", {}).get("allow_reflection"),
        "alignment_reflection_used": runtime.get("alignment", {}).get("reflection_used"),
        "alignment_scale": runtime.get("alignment", {}).get("scale"),
        "checkpoint": runtime.get("checkpoint"),
        "checkpoint_epoch": runtime.get("checkpoint_epoch"),
        "runtime_sec": runtime.get("runtime_sec"),
        "config_hash": stage_config["config_hash"],
        "reused": True,
    })
    runtime["reused"] = True
    return row, runtime

def run_luna_expression_self(out_dir, dataset_id, adata, nodes, environment, stage_config):
    started = time.perf_counter()
    measurement = LUNA_EXPRESSION_SELF
    persistent_dir = out_dir / CACHE_DIR / measurement
    scratch_dir = luna_scratch_dir(dataset_id, measurement) or (persistent_dir / "tmp")
    checkpoint_name = f"epoch={int(LUNA_EPOCHS) - 1}.ckpt"
    reuse_checkpoint = checkpoint_is_current(persistent_dir, stage_config, checkpoint_name)

    if persistent_dir.exists() and not reuse_checkpoint:
        shutil.rmtree(persistent_dir)
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)

    run_dir = persistent_dir / "run"
    log_dir = persistent_dir / "logs"
    results_dir = scratch_dir / "results"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    atomic_text(persistent_dir / "stage_config.json", json.dumps(stage_config, indent=2, sort_keys=True) + "\n")

    input_meta = write_luna_inputs(scratch_dir, dataset_id, adata, nodes)
    mode = "test_only" if reuse_checkpoint else "train_and_test"
    cmd = [
        LUNA_PYTHON,
        "main.py",
        *luna_overrides(dataset_id, input_meta, run_dir, results_dir, checkpoint_name, mode=mode),
    ]

    stdout_path = log_dir / "luna_stdout.log"
    stderr_path = log_dir / "luna_stderr.log"
    run_command(cmd, LUNA_SOURCE, luna_env(), stdout_path, stderr_path, LUNA_TIMEOUT_SEC)

    checkpoint_path = run_dir / "checkpoints" / checkpoint_name
    if not checkpoint_path.exists():
        raise LunaFailure(f"selected LUNA checkpoint was not created: {checkpoint_path}")

    pred_meta = find_metadata_predictions(results_dir)
    raw_pred = ordered_raw_predictions(pred_meta, nodes)
    real_xy = nodes[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float64)
    aligned, alignment = align_coordinates(raw_pred, real_xy, allow_reflection=LUNA_ALLOW_REFLECTION)

    pred_df, row = prediction_from_direct_coordinates(
        dataset_id,
        nodes,
        aligned,
        measurement,
        seed=int(LUNA_SEED),
        status="ok",
    )
    pred_df["raw_pred_x"] = raw_pred[:, 0]
    pred_df["raw_pred_y"] = raw_pred[:, 1]
    pred_df["alignment_method"] = alignment["alignment_method"]
    atomic_csv(pred_df, out_dir / f"predictions_{measurement}_k0.csv")

    runtime_sec = time.perf_counter() - started
    row.update({
        "status_reason": None,
        "alignment_method": alignment["alignment_method"],
        "alignment_allow_reflection": alignment["allow_reflection"],
        "alignment_reflection_used": alignment["reflection_used"],
        "alignment_scale": alignment["scale"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(LUNA_EPOCHS) - 1,
        "runtime_sec": runtime_sec,
        "config_hash": stage_config["config_hash"],
        "checkpoint_reused": reuse_checkpoint,
        "reused": False,
    })

    runtime = {
        "dataset": dataset_id,
        "measurement": measurement,
        "status": "ok",
        "source": str(LUNA_SOURCE),
        "luna_commit": luna_commit(),
        "environment": environment,
        "seed": int(LUNA_SEED),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(LUNA_EPOCHS) - 1,
        "input": input_meta,
        "stage_config": stage_config,
        "config_hash": stage_config["config_hash"],
        "checkpoint_reused": reuse_checkpoint,
        "alignment": alignment,
        "command": cmd,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "runtime_sec": runtime_sec,
    }
    atomic_text(out_dir / f"{measurement}_runtime.json", json.dumps(runtime, indent=2) + "\n")
    shutil.rmtree(scratch_dir, ignore_errors=True)

    return row, runtime

def run_luna_expression_transfer(out_dir, dataset_id, nodes):
    reason = "compatible transfer training dataset/checkpoint is not configured"
    row = status_row(
        dataset_id,
        nodes,
        LUNA_EXPRESSION_TRANSFER,
        "skipped",
        reason,
        seed=None,
    )
    return row, {
        "dataset": dataset_id,
        "measurement": LUNA_EXPRESSION_TRANSFER,
        "status": "skipped",
        "reason": reason,
        "fallback_used": False,
    }

def run_luna_measurements(out_dir, dataset_id, adata, nodes):
    started = time.perf_counter()
    rows = []
    stage_config = luna_stage_config(dataset_id, adata, nodes)
    status = {
        "dataset": dataset_id,
        "status": "started",
        "source": str(LUNA_SOURCE),
        "luna_commit": luna_commit(),
        "config_hash": stage_config["config_hash"],
        "measurements": {},
    }

    if not LUNA_SOURCE.exists():
        reason = f"LUNA source tree is missing: {LUNA_SOURCE}"
        for measurement in LUNA_MEASUREMENTS:
            row = status_row(dataset_id, nodes, measurement, "skipped", reason)
            rows.append(row)
            status["measurements"][measurement] = {"status": "skipped", "reason": reason}
        status["status"] = "skipped"
        status["runtime_sec"] = time.perf_counter() - started
        atomic_text(out_dir / "luna_status.json", json.dumps(status, indent=2) + "\n")
        print(f"LUNA skipped: {reason}", flush=True)
        return rows, {}

    reused = reuse_luna_expression_self(out_dir, dataset_id, nodes, stage_config)

    if reused is not None:
        row, meta = reused
        rows.append(row)
        status["measurements"][LUNA_EXPRESSION_SELF] = meta
        print(f"{LUNA_EXPRESSION_SELF} reused", flush=True)
    elif not LUNA_PYTHON:
        reason = "XENUM_LUNA_PYTHON is not configured"
        row = status_row(dataset_id, nodes, LUNA_EXPRESSION_SELF, "skipped", reason, seed=LUNA_SEED)
        rows.append(row)
        status["measurements"][LUNA_EXPRESSION_SELF] = {"status": "skipped", "reason": reason}
        print(f"{LUNA_EXPRESSION_SELF} skipped: {reason}", flush=True)
    else:
        try:
            environment = probe_luna_environment(LUNA_PYTHON)
            row, meta = run_luna_expression_self(out_dir, dataset_id, adata, nodes, environment, stage_config)
            rows.append(row)
            status["measurements"][LUNA_EXPRESSION_SELF] = meta
        except LunaSkip as e:
            reason = str(e)
            row = status_row(dataset_id, nodes, LUNA_EXPRESSION_SELF, "missing_dependency", reason, seed=LUNA_SEED)
            rows.append(row)
            status["measurements"][LUNA_EXPRESSION_SELF] = {"status": "missing_dependency", "reason": reason}
            print(f"{LUNA_EXPRESSION_SELF} skipped: {reason}", flush=True)
        except Exception as e:
            reason = "{}: {}".format(type(e).__name__, e)
            row = status_row(dataset_id, nodes, LUNA_EXPRESSION_SELF, "failed", reason, seed=LUNA_SEED)
            rows.append(row)
            status["measurements"][LUNA_EXPRESSION_SELF] = {"status": "failed", "reason": reason}
            print(f"{LUNA_EXPRESSION_SELF} failed: {reason}", flush=True)

    transfer_row, transfer_meta = run_luna_expression_transfer(out_dir, dataset_id, nodes)
    rows.append(transfer_row)
    status["measurements"][LUNA_EXPRESSION_TRANSFER] = transfer_meta
    print(f"{LUNA_EXPRESSION_TRANSFER} skipped: {transfer_meta['reason']}", flush=True)

    if any(row.get("status") == "ok" for row in rows):
        status["status"] = "ok"
    elif any(row.get("status") == "failed" for row in rows):
        status["status"] = "failed"
    else:
        status["status"] = "skipped"

    status["runtime_sec"] = time.perf_counter() - started
    atomic_text(out_dir / "luna_status.json", json.dumps(status, indent=2) + "\n")

    return rows, {}
