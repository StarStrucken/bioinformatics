#!/usr/bin/env bash
#SBATCH --job-name=xenum_dump
#SBATCH --account=lect0184
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --chdir=/work/hi202241/bioinformatics
#SBATCH --output=/work/hi202241/bioinformatics/logs/%x-%j.out
#SBATCH --error=/work/hi202241/bioinformatics/logs/%x-%j.err

set -euo pipefail

module purge
module load GCCcore/13.2.0 Python/3.11.5

readonly HPC_PROJECT_DIR="/work/hi202241/bioinformatics"
readonly HPC_ENV_ROOT="/hpcwork/hi202241/bioinformatics"
readonly HPC_MAIN_PYTHON="$HPC_ENV_ROOT/.venv/bin/python"
readonly HPC_LUNA_PYTHON="$HPC_ENV_ROOT/.venv-luna/bin/python"
readonly HPC_DATA_DIR="$HPC_PROJECT_DIR/data"
readonly HPC_OUTPUT_DIR="$HPC_PROJECT_DIR/outputs"
readonly HPC_SCRATCH_ROOT="$HPC_ENV_ROOT/tmp"
readonly HPC_SCRATCH_DIR="$HPC_SCRATCH_ROOT/${SLURM_JOB_ID:-manual}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -d "$HPC_PROJECT_DIR" ]] || fail "project directory is missing: $HPC_PROJECT_DIR"
[[ -d "$HPC_DATA_DIR" ]] || fail "dataset directory is missing: $HPC_DATA_DIR"
[[ -f "$HPC_PROJECT_DIR/datasets.tsv" ]] || fail "datasets.tsv is missing"
[[ -x "$HPC_MAIN_PYTHON" ]] || fail "main environment is missing: $HPC_MAIN_PYTHON"
[[ -x "$HPC_LUNA_PYTHON" ]] || fail "LUNA environment is missing: $HPC_LUNA_PYTHON"
[[ -d "$HPC_PROJECT_DIR/external/LUNA" ]] || fail \
  "LUNA checkout is missing: initialize/copy external/LUNA before submitting"

mkdir -p "$HPC_PROJECT_DIR/logs" "$HPC_OUTPUT_DIR" "$HPC_SCRATCH_DIR"
cd "$HPC_PROJECT_DIR"

export XENUM_MAIN_PYTHON="$HPC_MAIN_PYTHON"
export XENUM_LUNA_PYTHON="$HPC_LUNA_PYTHON"
export XENUM_DATA_DIR="$HPC_DATA_DIR"
export XENUM_OUTPUT_DIR="$HPC_OUTPUT_DIR"
export TMPDIR="$HPC_SCRATCH_DIR"

export XENUM_LUNA_GPUS_PER_NODE=1
export XENUM_LUNA_BATCH_SIZE=1
export XENUM_LUNA_SEED=0
export XENUM_LUNA_TIMEOUT_SEC=0
export XENUM_LUNA_ALLOW_REFLECTION=1
export XENUM_LUNA_EPOCHS="${XENUM_LUNA_EPOCHS:-1}"

export WANDB_DISABLED=true
export WANDB_MODE=disabled
export MPLBACKEND=Agg

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MALLOC_ARENA_MAX=2

echo "START   $(date --iso-8601=seconds)"
echo "JOB_ID  ${SLURM_JOB_ID:-manual}"
echo "ACCOUNT ${SLURM_JOB_ACCOUNT:-unknown}"
echo "HOST    $(hostname)"
echo "PWD     $(pwd)"
echo "MAIN    $XENUM_MAIN_PYTHON"
echo "LUNA    $XENUM_LUNA_PYTHON"
echo "DATA    $XENUM_DATA_DIR"
echo "OUT     $XENUM_OUTPUT_DIR"
echo "TMP     $TMPDIR"

"$XENUM_MAIN_PYTHON" --version

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  fail "nvidia-smi is not available in the allocated GPU job"
fi

"$XENUM_LUNA_PYTHON" - <<'PY'
import sys
import torch

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise SystemExit("LUNA environment cannot access CUDA")
PY

bash scripts/dump_all.sh

echo "DONE    $(date --iso-8601=seconds)"
