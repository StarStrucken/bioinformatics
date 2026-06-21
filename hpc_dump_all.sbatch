#!/usr/bin/env bash
#SBATCH --job-name=xenum_dump_all
#SBATCH --partition=c23ms
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs

module purge
module load GCCcore/13.2.0
module load Python/3.11.5

cd /work/hi202241/bioinformatics

source /hpcwork/hi202241/bioinformatics/.venv/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

export MALLOC_ARENA_MAX=2

echo "START $(date)"
echo "HOST  $(hostname)"
echo "PWD   $(pwd)"
echo "PY    $(which python)"
python --version
python -m pip check

bash scripts/dump_all.sh

echo "DONE $(date)"
