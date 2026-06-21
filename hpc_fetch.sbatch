#!/usr/bin/env bash
#SBATCH --job-name=xenum_fetch
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out

set -e

mkdir -p logs

cd "$SLURM_SUBMIT_DIR"

. .venv/bin/activate
bash scripts/fetch_all.sh
