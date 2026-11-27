#!/bin/bash
# Submit a training, resume or evaluation job to a GPU node.
# All arguments go to `python -m src.train` (see src/train/cli.py):
#   scripts/msi/submit_train.sh train --config configs/mlp_baseline.json
#   scripts/msi/submit_train.sh resume runs/mlp_baseline
#   scripts/msi/submit_train.sh evaluate runs/mlp_baseline --part test
# Resources come from config.sh (TRAIN_PARTITION, TRAIN_GRES, TRAIN_TIME,
# TRAIN_MEM, TRAIN_CPUS) and can be overridden from the environment, e.g.
#   TRAIN_TIME=02:00:00 scripts/msi/submit_train.sh train ...
# Runs are written to runs/<name>/ in this checkout; logs to logs/slurm/.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_DIR/scripts/msi/config.sh"
if [[ $# -eq 0 ]]; then
    echo "usage: $0 train|resume|evaluate [args...]" >&2
    exit 2
fi
mkdir -p "$REPO_DIR/logs/slurm"
sbatch \
    --account="$SLURM_ACCOUNT" \
    --partition="$TRAIN_PARTITION" \
    --gres="$TRAIN_GRES" \
    --time="$TRAIN_TIME" \
    --mem="$TRAIN_MEM" \
    --cpus-per-task="$TRAIN_CPUS" \
    --job-name="aneug-$1" \
    --output="$REPO_DIR/logs/slurm/%x-%j.out" \
    --export=ALL,REPO_DIR="$REPO_DIR" \
    "$REPO_DIR/scripts/msi/train.sbatch" "$@"
