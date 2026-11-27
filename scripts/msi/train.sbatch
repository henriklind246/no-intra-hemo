#!/bin/bash -l
# Slurm job that runs `python -m src.train` (train, resume or evaluate) on a GPU.
#
# Do not sbatch this file directly: scripts/msi/submit_train.sh passes the
# account, partition, GPU, time, memory, CPUs and log path, and sets REPO_DIR.
# All arguments go to `python -m src.train`.
#
# Slurm sends SIGTERM 5 minutes before the time limit (--signal below); it is
# forwarded to the trainer, which finishes its step, saves last.pt and exits
# with status 75. Continue with: scripts/msi/submit_train.sh resume runs/<name>
#SBATCH --job-name=aneug-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:TERM@300
set -euo pipefail

: "${REPO_DIR:?submit with scripts/msi/submit_train.sh}"
source "$REPO_DIR/scripts/msi/config.sh"
if [[ -n "$CONDA_MODULE" ]]; then
    module load "$CONDA_MODULE"
fi
# Activate the env directly (not `conda run`), so the trainer receives SIGTERM.
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_DIR"

echo "host=$(hostname) job=${SLURM_JOB_ID:-?} start=$(date -Is) args=$*"
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python -m src.train "$@" &
pid=$!
trap 'echo "SIGTERM: asking the trainer to save and stop"; kill -TERM "$pid" 2>/dev/null || true' TERM
set +e
wait "$pid"
code=$?
if kill -0 "$pid" 2>/dev/null; then  # wait was cut short by the trap
    wait "$pid"
    code=$?
fi
set -e
echo "end=$(date -Is) exit=$code"
exit "$code"
