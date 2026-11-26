#!/bin/bash -l
# Slurm array job that converts raw AneuG-Flow cases into the processed layout.
#
# Each array task runs `aneug convert` (through run.sh) on every NUM_SHARDS-th
# case of CASES_FILE, starting at its task id. Cases that are already complete
# are skipped, so resubmitting after a timeout or failure only redoes what is
# missing; failing cases are recorded in processed_data/_failures/.
#
# Do not sbatch this file directly. scripts/msi/submit_preprocess.sh passes the
# account, partition, array range, time, memory, CPUs and log paths, and sets
# the REPO_DIR, CASES_FILE, NUM_SHARDS and STAGE (convert or features)
# variables this script needs.
#SBATCH --job-name=aneug-preprocess
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail

: "${REPO_DIR:?submit with scripts/msi/submit_preprocess.sh}"
: "${CASES_FILE:?submit with scripts/msi/submit_preprocess.sh}"

echo "host=$(hostname) job=${SLURM_ARRAY_JOB_ID:-?} task=${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS} start=$(date -Is)"
"$REPO_DIR/scripts/msi/run.sh" "${STAGE:-convert}" \
    --cases-file "$CASES_FILE" \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$SLURM_ARRAY_TASK_ID" \
    "$@"
echo "end=$(date -Is)"
