#!/bin/bash
# Submit the batch conversion of the raw AneuG-Flow cases to Slurm. Run it on
# an MSI login node; it does three things:
#   1. Freezes the case list in processed_data/_jobs/cases_<time>.csv, by
#      scanning the raw data or copying the file you pass, so that shard
#      assignment cannot shift while the job runs.
#   2. Submits preprocess_array.sbatch as an array of min(NUM_SHARDS, #cases)
#      tasks, at most MAX_CONCURRENT at a time.
#   3. Submits an `aneug index` job that runs once the array has finished.
# Settings come from config.sh; logs go to logs/slurm/ in this checkout.
#
#   scripts/msi/submit_preprocess.sh                  # every raw case
#   scripts/msi/submit_preprocess.sh my_cases.txt     # a subset (one id per line, or a manifest CSV)
#   scripts/msi/submit_preprocess.sh "" --overwrite   # extra args go to `aneug convert`
#
# STAGE=features runs `aneug features` instead (scripts/msi/submit_features.sh
# does this); without a case file it then takes every processed case.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_DIR/scripts/msi/config.sh"
STAGE="${STAGE:-convert}"
case "$STAGE" in
    convert|features) ;;
    *) echo "STAGE must be convert or features, not '$STAGE'" >&2; exit 1 ;;
esac

CASES_ARG="${1:-}"
shift || true

OUT_DIR="$ANEUG_ROOT/processed_data"
JOBS_DIR="$OUT_DIR/_jobs"
LOG_DIR="$REPO_DIR/logs/slurm"
mkdir -p "$JOBS_DIR" "$LOG_DIR"
STAMP="$(date +%Y%m%dT%H%M%S)"
CASES_FILE="$JOBS_DIR/${STAGE}_cases_$STAMP.csv"

if [[ -n "$CASES_ARG" ]]; then
    cp "$CASES_ARG" "$CASES_FILE"
elif [[ "$STAGE" == features ]]; then
    # index.csv lists the complete processed cases.
    "$REPO_DIR/scripts/msi/run.sh" index > /dev/null || true
    cp "$OUT_DIR/index.csv" "$CASES_FILE"
else
    "$REPO_DIR/scripts/msi/run.sh" scan --manifest "$CASES_FILE"
fi

# Count case ids (skip a CSV header, blank lines and comments).
NUM_CASES="$(grep -v -e '^case_id' -e '^[[:space:]]*$' -e '^[[:space:]]*#' "$CASES_FILE" | wc -l | tr -d ' ')"
if [[ "$NUM_CASES" -eq 0 ]]; then
    echo "no cases in $CASES_FILE" >&2
    exit 1
fi
SHARDS=$(( NUM_SHARDS < NUM_CASES ? NUM_SHARDS : NUM_CASES ))

echo "stage: $STAGE  cases: $NUM_CASES  shards: $SHARDS  concurrent: $MAX_CONCURRENT  list: $CASES_FILE"

ARRAY_JOB="$(sbatch --parsable \
    --account="$SLURM_ACCOUNT" \
    --partition="$SLURM_PARTITION" \
    --array="0-$((SHARDS - 1))%$MAX_CONCURRENT" \
    --time="$TASK_TIME" \
    --mem="$TASK_MEM" \
    --cpus-per-task="$TASK_CPUS" \
    --output="$LOG_DIR/%x_%A_%a.out" \
    --job-name="aneug-$STAGE" \
    --export="ALL,REPO_DIR=$REPO_DIR,CASES_FILE=$CASES_FILE,NUM_SHARDS=$SHARDS,STAGE=$STAGE" \
    "$REPO_DIR/scripts/msi/preprocess_array.sbatch" "$@")"
echo "submitted array job $ARRAY_JOB"

INDEX_JOB="$(sbatch --parsable \
    --account="$SLURM_ACCOUNT" \
    --partition="$SLURM_PARTITION" \
    --job-name=aneug-index \
    --dependency="afterany:$ARRAY_JOB" \
    --time=00:30:00 --mem=4G --cpus-per-task=1 \
    --output="$LOG_DIR/%x_%j.out" \
    --export="ALL,REPO_DIR=$REPO_DIR" \
    --wrap="$REPO_DIR/scripts/msi/run.sh index")"
echo "submitted index job $INDEX_JOB (runs after $ARRAY_JOB finishes)"
echo "monitor: squeue -u $USER    logs: $LOG_DIR"
