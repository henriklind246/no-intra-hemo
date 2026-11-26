# Settings for running the AneuG-Flow preprocessing on MSI: dataset root,
# conda environment, and Slurm account, partition and resources.
#
# Not run directly: run.sh and submit_preprocess.sh source it, and the Slurm
# jobs pick it up through run.sh. Edit the defaults here, or override any of
# them in your environment before running a script, e.g.
#   NUM_SHARDS=8 scripts/msi/submit_preprocess.sh

# Dataset root holding transient_data/ (raw) and processed_data/ (output).
# NOTE: /scratch.global is purged after 30 days and is not backed up.
: "${ANEUG_ROOT:=/scratch.global/$USER/aneug_flow}"

# Conda environment with python, numpy and torch (and scipy for `aneug features`).
: "${CONDA_ENV:=tps}"
# Module that provides `conda`, if it is not already on PATH (e.g. "conda").
: "${CONDA_MODULE:=}"

# Slurm settings for the batch conversion.
: "${SLURM_ACCOUNT:=rdavis}"
: "${SLURM_PARTITION:=msismall}"
: "${NUM_SHARDS:=64}"        # array tasks; ~10 cases each for 634 cases
: "${MAX_CONCURRENT:=16}"    # array tasks running at once (limits scratch I/O)
: "${TASK_TIME:=04:00:00}"
: "${TASK_MEM:=16G}"         # mmap load needs far less; 16G covers the largest (~5.7 GB) archive even without mmap
: "${TASK_CPUS:=2}"

export ANEUG_ROOT CONDA_ENV CONDA_MODULE SLURM_ACCOUNT SLURM_PARTITION \
       NUM_SHARDS MAX_CONCURRENT TASK_TIME TASK_MEM TASK_CPUS
