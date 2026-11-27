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

# Slurm settings for training (submit_train.sh): one H100 on msigpu, as in the
# AhmedML benchmark jobs. Check the partition's time limit and GPUs with
#   sinfo -p msigpu -o "%P %G %l"
# (an A100 instead: TRAIN_PARTITION=a100-4 TRAIN_GRES=gpu:a100:1).
: "${TRAIN_PARTITION:=msigpu}"
: "${TRAIN_GRES:=gpu:h100:1}"
: "${TRAIN_TIME:=24:00:00}"
: "${TRAIN_MEM:=64G}"
: "${TRAIN_CPUS:=12}"        # DataLoader workers (num_workers, default 8) + main process

export ANEUG_ROOT CONDA_ENV CONDA_MODULE SLURM_ACCOUNT SLURM_PARTITION \
       NUM_SHARDS MAX_CONCURRENT TASK_TIME TASK_MEM TASK_CPUS \
       TRAIN_PARTITION TRAIN_GRES TRAIN_TIME TRAIN_MEM TRAIN_CPUS
