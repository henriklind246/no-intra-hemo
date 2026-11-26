#!/bin/bash
# Run the `aneug` preprocessing CLI (src/data/cli.py) on MSI.
#
# Loads config.sh, puts this checkout's root on PYTHONPATH and runs the CLI
# with `conda run` in $CONDA_ENV, so the package never has to be installed.
# Use it directly in an interactive session; the Slurm jobs call it too.
# All arguments are passed to `aneug`. Examples:
#   scripts/msi/run.sh scan
#   scripts/msi/run.sh inspect stable_0 -o stable_0_inspect.json
#   scripts/msi/run.sh convert stable_0
#   scripts/msi/run.sh index
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$REPO_DIR/scripts/msi/config.sh"

if [[ -n "$CONDA_MODULE" ]]; then
    module load "$CONDA_MODULE"
fi

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec conda run -n "$CONDA_ENV" --no-capture-output python -m src.data "$@"
