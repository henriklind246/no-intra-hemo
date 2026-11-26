#!/bin/bash
# Submit `aneug features` (geometry features for every processed case) as a
# Slurm array job followed by an index job. Same arguments as
# submit_preprocess.sh, which does the work with STAGE=features:
#   scripts/msi/submit_features.sh                    # every processed case
#   scripts/msi/submit_features.sh my_cases.txt       # a subset
#   scripts/msi/submit_features.sh "" --overwrite     # recompute existing features
# Needs the raw wall files, so run it before the scratch purge deletes them.
set -euo pipefail
STAGE=features exec "$(dirname "${BASH_SOURCE[0]}")/submit_preprocess.sh" "$@"
