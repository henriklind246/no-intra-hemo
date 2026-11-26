# no-intra-hemo

Neural operators for intracranial aneurysm hemodynamics on the
[AneuG-Flow](https://huggingface.co/datasets/whding123/AneuG-Flow) transient dataset.

This repository currently holds the preprocessing infrastructure: it converts
the raw `blood_data.pt` archives on MSI into a compact, memory-mappable layout
that separates static geometry from the time-dependent solution.

## Processed layout

The raw download is never modified. Output goes to a sibling directory:

```
aneug_flow/
├── transient_data/            raw, as downloaded
│   └── stable_0/blood_data.pt, flowsplit_ratio.txt, inlet_centroids.csv, ...
└── processed_data/
    ├── stable_0/
    │   ├── coordinates.npy    (N, 3)     float32   x, y, z
    │   ├── solution.npy       (T, N, 4)  float32   x/y/z velocity, absolute pressure
    │   └── metadata.json
    ├── index.csv              one row per processed case (from `aneug index`)
    ├── raw_manifest.csv       inventory of the raw cases (from `aneug scan`)
    ├── _failures/<case>.json  error + traceback for cases that failed
    └── _jobs/                 frozen case lists of submitted array jobs
```

`metadata.json` records the case id, N and T, the flow split exactly as stored
in `flowsplit_ratio.txt` (file order, not yet mapped to outlets), the raw
`inlet_centroids.csv` text, per-channel and per-snapshot statistics, the
coordinate bounding box, verification results, source file sizes and the git
commit that produced it.

Design choices, from the storage plan:

- **Geometry is static, the solution is dynamic.** Coordinates are stored once.
  Conversion fails if any snapshot's coordinates differ from snapshot 0, since
  that would mean a different point order and misaligned fields.
- **Time-major solution** so `solution[t]` and `solution[t, idx]` read one
  snapshot (or a point subset) from disk without loading the trajectory.
- **float32 at full resolution, no normalization.** Per-case statistics are
  stored so normalization constants for any train split can be pooled exactly
  later (`pooled_solution_stats`) without re-reading the data.
- **Raw geometric inputs only.** Learned geometry memory is computed during
  training. `geometry_features.npy` (wall distance etc.) will be a separate
  stage, so feature ablations never rewrite the solution files.
- **Safe to rerun.** Each case is written to a temporary directory, compared
  value-for-value against the source, and only then renamed into place.
  Completed cases are skipped, so a resubmitted job only does what is missing.

Not decided yet, and therefore not done: snapshot physical times
(`snapshot_times` is `null`), outlet labels for the flow-split values, unit
verification (`coordinate_units` is `"m"` per the dataset description, with
`coordinate_units_verified: false`), geometry features, and spatial sampling.

## Running on MSI

Everything runs from a clone of this repo using the `tps` conda env; no
install is needed. Settings (dataset root, conda env, Slurm account,
partition, shard count) are in [scripts/msi/config.sh](scripts/msi/config.sh)
and can be overridden from the environment.

1. Get the code onto MSI (after pushing this repo):

   ```bash
   git clone https://github.com/henriklind246/no-intra-hemo.git ~/no-intra-hemo && cd ~/no-intra-hemo
   ```

2. Start an interactive CPU session (adjust the partition if needed):

   ```bash
   srun -N 1 -n 1 -c 4 --mem=16G -t 01:00:00 -p interactive --pty bash
   ```

3. Inventory the raw data. This writes `processed_data/raw_manifest.csv`,
   lists cases with missing files, estimates the processed size, and reports
   how close the oldest raw file is to the 30-day scratch purge:

   ```bash
   scripts/msi/run.sh scan
   ```

4. Inspect one case to settle the open questions about the data (field
   shapes and dtypes, whether coordinates are static, flow split and inlet
   file contents, `wall_data.pt` structure, OBJ-vs-volume unit ratio):

   ```bash
   scripts/msi/run.sh inspect stable_0 -o ~/stable_0_inspect.json
   ```

5. Validate the storage design on one case:

   ```bash
   scripts/msi/run.sh convert stable_0
   ```

6. Convert everything as a Slurm array job (default 64 tasks, 16 at a time),
   followed by an index job:

   ```bash
   scripts/msi/submit_preprocess.sh
   ```

   Pass a file of case ids to convert a subset, e.g. a pilot set:
   `scripts/msi/submit_preprocess.sh pilot_cases.txt`. Logs go to `logs/slurm/`.

7. Check progress and failures at any time; rerun step 6 to resume:

   ```bash
   scripts/msi/run.sh index
   ```

   `scripts/msi/run.sh verify --all` re-checks every processed case against the raw files.

**Storage:** `/scratch.global` is purged after 30 days and not backed up. The
processed set is roughly 19% of the raw `blood_data.pt` size (about 440 GB if
all 634 cases are like `stable_0`). Check `groupquota -g rdavis -r`, and set
`ANEUG_ROOT` or `--out-dir` to persistent group storage if the processed data
must outlive the purge.

## Reading processed data

```python
from src.data import ProcessedCase, list_cases, pooled_solution_stats

case = ProcessedCase("/scratch.global/$USER/aneug_flow/processed_data/stable_0")
y = case.solution[35]            # (N, 4), memory-mapped
x = case.coordinates             # (N, 3)
r = case.flow_split              # as stored in flowsplit_ratio.txt

train = [ProcessedCase(root / c) for c in train_ids]
stats = pooled_solution_stats(train)   # exact per-channel mean/std/min/max
```

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

```bash
.venv/bin/python -m pytest
```

The tests build small synthetic cases with the same 21-field `blood_data.pt`
layout as the dataset's `stable_0` case.
