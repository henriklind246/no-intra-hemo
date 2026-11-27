# no-intra-hemo

Neural operators for intracranial aneurysm hemodynamics on the
[AneuG-Flow](https://huggingface.co/datasets/whding123/AneuG-Flow) transient dataset.

This repository holds the preprocessing infrastructure, which converts the
raw `blood_data.pt` archives on MSI into a compact, memory-mappable layout
that separates static geometry from the time-dependent solution and computes
per-point geometry features from the wall mesh, and the training code with a
pointwise MLP baseline.

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
    │   ├── metadata.json
    │   ├── geometry_features.npy  (N, 5)  float32   from `aneug features`, see below
    │   ├── wall_projection.npz    closest wall triangle + barycentric weights per point
    │   ├── wall/                  wall_mesh.npz + copies of the raw wall files
    │   └── geometry.json          feature channels, checks, statistics
    ├── index.csv              one row per processed case (from `aneug index`)
    ├── raw_manifest.csv       inventory of the raw cases (from `aneug scan`)
    ├── _failures/<case>.json  error + traceback for cases that failed
    ├── _failures/features/    the same for `aneug features`
    ├── _jobs/                 frozen case lists of submitted array jobs
    ├── splits/<name>.json     train/val/test case ids + training statistics (`aneug split`)
    ├── splits/<name>.normalization.json   operator normalization constants (`aneug normalization`)
    └── outlets.csv            measured outlet flow fractions vs flowsplit_ratio.txt (`aneug outlets`)
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
(`snapshot_times` is `null`), which outlet each flow-split value belongs to,
and spatial sampling. Coordinates are in metres (`coordinate_units_verified`
is still `false` in metadata.json, but the inspect report confirmed it: the
OBJ surfaces, in mm, are exactly 1000 times larger in the same frame). The
`inlet_centroids_raw` metadata field is only a 4 KB text preview of a file
that actually lists the inlet cap's triangle vertices; use `wall/` instead.

## Geometry features

`aneug features` writes `geometry_features.npy`, shape (N, 5) float32 in
metres, one row per volume point, in this channel order:

| channel | meaning |
| --- | --- |
| `wall_distance` | Euclidean distance to the vessel wall (exact distance to the closest point on the wall triangle mesh) |
| `geodesic_inlet` | geodesic distance through the fluid from the inlet cross-section (the cap of the node-type-2 rim) |
| `geodesic_outlet3` | the same from the outlet cap of node type 3 |
| `geodesic_outlet4` | the same from the outlet cap of node type 4 |
| `geodesic_aneurysm` | the same from the aneurysm sac wall (node type 1) |

Model C (coordinates, time, wall distance, inlet and two outlet distances)
uses the first four; the aneurysm distance is extra.

- **Wall mesh.** `wall_mesh_registration.pt`: 35k nodes and 71k triangles for
  stable_0, in the same frame as the volume points. Node types, as identified
  on stable_0: 0 wall, 1 aneurysm sac, 2 inlet rim, 3 and 4 outlet rims.
- **Volume geodesics.** The volume points have no stored connectivity.
  1. Each point is joined to its 32 nearest neighbours.
  2. Edges that leave the fluid are dropped: an edge goes if it crosses a
     wall triangle or its midpoint lies outside the wall. Paths therefore
     cannot jump between vessels that nearly touch.
  3. Dijkstra runs from each source:
     - inlet and outlets: every point on the cap plane fitted to that open
       rim of the wall mesh, so the whole cross-section starts at about 0;
     - aneurysm: every point whose closest wall is the sac, starting from its
       wall distance.
- **Accuracy and cost.** On a synthetic 885k-point tube with boundary-layer
  point stacks, graph paths came out 0.6% longer than the true distance at
  the median (2.5% at the 95th percentile). The case took about 70 s and
  3 GB.
- **The dataset's own geodesics are only used for checks.** They are surface
  distances divided by a per-case maximum, which loses the length scale.
  Surface geodesics recomputed in metres on the wall mesh must line up with
  them, which confirms the node-type labels.
- **Checks on every case,** recorded in `geometry.json`. A case fails if:
  - the registration and graph files disagree;
  - a node type is missing;
  - the wall and volume frames differ;
  - `inlet_centroids.csv` is not closest to the type-2 nodes.

  A case also fails if no volume points lie on a cap, or if more than 0.1% of
  the points are cut off from a source. Fewer cut-off points take their
  nearest connected neighbour's value. Nearly every case has two or three
  of them; their count is in `checks.geodesic_sources` and they only give a
  warning above 0.001% of the points.

  It gets a warning if:
  - the recomputed surface geodesics don't line up with the dataset's
    normalized ones (correlation below 0.99, or the expected type-2/3/4
    column order doesn't fit);
  - the flow split differs between files;
  - the wall triangles aren't consistently wound, or a cap needed a fallback.

  On the full dataset the only warning left is in 26 cases, where the sac
  geodesic correlates about 0.98 with the dataset's aneurysm distance. Both
  of its fields (`global_geo_dist` and the registration `geo_dist`) give the
  same number, so the dataset probably measures from a slightly different
  set of sac nodes. The labels still match, and the feature is our own.
- **For later.**
  - `geometry.json` records each boundary's size (`checks.boundaries`), for
    matching flow-split values to outlets.
  - `wall_projection.npz` and `ProcessedCase.wall_to_volume` carry any
    per-wall-node field into the volume without the raw files, for example
    the fluid-facing normals stored in `wall/wall_mesh.npz`, Laplacian
    eigenvectors, or wall shear.

## Train/validation/test split

`aneug split` splits by geometry: the unit is the case, so all 80 snapshots of
a geometry land in the same part and no snapshot of a test geometry is ever
seen in training. The default is 80/10/10 with seed 0. For the 634 cases
that is 508/63/63 (`splits/default.json`, committed).

- **Which cases.** Every case with a complete conversion and geometry
  features, minus an optional `--exclude-file`. Cases left out are listed
  with the reason.
- **Reproducible.** The assignment depends only on the set of eligible case
  ids and the seed. The file is never overwritten without `--overwrite`.
- **Related geometries.** If some cases share a parent vessel, pass
  `--groups-file` (CSV with `case_id,group`) and each group stays in one part.
- **Training statistics.** The file also stores exact per-channel statistics
  of the solution and the geometry features over the training cases only,
  ready for the normalization constants.

Keep the split in the repository, since scratch is purged:

```bash
mkdir -p splits && cp /scratch.global/$USER/aneug_flow/processed_data/splits/default.json splits/
```

```python
from src.data import Split

split = Split.load("splits/default.json")
train = split.cases(root, "train")       # list of ProcessedCase
val_ids = split.ids("val")
stats = split.train_stats["solution"]    # training-only mean/std/min/max
```

## Operator inputs, normalization and loading

The operator maps, for one geometry, per-point inputs plus two global inputs
to the flow at that time:

| | inputs | shape |
|---|---|---|
| per point | x, y, z, wall_distance, geodesic_inlet, geodesic_outlet3, geodesic_outlet4, geodesic_aneurysm | (N, 8) |
| per case | outlet flow split | (2,) |
| per query | prediction time, as a phase of the cycle in [0, 1) | scalar |
| **output** | u, v, w, gauge pressure | (N, 4) |

`Normalization` ([src/data/normalize.py](src/data/normalize.py)) takes its
constants from the split's training cases only:

- **Outputs and distances are scaled, never shifted**, so physical zeros stay
  zero: no-slip at the wall, zero distance on the wall or a cap, zero gauge
  pressure at the reference point.
- **Velocity:** one scale for u, v and w (the RMS component), so vectors keep
  their direction.
- **Pressure:** the target is gauge pressure `p - 101325 Pa`, divided by its
  RMS. The solver pins the pressure at the coordinate origin to the operating
  pressure (the point nearest the origin stays within 15 Pa of 101325 in
  every snapshot), so the offset carries no physics. Report errors also with
  each snapshot's mean removed.
- **Coordinates:** divided by the largest absolute training coordinate, which
  keeps the frame and puts the origin (the pressure reference) at zero.
- **Geometry features:** each divided by its own RMS.
- **Flow split:** each value standardized over the training cases.
- **Time:** snapshot `k` of `T` is phase `k / T`, which assumes evenly spaced
  snapshots over one cycle (the dataset has no physical times).
  `periodic_time_features` gives sin/cos harmonics.

**Flow-split order.** The operator needs the split in outlet order (node
type 3, node type 4), matching the outlet geodesic channels.
`flowsplit_ratio.txt` has no documented order. `aneug outlets` measures the
flow leaving each outlet cap; on the full dataset the measured fractions match
the file within 0.02 (median) and inflow matches outflow within 6%.

- The file is in outlet order for most cases.
- 14 cases are confidently reversed: the measured and file fractions lie on
  opposite sides of 0.5, each at least 0.05 away from it.
- 8 cases have an unreliable measurement and keep the file order.

`aneug normalization` reads the report and stores both lists in the
normalization file. `Normalization.case_flow_split` and the datasets then
reverse the swapped cases, so nothing else needs to know about it.

**Inflow is not a separate input.** The inlet flow differs between cases
(15% spread), but the mean inlet velocity is the same, 0.39 m/s (2.5% spread;
flow and inlet area correlate 0.99), and it peaks at snapshot 12 in every
case. So every case has the same inlet velocity waveform, and the inflow
follows from the geometry.

Two map-style datasets ([src/data/dataset.py](src/data/dataset.py)) read only
what they need from the memory maps and work with a PyTorch `DataLoader`:

- `SampledPoints` for training: item `i` is one (case, snapshot) pair with a
  fresh random subset of `points` points; an epoch visits every pair once.
  `snapshots=s` adds s - 1 other snapshots at the same points. Call
  `set_epoch(e)` each epoch for new points.
- `FullField` for evaluation: every point of every snapshot, in contiguous
  chunks; use `batch_size=None`.

```python
from torch.utils.data import DataLoader
from src.data import FullField, Normalization, SampledPoints, Split

root = "/scratch.global/lind1775/aneug_flow/processed_data"
split = Split.load("splits/default.json")
norm = Normalization.load("splits/default.normalization.json")
train = SampledPoints(split, "train", root, norm, points=16384)
loader = DataLoader(train, batch_size=8, shuffle=True, num_workers=4)
for epoch in range(epochs):
    train.set_epoch(epoch)
    for b in loader:
        # b["inputs"] (B, K, 8), b["flow_split"] (B, 2), b["time"] (B, 1), b["targets"] (B, 1, K, 4)
        ...
val = FullField(split, "val", root, norm, chunk=65536)
for c in DataLoader(val, batch_size=None):
    pred_si = norm.physical(model(c).numpy())   # m/s and gauge Pa
```

## Training

`python -m src.train` trains, resumes and evaluates runs
([src/train/](src/train)); the models are in [src/models/](src/models).

**Baseline: pointwise MLP** ([src/models/mlp.py](src/models/mlp.py)). Every
point is predicted on its own from its 8 point inputs, the flow split and
sin/cos harmonics of the phase (4 by default), so all it knows about the
geometry comes from the distance features. Default: width 256, depth 6,
GELU, about 335k parameters.

**Time-conditioned LinearNO** ([src/models/linearno.py](src/models/linearno.py),
[configs/linearno.json](configs/linearno.json)). The chunk-capable LinearNO
from the AhmedML benchmark at its efficient size (8 blocks, width 256,
8 heads, MLP ratio 2, key ratio 1). Every point attends to a softmax-weighted
average over all points of its field, in time linear in the number of points.
Added for AneuG-Flow:

- **Time and flow-split conditioning:** sin/cos harmonics of the phase and
  the flow split give a condition vector that scales, shifts and gates every
  block's LayerNorms and residual branches (adaptive LayerNorm). One
  modulation layer is shared by all blocks, with a small per-block table, to
  keep the model small.
- **Early geometry memory injection (FiLM):** following Xia and
  Aviles-Rivero, "Do Neural Operators Forget Geometry?" (ICML 2026). A
  shared pointwise MLP on the original point inputs gives per-point gamma
  and beta, and after each of the first half of the blocks (0-3 of 8) the
  hidden state becomes `gamma * x + beta`. Injecting only early avoids the
  "geometric shortcut" the paper finds with late injection. The geometry
  encoder does not depend on time. `--injection-blocks n` changes how many
  blocks, `--no-geometry-injection` turns it off (the plain LinearNO, for
  comparison).

Both start as identities (zero-initialized modulation, gamma = 1, beta = 0),
so the untrained model is exactly the original LinearNO. 4.6M parameters
(3.85M of them the AhmedML backbone, 0.6M conditioning, 0.13M geometry
encoder).
Training attends over the sampled points of each item; evaluation attends
over every point of the field, accumulated chunk by chunk, which gives the
same result as one pass over all points.

Every model follows the same interface,
`forward(inputs (B, K, 8), flow_split (B, 2), time (B, s)) -> (B, s, K, 4)`,
so others can be added to `src.models` and trained with the same loop.

**Training.** A step is `batch_size` (case, snapshot) items of `points`
random points each, from `SampledPoints` with an `EpochSampler`: 8 x 32768
for the MLP, 4 x 65536 for LinearNO (more points per item give its attention
a better estimate of the field). The loss is the MSE of the normalized
outputs (`--loss huber` is less sensitive to the few cases with extreme
pressure drops). bfloat16 autocast on GPUs.

| | MLP baseline | LinearNO |
|---|---|---|
| optimizer | AdamW, betas (0.9, 0.999), weight decay 1e-4 | the same |
| schedule | linear warmup (1000 steps) to 1e-3, cosine to 1e-5 | OneCycleLR, max 5e-4, 5% warmup, cosine annealing, no momentum cycling |
| gradient clipping | 1.0 | 1.0 |
| steps | 50000 | 50000 |

OneCycleLR has a fixed length, so a LinearNO run can be resumed but not
extended.

**I/O.** Each item reads a whole snapshot slab (~8 MB), however few points it
keeps, so an epoch (508 cases x 80 snapshots) reads about 325 GB. The log
line reports `data wait`: when it is high, raise `points` (nearly free) or
`num_workers` rather than `batch_size`.

**Validation** runs every `eval_every` steps on full fields of all validation
cases at snapshots 2, 12 (peak inflow), 22, ..., 72. Metrics
([src/train/metrics.py](src/train/metrics.py)), per (case, snapshot):

| metric | |
|---|---|
| `velocity_rel_l2` | relative L2 error of the velocity vectors |
| `pressure_rel_l2` | the same for gauge pressure |
| `pressure_centered_rel_l2` | the same with each field's mean pressure removed |
| `velocity_mae`, `pressure_mae`, `pressure_centered_rmse` | in m/s and Pa |
| `score` | mean of the velocity and centered pressure errors; picks `best.pt` |

All points weigh the same (the mesh is finer near the wall, so this is not a
volume average).

**Run directory** `runs/<name>/`: `config.json`, copies of the split and
normalization, `metrics.jsonl`, `train.log`, `last.pt`, `best.pt`,
`status.json`, and `eval_<part>_<checkpoint>.json/.csv` from `evaluate`.

```bash
python -m src.train train --config configs/mlp_baseline.json            # flags override the file
python -m src.train train --config configs/linearno.json
python -m src.train resume runs/mlp_baseline                            # after a time limit
python -m src.train evaluate runs/mlp_baseline --part test              # all 80 snapshots
```

On SIGTERM (Slurm sends it 5 minutes before the time limit) the trainer
finishes the current step, saves `last.pt` and exits with status 75; `resume`
continues with exactly the batches the run would have seen.

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

8. Compute the geometry features. They need scipy in the conda env
   (`conda install -n tps scipy` if the first command fails) and the raw wall
   files, so run this before the scratch purge removes them:

   ```bash
   conda run -n tps python -c "import scipy; print(scipy.__version__)"
   ```

   ```bash
   scripts/msi/run.sh features stable_0
   ```

   ```bash
   scripts/msi/submit_features.sh
   ```

   The case list defaults to every processed case. `run.sh index` then
   reports `num_with_geometry`, `geometry_failed` and
   `geometry_with_warnings`.

9. Make the split once the features are done:

   ```bash
   scripts/msi/run.sh split
   ```

10. Measure which flow-split value belongs to which outlet (about 1 s per
    case, so use an interactive session), then write the normalization
    constants for the committed split, which read that report:

    ```bash
    scripts/msi/run.sh outlets
    ```

    ```bash
    scripts/msi/run.sh normalization --split splits/default.json
    ```

    Commit `splits/default.normalization.json`.

11. Train on a GPU. Jobs get one H100 on `msigpu` by default
    (`TRAIN_PARTITION`, `TRAIN_GRES` in
    [scripts/msi/config.sh](scripts/msi/config.sh)); check its time limit:

    ```bash
    sinfo -p msigpu -o "%P %G %l"
    ```

    A short LinearNO smoke run first (300 steps, validation on 4 cases). Its
    log gives the throughput, peak GPU memory and an ETA, which scales to the
    full run:

    ```bash
    TRAIN_TIME=00:45:00 scripts/msi/submit_train.sh train --config configs/linearno.json --name linearno_smoke --steps 300 --eval-every 150 --eval-max-cases 4 --log-every 20
    ```

    Then the full runs (each its own job), and after each the test evaluation:

    ```bash
    scripts/msi/submit_train.sh train --config configs/linearno.json
    ```

    ```bash
    scripts/msi/submit_train.sh train --config configs/mlp_baseline.json
    ```

    ```bash
    scripts/msi/submit_train.sh evaluate runs/linearno --part test
    ```

    **Following a run.** `squeue --me` shows the job and its ID. Every
    `log_every` steps `runs/<name>/train.log` gets a line with step, loss,
    learning rate, points/s, data wait, epoch, peak GPU memory and ETA (the
    ETA counts evaluations once the first has run), and every validation a
    line with its scores; `runs/<name>/metrics.jsonl` has the same as JSON.
    `logs/slurm/aneug-<command>-<jobid>.out` has the job's own output and any
    traceback.

    ```bash
    tail -f runs/linearno/train.log
    ```

    ```bash
    srun --jobid <jobid> --overlap nvidia-smi
    ```

    If a job hits its time limit, it saves `last.pt` 5 minutes before; continue
    with `scripts/msi/submit_train.sh resume runs/<name>`.

**Storage:** `/scratch.global` is purged after 30 days and not backed up. The
processed set is roughly 19% of the raw `blood_data.pt` size (about 440 GB if
all 634 cases are like `stable_0`). Check `groupquota -g rdavis -r`, and set
`ANEUG_ROOT` or `--out-dir` to persistent group storage if the processed data
must outlive the purge.

## Reading processed data

```python
from src.data import ProcessedCase, list_cases, pooled_geometry_stats, pooled_solution_stats

case = ProcessedCase("/scratch.global/$USER/aneug_flow/processed_data/stable_0")
y = case.solution[35]            # (N, 4), memory-mapped
x = case.coordinates             # (N, 3)
r = case.flow_split              # as stored in flowsplit_ratio.txt
g = case.geometry_features[:, :4]   # (N, 4) Model C geometry channels, metres

train = [ProcessedCase(root / c) for c in train_ids]
stats = pooled_solution_stats(train)   # exact per-channel mean/std/min/max
gstats = pooled_geometry_stats(train)  # the same for the geometry features
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
