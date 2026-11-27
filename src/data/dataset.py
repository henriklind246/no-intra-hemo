"""Datasets for training and evaluating the operator, read from the memory maps.

A whole case is about 640 MB (80 snapshots x ~500k points x 4 values), so
nothing loads cases whole. Two map-style datasets, usable directly with
``torch.utils.data.DataLoader`` (they return dicts of numpy arrays, which the
default collate turns into tensors):

``SampledPoints`` (training). Item ``i`` is case ``i // T`` at snapshot
``i % T``, so an epoch visits every (case, snapshot) pair once, with a fresh
random subset of ``points`` points each time. With ``snapshots > 1`` the item
also holds that many - 1 other random snapshots of the same case at the same
points, which reads the geometry once for several times. Reading one item
costs about one snapshot slab (~8 MB) per snapshot plus the point subset of
the coordinates and geometry features, whatever ``points`` is, so larger
point subsets are nearly free. An item's draw depends only on (seed, epoch,
index), not on the worker that reads it. For step-based training use
``EpochSampler``, whose keys are (epoch, index) pairs: the epoch then travels
with each key, which works with persistent DataLoader workers and resumes at
an exact item. With plain integer indices, ``set_epoch`` sets the epoch (it
only reaches workers that are started afterwards).

``FullField`` (evaluation). Every point of every snapshot, in chunks of
``chunk`` consecutive points (contiguous reads). Chunks differ in length, so
use ``DataLoader(dataset, batch_size=None)``, or put the chunks back together
per (case, snapshot) with the ``case``, ``snapshots`` and ``points`` keys.

Both return, for K points and s snapshots:
    inputs      (K, 8)    x, y, z and the 5 geometry features, normalized
    flow_split  (S,)      the case's flow split in outlet order (type 3, type 4), normalized
    time        (s,)      prediction time as a phase of the cycle in [0, 1)
    targets     (s, K, 4) u, v, w, gauge pressure, normalized
    case        ()        index into ``dataset.case_ids``
    snapshots   (s,)      snapshot indices
    points      (K,)      point indices within the case

``Normalization.physical`` converts predictions back to m/s and Pa.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .normalize import Normalization
from .processed import ProcessedCase


@dataclass(frozen=True)
class _CaseInfo:
    case_id: str
    num_points: int
    num_snapshots: int
    flow_split: np.ndarray  # normalized


class _CaseDataset:
    """Case bookkeeping shared by both datasets: metadata up front, memory maps on demand."""

    def __init__(self, split, part: str, out_root: Path, normalization: Normalization, *,
                 max_open_cases: int = 32, cases: Optional[Sequence[str]] = None):
        self.out_root = Path(out_root)
        self.part = part
        self.normalization = normalization
        self.max_open_cases = max_open_cases
        ids = list(split.ids(part))
        if cases is not None:
            outside = sorted(set(cases) - set(ids))
            if outside:
                raise ValueError(f"not {part} cases of split {split.name!r}: {outside[:5]}")
            ids = [c for c in ids if c in set(cases)]
        self.case_ids: List[str] = ids
        if not self.case_ids:
            raise ValueError(f"split {split.name!r} has no {part} cases")
        self.cases: List[_CaseInfo] = []
        for cid in self.case_ids:
            c = ProcessedCase(self.out_root / cid)
            if not c.has_geometry:
                raise FileNotFoundError(f"{cid}: no geometry features")
            if c.geometry_channels != normalization.geometry_channels:
                raise ValueError(f"{c.case_id}: geometry channels {c.geometry_channels} differ from the "
                                 f"normalization's {normalization.geometry_channels}")
            fs = normalization.case_flow_split(c)  # outlet order
            if fs is None:
                raise ValueError(f"{c.case_id}: no flow split")
            self.cases.append(_CaseInfo(c.case_id, c.num_points, c.num_snapshots, normalization.flow_split(fs)))
        self._open: "OrderedDict[int, ProcessedCase]" = OrderedDict()

    def _case(self, i: int) -> ProcessedCase:
        """Open case ``i``, keeping at most ``max_open_cases`` cases' memory maps open."""
        c = self._open.pop(i, None)
        if c is None:
            c = ProcessedCase(self.out_root / self.case_ids[i])
        self._open[i] = c
        while len(self._open) > self.max_open_cases:
            self._open.popitem(last=False)
        return c

    def __getstate__(self):
        # DataLoader workers start without open memory maps.
        state = self.__dict__.copy()
        state["_open"] = OrderedDict()
        return state

    def _item(self, i: int, snapshots: np.ndarray, points: np.ndarray) -> Dict[str, np.ndarray]:
        c, info, norm = self._case(i), self.cases[i], self.normalization
        x, g = c.coordinates[points], c.geometry_features[points]
        y = np.stack([c.solution[t, points] for t in snapshots])
        if isinstance(points, slice):
            points = np.arange(points.start, points.stop)
        return {
            "inputs": norm.point_inputs(x, g),
            "flow_split": info.flow_split,
            "time": norm.phase(snapshots, info.num_snapshots),
            "targets": norm.targets(y),
            "case": np.int64(i),
            "snapshots": np.asarray(snapshots, dtype=np.int64),
            "points": np.asarray(points, dtype=np.int64),
        }


class SampledPoints(_CaseDataset):
    """Random point subsets of every (case, snapshot) pair; for training."""

    def __init__(self, split, part: str, out_root: Path, normalization: Normalization, *,
                 points: int = 16384, snapshots: int = 1, seed: int = 0, max_open_cases: int = 32,
                 cases: Optional[Sequence[str]] = None):
        super().__init__(split, part, out_root, normalization, max_open_cases=max_open_cases, cases=cases)
        if points < 1 or snapshots < 1:
            raise ValueError("points and snapshots must be at least 1")
        self.points, self.snapshots, self.seed, self.epoch = points, snapshots, seed, 0
        T = {c.num_snapshots for c in self.cases}
        if len(T) != 1:
            raise ValueError(f"cases have different snapshot counts: {sorted(T)}")
        self.num_snapshots = T.pop()
        if snapshots > self.num_snapshots:
            raise ValueError(f"snapshots={snapshots} but cases have {self.num_snapshots}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.cases) * self.num_snapshots

    def __getitem__(self, key) -> Dict[str, np.ndarray]:
        """``key`` is an index, or an (epoch, index) pair as made by ``EpochSampler``."""
        epoch, index = (int(key[0]), int(key[1])) if isinstance(key, (tuple, list)) else (self.epoch, key)
        if not 0 <= index < len(self):
            raise IndexError(index)
        i, t = divmod(index, self.num_snapshots)
        rng = np.random.default_rng([self.seed, epoch, index])
        n = self.cases[i].num_points
        # Sorted indices read the memory maps front to back.
        pts = np.sort(rng.choice(n, self.points, replace=self.points > n))
        ts = np.array([t])
        if self.snapshots > 1:
            others = np.delete(np.arange(self.num_snapshots), t)
            ts = np.concatenate([ts, rng.choice(others, self.snapshots - 1, replace=False)])
        return self._item(i, ts, pts)


class FullField(_CaseDataset):
    """Every point of every snapshot, in contiguous chunks; for evaluation."""

    def __init__(self, split, part: str, out_root: Path, normalization: Normalization, *,
                 chunk: int = 65536, snapshots: Optional[Sequence[int]] = None, max_open_cases: int = 4,
                 cases: Optional[Sequence[str]] = None):
        super().__init__(split, part, out_root, normalization, max_open_cases=max_open_cases, cases=cases)
        if chunk < 1:
            raise ValueError("chunk must be at least 1")
        self.chunk = chunk
        self._index = []  # (case, snapshot, start)
        for i, c in enumerate(self.cases):
            ts = range(c.num_snapshots) if snapshots is None else snapshots
            for t in ts:
                if not 0 <= t < c.num_snapshots:
                    raise ValueError(f"{c.case_id} has no snapshot {t}")
                self._index.extend((i, t, s) for s in range(0, c.num_points, chunk))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        i, t, s = self._index[index]
        return self._item(i, np.array([t]), slice(s, min(s + self.chunk, self.cases[i].num_points)))


class EpochSampler:
    """Endless shuffled (epoch, index) keys for ``SampledPoints``, for step-based training.

    Each epoch is a fresh permutation of the dataset's indices, from
    (seed, epoch). ``start`` skips that many keys, so resuming at step ``n``
    with batch size ``b`` and ``start = n * b`` continues exactly where the
    interrupted run stopped. Pass it as ``DataLoader(..., sampler=...)``.
    """

    def __init__(self, num_items: int, *, seed: int = 0, start: int = 0):
        if num_items < 1:
            raise ValueError("num_items must be at least 1")
        self.num_items, self.seed, self.start = num_items, seed, start

    def __iter__(self):
        epoch, offset = divmod(self.start, self.num_items)
        while True:
            order = np.random.default_rng([self.seed, epoch, 1]).permutation(self.num_items)
            for i in order[offset:]:
                yield (epoch, int(i))
            offset, epoch = 0, epoch + 1
