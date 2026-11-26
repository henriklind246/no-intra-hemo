"""Reading processed cases, and indexing what has been processed.

``ProcessedCase`` opens one processed case: its metadata.json, plus
``coordinates`` (N, 3) and ``solution`` (T, N, 4) as memory-mapped arrays, so
one snapshot or a subset of points is read without loading the trajectory. It
refuses directories that are not complete cases of the current format.

``list_cases`` lists the complete cases in a directory.
``pooled_solution_stats`` combines the per-case statistics saved at
conversion into exact mean/std/min/max over any set of cases (use the
training split for normalization). ``build_index`` (``aneug index``) writes
index.csv and reports failed, incomplete and not-yet-processed cases and
leftover temporary directories.
"""

from __future__ import annotations

import csv
import json
from functools import cached_property
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

from . import layout
from .stats import ChannelStats, pool


class ProcessedCase:
    """Memory-mapped access to one processed case.

    >>> case = ProcessedCase(root / "processed_data" / "stable_0")
    >>> y = case.solution[35]                  # (N, 4), read from disk on demand
    >>> y = case.solution[35, idx]             # subsampled points
    >>> x = case.coordinates[idx]              # (len(idx), 3)
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        if not layout.is_complete_case(self.path):
            raise FileNotFoundError(
                f"{self.path} is not a complete processed case (format v{layout.FORMAT_VERSION})"
            )
        with open(self.path / layout.METADATA_FILE) as f:
            self.metadata = json.load(f)

    @property
    def case_id(self) -> str:
        return self.metadata["case_id"]

    @property
    def num_points(self) -> int:
        return self.metadata["num_points"]

    @property
    def num_snapshots(self) -> int:
        return self.metadata["num_snapshots"]

    @property
    def flow_split(self) -> Optional[List[float]]:
        return self.metadata["flow_split"]

    @property
    def solution_channels(self) -> List[str]:
        return self.metadata["solution_channels"]

    @cached_property
    def coordinates(self) -> np.ndarray:
        """(N, 3) float32, memory-mapped."""
        return np.load(self.path / layout.COORDINATES_FILE, mmap_mode="r")

    @cached_property
    def solution(self) -> np.ndarray:
        """(T, N, 4) float32, memory-mapped."""
        return np.load(self.path / layout.SOLUTION_FILE, mmap_mode="r")

    def snapshot(self, t: int, points: Optional[np.ndarray] = None) -> np.ndarray:
        y = self.solution[t] if points is None else self.solution[t, points]
        return np.asarray(y)

    def solution_stats(self) -> ChannelStats:
        return ChannelStats.from_dict(self.metadata["solution_stats"], self.solution_channels)

    def __repr__(self) -> str:
        return f"ProcessedCase({self.case_id!r}, T={self.num_snapshots}, N={self.num_points})"


def list_cases(out_root: Path) -> List[str]:
    """Ids of complete processed cases, naturally sorted."""
    out_root = Path(out_root)
    if not out_root.is_dir():
        return []
    ids = [
        p.name
        for p in out_root.iterdir()
        if p.is_dir() and not p.name.startswith((".", "_")) and layout.is_complete_case(p)
    ]
    return sorted(ids, key=layout.natural_key)


def pooled_solution_stats(cases: Iterable[ProcessedCase]) -> dict:
    """Exact per-channel mean/std/min/max over all points and snapshots of ``cases``.

    Use with the training split only when deriving normalization constants.
    """
    cases = list(cases)
    if not cases:
        raise ValueError("no cases given")
    channels: Sequence[str] = cases[0].solution_channels
    total = pool((c.solution_stats() for c in cases), len(channels))
    return total.to_dict(channels)


def build_index(out_root: Path, raw_case_ids: Optional[Sequence[str]] = None) -> dict:
    """Write ``index.csv`` for complete cases and report incomplete/failed/missing ones."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    complete = list_cases(out_root)
    rows = []
    warned = []
    for cid in complete:
        md = ProcessedCase(out_root / cid).metadata
        sol = out_root / cid / layout.SOLUTION_FILE
        prov = md.get("provenance", {})
        rows.append(
            {
                "case_id": cid,
                "num_points": md["num_points"],
                "num_snapshots": md["num_snapshots"],
                "flow_split": json.dumps(md.get("flow_split")),
                "solution_bytes": sol.stat().st_size,
                "num_warnings": len(md.get("warnings", [])),
                "created_utc": prov.get("created_utc", ""),
                "git_commit": prov.get("git_commit") or "",
            }
        )
        if md.get("warnings"):
            warned.append(cid)
    index_path = out_root / layout.INDEX_FILE
    columns = ["case_id", "num_points", "num_snapshots", "flow_split", "solution_bytes",
               "num_warnings", "created_utc", "git_commit"]
    with open(index_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)

    failures_dir = out_root / layout.FAILURES_SUBDIR
    failed = sorted(
        (p.stem for p in failures_dir.glob("*.json")) if failures_dir.is_dir() else [],
        key=layout.natural_key,
    )
    leftovers = sorted(p.name for p in out_root.iterdir() if p.name.startswith((".tmp-", ".old-")))
    incomplete = sorted(
        (p.name for p in out_root.iterdir()
         if p.is_dir() and not p.name.startswith((".", "_")) and p.name not in set(complete)),
        key=layout.natural_key,
    )
    summary = {
        "index": str(index_path),
        "num_complete": len(complete),
        "processed_gb": sum(r["solution_bytes"] for r in rows) / 1e9,
        "failed": failed,
        "with_warnings": warned,
        "incomplete_or_foreign_dirs": incomplete,
        "leftover_temp_dirs": leftovers,
    }
    if raw_case_ids is not None:
        done = set(complete)
        summary["num_raw"] = len(raw_case_ids)
        summary["not_yet_processed"] = [c for c in raw_case_ids if c not in done]
    return summary
