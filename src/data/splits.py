"""Train/validation/test splits by geometry (``aneug split``).

Every case is one patient-specific geometry with 80 snapshots of one cardiac
cycle. Snapshots of the same geometry are strongly correlated, so a split is a
partition of *case ids*: all snapshots of a case go to the same part, and
nothing is split by snapshot. ``Split`` checks this when a split file is
loaded (the parts must be disjoint), and the training code should only ever
reach data through ``Split.cases``.

A split file, ``<processed_data>/splits/<name>.json``, holds:
    name, seed, fractions    how it was made
    parts                    {"train": [...], "val": [...], "test": [...]}
    groups                   case -> group, if a groups file was given
    excluded                 cases left out, with the reason
    train_stats              exact per-channel statistics of the solution
                             and the geometry features over the training
                             cases only (pooled from the per-case statistics),
                             for deriving normalization constants later
    eligible_sha256          hash of the sorted eligible case ids
    provenance

How cases are assigned. The eligible cases (complete conversion and, unless
``require_geometry=False``, complete geometry features, minus an exclusion
list) are sorted, shuffled with ``seed`` and cut into parts of
``round(fraction * count)`` cases, with train taking the rest. The result
depends only on the eligible set and the seed, not on directory listing
order. With ``groups`` (e.g. several cases derived from one parent vessel),
whole groups are shuffled and assigned instead, so related geometries never
straddle two parts; the part sizes are then as close to the targets as whole
groups allow.

Split files are never overwritten unless asked: a fixed split is what makes
results comparable, and it should be copied into the repository (the scratch
filesystem is purged).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from . import layout
from .processed import ProcessedCase, list_cases, pooled_geometry_stats, pooled_solution_stats
from .provenance import provenance

PARTS = ("train", "val", "test")
SPLITS_SUBDIR = "splits"
SPLIT_VERSION = 1


def assign_parts(
    case_ids: Sequence[str],
    *,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 0,
    groups: Optional[Mapping[str, str]] = None,
) -> Dict[str, List[str]]:
    """Partition ``case_ids`` into train/val/test, keeping each group in one part."""
    if not (0 <= val < 1 and 0 <= test < 1 and val + test < 1):
        raise ValueError(f"fractions must satisfy 0 <= val, test and val + test < 1 (got {val}, {test})")
    ids = sorted(set(case_ids), key=layout.natural_key)
    if len(ids) != len(case_ids):
        raise ValueError("duplicate case ids")
    n = len(ids)
    targets = {"val": round(val * n), "test": round(test * n)}
    rng = np.random.default_rng(seed)

    if groups is None:
        order = [ids[i] for i in rng.permutation(n)]
        parts = {
            "test": order[: targets["test"]],
            "val": order[targets["test"] : targets["test"] + targets["val"]],
            "train": order[targets["test"] + targets["val"] :],
        }
    else:
        missing = [c for c in ids if c not in groups]
        if missing:
            raise ValueError(f"{len(missing)} case(s) have no group, e.g. {missing[:3]}")
        members: Dict[str, List[str]] = {}
        for c in ids:
            members.setdefault(str(groups[c]), []).append(c)
        names = sorted(members, key=layout.natural_key)
        parts = {p: [] for p in PARTS}
        for g in (names[i] for i in rng.permutation(len(names))):
            # Fill test, then val, as long as the group fits better than not.
            for p in ("test", "val"):
                have, want = len(parts[p]), targets[p]
                if have < want and abs(have + len(members[g]) - want) <= abs(have - want):
                    parts[p].extend(members[g])
                    break
            else:
                parts["train"].extend(members[g])
    return {p: sorted(parts[p], key=layout.natural_key) for p in PARTS}


def _sha(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids, key=layout.natural_key)).encode()).hexdigest()


def split_path(out_root: Path, name: str) -> Path:
    return Path(out_root) / SPLITS_SUBDIR / f"{name}.json"


def create_split(
    out_root: Path,
    name: str = "default",
    *,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 0,
    groups: Optional[Mapping[str, str]] = None,
    exclude: Sequence[str] = (),
    require_geometry: bool = True,
    output: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    """Choose the eligible cases, split them, and write the split file."""
    out_root = Path(out_root)
    path = Path(output) if output else split_path(out_root, name)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; splits are fixed once made (use --overwrite or another --name)")

    excluded: Dict[str, str] = {}
    eligible = []
    exclude = set(exclude)
    for cid in list_cases(out_root):
        if cid in exclude:
            excluded[cid] = "exclusion list"
        elif require_geometry and not layout.has_geometry(out_root / cid):
            excluded[cid] = "no geometry features"
        else:
            eligible.append(cid)
    unknown = sorted(exclude - set(excluded), key=layout.natural_key)
    if not eligible:
        raise ValueError(f"no eligible cases in {out_root}")

    parts = assign_parts(eligible, val=val, test=test, seed=seed, groups=groups)
    train = [ProcessedCase(out_root / c) for c in parts["train"]]
    stats = {"solution": pooled_solution_stats(train)}
    if require_geometry:
        stats["geometry"] = pooled_geometry_stats(train)

    doc = {
        "split_version": SPLIT_VERSION,
        "name": name,
        "seed": seed,
        "fractions": {"val": val, "test": test, "train": round(1 - val - test, 10)},
        "counts": {p: len(parts[p]) for p in PARTS},
        "unit": "case (geometry); all snapshots of a case are in the same part",
        "parts": parts,
        "groups": {c: str(groups[c]) for c in eligible} if groups is not None else None,
        "excluded": excluded,
        "exclusions_not_found": unknown,
        "eligible_sha256": _sha(eligible),
        "train_stats": stats,
        "processed_root": str(out_root),
        "provenance": provenance(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    tmp.replace(path)
    return path


@dataclass
class Split:
    """A loaded split file. ``split.cases(root, "train")`` opens that part's cases."""

    name: str
    parts: Dict[str, List[str]]
    document: dict = field(repr=False)

    @classmethod
    def load(cls, path: Path) -> "Split":
        with open(path) as f:
            doc = json.load(f)
        parts = {p: list(doc["parts"][p]) for p in PARTS}
        seen: Dict[str, str] = {}
        for p in PARTS:
            for c in parts[p]:
                if c in seen:
                    raise ValueError(f"{path}: case {c} is in both {seen[c]} and {p}")
                seen[c] = p
        groups = doc.get("groups")
        if groups:
            where: Dict[str, str] = {}
            for c, p in seen.items():
                g = groups[c]
                if where.setdefault(g, p) != p:
                    raise ValueError(f"{path}: group {g} is in both {where[g]} and {p}")
        return cls(doc["name"], parts, doc)

    def ids(self, part: str) -> List[str]:
        if part not in PARTS:
            raise KeyError(f"part must be one of {PARTS}, not {part!r}")
        return self.parts[part]

    def part_of(self, case_id: str) -> str:
        for p in PARTS:
            if case_id in self.parts[p]:
                return p
        raise KeyError(f"{case_id} is not in split {self.name!r}")

    def cases(self, out_root: Path, part: str, *, require_geometry: bool = True) -> List[ProcessedCase]:
        """Open the cases of one part, failing if any is missing or incomplete."""
        out_root = Path(out_root)
        cases = [ProcessedCase(out_root / c) for c in self.ids(part)]
        if require_geometry:
            missing = [c.case_id for c in cases if not c.has_geometry]
            if missing:
                raise FileNotFoundError(f"{len(missing)} {part} case(s) lack geometry features, e.g. {missing[:3]}")
        return cases

    @property
    def train_stats(self) -> dict:
        return self.document["train_stats"]


def read_groups(path: Path) -> Dict[str, str]:
    """CSV with ``case_id`` and ``group`` columns."""
    import csv

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"case_id", "group"} <= set(rows[0]):
        raise ValueError(f"{path}: need a CSV with case_id and group columns")
    return {r["case_id"].strip(): r["group"].strip() for r in rows}
