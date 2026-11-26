"""Inventory of the raw transient cases, used by ``aneug scan`` and the Slurm submit script.

``scan_raw`` walks <root>/transient_data/ and records, for each case
directory, which of the expected files exist, their total size and the oldest
modification time. ``write_manifest`` saves this as a CSV with one row per
case (``convertible`` means blood_data.pt and flowsplit_ratio.txt are both
present). ``summarize`` reports totals, missing files, the estimated processed
size (about 19% of the blood_data.pt bytes) and how close the oldest file is
to the 30-day /scratch.global purge. ``read_case_list`` reads a manifest CSV or
a plain list of case ids; that is how a subset is passed to ``aneug convert``.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from . import layout, raw

# Processed bytes per raw blood_data.pt byte, assuming the audited layout:
# 21 equally sized float32 fields of shape (T, N, 1), of which 4 are kept
# time-dependent and 3 are kept once. With T = 80: (4*80 + 3) / (21*80).
_PROCESSED_FRACTION = (4 * 80 + 3) / (21 * 80)
SCRATCH_PURGE_DAYS = 30


@dataclass
class RawCase:
    case_id: str
    path: Path
    present: Dict[str, bool] = field(default_factory=dict)
    blood_bytes: int = 0
    total_bytes: int = 0
    oldest_mtime: Optional[float] = None

    @property
    def convertible(self) -> bool:
        return all(self.present.get(name, False) for name in layout.REQUIRED_FOR_CONVERT)


def scan_raw(raw_dir: Path) -> List[RawCase]:
    cases = []
    for d in raw.raw_case_dirs(Path(raw_dir)):
        c = RawCase(d.name, d)
        for name in layout.RAW_CASE_FILES:
            p = d / name
            c.present[name] = p.is_file()
            if c.present[name]:
                info = raw.file_info(p)
                c.total_bytes += info["size_bytes"]
                if name == layout.BLOOD_FILE:
                    c.blood_bytes = info["size_bytes"]
                if c.oldest_mtime is None or info["mtime"] < c.oldest_mtime:
                    c.oldest_mtime = info["mtime"]
        cases.append(c)
    return cases


def write_manifest(cases: List[RawCase], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["case_id", "convertible", "blood_bytes", "total_bytes", "oldest_mtime_utc"]
    columns += [f"has_{name}" for name in layout.RAW_CASE_FILES]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for c in cases:
            w.writerow(
                [c.case_id, int(c.convertible), c.blood_bytes, c.total_bytes, _iso(c.oldest_mtime)]
                + [int(c.present[name]) for name in layout.RAW_CASE_FILES]
            )


def read_case_list(path: Path) -> List[str]:
    """Case ids from a manifest CSV (``case_id`` column) or a plain list, one per line."""
    path = Path(path)
    with open(path, newline="") as f:
        first = f.readline()
        f.seek(0)
        if first.startswith("case_id"):
            return [row["case_id"] for row in csv.DictReader(f)]
        return [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]


def summarize(cases: List[RawCase], now: Optional[float] = None) -> dict:
    now = time.time() if now is None else now
    missing = {
        name: [c.case_id for c in cases if not c.present[name]] for name in layout.RAW_CASE_FILES
    }
    blood = sum(c.blood_bytes for c in cases if c.convertible)
    mtimes = [c.oldest_mtime for c in cases if c.oldest_mtime is not None]
    oldest = min(mtimes) if mtimes else None
    return {
        "num_cases": len(cases),
        "num_convertible": sum(c.convertible for c in cases),
        "missing_files": {name: ids for name, ids in missing.items() if ids},
        "raw_total_gb": sum(c.total_bytes for c in cases) / 1e9,
        "blood_data_gb": blood / 1e9,
        "estimated_processed_gb": blood * _PROCESSED_FRACTION / 1e9,
        "oldest_file_mtime_utc": _iso(oldest),
        "oldest_file_age_days": None if oldest is None else (now - oldest) / 86400,
        "days_until_oldest_reaches_scratch_purge_age": (
            None if oldest is None else SCRATCH_PURGE_DAYS - (now - oldest) / 86400
        ),
    }


def _iso(ts: Optional[float]) -> str:
    if ts is None:
        return ""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))
