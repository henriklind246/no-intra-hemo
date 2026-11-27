"""Which flow-split value belongs to which outlet (``aneug outlets``).

flowsplit_ratio.txt lists one fraction per outlet, in an order the dataset
does not document. The operator takes the split as an input next to the
geodesic_outlet3/4 channels, so the order has to match node types 3 and 4.
This module measures it from the solution: the volumetric flow through each
cap, estimated from the volume points on the cap (the same cap plane, radius
and slab that seeded the geodesics, read from geometry.json).

Flow through a cap: the points within two slabs of the cap plane and inside
its radius are binned on a square grid in the plane (cell = radius / 8), the
normal velocity is averaged per cell and then over the cells, and multiplied
by the cap area. Averaging per cell first stops the dense near-wall layers
from outweighing the core. The normal points into the fluid, so inflow is
positive. It is an estimate: the check that it is good enough is that the
inlet flow matches the sum of the outlet flows (``balance`` near 1).

Per case the report gives the measured fraction leaving through each outlet,
averaged over the snapshots, the file's values, and which order fits:
``file order`` (first value = node type 3), ``swapped``, or ``ambiguous``
when the two values are too close to tell.

That verdict only says which order is closer, which near a 50/50 split is
decided by the 2-6% measurement error. ``decide`` is the rule actually used
for the operator's input (``flow_split_order``):
    unreliable  the measurement fits neither order (error above 0.1 in both);
                the file order is kept
    swap        the measured fraction and the file's first value lie on
                opposite sides of 0.5, each at least 0.05 away from it
    keep        everything else (where the two values are within 0.05 of
                0.5, which order is used hardly matters)
On the full dataset (2026-11-27): 14 swaps, 8 unreliable.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from .processed import ProcessedCase

CAPS = ("geodesic_inlet", "geodesic_outlet3", "geodesic_outlet4")
GRID_CELLS_PER_RADIUS = 8
AMBIGUOUS = 0.02  # file values closer than this cannot be told apart
SWAP_MARGIN = 0.05  # both fractions at least this far from 0.5 to swap
UNRELIABLE_ERROR = 0.1  # measurement fits neither order


def cap_flow(case: ProcessedCase, snapshots: Optional[Iterable[int]] = None) -> Dict[str, np.ndarray]:
    """Estimated flow into the fluid through each cap (m^3/s), one value per snapshot."""
    snapshots = list(range(0, case.num_snapshots, 4) if snapshots is None else snapshots)
    sources = case.geometry["checks"]["geodesic_sources"]
    x = np.asarray(case.coordinates, dtype=np.float64)
    g = case.geometry_features
    out = {}
    for name in CAPS:
        info = sources[name]
        c, n, r = np.asarray(info["centroid"]), np.asarray(info["normal"]), float(info["radius"])
        slab = float(info.get("slab", r / 10))
        geo = np.asarray(g[:, case.geometry_channels.index(name)], dtype=np.float64)
        near = np.flatnonzero(geo <= 6 * slab)
        rel = x[near] - c
        pd = rel @ n
        # Orient the normal into the fluid: points further along the geodesic lie on that side.
        far = geo[near] > 2 * slab
        if far.any() and pd[far].mean() < 0:
            n, pd = -n, -pd
        inplane = rel - np.outer(pd, n)
        on = (np.abs(pd) <= 2 * slab) & (np.linalg.norm(inplane, axis=1) <= r)
        idx = near[on]
        if idx.size == 0:
            out[name] = np.full(len(snapshots), np.nan)
            continue
        e1 = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        h = r / GRID_CELLS_PER_RADIUS
        cells = np.floor(np.stack([inplane[on] @ e1, inplane[on] @ e2], axis=1) / h).astype(np.int64)
        _, cell = np.unique(cells, axis=0, return_inverse=True)
        cell = cell.ravel()
        counts = np.bincount(cell)
        order = np.argsort(idx)
        idx_sorted, cell_sorted = idx[order], cell[order]
        q = []
        for t in snapshots:
            un = np.asarray(case.solution[t, idx_sorted, :3], dtype=np.float64) @ n
            q.append((np.bincount(cell_sorted, weights=un) / counts).mean() * np.pi * r**2)
        out[name] = np.asarray(q)
    out["snapshots"] = np.asarray(snapshots)
    out["inlet_area"] = np.pi * float(sources["geodesic_inlet"]["radius"]) ** 2
    return out


def decide(measured_fraction_outlet3: float, file_split: List[float]) -> str:
    """``keep``, ``swap`` or ``unreliable`` for one case (see the module docstring)."""
    if len(file_split) != 2 or not np.isfinite(measured_fraction_outlet3):
        return "unreliable"
    f3, a = measured_fraction_outlet3, file_split[0]
    if min(abs(f3 - a), abs(f3 - file_split[1])) > UNRELIABLE_ERROR:
        return "unreliable"
    if (f3 - 0.5) * (a - 0.5) < 0 and abs(f3 - 0.5) >= SWAP_MARGIN and abs(a - 0.5) >= SWAP_MARGIN:
        return "swap"
    return "keep"


def flow_split_order(report: Path) -> dict:
    """Swapped and unreliable case ids from an ``aneug outlets`` report CSV."""
    with open(report, newline="") as f:
        rows = list(csv.DictReader(f))
    out: Dict[str, List[str]] = {"swap": [], "unreliable": [], "keep": []}
    for r in rows:
        fs = [float(v) for v in r["file_flow_split"].split()]
        out[decide(float(r["measured_fraction_outlet3"]), fs)].append(r["case_id"])
    return {"swapped": out["swap"], "unreliable": out["unreliable"], "num_cases": len(rows),
            "rule": f"swap if measured and file fractions lie on opposite sides of 0.5, each >= {SWAP_MARGIN} "
                    f"away; unreliable if the error exceeds {UNRELIABLE_ERROR} in both orders"}


def outlet_report(case: ProcessedCase, snapshots: Optional[Iterable[int]] = None) -> dict:
    """One report row: measured outlet fractions vs the file's flow split."""
    q = cap_flow(case, snapshots)
    q_in = q["geodesic_inlet"].mean()
    out3, out4 = -q["geodesic_outlet3"].mean(), -q["geodesic_outlet4"].mean()
    f3 = out3 / (out3 + out4)
    fs = case.flow_split or []
    row = {
        "case_id": case.case_id,
        "measured_fraction_outlet3": round(float(f3), 4),
        "measured_fraction_outlet4": round(float(1 - f3), 4),
        "file_flow_split": " ".join(str(v) for v in fs),
        "balance": round(float((out3 + out4) / q_in), 4) if q_in else float("nan"),
        "inlet_flow_mean_m3s": float(q_in),
        "inlet_flow_max_m3s": float(q["geodesic_inlet"].max()),
        "inlet_flow_argmax_snapshot": int(q["snapshots"][np.argmax(q["geodesic_inlet"])]),
        "inlet_mean_velocity_ms": float(q_in / q["inlet_area"]),
        "decision": decide(float(f3), list(fs)),
    }
    if len(fs) == 2:
        err_file, err_swap = abs(f3 - fs[0]), abs(f3 - fs[1])
        row["error_file_order"] = round(float(err_file), 4)
        row["error_swapped"] = round(float(err_swap), 4)
        if abs(fs[0] - fs[1]) < AMBIGUOUS:
            row["verdict"] = "ambiguous"
        else:
            row["verdict"] = "file order" if err_file < err_swap else "swapped"
    else:
        row["verdict"] = f"{len(fs)} flow-split values"
    return row


def write_report(rows: List[dict], path: Path) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def summarize(rows: List[dict]) -> dict:
    verdicts: Dict[str, int] = {}
    decisions: Dict[str, int] = {}
    for r in rows:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
        decisions[r["decision"]] = decisions.get(r["decision"], 0) + 1
    bal = np.array([r["balance"] for r in rows], dtype=np.float64)
    e = np.array([min(r.get("error_file_order", np.nan), r.get("error_swapped", np.nan)) for r in rows])
    q = np.array([r["inlet_flow_mean_m3s"] for r in rows])
    v = np.array([r["inlet_mean_velocity_ms"] for r in rows])
    return {
        "cases": len(rows),
        "verdicts": verdicts,
        "decisions": decisions,
        "inlet_mean_velocity_cv": float(v.std() / v.mean()),
        "inlet_flow_cv": float(q.std() / q.mean()),
        "balance_median": float(np.nanmedian(bal)),
        "balance_p05_p95": [float(np.nanpercentile(bal, 5)), float(np.nanpercentile(bal, 95))],
        "best_order_error_median": float(np.nanmedian(e)) if np.isfinite(e).any() else None,
        "best_order_error_p95": float(np.nanpercentile(e, 95)) if np.isfinite(e).any() else None,
        "inlet_flow_mean_m3s_min_median_max": [float(q.min()), float(np.median(q)), float(q.max())],
        "inlet_peak_snapshots": sorted({r["inlet_flow_argmax_snapshot"] for r in rows}),
    }
