"""Shared test fixtures: small synthetic raw AneuG-Flow cases.

``make_blood`` builds a blood_data.pt dict with the same 21 fields as the
audited stable_0 archive, as (T, N, 1) float32 tensors (T=6, N=37 by
default), with coordinates identical in every snapshot and pressure near
atmospheric. ``write_case`` saves one as a case directory together with
flowsplit_ratio.txt, inlet_centroids.csv and a tiny shape.obj.

The ``raw_root`` fixture is <tmp>/aneug_flow/ whose transient_data/ holds
stable_0, stable_2 and stable_10, plus stable_5, which lacks blood_data.pt.
Every test is skipped if torch is not installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# The 21 fields found in stable_0's blood_data.pt.
BLOOD_FIELDS = (
    "x_coordinate", "y_coordinate", "z_coordinate",
    "x_velocity", "y_velocity", "z_velocity", "absolute_pressure",
    "viscosity_lam", "dp_dt", "dp_dx", "dp_dy", "dp_dz",
    "dx_velocity_dx", "dx_velocity_dy", "dx_velocity_dz",
    "dy_velocity_dx", "dy_velocity_dy", "dy_velocity_dz",
    "dz_velocity_dx", "dz_velocity_dy", "dz_velocity_dz",
)

T, N = 6, 37


def make_blood(seed: int, *, T: int = T, N: int = N) -> dict:
    rng = np.random.default_rng(seed)
    coords = rng.uniform(-0.02, 0.02, size=(N, 3)).astype(np.float32)
    data = {}
    for j, axis in enumerate("xyz"):
        data[f"{axis}_coordinate"] = np.broadcast_to(coords[:, j], (T, N)).copy()
    for name in BLOOD_FIELDS:
        if name not in data:
            data[name] = rng.normal(size=(T, N)).astype(np.float32)
    data["absolute_pressure"] = (101325 + 500 * data["absolute_pressure"]).astype(np.float32)
    return {k: torch.from_numpy(v[..., None].copy()) for k, v in data.items()}


def write_case(raw_dir: Path, case_id: str, blood: dict, flow_split="0.4 0.6\n") -> Path:
    d = raw_dir / case_id
    d.mkdir(parents=True)
    torch.save(blood, d / "blood_data.pt")
    if flow_split is not None:
        (d / "flowsplit_ratio.txt").write_text(flow_split)
    (d / "inlet_centroids.csv").write_text("x,y,z\n1.0,2.0,3.0\n")
    (d / "shape.obj").write_text("v 0 0 0\nv 20 30 40\nv 1 1 1\nf 1 2 3\n")
    return d


@pytest.fixture
def raw_root(tmp_path: Path) -> Path:
    """<root>/transient_data/ with three good cases and one missing blood_data.pt."""
    root = tmp_path / "aneug_flow"
    raw_dir = root / "transient_data"
    for i in (0, 2, 10):
        write_case(raw_dir, f"stable_{i}", make_blood(i))
    (raw_dir / "stable_5").mkdir()
    (raw_dir / "stable_5" / "flowsplit_ratio.txt").write_text("0.5 0.5\n")
    return root
