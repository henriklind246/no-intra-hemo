"""Normalization of the operator's inputs and outputs (``aneug normalization``).

The operator maps, for one geometry,

    per point   x, y, z, wall_distance, geodesic_inlet, geodesic_outlet3,
                geodesic_outlet4, geodesic_aneurysm             (N, 8)
    per case    the outlet flow split                           (S,)
    per query   the prediction time, as a phase of the cycle    scalar
        ->
    per point   u, v, w, gauge pressure                         (N, 4)

``Normalization`` holds the constants, all taken from the training cases of a
split only, and converts both ways. The rules:

* Outputs and distances are only scaled, never shifted, so their physical
  zeros stay at zero: no-slip velocity at the wall, zero distance on the wall
  or a cap, and zero gauge pressure at the reference point.
* Velocity: one scale shared by u, v and w (the RMS velocity component), so a
  velocity vector keeps its direction.
* Pressure: the solver pins the pressure level at the coordinate origin to the
  operating pressure, 101325 Pa (checked on MSI: the volume point nearest the
  origin stays within 15 Pa of it in every snapshot). The target is gauge
  pressure ``p - 101325`` divided by its RMS. So only pressure differences
  carry physics; the offset is a convention the model can learn, because the
  origin is always inside the fluid.
* Distances: each geometry channel divided by its own RMS (wall distances are
  about 50 times smaller than geodesics).
* Coordinates: divided by one length, the largest absolute coordinate of any
  training case, so the frame is kept (no rotation, no per-axis stretch) and
  the origin, the pressure reference, stays at zero. Values lie in [-1, 1].
* Flow split: each value standardized over the training cases (z-score); the
  raw fractions vary over a narrow range.
* Time: snapshot ``k`` of ``T`` is phase ``k / T`` in [0, 1). The physical
  snapshot times are not in the dataset, so the phase assumes evenly spaced
  snapshots over one cycle. ``periodic_time_features`` turns phases into
  sin/cos pairs for models that need the periodicity built in.

Flow-split order: the operator needs the values in outlet order (node type 3,
node type 4), the order of the geodesic_outlet3/4 channels. flowsplit_ratio.txt
is in that order for most cases but reversed for some: ``aneug outlets``
measures the flow leaving each outlet, and the cases it confidently finds
reversed are stored here (``flow_split_swapped``) and reversed by
``case_flow_split``, both for the statistics and in the datasets. Cases whose
measurement failed are listed in ``flow_split_unreliable`` and keep the file
order.

Inflow: the inlet flow differs between cases (15% spread), but the mean inlet
velocity does not (2.5%; flow and inlet area correlate 0.99), so the inlet
condition is one velocity waveform for every case and the inflow follows from
the geometry. It is therefore not a separate input.

The constants are saved as JSON next to the split (``aneug normalization``)
and should be committed with it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

from .layout import natural_key
from .provenance import provenance

PRESSURE_REFERENCE_PA = 101325.0
VELOCITY_CHANNELS = ("x_velocity", "y_velocity", "z_velocity")
PRESSURE_CHANNEL = "absolute_pressure"
TARGET_CHANNELS = ("u", "v", "w", "gauge_pressure")
COORDINATE_INPUTS = ("x", "y", "z")
NORMALIZATION_VERSION = 2


def _ordered(values: Optional[Sequence[float]], swap: bool) -> Optional[List[float]]:
    if values is None:
        return None
    return list(values)[::-1] if swap else list(values)


def _rms(stats: dict, offset: float = 0.0) -> float:
    """Root mean square of a channel about ``offset``, from its mean and std."""
    return float(np.sqrt(stats["std"] ** 2 + (stats["mean"] - offset) ** 2))


def periodic_time_features(phase: np.ndarray, harmonics: int = 1) -> np.ndarray:
    """(..., 2 * harmonics): sin and cos of 2 pi k phase for k = 1..harmonics."""
    phase = np.asarray(phase, dtype=np.float64)[..., None]
    k = np.arange(1, harmonics + 1)
    a = 2 * np.pi * k * phase
    return np.concatenate([np.sin(a), np.cos(a)], axis=-1).astype(np.float32)


@dataclass
class Normalization:
    """Constants and conversions for the operator's inputs and outputs."""

    coordinate_scale: float
    geometry_channels: List[str]
    geometry_scale: List[float]
    velocity_scale: float
    pressure_scale: float
    flow_split_mean: List[float]
    flow_split_std: List[float]
    pressure_reference: float = PRESSURE_REFERENCE_PA
    flow_split_swapped: List[str] = field(default_factory=list)
    flow_split_unreliable: List[str] = field(default_factory=list)
    source: dict = field(default_factory=dict)

    # --- construction ------------------------------------------------------

    @classmethod
    def from_split(cls, split, out_root: Path, *, swapped: Iterable[str] = (),
                   unreliable: Iterable[str] = (), source: Optional[dict] = None) -> "Normalization":
        """Constants from the training cases of ``split`` (a ``Split``).

        ``swapped``: cases whose flowsplit_ratio.txt lists the outlets in the
        reverse order (from ``outlets.flow_split_order``); ``unreliable``:
        cases where that could not be measured. Both are kept in the result.
        """
        swapped = sorted(set(swapped), key=natural_key)
        cases = split.cases(out_root, "train")
        stats = split.train_stats
        sol, geo = stats["solution"], stats["geometry"]

        coord = max(float(np.abs(np.r_[c.metadata["coordinate_bbox"]["min"],
                                       c.metadata["coordinate_bbox"]["max"]]).max()) for c in cases)
        splits = [_ordered(c.flow_split, c.case_id in swapped) for c in cases]
        if any(s is None for s in splits):
            missing = [c.case_id for c, s in zip(cases, splits) if s is None]
            raise ValueError(f"{len(missing)} training case(s) have no flow split, e.g. {missing[:3]}")
        lengths = {len(s) for s in splits}
        if len(lengths) != 1:
            raise ValueError(f"flow splits have different lengths across training cases: {sorted(lengths)}")
        fs = np.asarray(splits, dtype=np.float64)
        fs_std = fs.std(axis=0)

        channels = list(geo)
        return cls(
            coordinate_scale=coord,
            geometry_channels=channels,
            geometry_scale=[_rms(geo[c]) for c in channels],
            velocity_scale=float(np.sqrt(np.mean([_rms(sol[c]) ** 2 for c in VELOCITY_CHANNELS]))),
            pressure_scale=_rms(sol[PRESSURE_CHANNEL], PRESSURE_REFERENCE_PA),
            flow_split_mean=fs.mean(axis=0).tolist(),
            # A split that never varies would divide by zero; leave it unscaled.
            flow_split_std=np.where(fs_std > 1e-12, fs_std, 1.0).tolist(),
            flow_split_swapped=swapped,
            flow_split_unreliable=sorted(set(unreliable), key=natural_key),
            source={"split": split.name, "eligible_sha256": split.document.get("eligible_sha256"),
                    "num_train_cases": len(cases), **(source or {})},
        )

    # --- inputs ------------------------------------------------------------

    @property
    def input_channels(self) -> List[str]:
        return list(COORDINATE_INPUTS) + list(self.geometry_channels)

    @property
    def num_flow_split(self) -> int:
        return len(self.flow_split_mean)

    def point_inputs(self, coordinates: np.ndarray, geometry_features: np.ndarray) -> np.ndarray:
        """(K, 3) coordinates in m and (K, G) geometry features in m -> (K, 3 + G) float32."""
        x = np.asarray(coordinates, dtype=np.float32) / np.float32(self.coordinate_scale)
        g = np.asarray(geometry_features, dtype=np.float32) / np.asarray(self.geometry_scale, dtype=np.float32)
        return np.concatenate([x, g], axis=-1)

    def case_flow_split(self, case) -> Optional[List[float]]:
        """A case's flow split in outlet order (node type 3, node type 4), not normalized."""
        return _ordered(case.flow_split, case.case_id in self.flow_split_swapped)

    def flow_split(self, values: Sequence[float]) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64)
        if v.shape[-1] != self.num_flow_split:
            raise ValueError(f"expected {self.num_flow_split} flow-split values, got {v.shape[-1]}")
        return ((v - self.flow_split_mean) / self.flow_split_std).astype(np.float32)

    @staticmethod
    def phase(snapshot: np.ndarray, num_snapshots: int) -> np.ndarray:
        """Snapshot index -> phase in [0, 1), assuming evenly spaced snapshots over one cycle."""
        return (np.asarray(snapshot, dtype=np.float64) / num_snapshots).astype(np.float32)

    # --- outputs -----------------------------------------------------------

    @property
    def target_scale(self) -> np.ndarray:
        v = np.float32(self.velocity_scale)
        return np.array([v, v, v, self.pressure_scale], dtype=np.float32)

    def targets(self, solution: np.ndarray) -> np.ndarray:
        """(..., 4) u, v, w in m/s and absolute pressure in Pa -> normalized (..., 4) float32."""
        y = np.array(solution, dtype=np.float32)
        y[..., 3] -= np.float32(self.pressure_reference)
        return y / self.target_scale

    def physical(self, normalized: np.ndarray, *, absolute_pressure: bool = False) -> np.ndarray:
        """Normalized outputs -> u, v, w in m/s and gauge pressure in Pa (or absolute, if asked)."""
        y = np.asarray(normalized, dtype=np.float32) * self.target_scale
        if absolute_pressure:
            y = y.copy()
            y[..., 3] += np.float32(self.pressure_reference)
        return y

    # --- storage -----------------------------------------------------------

    def to_dict(self) -> dict:
        return {"normalization_version": NORMALIZATION_VERSION, "input_channels": self.input_channels,
                "target_channels": list(TARGET_CHANNELS), **asdict(self)}

    @classmethod
    def from_dict(cls, d: dict) -> "Normalization":
        if d.get("normalization_version") != NORMALIZATION_VERSION:
            raise ValueError(f"normalization version {d.get('normalization_version')}, "
                             f"expected {NORMALIZATION_VERSION}")
        names = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in names})

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = self.to_dict()
        doc["provenance"] = provenance()
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(json.dumps(doc, indent=2) + "\n")
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: Path) -> "Normalization":
        return cls.from_dict(json.loads(Path(path).read_text()))


def normalization_path(split_path: Path) -> Path:
    """splits/default.json -> splits/default.normalization.json"""
    split_path = Path(split_path)
    return split_path.with_name(f"{split_path.stem}.normalization.json")
