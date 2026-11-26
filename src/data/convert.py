"""Convert one raw transient case into the processed layout, and verify it.

The conversion keeps the static geometry and the dynamic solution apart:

* ``coordinates.npy``  (N, 3)     float32, taken from snapshot 0 after
  checking that every snapshot has the same coordinates;
* ``solution.npy``     (T, N, 4)  float32, time-major, channels
  ``x_velocity, y_velocity, z_velocity, absolute_pressure``;
* ``metadata.json``    case id, sizes, flow split (as stored), per-channel
  statistics, verification results and provenance.

No normalization, resampling or geometry features are applied. Output is
written to a temporary directory and renamed into place only after it has been
compared value-for-value against the source, so an interrupted job never
leaves a case that looks complete.

A case fails, leaving no output, if a required field is missing or
mis-shaped, the coordinates change between snapshots (beyond ``coord_atol``),
or the solution contains NaN/Inf (unless ``allow_nonfinite``).

Entry points: ``convert_case`` (``aneug convert``), ``verify_case``
(``aneug verify``), and ``record_failure`` / ``clear_failure``, which manage
the per-case error records in processed_data/_failures/.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import traceback
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np
from numpy.lib.format import open_memmap

from . import layout, raw
from .provenance import provenance
from .raw import DataValidationError
from .stats import ChannelStats

log = logging.getLogger(__name__)

Loader = Callable[[Path], Mapping[str, Any]]


@dataclass
class ConvertResult:
    case_id: str
    status: str  # "converted" | "skipped" | "failed"
    out_dir: Path
    seconds: float = 0.0
    error: Optional[str] = None


def convert_case(
    src_dir: Path,
    out_root: Path,
    *,
    overwrite: bool = False,
    coord_atol: float = 0.0,
    allow_nonfinite: bool = False,
    time_chunk: int = 8,
    loader: Optional[Loader] = None,
) -> ConvertResult:
    """Convert ``src_dir`` into ``out_root/<case>``. Raises on any validation failure."""
    src_dir, out_root = Path(src_dir), Path(out_root)
    case_id = src_dir.name
    dst = out_root / case_id
    if dst.exists() and not overwrite:
        if layout.is_complete_case(dst):
            return ConvertResult(case_id, "skipped", dst)
        raise FileExistsError(
            f"{dst} exists but was not written by this code (format v{layout.FORMAT_VERSION}); "
            "rerun with --overwrite to replace it"
        )

    start = time.time()
    out_root.mkdir(parents=True, exist_ok=True)
    _remove_stale_temp_dirs(out_root, case_id)
    tmp = out_root / f".tmp-{case_id}-{uuid.uuid4().hex[:8]}"
    tmp.mkdir()
    try:
        data = (loader or raw.load_blood_data)(src_dir / layout.BLOOD_FILE)
        metadata = _write_arrays(
            data, tmp, coord_atol=coord_atol, allow_nonfinite=allow_nonfinite, time_chunk=time_chunk
        )
        metadata["verification"] = verify_arrays(tmp, data, time_chunk=time_chunk)
        metadata = _assemble_metadata(case_id, src_dir, data, metadata)
        _write_json(tmp / layout.METADATA_FILE, metadata)  # last: marks the case complete
        _publish(tmp, dst)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    clear_failure(out_root, case_id)
    return ConvertResult(case_id, "converted", dst, seconds=time.time() - start)


def _write_arrays(
    data: Mapping[str, Any],
    out_dir: Path,
    *,
    coord_atol: float,
    allow_nonfinite: bool,
    time_chunk: int,
) -> dict:
    T, N = raw.field_time_points(data, layout.SOLUTION_FIELDS[0])
    for name in layout.SOLUTION_FIELDS + layout.COORDINATE_FIELDS:
        if raw.field_time_points(data, name) != (T, N):
            raise DataValidationError(
                f"field {name!r} has shape {tuple(data[name].shape)}; expected ({T}, {N}, 1)"
            )
    chunks = [(t0, min(t0 + time_chunk, T)) for t0 in range(0, T, time_chunk)]

    # --- static geometry ---------------------------------------------------
    ref = [raw.read_field(data, f, 0, 1)[0] for f in layout.COORDINATE_FIELDS]
    coords = np.stack(ref, axis=-1).astype(np.float32)
    if not np.isfinite(coords).all():
        raise DataValidationError("coordinates contain NaN/Inf")
    deviation = {}
    for ch, f, r in zip(layout.COORDINATE_CHANNELS, layout.COORDINATE_FIELDS, ref):
        dev = 0.0
        for t0, t1 in chunks:
            block = raw.read_field(data, f, t0, t1)
            dev = max(dev, float(np.max(np.abs(block - r))))
        deviation[ch] = dev
    worst = max(deviation.values())
    if not worst <= coord_atol:  # also catches NaN
        raise DataValidationError(
            f"coordinates change across snapshots (max |x_t - x_0| per axis: {deviation}, "
            f"tolerance {coord_atol}); the geometry is not static or the point order differs"
        )
    np.save(out_dir / layout.COORDINATES_FILE, coords)

    # --- dynamic solution --------------------------------------------------
    C = len(layout.SOLUTION_FIELDS)
    solution = open_memmap(
        out_dir / layout.SOLUTION_FILE, mode="w+", dtype=np.float32, shape=(T, N, C)
    )
    stats = ChannelStats(C)
    snap_mean = np.empty((T, C))
    snap_std = np.empty((T, C))
    nonfinite = np.zeros(C, dtype=np.int64)
    for t0, t1 in chunks:
        block = np.stack(
            [raw.read_field(data, f, t0, t1) for f in layout.SOLUTION_FIELDS], axis=-1
        ).astype(np.float32, copy=False)
        finite = np.isfinite(block)
        if finite.all():
            stats.update(block)
            snap_mean[t0:t1] = block.mean(axis=1, dtype=np.float64)
            snap_std[t0:t1] = block.std(axis=1, dtype=np.float64)
        else:
            bad = (~finite).sum(axis=(0, 1))
            if not allow_nonfinite:
                raise DataValidationError(
                    f"non-finite values in snapshots {t0}-{t1 - 1}: "
                    f"{dict(zip(layout.SOLUTION_FIELDS, bad.tolist()))}"
                )
            nonfinite += bad
            stats.update(block, valid=finite)
            masked = np.where(finite, block, np.nan).astype(np.float64)
            with warnings.catch_warnings():  # all-NaN channels: "Mean of empty slice"
                warnings.simplefilter("ignore", RuntimeWarning)
                snap_mean[t0:t1] = np.nanmean(masked, axis=1)
                snap_std[t0:t1] = np.nanstd(masked, axis=1)
        solution[t0:t1] = block
    solution.flush()
    del solution

    return {
        "num_points": N,
        "num_snapshots": T,
        "coordinate_bbox": {"min": coords.min(axis=0).tolist(), "max": coords.max(axis=0).tolist()},
        "coordinate_max_time_deviation": deviation,
        "solution_stats": stats.to_dict(layout.SOLUTION_FIELDS),
        "per_snapshot": {
            "channels": list(layout.SOLUTION_FIELDS),
            "mean": _round_list(snap_mean),
            "std": _round_list(snap_std),
        },
        "nonfinite_counts": dict(zip(layout.SOLUTION_FIELDS, nonfinite.tolist())),
    }


def verify_arrays(case_dir: Path, data: Mapping[str, Any], *, time_chunk: int = 8) -> dict:
    """Compare stored arrays value-for-value with the raw fields. Raises on mismatch."""
    T, N = raw.field_time_points(data, layout.SOLUTION_FIELDS[0])
    C = len(layout.SOLUTION_FIELDS)
    X = np.load(case_dir / layout.COORDINATES_FILE, mmap_mode="r")
    Y = np.load(case_dir / layout.SOLUTION_FILE, mmap_mode="r")
    if X.shape != (N, 3) or X.dtype != np.float32:
        raise DataValidationError(f"coordinates.npy is {X.shape} {X.dtype}; expected ({N}, 3) float32")
    if Y.shape != (T, N, C) or Y.dtype != np.float32:
        raise DataValidationError(f"solution.npy is {Y.shape} {Y.dtype}; expected ({T}, {N}, {C}) float32")

    for j, f in enumerate(layout.COORDINATE_FIELDS):
        src = raw.read_field(data, f, 0, 1)[0].astype(np.float32, copy=False)
        if not np.array_equal(X[:, j], src):
            raise DataValidationError(f"coordinates.npy channel {j} does not match {f}")
    for t0 in range(0, T, time_chunk):
        t1 = min(t0 + time_chunk, T)
        for j, f in enumerate(layout.SOLUTION_FIELDS):
            src = raw.read_field(data, f, t0, t1).astype(np.float32, copy=False)
            if not np.array_equal(Y[t0:t1, :, j], src, equal_nan=True):
                raise DataValidationError(f"solution.npy[{t0}:{t1}, :, {j}] does not match {f}")
    return {
        "coordinates_exact_match": True,
        "solution_exact_match": True,
        "snapshots_checked": T,
        "source_fields_float32": all(
            raw.dtype_name(data[f]) == "float32"
            for f in layout.SOLUTION_FIELDS + layout.COORDINATE_FIELDS
        ),
    }


def verify_case(src_dir: Path, case_dir: Path, *, time_chunk: int = 8, loader: Optional[Loader] = None) -> dict:
    """Re-check an already processed case against its raw source."""
    src_dir, case_dir = Path(src_dir), Path(case_dir)
    if not layout.is_complete_case(case_dir):
        raise DataValidationError(f"{case_dir} is not a complete v{layout.FORMAT_VERSION} case")
    with open(case_dir / layout.METADATA_FILE) as f:
        meta = json.load(f)
    data = (loader or raw.load_blood_data)(src_dir / layout.BLOOD_FILE)
    T, N = raw.field_time_points(data, layout.SOLUTION_FIELDS[0])
    if (meta["num_snapshots"], meta["num_points"]) != (T, N):
        raise DataValidationError(
            f"metadata says ({meta['num_snapshots']}, {meta['num_points']}), source has ({T}, {N})"
        )
    return verify_arrays(case_dir, data, time_chunk=time_chunk)


def _assemble_metadata(case_id: str, src_dir: Path, data: Mapping[str, Any], arrays: dict) -> dict:
    warnings = []
    flow_split = flow_split_raw = None
    flow_path = src_dir / layout.FLOWSPLIT_FILE
    if flow_path.is_file():
        flow_split, flow_split_raw, warn = raw.read_flow_split(flow_path)
        if warn:
            warnings.append(warn)
    else:
        warnings.append(f"{layout.FLOWSPLIT_FILE} missing")
    inlet_path = src_dir / layout.INLET_FILE
    inlet_raw = raw.read_text_preview(inlet_path) if inlet_path.is_file() else None
    if any(arrays["nonfinite_counts"].values()):
        warnings.append(f"non-finite solution values: {arrays['nonfinite_counts']}")
    if not arrays["verification"]["source_fields_float32"]:
        warnings.append("source fields are not all float32; values were cast")

    used = layout.SOLUTION_FIELDS + layout.COORDINATE_FIELDS
    T, N = arrays["num_snapshots"], arrays["num_points"]
    return {
        "format_version": layout.FORMAT_VERSION,
        "case_id": case_id,
        "num_points": N,
        "num_snapshots": T,
        "dtype": layout.DTYPE,
        "files": {
            layout.COORDINATES_FILE: {"shape": [N, 3], "axes": ["point", "coordinate"]},
            layout.SOLUTION_FILE: {"shape": [T, N, len(layout.SOLUTION_FIELDS)], "axes": ["snapshot", "point", "channel"]},
        },
        "coordinate_channels": list(layout.COORDINATE_CHANNELS),
        # Units as stated in the dataset description; not yet checked against
        # the surface files (which appear to be in mm). See `aneug inspect`.
        "coordinate_units": "m",
        "coordinate_units_verified": False,
        "coordinate_bbox": arrays["coordinate_bbox"],
        "coordinate_max_time_deviation": arrays["coordinate_max_time_deviation"],
        "solution_channels": list(layout.SOLUTION_FIELDS),
        "solution_stats": arrays["solution_stats"],
        "per_snapshot": arrays["per_snapshot"],
        "nonfinite_counts": arrays["nonfinite_counts"],
        # Physical times of the 80 retained snapshots are not yet verified.
        "snapshot_times": None,
        # Kept in file order; which value belongs to which outlet is not yet mapped.
        "flow_split": flow_split,
        "flow_split_raw": flow_split_raw,
        "inlet_centroids_raw": inlet_raw,
        "warnings": warnings,
        "verification": arrays["verification"],
        "source": {
            "case_dir": str(src_dir),
            "files": {
                name: raw.file_info(src_dir / name)
                for name in layout.RAW_CASE_FILES
                if (src_dir / name).is_file()
            },
            "field_dtypes": {f: raw.dtype_name(data[f]) for f in used},
            "available_fields": sorted(data),
            "dropped_fields": sorted(set(data) - set(used)),
        },
        "provenance": provenance(),
    }


# Longer than any Slurm task's time limit, so a live conversion is never removed.
STALE_TEMP_SECONDS = 12 * 3600


def _remove_stale_temp_dirs(out_root: Path, case_id: str) -> None:
    """Remove temp dirs of this case left by a killed job (SIGKILL skips cleanup)."""
    now = time.time()
    for p in out_root.glob(f".tmp-{case_id}-*"):
        try:
            if now - p.stat().st_mtime > STALE_TEMP_SECONDS:
                log.warning("removing stale temporary directory %s", p)
                shutil.rmtree(p, ignore_errors=True)
        except FileNotFoundError:
            pass


def _publish(tmp: Path, dst: Path) -> None:
    """Rename ``tmp`` to ``dst``, replacing any existing ``dst``."""
    if dst.exists():
        old = dst.with_name(f".old-{dst.name}-{uuid.uuid4().hex[:8]}")
        dst.rename(old)
        tmp.rename(dst)
        shutil.rmtree(old, ignore_errors=True)
    else:
        tmp.rename(dst)


# --- failure records -------------------------------------------------------


def failure_path(out_root: Path, case_id: str) -> Path:
    return Path(out_root) / layout.FAILURES_SUBDIR / f"{case_id}.json"


def record_failure(out_root: Path, case_id: str, exc: BaseException) -> Path:
    path = failure_path(out_root, case_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        path,
        {
            "case_id": case_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__),
            "provenance": provenance(),
        },
    )
    return path


def clear_failure(out_root: Path, case_id: str) -> None:
    path = failure_path(out_root, case_id)
    if path.exists():
        path.unlink()


# --- helpers ---------------------------------------------------------------


def _write_json(path: Path, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, allow_nan=True)
        f.write("\n")


def _round_list(a: np.ndarray, digits: int = 9) -> list:
    return [[float(f"{v:.{digits}g}") for v in row] for row in a]
