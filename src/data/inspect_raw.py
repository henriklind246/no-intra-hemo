"""Describe everything in one raw case directory, to settle open questions about the data.

``aneug inspect stable_0 -o stable_0_inspect.json`` reports, without
modifying anything:

* the key / shape / dtype tree of every ``.pt`` file (loaded with
  ``weights_only=True``; files needing arbitrary pickled classes report the
  error instead of being unpickled);
* for ``blood_data.pt``: per-field summaries at the first, middle and last
  snapshot, whether coordinates are identical across snapshots, and the
  coordinate bounding box;
* the raw text of ``flowsplit_ratio.txt`` and ``inlet_centroids.csv``;
* vertex/face counts and bounding boxes of the OBJ surfaces, and their extent
  relative to the volume coordinates (a ratio near 1000 means mm vs m).

Entry point: ``inspect_case(case_dir) -> dict``. ``describe`` builds the
generic structure summary used for any loaded object.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import layout, raw

_STATS_MAX_ELEMENTS = 5_000_000


def describe(obj: Any, *, depth: int = 0, max_items: int = 64) -> Any:
    """JSON-able structural summary of a loaded object."""
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        out = {"type": type(obj).__name__, "shape": list(obj.shape), "dtype": raw.dtype_name(obj)}
        numel = int(np.prod(obj.shape)) if len(obj.shape) else 1
        if 0 < numel <= _STATS_MAX_ELEMENTS:
            out.update(_summary(raw.to_numpy(obj)))
        return out
    if depth >= 6:
        return {"type": type(obj).__name__, "truncated": True}
    if isinstance(obj, Mapping):
        keys = list(obj)
        out = {"type": type(obj).__name__, "len": len(keys)}
        out["items"] = {str(k): describe(obj[k], depth=depth + 1) for k in keys[:max_items]}
        return out
    if isinstance(obj, (list, tuple)):
        out = {"type": type(obj).__name__, "len": len(obj)}
        out["items"] = [describe(v, depth=depth + 1) for v in obj[:8]]
        return out
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return {"type": type(obj).__name__, "value": obj if not isinstance(obj, str) else obj[:500]}
    return {"type": f"{type(obj).__module__}.{type(obj).__qualname__}", "repr": repr(obj)[:500]}


def _summary(a: np.ndarray) -> dict:
    a = np.asarray(a)
    if a.dtype.kind not in "fiub":
        return {}
    a64 = a.astype(np.float64, copy=False)
    finite = np.isfinite(a64)
    out = {"nonfinite": int((~finite).sum())}
    if finite.any():
        v = a64[finite]
        out.update(min=float(v.min()), max=float(v.max()), mean=float(v.mean()))
    return out


def _blood_details(data: Mapping[str, Any]) -> dict:
    out: dict = {"fields": {}}
    for name in sorted(data):
        x = data[name]
        entry = {"shape": list(x.shape), "dtype": raw.dtype_name(x)}
        if len(x.shape) >= 2:
            T = x.shape[0]
            entry["snapshots"] = {
                str(t): _summary(raw.to_numpy(x[t])) for t in sorted({0, T // 2, T - 1})
            }
        out["fields"][name] = entry

    if all(f in data for f in layout.COORDINATE_FIELDS):
        T, _ = raw.field_time_points(data, layout.COORDINATE_FIELDS[0])
        dev = {}
        for ch, f in zip(layout.COORDINATE_CHANNELS, layout.COORDINATE_FIELDS):
            ref = raw.read_field(data, f, 0, 1)[0]
            dev[ch] = max(
                (float(np.max(np.abs(raw.read_field(data, f, t0, min(t0 + 8, T)) - ref)))
                 for t0 in range(0, T, 8)),
                default=0.0,
            )
        coords = np.stack([raw.read_field(data, f, 0, 1)[0] for f in layout.COORDINATE_FIELDS], -1)
        out["coordinates_static"] = all(v == 0.0 for v in dev.values())
        out["coordinate_max_time_deviation"] = dev
        out["coordinate_bbox"] = {"min": coords.min(0).tolist(), "max": coords.max(0).tolist()}
    return out


def _obj_summary(path: Path) -> dict:
    verts, faces = [], 0
    with open(path, "rb") as f:
        for line in f:
            if line.startswith(b"v "):
                verts.append(line.split()[1:4])
            elif line.startswith(b"f "):
                faces += 1
    v = np.asarray(verts, dtype=np.float64)
    out: dict = {"vertices": len(v), "faces": faces}
    if len(v):
        out["bbox"] = {"min": v.min(0).tolist(), "max": v.max(0).tolist()}
    return out


def inspect_case(case_dir: Path) -> dict:
    case_dir = Path(case_dir)
    report: dict = {"case_dir": str(case_dir), "files": {}}
    for p in sorted(case_dir.iterdir()):
        if p.is_file():
            report["files"][p.name] = raw.file_info(p)

    for p in sorted(case_dir.glob("*.pt")):
        try:
            obj = raw.torch_load(p)
        except Exception as exc:  # report, don't unpickle unsafely
            report[p.name] = {"error": f"{type(exc).__name__}: {str(exc)[:2000]}"}
            continue
        if p.name == layout.BLOOD_FILE and isinstance(obj, Mapping):
            report[p.name] = _blood_details(obj)
        else:
            report[p.name] = describe(obj)

    for name in (layout.FLOWSPLIT_FILE, layout.INLET_FILE):
        p = case_dir / name
        if p.is_file():
            report[name] = {"text": raw.read_text_preview(p)}
    if (case_dir / layout.FLOWSPLIT_FILE).is_file():
        values, _, warn = raw.read_flow_split(case_dir / layout.FLOWSPLIT_FILE)
        report[layout.FLOWSPLIT_FILE].update(parsed=values, warning=warn)

    for p in sorted(case_dir.glob("*.npy")):
        try:
            a = np.load(p, mmap_mode="r", allow_pickle=False)
            report[p.name] = {"shape": list(a.shape), "dtype": str(a.dtype), **_summary(a)}
        except Exception as exc:
            report[p.name] = {"error": f"{type(exc).__name__}: {str(exc)[:500]}"}

    blood_bbox = report.get(layout.BLOOD_FILE, {}).get("coordinate_bbox")
    for p in sorted(case_dir.glob("*.obj")):
        s = _obj_summary(p)
        if blood_bbox and "bbox" in s:
            obj_ext = np.subtract(s["bbox"]["max"], s["bbox"]["min"])
            vol_ext = np.subtract(blood_bbox["max"], blood_bbox["min"])
            with np.errstate(divide="ignore", invalid="ignore"):
                s["extent_ratio_to_volume_coordinates"] = (obj_ext / vol_ext).tolist()
        report[p.name] = s
    return report
