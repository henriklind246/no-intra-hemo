"""Reading the raw AneuG-Flow transient case files.

``load_blood_data`` opens a blood_data.pt archive with ``torch.load``
(``weights_only=True``, memory-mapped on torch >= 2.1) as a dict of tensors
shaped (T, N, 1). ``field_time_points`` checks a field's shape, and
``read_field`` reads snapshots t0:t1 of one field as a NumPy (t, N) array, so
callers stream a few snapshots at a time instead of loading the archive.

Also here: ``read_flow_split`` (flowsplit_ratio.txt, kept in file order),
``read_text_preview``, ``file_info`` (size and mtime), ``raw_case_dirs``
(case directories in natural order) and ``DataValidationError``, raised when
the data does not match what preprocessing assumes. torch is imported only
when an archive is loaded, so the rest of the package works without it.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import layout


class DataValidationError(RuntimeError):
    """The raw data does not match what preprocessing assumes."""


def torch_load(path: Path, *, mmap: bool = True) -> Any:
    """``torch.load`` on CPU, restricted to tensors/containers, memory-mapped when possible.

    ``weights_only=True`` refuses arbitrary pickled objects; the transient
    ``blood_data.pt`` archives are plain dicts of tensors, so they load fine.
    ``mmap=True`` (torch >= 2.1) keeps the multi-GB archive on disk and pages
    in only the slices that are read.
    """
    import torch  # deferred: only needed for raw archives

    params = inspect.signature(torch.load).parameters
    kwargs: dict = {"map_location": "cpu"}
    if "weights_only" in params:
        kwargs["weights_only"] = True
    if mmap and "mmap" in params:
        kwargs["mmap"] = True
    try:
        return torch.load(str(path), **kwargs)
    except RuntimeError as exc:
        # Legacy (non-zip) archives cannot be memory-mapped.
        if kwargs.pop("mmap", False) and "mmap" in str(exc):
            return torch.load(str(path), **kwargs)
        raise


def load_blood_data(path: Path) -> Mapping[str, Any]:
    data = torch_load(path)
    if not isinstance(data, Mapping):
        raise DataValidationError(f"{path}: expected a dict of tensors, got {type(data).__name__}")
    return data


def to_numpy(x: Any) -> np.ndarray:
    """Tensor or array-like to a NumPy array (zero-copy for CPU tensors)."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def dtype_name(x: Any) -> str:
    return str(getattr(x, "dtype", type(x).__name__)).replace("torch.", "")


def field_time_points(data: Mapping[str, Any], name: str) -> tuple:
    """Return (T, N) for a field stored as (T, N) or (T, N, 1)."""
    if name not in data:
        raise DataValidationError(f"missing field {name!r}; available: {sorted(data)}")
    shape = tuple(data[name].shape)
    if len(shape) == 3 and shape[2] == 1:
        return shape[0], shape[1]
    if len(shape) == 2:
        return shape
    raise DataValidationError(f"field {name!r} has shape {shape}; expected (T, N, 1) or (T, N)")


def read_field(data: Mapping[str, Any], name: str, t0: int, t1: int) -> np.ndarray:
    """Snapshots ``t0:t1`` of a scalar field as an array of shape (t1 - t0, N)."""
    block = to_numpy(data[name][t0:t1])
    return block.reshape(block.shape[0], block.shape[1])


def read_flow_split(path: Path) -> tuple:
    """(values or None, raw text, warning or None). Ordering is kept exactly as stored."""
    text = path.read_text()
    try:
        values = np.loadtxt(path, ndmin=1).ravel().tolist()
    except ValueError as exc:
        return None, text, f"could not parse {path.name}: {exc}"
    return values, text, None


def read_text_preview(path: Path, max_bytes: int = 4096) -> str:
    with open(path, "rb") as f:
        return f.read(max_bytes).decode("utf-8", errors="replace")


def file_info(path: Path) -> dict:
    st = os.stat(path)
    return {"size_bytes": st.st_size, "mtime": st.st_mtime}


def raw_case_dirs(raw_dir: Path) -> list:
    """Case directories under ``raw_dir``, naturally sorted (stable_2 < stable_10)."""
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw data directory not found: {raw_dir}")
    dirs = [p for p in raw_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    return sorted(dirs, key=lambda p: layout.natural_key(p.name))
