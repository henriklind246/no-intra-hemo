"""Names, paths and constants shared by the raw and processed AneuG-Flow layouts.

Raw (downloaded, never modified)::

    <root>/transient_data/<case>/blood_data.pt
                                 flowsplit_ratio.txt
                                 inlet_centroids.csv
                                 ...

Processed (written by ``aneug convert``)::

    <root>/processed_data/<case>/coordinates.npy   (N, 3)     float32
                                 solution.npy      (T, N, 4)  float32, time-major
                                 metadata.json

Geometry stage (written by ``aneug features`` into the same case directory)::

    <case>/geometry_features.npy  (N, 5)  float32
           wall_projection.npz    closest wall triangle + barycentric weights per point
           wall/                  wall_mesh.npz and copies of the raw wall files
           geometry.json          written last; marks the stage complete

Every other module takes its names from here so they are defined once: the
raw and processed file names, the coordinate and solution field names (and
so the channel order of solution.npy), ``FORMAT_VERSION`` (stored in each
metadata.json), ``default_root()`` and ``is_complete_case()``.
"""

from __future__ import annotations

import getpass
import json
import os
import re
from pathlib import Path

# Bump when the processed layout or metadata schema changes incompatibly.
FORMAT_VERSION = 1

RAW_SUBDIR = "transient_data"
PROCESSED_SUBDIR = "processed_data"

BLOOD_FILE = "blood_data.pt"
FLOWSPLIT_FILE = "flowsplit_ratio.txt"
INLET_FILE = "inlet_centroids.csv"
# Every file the audit found in each transient case directory.
RAW_CASE_FILES = (
    BLOOD_FILE,
    FLOWSPLIT_FILE,
    INLET_FILE,
    "wall_data.pt",
    "wall_graph_static.pt",
    "wall_mesh_registration.pt",
    "checkpoint.npy",
    "shape.obj",
    "shape_remeshed.obj",
)
# Files a case needs before it can be converted.
REQUIRED_FOR_CONVERT = (BLOOD_FILE, FLOWSPLIT_FILE)

COORDINATES_FILE = "coordinates.npy"
SOLUTION_FILE = "solution.npy"
METADATA_FILE = "metadata.json"
FAILURES_SUBDIR = "_failures"
INDEX_FILE = "index.csv"
RAW_MANIFEST_FILE = "raw_manifest.csv"

# Geometry stage.
WALL_DATA_FILE = "wall_data.pt"
WALL_GRAPH_FILE = "wall_graph_static.pt"
WALL_REGISTRATION_FILE = "wall_mesh_registration.pt"
REQUIRED_FOR_FEATURES = (WALL_REGISTRATION_FILE, WALL_GRAPH_FILE, INLET_FILE)
GEOMETRY_FEATURES_FILE = "geometry_features.npy"
WALL_PROJECTION_FILE = "wall_projection.npz"
WALL_SUBDIR = "wall"
WALL_MESH_FILE = "wall_mesh.npz"
GEOMETRY_METADATA_FILE = "geometry.json"
# Bump when the geometry features or their files change incompatibly.
GEOMETRY_VERSION = 1

COORDINATE_FIELDS = ("x_coordinate", "y_coordinate", "z_coordinate")
COORDINATE_CHANNELS = ("x", "y", "z")
SOLUTION_FIELDS = ("x_velocity", "y_velocity", "z_velocity", "absolute_pressure")

DTYPE = "float32"

DEFAULT_SCRATCH = Path("/scratch.global")
ROOT_ENV_VAR = "ANEUG_ROOT"


def default_root() -> Path:
    """``$ANEUG_ROOT`` if set, else ``/scratch.global/$USER/aneug_flow``."""
    env = os.environ.get(ROOT_ENV_VAR)
    if env:
        return Path(env)
    user = os.environ.get("USER") or getpass.getuser()
    return DEFAULT_SCRATCH / user / "aneug_flow"


def natural_key(name: str):
    """Sort ``stable_2`` before ``stable_10``."""
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", name)]


def is_complete_case(case_dir: Path) -> bool:
    """True if ``case_dir`` was fully written by this code's current format.

    Cases are written to a temporary directory and renamed into place as the
    final step, so a directory named after the case is never half-written.
    The format check rejects output from older formats or ad-hoc scripts.
    """
    try:
        with open(case_dir / METADATA_FILE) as f:
            return json.load(f).get("format_version") == FORMAT_VERSION
    except (OSError, ValueError):
        return False


def has_geometry(case_dir: Path) -> bool:
    """True if ``aneug features`` finished for ``case_dir`` with the current version."""
    try:
        with open(case_dir / GEOMETRY_METADATA_FILE) as f:
            return json.load(f).get("geometry_version") == GEOMETRY_VERSION
    except (OSError, ValueError):
        return False
