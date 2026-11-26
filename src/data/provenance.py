"""Record which code and environment produced a processed case.

``provenance()`` returns the package version, the git commit and whether the
working tree had uncommitted changes, the UTC creation time, the host name,
and the Python, NumPy and torch versions. It is stored under "provenance" in
every metadata.json and failure record, so each processed file can be traced
to the code that wrote it. If git or torch is unavailable, those entries are
null instead of raising an error.
"""

from __future__ import annotations

import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from .. import __version__


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def provenance() -> dict:
    try:
        import torch

        torch_version: Optional[str] = torch.__version__
    except ImportError:
        torch_version = None
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no") if commit else None
    return {
        "package_version": __version__,
        "git_commit": commit,
        "git_dirty": bool(status) if status is not None else None,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch_version,
    }
