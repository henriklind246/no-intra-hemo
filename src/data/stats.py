"""Mergeable per-channel statistics (count, mean, M2, min, max).

Each case stores its own statistics so that normalization constants for any
train split can be pooled exactly later, without re-reading the solutions.

``ChannelStats`` accumulates in float64 chunk by chunk, optionally masking
NaN/Inf per channel, and converts to and from the dict stored under
"solution_stats" in metadata.json. ``pool`` merges several into one.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np


class ChannelStats:
    """Running statistics over the last axis, merged with Chan et al.'s formula.

    Counts are kept per channel so that masked (non-finite) values can be
    excluded from one channel without affecting the others.
    """

    def __init__(self, num_channels: int):
        self.count = np.zeros(num_channels, dtype=np.int64)
        self.mean = np.zeros(num_channels, dtype=np.float64)
        self.m2 = np.zeros(num_channels, dtype=np.float64)
        self.min = np.full(num_channels, np.inf)
        self.max = np.full(num_channels, -np.inf)

    @property
    def num_channels(self) -> int:
        return self.mean.size

    def update(self, block: np.ndarray, valid: Optional[np.ndarray] = None) -> None:
        """Add samples; ``block`` is (..., C). ``valid`` masks samples to keep."""
        c = self.num_channels
        x = np.asarray(block, dtype=np.float64).reshape(-1, c)
        if x.shape[0] == 0:
            return
        if valid is None:
            n = np.full(c, x.shape[0], dtype=np.int64)
            mean = x.mean(axis=0)
            m2 = ((x - mean) ** 2).sum(axis=0)
            lo, hi = x.min(axis=0), x.max(axis=0)
        else:
            v = np.asarray(valid, dtype=bool).reshape(-1, c)
            n = v.sum(axis=0).astype(np.int64)
            mean = np.where(v, x, 0.0).sum(axis=0) / np.maximum(n, 1)
            m2 = (np.where(v, x - mean, 0.0) ** 2).sum(axis=0)
            lo = np.where(v, x, np.inf).min(axis=0)
            hi = np.where(v, x, -np.inf).max(axis=0)
        self._merge(n, mean, m2, lo, hi)

    def merge(self, other: "ChannelStats") -> None:
        self._merge(other.count, other.mean, other.m2, other.min, other.max)

    def _merge(self, n_b, mean_b, m2_b, min_b, max_b) -> None:
        n_a = self.count
        n = n_a + n_b
        safe_n = np.maximum(n, 1)
        delta = mean_b - self.mean
        self.mean = np.where(n > 0, self.mean + delta * (n_b / safe_n), 0.0)
        self.m2 = self.m2 + m2_b + delta**2 * (n_a * n_b / safe_n)
        self.min = np.minimum(self.min, min_b)
        self.max = np.maximum(self.max, max_b)
        self.count = n

    @property
    def std(self) -> np.ndarray:
        """Population standard deviation (NaN for empty channels)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.count > 0, np.sqrt(self.m2 / self.count), np.nan)

    def to_dict(self, channels: Sequence[str]) -> dict:
        if len(channels) != self.num_channels:
            raise ValueError(f"expected {self.num_channels} channel names, got {len(channels)}")
        std = self.std
        return {
            name: {
                "count": int(self.count[i]),
                "mean": float(self.mean[i]),
                "std": float(std[i]),
                "min": float(self.min[i]),
                "max": float(self.max[i]),
            }
            for i, name in enumerate(channels)
        }

    @classmethod
    def from_dict(cls, d: dict, channels: Sequence[str]) -> "ChannelStats":
        s = cls(len(channels))
        s.count = np.array([d[c]["count"] for c in channels], dtype=np.int64)
        s.mean = np.array([d[c]["mean"] for c in channels], dtype=np.float64)
        s.m2 = np.array([d[c]["std"] ** 2 * d[c]["count"] for c in channels], dtype=np.float64)
        s.min = np.array([d[c]["min"] for c in channels], dtype=np.float64)
        s.max = np.array([d[c]["max"] for c in channels], dtype=np.float64)
        return s


def pool(stats: Iterable[ChannelStats], num_channels: int) -> ChannelStats:
    total = ChannelStats(num_channels)
    for s in stats:
        total.merge(s)
    return total
