"""Tests for ChannelStats (stats.py).

Covers: chunked and pooled statistics match a direct computation on the
concatenated data, the metadata dict form round-trips, masked values are
excluded per channel, and a fully masked channel gives NaN rather than an error.
"""

import numpy as np
import pytest

from src.data.stats import ChannelStats, pool


def test_chunked_and_pooled_stats_match_direct_computation():
    rng = np.random.default_rng(0)
    parts = [rng.normal(loc=1e5, scale=300, size=(n, 3)) for n in (10, 1, 257, 64)]
    full = np.concatenate(parts)

    per_part = []
    for p in parts:
        s = ChannelStats(3)
        for chunk in np.array_split(p, 3):
            s.update(chunk)
        per_part.append(s)
    total = pool(per_part, 3)

    np.testing.assert_array_equal(total.count, len(full))
    np.testing.assert_allclose(total.mean, full.mean(0), rtol=1e-13)
    np.testing.assert_allclose(total.std, full.std(0), rtol=1e-9)
    np.testing.assert_array_equal(total.min, full.min(0))
    np.testing.assert_array_equal(total.max, full.max(0))


def test_round_trip_through_dict():
    rng = np.random.default_rng(1)
    s = ChannelStats(2)
    s.update(rng.normal(size=(50, 2)))
    names = ["a", "b"]
    r = ChannelStats.from_dict(s.to_dict(names), names)
    for attr in ("count", "mean", "std", "min", "max"):
        np.testing.assert_allclose(getattr(r, attr), getattr(s, attr), rtol=1e-12)


def test_masked_update_excludes_values_per_channel():
    x = np.array([[1.0, 10.0], [np.nan, 20.0], [3.0, 30.0]])
    s = ChannelStats(2)
    s.update(x, valid=np.isfinite(x))
    np.testing.assert_array_equal(s.count, [2, 3])
    np.testing.assert_allclose(s.mean, [2.0, 20.0])
    assert s.min[0] == 1.0 and s.max[0] == 3.0


def test_fully_masked_channel_is_nan_not_an_error():
    s = ChannelStats(1)
    s.update(np.array([[np.nan]]), valid=np.array([[False]]))
    assert s.count[0] == 0 and np.isnan(s.std[0])
    d = s.to_dict(["c"])
    assert d["c"]["count"] == 0
    with pytest.raises(ValueError):
        s.to_dict(["a", "b"])
