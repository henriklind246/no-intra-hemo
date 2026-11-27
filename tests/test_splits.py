"""Tests of the geometry-level train/val/test split (splits.py, ``aneug split``).

Covers: assign_parts partitions the case ids exactly, gives the requested
sizes, depends only on the case set and seed (not input order), keeps groups
whole, and rejects bad fractions; create_split excludes cases without
geometry features or on the exclusion list, stores training-only statistics
equal to pooling the training cases directly, and refuses to overwrite;
Split.load rejects overlapping parts and split groups; the CLI command.
"""

import json

import numpy as np
import pytest

from src.data import cli, layout, splits
from src.data.geometry import FEATURE_CHANNELS
from src.data.processed import ProcessedCase, pooled_solution_stats
from src.data.stats import ChannelStats

from conftest import make_blood, write_case

IDS = [f"stable_{i}" for i in range(40)]


def test_assign_parts_is_an_exact_partition():
    parts = splits.assign_parts(IDS, val=0.1, test=0.2, seed=3)
    assert {p: len(v) for p, v in parts.items()} == {"train": 28, "val": 4, "test": 8}
    together = parts["train"] + parts["val"] + parts["test"]
    assert sorted(together) == sorted(IDS) and len(set(together)) == len(IDS)


def test_assign_parts_depends_only_on_set_and_seed():
    a = splits.assign_parts(IDS, seed=1)
    assert splits.assign_parts(list(reversed(IDS)), seed=1) == a
    assert splits.assign_parts(IDS, seed=2) != a


def test_assign_parts_keeps_groups_whole():
    groups = {c: f"g{i // 4}" for i, c in enumerate(IDS)}  # 10 groups of 4
    parts = splits.assign_parts(IDS, val=0.1, test=0.1, seed=0, groups=groups)
    where = {}
    for p, ids in parts.items():
        for c in ids:
            assert where.setdefault(groups[c], p) == p
    assert len(parts["val"]) == 4 and len(parts["test"]) == 4


@pytest.mark.parametrize("val,test", [(-0.1, 0.1), (0.5, 0.5), (1.0, 0.0)])
def test_assign_parts_rejects_bad_fractions(val, test):
    with pytest.raises(ValueError):
        splits.assign_parts(IDS, val=val, test=test)


def write_fake_geometry(case_dir, seed):
    """A minimal geometry.json, enough for has_geometry and the pooled statistics."""
    rng = np.random.default_rng(seed)
    st = ChannelStats(len(FEATURE_CHANNELS))
    st.update(rng.uniform(0, 0.02, size=(50, len(FEATURE_CHANNELS))))
    doc = {"geometry_version": layout.GEOMETRY_VERSION, "feature_channels": list(FEATURE_CHANNELS),
           "feature_stats": st.to_dict(FEATURE_CHANNELS), "warnings": []}
    (case_dir / layout.GEOMETRY_METADATA_FILE).write_text(json.dumps(doc))


@pytest.fixture
def many_cases(tmp_path):
    """20 converted cases; all but stable_7 and stable_13 have geometry features."""
    root = tmp_path / "aneug_flow"
    for i in range(20):
        write_case(root / "transient_data", f"stable_{i}", make_blood(i, T=3, N=11))
    assert cli.main(["convert", "--root", str(root), "--all"]) == 0
    out = root / "processed_data"
    for i in range(20):
        if i not in (7, 13):
            write_fake_geometry(out / f"stable_{i}", i)
    return root


def test_create_split(many_cases):
    out = many_cases / "processed_data"
    path = splits.create_split(out, "s", val=0.2, test=0.2, seed=5, exclude=["stable_3", "nope"])
    assert path == out / "splits" / "s.json"
    split = splits.Split.load(path)
    doc = split.document
    assert doc["excluded"] == {"stable_3": "exclusion list", "stable_7": "no geometry features",
                               "stable_13": "no geometry features"}
    assert doc["exclusions_not_found"] == ["nope"]
    assert doc["counts"] == {"train": 11, "val": 3, "test": 3}
    for c in ("stable_3", "stable_7", "stable_13"):
        with pytest.raises(KeyError):
            split.part_of(c)

    # Statistics come from the training cases only.
    train = split.cases(out, "train")
    assert [c.case_id for c in train] == split.ids("train")
    assert split.train_stats["solution"] == pooled_solution_stats(train)
    n_train = sum(c.num_points * c.num_snapshots for c in train)
    assert split.train_stats["solution"]["x_velocity"]["count"] == n_train
    assert split.train_stats["geometry"]["wall_distance"]["count"] == 50 * len(train)

    with pytest.raises(FileExistsError):
        splits.create_split(out, "s")
    splits.create_split(out, "s", seed=6, overwrite=True)
    assert splits.Split.load(path).parts != split.parts


def test_allow_missing_geometry_includes_every_case(many_cases):
    out = many_cases / "processed_data"
    split = splits.Split.load(splits.create_split(out, require_geometry=False))
    assert sum(split.document["counts"].values()) == 20
    assert "geometry" not in split.train_stats


def test_load_rejects_overlap_and_split_groups(many_cases, tmp_path):
    out = many_cases / "processed_data"
    doc = splits.Split.load(splits.create_split(out)).document
    bad = json.loads(json.dumps(doc))
    bad["parts"]["val"].append(bad["parts"]["train"][0])
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="in both"):
        splits.Split.load(p)

    bad = json.loads(json.dumps(doc))
    a, b = bad["parts"]["train"][0], bad["parts"]["test"][0]
    bad["groups"] = {c: c for p in splits.PARTS for c in bad["parts"][p]}
    bad["groups"][b] = a
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="group"):
        splits.Split.load(p)


def test_split_command_with_groups(many_cases, tmp_path, capsys):
    groups = tmp_path / "groups.csv"
    rows = [f"stable_{i},parent{i // 2}" for i in range(20)]
    groups.write_text("case_id,group\n" + "\n".join(rows) + "\n")
    assert cli.main(["split", "--root", str(many_cases), "--name", "g", "--groups-file", str(groups)]) == 0
    summary = json.loads(capsys.readouterr().out)
    split = splits.Split.load(summary["split"])
    for i in range(0, 20, 2):
        a, b = f"stable_{i}", f"stable_{i + 1}"
        if a in split.document["groups"] and b in split.document["groups"]:
            assert split.part_of(a) == split.part_of(b)
    # An existing split is never replaced silently.
    assert cli.main(["split", "--root", str(many_cases), "--name", "g"]) == 1
