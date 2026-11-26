"""Tests for converting and verifying single cases (convert.py).

Covers: output shapes, dtypes and exact values; metadata contents and
statistics; skipping, overwriting, and refusing directories not written by
this code; failures for moving coordinates, NaN/Inf, missing fields and bad
shapes, each leaving no partial output; cleanup after interrupted jobs and of
stale temporary directories; unparseable flow-split files; (T, N) fields
without the trailing axis; and detection of corrupted output by verify_case.
"""

import json

import numpy as np
import pytest

from src.data import convert, layout
from src.data.processed import ProcessedCase
from src.data.raw import DataValidationError

from conftest import N, T, make_blood, write_case


def _source(blood, name):
    return blood[name].numpy()[..., 0]


def test_convert_writes_static_geometry_and_time_major_solution(raw_root, tmp_path):
    out = tmp_path / "processed"
    blood = make_blood(0)
    res = convert.convert_case(raw_root / "transient_data" / "stable_0", out, time_chunk=4)

    assert res.status == "converted"
    case_dir = out / "stable_0"
    assert sorted(p.name for p in case_dir.iterdir()) == [
        "coordinates.npy", "metadata.json", "solution.npy"
    ]
    X = np.load(case_dir / "coordinates.npy")
    Y = np.load(case_dir / "solution.npy", mmap_mode="r")
    assert X.shape == (N, 3) and X.dtype == np.float32
    assert Y.shape == (T, N, 4) and Y.dtype == np.float32
    for j, axis in enumerate("xyz"):
        np.testing.assert_array_equal(X[:, j], _source(blood, f"{axis}_coordinate")[0])
    for j, f in enumerate(layout.SOLUTION_FIELDS):
        np.testing.assert_array_equal(Y[:, :, j], _source(blood, f))
    # No temp dirs left behind.
    assert [p.name for p in out.iterdir()] == ["stable_0"]


def test_metadata_contents(raw_root, tmp_path):
    out = tmp_path / "processed"
    convert.convert_case(raw_root / "transient_data" / "stable_0", out)
    md = json.loads((out / "stable_0" / "metadata.json").read_text())
    blood = make_blood(0)

    assert md["format_version"] == layout.FORMAT_VERSION
    assert (md["case_id"], md["num_points"], md["num_snapshots"]) == ("stable_0", N, T)
    assert md["solution_channels"] == list(layout.SOLUTION_FIELDS)
    assert md["flow_split"] == [0.4, 0.6]
    assert md["snapshot_times"] is None
    assert md["warnings"] == []
    assert md["verification"]["solution_exact_match"] is True
    assert md["verification"]["snapshots_checked"] == T
    assert md["coordinate_max_time_deviation"] == {"x": 0.0, "y": 0.0, "z": 0.0}
    assert len(md["source"]["dropped_fields"]) == 21 - 7
    assert md["provenance"]["package_version"]

    p = _source(blood, "absolute_pressure").astype(np.float64)
    s = md["solution_stats"]["absolute_pressure"]
    assert s["count"] == T * N
    assert s["mean"] == pytest.approx(p.mean(), rel=1e-12)
    assert s["std"] == pytest.approx(p.std(), rel=1e-9)
    assert s["min"] == pytest.approx(p.min()) and s["max"] == pytest.approx(p.max())
    np.testing.assert_allclose(
        np.array(md["per_snapshot"]["mean"])[:, 3], p.mean(axis=1), rtol=1e-8
    )


def test_existing_case_is_skipped_unless_overwrite(raw_root, tmp_path):
    out = tmp_path / "processed"
    src = raw_root / "transient_data" / "stable_0"
    assert convert.convert_case(src, out).status == "converted"
    marker = out / "stable_0" / "marker"
    marker.touch()
    assert convert.convert_case(src, out).status == "skipped"
    assert marker.exists()
    assert convert.convert_case(src, out, overwrite=True).status == "converted"
    assert not marker.exists()  # directory was replaced, not merged
    assert [p.name for p in out.iterdir()] == ["stable_0"]


def test_stale_temp_dirs_of_a_killed_job_are_removed(raw_root, tmp_path):
    import os

    out = tmp_path / "processed"
    stale = out / ".tmp-stable_0-deadbeef"
    fresh = out / ".tmp-stable_0-cafef00d"  # could be a live conversion
    for d in (stale, fresh):
        d.mkdir(parents=True)
    old = stale.stat().st_mtime - convert.STALE_TEMP_SECONDS - 60
    os.utime(stale, (old, old))
    convert.convert_case(raw_root / "transient_data" / "stable_0", out)
    assert not stale.exists() and fresh.exists()


def test_interrupt_removes_partial_output(raw_root, tmp_path):
    out = tmp_path / "processed"

    def killed_loader(path):
        raise SystemExit(143)  # what the SIGTERM handler raises

    with pytest.raises(SystemExit):
        convert.convert_case(raw_root / "transient_data" / "stable_0", out, loader=killed_loader)
    assert list(out.iterdir()) == []


def test_foreign_directory_is_not_silently_reused(raw_root, tmp_path):
    out = tmp_path / "processed"
    (out / "stable_0").mkdir(parents=True)
    (out / "stable_0" / "metadata.json").write_text('{"case_id": "stable_0"}')  # e.g. ad-hoc script
    with pytest.raises(FileExistsError, match="--overwrite"):
        convert.convert_case(raw_root / "transient_data" / "stable_0", out)
    assert convert.convert_case(
        raw_root / "transient_data" / "stable_0", out, overwrite=True
    ).status == "converted"


def test_moving_coordinates_fail_and_leave_nothing(tmp_path):
    raw_dir = tmp_path / "raw"
    blood = make_blood(1)
    blood["y_coordinate"][3, 5, 0] += 1e-4
    write_case(raw_dir, "stable_1", blood)
    out = tmp_path / "processed"
    with pytest.raises(DataValidationError, match="coordinates change"):
        convert.convert_case(raw_dir / "stable_1", out)
    assert list(out.iterdir()) == []
    # A tolerance can be granted explicitly; the deviation is recorded.
    convert.convert_case(raw_dir / "stable_1", out, coord_atol=1e-3)
    md = ProcessedCase(out / "stable_1").metadata
    assert md["coordinate_max_time_deviation"]["y"] == pytest.approx(1e-4, rel=1e-2)


def test_nonfinite_values_fail_unless_allowed(tmp_path):
    raw_dir = tmp_path / "raw"
    blood = make_blood(2)
    blood["z_velocity"][2, 7, 0] = float("nan")
    write_case(raw_dir, "stable_2", blood)
    out = tmp_path / "processed"
    with pytest.raises(DataValidationError, match="non-finite"):
        convert.convert_case(raw_dir / "stable_2", out)
    assert list(out.iterdir()) == []

    convert.convert_case(raw_dir / "stable_2", out, allow_nonfinite=True)
    md = ProcessedCase(out / "stable_2").metadata
    assert md["nonfinite_counts"]["z_velocity"] == 1
    assert md["solution_stats"]["z_velocity"]["count"] == T * N - 1
    assert md["solution_stats"]["x_velocity"]["count"] == T * N
    assert any("non-finite" in w for w in md["warnings"])
    assert np.isnan(np.load(out / "stable_2" / "solution.npy")[2, 7, 2])


def test_missing_field_and_bad_shape_are_reported(tmp_path):
    raw_dir = tmp_path / "raw"
    blood = make_blood(3)
    del blood["absolute_pressure"]
    write_case(raw_dir, "a", blood)
    with pytest.raises(DataValidationError, match="absolute_pressure"):
        convert.convert_case(raw_dir / "a", tmp_path / "out")

    blood = make_blood(3)
    blood["y_velocity"] = blood["y_velocity"][:, :-1]
    write_case(raw_dir, "b", blood)
    with pytest.raises(DataValidationError, match="y_velocity"):
        convert.convert_case(raw_dir / "b", tmp_path / "out")


def test_unparseable_flow_split_is_a_warning(tmp_path):
    raw_dir = tmp_path / "raw"
    write_case(raw_dir, "c", make_blood(4), flow_split="outlet1: 0.3\n")
    convert.convert_case(raw_dir / "c", tmp_path / "out")
    md = ProcessedCase(tmp_path / "out" / "c").metadata
    assert md["flow_split"] is None
    assert md["flow_split_raw"] == "outlet1: 0.3\n"
    assert any("flowsplit_ratio.txt" in w for w in md["warnings"])


def test_verify_case_detects_corruption(raw_root, tmp_path):
    out = tmp_path / "processed"
    src = raw_root / "transient_data" / "stable_0"
    convert.convert_case(src, out)
    assert convert.verify_case(src, out / "stable_0")["solution_exact_match"]

    Y = np.load(out / "stable_0" / "solution.npy", mmap_mode="r+")
    Y[T - 1, 0, 1] += 1.0
    Y.flush()
    del Y
    with pytest.raises(DataValidationError, match="y_velocity"):
        convert.verify_case(src, out / "stable_0")


def test_two_dimensional_fields_are_accepted(tmp_path):
    raw_dir = tmp_path / "raw"
    blood = {k: v[..., 0].clone() for k, v in make_blood(5).items()}  # (T, N)
    write_case(raw_dir, "d", blood)
    convert.convert_case(raw_dir / "d", tmp_path / "out")
    assert ProcessedCase(tmp_path / "out" / "d").solution.shape == (T, N, 4)
