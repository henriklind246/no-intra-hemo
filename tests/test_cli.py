"""End-to-end tests of the ``aneug`` commands (cli.py) on synthetic raw data.

Covers: scan's manifest and missing-file report; sharded convert processing
every case exactly once and recording failures; --cases-file subsets; index
progress reporting; verify --all; inspect's report, including the units hint;
reading processed cases and pooled statistics through ProcessedCase; and the
default dataset root.
"""

import csv
import json

import numpy as np
import pytest

from src.data import cli, layout
from src.data.processed import ProcessedCase, list_cases, pooled_solution_stats

from conftest import N, T, make_blood


def run(*argv):
    return cli.main([str(a) for a in argv])


def test_scan_writes_manifest_and_flags_missing_files(raw_root, capsys):
    assert run("scan", "--root", raw_root) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["num_cases"] == 4
    assert summary["num_convertible"] == 3
    assert summary["missing_files"]["blood_data.pt"] == ["stable_5"]
    rows = list(csv.DictReader(open(raw_root / "processed_data" / "raw_manifest.csv")))
    assert [r["case_id"] for r in rows] == ["stable_0", "stable_2", "stable_5", "stable_10"]
    assert [r["convertible"] for r in rows] == ["1", "1", "0", "1"]


def test_sharded_convert_covers_every_case_once_and_records_failures(raw_root):
    root = raw_root
    codes = [run("convert", "--root", root, "--all", "--num-shards", 3, "--shard-index", i)
             for i in range(3)]
    # stable_5 (no blood_data.pt) fails in exactly one shard.
    assert sorted(codes) == [0, 0, 1]
    out = root / "processed_data"
    assert list_cases(out) == ["stable_0", "stable_2", "stable_10"]
    failure = json.loads((out / "_failures" / "stable_5.json").read_text())
    assert failure["error_type"] == "FileNotFoundError"

    # Rerunning skips completed cases.
    assert run("convert", "--root", root, "stable_0", "stable_2") == 0


def test_cases_file_selects_subset(raw_root, tmp_path):
    ids = tmp_path / "ids.txt"
    ids.write_text("# pilot\nstable_10\n\nstable_0\n")
    assert run("convert", "--root", raw_root, "--cases-file", ids) == 0
    assert list_cases(raw_root / "processed_data") == ["stable_0", "stable_10"]


def test_index_reports_progress(raw_root, capsys):
    run("convert", "--root", raw_root, "stable_0", "stable_10")
    capsys.readouterr()
    assert run("index", "--root", raw_root) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["num_complete"] == 2
    assert summary["num_raw"] == 4
    assert summary["not_yet_processed"] == ["stable_2", "stable_5"]
    rows = list(csv.DictReader(open(raw_root / "processed_data" / "index.csv")))
    assert [r["case_id"] for r in rows] == ["stable_0", "stable_10"]
    assert int(rows[0]["solution_bytes"]) > T * N * 4 * 4


def test_verify_all(raw_root):
    run("convert", "--root", raw_root, "stable_0", "stable_2")
    assert run("verify", "--root", raw_root, "--all") == 0


def test_inspect_reports_structure_and_units_hint(raw_root, tmp_path):
    out = tmp_path / "inspect.json"
    assert run("inspect", "--root", raw_root, "stable_0", "-o", out) == 0
    report = json.loads(out.read_text())
    blood = report["blood_data.pt"]
    assert blood["coordinates_static"] is True
    assert blood["fields"]["x_velocity"]["shape"] == [T, N, 1]
    assert set(blood["fields"]["x_velocity"]["snapshots"]) == {"0", str(T // 2), str(T - 1)}
    assert report["flowsplit_ratio.txt"]["parsed"] == [0.4, 0.6]
    assert report["inlet_centroids.csv"]["text"].startswith("x,y,z")
    assert report["shape.obj"]["vertices"] == 3
    # OBJ spans 20-40 units vs ~0.04 m volume extent: ratio ~1000 suggests mm.
    assert min(report["shape.obj"]["extent_ratio_to_volume_coordinates"]) > 100


def test_reader_and_pooled_stats(raw_root):
    run("convert", "--root", raw_root, "--all")
    out = raw_root / "processed_data"
    cases = [ProcessedCase(out / c) for c in list_cases(out)]
    case = cases[0]
    idx = np.array([0, 5, 9])
    assert case.snapshot(2).shape == (N, 4)
    assert case.snapshot(2, idx).shape == (3, 4)
    assert case.coordinates[idx].shape == (3, 3)
    assert case.flow_split == [0.4, 0.6]

    pooled = pooled_solution_stats(cases)
    seeds = {"stable_0": 0, "stable_2": 2, "stable_10": 10}
    vx = np.concatenate([make_blood(seeds[c.case_id])["x_velocity"].numpy().ravel() for c in cases])
    assert pooled["x_velocity"]["count"] == vx.size
    assert pooled["x_velocity"]["mean"] == pytest.approx(vx.astype(np.float64).mean(), rel=1e-10)
    assert pooled["x_velocity"]["std"] == pytest.approx(vx.astype(np.float64).std(), rel=1e-8)


def test_default_root_uses_env(monkeypatch, tmp_path):
    monkeypatch.setenv(layout.ROOT_ENV_VAR, str(tmp_path))
    assert layout.default_root() == tmp_path
    monkeypatch.delenv(layout.ROOT_ENV_VAR)
    monkeypatch.setenv("USER", "someone")
    assert str(layout.default_root()) == "/scratch.global/someone/aneug_flow"
