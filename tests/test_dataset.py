"""Tests of the operator normalization (normalize.py), the datasets (dataset.py)
and the outlet flow check (outlets.py).

The normalization and dataset tests use two converted synthetic tubes with
geometry features (from test_geometry), split one for training and one for
testing. The outlet check uses a hand-built case with three flat caps and a
known flow through each.
"""

import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from src.data import cli, splits
from src.data.dataset import FullField, SampledPoints
from src.data.normalize import (PRESSURE_REFERENCE_PA, Normalization, normalization_path,
                                periodic_time_features)
from src.data.outlets import cap_flow, decide, flow_split_order, outlet_report, summarize, write_report
from src.data.processed import ProcessedCase

from conftest import torch, write_case
from test_geometry import blood_from_points, tube, tube_volume, write_wall_files

pytest.importorskip("scipy")


@pytest.fixture
def featured(tmp_path):
    """Two tube cases with geometry features and a split: stable_0 train, stable_1 test."""
    root = tmp_path / "aneug_flow"
    pos, faces, nt = tube()
    for i, fs in ((0, (0.4, 0.6)), (1, (0.3, 0.7))):
        d = write_case(root / "transient_data", f"stable_{i}", blood_from_points(tube_volume(i), seed=i),
                       flow_split=f"{fs[0]} {fs[1]}\n")
        write_wall_files(d, pos, faces, nt, flow_split=fs)
    assert cli.main(["convert", "--root", str(root), "--all"]) == 0
    assert cli.main(["features", "--root", str(root), "--all"]) == 0
    out = root / "processed_data"
    path = splits.create_split(out, val=0.0, test=0.5, seed=0)
    split = splits.Split.load(path)
    return out, split, path


def test_normalization_constants(featured):
    out, split, _ = featured
    norm = Normalization.from_split(split, out)
    (train,) = split.ids("train")
    c = ProcessedCase(out / train)
    sol = np.asarray(c.solution, dtype=np.float64)
    assert np.isclose(norm.velocity_scale, np.sqrt((sol[..., :3] ** 2).mean()), rtol=1e-5)
    assert np.isclose(norm.pressure_scale, np.sqrt(((sol[..., 3] - PRESSURE_REFERENCE_PA) ** 2).mean()), rtol=1e-5)
    assert np.isclose(norm.coordinate_scale, np.abs(np.asarray(c.coordinates)).max())
    g = np.asarray(c.geometry_features, dtype=np.float64)
    assert np.allclose(norm.geometry_scale, np.sqrt((g**2).mean(0)), rtol=1e-5)
    # One training case: the flow split cannot vary, so it is centred but not scaled.
    assert norm.flow_split_mean == pytest.approx(c.flow_split) and norm.flow_split_std == [1.0, 1.0]
    assert norm.input_channels[:3] == ["x", "y", "z"] and len(norm.input_channels) == 8


def test_normalization_round_trip(featured, tmp_path):
    out, split, _ = featured
    norm = Normalization.from_split(split, out)
    path = norm.save(tmp_path / "n.json")
    again = Normalization.load(path)
    assert again == norm
    y = np.array([[0.1, -0.2, 0.3, PRESSURE_REFERENCE_PA], [1.0, 0.0, -1.0, 101000.0]], dtype=np.float32)
    z = norm.targets(y)
    assert z[0, 3] == 0  # zero gauge pressure stays zero
    assert np.isclose(z[0, 0] / z[0, 1], -0.5)  # one velocity scale: direction is kept
    assert np.allclose(norm.physical(z, absolute_pressure=True), y, rtol=1e-6)
    assert np.allclose(norm.physical(z)[:, 3], y[:, 3] - PRESSURE_REFERENCE_PA, atol=1e-2)
    with pytest.raises(ValueError):
        norm.flow_split([0.5])


def test_time_encoding():
    ph = Normalization.phase(np.arange(80), 80)
    assert ph[0] == 0 and ph[-1] == pytest.approx(79 / 80) and ph.max() < 1
    f = periodic_time_features(np.array([0.0, 0.25, 1.0]), harmonics=2)
    assert f.shape == (3, 4)
    assert np.allclose(f[0], f[2], atol=1e-6)  # periodic
    assert np.allclose(f[1, [0, 2]], [1, 0], atol=1e-6)


def test_sampled_points(featured):
    out, split, _ = featured
    norm = Normalization.from_split(split, out)
    ds = SampledPoints(split, "train", out, norm, points=64, seed=3)
    c = ProcessedCase(out / ds.case_ids[0])
    assert len(ds) == c.num_snapshots
    item = ds[2]
    assert item["inputs"].shape == (64, 8) and item["targets"].shape == (1, 64, 4)
    assert item["flow_split"].shape == (2,) and item["time"].tolist() == [2 / c.num_snapshots]
    pts = item["points"]
    assert (np.diff(pts) > 0).all()
    assert np.allclose(item["targets"][0], norm.targets(c.solution[2, pts]))
    assert np.allclose(item["inputs"], norm.point_inputs(c.coordinates[pts], c.geometry_features[pts]))
    # Same (seed, epoch, index) -> same points; a new epoch draws new ones.
    assert (ds[2]["points"] == pts).all()
    ds.set_epoch(1)
    assert not (ds[2]["points"] == pts).all()
    # Several snapshots at the same points, the first being the item's own.
    multi = SampledPoints(split, "train", out, norm, points=32, snapshots=3)
    m = multi[1]
    assert m["targets"].shape == (3, 32, 4) and m["snapshots"][0] == 1 and len(set(m["snapshots"])) == 3
    for k, t in enumerate(m["snapshots"]):
        assert np.allclose(m["targets"][k], norm.targets(c.solution[t, m["points"]]))
    # More points than the case has: drawn with replacement.
    big = SampledPoints(split, "train", out, norm, points=c.num_points + 5)
    assert big[0]["inputs"].shape[0] == c.num_points + 5
    with pytest.raises(IndexError):
        ds[len(ds)]


def test_sampled_points_with_dataloader(featured):
    out, split, _ = featured
    norm = Normalization.from_split(split, out)
    ds = SampledPoints(split, "train", out, norm, points=16)
    ds[0]  # opens a case
    clone = pickle.loads(pickle.dumps(ds))  # what a worker process receives
    assert len(clone._open) == 0 and (clone[0]["points"] == ds[0]["points"]).all()
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=3, shuffle=True)))
    assert tuple(batch["inputs"].shape) == (3, 16, 8) and tuple(batch["targets"].shape) == (3, 1, 16, 4)
    assert batch["inputs"].dtype == torch.float32


def test_full_field_covers_every_point_once(featured):
    out, split, _ = featured
    norm = Normalization.from_split(split, out)
    ds = FullField(split, "test", out, norm, chunk=100, snapshots=[0, 3])
    c = ProcessedCase(out / ds.case_ids[0])
    seen = {0: [], 3: []}
    for item in torch.utils.data.DataLoader(ds, batch_size=None):
        t = int(item["snapshots"][0])
        seen[t].append(item["points"].numpy())
        assert np.allclose(item["targets"][0].numpy(), norm.targets(c.solution[t, item["points"].numpy()]))
    for t in seen:
        assert (np.concatenate(seen[t]) == np.arange(c.num_points)).all()
    with pytest.raises(ValueError):
        FullField(split, "test", out, norm, snapshots=[c.num_snapshots])


def test_dataset_rejects_other_geometry_channels(featured):
    out, split, _ = featured
    norm = Normalization.from_split(split, out)
    norm.geometry_channels = norm.geometry_channels[::-1]
    with pytest.raises(ValueError, match="geometry channels"):
        SampledPoints(split, "train", out, norm)


def test_swapped_flow_split_is_reversed_everywhere(featured):
    out, split, _ = featured
    (train,) = split.ids("train")
    (test,) = split.ids("test")
    plain = Normalization.from_split(split, out)
    norm = Normalization.from_split(split, out, swapped=[train, test], unreliable=["stable_99"])
    c = ProcessedCase(out / train)
    assert norm.case_flow_split(c) == c.flow_split[::-1] and plain.case_flow_split(c) == c.flow_split
    assert norm.flow_split_mean == pytest.approx(plain.flow_split_mean[::-1])
    assert norm.flow_split_unreliable == ["stable_99"]
    assert Normalization.load(norm.save(out / "n.json")).flow_split_swapped == sorted([train, test])
    t = ProcessedCase(out / test)
    item = FullField(split, "test", out, norm, chunk=10_000)[0]
    assert np.allclose(item["flow_split"], norm.flow_split(t.flow_split[::-1]))


def test_decide_rule():
    assert decide(0.60, [0.40, 0.60]) == "swap"
    assert decide(0.60, [0.60, 0.40]) == "keep"
    assert decide(0.52, [0.47, 0.53]) == "keep"  # too close to 0.5 to swap
    assert decide(0.56, [0.45, 0.55]) == "keep"  # file value within the margin
    assert decide(0.93, [0.71, 0.29]) == "unreliable"
    assert decide(float("nan"), [0.4, 0.6]) == "unreliable"


def write_outlets_csv(path, rows):
    write_report([{"case_id": c, "measured_fraction_outlet3": f3, "file_flow_split": f"{a} {1 - a}"}
                  for c, f3, a in rows], path)


def test_normalization_command(featured, capsys):
    out, split, split_path = featured
    args = ["normalization", "--out-dir", str(out), "--split", str(split_path)]
    assert cli.main(args) == 1  # no outlets report yet
    (train,) = split.ids("train")
    write_outlets_csv(out / "outlets.csv", [(train, 0.62, 0.4), ("stable_1", 0.93, 0.7)])
    assert flow_split_order(out / "outlets.csv")["swapped"] == [train]
    assert cli.main(args) == 0
    path = normalization_path(split_path)
    assert path.name == "default.normalization.json" and path.is_file()
    norm = Normalization.load(path)
    assert norm.flow_split_swapped == [train] and norm.flow_split_unreliable == ["stable_1"]
    assert cli.main(args) == 1  # exists
    assert cli.main(args + ["--overwrite", "--ignore-outlets"]) == 0
    assert Normalization.load(path).flow_split_swapped == []


# --- outlet flow -------------------------------------------------------------


def three_cap_case(f3=0.3, file_split=(0.7, 0.3), T=3):
    """An inlet cap at z=0 (fluid above) and outlet caps at x=+-0.05 (fluid inside), radius 2 mm.

    Plug flow: speed 1 m/s into the inlet, and out of the outlets at speeds
    giving fractions f3 and 1 - f3. Points are denser near each rim, as in a
    boundary layer, so an unweighted average would be biased.
    """
    rng = np.random.default_rng(0)
    r = 0.002
    caps = {"geodesic_inlet": (np.zeros(3), np.array([0, 0, 1.0])),
            "geodesic_outlet3": (np.array([0.05, 0, 0.01]), np.array([1.0, 0, 0])),
            "geodesic_outlet4": (np.array([-0.05, 0, 0.01]), np.array([-1.0, 0, 0]))}
    speed = {"geodesic_inlet": 1.0, "geodesic_outlet3": f3, "geodesic_outlet4": 1 - f3}
    xs, us, gs, sources = [], [], [], {}
    names = list(caps) + ["wall_distance", "geodesic_aneurysm"]
    for k, (name, (c, n)) in enumerate(caps.items()):
        e1 = np.cross(n, [0, 1.0, 0] if abs(n[1]) < 0.9 else [1.0, 0, 0]); e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        m = 4000
        rad = r * np.concatenate([np.sqrt(rng.uniform(0, 1, m // 2)), 1 - 0.1 * rng.uniform(0, 1, m // 2)])
        th = rng.uniform(0, 2 * np.pi, m)
        depth = rng.uniform(0, 0.003, m)  # into the fluid
        inward = -n if name != "geodesic_inlet" else n
        p = c + np.outer(rad * np.cos(th), e1) + np.outer(rad * np.sin(th), e2) + np.outer(depth, inward)
        # Velocity along the axis: into the fluid at the inlet, out of it at the outlets.
        u = np.outer(np.full(m, speed[name]), n if name == "geodesic_inlet" else n)
        g = np.full((m, len(names)), 0.05)
        g[:, k] = depth
        xs.append(p); us.append(u); gs.append(g)
        sources[name] = {"centroid": c.tolist(), "normal": (-n).tolist(), "radius": r, "slab": 1e-4}
    x, u, g = np.concatenate(xs), np.concatenate(us), np.concatenate(gs)
    sol = np.repeat(np.concatenate([u, np.full((len(x), 1), PRESSURE_REFERENCE_PA)], 1)[None], T, axis=0)
    return SimpleNamespace(case_id="synthetic", num_snapshots=T, coordinates=x, solution=sol,
                           geometry_features=g, geometry_channels=names, flow_split=list(file_split),
                           geometry={"checks": {"geodesic_sources": sources}})


def test_cap_flow_measures_known_flows():
    case = three_cap_case()
    q = cap_flow(case)
    area = np.pi * 0.002**2
    assert np.allclose(q["geodesic_inlet"], area, rtol=0.02)  # into the fluid: positive
    assert np.allclose(q["geodesic_outlet3"], -0.3 * area, rtol=0.02)  # out of it: negative
    assert np.allclose(q["geodesic_outlet4"], -0.7 * area, rtol=0.02)
    row = outlet_report(case)
    assert row["measured_fraction_outlet3"] == pytest.approx(0.3, abs=0.01)
    assert row["balance"] == pytest.approx(1.0, abs=0.03)
    assert row["verdict"] == "swapped" and row["decision"] == "swap"
    assert row["inlet_mean_velocity_ms"] == pytest.approx(1.0, rel=0.02)
    assert outlet_report(three_cap_case(file_split=(0.3, 0.7)))["verdict"] == "file order"
    assert outlet_report(three_cap_case(file_split=(0.5, 0.5)))["verdict"] == "ambiguous"
    s = summarize([row, outlet_report(three_cap_case(file_split=(0.3, 0.7)))])
    assert s["verdicts"] == {"swapped": 1, "file order": 1} and s["decisions"] == {"swap": 1, "keep": 1}
