"""Tests of the geometry features stage (geometry.py, ``aneug features``).

A synthetic straight tube stands in for a vessel: the wall is a triangulated
cylinder whose bottom rim is the inlet (node type 2), whose top rim is split
into two outlets (types 3 and 4), with an "aneurysm" patch (type 1) on the
side and one orphan node. Along such a tube the distance from the inlet,
along the wall or through the fluid, is the height z, which the tests use
as ground truth.

Covers: the point-triangle closest point against brute force, the exact
surface projection (including the candidate-doubling path), segment-wall
crossing, the fluid graph refusing to join two touching tubes, the volume
geodesics (bounded below by the straight-line distance to the cap plane),
the full command on a converted case, rerun skipping, the checks that fail a
case (wrong inlet, mismatched files) and warn (permuted dataset geodesics),
index reporting, and ProcessedCase access including wall_to_volume.
"""

import json
import logging

import numpy as np
import pytest

from src.data import cli, geometry, layout
from src.data.processed import ProcessedCase, pooled_geometry_stats

from conftest import make_blood, torch, write_case

pytest.importorskip("scipy")

R, L, NT, NZ = 0.002, 0.02, 32, 21


def tube():
    th = 2 * np.pi * np.arange(NT) / NT
    z = np.linspace(0, L, NZ)
    zz, tt = np.meshgrid(z, th, indexing="ij")
    pos = np.stack([R * np.cos(tt), R * np.sin(tt), zz], -1).reshape(-1, 3)
    faces = []
    for i in range(NZ - 1):
        for j in range(NT):
            a, b = i * NT + j, i * NT + (j + 1) % NT
            c, d = a + NT, b + NT
            faces += [[a, b, d], [a, d, c]]
    node_type = np.zeros(len(pos), dtype=np.int64)
    iz = np.repeat(np.arange(NZ), NT)
    it = np.tile(np.arange(NT), NZ)
    node_type[iz == 0] = 2
    node_type[(iz == NZ - 1) & (it < NT // 2)] = 3
    node_type[(iz == NZ - 1) & (it >= NT // 2)] = 4
    node_type[(iz >= 8) & (iz <= 12) & (it >= 2) & (it <= 6)] = 1
    # One orphan node, referenced by no face.
    pos = np.vstack([pos, [[0.0, 0.0, L / 2]]])
    node_type = np.append(node_type, 0)
    return pos.astype(np.float32), np.array(faces), node_type


def write_wall_files(case_dir, pos, faces, node_type, *, permute_geo=False, flow_split=(0.4, 0.6)):
    p64 = pos.astype(np.float64)
    geo = np.stack([geometry.surface_geodesic(p64, faces, node_type == t) for t in (2, 3, 4, 1)], 1)
    geo[~np.isfinite(geo)] = 0.0
    norm = geo / geo.max(0)
    ds = norm[:, [1, 0, 2]] if permute_geo else norm[:, :3]
    orphan = np.zeros(len(pos), dtype=bool)
    orphan[-1] = True
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    reg = {"pos": t(pos), "faces": t(faces), "node_type": t(node_type),
           "flow_split": t(np.array(flow_split, np.float32)), "aneurysm_type": torch.tensor(0),
           "geo_dist": t(geo[:, 3].astype(np.float32))}
    graph = {"pos": t((p64 - p64.mean(0)).astype(np.float32)), "faces": t(faces), "node_type": t(node_type),
             "flow_split": t(np.array(flow_split, np.float32)), "geo_dist": t(ds.astype(np.float32)),
             "global_geo_dist": t(norm[:, 3].astype(np.float32)), "orphan_mask": t(orphan)}
    torch.save(reg, case_dir / layout.WALL_REGISTRATION_FILE)
    torch.save(graph, case_dir / layout.WALL_GRAPH_FILE)
    rim = pos[node_type == 2][::4]
    rows = "\n".join(",".join(f"{v:.9g}" for v in r) for r in np.vstack([rim, [[0, 0, 0]]]))
    (case_dir / layout.INLET_FILE).write_text("x,y,z\n" + rows + "\n")


def tube_volume(seed=0, n=400):
    """Points inside the tube plus the wall nodes themselves."""
    rng = np.random.default_rng(seed)
    r = R * np.sqrt(rng.uniform(0, 0.8, n))
    th = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(0, L, n)
    inside = np.stack([r * np.cos(th), r * np.sin(th), z], -1)
    pos, faces, _ = tube()
    return np.vstack([inside, pos[: NZ * NT]]).astype(np.float32)


def blood_from_points(points, T=4, seed=0):
    blood = make_blood(seed, T=T, N=len(points))
    for j, a in enumerate("xyz"):
        blood[f"{a}_coordinate"] = torch.from_numpy(np.broadcast_to(points[:, j], (T, len(points))).copy()[..., None])
    return blood


@pytest.fixture
def tube_root(tmp_path):
    root = tmp_path / "aneug_flow"
    raw_dir = root / "transient_data"
    pos, faces, nt = tube()
    for i, kw in ((0, {}), (1, {"permute_geo": True})):
        d = write_case(raw_dir, f"stable_{i}", blood_from_points(tube_volume(i), seed=i))
        write_wall_files(d, pos, faces, nt, **kw)
    return root


def run(*argv):
    return cli.main([str(a) for a in argv])


# --- geometry primitives -----------------------------------------------------


def test_closest_point_matches_brute_force():
    rng = np.random.default_rng(1)
    a, b, c = (rng.normal(size=(200, 3)) for _ in range(3))
    p = rng.normal(size=(200, 3)) * 2
    q, bary = geometry.closest_point_on_triangles(p, a, b, c)
    assert np.allclose(bary.sum(1), 1) and (bary >= -1e-12).all()
    assert np.allclose(q, bary[:, :1] * a + bary[:, 1:2] * b + bary[:, 2:] * c)
    # Dense barycentric sampling: no sample may be closer than our answer.
    s = np.linspace(0, 1, 151)
    u, v = np.meshgrid(s, s)
    keep = u + v <= 1
    u, v = u[keep], v[keep]
    w = 1 - u - v
    samples = w[None, :, None] * a[:, None] + u[None, :, None] * b[:, None] + v[None, :, None] * c[:, None]
    d_samp = np.linalg.norm(samples - p[:, None], axis=2).min(1)
    d = np.linalg.norm(q - p, axis=1)
    assert (d <= d_samp + 1e-12).all()
    assert np.allclose(d, d_samp, atol=0.05)


def test_projection_is_exact_even_with_few_candidates():
    pos, faces, _ = tube()
    rng = np.random.default_rng(2)
    pts = np.concatenate([tube_volume(3, 150), rng.uniform(-0.004, 0.024, (50, 3))])
    p64 = pos.astype(np.float64)
    tri = p64[faces]
    q, _ = geometry.closest_point_on_triangles(pts[:, None].astype(np.float64), tri[None, :, 0], tri[None, :, 1], tri[None, :, 2])
    brute = np.linalg.norm(q - pts[:, None], axis=2).min(1)
    for k in (1, 16):
        d, f, bary = geometry.project_to_surface(pts, pos, faces, k=k)
        np.testing.assert_allclose(d, brute, rtol=0, atol=1e-12)
        q2 = (bary[..., None] * p64[faces[f]]).sum(1)
        np.testing.assert_allclose(np.linalg.norm(q2 - pts, axis=1), d, atol=1e-12)


def test_segment_crossing():
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2]])
    a = np.array([[0.2, 0.2, -1], [0.2, 0.2, 0], [0.9, 0.9, -1], [0.2, 0.2, 0.5], [0.1, 0.1, 0]])
    b = np.array([[0.2, 0.2, 1], [0.2, 0.2, 1], [0.9, 0.9, 1], [0.3, 0.3, 2], [0.3, 0.1, 0]])
    # crosses / starts on the wall / misses / misses / lies in the wall
    assert geometry.segments_cross_wall(a, b, pos, faces, k=1).tolist() == [True, False, False, False, False]


def test_graph_does_not_jump_between_touching_tubes():
    """Two parallel tubes 0.2 mm apart: no graph edge may join them."""
    pos, faces, _ = tube()
    pos, faces = pos[:-1].astype(np.float64), faces
    shift = np.array([2 * R + 2e-4, 0, 0])
    both_pos = np.vstack([pos, pos + shift])
    both_faces = np.vstack([faces, faces + len(pos)])
    vol_a = tube_volume(5, 300).astype(np.float64)
    pts = np.vstack([vol_a, vol_a + shift])
    tube_id = np.repeat([0, 1], len(vol_a))
    d, f, b = geometry.project_to_surface(pts, both_pos, both_faces)
    _, vn, agreement = geometry.fluid_normals(both_pos, both_faces, pts, d, f, b)
    assert agreement == 1.0
    edges, _, _, info = geometry.volume_graph(pts, d, both_pos, both_faces, vn, k=16)
    assert info["edges_removed_midpoint_outside"] > 0
    assert (tube_id[edges[:, 0]] == tube_id[edges[:, 1]]).all()
    # Without the checks the naive neighbour graph would have joined them.
    from scipy.spatial import cKDTree

    _, nn = cKDTree(pts).query(pts, k=17)
    assert (tube_id[nn] != tube_id[:, None]).any()


def test_surface_geodesic_from_inlet_rim_is_height():
    pos, faces, nt = tube()
    g = geometry.surface_geodesic(pos.astype(np.float64), faces, nt == 2)
    np.testing.assert_allclose(g[:-1], pos[:-1, 2], atol=1e-9)
    assert np.isinf(g[-1])  # orphan


# --- the stage ---------------------------------------------------------------


def test_features_command(tube_root, capsys, caplog):
    assert run("convert", "--root", tube_root, "--all") == 0
    assert run("features", "--root", tube_root, "--all") == 0
    out = tube_root / "processed_data"
    case = ProcessedCase(out / "stable_0")
    assert case.has_geometry
    f = np.asarray(case.geometry_features)
    x = np.asarray(case.coordinates, dtype=np.float64)
    assert f.shape == (len(x), 5) and f.dtype == np.float32
    assert case.geometry_channels == list(geometry.FEATURE_CHANNELS)

    # Wall distance: between the facet distance and the circle distance.
    r = np.hypot(x[:, 0], x[:, 1])
    assert (f[:, 0] <= R - r + 1e-7).all()
    assert (f[:, 0] >= R * np.cos(np.pi / NT) - r - 1e-7).all()
    wall_pts = np.isclose(r, R, rtol=1e-5)
    assert wall_pts.sum() == NZ * NT and np.abs(f[wall_pts, 0]).max() < 1e-7
    # Volume geodesic from the inlet cap of a straight tube is the height z:
    # never below it (straight-line bound), and graph paths only a little longer.
    spacing = case.geometry["checks"]["volume_graph"]["median_nearest_neighbour_distance"]
    assert (f[:, 1] >= x[:, 2] - 1e-7).all()
    assert (f[:, 1] <= 1.15 * x[:, 2] + 2 * spacing).all()
    # The whole inlet cross-section starts near 0, not only its rim.
    centre_bottom = (x[:, 2] < 0.05 * L) & (r < 0.5 * R)
    assert centre_bottom.any() and (f[centre_bottom, 1] < 0.1 * L).all()
    # Outlet geodesics shrink towards the top.
    assert np.corrcoef(f[:, 2], x[:, 2])[0, 1] < -0.9
    assert np.corrcoef(f[:, 3], x[:, 2])[0, 1] < -0.9
    # Points whose closest wall is the sac are at most their wall distance from it.
    mesh = case.wall_mesh()
    sac_share = case.wall_to_volume((mesh["node_type"] == 1).astype(float))
    on_sac = sac_share >= 0.5
    assert on_sac.any() and (f[on_sac, 4] <= f[on_sac, 0] + 1e-7).all()
    assert (f[~on_sac, 4] > 0).all()

    # Surface geodesics on the wall nodes (for the label checks) are exact on a tube.
    np.testing.assert_allclose(mesh["surface_geodesic"][:-1, 0], mesh["pos"][:-1, 2], atol=1e-6)
    assert np.isnan(mesh["surface_geodesic"][-1]).all()  # orphan
    # wall_to_volume carries wall fields into the volume.
    np.testing.assert_allclose(case.wall_to_volume(np.nan_to_num(mesh["surface_geodesic"]))[:, 0], x[:, 2], atol=1e-6)
    idx = np.array([0, 5, 9])
    assert case.wall_to_volume(mesh["pos"], idx).shape == (3, 3)
    # Normals point into the fluid (towards the axis).
    radial = mesh["pos"][:-1, :2] / np.linalg.norm(mesh["pos"][:-1, :2], axis=1, keepdims=True)
    assert (np.einsum("ij,ij->i", mesh["fluid_vertex_normals"][:-1, :2], -radial) > 0.9).all()

    md = case.geometry
    assert md["warnings"] == []
    g = md["checks"]["volume_graph"]
    assert g["edges_removed_crossing_wall"] == 0 and g["edges_removed_midpoint_outside"] == 0
    assert md["checks"]["wall_orientation_agreement"] == 1.0
    for c in ("geodesic_inlet", "geodesic_outlet3", "geodesic_outlet4"):
        src = md["checks"]["geodesic_sources"][c]
        assert src["seeds"] > 0 and src["unreachable"] == 0 and src["below_cap_plane_distance"] == 0
    assert md["checks"]["dataset_geo_dist_best_column"] == [0, 1, 2]
    assert md["checks"]["boundaries"]["inlet"]["nodes"] == NT
    for name in (layout.WALL_REGISTRATION_FILE, layout.WALL_GRAPH_FILE, layout.INLET_FILE):
        assert (case.path / layout.WALL_SUBDIR / name).is_file()

    # Permuted dataset geodesics only warn.
    other = ProcessedCase(out / "stable_1")
    assert other.has_geometry
    assert any("do not match" in w for w in other.geometry["warnings"])

    pooled = pooled_geometry_stats([case, other])
    assert pooled["wall_distance"]["count"] == len(x) + other.num_points

    # Rerun skips; index reports.
    caplog.clear()
    caplog.set_level(logging.INFO)
    assert run("features", "--root", tube_root, "stable_0") == 0
    assert "stable_0 skipped" in caplog.text
    capsys.readouterr()
    assert run("index", "--root", tube_root) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["num_with_geometry"] == 2
    assert summary["geometry_with_warnings"] == ["stable_1"]
    assert summary["geometry_failed"] == []


def test_wrong_inlet_fails_and_leaves_no_features(tube_root, capsys):
    raw_case = tube_root / "transient_data" / "stable_0"
    pos, faces, nt = tube()
    top = pos[nt == 3][::4]
    (raw_case / layout.INLET_FILE).write_text("x,y,z\n" + "\n".join(",".join(map(str, r)) for r in top) + "\n")
    run("convert", "--root", tube_root, "stable_0")
    assert run("features", "--root", tube_root, "stable_0") == 1
    out = tube_root / "processed_data"
    assert not layout.has_geometry(out / "stable_0")
    rec = json.loads(geometry.failure_path(out, "stable_0").read_text())
    assert "not closest to node type 2" in rec["error"]
    capsys.readouterr()
    assert run("index", "--root", tube_root) == 1
    assert json.loads(capsys.readouterr().out)["geometry_failed"] == ["stable_0"]

    # Fixing the input and rerunning clears the failure record.
    write_wall_files(raw_case, pos, faces, nt)
    assert run("features", "--root", tube_root, "stable_0") == 0
    assert not geometry.failure_path(out, "stable_0").exists()


def test_mismatched_wall_files_fail(tube_root):
    raw_case = tube_root / "transient_data" / "stable_0"
    graph = torch.load(raw_case / layout.WALL_GRAPH_FILE)
    graph["pos"] = graph["pos"] + 1e-3
    torch.save(graph, raw_case / layout.WALL_GRAPH_FILE)
    run("convert", "--root", tube_root, "stable_0")
    with pytest.raises(geometry.DataValidationError, match="centred registration pos"):
        geometry.features_case(tube_root / "processed_data" / "stable_0", raw_case)


def test_features_need_a_converted_case(tube_root):
    with pytest.raises(FileNotFoundError, match="aneug convert"):
        geometry.features_case(tube_root / "processed_data" / "stable_0", tube_root / "transient_data" / "stable_0")
