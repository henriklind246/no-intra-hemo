"""Geometry features for each volume point of a processed case (``aneug features``).

For every point of ``coordinates.npy`` this writes ``geometry_features.npy``
(N, 5) float32, in metres, channels in this order:

    wall_distance       Euclidean distance to the vessel wall: the exact
                        distance to the closest point on the triangulated
                        wall surface (wall_mesh_registration.pt)
    geodesic_inlet      geodesic distance through the fluid from the inlet
                        cross-section (the cap closing the rim of wall
                        node_type 2)
    geodesic_outlet3    the same from the outlet cap of node_type 3
    geodesic_outlet4    the same from the outlet cap of node_type 4
    geodesic_aneurysm   the same from the aneurysm sac wall (node_type 1)

Model C uses the first four channels; the aneurysm distance is extra.

How the geodesics are computed. The volume points have no mesh connectivity,
so a graph is built: each point is joined to its ``GRAPH_K`` (32) nearest
neighbours, and an edge is dropped if it leaves the fluid. Edges no longer
than the sum of their end points' wall distances provably stay inside and are
kept; the rest (near the wall) are dropped if they pass through a wall
triangle or their midpoint lies outside the wall (judged with wall normals
oriented into the fluid). This stops paths jumping between vessels that
nearly touch. Dijkstra then runs from all seeds of a source at once, each seed
starting at its straight-line distance to the source:

    inlet / outlets  the cap plane is fitted to the open rim of the wall mesh
                     with that node type; seeds are the points inside the rim
                     and within one local point spacing of the plane, so the
                     whole cross-section starts at ~0, not only its edge
    aneurysm         seeds are the points whose closest wall point is on the
                     sac, starting at their wall distance

Graph paths are slightly longer than true distances: on a synthetic
850k-point tube with boundary-layer-like point stacks, 0.6% at the median and
2.5% at the 95th percentile with 32 neighbours (16 gives 3.5% and 5%). A path
can never be shorter than the straight line to the cap plane; this is
checked. The dataset's own geodesics (``wall_graph_static.pt``) are surface
distances divided by a per-case maximum, so they are used only for checks:
surface geodesics recomputed in metres on the wall mesh must line up with
them, which confirms the node-type labels.

Node types, as identified from stable_0: 0 vessel wall, 1 aneurysm sac,
2 inlet rim, 3 and 4 the two outlet rims. Which outlet each flow-split value
belongs to is not known yet, so the outlet channels are named by node type.

Checks per case, recorded in geometry.json. These fail the case:
    * the registration and graph files disagree on faces or node types, or
      the graph positions are not the registration positions minus their mean;
    * node type 1, 2, 3 or 4 is missing;
    * the wall does not span the same bounding box as the volume points
      (different coordinate frames);
    * the points of inlet_centroids.csv are not closest to the type-2 nodes
      (so type 2 is not the inlet);
    * a wall-triangle vertex is unreachable from a source on the surface;
    * no volume points lie on a cap, or none have the sac as closest wall;
    * more than 0.1% of the volume points are not connected to a source.
      Fewer are filled from their nearest connected neighbour (its value plus
      the distance to it) and counted in ``checks.geodesic_sources``; above
      0.001% this also gives a warning. A few such points are normal: in
      almost every case two or three points, apparently just outside the
      triangulated wall, lose all their edges to the inside/outside test.
These give warnings only:
    * the recomputed geodesics, normalized by their maximum, correlate below
      0.99 with the dataset's normalized ones, or the best-matching dataset
      channel is not the expected one;
    * the flow split differs between the wall files and flowsplit_ratio.txt;
    * the wall triangle winding is inconsistent, a cap needed the fallback
      seeds, or a path came out shorter than the straight line to its cap.
Also recorded: graph edge counts and removals, seeds per source, and, for
points on the wall, the ratio of volume to surface geodesic.

Other outputs, so later features never need the raw files (which the scratch
purge will delete):
    wall_projection.npz  ``face`` (N,) int32 and ``barycentric`` (N, 3)
                         float32: each point's closest wall triangle and its
                         weights. ``ProcessedCase.wall_to_volume`` uses them to
                         carry any per-wall-node field into the volume.
    wall/wall_mesh.npz   wall ``pos``, ``faces``, ``node_type``,
                         ``orphan_mask``, ``surface_geodesic`` (M, 4) in
                         metres, and ``fluid_vertex_normals`` (unit normals
                         pointing into the fluid).
    wall/                copies of wall_mesh_registration.pt,
                         wall_graph_static.pt (normals, Laplacian
                         eigenvectors, the dataset's geodesics) and
                         inlet_centroids.csv.

Files are written under temporary names and renamed into place;
geometry.json is written last and marks the stage complete, so an
interrupted run is redone on the next one. The solution files are never
touched.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from . import layout, raw
from .provenance import provenance
from .raw import DataValidationError
from .stats import ChannelStats

log = logging.getLogger(__name__)

FEATURE_CHANNELS = (
    "wall_distance",
    "geodesic_inlet",
    "geodesic_outlet3",
    "geodesic_outlet4",
    "geodesic_aneurysm",
)
# Wall node type each geodesic channel is measured from.
GEODESIC_SOURCES = {
    "geodesic_inlet": 2,
    "geodesic_outlet3": 3,
    "geodesic_outlet4": 4,
    "geodesic_aneurysm": 1,
}
NODE_TYPE_NAMES = {0: "wall", 1: "aneurysm", 2: "inlet", 3: "outlet3", 4: "outlet4"}
# wall_graph_static.pt geo_dist column -> source node type (from stable_0).
DATASET_GEO_DIST_SOURCES = (2, 3, 4)

POS_ATOL = 1e-6          # m; graph pos vs registration pos minus mean
BBOX_RTOL = 0.01         # fraction of the volume extent
CORRELATION_WARN = 0.99
GRAPH_K = 32             # neighbours per volume point in the geodesic graph (16: ~3.5% longer paths, 32: ~0.6%)
CROSS_TOL = 1e-7         # m; how far inside a segment a wall hit must be to count
UNREACHABLE_MAX = 1e-3   # fraction of points the graph may leave unconnected
UNREACHABLE_WARN = 1e-5  # above this fraction the filled points give a warning


@dataclass
class FeaturesResult:
    case_id: str
    status: str  # "computed" | "skipped"
    seconds: float = 0.0
    num_warnings: int = 0


# --- surface geometry ------------------------------------------------------


def mesh_edges(faces: np.ndarray) -> np.ndarray:
    """Unique undirected edges (E, 2) of a triangle mesh."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def surface_geodesic(pos: np.ndarray, faces: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Shortest-path distance along mesh edges from the nearest source node.

    ``sources`` is a boolean mask or index array of wall nodes. Returns (M,)
    float64 in the units of ``pos``; inf for nodes not connected to a source.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    pos = np.asarray(pos, dtype=np.float64)
    src = np.flatnonzero(sources) if np.asarray(sources).dtype == bool else np.asarray(sources)
    if src.size == 0:
        raise ValueError("no source nodes")
    e = mesh_edges(faces)
    w = np.linalg.norm(pos[e[:, 0]] - pos[e[:, 1]], axis=1)
    m = len(pos)
    graph = coo_matrix((w, (e[:, 0], e[:, 1])), shape=(m, m)).tocsr()
    return dijkstra(graph, directed=False, indices=src, min_only=True)


def _dot(a, b):
    return np.einsum("...i,...i->...", a, b)


def closest_point_on_triangles(p, a, b, c) -> Tuple[np.ndarray, np.ndarray]:
    """Closest point to ``p`` on triangles ``abc`` (Ericson, Real-Time Collision Detection 5.1.5).

    All inputs broadcast to (..., 3). Returns (closest point, barycentric
    weights for a, b, c), both (..., 3).
    """
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    d1, d2 = _dot(ab, ap), _dot(ac, ap)
    d3, d4 = _dot(ab, bp), _dot(ac, bp)
    d5, d6 = _dot(ab, cp), _dot(ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    def div(n, d):
        return np.divide(n, d, out=np.zeros(np.broadcast(n, d).shape), where=d != 0)

    shape = d1.shape
    u = np.empty(shape)
    v = np.empty(shape)
    w = np.empty(shape)

    # Face region first; each later region has higher precedence and overrides.
    denom = va + vb + vc
    v[...] = div(vb, denom)
    w[...] = div(vc, denom)
    u[...] = 1.0 - v - w
    degenerate = denom == 0
    u[degenerate], v[degenerate], w[degenerate] = 1.0, 0.0, 0.0

    def assign(mask, uu, vv, ww):
        u[mask] = uu[mask] if isinstance(uu, np.ndarray) else uu
        v[mask] = vv[mask] if isinstance(vv, np.ndarray) else vv
        w[mask] = ww[mask] if isinstance(ww, np.ndarray) else ww

    # Edge BC
    t = div(d4 - d3, (d4 - d3) + (d5 - d6))
    assign((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), 0.0, 1.0 - t, t)
    # Edge AC
    t = div(d2, d2 - d6)
    assign((vb <= 0) & (d2 >= 0) & (d6 <= 0), 1.0 - t, 0.0, t)
    # Vertex C
    assign((d6 >= 0) & (d5 <= d6), 0.0, 0.0, 1.0)
    # Edge AB
    t = div(d1, d1 - d3)
    assign((vc <= 0) & (d1 >= 0) & (d3 <= 0), 1.0 - t, t, 0.0)
    # Vertex B
    assign((d3 >= 0) & (d4 <= d3), 0.0, 1.0, 0.0)
    # Vertex A
    assign((d1 <= 0) & (d2 <= 0), 1.0, 0.0, 0.0)

    bary = np.stack([u, v, w], axis=-1)
    q = u[..., None] * a + v[..., None] * b + w[..., None] * c
    return q, bary


def project_to_surface(
    points: np.ndarray,
    pos: np.ndarray,
    faces: np.ndarray,
    *,
    k: int = 16,
    max_pairs: int = 1 << 21,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact closest point on a triangle mesh for every point.

    Candidates are the ``k`` triangles with the nearest centroids. A triangle
    outside that set is at least (k-th centroid distance - largest
    centroid-to-vertex distance) away, so points whose best distance exceeds
    that bound are searched again with twice as many candidates. The result
    is therefore exact, not approximate.

    Returns (distance (N,), face index (N,), barycentric weights (N, 3)).
    """
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float64)
    tri = np.asarray(pos, dtype=np.float64)[faces]  # (F, 3, 3)
    cent = tri.mean(axis=1)
    rmax = float(np.linalg.norm(tri - cent[:, None], axis=2).max())
    tree = cKDTree(cent)
    nf = len(faces)

    n = len(points)
    dist = np.full(n, np.inf)
    face = np.zeros(n, dtype=np.int64)
    bary = np.zeros((n, 3))

    pending = np.arange(n)
    kk = min(k, nf)
    while pending.size:
        unsure = []
        step = max(1, max_pairs // kk)
        for s in range(0, pending.size, step):
            idx = pending[s : s + step]
            cd, ci = tree.query(points[idx], k=kk)
            cd = cd.reshape(len(idx), kk)
            ci = ci.reshape(len(idx), kk)
            p = points[idx][:, None, :]
            t = tri[ci]
            q, b = closest_point_on_triangles(p, t[..., 0, :], t[..., 1, :], t[..., 2, :])
            d = np.linalg.norm(q - p, axis=-1)
            j = d.argmin(axis=1)
            r = np.arange(len(idx))
            dist[idx] = d[r, j]
            face[idx] = ci[r, j]
            bary[idx] = b[r, j]
            if kk < nf:
                unsure.append(idx[dist[idx] > cd[:, -1] - rmax])
        pending = np.concatenate(unsure) if unsure else np.empty(0, dtype=np.int64)
        kk = min(2 * kk, nf)
    return dist, face, bary


def interpolate(values: np.ndarray, faces: np.ndarray, face: np.ndarray, bary: np.ndarray) -> np.ndarray:
    """Barycentric interpolation of per-node ``values`` (M, ...) at projected points."""
    v = np.asarray(values)[faces[face]]  # (N, 3, ...)
    w = bary.reshape(bary.shape + (1,) * (v.ndim - 2))
    return (v * w).sum(axis=1)


# --- volume geodesics --------------------------------------------------------


def boundary_nodes(faces: np.ndarray) -> np.ndarray:
    """Nodes on edges used by exactly one triangle: the open rims of the wall mesh."""
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    return np.unique(uniq[counts == 1])


def fluid_normals(
    pos: np.ndarray, faces: np.ndarray, points: np.ndarray, distance: np.ndarray, face: np.ndarray, bary: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Wall normals oriented into the fluid.

    Each triangle's orientation is voted on by the volume points that project
    onto it from off the wall; triangles without votes follow the majority
    (valid when the mesh winding is consistent, which ``agreement`` measures).
    Returns (face normals (F, 3), area-weighted vertex normals (M, 3),
    agreement = fraction of votes that match the majority winding).
    """
    pos = np.asarray(pos, dtype=np.float64)
    tri = pos[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # length = 2 * area
    q = (bary[:, :, None] * tri[face]).sum(axis=1)
    off = distance > CROSS_TOL
    s = np.sign(_dot(np.asarray(points, dtype=np.float64)[off] - q[off], n[face[off]]))
    votes = np.zeros(len(faces))
    np.add.at(votes, face[off], s)
    majority = 1.0 if s.sum() >= 0 else -1.0
    agreement = float((s == majority).mean()) if s.size else float("nan")
    sign = np.where(votes > 0, 1.0, np.where(votes < 0, -1.0, majority))
    n = n * sign[:, None]
    vn = np.zeros_like(pos)
    for j in range(3):
        np.add.at(vn, faces[:, j], n)
    norm = np.linalg.norm(vn, axis=1, keepdims=True)
    vn = np.divide(vn, norm, out=np.zeros_like(vn), where=norm > 0)
    fn = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    return fn, vn, agreement


def _segment_triangle_hits(a, b, v0, v1, v2, tol):
    """Moller-Trumbore: does segment ab cross triangle v0v1v2 strictly inside the segment?"""
    d = b - a
    e1, e2 = v1 - v0, v2 - v0
    h = np.cross(d, e2)
    det = _dot(e1, h)
    scale = np.linalg.norm(d, axis=-1) * np.linalg.norm(e1, axis=-1) * np.linalg.norm(e2, axis=-1)
    ok = np.abs(det) > 1e-12 * scale
    f = np.divide(1.0, det, out=np.zeros_like(det), where=ok)
    s = a - v0
    u = f * _dot(s, h)
    qv = np.cross(s, e1)
    v = f * _dot(d, qv)
    t = f * _dot(e2, qv)
    length = np.linalg.norm(d, axis=-1)
    return ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t * length > tol) & ((1 - t) * length > tol)


def segments_cross_wall(
    a: np.ndarray, b: np.ndarray, pos: np.ndarray, faces: np.ndarray, *, k: int = 16, max_pairs: int = 1 << 21,
    tol: float = None,
) -> np.ndarray:
    """True for segments that pass through a wall triangle (not just touch it at an end).

    Exact: a triangle that meets the segment has its centroid within (half
    the segment length + its circumradius about the centroid) of the
    segment's midpoint, and the candidate set is enlarged until it provably
    contains every such triangle.
    """
    from scipy.spatial import cKDTree

    tol = CROSS_TOL if tol is None else tol
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    tri = np.asarray(pos, dtype=np.float64)[faces]
    cent = tri.mean(axis=1)
    rmax = float(np.linalg.norm(tri - cent[:, None], axis=2).max())
    tree = cKDTree(cent)
    nf = len(faces)
    mid = 0.5 * (a + b)
    half = 0.5 * np.linalg.norm(b - a, axis=1)
    hit = np.zeros(len(a), dtype=bool)
    pending = np.arange(len(a))
    kk = min(k, nf)
    while pending.size:
        more = []
        step = max(1, max_pairs // kk)
        for s in range(0, pending.size, step):
            idx = pending[s : s + step]
            cd, ci = tree.query(mid[idx], k=kk)
            cd = cd.reshape(len(idx), kk)
            ci = ci.reshape(len(idx), kk)
            t = tri[ci]
            h = _segment_triangle_hits(a[idx, None], b[idx, None], t[..., 0, :], t[..., 1, :], t[..., 2, :], tol)
            hit[idx] = h.any(axis=1)
            if kk < nf:
                incomplete = ~hit[idx] & (cd[:, -1] <= half[idx] + rmax)
                more.append(idx[incomplete])
        pending = np.concatenate(more) if more else np.empty(0, dtype=np.int64)
        kk = min(2 * kk, nf)
    return hit


def volume_graph(
    points: np.ndarray, wall_distance: np.ndarray, pos: np.ndarray, faces: np.ndarray, vertex_normals: np.ndarray,
    *, k: int = GRAPH_K,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Edges of the k-nearest-neighbour graph of the volume points that stay inside the fluid.

    An edge is kept outright if it is no longer than the sum of its end
    points' wall distances: the two wall-free balls around the ends then
    cover it. Other edges (near the wall) are dropped if they pass through a
    wall triangle, or if their midpoint lies outside the wall, which catches
    edges between two facing walls across a gap (kissing vessels, a sac
    touching its parent artery).

    Returns (edges (E, 2), lengths (E,), nearest-neighbour distance per point (N,), info).
    """
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    k = min(k, n - 1)
    nd, nn = cKDTree(points).query(points, k=k + 1)
    spacing = nd[:, 1]
    i = np.repeat(np.arange(n), k + 1)
    j = nn.ravel()
    keep = i != j
    i, j = np.minimum(i[keep], j[keep]), np.maximum(i[keep], j[keep])
    code = np.unique(i.astype(np.int64) * n + j)
    edges = np.stack([code // n, code % n], axis=1)
    length = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    safe = length <= wall_distance[edges[:, 0]] + wall_distance[edges[:, 1]]
    check = np.flatnonzero(~safe)
    a, b = points[edges[check, 0]], points[edges[check, 1]]
    crosses = segments_cross_wall(a, b, pos, faces)
    mid = 0.5 * (a + b)
    _, mface, mbary = project_to_surface(mid, pos, faces)
    tri = np.asarray(pos, dtype=np.float64)[faces[mface]]
    q = (mbary[:, :, None] * tri).sum(axis=1)
    normal = interpolate(vertex_normals, faces, mface, mbary)
    outside = _dot(mid - q, normal) < -CROSS_TOL
    bad = np.zeros(len(edges), dtype=bool)
    bad[check] = crosses | outside
    info = {
        "k": int(k),
        "edges": int(len(edges)),
        "edges_checked_near_wall": int(check.size),
        "edges_removed_crossing_wall": int(crosses.sum()),
        "edges_removed_midpoint_outside": int((outside & ~crosses).sum()),
        "median_nearest_neighbour_distance": float(np.median(spacing)),
    }
    return edges[~bad], length[~bad], spacing, info


def cap_seeds(points: np.ndarray, rim: np.ndarray, spacing: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Volume points on (or just inside) the planar cap bounded by ``rim`` nodes.

    The cap plane is fitted to the rim. Seeds are points within the rim's
    radius of its centroid, measured in the plane, and within one local point
    spacing of the plane; their start distance is their distance to the plane.
    Returns (seed indices, start distances, info).
    """
    c = rim.mean(axis=0)
    _, sv, vt = np.linalg.svd(rim - c, full_matrices=False)
    normal = vt[-1]
    residual = float(np.abs((rim - c) @ normal).max())
    radius = float(np.linalg.norm((rim - c) - np.outer((rim - c) @ normal, normal), axis=1).max())
    rel = points - c
    pd = np.abs(rel @ normal)
    inplane = np.linalg.norm(rel - np.outer(rel @ normal, normal), axis=1)
    near = np.flatnonzero((inplane <= radius * (1 + 1e-6)) & (pd <= radius))
    info = {"centroid": [float(x) for x in c], "normal": [float(x) for x in normal], "radius": radius,
            "rim_plane_residual": residual}
    if near.size == 0:
        info["seeds"] = 0
        return near, pd[near], info
    slab = max(1.5 * residual, float(np.median(spacing[near])))
    seeds = near[pd[near] <= slab]
    if seeds.size == 0:
        seeds = near[np.argsort(pd[near])[:10]]
        info["fallback"] = "closest points to the cap plane"
    info["slab"] = slab
    info["seeds"] = int(seeds.size)
    return seeds, pd[seeds], info


def graph_distance(n: int, edges: np.ndarray, length: np.ndarray, seeds: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Shortest path length through the graph from any seed, starting from ``start`` at each seed."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    s = n  # virtual source joined to every seed
    rows = np.concatenate([edges[:, 0], np.full(len(seeds), s)])
    cols = np.concatenate([edges[:, 1], seeds])
    w = np.maximum(np.concatenate([length, start]), 1e-15)  # explicit zeros would drop edges
    g = coo_matrix((w, (rows, cols)), shape=(n + 1, n + 1)).tocsr()
    d = dijkstra(g, directed=False, indices=s)
    return d[:n]


# --- one case --------------------------------------------------------------


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def _flow_split(obj) -> Optional[list]:
    if obj is None:
        return None
    return [float(x) for x in np.asarray(raw.to_numpy(obj), dtype=np.float64).ravel()]


def _load_wall(raw_case: Path) -> Dict[str, Any]:
    reg = raw.torch_load(raw_case / layout.WALL_REGISTRATION_FILE)
    graph = raw.torch_load(raw_case / layout.WALL_GRAPH_FILE)
    for name, obj, keys in (
        (layout.WALL_REGISTRATION_FILE, reg, ("pos", "faces", "node_type")),
        (layout.WALL_GRAPH_FILE, graph, ("pos", "faces", "node_type", "geo_dist")),
    ):
        missing = [k for k in keys if k not in obj]
        if missing:
            raise DataValidationError(f"{name} lacks {missing}")
    return {"reg": reg, "graph": graph}


def compute_features(
    coordinates: np.ndarray,
    raw_case: Path,
    *,
    expected_flow_split: Optional[list] = None,
) -> Dict[str, Any]:
    """Compute the features and checks for one case. Nothing is written.

    Returns a dict with ``features`` (N, 5) float32, ``face``, ``barycentric``,
    the wall mesh arrays, nodal ``surface_geodesic`` (M, 4) (used for the label
    checks), ``fluid_vertex_normals``, ``checks`` and ``warnings``.
    """
    wall = _load_wall(raw_case)
    reg, graph = wall["reg"], wall["graph"]
    pos = np.asarray(raw.to_numpy(reg["pos"]), dtype=np.float64)
    faces = np.asarray(raw.to_numpy(reg["faces"]), dtype=np.int64)
    node_type = np.asarray(raw.to_numpy(reg["node_type"]), dtype=np.int64).ravel()
    m = len(pos)
    checks: Dict[str, Any] = {}
    warnings = []

    # Registration and graph describe the same mesh.
    gpos = np.asarray(raw.to_numpy(graph["pos"]), dtype=np.float64)
    gfaces = np.asarray(raw.to_numpy(graph["faces"]), dtype=np.int64)
    gtype = np.asarray(raw.to_numpy(graph["node_type"]), dtype=np.int64).ravel()
    if pos.shape != (m, 3) or gpos.shape != pos.shape or faces.ndim != 2 or faces.shape[1] != 3:
        raise DataValidationError(f"unexpected wall shapes: pos {pos.shape}, graph pos {gpos.shape}, faces {faces.shape}")
    if gfaces.shape != faces.shape or not np.array_equal(gfaces, faces):
        raise DataValidationError("wall registration and graph faces differ")
    if not np.array_equal(gtype, node_type):
        raise DataValidationError("wall registration and graph node types differ")
    if faces.min() < 0 or faces.max() >= m:
        raise DataValidationError("wall faces index outside the node array")
    pos_err = float(np.abs(gpos - (pos - pos.mean(axis=0))).max())
    checks["graph_pos_minus_centered_registration_pos_max"] = pos_err
    if pos_err > POS_ATOL:
        raise DataValidationError(f"graph pos is not the centred registration pos (max diff {pos_err:.3g} m)")

    counts = {int(t): int((node_type == t).sum()) for t in np.unique(node_type)}
    checks["node_type_counts"] = {NODE_TYPE_NAMES.get(t, str(t)): c for t, c in counts.items()}
    missing = [NODE_TYPE_NAMES[t] for t in (1, 2, 3, 4) if counts.get(t, 0) == 0]
    if missing:
        raise DataValidationError(f"wall node types missing: {missing}")

    # Same coordinate frame as the volume points.
    vol = np.asarray(coordinates, dtype=np.float64)
    vmin, vmax = vol.min(axis=0), vol.max(axis=0)
    extent = vmax - vmin
    used = np.unique(faces)
    bbox_err = np.maximum(np.abs(pos[used].min(axis=0) - vmin), np.abs(pos[used].max(axis=0) - vmax)) / extent
    checks["wall_vs_volume_bbox_rel_diff"] = [float(x) for x in bbox_err]
    if (bbox_err > BBOX_RTOL).any():
        raise DataValidationError(
            f"wall and volume bounding boxes differ by up to {bbox_err.max():.1%} of the extent; different frames?"
        )

    # The inlet file sits on the type-2 rim.
    inlet = np.loadtxt(raw_case / layout.INLET_FILE, delimiter=",", skiprows=1, ndmin=2)
    from scipy.spatial import cKDTree

    near = {}
    for t in (2, 3, 4):
        d, _ = cKDTree(pos[node_type == t]).query(inlet)
        near[NODE_TYPE_NAMES[t]] = float(d.mean())
    checks["inlet_csv_rows"] = int(len(inlet))
    checks["inlet_csv_mean_distance_to_node_type"] = near
    if min(near, key=near.get) != "inlet":
        raise DataValidationError(f"inlet_centroids.csv is not closest to node type 2: {near}")

    # Surface geodesics on the wall nodes, in metres: compared with the
    # dataset's own geodesics to confirm the node-type labels.
    names = list(GEODESIC_SOURCES)
    geo = np.stack([surface_geodesic(pos, faces, node_type == GEODESIC_SOURCES[c]) for c in names], axis=1)
    if not np.isfinite(geo[used]).all():
        raise DataValidationError("some wall-triangle vertices are not connected to every source")
    reachable = np.isfinite(geo).all(axis=1)
    geo[~reachable] = np.nan
    checks["wall_nodes_unreachable"] = int((~reachable).sum())

    # Compare with the dataset's own (normalized) geodesics.
    orphan = (np.asarray(raw.to_numpy(graph["orphan_mask"]), dtype=bool).ravel()
              if "orphan_mask" in graph else np.zeros(m, dtype=bool))
    ok = reachable & ~orphan
    ds = np.asarray(raw.to_numpy(graph["geo_dist"]), dtype=np.float64).reshape(m, -1)
    ours = geo[:, [names.index(c) for c in ("geodesic_inlet", "geodesic_outlet3", "geodesic_outlet4")]]
    ours_n = ours[ok] / ours[ok].max(axis=0)
    corr = [[_pearson(ours_n[:, i], ds[ok, j]) for j in range(ds.shape[1])] for i in range(ours_n.shape[1])]
    checks["dataset_geo_dist_correlation"] = corr
    expected = list(range(min(3, ds.shape[1])))
    best = [int(np.nanargmax(row)) for row in corr]
    checks["dataset_geo_dist_best_column"] = best
    if best[: len(expected)] != expected:
        warnings.append(f"dataset geo_dist columns do not match node types 2/3/4 in order (best columns {best})")
    for i in expected:
        if corr[i][i] < CORRELATION_WARN:
            warnings.append(f"{names[i]} correlates {corr[i][i]:.3f} with dataset geo_dist column {i}")
    an = geo[:, names.index("geodesic_aneurysm")]
    if "global_geo_dist" in graph:
        g = np.asarray(raw.to_numpy(graph["global_geo_dist"]), dtype=np.float64).ravel()
        r = _pearson(an[ok] / an[ok].max(), g[ok])
        checks["dataset_global_geo_dist_correlation"] = r
        if r < CORRELATION_WARN:
            warnings.append(f"geodesic_aneurysm correlates {r:.3f} with dataset global_geo_dist")
    if "geo_dist" in reg:
        g = np.asarray(raw.to_numpy(reg["geo_dist"]), dtype=np.float64).ravel()
        if g.shape == (m,):
            sel = ok & (g > 0)
            ratio = an[sel] / g[sel]
            checks["aneurysm_geodesic_over_dataset_metres"] = {
                "median": float(np.median(ratio)), "p05": float(np.percentile(ratio, 5)),
                "p95": float(np.percentile(ratio, 95)),
            }
            r = _pearson(an[ok], g[ok])
            checks["dataset_registration_geo_dist_correlation"] = r
            if r < CORRELATION_WARN:
                warnings.append(f"geodesic_aneurysm correlates {r:.3f} with registration geo_dist (m)")

    # Flow split consistency.
    fs = {"registration": _flow_split(reg.get("flow_split")), "graph": _flow_split(graph.get("flow_split")),
          "flowsplit_ratio.txt": expected_flow_split}
    checks["flow_split"] = fs
    known = [v for v in fs.values() if v is not None]
    if any(len(v) != len(known[0]) or not np.allclose(v, known[0], atol=1e-6) for v in known[1:]):
        warnings.append(f"flow split differs between sources: {fs}")

    # Boundary sizes, for later matching flow-split values to outlets.
    rims = {}
    for t in (2, 3, 4):
        p = pos[node_type == t]
        c = p.mean(axis=0)
        rims[NODE_TYPE_NAMES[t]] = {"nodes": int(len(p)), "centroid": [float(x) for x in c],
                                    "mean_radius": float(np.linalg.norm(p - c, axis=1).mean())}
    checks["boundaries"] = rims

    # Volume points: exact wall distance and closest wall point.
    n = len(vol)
    wall_distance, face, bary = project_to_surface(vol, pos, faces)
    d_node, _ = cKDTree(vol).query(pos[used])
    checks["wall_node_to_nearest_volume_point"] = {"median": float(np.median(d_node)), "max": float(d_node.max())}

    # Graph through the fluid.
    _, vertex_normals, agreement = fluid_normals(pos, faces, vol, wall_distance, face, bary)
    checks["wall_orientation_agreement"] = agreement
    if not agreement >= 0.99:
        warnings.append(f"wall triangle winding is inconsistent ({agreement:.3f} agreement); "
                        "the inside/outside test for near-wall edges is less reliable")
    edges, length, spacing, ginfo = volume_graph(vol, wall_distance, pos, faces, vertex_normals)
    checks["volume_graph"] = ginfo

    # Volume geodesics from each source.
    rim_nodes = boundary_nodes(faces)
    on_wall = wall_distance <= 1e-6
    vol_geo = np.empty((n, len(names)))
    sources = {}
    for col, c in enumerate(names):
        t = GEODESIC_SOURCES[c]
        if t == 1:
            sac = interpolate((node_type == 1).astype(np.float64), faces, face, bary) >= 0.5
            seeds = np.flatnonzero(sac)
            start = wall_distance[seeds]
            info = {"seeds": int(seeds.size), "start": "distance to the sac wall"}
            if seeds.size == 0:
                raise DataValidationError("no volume point has the aneurysm wall as its closest wall")
        else:
            rim = rim_nodes[node_type[rim_nodes] == t]
            if len(rim) < 3:
                rim = np.flatnonzero(node_type == t)
                warnings.append(f"{c}: no open mesh rim with node type {t}; fitted the cap to all type-{t} nodes")
            seeds, start, info = cap_seeds(vol, pos[rim], spacing)
            info["rim_nodes"] = int(len(rim))
            if seeds.size == 0:
                raise DataValidationError(f"no volume points on the {NODE_TYPE_NAMES[t]} cap")
            if "fallback" in info:
                warnings.append(f"{c}: no volume points within one spacing of the cap plane; "
                                f"used the {seeds.size} closest")
        d = graph_distance(n, edges, length, seeds, start)
        lost = ~np.isfinite(d)
        info["unreachable"] = int(lost.sum())
        if lost.mean() > UNREACHABLE_MAX:
            raise DataValidationError(f"{c}: {lost.sum()} of {n} points are not connected to the source")
        if lost.any():
            dd, ii = cKDTree(vol[~lost]).query(vol[lost])
            d[lost] = d[~lost][ii] + dd
            if lost.mean() > UNREACHABLE_WARN:
                warnings.append(f"{c}: {lost.sum()} unconnected points took their nearest connected neighbour's value")
        if t != 1:
            cn = np.asarray(info["normal"])
            below = d < np.abs((vol - np.asarray(info["centroid"])) @ cn) - 1e-9
            info["below_cap_plane_distance"] = int(below.sum())
            if below.any():
                warnings.append(f"{c}: {below.sum()} points closer by path than by straight line to the cap plane")
        surf = interpolate(np.nan_to_num(geo[:, col]), faces, face, bary)
        sel = on_wall & (surf > 0.1 * np.nanmax(geo[:, col]))
        if sel.any():
            ratio = d[sel] / surf[sel]
            info["on_wall_volume_over_surface"] = {"median": float(np.median(ratio)),
                                                    "p95": float(np.percentile(ratio, 95))}
        vol_geo[:, col] = d
        sources[c] = info
    checks["geodesic_sources"] = sources

    features = np.concatenate([wall_distance[:, None], vol_geo], axis=1).astype(np.float32)
    if not np.isfinite(features).all():
        raise DataValidationError("non-finite geometry features")

    return {
        "features": features,
        "face": face.astype(np.int32),
        "barycentric": bary.astype(np.float32),
        "pos": pos.astype(np.float32),
        "faces": faces.astype(np.int32),
        "node_type": node_type.astype(np.int8),
        "orphan_mask": orphan,
        "surface_geodesic": geo.astype(np.float32),
        "fluid_vertex_normals": vertex_normals.astype(np.float32),
        "aneurysm_type": int(np.asarray(raw.to_numpy(reg["aneurysm_type"])).ravel()[0]) if "aneurysm_type" in reg else None,
        "checks": checks,
        "warnings": warnings,
    }


def _replace_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def _replace_npz(path: Path, **arrays) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def _replace_copy(src: Path, dst: Path) -> None:
    tmp = dst.with_name(f".{dst.name}.tmp-{uuid.uuid4().hex[:8]}")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def features_case(case_dir: Path, raw_case: Path, *, overwrite: bool = False) -> FeaturesResult:
    """Compute and write the geometry stage for one processed case."""
    case_dir = Path(case_dir)
    raw_case = Path(raw_case)
    cid = case_dir.name
    if not layout.is_complete_case(case_dir):
        raise FileNotFoundError(f"{case_dir} is not a complete processed case; run `aneug convert {cid}` first")
    if layout.has_geometry(case_dir) and not overwrite:
        return FeaturesResult(cid, "skipped")
    for name in layout.REQUIRED_FOR_FEATURES:
        if not (raw_case / name).is_file():
            raise FileNotFoundError(f"{raw_case / name} not found")

    t0 = time.time()
    with open(case_dir / layout.METADATA_FILE) as f:
        md = json.load(f)
    coords = np.load(case_dir / layout.COORDINATES_FILE, mmap_mode="r")
    res = compute_features(coords, raw_case, expected_flow_split=md.get("flow_split"))

    marker = case_dir / layout.GEOMETRY_METADATA_FILE
    if marker.exists():
        marker.unlink()
    wall_dir = case_dir / layout.WALL_SUBDIR
    wall_dir.mkdir(exist_ok=True)
    _replace_npy(case_dir / layout.GEOMETRY_FEATURES_FILE, res["features"])
    _replace_npz(case_dir / layout.WALL_PROJECTION_FILE, face=res["face"], barycentric=res["barycentric"])
    _replace_npz(
        wall_dir / layout.WALL_MESH_FILE,
        pos=res["pos"], faces=res["faces"], node_type=res["node_type"],
        orphan_mask=res["orphan_mask"], surface_geodesic=res["surface_geodesic"],
        surface_geodesic_channels=np.array(list(GEODESIC_SOURCES)),
        fluid_vertex_normals=res["fluid_vertex_normals"],
    )
    copied = {}
    for name in (layout.WALL_REGISTRATION_FILE, layout.WALL_GRAPH_FILE, layout.INLET_FILE):
        _replace_copy(raw_case / name, wall_dir / name)
        copied[name] = (wall_dir / name).stat().st_size

    stats = ChannelStats(len(FEATURE_CHANNELS))
    stats.update(res["features"].astype(np.float64))
    meta = {
        "geometry_version": layout.GEOMETRY_VERSION,
        "case_id": cid,
        "num_points": int(len(res["features"])),
        "num_wall_nodes": int(len(res["pos"])),
        "num_wall_faces": int(len(res["faces"])),
        "units": "m",
        "feature_channels": list(FEATURE_CHANNELS),
        "geodesic_sources": {c: {"node_type": t, "name": NODE_TYPE_NAMES[t]} for c, t in GEODESIC_SOURCES.items()},
        "node_types": {str(t): n for t, n in NODE_TYPE_NAMES.items()},
        "method": {
            "wall_distance": "exact Euclidean distance to the closest point on the wall triangle mesh",
            "geodesic": f"Dijkstra through the fluid on the {GRAPH_K}-nearest-neighbour graph of the volume "
                        "points, without edges that leave the fluid; sources are the inlet and outlet caps "
                        "(fitted to the open mesh rims) and the aneurysm sac wall",
            "wall_mesh_source": layout.WALL_REGISTRATION_FILE,
        },
        "feature_stats": stats.to_dict(FEATURE_CHANNELS),
        "aneurysm_type": res["aneurysm_type"],
        "checks": res["checks"],
        "warnings": res["warnings"],
        "files": {
            layout.GEOMETRY_FEATURES_FILE: {"shape": list(res["features"].shape), "dtype": "float32"},
            layout.WALL_PROJECTION_FILE: ["face", "barycentric"],
            f"{layout.WALL_SUBDIR}/{layout.WALL_MESH_FILE}": ["pos", "faces", "node_type", "orphan_mask",
                                                             "surface_geodesic", "surface_geodesic_channels",
                                                             "fluid_vertex_normals"],
            "copied_raw_files": copied,
        },
        "source": {"case_dir": str(raw_case)},
        "provenance": provenance(),
    }
    tmp = marker.with_name(f".{marker.name}.tmp-{uuid.uuid4().hex[:8]}")
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2, allow_nan=True)
        f.write("\n")
    os.replace(tmp, marker)
    for w in res["warnings"]:
        log.warning("%s: %s", cid, w)
    return FeaturesResult(cid, "computed", time.time() - t0, len(res["warnings"]))


def failure_path(out_root: Path, case_id: str) -> Path:
    """Features failures go to _failures/features/, apart from conversion failures."""
    return Path(out_root) / layout.FAILURES_SUBDIR / "features" / f"{case_id}.json"


def record_failure(out_root: Path, case_id: str, exc: BaseException) -> Path:
    import traceback

    path = failure_path(out_root, case_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "case_id": case_id,
            "stage": "features",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__),
            "provenance": provenance(),
        }, f, indent=2)
        f.write("\n")
    return path


def clear_failure(out_root: Path, case_id: str) -> None:
    path = failure_path(out_root, case_id)
    if path.exists():
        path.unlink()
