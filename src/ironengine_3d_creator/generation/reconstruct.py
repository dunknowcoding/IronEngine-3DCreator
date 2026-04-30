"""Mesh reconstruction from a point cloud, with a small in-process cache.

Returns plain numpy arrays — positions (V, 3), indices (T*3,), normals (V, 3) —
which is what every renderer (ours, Sim's, Open3D's) needs at the boundary.

Uses Open3D's Poisson reconstruction by default, falling back to ball-pivot
with adaptive radii (chosen from the point cloud's nearest-neighbour spacing)
and as a last resort scipy ConvexHull. Sim's `build_point_cloud_reconstruct`
has buggy radii on multi-component shapes (it produces ~one component per
point) so we do **not** call it.

Cache key is `(id(positions), len(positions), method, radius)` so toggling
the mesh preview on/off in the viewport is instant after the first build.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

_log = logging.getLogger(__name__)


@dataclass
class ReconstructedMesh:
    positions: np.ndarray   # (V, 3) float32
    normals: np.ndarray     # (V, 3) float32
    indices: np.ndarray     # (T*3,) uint32
    source: str = ""        # "open3d_poisson" | "open3d_ball_pivot" | "convex_hull"


_cache: dict[tuple, ReconstructedMesh] = {}


def _cache_key(positions: np.ndarray, method: str, radius: float) -> tuple:
    return (id(positions), int(positions.shape[0]), method, round(radius, 4))


def reconstruct(
    positions: np.ndarray,
    *,
    radius: float = 0.0,
    method: str = "auto",
    use_cache: bool = True,
) -> ReconstructedMesh:
    """Triangulate `positions`.

    Args:
        positions: (N, 3) float32 point cloud.
        radius: ball-pivot radius. 0 means "auto" (computed from point spacing).
        method: "auto" | "poisson" | "ball_pivot" | "convex_hull".
        use_cache: reuse the previous mesh for the same (id, method, radius).
    """
    if use_cache:
        cached = _cache.get(_cache_key(positions, method, radius))
        if cached is not None:
            return cached

    mesh: Optional[ReconstructedMesh] = None
    # Ball-pivot first: it respects the cloud's actual openings (fence bars,
    # chair gaps, archway interior) rather than closing them with Poisson's
    # implicit watertight lid. Poisson is the fallback for clouds with very
    # uneven density that ball-pivot can't bridge.
    if method in ("auto", "ball_pivot"):
        mesh = _try_ball_pivot(positions, radius)
    if mesh is None and method in ("auto", "poisson"):
        mesh = _try_poisson(positions)
    if mesh is None and method in ("auto", "convex_hull"):
        mesh = _try_convex_hull(positions)
    if mesh is None:
        raise ImportError(
            "Mesh reconstruction needs `open3d` or `scipy`. Install with: "
            "`conda run -n IronEngineWorld pip install open3d`."
        )

    if use_cache:
        _cache[_cache_key(positions, method, radius)] = mesh
        if len(_cache) > 16:
            _cache.pop(next(iter(_cache)))
    return mesh


# ---------------------------------------------------------------- Poisson


def _try_poisson(positions: np.ndarray) -> Optional[ReconstructedMesh]:
    """Screened Poisson reconstruction. Robust, watertight, slower than
    ball-pivot but produces a single connected manifold rather than a
    pile of fragments. Best general-purpose choice."""
    o3d = _try_open3d()
    if o3d is None:
        return None
    pcd = _build_pcd_with_normals(positions, o3d)
    if pcd is None:
        return None
    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9, scale=1.1, linear_fit=False,
        )
        # Trim only the *very* sparse halo (Poisson extends the surface
        # beyond the cloud's true extent in low-density regions). Trimming
        # too aggressively at, say, the 4th percentile leaves visible holes
        # over the surface — 0.5% removes only the outer halo.
        densities = np.asarray(densities)
        if densities.size:
            threshold = np.quantile(densities, 0.005)
            verts_to_remove = densities < threshold
            mesh.remove_vertices_by_mask(verts_to_remove)
        # Stitch any small holes the trim may have introduced.
        mesh = _close_holes(mesh, o3d)
        # And drop dust — keep only triangle clusters that exceed 1% of total
        # surface area, which removes Poisson's stray tiny components without
        # eating the main body.
        mesh = _keep_large_clusters(mesh, o3d)
        mesh.compute_vertex_normals()
        return _to_arrays(mesh, "open3d_poisson")
    except Exception:
        _log.exception("Poisson reconstruction failed; will try ball-pivot")
        return None


def _close_holes(mesh, o3d):
    """Best-effort hole-fill via Open3D's tensor mesh API. Falls back to a
    no-op if the version of Open3D in use doesn't expose `fill_holes`."""
    try:
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        filled = tmesh.fill_holes(hole_size=0.05)
        return filled.to_legacy()
    except Exception:
        return mesh


def _keep_large_clusters(mesh, o3d):
    """Drop tiny disconnected components (Poisson dust) that the user wouldn't
    want to see. Keep clusters whose triangle count is >= 1% of the largest."""
    try:
        cluster_idx, cluster_n_tri, _ = mesh.cluster_connected_triangles()
        cluster_idx = np.asarray(cluster_idx)
        cluster_n_tri = np.asarray(cluster_n_tri)
        if cluster_n_tri.size == 0:
            return mesh
        keep_min = max(20, int(cluster_n_tri.max() * 0.01))
        small = (cluster_n_tri < keep_min)[cluster_idx]
        mesh.remove_triangles_by_mask(small)
        mesh.remove_unreferenced_vertices()
        return mesh
    except Exception:
        return mesh


# ---------------------------------------------------------------- ball-pivot


def _try_ball_pivot(positions: np.ndarray, radius: float) -> Optional[ReconstructedMesh]:
    o3d = _try_open3d()
    if o3d is None:
        return None
    pcd = _build_pcd_with_normals(positions, o3d)
    if pcd is None:
        return None
    if radius <= 0.0:
        # Use the average nearest-neighbour distance as a baseline radius.
        # A 2-3x spread of radii catches both dense and sparse regions.
        spacing = _avg_nn_distance(positions, sample=2048)
        radius = max(spacing * 1.5, 1e-3)
    radii = [radius * 0.5, radius, radius * 2.0, radius * 3.0]
    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii),
        )
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh = _keep_large_clusters(mesh, o3d)
        mesh.compute_vertex_normals()
        return _to_arrays(mesh, "open3d_ball_pivot")
    except Exception:
        _log.exception("ball-pivot failed")
        return None


# ---------------------------------------------------------------- convex hull


def _try_convex_hull(positions: np.ndarray) -> Optional[ReconstructedMesh]:
    try:
        from scipy.spatial import ConvexHull  # type: ignore
    except Exception:
        return None
    try:
        hull = ConvexHull(positions.astype(np.float64))
    except Exception:
        return None
    pos = np.asarray(positions[hull.vertices], dtype=np.float32)
    remap = {old: new for new, old in enumerate(hull.vertices)}
    idx = np.asarray(
        [[remap[v] for v in tri] for tri in hull.simplices], dtype=np.uint32
    ).ravel()
    nrm = _compute_vertex_normals(pos, idx)
    return ReconstructedMesh(positions=pos, normals=nrm, indices=idx, source="convex_hull")


# ---------------------------------------------------------------- helpers


def _try_open3d():
    try:
        import open3d as o3d  # type: ignore
        return o3d
    except Exception:
        return None


def _build_pcd_with_normals(positions: np.ndarray, o3d):
    """Build an Open3D PointCloud with consistently oriented normals.

    Consistent normals are critical for both Poisson (otherwise the surface
    inverts in patches) and for the half-Lambert shader in our viewport.
    """
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(positions.astype(np.float64))
        spacing = _avg_nn_distance(positions, sample=2048)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=max(spacing * 4, 0.01),
                max_nn=30,
            ),
        )
        try:
            pcd.orient_normals_consistent_tangent_plane(k=12)
        except Exception:
            # Fallback: orient normals away from the cloud centroid.
            pcd.orient_normals_towards_camera_location(
                np.array(positions.mean(axis=0), dtype=np.float64) + np.array([0, 5, 0]),
            )
        return pcd
    except Exception:
        _log.exception("Open3D PCD construction failed")
        return None


def _avg_nn_distance(positions: np.ndarray, sample: int = 2048) -> float:
    """Estimate point-to-nearest-neighbour distance from a random subset."""
    n = positions.shape[0]
    if n < 2:
        return 0.01
    idx = np.random.default_rng(0).choice(n, size=min(sample, n), replace=False)
    pts = positions[idx]
    # Brute force kNN inside the subset (small enough).
    diffs = pts[:, None, :] - pts[None, :, :]
    d2 = (diffs * diffs).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    nn = np.sqrt(d2.min(axis=1))
    return float(np.median(nn))


def _to_arrays(mesh, source: str) -> ReconstructedMesh:
    pos = np.asarray(mesh.vertices, dtype=np.float32)
    idx = np.asarray(mesh.triangles, dtype=np.uint32).ravel()
    if mesh.has_vertex_normals():
        nrm = np.asarray(mesh.vertex_normals, dtype=np.float32)
    else:
        nrm = _compute_vertex_normals(pos, idx)
    return ReconstructedMesh(positions=pos, normals=nrm, indices=idx, source=source)


def _compute_vertex_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        return np.zeros_like(positions)
    tri = indices.reshape(-1, 3)
    a = positions[tri[:, 0]]; b = positions[tri[:, 1]]; c = positions[tri[:, 2]]
    face_n = np.cross(b - a, c - a)
    out = np.zeros_like(positions)
    np.add.at(out, tri[:, 0], face_n)
    np.add.at(out, tri[:, 1], face_n)
    np.add.at(out, tri[:, 2], face_n)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    return (out / norm).astype(np.float32)


def clear_cache() -> None:
    _cache.clear()
