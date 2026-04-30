"""PLY / PCD / GLB / OBJ exporters.

PLY and PCD are pure-Python and always available. GLB and OBJ require a
ball-pivoting reconstruction pass and therefore Open3D — they raise a
descriptive ImportError if Open3D is missing.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def write_ply(
    path: Path,
    positions: np.ndarray,
    colors: np.ndarray | None = None,
) -> Path:
    """Write an ASCII PLY round-trippable with ironengine_sim.assets.point_cloud.load_ply."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = positions.shape[0]
    has_rgb = colors is not None
    lines: list[str] = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_rgb:
        lines += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    lines.append("end_header")
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
        if has_rgb:
            rgb = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
            for i in range(n):
                p, c = positions[i], rgb[i]
                fh.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        else:
            for i in range(n):
                p = positions[i]
                fh.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    return path


def write_pcd(
    path: Path,
    positions: np.ndarray,
    colors: np.ndarray | None = None,
) -> Path:
    """Write an ASCII PCD compatible with ironengine_sim.assets.point_cloud.load_pcd."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = positions.shape[0]
    has_rgb = colors is not None
    fields = ["x", "y", "z"] + (["rgb"] if has_rgb else [])
    sizes = [4, 4, 4] + ([4] if has_rgb else [])
    types = ["F", "F", "F"] + (["F"] if has_rgb else [])
    counts = [1, 1, 1] + ([1] if has_rgb else [])
    header = [
        "# .PCD v0.7 — IronEngine-3DCreator",
        "VERSION 0.7",
        f"FIELDS {' '.join(fields)}",
        f"SIZE {' '.join(map(str, sizes))}",
        f"TYPE {' '.join(types)}",
        f"COUNT {' '.join(map(str, counts))}",
        f"WIDTH {n}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {n}",
        "DATA ascii",
    ]
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(header) + "\n")
        if has_rgb:
            rgb_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint32)
            packed = (rgb_u8[:, 0] << 16) | (rgb_u8[:, 1] << 8) | rgb_u8[:, 2]
            for i in range(n):
                p = positions[i]
                # PCL-style: float32 reinterpretation of the packed integer
                f = struct.unpack("<f", struct.pack("<I", int(packed[i])))[0]
                fh.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {f:.6f}\n")
        else:
            for i in range(n):
                p = positions[i]
                fh.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    return path


def _require_open3d():
    try:
        import open3d as o3d  # type: ignore
        return o3d
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Mesh export (GLB/OBJ) requires Open3D. Install with: "
            "`conda run -n IronEngineWorld pip install open3d`."
        ) from e


def _reconstruct_to_mesh(positions: np.ndarray, colors: np.ndarray | None):
    o3d = _require_open3d()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions.astype(np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1).astype(np.float64))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    radii = o3d.utility.DoubleVector([0.005, 0.01, 0.02, 0.04])
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
    mesh.compute_vertex_normals()
    return mesh


def write_glb(
    path: Path,
    positions: np.ndarray,
    colors: np.ndarray | None = None,
) -> Path:
    o3d = _require_open3d()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = _reconstruct_to_mesh(positions, colors)
    # Open3D writes glTF/GLB based on suffix.
    o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False)
    return path


def write_obj(
    path: Path,
    positions: np.ndarray,
    colors: np.ndarray | None = None,
) -> Path:
    o3d = _require_open3d()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = _reconstruct_to_mesh(positions, colors)
    o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=True)
    return path


def export(
    path: Path,
    positions: np.ndarray,
    colors: np.ndarray | None = None,
    *,
    fmt: str | None = None,
) -> Path:
    """Dispatch by extension or explicit `fmt` (one of ply, pcd, glb, obj)."""
    path = Path(path)
    fmt = (fmt or path.suffix.lstrip(".")).lower()
    if fmt == "ply":
        return write_ply(path, positions, colors)
    if fmt == "pcd":
        return write_pcd(path, positions, colors)
    if fmt == "glb":
        return write_glb(path, positions, colors)
    if fmt == "obj":
        return write_obj(path, positions, colors)
    raise ValueError(f"unknown export format: {fmt!r}")
