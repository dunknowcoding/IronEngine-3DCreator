"""PLY / PCD / GLB / OBJ exporters.

PLY and PCD are pure-Python/NumPy and always available. Spec-driven GLB/OBJ
export builds exact analytic meshes per primitive (generation.analytic_mesh)
with PBR materials, baked albedo textures, UVs, and COLOR_0 vertex colors;
point-cloud reconstruction (ball-pivot / Poisson via Open3D) is kept only as
a fallback for code-mode / freeform clouds, and raises a descriptive
ImportError if Open3D is missing.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

_log = logging.getLogger(__name__)

_PLY_FMT_RGB = ["%.6f", "%.6f", "%.6f", "%d", "%d", "%d"]
_PLY_FMT_XYZ = ["%.6f", "%.6f", "%.6f"]


def write_ply(
    path: Path,
    positions: np.ndarray,
    colors: np.ndarray | None = None,
    *,
    binary: bool = False,
) -> Path:
    """Write a PLY round-trippable with ironengine_sim.assets.point_cloud.load_ply.

    ASCII is the default because Sim's reader only parses ASCII (downstream
    contract, W15). The body is written with a single vectorized np.savetxt
    call instead of a per-point Python loop. Pass `binary=True` for a
    binary_little_endian file (~10x faster, ~3x smaller) when the consumer
    supports it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    n = positions.shape[0]
    has_rgb = colors is not None and len(colors)
    header = [
        "ply",
        f"format {'binary_little_endian' if binary else 'ascii'} 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_rgb:
        header += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    header.append("end_header")
    if binary:
        fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
        if has_rgb:
            fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        arr = np.empty(n, dtype=fields)
        arr["x"], arr["y"], arr["z"] = positions[:, 0], positions[:, 1], positions[:, 2]
        if has_rgb:
            rgb = np.clip(np.asarray(colors) * 255.0, 0, 255).astype(np.uint8).reshape(-1, 3)
            arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        with path.open("wb") as fh:
            fh.write(("\n".join(header) + "\n").encode("ascii"))
            arr.tofile(fh)
        return path
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(header) + "\n")
        if has_rgb:
            rgb = np.clip(np.asarray(colors) * 255.0, 0, 255).astype(np.uint8).reshape(-1, 3)
            body = np.column_stack([positions.astype(np.float64), rgb])
            np.savetxt(fh, body, fmt=_PLY_FMT_RGB)
        else:
            np.savetxt(fh, positions.astype(np.float64), fmt=_PLY_FMT_XYZ)
    return path


def write_pcd(
    path: Path,
    positions: np.ndarray,
    colors: np.ndarray | None = None,
) -> Path:
    """Write an ASCII PCD compatible with ironengine_sim.assets.point_cloud.load_pcd.

    The packed rgb channel is written as a plain integer-valued float (e.g.
    ``16711680.0``) — Sim's reader recovers it with ``int(float(tok))`` (W2).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    n = positions.shape[0]
    has_rgb = colors is not None and len(colors)
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
            rgb_u8 = np.clip(np.asarray(colors) * 255.0, 0, 255).astype(np.uint32).reshape(-1, 3)
            packed = (rgb_u8[:, 0] << 16) | (rgb_u8[:, 1] << 8) | rgb_u8[:, 2]
            body = np.column_stack([positions.astype(np.float64), packed.astype(np.float64)])
            np.savetxt(fh, body, fmt=["%.6f", "%.6f", "%.6f", "%.1f"])
        else:
            np.savetxt(fh, positions.astype(np.float64), fmt=_PLY_FMT_XYZ)
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
    """Triangulate via generation.reconstruct (auto-radius ball-pivot with
    oriented normals, Poisson fallback) and wrap the result as an Open3D
    TriangleMesh, transferring point colors to mesh vertices."""
    o3d = _require_open3d()
    from ..generation.reconstruct import reconstruct

    positions = np.asarray(positions, dtype=np.float32)
    rec = reconstruct(positions, use_cache=True)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(rec.positions.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(rec.indices.reshape(-1, 3).astype(np.int32))
    if rec.normals.size == rec.positions.size:
        mesh.vertex_normals = o3d.utility.Vector3dVector(rec.normals.astype(np.float64))
    if colors is not None and len(colors) and len(rec.positions):
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            _transfer_colors(positions, colors, rec.positions)
        )
    return mesh


def _transfer_colors(src_pos: np.ndarray, src_col: np.ndarray, dst_pos: np.ndarray) -> np.ndarray:
    """Map per-point colors onto mesh vertices by nearest neighbour."""
    src_col = np.clip(np.asarray(src_col, dtype=np.float64).reshape(-1, 3), 0, 1)
    if dst_pos.shape[0] == src_pos.shape[0]:
        # Ball-pivot keeps the input points 1:1 — no lookup needed.
        return src_col
    try:
        from scipy.spatial import cKDTree  # type: ignore
        _, nn = cKDTree(src_pos.astype(np.float64)).query(dst_pos.astype(np.float64), k=1)
        return src_col[nn]
    except Exception:
        # Chunked brute-force fallback when scipy is unavailable.
        out = np.empty((dst_pos.shape[0], 3), dtype=np.float64)
        src = src_pos.astype(np.float64)
        dst = dst_pos.astype(np.float64)
        for lo in range(0, dst.shape[0], 4096):
            chunk = dst[lo:lo + 4096]
            d2 = ((chunk[:, None, :] - src[None, :, :]) ** 2).sum(axis=2)
            out[lo:lo + 4096] = src_col[d2.argmin(axis=1)]
        return out


# ---------------------------------------------------------------------------
# Spec-driven analytic mesh export (F5)
# ---------------------------------------------------------------------------


def _build_parts(spec):
    """Analytic part meshes for a spec, or None when unavailable."""
    if spec is None or not getattr(spec, "primitives", None):
        return None
    try:
        from ..generation.analytic_mesh import build_spec_meshes
        parts = build_spec_meshes(spec)
        return parts or None
    except Exception:
        _log.exception("analytic mesh build failed; falling back to reconstruction")
        return None


def _part_vertex_colors(parts, positions, colors) -> list[np.ndarray]:
    """Per-part (V, 3) float64 colors in [0, 1] via 3D nearest-neighbour."""
    if colors is None or not len(colors) or not len(positions):
        grey = np.array([0.7, 0.7, 0.7])
        return [np.tile(grey, (p.vertices.shape[0], 1)) for p in parts]
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    colors = np.clip(np.asarray(colors, dtype=np.float64).reshape(-1, 3), 0.0, 1.0)
    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(positions)
        return [colors[tree.query(p.vertices.astype(np.float64), k=1, workers=-1)[1]] for p in parts]
    except Exception:
        out = []
        for p in parts:
            dst = p.vertices.astype(np.float64)
            cols = np.empty((dst.shape[0], 3), dtype=np.float64)
            for lo in range(0, dst.shape[0], 4096):
                chunk = dst[lo:lo + 4096]
                d2 = ((chunk[:, None, :] - positions[None, :, :]) ** 2).sum(axis=2)
                cols[lo:lo + 4096] = colors[d2.argmin(axis=1)]
            out.append(cols)
        return out


def _bake_uv_texture(uvs: np.ndarray, colors: np.ndarray, size: int):
    """Rasterize per-vertex colors onto the part's UV grid (nearest-KDTree fill).

    `uvs` are the *exported* (glTF-convention) coordinates: v = 0 is the top
    image row. Returns a PIL RGB image.
    """
    from PIL import Image  # type: ignore

    px = np.clip(np.round(uvs[:, 0] * (size - 1)).astype(np.int64), 0, size - 1)
    py = np.clip(np.round(uvs[:, 1] * (size - 1)).astype(np.int64), 0, size - 1)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=bool)
    rgb = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    img[py, px] = rgb
    mask[py, px] = True
    if not mask.all():
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            img[:] = rgb.mean(axis=0).astype(np.uint8) if len(rgb) else 128
        else:
            miss_y, miss_x = np.nonzero(~mask)
            try:
                from scipy.spatial import cKDTree  # type: ignore
                tree = cKDTree(np.column_stack([ys, xs]))
                _, nn = tree.query(np.column_stack([miss_y, miss_x]), k=1)
            except Exception:
                # Nearest-index fallback without scipy: brute-force in chunks.
                src = np.column_stack([ys, xs]).astype(np.float64)
                dst = np.column_stack([miss_y, miss_x]).astype(np.float64)
                nn = np.empty(len(dst), dtype=np.int64)
                for lo in range(0, len(dst), 4096):
                    chunk = dst[lo:lo + 4096]
                    d2 = ((chunk[:, None, :] - src[None, :, :]) ** 2).sum(axis=1)
                    nn[lo:lo + 4096] = d2.argmin(axis=1)
            img[miss_y, miss_x] = img[ys[nn], xs[nn]]
    return Image.fromarray(img, "RGB")


def _write_glb_scene(path: Path, parts, positions, colors, texture_size: int) -> None:
    """Write a GLB with one named node per part, PBR material + baked albedo
    texture + COLOR_0 vertex colors per part (F5/W3/W7)."""
    import trimesh  # type: ignore
    from trimesh.visual.material import PBRMaterial  # type: ignore
    from trimesh.visual.texture import TextureVisuals  # type: ignore

    from ..generation.materials import MATERIAL_PRESETS, default_preset

    vert_colors = _part_vertex_colors(parts, positions, colors)
    scene = trimesh.Scene()
    for part, vc in zip(parts, vert_colors):
        preset = MATERIAL_PRESETS.get(part.material, default_preset())
        mean_col = vc.mean(axis=0) if len(vc) else np.array([0.7, 0.7, 0.7])
        # glTF UV convention: v = 0 is the image top row. trimesh flips
        # uv[:, 1] on export, so bake against the flipped coordinates.
        uv_gltf = part.uvs.astype(np.float64).copy()
        uv_gltf[:, 1] = 1.0 - uv_gltf[:, 1]
        texture = _bake_uv_texture(uv_gltf, vc, texture_size)
        material = PBRMaterial(
            name=part.material,
            baseColorTexture=texture,
            baseColorFactor=np.append(np.clip(mean_col * 255.0, 0, 255), 255).astype(np.uint8),
            metallicFactor=float(preset["metallic"]),
            roughnessFactor=float(preset["roughness"]),
        )
        tmesh = trimesh.Trimesh(
            vertices=part.vertices,
            faces=part.faces,
            vertex_normals=part.normals,
            process=False,
        )
        visuals = TextureVisuals(uv=part.uvs, material=material)
        rgba = np.concatenate(
            [np.clip(vc * 255.0, 0, 255).astype(np.uint8),
             np.full((vc.shape[0], 1), 255, dtype=np.uint8)],
            axis=1,
        )
        # Kept alongside TEXCOORD_0: trimesh exports this as COLOR_0 (contract 2).
        visuals.vertex_attributes["color"] = rgba
        tmesh.visual = visuals
        scene.add_geometry(tmesh, node_name=part.label, geom_name=part.label)
    scene.export(str(path), file_type="glb")


def write_glb_parts(
    path: Path,
    parts,
    positions: np.ndarray | None = None,
    colors: np.ndarray | None = None,
    *,
    texture_size: int = 256,
) -> Path:
    """Write a GLB directly from pre-built analytic parts (one named node per
    part, per-part PBR materials, baked albedo textures, COLOR_0 colors).

    This is the export path for generators that build their own exact meshes
    instead of spec primitives — e.g. ``generation.soft_author`` cloth sheets,
    ropes, frangible vessels, and ragdoll body parts.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_glb_scene(path, list(parts), positions, colors, max(16, int(texture_size)))
    return path


def write_glb(
    path: Path,
    positions: np.ndarray,
    colors: np.ndarray | None = None,
    *,
    spec=None,
    texture_size: int = 256,
) -> Path:
    """Write a binary GLB.

    With a spec (and trimesh available) the export uses exact analytic meshes
    — one named node per primitive label with a PBR material, a baked albedo
    texture, UVs, and COLOR_0 vertex colors (F5). Without a spec it falls back
    to ball-pivot / Poisson reconstruction of the point cloud (code-mode /
    freeform clouds).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = _build_parts(spec)
    if parts is not None:
        try:
            _write_glb_scene(path, parts, positions, colors, max(16, int(texture_size)))
            return path
        except Exception:
            _log.exception("analytic GLB export failed; falling back to reconstruction")
    o3d = _require_open3d()
    mesh = _reconstruct_to_mesh(positions, colors)
    # Open3D 0.19 writes GLBs with a base64 data-URI buffer that ASSIMP
    # (its own reader, and SceneEditor's) rejects — prefer trimesh's binary
    # GLB writer when available, keeping vertex colors and normals.
    try:
        _write_glb_trimesh(path, mesh)
    except ImportError:
        o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False)
    return path


def _write_glb_trimesh(path: Path, mesh) -> None:
    import trimesh  # type: ignore

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    tmesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if mesh.has_vertex_colors():
        tmesh.visual.vertex_colors = (
            np.clip(np.asarray(mesh.vertex_colors) * 255.0, 0, 255).astype(np.uint8)
        )
    if mesh.has_vertex_normals():
        tmesh.vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    tmesh.export(str(path), file_type="glb")


def write_obj(
    path: Path,
    positions: np.ndarray,
    colors: np.ndarray | None = None,
    *,
    spec=None,
    albedo=None,
) -> Path:
    """Write an ASCII OBJ plus a sibling .mtl carrying the material albedo (W17).

    OBJ has no vertex-color channel in the downstream toolchain, so colors are
    reduced to per-part ``Kd`` entries in the .mtl; a warning is logged that
    per-point color variation does not survive this format.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = _build_parts(spec)
    mtl_path = path.with_suffix(".mtl")
    if parts is not None:
        vert_colors = _part_vertex_colors(parts, positions, colors)
        _write_obj_parts(path, mtl_path, parts, vert_colors)
        _log.warning(
            "OBJ export: per-vertex colors are not stored in OBJ; only per-part "
            "Kd colors in %s survive. Use GLB to keep full color detail.",
            mtl_path.name,
        )
        return path

    # Fallback: reconstructed mesh, single material.
    o3d = _require_open3d()
    mesh = _reconstruct_to_mesh(positions, colors)
    if albedo is None:
        if colors is not None and len(colors):
            albedo = np.asarray(colors, dtype=np.float64).reshape(-1, 3).mean(axis=0).tolist()
        else:
            albedo = [0.7, 0.7, 0.7]
    _write_obj_single(path, mtl_path, mesh, albedo)
    _log.warning(
        "OBJ export: per-vertex colors are not stored in OBJ; only the mean "
        "albedo in %s survives. Use GLB to keep full color detail.",
        mtl_path.name,
    )
    return path


def _write_obj_parts(path: Path, mtl_path: Path, parts, vert_colors) -> None:
    """One `o <label>` group + `usemtl` per part; .mtl with per-part Kd."""
    # First part + mean color per unique material name.
    mat_kd: dict[str, np.ndarray] = {}
    for part, vc in zip(parts, vert_colors):
        if part.material not in mat_kd:
            mat_kd[part.material] = vc.mean(axis=0) if len(vc) else np.array([0.7, 0.7, 0.7])
    with mtl_path.open("w", encoding="utf-8") as fh:
        fh.write("# IronEngine-3DCreator material file\n")
        for name, kd in mat_kd.items():
            fh.write(f"newmtl {name}\n")
            fh.write(f"Kd {kd[0]:.4f} {kd[1]:.4f} {kd[2]:.4f}\n")
            fh.write("Ka 0.0000 0.0000 0.0000\nKs 0.0400 0.0400 0.0400\nNs 32.0\n\n")
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"mtllib {mtl_path.name}\n")
        offset = 1  # OBJ indices are 1-based
        for part in parts:
            fh.write(f"o {part.label}\n")
            fh.write(f"usemtl {part.material}\n")
            for v in part.vertices:
                fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for n in part.normals:
                fh.write(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
            for tri in part.faces:
                a, b, c = int(tri[0]) + offset, int(tri[1]) + offset, int(tri[2]) + offset
                fh.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")
            offset += part.vertices.shape[0]


def _write_obj_single(path: Path, mtl_path: Path, mesh, albedo) -> None:
    """Fallback reconstructed-mesh OBJ with a single material."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    normals = (
        np.asarray(mesh.vertex_normals, dtype=np.float64)
        if mesh.has_vertex_normals()
        else np.zeros((0, 3))
    )
    with mtl_path.open("w", encoding="utf-8") as fh:
        fh.write("# IronEngine-3DCreator material file\n")
        fh.write("newmtl creator_material\n")
        fh.write(f"Kd {float(albedo[0]):.4f} {float(albedo[1]):.4f} {float(albedo[2]):.4f}\n")
        fh.write("Ka 0.0000 0.0000 0.0000\nKs 0.0400 0.0400 0.0400\nNs 32.0\n")
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"mtllib {mtl_path.name}\n")
        fh.write("o creator_model\nusemtl creator_material\n")
        for v in vertices:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for n in normals:
            fh.write(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
        has_n = normals.shape[0] == vertices.shape[0]
        for tri in faces:
            a, b, c = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
            if has_n:
                fh.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")
            else:
                fh.write(f"f {a} {b} {c}\n")


def export(
    path: Path,
    positions: np.ndarray,
    colors: np.ndarray | None = None,
    *,
    fmt: str | None = None,
    spec=None,
    binary_ply: bool = False,
) -> Path:
    """Dispatch by extension or explicit `fmt` (one of ply, pcd, glb, obj)."""
    path = Path(path)
    fmt = (fmt or path.suffix.lstrip(".")).lower()
    if fmt == "ply":
        return write_ply(path, positions, colors, binary=binary_ply)
    if fmt == "pcd":
        return write_pcd(path, positions, colors)
    if fmt == "glb":
        return write_glb(path, positions, colors, spec=spec)
    if fmt == "obj":
        return write_obj(path, positions, colors, spec=spec)
    raise ValueError(f"unknown export format: {fmt!r}")
