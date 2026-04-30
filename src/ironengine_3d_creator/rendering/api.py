"""Public renderer API.

Three high-level operations:

- ``reconstruct_mesh(positions)`` — ball-pivot triangle mesh from a cloud
  (cached, returns numpy arrays).
- ``colorize_plain(positions, rgb)`` — single-colour cloud array.
- ``render_points_offscreen(positions, colors, ...)`` — RGBA image of the cloud,
  no Qt UI required (uses an offscreen QOffscreenSurface + QOpenGLContext).
- ``render_mesh_offscreen(positions, indices, normals, colors, ...)`` — RGBA
  image of the reconstructed mesh, same offscreen surface.

These are the same code paths the in-UI viewport uses — the UI just adds a
QOpenGLWidget on top. External callers (SceneEditor, batch scripts, tests)
can render to images without standing up a window.
"""
from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..generation.reconstruct import ReconstructedMesh, reconstruct as _reconstruct


# ---------------------------------------------------------------- public types


@dataclass
class RenderOptions:
    width: int = 800
    height: int = 600
    point_size: float = 4.0
    bg_color: tuple[float, float, float] = (0.05, 0.06, 0.10)
    light_dir: tuple[float, float, float] = (0.4, 0.8, 0.6)
    yaw_deg: float = 35.0
    pitch_deg: float = -20.0
    distance: float = 0.0  # 0 → auto-frame from cloud extent
    target: Optional[tuple[float, float, float]] = None


# ---------------------------------------------------------------- helpers


def reconstruct_mesh(positions: np.ndarray, *, radius: float = 0.0) -> ReconstructedMesh:
    """Triangulate a point cloud. `radius=0` (the default) picks an adaptive
    radius from the cloud's nearest-neighbour spacing — recommended.

    Cached: repeat calls with the same array return the previous result.
    """
    return _reconstruct(positions, radius=radius)


def colorize_plain(positions: np.ndarray, rgb: tuple[float, float, float]) -> np.ndarray:
    n = positions.shape[0]
    return np.broadcast_to(np.asarray(rgb, dtype=np.float32), (n, 3)).copy()


# ---------------------------------------------------------------- offscreen GL


_POINT_VS = """
#version 330 core
layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec3 a_color;
uniform mat4 u_mvp;
uniform float u_point_size;
out vec3 v_color;
void main() {
    gl_Position = u_mvp * vec4(a_pos, 1.0);
    gl_PointSize = u_point_size;
    v_color = a_color;
}
"""

_POINT_FS = """
#version 330 core
in vec3 v_color;
out vec4 frag_color;
void main() {
    vec2 d = gl_PointCoord - vec2(0.5);
    if (length(d) > 0.5) discard;
    frag_color = vec4(v_color, 1.0);
}
"""

_MESH_VS = """
#version 330 core
layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec3 a_normal;
layout (location = 2) in vec3 a_color;
uniform mat4 u_mvp;
out vec3 v_color;
out vec3 v_normal;
void main() {
    gl_Position = u_mvp * vec4(a_pos, 1.0);
    v_normal = a_normal;
    v_color = a_color;
}
"""

_MESH_FS = """
#version 330 core
in vec3 v_color;
in vec3 v_normal;
uniform vec3 u_light_dir;
out vec4 frag_color;
void main() {
    vec3 N = normalize(v_normal);
    float l = max(dot(N, normalize(u_light_dir)), 0.0);
    frag_color = vec4(v_color * (0.35 + 0.65 * (l * 0.5 + 0.5)), 1.0);
}
"""


def _auto_frame(positions: np.ndarray) -> tuple[np.ndarray, float]:
    centre = positions.mean(axis=0)
    extent = float(np.linalg.norm(positions.max(0) - positions.min(0)))
    return centre.astype(np.float32), max(0.5, extent * 1.4)


def _orbit_mvp(target: np.ndarray, distance: float, yaw_rad: float, pitch_rad: float, aspect: float) -> np.ndarray:
    eye = target + np.array([
        distance * math.cos(pitch_rad) * math.sin(yaw_rad),
        distance * math.sin(pitch_rad),
        distance * math.cos(pitch_rad) * math.cos(yaw_rad),
    ], dtype=np.float32)
    f = (target - eye); f /= np.linalg.norm(f) + 1e-9
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    s = np.cross(f, up); s /= np.linalg.norm(s) + 1e-9
    u = np.cross(s, f)
    view = np.eye(4, dtype=np.float32)
    view[0, :3] = s; view[1, :3] = u; view[2, :3] = -f
    view[0, 3] = -s @ eye; view[1, 3] = -u @ eye; view[2, 3] = f @ eye
    fovy = math.radians(45.0); near, far = 0.05, 500.0
    fy = 1.0 / math.tan(fovy / 2)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = fy / aspect; proj[1, 1] = fy
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2 * far * near) / (near - far)
    proj[3, 2] = -1.0
    return (proj @ view).astype(np.float32)


class _OffscreenContext:
    """Lazily-built offscreen GL context + FBO, reused across calls."""

    _instance: Optional["_OffscreenContext"] = None

    def __init__(self) -> None:
        from PySide6.QtCore import QSize
        from PySide6.QtGui import QOffscreenSurface, QOpenGLContext, QSurfaceFormat

        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setDepthBufferSize(24)
        QSurfaceFormat.setDefaultFormat(fmt)
        self._surface = QOffscreenSurface()
        self._surface.setFormat(fmt)
        self._surface.create()
        self._ctx = QOpenGLContext()
        self._ctx.setFormat(fmt)
        if not self._ctx.create():
            raise RuntimeError("offscreen GL context creation failed")
        self._programs: dict[str, int] = {}
        self._fbo = 0
        self._fbo_size = (0, 0)
        self._color_tex = 0
        self._depth_rbo = 0

    @classmethod
    def get(cls) -> "_OffscreenContext":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def make_current(self) -> None:
        if not self._ctx.makeCurrent(self._surface):
            raise RuntimeError("offscreen makeCurrent failed")

    def ensure_fbo(self, w: int, h: int) -> int:
        from OpenGL import GL
        if self._fbo and self._fbo_size == (w, h):
            return self._fbo
        if self._fbo:
            GL.glDeleteFramebuffers(1, [self._fbo])
            GL.glDeleteTextures(1, [self._color_tex])
            GL.glDeleteRenderbuffers(1, [self._depth_rbo])
        self._fbo = GL.glGenFramebuffers(1)
        self._color_tex = GL.glGenTextures(1)
        self._depth_rbo = GL.glGenRenderbuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._color_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, w, h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self._color_tex, 0)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self._depth_rbo)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, w, h)
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, self._depth_rbo)
        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("offscreen FBO incomplete")
        self._fbo_size = (w, h)
        return self._fbo

    def get_program(self, name: str, vs: str, fs: str) -> int:
        if name in self._programs:
            return self._programs[name]
        from OpenGL import GL
        v = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(v, vs); GL.glCompileShader(v)
        if GL.glGetShaderiv(v, GL.GL_COMPILE_STATUS) != GL.GL_TRUE:
            raise RuntimeError(GL.glGetShaderInfoLog(v).decode())
        f = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(f, fs); GL.glCompileShader(f)
        if GL.glGetShaderiv(f, GL.GL_COMPILE_STATUS) != GL.GL_TRUE:
            raise RuntimeError(GL.glGetShaderInfoLog(f).decode())
        prog = GL.glCreateProgram()
        GL.glAttachShader(prog, v); GL.glAttachShader(prog, f); GL.glLinkProgram(prog)
        if GL.glGetProgramiv(prog, GL.GL_LINK_STATUS) != GL.GL_TRUE:
            raise RuntimeError(GL.glGetProgramInfoLog(prog).decode())
        GL.glDeleteShader(v); GL.glDeleteShader(f)
        self._programs[name] = prog
        return prog


def _read_fbo_pixels(w: int, h: int) -> np.ndarray:
    from OpenGL import GL
    data = GL.glReadPixels(0, 0, w, h, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
    arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 4)
    # GL origin is bottom-left; flip so the resulting image reads top-down.
    return np.ascontiguousarray(arr[::-1])


def render_points_offscreen(
    positions: np.ndarray,
    colors: np.ndarray,
    *,
    options: Optional[RenderOptions] = None,
) -> np.ndarray:
    """Render the cloud to an (H, W, 4) uint8 RGBA image. Headless-safe."""
    from OpenGL import GL
    opt = options or RenderOptions()
    ctx = _OffscreenContext.get()
    ctx.make_current()
    ctx.ensure_fbo(opt.width, opt.height)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, ctx._fbo)
    GL.glViewport(0, 0, opt.width, opt.height)
    GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)
    GL.glClearColor(*opt.bg_color, 1.0)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    if positions.shape[0] == 0:
        return _read_fbo_pixels(opt.width, opt.height)

    program = ctx.get_program("points", _POINT_VS, _POINT_FS)
    GL.glUseProgram(program)
    target = (np.asarray(opt.target, dtype=np.float32) if opt.target is not None
              else _auto_frame(positions)[0])
    distance = opt.distance or _auto_frame(positions)[1]
    mvp = _orbit_mvp(target, distance, math.radians(opt.yaw_deg), math.radians(opt.pitch_deg),
                     opt.width / max(1, opt.height))
    GL.glUniformMatrix4fv(GL.glGetUniformLocation(program, "u_mvp"), 1, GL.GL_TRUE, mvp)
    GL.glUniform1f(GL.glGetUniformLocation(program, "u_point_size"), float(opt.point_size))

    vao = GL.glGenVertexArrays(1); vbo_p = GL.glGenBuffers(1); vbo_c = GL.glGenBuffers(1)
    GL.glBindVertexArray(vao)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_p)
    pos = np.ascontiguousarray(positions, dtype=np.float32)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, pos.nbytes, pos, GL.GL_STATIC_DRAW)
    GL.glEnableVertexAttribArray(0)
    GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_c)
    col = np.ascontiguousarray(colors, dtype=np.float32)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, col.nbytes, col, GL.GL_STATIC_DRAW)
    GL.glEnableVertexAttribArray(1)
    GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))
    GL.glDrawArrays(GL.GL_POINTS, 0, pos.shape[0])
    GL.glBindVertexArray(0)
    GL.glDeleteBuffers(2, [vbo_p, vbo_c])
    GL.glDeleteVertexArrays(1, [vao])

    img = _read_fbo_pixels(opt.width, opt.height)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
    return img


def render_mesh_offscreen(
    positions: np.ndarray,
    indices: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    *,
    options: Optional[RenderOptions] = None,
) -> np.ndarray:
    """Render a triangle mesh to an (H, W, 4) RGBA image."""
    from OpenGL import GL
    opt = options or RenderOptions()
    ctx = _OffscreenContext.get()
    ctx.make_current()
    ctx.ensure_fbo(opt.width, opt.height)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, ctx._fbo)
    GL.glViewport(0, 0, opt.width, opt.height)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glClearColor(*opt.bg_color, 1.0)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    if positions.shape[0] == 0 or indices.size == 0:
        return _read_fbo_pixels(opt.width, opt.height)

    program = ctx.get_program("mesh", _MESH_VS, _MESH_FS)
    GL.glUseProgram(program)
    target = (np.asarray(opt.target, dtype=np.float32) if opt.target is not None
              else _auto_frame(positions)[0])
    distance = opt.distance or _auto_frame(positions)[1]
    mvp = _orbit_mvp(target, distance, math.radians(opt.yaw_deg), math.radians(opt.pitch_deg),
                     opt.width / max(1, opt.height))
    GL.glUniformMatrix4fv(GL.glGetUniformLocation(program, "u_mvp"), 1, GL.GL_TRUE, mvp)
    GL.glUniform3f(GL.glGetUniformLocation(program, "u_light_dir"), *opt.light_dir)

    vertices = np.concatenate(
        [
            np.ascontiguousarray(positions, dtype=np.float32),
            np.ascontiguousarray(normals, dtype=np.float32),
            np.ascontiguousarray(colors, dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    idx = np.ascontiguousarray(indices, dtype=np.uint32)

    vao = GL.glGenVertexArrays(1); vbo = GL.glGenBuffers(1); ibo = GL.glGenBuffers(1)
    GL.glBindVertexArray(vao)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)
    stride = 9 * 4
    for loc, off in ((0, 0), (1, 3 * 4), (2, 6 * 4)):
        GL.glEnableVertexAttribArray(loc)
        GL.glVertexAttribPointer(loc, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(off))
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ibo)
    GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL.GL_STATIC_DRAW)
    GL.glDrawElements(GL.GL_TRIANGLES, idx.size, GL.GL_UNSIGNED_INT, ctypes.c_void_p(0))
    GL.glBindVertexArray(0)
    GL.glDeleteBuffers(2, [vbo, ibo])
    GL.glDeleteVertexArrays(1, [vao])

    img = _read_fbo_pixels(opt.width, opt.height)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
    GL.glDisable(GL.GL_DEPTH_TEST)
    return img
