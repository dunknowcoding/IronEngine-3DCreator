"""QOpenGLWidget that previews a point cloud and (optionally) its
ball-pivot mesh.

Two render modes (toggleable independently):
- POINTS — GL_POINTS pass with soft-disc fragments
- MESH   — Triangle pass with simple Lambertian shading

Two color modes:
- TEXTURED — per-vertex colors from the generator (procedural materials)
- PLAIN    — single uniform color (`set_plain_color`)

Camera controls:
- Left drag         — pan (translates the look-at; rotation pivot is unchanged)
- Right drag        — orbit around the cloud's volume centre (always)
- Middle drag       — orbit (alias)
- Shift + Left drag — orbit (alias)
- Wheel             — zoom
- F key             — frame all (recenter + auto-distance on the current cloud)

Rotation always pivots on the cloud's volume centre (`_pivot`), no matter
how far the user has panned. Pan moves an independent screen-plane offset
(`_pan_offset`) that is applied to both the eye and the look-at target.

Uses PyOpenGL throughout (no PySide6 ↔ PyOpenGL context mixing).
"""
from __future__ import annotations

import ctypes
import math
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import (
    QColor, QFont, QKeyEvent, QMouseEvent, QPainter, QSurfaceFormat, QWheelEvent,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from ..generation.reconstruct import ReconstructedMesh, reconstruct
from .theme import current as theme_current


# ---------------------------------------------------------------- shaders

POINT_VS = """
#version 330 core
layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec3 a_color;
uniform mat4 u_mvp;
uniform float u_point_size;
uniform vec3 u_plain_color;
uniform int u_use_plain;
out vec3 v_color;
void main() {
    gl_Position = u_mvp * vec4(a_pos, 1.0);
    gl_PointSize = u_point_size;
    v_color = (u_use_plain == 1) ? u_plain_color : a_color;
}
"""

POINT_FS = """
#version 330 core
in vec3 v_color;
out vec4 frag_color;
void main() {
    vec2 d = gl_PointCoord - vec2(0.5);
    float r = length(d);
    if (r > 0.5) discard;
    float alpha = smoothstep(0.5, 0.32, r);
    frag_color = vec4(v_color, alpha);
}
"""

MESH_VS = """
#version 330 core
layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec3 a_normal;
layout (location = 2) in vec3 a_color;
uniform mat4 u_mvp;
out vec3 v_normal_world;
out vec3 v_color;
void main() {
    gl_Position = u_mvp * vec4(a_pos, 1.0);
    v_normal_world = a_normal;
    v_color = a_color;
}
"""

MESH_FS = """
#version 330 core
in vec3 v_normal_world;
in vec3 v_color;
uniform vec3 u_light_dir;     // world-space direction TO the light
uniform vec3 u_plain_color;
uniform int u_use_plain;
uniform int u_wireframe;
out vec4 frag_color;
void main() {
    vec3 base = (u_use_plain == 1) ? u_plain_color : v_color;
    vec3 N = normalize(v_normal_world);
    float lambert = max(dot(N, normalize(u_light_dir)), 0.0);
    // Half-Lambert keeps shadowed sides visible — better preview UX than
    // proper Lambertian which goes black on backfacing surfaces.
    float light = 0.35 + 0.65 * (lambert * 0.5 + 0.5);
    frag_color = vec4(base * light, 1.0);
}
"""


# ---------------------------------------------------------------- widget


class PointCloudViewport(QOpenGLWidget):
    """Public Qt signals."""
    cursor_ndc_changed = Signal(float, float)
    mode_changed = Signal()

    # Render-mode constants (kept simple — strings are easy to log/serialise)
    MODE_POINTS = "points"
    MODE_MESH = "mesh"
    MODE_BOTH = "both"

    COLOR_TEXTURED = "textured"
    COLOR_PLAIN = "plain"

    def __init__(self, parent=None) -> None:
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setDepthBufferSize(24)
        fmt.setSamples(4)
        QSurfaceFormat.setDefaultFormat(fmt)
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(480, 360)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # so F key works

        # ---- cloud
        self._positions: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self._colors: np.ndarray = np.empty((0, 3), dtype=np.float32)

        # ---- mesh (lazily built when MODE_MESH or MODE_BOTH is selected)
        self._mesh: Optional[ReconstructedMesh] = None
        self._mesh_dirty = True

        # ---- modes
        self._render_mode = self.MODE_POINTS
        self._color_mode = self.COLOR_TEXTURED
        self._plain_color = (0.8, 0.85, 0.95)
        self._wireframe = False

        # ---- camera (orbit)
        self._yaw = math.radians(35.0)
        self._pitch = math.radians(-20.0)
        self._distance = 3.0
        # `_pivot` is the rotation centre — fixed at the cloud's volume
        # centroid when set_cloud / frame_all is called. Panning never moves it.
        self._pivot = np.zeros(3, dtype=np.float32)
        # `_pan_offset` is a screen-plane offset added equally to the eye and
        # the look-at target, so the cloud appears to slide on screen while
        # the rotation centre stays put.
        self._pan_offset = np.zeros(3, dtype=np.float32)
        self._auto_distance = 3.0
        self._point_size = 4.0

        # ---- mouse
        self._last_mouse = None
        self._drag_button = None
        self._drag_modifiers = Qt.KeyboardModifier.NoModifier
        self._cursor_ndc = (0.0, 0.0)
        self._edit_callback = None
        self._edit_active = False
        # Whether vertex-editing mode is currently engaged. When False, plain
        # left-click pans; when True, plain left-click drives the edit op.
        self._edit_mode_active = False

        # ---- GL handles (raw integer ids)
        self._point_program = 0
        self._mesh_program = 0
        self._point_uloc = {}
        self._mesh_uloc = {}

        self._vao_points = 0
        self._vbo_pos = 0
        self._vbo_col = 0

        self._vao_mesh = 0
        self._vbo_mesh = 0       # interleaved (pos, normal, color)
        self._ibo_mesh = 0
        self._mesh_index_count = 0

        self._gl_ready = False
        self._dirty_points = True

    # ============================================================== public

    def set_cloud(self, positions: np.ndarray, colors: np.ndarray) -> None:
        self._positions = np.ascontiguousarray(positions, dtype=np.float32)
        self._colors = np.ascontiguousarray(colors, dtype=np.float32)
        self._mesh = None
        self._mesh_dirty = True
        if positions.size:
            # Rotation pivot = volume centroid of the cloud. Doesn't move on pan.
            self._pivot = positions.mean(axis=0).astype(np.float32)
            extent = float(np.linalg.norm(positions.max(0) - positions.min(0)))
            self._auto_distance = max(0.5, extent * 1.4)
            self._distance = self._auto_distance
            self._pan_offset = np.zeros(3, dtype=np.float32)
        self._dirty_points = True
        self.update()

    def set_render_mode(self, mode: str) -> None:
        if mode in (self.MODE_POINTS, self.MODE_MESH, self.MODE_BOTH):
            self._render_mode = mode
            self.mode_changed.emit()
            self.update()

    def render_mode(self) -> str:
        return self._render_mode

    def set_color_mode(self, mode: str) -> None:
        if mode in (self.COLOR_TEXTURED, self.COLOR_PLAIN):
            self._color_mode = mode
            self.mode_changed.emit()
            self.update()

    def color_mode(self) -> str:
        return self._color_mode

    def set_plain_color(self, rgb: tuple[float, float, float]) -> None:
        self._plain_color = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
        self.update()

    def set_wireframe(self, on: bool) -> None:
        self._wireframe = bool(on); self.update()

    def set_point_size(self, px: float) -> None:
        self._point_size = float(px); self.update()

    def set_edit_callback(self, fn) -> None:
        self._edit_callback = fn

    def set_edit_mode_active(self, active: bool) -> None:
        """Tell the viewport whether vertex-editing mode is engaged.

        When active and an edit callback is registered, plain left-click +
        drag invokes the callback instead of panning the camera.
        """
        self._edit_mode_active = bool(active)

    def cursor_ndc(self) -> tuple[float, float]:
        return self._cursor_ndc

    def view_proj(self) -> np.ndarray:
        return _orbit_mvp(
            self._yaw, self._pitch, self._distance,
            pivot=self._pivot, pan_offset=self._pan_offset,
            aspect=self.width() / max(1, self.height()),
        )

    def cloud(self) -> tuple[np.ndarray, np.ndarray]:
        return self._positions, self._colors

    def mesh(self) -> Optional[ReconstructedMesh]:
        return self._mesh

    def has_cloud(self) -> bool:
        return self._positions.size > 0

    def frame_all(self) -> None:
        """Reset camera pivot to the cloud's centroid, clear pan, restore distance."""
        if not self.has_cloud():
            return
        self._pivot = self._positions.mean(axis=0).astype(np.float32)
        self._pan_offset = np.zeros(3, dtype=np.float32)
        self._distance = self._auto_distance
        self.update()

    def mark_buffers_dirty(self) -> None:
        self._dirty_points = True
        # The reconstructed mesh is now stale — drop it. A subsequent
        # mode toggle (or invalidate_mesh follow-up) will rebuild it.
        self.invalidate_mesh()
        self.update()

    def invalidate_mesh(self) -> None:
        """Forget the cached mesh. The next paintGL pass with a mesh mode
        active will trigger reconstruction."""
        self._mesh = None
        self._mesh_dirty = True

    def ensure_mesh(self, *, radius: float = 0.0) -> Optional[ReconstructedMesh]:
        """Build (or fetch from cache) the reconstructed mesh.
        `radius=0` lets the reconstructor pick adaptive radii from the
        cloud's nearest-neighbour spacing — recommended."""
        if not self.has_cloud():
            return None
        try:
            self._mesh = reconstruct(self._positions, radius=radius)
        except ImportError:
            self._mesh = None
            return None
        self._mesh_dirty = True
        return self._mesh

    # ============================================================== GL

    def initializeGL(self) -> None:
        from OpenGL import GL

        self._point_program = _build_program(POINT_VS, POINT_FS)
        self._mesh_program = _build_program(MESH_VS, MESH_FS)

        self._point_uloc = {
            "u_mvp": GL.glGetUniformLocation(self._point_program, "u_mvp"),
            "u_point_size": GL.glGetUniformLocation(self._point_program, "u_point_size"),
            "u_plain_color": GL.glGetUniformLocation(self._point_program, "u_plain_color"),
            "u_use_plain": GL.glGetUniformLocation(self._point_program, "u_use_plain"),
        }
        self._mesh_uloc = {
            "u_mvp": GL.glGetUniformLocation(self._mesh_program, "u_mvp"),
            "u_light_dir": GL.glGetUniformLocation(self._mesh_program, "u_light_dir"),
            "u_plain_color": GL.glGetUniformLocation(self._mesh_program, "u_plain_color"),
            "u_use_plain": GL.glGetUniformLocation(self._mesh_program, "u_use_plain"),
            "u_wireframe": GL.glGetUniformLocation(self._mesh_program, "u_wireframe"),
        }

        self._vao_points = GL.glGenVertexArrays(1)
        self._vbo_pos = GL.glGenBuffers(1)
        self._vbo_col = GL.glGenBuffers(1)
        self._vao_mesh = GL.glGenVertexArrays(1)
        self._vbo_mesh = GL.glGenBuffers(1)
        self._ibo_mesh = GL.glGenBuffers(1)

        GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        self._gl_ready = True

    def _upload_points(self) -> None:
        from OpenGL import GL
        GL.glBindVertexArray(self._vao_points)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo_pos)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self._positions.nbytes, self._positions, GL.GL_DYNAMIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo_col)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self._colors.nbytes, self._colors, GL.GL_DYNAMIC_DRAW)
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))
        GL.glBindVertexArray(0)
        self._dirty_points = False

    def _upload_mesh(self) -> None:
        from OpenGL import GL
        if self._mesh is None:
            return
        n = self._mesh.positions.shape[0]
        # Reuse cloud color per vertex by nearest-neighbour lookup so the mesh
        # gets the same procedural texture as the points. Open3D's reconstructed
        # vertices are a subset of the input points but reordered, so we can't
        # directly index — fall back to the closest-input-point match, which is
        # cheap with KD-tree below for moderate sizes.
        if self._color_mode == self.COLOR_TEXTURED and self._colors.size > 0:
            colors_v = _nearest_colors(self._mesh.positions, self._positions, self._colors)
        else:
            base = np.asarray(self._plain_color, dtype=np.float32)
            colors_v = np.broadcast_to(base, (n, 3)).copy()
        # Interleave: pos(3) + normal(3) + color(3) = 9 floats per vertex.
        interleaved = np.concatenate([self._mesh.positions, self._mesh.normals, colors_v], axis=1).astype(np.float32)

        GL.glBindVertexArray(self._vao_mesh)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo_mesh)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, interleaved.nbytes, interleaved, GL.GL_DYNAMIC_DRAW)
        stride = 9 * 4
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(3 * 4))
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(2, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(6 * 4))

        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._ibo_mesh)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self._mesh.indices.nbytes, self._mesh.indices, GL.GL_DYNAMIC_DRAW)
        self._mesh_index_count = int(self._mesh.indices.size)
        GL.glBindVertexArray(0)
        self._mesh_dirty = False

    def paintGL(self) -> None:
        from OpenGL import GL
        pal = theme_current()
        bg = _hex_to_rgbf(pal.bg_window)
        GL.glClearColor(bg[0], bg[1], bg[2], 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        if not self._gl_ready or not self.has_cloud():
            self._draw_empty_hint()
            return

        if self._dirty_points:
            self._upload_points()

        # Build mesh on demand the first time MESH/BOTH is requested.
        wants_mesh = self._render_mode in (self.MODE_MESH, self.MODE_BOTH)
        if wants_mesh and self._mesh is None:
            self.ensure_mesh()
        if wants_mesh and self._mesh is not None and self._mesh_dirty:
            self._upload_mesh()

        mvp = self.view_proj().astype(np.float32)
        plain_used = 1 if self._color_mode == self.COLOR_PLAIN else 0
        plain_rgb = self._plain_color

        # Enable depth test for the entire pass so points and mesh occlude
        # each other correctly in BOTH mode (otherwise points would render
        # on top of the mesh regardless of depth).
        GL.glEnable(GL.GL_DEPTH_TEST)
        if self._mesh is not None and wants_mesh and self._mesh_index_count > 0:
            if self._wireframe:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
            else:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
            GL.glUseProgram(self._mesh_program)
            GL.glUniformMatrix4fv(self._mesh_uloc["u_mvp"], 1, GL.GL_TRUE, mvp)
            GL.glUniform3f(self._mesh_uloc["u_light_dir"], 0.4, 0.8, 0.6)
            GL.glUniform3f(self._mesh_uloc["u_plain_color"], *plain_rgb)
            GL.glUniform1i(self._mesh_uloc["u_use_plain"], plain_used)
            GL.glUniform1i(self._mesh_uloc["u_wireframe"], 1 if self._wireframe else 0)
            GL.glBindVertexArray(self._vao_mesh)
            GL.glDrawElements(GL.GL_TRIANGLES, self._mesh_index_count, GL.GL_UNSIGNED_INT, ctypes.c_void_p(0))
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        if self._render_mode in (self.MODE_POINTS, self.MODE_BOTH):
            GL.glUseProgram(self._point_program)
            GL.glUniformMatrix4fv(self._point_uloc["u_mvp"], 1, GL.GL_TRUE, mvp)
            GL.glUniform1f(self._point_uloc["u_point_size"], float(self._point_size))
            GL.glUniform3f(self._point_uloc["u_plain_color"], *plain_rgb)
            GL.glUniform1i(self._point_uloc["u_use_plain"], plain_used)
            GL.glBindVertexArray(self._vao_points)
            GL.glDrawArrays(GL.GL_POINTS, 0, self._positions.shape[0])

        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)

    def resizeGL(self, w: int, h: int) -> None:
        from OpenGL import GL
        GL.glViewport(0, 0, w, h)

    def _draw_empty_hint(self) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pal = theme_current()
        title_color = QColor(pal.accent); title_color.setAlphaF(0.75)
        hint_color = QColor(pal.text_dim); hint_color.setAlphaF(0.85)
        title_f = QFont("Segoe UI Variable"); title_f.setPointSize(15); title_f.setBold(True)
        hint_f = QFont("Cascadia Code"); hint_f.setPointSize(10)
        p.setPen(title_color); p.setFont(title_f)
        p.drawText(self.rect().adjusted(0, -32, 0, 0), Qt.AlignmentFlag.AlignCenter, "✦ IronEngine 3D Creator")
        p.setPen(hint_color); p.setFont(hint_f)
        p.drawText(self.rect().adjusted(0, 6, 0, 0), Qt.AlignmentFlag.AlignCenter,
                   "Click ✦ Auto for an instant model, or describe one and press ▶ Generate.")
        p.drawText(self.rect().adjusted(0, 38, 0, 0), Qt.AlignmentFlag.AlignCenter,
                   "Left-drag to pan · Right-drag to rotate around centre · Wheel to zoom · F to frame")

    # ============================================================== input

    def mousePressEvent(self, e: QMouseEvent) -> None:
        self.setFocus()
        self._last_mouse = (e.position().x(), e.position().y())
        self._drag_button = e.button()
        self._drag_modifiers = e.modifiers()
        # Edit mode hijacks left-click only when the user has explicitly
        # selected an edit tool; otherwise plain LMB pans.
        if (
            self._edit_callback
            and e.button() == Qt.MouseButton.LeftButton
            and not (e.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            and self._edit_mode_active
        ):
            self._edit_active = True
            self._edit_callback(self._cursor_ndc, "press", False)

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        if self._edit_active and e.button() == Qt.MouseButton.LeftButton:
            self._edit_active = False
            if self._edit_callback:
                self._edit_callback(self._cursor_ndc, "release", False)
        self._drag_button = None

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        x = e.position().x(); y = e.position().y()
        ndc = (2.0 * x / max(1, self.width()) - 1.0, 1.0 - 2.0 * y / max(1, self.height()))
        self._cursor_ndc = ndc
        self.cursor_ndc_changed.emit(ndc[0], ndc[1])

        if self._edit_active and self._edit_callback:
            self._edit_callback(self._cursor_ndc, "drag", True)
            self._last_mouse = (x, y)
            return

        if self._last_mouse is None:
            return
        dx = x - self._last_mouse[0]; dy = y - self._last_mouse[1]
        self._last_mouse = (x, y)

        # Decide gesture:
        # - left-drag (no modifier)  → pan (translates the view; pivot unchanged)
        # - right-drag / middle-drag → orbit around _pivot (volume centre)
        # - shift + left-drag        → orbit (alias)
        is_orbit = (
            self._drag_button == Qt.MouseButton.RightButton
            or self._drag_button == Qt.MouseButton.MiddleButton
            or (self._drag_button == Qt.MouseButton.LeftButton
                and (self._drag_modifiers & Qt.KeyboardModifier.ShiftModifier))
        )
        if is_orbit:
            self._yaw -= dx * 0.01
            self._pitch -= dy * 0.01
            self._pitch = max(min(self._pitch, math.pi / 2 - 0.01), -math.pi / 2 + 0.01)
            self.update()
        elif self._drag_button == Qt.MouseButton.LeftButton:
            self._pan(dx, dy)
            self.update()

    def _pan(self, dx: float, dy: float) -> None:
        """Translate the view by accumulating a screen-plane offset.

        We never touch `_pivot` here — the rotation centre stays at the
        cloud's volume centroid no matter how far the user pans.
        """
        eye, _ = _eye_and_up(self._yaw, self._pitch, self._distance,
                             self._pivot + self._pan_offset)
        forward = (self._pivot + self._pan_offset) - eye
        forward /= np.linalg.norm(forward) + 1e-9
        right = np.cross(forward, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        right /= np.linalg.norm(right) + 1e-9
        up = np.cross(right, forward)
        scale = self._distance * 0.0025
        self._pan_offset += (-right * dx + up * dy) * scale

    def wheelEvent(self, e: QWheelEvent) -> None:
        delta = e.angleDelta().y() / 120.0
        self._distance *= 0.9 ** delta
        self._distance = max(0.05, min(self._distance, 500.0))
        self.update()

    def keyPressEvent(self, e: QKeyEvent) -> None:
        if e.key() == Qt.Key.Key_F:
            self.frame_all()
            return
        if e.key() == Qt.Key.Key_1:
            self.set_render_mode(self.MODE_POINTS); return
        if e.key() == Qt.Key.Key_2:
            self.set_render_mode(self.MODE_MESH); return
        if e.key() == Qt.Key.Key_3:
            self.set_render_mode(self.MODE_BOTH); return
        super().keyPressEvent(e)


# ---------------------------------------------------------------- helpers


def _build_program(vs_src: str, fs_src: str) -> int:
    from OpenGL import GL
    vs = GL.glCreateShader(GL.GL_VERTEX_SHADER)
    GL.glShaderSource(vs, vs_src); GL.glCompileShader(vs)
    if GL.glGetShaderiv(vs, GL.GL_COMPILE_STATUS) != GL.GL_TRUE:
        raise RuntimeError("vert shader: " + GL.glGetShaderInfoLog(vs).decode("utf-8", "replace"))
    fs = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
    GL.glShaderSource(fs, fs_src); GL.glCompileShader(fs)
    if GL.glGetShaderiv(fs, GL.GL_COMPILE_STATUS) != GL.GL_TRUE:
        raise RuntimeError("frag shader: " + GL.glGetShaderInfoLog(fs).decode("utf-8", "replace"))
    program = GL.glCreateProgram()
    GL.glAttachShader(program, vs); GL.glAttachShader(program, fs)
    GL.glLinkProgram(program)
    if GL.glGetProgramiv(program, GL.GL_LINK_STATUS) != GL.GL_TRUE:
        raise RuntimeError("link: " + GL.glGetProgramInfoLog(program).decode("utf-8", "replace"))
    GL.glDeleteShader(vs); GL.glDeleteShader(fs)
    return program


def _hex_to_rgbf(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


def _eye_and_up(yaw, pitch, distance, target):
    eye = target + np.array([
        distance * math.cos(pitch) * math.sin(yaw),
        distance * math.sin(pitch),
        distance * math.cos(pitch) * math.cos(yaw),
    ], dtype=np.float32)
    return eye, np.array([0.0, 1.0, 0.0], dtype=np.float32)


def _orbit_mvp(yaw, pitch, distance, *, pivot, pan_offset, aspect):
    """Build the view-projection matrix.

    The camera orbits around `pivot` (cloud volume centre) at `distance`.
    `pan_offset` is added equally to the eye and the look-at target — that
    slides the cloud across the screen without changing the rotation centre.
    """
    eye_base, up = _eye_and_up(yaw, pitch, distance, pivot)
    eye = eye_base + pan_offset
    target = pivot + pan_offset
    view = _look_at(eye, target, up)
    proj = _perspective(math.radians(45.0), aspect, 0.05, 500.0)
    return (proj @ view).astype(np.float32)


def _look_at(eye, target, up):
    f = (target - eye); f /= np.linalg.norm(f) + 1e-9
    s = np.cross(f, up); s /= np.linalg.norm(s) + 1e-9
    u = np.cross(s, f)
    M = np.eye(4, dtype=np.float32)
    M[0, :3] = s; M[1, :3] = u; M[2, :3] = -f
    M[0, 3] = -s @ eye; M[1, 3] = -u @ eye; M[2, 3] = f @ eye
    return M


def _perspective(fovy, aspect, near, far):
    f = 1.0 / math.tan(fovy / 2.0)
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (far + near) / (near - far)
    M[2, 3] = (2 * far * near) / (near - far)
    M[3, 2] = -1.0
    return M


def _nearest_colors(query: np.ndarray, source_pos: np.ndarray, source_col: np.ndarray) -> np.ndarray:
    """For each query vertex, pick the colour of the nearest input point.

    O(N·M) but blocked into chunks to keep memory bounded. Works for the sizes
    we typically deal with (mesh: a few k vertices; cloud: 50k–500k points).
    """
    if query.shape[0] == 0 or source_pos.shape[0] == 0:
        return np.zeros((query.shape[0], 3), dtype=np.float32)
    out = np.empty((query.shape[0], 3), dtype=np.float32)
    chunk = 4096
    for i in range(0, query.shape[0], chunk):
        q = query[i:i + chunk]
        d2 = ((q[:, None, :] - source_pos[None, :, :]) ** 2).sum(axis=2)
        idx = np.argmin(d2, axis=1)
        out[i:i + chunk] = source_col[idx]
    return out
