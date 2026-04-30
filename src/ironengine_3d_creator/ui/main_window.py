"""Main window — wires panels, viewport, menus, toolbar, and worker threads."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from PySide6.QtCore import QByteArray, QThread, QTimer, Qt
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QScrollArea, QSizePolicy, QStatusBar, QToolBar,
    QVBoxLayout, QWidget,
)

from .. import __version__
from ..alignment.schema import GenerationSpec
from ..core import exporter
from ..core.pipeline import PipelineRequest, PipelineResult, replay_spec
from ..core.session import Session, SessionRequirements
from ..core.settings import Settings
from ..editing import ops as editops
from ..editing.history import History
from .panels.editing_panel import EditingPanel
from .panels.llm_config_panel import LLMConfigPanel
from .panels.requirements_panel import RequirementsPanel
from .panels.resources_panel import ResourcesPanel
from .panels.spec_preview_panel import SpecPreviewPanel
from .panels.token_stream_widget import TokenStreamWidget
from .theme import (
    DARK_THEMES, LIGHT_THEMES, current as theme_current, names as theme_names,
    set_theme, stylesheet,
)
from .viewport import PointCloudViewport
from .widgets.cyber_button import CyberButton
from .widgets.shimmer_progress import ShimmerProgress
from .workers import start_worker

_log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, *, target_dir_override: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle(f"IronEngine 3D Creator · v{__version__}")
        self.resize(1480, 900)

        self._settings = Settings.load()
        if target_dir_override:
            self._settings.set("export", "target_dir", value=str(target_dir_override))
        self.setStyleSheet(stylesheet(theme_current()))

        # Restore previous window geometry if available.
        geo_b64 = self._settings.get("ui", "geometry", default=None)
        if geo_b64:
            try:
                self.restoreGeometry(QByteArray.fromBase64(geo_b64.encode("ascii")))
            except Exception:
                _log.exception("failed to restore window geometry")

        # State -------------------------------------------------------------
        self._latest: Optional[PipelineResult] = None
        self._history = History()
        self._edit_mode = "none"
        self._brush = 0.12
        self._paint_rgb = (1.0, 0.5, 0.2)
        self._gen_thread: QThread | None = None
        self._gen_worker = None  # held to defeat Python GC while the thread runs
        self._mesh_thread: QThread | None = None
        self._mesh_worker = None
        # Edit-stroke throttle: redraws are rate-limited to ~60 fps so a brush
        # drag on a 500k-point cloud doesn't spend its whole budget re-uploading
        # the VBO. The flag is set on every edit step and cleared by the timer.
        self._edit_pending = False
        self._edit_timer = QTimer(self)
        self._edit_timer.setInterval(16)  # ~60 fps
        self._edit_timer.timeout.connect(self._flush_edit)

        # Layout ------------------------------------------------------------
        # Build the centre first because the toolbar wires signals to the
        # viewport (Wireframe toggle, Frame button), and panels in the
        # right column drive status-bar handlers.
        self._build_menus()
        self._build_central()
        self._build_toolbar()
        self._build_status_bar()

        # Wire panels -------------------------------------------------------
        self.requirements.generate_clicked.connect(self._on_generate)
        self.spec_preview.rerun_clicked.connect(self._on_rerun_from_spec)
        self.resources.target_dir_changed.connect(lambda d: (
            self.status.showMessage(f"Export target: {d}", 4000),
            self._refresh_status_info(),
        ))
        self.editing.mode_changed.connect(self._set_edit_mode)
        self.editing.brush_changed.connect(self._set_brush)
        self.editing.paint_color_changed.connect(self._set_paint_color)
        self.editing.undo_clicked.connect(self._undo)
        self.editing.redo_clicked.connect(self._redo)

        self.viewport.set_edit_callback(self._on_viewport_edit)
        # Initial state — nothing generated yet, so cloud-dependent buttons
        # should be disabled.
        self._refresh_action_states()

    # ------------------------------------------------------------------ menus
    def _build_menus(self) -> None:
        bar = self.menuBar()

        m_file = bar.addMenu("&File")
        a_new = QAction("New session", self); a_new.triggered.connect(self._new_session); m_file.addAction(a_new)
        a_open = QAction("Open session…", self); a_open.triggered.connect(self._open_session); m_file.addAction(a_open)
        a_save = QAction("Save session…", self); a_save.triggered.connect(self._save_session); m_file.addAction(a_save)
        m_file.addSeparator()
        a_export = QAction("Export…", self); a_export.triggered.connect(self._export); m_file.addAction(a_export)
        m_file.addSeparator()
        a_quit = QAction("Quit", self); a_quit.triggered.connect(self.close); m_file.addAction(a_quit)

        m_view = bar.addMenu("&View")
        m_theme = m_view.addMenu("Theme")
        m_dark = m_theme.addMenu("Dark")
        for name in DARK_THEMES:
            act = QAction(name.replace("_", " ").title(), self)
            act.triggered.connect(lambda _=False, n=name: self._set_theme(n))
            m_dark.addAction(act)
        m_light = m_theme.addMenu("Light")
        for name in LIGHT_THEMES:
            act = QAction(name.replace("_", " ").title(), self)
            act.triggered.connect(lambda _=False, n=name: self._set_theme(n))
            m_light.addAction(act)

        m_help = bar.addMenu("&Help")
        a_guide = QAction("User Guide…", self)
        a_guide.setShortcut(QKeySequence("F1"))
        a_guide.triggered.connect(self._show_user_guide)
        m_help.addAction(a_guide)
        a_shortcuts = QAction("Keyboard shortcuts", self)
        a_shortcuts.triggered.connect(self._show_shortcuts)
        m_help.addAction(a_shortcuts)
        m_help.addSeparator()
        a_about = QAction("About", self)
        a_about.triggered.connect(lambda: QMessageBox.information(self, "About",
            f"IronEngine 3D Creator v{__version__}\n\nLLM-driven point cloud generator."))
        m_help.addAction(a_about)

    # --------------------------------------------------------------- toolbar
    def _build_toolbar(self) -> None:
        tb = QToolBar("Viewport", self)
        tb.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)

        # Render mode buttons (mutually exclusive). The mesh option lazily
        # builds a ball-pivot reconstruction the first time it's chosen.
        self._mode_btns: dict[str, CyberButton] = {}
        for key, label, tip in (
            ("points", "⊙ Points", "Render the raw point cloud (1)"),
            ("mesh",   "△ Mesh",   "Render the ball-pivot reconstructed surface (2)"),
            ("both",   "⊙ + △",     "Render both points and mesh together (3)"),
        ):
            btn = CyberButton(label); btn.setCheckable(True); btn.setToolTip(tip)
            btn.clicked.connect(lambda _=False, k=key: self._set_render_mode(k))
            self._mode_btns[key] = btn
            tb.addWidget(btn)
        self._mode_btns["points"].setChecked(True)

        tb.addSeparator()

        # Color mode buttons (textured uses procedural materials, plain is a
        # single uniform colour — useful for inspecting geometry in isolation).
        self._color_btns: dict[str, CyberButton] = {}
        for key, label, tip in (
            ("textured", "🎨 Textured", "Per-vertex procedural colours (wood/stone/...)"),
            ("plain",    "⚪ Plain",    "Pure single colour, no per-point variation"),
        ):
            btn = CyberButton(label); btn.setCheckable(True); btn.setToolTip(tip)
            btn.clicked.connect(lambda _=False, k=key: self._set_color_mode(k))
            self._color_btns[key] = btn
            tb.addWidget(btn)
        self._color_btns["textured"].setChecked(True)

        self._wire_btn = CyberButton("Wireframe")
        self._wire_btn.setCheckable(True)
        self._wire_btn.setToolTip("Render the mesh as wireframe (when mesh is on)")
        self._wire_btn.toggled.connect(self.viewport.set_wireframe)
        tb.addWidget(self._wire_btn)

        tb.addSeparator()

        # Camera helpers.
        frame = CyberButton("⌂ Frame")
        frame.setToolTip("Recenter the camera on the cloud and reset zoom (F)")
        frame.clicked.connect(self.viewport.frame_all)
        tb.addWidget(frame)
        ps = CyberButton("+")
        ps.setToolTip("Increase point size"); ps.setMaximumWidth(32)
        ps.clicked.connect(lambda: self.viewport.set_point_size(min(20.0, self.viewport._point_size + 1)))
        tb.addWidget(ps)
        ms = CyberButton("−")
        ms.setToolTip("Decrease point size"); ms.setMaximumWidth(32)
        ms.clicked.connect(lambda: self.viewport.set_point_size(max(1.0, self.viewport._point_size - 1)))
        tb.addWidget(ms)

        tb.addSeparator()

        send_btn = CyberButton("⮕ Send to SceneEditor")
        send_btn.setToolTip("Export PLY to the user_models library and open SceneEditor")
        send_btn.clicked.connect(self._render_in_sim)
        tb.addWidget(send_btn)

        # Right-side widgets ----------------------------------------------
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)

        tb.addWidget(QLabel("Theme:"))
        self._theme_combo = QComboBox()
        self._theme_combo.addItem("── Dark ──"); _disable_combo_item(self._theme_combo, 0)
        for name in DARK_THEMES:
            self._theme_combo.addItem(name.replace("_", " ").title())
        self._theme_combo.addItem("── Light ──"); _disable_combo_item(self._theme_combo, self._theme_combo.count() - 1)
        for name in LIGHT_THEMES:
            self._theme_combo.addItem(name.replace("_", " ").title())
        self._select_theme_in_combo(theme_current().name)
        self._theme_combo.currentTextChanged.connect(self._on_theme_combo)
        tb.addWidget(self._theme_combo)

        # Keyboard shortcuts ----------------------------------------------
        QShortcut(QKeySequence("Ctrl+G"), self, activated=self._shortcut_generate)
        QShortcut(QKeySequence("Ctrl+Shift+G"), self, activated=self._shortcut_auto)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self._undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=self._redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, activated=self._redo)
        QShortcut(QKeySequence("Ctrl+E"), self, activated=self._export)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self._save_session)
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self._open_session)

    # ---------------------------------------------------------------- centre
    def _build_central(self) -> None:
        central = QWidget(self)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8); outer.setSpacing(8)

        # Left column ------------------------------------------------------
        left = QWidget(); l_lay = QVBoxLayout(left)
        l_lay.setContentsMargins(0, 0, 0, 0); l_lay.setSpacing(8)
        self.requirements = RequirementsPanel()
        self.spec_preview = SpecPreviewPanel()
        l_lay.addWidget(self.requirements, 1)
        l_lay.addWidget(self.spec_preview)
        left_scroll = QScrollArea(); left_scroll.setWidget(left); left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        left_scroll.setMinimumWidth(300); left_scroll.setMaximumWidth(360)

        # Centre column (viewport + token stream) -------------------------
        centre = QWidget(); c_lay = QVBoxLayout(centre)
        c_lay.setContentsMargins(0, 0, 0, 0); c_lay.setSpacing(8)
        self.viewport = PointCloudViewport()
        self.token_stream = TokenStreamWidget()
        c_lay.addWidget(self.viewport, 3)
        c_lay.addWidget(self.token_stream, 1)

        # Right column -----------------------------------------------------
        right = QWidget(); r_lay = QVBoxLayout(right)
        r_lay.setContentsMargins(0, 0, 0, 0); r_lay.setSpacing(8)
        self.llm_config = LLMConfigPanel(self._settings)
        self.resources = ResourcesPanel(self._settings)
        self.editing = EditingPanel()
        r_lay.addWidget(self.llm_config)
        r_lay.addWidget(self.resources)
        r_lay.addWidget(self.editing)
        r_lay.addStretch(1)
        right_scroll = QScrollArea(); right_scroll.setWidget(right); right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        right_scroll.setMinimumWidth(320); right_scroll.setMaximumWidth(400)

        outer.addWidget(left_scroll)
        outer.addWidget(centre, 1)
        outer.addWidget(right_scroll)
        self.setCentralWidget(central)

    # ----------------------------------------------------------- status bar
    def _build_status_bar(self) -> None:
        sb = QStatusBar(self); self.setStatusBar(sb); self.status = sb
        self.progress = ShimmerProgress(); self.progress.setMinimumWidth(280)
        sb.addPermanentWidget(self.progress, 1)
        self._status_info = QLabel("")
        self._status_info.setObjectName("sectionHint")
        sb.addPermanentWidget(self._status_info)
        self._refresh_status_info()
        sb.showMessage("ready — press Ctrl+G to generate, Ctrl+Shift+G for auto")

    def _refresh_status_info(self) -> None:
        from ..core import resources as res
        backend = res.active_backend()
        gpu_glyph = "⚡ " if res.is_gpu(backend) else ""
        self._status_info.setText(f"theme: {theme_current().name}  ·  {gpu_glyph}backend: {backend}")

    # ------------------------------------------------------------ generation
    def _on_generate(self, req: PipelineRequest) -> None:
        provider = None
        try:
            if req.user_prompt.strip():
                provider = self.llm_config.build_provider()
                req.code_mode = self.llm_config.is_code_mode()
        except Exception as e:
            self.status.showMessage(f"LLM unavailable, using auto: {e}", 5000)
            provider = None
            req.user_prompt = ""

        think_on = bool(self.llm_config.think_mode.isChecked())
        self.token_stream.begin("connecting…", think_mode_on=think_on)
        self.progress.start("aligning…")

        thread, worker = start_worker(self, req, provider)
        worker.token.connect(self.token_stream.append_chunk)
        worker.stage.connect(self._on_stage)
        worker.result.connect(self._on_result)
        worker.error.connect(self._on_error)
        worker.finished.connect(self._on_worker_finished)
        # Keep both on `self`. Without holding `worker`, Python GC would destroy
        # it after this method returns — the thread would then run against a
        # dangling C++ object and no signals would ever reach the main thread.
        self._gen_thread = thread
        self._gen_worker = worker
        thread.start()

    def _on_rerun_from_spec(self, spec: GenerationSpec) -> None:
        self.progress.start("re-running…")
        try:
            res = replay_spec(spec)
            self._on_result(res)
        except Exception as e:
            self._on_error(f"{type(e).__name__}: {e}")
        finally:
            self.progress.stop()

    def _on_stage(self, stage: str) -> None:
        self.progress.set_stage(stage)
        self.status.showMessage(stage)

    def _on_result(self, res: PipelineResult) -> None:
        self._latest = res
        self.viewport.set_cloud(res.generation.positions, res.generation.colors)
        self.spec_preview.set_spec(res.spec)
        self._history.clear()
        self.token_stream.end()
        self._refresh_action_states()
        if res.warnings:
            self.status.showMessage("warnings: " + "; ".join(res.warnings), 8000)
        else:
            self.status.showMessage(
                f"generated {res.generation.positions.shape[0]:,} points · {res.spec.shape}", 5000,
            )

    def _refresh_action_states(self) -> None:
        """Enable/disable cloud-dependent toolbar actions so the user can't
        click buttons that would no-op without a generated model."""
        has_cloud = self.viewport.has_cloud()
        for btn in (
            *self._mode_btns.values(),
            *self._color_btns.values(),
            getattr(self, "_wire_btn", None),
        ):
            if btn is not None:
                btn.setEnabled(has_cloud)

    def _on_error(self, msg: str) -> None:
        self.token_stream.end()
        QMessageBox.warning(self, "Generation failed", msg)
        self.status.showMessage("error: " + msg, 8000)

    def _on_worker_finished(self) -> None:
        self.progress.stop()
        self.requirements.re_enable()
        self._gen_thread = None
        self._gen_worker = None

    # ------------------------------------------------------------- editing
    def _set_edit_mode(self, mode: str) -> None:
        self._edit_mode = mode
        # Tell the viewport whether plain LMB should run an edit op or pan.
        self.viewport.set_edit_mode_active(mode != "none")
        self.status.showMessage(f"edit mode: {mode}", 2000)

    def _set_brush(self, v: float) -> None:
        self._brush = v

    def _set_paint_color(self, rgb: tuple[float, float, float]) -> None:
        self._paint_rgb = rgb

    def _on_viewport_edit(self, cursor_ndc, phase, dragging) -> None:
        if self._edit_mode == "none" or self._latest is None:
            return
        if phase == "press":
            # Snapshot before the first drag step.
            self._history.push(*self.viewport.cloud(), op_name=self._edit_mode)
            return
        if phase == "release":
            # If the user is in mesh-or-both mode, kick off an async
            # rebuild now that editing is done. We do NOT rebuild during
            # the drag — ball-pivot would block for seconds per stroke.
            mode = self.viewport.render_mode()
            if mode in ("mesh", "both"):
                self._kick_mesh_worker(mode)
            return
        if phase != "drag":
            return
        positions, colors = self.viewport.cloud()
        mvp = self.viewport.view_proj()
        if self._edit_mode == "brush":
            editops.brush_move(positions, mvp, cursor_ndc, self._brush,
                               direction=np.array([0.0, 0.05, 0.0], dtype=np.float32),
                               strength=1.0)
        elif self._edit_mode == "warp":
            editops.radial_warp(positions, mvp, cursor_ndc, self._brush, factor=1.05)
        elif self._edit_mode == "paint":
            editops.point_paint(colors, mvp, positions, cursor_ndc, self._brush, self._paint_rgb, 0.4)
        elif self._edit_mode == "smooth":
            editops.smooth(positions, mvp, cursor_ndc, self._brush, 0.4)
        # Throttle VBO re-uploads to ~60 fps so big clouds stay
        # responsive during a continuous drag.
        self._mark_edit_pending()

    def _mark_edit_pending(self) -> None:
        """Schedule a viewport redraw after the next 16 ms tick."""
        self._edit_pending = True
        if not self._edit_timer.isActive():
            self._edit_timer.start()

    def _flush_edit(self) -> None:
        if not self._edit_pending:
            self._edit_timer.stop()
            return
        self._edit_pending = False
        # `mark_buffers_dirty` invalidates the cached mesh too — we don't
        # rebuild it during the drag (too slow); a release event triggers
        # `_kick_mesh_worker` once the user lifts the mouse.
        self.viewport.mark_buffers_dirty()

    def _undo(self) -> None:
        if self._latest is None or not self._history.can_undo():
            return
        positions, colors = self.viewport.cloud()
        snap = self._history.undo(positions, colors)
        if snap is not None:
            self._latest.generation.positions = snap.positions
            self._latest.generation.colors = snap.colors
            self.viewport.set_cloud(snap.positions, snap.colors)

    def _redo(self) -> None:
        if self._latest is None or not self._history.can_redo():
            return
        positions, colors = self.viewport.cloud()
        snap = self._history.redo(positions, colors)
        if snap is not None:
            self._latest.generation.positions = snap.positions
            self._latest.generation.colors = snap.colors
            self.viewport.set_cloud(snap.positions, snap.colors)

    # ------------------------------------------------------------- mesh preview
    def _set_render_mode(self, mode: str) -> None:
        for k, btn in self._mode_btns.items():
            btn.setChecked(k == mode)
        if mode in ("mesh", "both") and self._latest is not None and self.viewport.mesh() is None:
            # Reconstruct on a worker thread — ball-pivoting takes seconds on
            # large clouds and would freeze the UI if run inline.
            self._kick_mesh_worker(mode)
            return
        self.viewport.set_render_mode(mode)
        self.status.showMessage(f"render mode: {mode}", 2500)

    def _kick_mesh_worker(self, mode: str) -> None:
        from .workers import start_mesh_worker

        if self._mesh_thread is not None:
            return  # already building
        self.progress.start(f"reconstructing mesh for '{mode}'…")
        self.status.showMessage("ball-pivot reconstruction running on a worker thread", 4000)
        thread, worker = start_mesh_worker(self, self._latest.generation.positions)
        worker.done.connect(lambda mesh, _m=mode: self._on_mesh_ready(mesh, _m))
        worker.error.connect(self._on_mesh_error)
        worker.finished.connect(self._on_mesh_finished)
        self._mesh_thread = thread
        self._mesh_worker = worker
        thread.start()

    def _on_mesh_ready(self, mesh, mode: str) -> None:
        # If the cloud was replaced (new generation) while the worker was
        # in flight, the result is for the OLD cloud — discard it.
        if self._latest is None or not self.viewport.has_cloud():
            return
        self.viewport._mesh = mesh
        self.viewport._mesh_dirty = True
        self.viewport.set_render_mode(mode)
        v = mesh.positions.shape[0] if mesh is not None else 0
        t = (mesh.indices.size // 3) if mesh is not None else 0
        self.status.showMessage(f"mesh built: {v:,} verts · {t:,} tris", 5000)

    def _on_mesh_error(self, msg: str) -> None:
        QMessageBox.warning(self, "Mesh preview failed", msg)
        # Keep render mode at points so the user still sees something.
        for k, btn in self._mode_btns.items():
            btn.setChecked(k == "points")
        self.viewport.set_render_mode("points")
        self.status.showMessage("mesh reconstruction failed — see warning", 6000)

    def _on_mesh_finished(self) -> None:
        self.progress.stop()
        self._mesh_thread = None
        self._mesh_worker = None

    def _set_color_mode(self, mode: str) -> None:
        for k, btn in self._color_btns.items():
            btn.setChecked(k == mode)
        self.viewport.set_color_mode(mode)
        self.status.showMessage(f"color mode: {mode}", 2500)

    # -------------------------------------------------------------- camera
    def _reset_camera(self) -> None:
        self.viewport.frame_all()

    def _render_in_sim(self) -> None:
        """Export to PLY in the configured target dir, then hand off to a
        renderer. Tries IronEngine-SceneEditor first (which loads PLY through
        its asset library), then falls back to the system default viewer.
        """
        if self._latest is None:
            QMessageBox.information(self, "Nothing to render", "Generate a model first.")
            return
        target = self.resources.export_target()
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"creator_preview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ply"
        try:
            exporter.write_ply(path, self._latest.generation.positions, self._latest.generation.colors)
        except Exception as e:
            QMessageBox.warning(self, "Export failed", str(e))
            return

        import importlib.util
        import os
        import subprocess
        import sys

        # Try installed renderers in priority order. SceneEditor doesn't take
        # a CLI path — it discovers PLY/PCD via its asset library scan, which
        # we already taught about our suffixes. So we just launch it.
        candidates: list[tuple[str, list[str], bool]] = [
            ("ironengine_scene_editor", [], False),  # SceneEditor: no path arg
            ("ironengine_sim.tools.point_cloud_viewer", [], True),
            ("ironengine_sim", ["--render"], True),
        ]
        launched_module: str | None = None
        for module, extra, pass_path in candidates:
            try:
                if importlib.util.find_spec(module) is None:
                    continue
            except (ImportError, ValueError):
                continue
            argv = [sys.executable, "-m", module, *extra]
            if pass_path:
                argv.append(str(path))
            try:
                subprocess.Popen(argv, close_fds=True)
                launched_module = module
                break
            except Exception:
                continue

        if launched_module == "ironengine_scene_editor":
            self.status.showMessage(
                f"Wrote {path.name} → SceneEditor opening; refresh Asset Browser to see it",
                7000,
            )
            return
        if launched_module:
            self.status.showMessage(
                f"Wrote {path.name}, opening in {launched_module}", 5000,
            )
            return
        try:
            os.startfile(str(path))
            self.status.showMessage(
                f"Wrote {path} (no Sim/SceneEditor module found — opened in system viewer)",
                7000,
            )
        except Exception as e:
            QMessageBox.warning(
                self, "Launch failed",
                f"Wrote {path}\n\nCould not launch a viewer: {e}",
            )

    # --------------------------------------------------------- session I/O
    def _new_session(self) -> None:
        self._latest = None
        self._history.clear()
        self.viewport.set_cloud(np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32))
        self.token_stream.end()
        self._refresh_action_states()
        self.status.showMessage("new session", 2000)

    def _save_session(self) -> None:
        if self._latest is None:
            QMessageBox.information(self, "Nothing to save", "Generate a model first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save session", "session.iecreator.json", "Session (*.iecreator.json)")
        if not path:
            return
        sess = Session(
            requirements=SessionRequirements(
                prompt=self.requirements.details.toPlainText(),
                shape=self.requirements.shape.currentText(),
                n_points=self.requirements.points.value(),
                bbox=(self.requirements.bx.value(), self.requirements.by.value(), self.requirements.bz.value()),
                legs=self.requirements.legs.value(),
                details=self.requirements.details.toPlainText(),
            ),
            spec=self._latest.spec.to_json(),
            seed=self._latest.spec.seed,
            export={},
        )
        sess.save(Path(path))
        self.status.showMessage(f"saved {path}", 4000)

    def _open_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open session", "", "Session (*.iecreator.json)")
        if not path:
            return
        sess = Session.load(Path(path))
        # Restore requirements UI.
        self.requirements.details.setPlainText(sess.requirements.details)
        self.requirements.points.setValue(sess.requirements.n_points)
        self.requirements.bx.setValue(sess.requirements.bbox[0])
        self.requirements.by.setValue(sess.requirements.bbox[1])
        self.requirements.bz.setValue(sess.requirements.bbox[2])
        self.requirements.legs.setValue(sess.requirements.legs)
        # Replay the spec deterministically.
        spec = GenerationSpec.from_json(sess.spec)
        self._on_rerun_from_spec(spec)
        self.status.showMessage(f"opened {path}", 4000)

    def _export(self) -> None:
        if self._latest is None:
            QMessageBox.information(self, "Nothing to export", "Generate a model first.")
            return
        target = self.resources.export_target()
        target.mkdir(parents=True, exist_ok=True)
        suggested = target / f"{self._latest.spec.shape}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ply"
        path, fmt_filter = QFileDialog.getSaveFileName(
            self, "Export point cloud", str(suggested),
            "Point cloud PLY (*.ply);;Point cloud PCD (*.pcd);;Mesh GLB (*.glb);;Mesh OBJ (*.obj)",
        )
        if not path:
            return
        try:
            written = exporter.export(Path(path), self._latest.generation.positions, self._latest.generation.colors)
            self.status.showMessage(f"exported {written}", 6000)
        except ImportError as e:
            QMessageBox.warning(self, "Export needs Open3D", str(e))
        except Exception as e:
            QMessageBox.warning(self, "Export failed", str(e))

    # --------------------------------------------------------------- theme
    def _show_user_guide(self) -> None:
        from .dialogs.user_guide import show_user_guide
        show_user_guide(self)

    def _show_shortcuts(self) -> None:
        from .dialogs.user_guide import show_shortcuts
        show_shortcuts(self)

    def _set_theme(self, name: str) -> None:
        set_theme(name)
        self.setStyleSheet(stylesheet(theme_current()))
        self._settings.set("ui", "theme", value=name); self._settings.save()
        if hasattr(self, "_theme_combo"):
            self._select_theme_in_combo(name)
        self._refresh_status_info()
        self.update()
        self.viewport.update()  # background is theme-coloured

    def _select_theme_in_combo(self, name: str) -> None:
        title = name.replace("_", " ").title()
        idx = self._theme_combo.findText(title)
        if idx >= 0:
            self._theme_combo.blockSignals(True)
            self._theme_combo.setCurrentIndex(idx)
            self._theme_combo.blockSignals(False)

    def _on_theme_combo(self, text: str) -> None:
        if text.startswith("──"):
            return
        slug = text.lower().replace(" ", "_")
        self._set_theme(slug)

    # ------------------------------------------------------------ shortcuts
    def _shortcut_generate(self) -> None:
        self.requirements._emit_generate()

    def _shortcut_auto(self) -> None:
        self.requirements._emit_auto()

    # ------------------------------------------------------------ persistence
    def closeEvent(self, e) -> None:
        try:
            geo = bytes(self.saveGeometry().toBase64()).decode("ascii")
            self._settings.set("ui", "geometry", value=geo)
            self._settings.save()
        except Exception:
            _log.exception("failed to save window geometry")
        super().closeEvent(e)


def _disable_combo_item(combo: QComboBox, index: int) -> None:
    """Render the item as a non-selectable header row."""
    item = combo.model().item(index)
    if item is not None:
        item.setEnabled(False)
        item.setSelectable(False)
