"""QThread workers for the LLM/generation pipeline.

The UI must never block on a streaming LLM call. We move every pipeline
invocation into a worker thread that emits Qt signals back to the main thread:
- token(str) for each streaming chunk
- stage(str) for stage transitions
- result(PipelineResult) when complete
- error(str) on failure
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from PySide6.QtCore import QObject, QThread, Signal

from ..core.pipeline import PipelineRequest, PipelineResult, run
from ..llm.base import LLMProvider

_log = logging.getLogger(__name__)


class GenerationWorker(QObject):
    token = Signal(str)
    stage = Signal(str)
    result = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, req: PipelineRequest, provider: Optional[LLMProvider]) -> None:
        super().__init__()
        self._req = req
        self._provider = provider
        self._first_token_seen = False
        self._t_start = 0.0

    def _on_token(self, chunk: str) -> None:
        if not self._first_token_seen:
            self._first_token_seen = True
            self.stage.emit(f"streaming · first token in {time.perf_counter() - self._t_start:.1f}s")
        self.token.emit(chunk)

    def run(self) -> None:
        self._t_start = time.perf_counter()
        try:
            _log.warning("pipeline starting (provider=%s, code_mode=%s, prompt=%r)",
                         getattr(self._provider, "name", None),
                         self._req.code_mode,
                         self._req.user_prompt[:60])
            res = run(
                self._req,
                self._provider,
                on_token=self._on_token,
                on_stage=lambda s: (_log.warning("stage: %s", s), self.stage.emit(s))[-1],
            )
            _log.warning("pipeline produced %d points", res.generation.positions.shape[0])
            self.result.emit(res)
        except Exception as e:
            _log.exception("pipeline failed")
            self.error.emit(f"{type(e).__name__}: {e}")
        finally:
            self.finished.emit()


def start_worker(parent: QObject, req: PipelineRequest, provider: Optional[LLMProvider]) -> tuple[QThread, GenerationWorker]:
    """Construct a worker + thread, wire ownership, and start. Caller connects signals."""
    thread = QThread(parent)
    worker = GenerationWorker(req, provider)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    return thread, worker


# ---------------------------------------------------------------- mesh worker


class MeshWorker(QObject):
    """Run ball-pivot reconstruction off the main thread. Ball-pivot can take
    several seconds on large clouds — doing it inline freezes the UI."""

    done = Signal(object)         # ReconstructedMesh
    error = Signal(str)
    finished = Signal()

    def __init__(self, positions, *, radius: float = 0.0) -> None:
        super().__init__()
        self._positions = positions
        # 0.0 means "auto" — reconstruct.py picks a radius from point spacing.
        self._radius = float(radius)

    def run(self) -> None:
        try:
            from ..generation.reconstruct import reconstruct
            self.done.emit(reconstruct(self._positions, radius=self._radius))
        except Exception as e:
            _log.exception("mesh reconstruction failed")
            self.error.emit(f"{type(e).__name__}: {e}")
        finally:
            self.finished.emit()


def start_mesh_worker(parent: QObject, positions, *, radius: float = 0.0) -> tuple[QThread, MeshWorker]:
    thread = QThread(parent)
    worker = MeshWorker(positions, radius=radius)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    return thread, worker


# ------------------------------------------------------------- handoff worker


@dataclass
class HandoffResult:
    """Artifacts written by HandoffWorker for one 'Send to SceneEditor' run."""
    ply_path: Path
    glb_path: Optional[Path] = None
    manifest_path: Optional[Path] = None
    handoff_path: Optional[Path] = None
    mesh_stats: Optional[dict] = None
    warnings: list[str] = field(default_factory=list)


def write_handoff_pointer(directory: Path, manifest_path: Path) -> Path:
    """Write the SceneEditor handoff pointer next to the exported triple.

    Thin local helper: core/exporter.py + core/manifest.py own the artifact
    writers, but the handoff.json pointer is a UI-flow contract (consumed by
    IronEngine-SceneEditor on launch), so it lives here. In the default
    configuration `directory` is %LOCALAPPDATA%/IronEngine/user_models.
    """
    payload = {
        "manifest": str(Path(manifest_path).resolve()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = Path(directory) / "handoff.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


class HandoffWorker(QObject):
    """Write the PLY + GLB + .iemodel.json handoff triple off the main thread.

    Ball-pivot reconstruction and the per-point PLY writer take seconds on
    large clouds — running them inline in the button handler freezes the UI
    (the preview path already uses MeshWorker for the same reason). Only the
    subprocess launch happens back on the main thread, in the `done` slot.
    """

    stage = Signal(str)
    done = Signal(object)         # HandoffResult
    error = Signal(str)
    finished = Signal()

    def __init__(self, spec, positions: np.ndarray, colors: Optional[np.ndarray], base: Path,
                 labels: Optional[np.ndarray] = None) -> None:
        super().__init__()
        self._spec = spec
        self._positions = positions
        self._colors = colors
        # Per-point primitive indices from GenerationResult — enable measured
        # per-part AABBs / albedos in the iemodel/2 manifest.
        self._labels = labels
        # Base path without suffix; <base>.ply/.glb/.iemodel.json are written.
        self._base = Path(base)

    def run(self) -> None:
        from ..core import exporter, manifest as iemanifest

        warnings: list[str] = []
        try:
            self.stage.emit("writing point cloud…")
            ply_path = exporter.write_ply(self._base.with_suffix(".ply"),
                                          self._positions, self._colors)

            glb_path: Optional[Path] = None
            mesh_stats: Optional[dict] = None
            try:
                # Spec-driven path: exact analytic meshes with PBR materials,
                # baked baseColorTexture and UVs (F5). Stats come from the
                # analytic parts so the manifest records analytic=true.
                parts = None
                if self._spec is not None and getattr(self._spec, "primitives", None):
                    from ..generation.analytic_mesh import build_spec_meshes
                    try:
                        parts = build_spec_meshes(self._spec) or None
                    except Exception:
                        _log.exception("analytic mesh build failed; falling back to reconstruction")
                if parts is not None:
                    mesh_stats = {
                        "vertices": int(sum(p.vertices.shape[0] for p in parts)),
                        "faces": int(sum(p.faces.shape[0] for p in parts)),
                        "has_uvs": True,
                        "has_vertex_colors": True,
                        "analytic": True,
                    }
                    self.stage.emit("writing analytic GLB mesh…")
                else:
                    self.stage.emit("reconstructing mesh…")
                    from ..generation.reconstruct import reconstruct
                    rec = reconstruct(self._positions)
                    mesh_stats = {"vertices": int(rec.positions.shape[0]),
                                  "faces": int(rec.indices.size // 3),
                                  "analytic": False}
                    self.stage.emit("writing GLB mesh…")
                glb_path = exporter.write_glb(self._base.with_suffix(".glb"),
                                              self._positions, self._colors,
                                              spec=self._spec)
            except Exception as e:
                _log.exception("handoff mesh export failed; manifest will have mesh=null")
                warnings.append(f"mesh/GLB export skipped: {type(e).__name__}: {e}")

            manifest_path: Optional[Path] = None
            handoff_path: Optional[Path] = None
            try:
                self.stage.emit("writing manifest…")
                manifest = iemanifest.build_manifest(
                    self._spec, self._positions, self._colors,
                    mesh_path=glb_path, point_cloud_path=ply_path, mesh_stats=mesh_stats,
                    labels=self._labels,
                )
                manifest_path = self._base.with_suffix(".iemodel.json")
                iemanifest.write_manifest(manifest_path, manifest)
                handoff_path = write_handoff_pointer(self._base.parent, manifest_path)
            except Exception as e:
                _log.exception("failed to write handoff manifest")
                warnings.append(f"manifest/handoff write failed: {type(e).__name__}: {e}")

            self.done.emit(HandoffResult(
                ply_path=ply_path, glb_path=glb_path, manifest_path=manifest_path,
                handoff_path=handoff_path, mesh_stats=mesh_stats, warnings=warnings,
            ))
        except Exception as e:
            _log.exception("handoff export failed")
            self.error.emit(f"{type(e).__name__}: {e}")
        finally:
            self.finished.emit()


def start_handoff_worker(parent: QObject, spec, positions, colors, base: Path,
                         labels: Optional[np.ndarray] = None) -> tuple[QThread, HandoffWorker]:
    thread = QThread(parent)
    worker = HandoffWorker(spec, positions, colors, base, labels=labels)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    return thread, worker
