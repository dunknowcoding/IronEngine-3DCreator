"""QThread workers for the LLM/generation pipeline.

The UI must never block on a streaming LLM call. We move every pipeline
invocation into a worker thread that emits Qt signals back to the main thread:
- token(str) for each streaming chunk
- stage(str) for stage transitions
- result(PipelineResult) when complete
- error(str) on failure
"""
from __future__ import annotations

import logging
import time
from typing import Optional

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
