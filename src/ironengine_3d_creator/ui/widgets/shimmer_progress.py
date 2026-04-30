"""Indeterminate shimmer progress bar with stage label."""
from __future__ import annotations

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter
from PySide6.QtWidgets import QWidget

from ..theme import current as theme_current


class ShimmerProgress(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(10)
        self._t = 0.0
        self._active = False
        self._stage = ""
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)

    def start(self, stage: str = "") -> None:
        self._active = True
        self._stage = stage
        self._timer.start()
        self.update()

    def set_stage(self, stage: str) -> None:
        self._stage = stage
        self.update()

    def stop(self) -> None:
        self._active = False
        self._timer.stop()
        self._stage = ""
        self.update()

    def _tick(self) -> None:
        self._t = (self._t + 0.018) % 1.0
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pal = theme_current()
        bg = QColor(pal.bg_input)
        p.fillRect(self.rect(), bg)
        if self._active:
            w = self.width(); h = self.height()
            sweep_x = int((self._t * 2.0 - 1.0) * w)
            grad = QLinearGradient(sweep_x - w * 0.3, 0, sweep_x + w * 0.3, 0)
            base = QColor(pal.accent); base.setAlphaF(0.0)
            mid = QColor(pal.accent); mid.setAlphaF(0.7)
            grad.setColorAt(0.0, base)
            grad.setColorAt(0.5, mid)
            grad.setColorAt(1.0, base)
            p.fillRect(self.rect(), grad)
        if self._stage:
            p.setPen(QColor(pal.text_dim))
            p.drawText(self.rect().adjusted(8, 0, -8, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self._stage)
