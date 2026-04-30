"""Card-style panel with a soft animated accent border."""
from __future__ import annotations

import math

from PySide6.QtCore import Property, QPropertyAnimation, QTimer, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QFrame, QSizePolicy, QVBoxLayout

from ..theme import current as theme_current


class AnimatedPanel(QFrame):
    def __init__(self, title: str | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self._pulse_t = 0.0
        self._intensity = 0.5
        self._timer = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(12, 12, 12, 12)
        self._layout.setSpacing(8)
        if title:
            from PySide6.QtWidgets import QLabel
            self._title = QLabel(title)
            self._title.setObjectName("sectionTitle")
            self._layout.addWidget(self._title)

    def panel_layout(self) -> QVBoxLayout:
        return self._layout

    def _get_intensity(self) -> float:
        return self._intensity

    def _set_intensity(self, v: float) -> None:
        self._intensity = float(v); self.update()

    intensity = Property(float, _get_intensity, _set_intensity)

    def _tick(self) -> None:
        self._pulse_t += 0.06
        self._intensity = 0.45 + 0.15 * math.sin(self._pulse_t)
        self.update()

    def flash(self) -> None:
        anim = QPropertyAnimation(self, b"intensity", self)
        anim.setDuration(380)
        anim.setStartValue(1.0)
        anim.setEndValue(self._intensity)
        anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)

    def paintEvent(self, e):
        super().paintEvent(e)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pal = theme_current()
        accent = QColor(pal.accent)
        # Light themes need higher base alpha to remain visible against white panels.
        base = 0.42 if pal.is_light else 0.18
        accent.setAlphaF(min(1.0, base * self._intensity))
        pen = QPen(accent, 1)
        p.setPen(pen)
        p.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 6, 6)
        accent.setAlphaF(min(1.0, (0.85 if pal.is_light else 0.7) * self._intensity))
        pen = QPen(accent, 2)
        p.setPen(pen)
        p.drawLine(2, 2, 14, 2)
        p.drawLine(2, 2, 2, 14)
        w, h = self.width(), self.height()
        p.drawLine(w - 14, h - 2, w - 2, h - 2)
        p.drawLine(w - 2, h - 14, w - 2, h - 2)
