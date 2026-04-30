"""Neon-glow primary button with hover pulse and click sweep animation."""
from __future__ import annotations

from PySide6.QtCore import Property, QEasingCurve, QPropertyAnimation, QTimer, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import QPushButton

from ..theme import current as theme_current


class CyberButton(QPushButton):
    def __init__(self, text: str = "", *, primary: bool = False, parent=None) -> None:
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._primary = primary
        if primary:
            self.setObjectName("primary")
        self._glow = 0.2
        self._sweep = 0.0
        self._anim_glow = QPropertyAnimation(self, b"glowIntensity", self)
        self._anim_glow.setDuration(220)
        self._anim_glow.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self._anim_sweep = QPropertyAnimation(self, b"sweep", self)
        self._anim_sweep.setDuration(450)
        self._anim_sweep.setEasingCurve(QEasingCurve.Type.OutCubic)
        # Idle pulse so the button looks "alive" even when nothing is happening.
        self._pulse_t = 0.0
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(80)
        self._pulse_timer.timeout.connect(self._tick_pulse)
        if primary:
            self._pulse_timer.start()

    # animatable props ---------------------------------------------------------

    def _get_glow(self) -> float: return self._glow
    def _set_glow(self, v: float) -> None:
        self._glow = float(v); self.update()
    glowIntensity = Property(float, _get_glow, _set_glow)

    def _get_sweep(self) -> float: return self._sweep
    def _set_sweep(self, v: float) -> None:
        self._sweep = float(v); self.update()
    sweep = Property(float, _get_sweep, _set_sweep)

    # interaction -------------------------------------------------------------

    def enterEvent(self, e):
        self._anim_glow.stop()
        self._anim_glow.setStartValue(self._glow)
        self._anim_glow.setEndValue(1.0)
        self._anim_glow.start()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._anim_glow.stop()
        self._anim_glow.setStartValue(self._glow)
        self._anim_glow.setEndValue(0.2)
        self._anim_glow.start()
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        self._anim_sweep.stop()
        self._sweep = 0.0
        self._anim_sweep.setStartValue(0.0)
        self._anim_sweep.setEndValue(1.0)
        self._anim_sweep.start()
        super().mousePressEvent(e)

    def _tick_pulse(self):
        import math
        self._pulse_t += 0.08
        # Modulate glow when not hovered.
        if self._anim_glow.state() != QPropertyAnimation.State.Running:
            self._glow = 0.18 + 0.10 * (0.5 + 0.5 * math.sin(self._pulse_t))
            self.update()

    # paint -------------------------------------------------------------------

    def paintEvent(self, e):
        super().paintEvent(e)
        if self._glow <= 0.0 and self._sweep <= 0.0:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pal = theme_current()
        accent = QColor(pal.accent)
        accent.setAlphaF(min(1.0, 0.35 * self._glow))
        # Glow border.
        pen = QPen(accent, 2)
        p.setPen(pen)
        p.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 4, 4)
        # Click sweep.
        if self._sweep > 0:
            x = int((self._sweep) * self.width())
            grad = QLinearGradient(x - 40, 0, x + 40, 0)
            base = QColor(pal.accent); base.setAlphaF(0.0)
            mid = QColor(pal.accent); mid.setAlphaF(0.55 * (1.0 - self._sweep))
            grad.setColorAt(0.0, base)
            grad.setColorAt(0.5, mid)
            grad.setColorAt(1.0, base)
            p.fillRect(self.rect(), grad)
