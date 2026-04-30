"""Real-time vertex editing panel.

Active only after a generation has produced a point cloud. Houses the brush
mode buttons, brush size slider, paint color picker, and undo/redo.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup, QColorDialog, QHBoxLayout, QLabel, QSlider, QVBoxLayout,
)
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt

from ..widgets.animated_panel import AnimatedPanel
from ..widgets.cyber_button import CyberButton


class EditingPanel(AnimatedPanel):
    mode_changed = Signal(str)
    brush_changed = Signal(float)
    paint_color_changed = Signal(tuple)  # (r, g, b)
    undo_clicked = Signal()
    redo_clicked = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(title="Edit", parent=parent)
        L = self.panel_layout()

        modes = [("none", "Off"), ("brush", "Move"), ("warp", "Warp"), ("paint", "Paint"), ("smooth", "Smooth")]
        row = QHBoxLayout()
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        for key, label in modes:
            btn = CyberButton(label)
            btn.setCheckable(True)
            btn.setProperty("mode_key", key)
            row.addWidget(btn)
            self.group.addButton(btn)
            btn.clicked.connect(self._on_mode_clicked)
            if key == "none":
                btn.setChecked(True)
        L.addLayout(row)

        L.addWidget(QLabel("Brush radius (NDC)"))
        self.brush = QSlider(Qt.Orientation.Horizontal)
        self.brush.setRange(2, 50)
        self.brush.setValue(12)
        self.brush.valueChanged.connect(lambda v: self.brush_changed.emit(v / 100.0))
        L.addWidget(self.brush)

        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("Paint color"))
        self.color_btn = CyberButton("Pick…")
        self.color_btn.clicked.connect(self._pick_color)
        color_row.addWidget(self.color_btn)
        self._paint_rgb = (1.0, 0.5, 0.2)
        L.addLayout(color_row)

        hist = QHBoxLayout()
        self.undo_btn = CyberButton("↶ Undo")
        self.redo_btn = CyberButton("↷ Redo")
        hist.addWidget(self.undo_btn); hist.addWidget(self.redo_btn)
        L.addLayout(hist)
        self.undo_btn.clicked.connect(self.undo_clicked.emit)
        self.redo_btn.clicked.connect(self.redo_clicked.emit)

    def _on_mode_clicked(self) -> None:
        btn = self.group.checkedButton()
        if btn is not None:
            self.mode_changed.emit(btn.property("mode_key"))

    def _pick_color(self) -> None:
        c = QColorDialog.getColor(QColor(int(self._paint_rgb[0] * 255),
                                         int(self._paint_rgb[1] * 255),
                                         int(self._paint_rgb[2] * 255)))
        if c.isValid():
            self._paint_rgb = (c.redF(), c.greenF(), c.blueF())
            self.paint_color_changed.emit(self._paint_rgb)

    def current_brush(self) -> float:
        return self.brush.value() / 100.0

    def current_paint_rgb(self) -> tuple[float, float, float]:
        return self._paint_rgb
