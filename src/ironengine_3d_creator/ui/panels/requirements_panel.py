"""Left-side requirements panel.

Captures the user's free-form description plus the structured controls (shape
combo, point count slider, bbox sizes, legs spinner). Emits two signals:
- generate(PipelineRequest) when the user clicks Generate or Auto.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QPlainTextEdit,
    QSlider, QSpinBox, QVBoxLayout,
)
from PySide6.QtCore import Qt

from ...alignment.defaults import available_templates
from ...alignment.schema import SHAPE_KINDS
from ...core.pipeline import PipelineRequest
from ..widgets.animated_panel import AnimatedPanel
from ..widgets.cyber_button import CyberButton


class RequirementsPanel(AnimatedPanel):
    generate_clicked = Signal(object)  # emits PipelineRequest

    def __init__(self, parent=None) -> None:
        super().__init__(title="Requirements", parent=parent)
        L = self.panel_layout()

        L.addWidget(QLabel("Shape style"))
        self.shape = QComboBox()
        self.shape.addItem("(auto)")
        for s in SHAPE_KINDS:
            self.shape.addItem(s)
        L.addWidget(self.shape)

        L.addWidget(QLabel("Points"))
        self.points = QSlider(Qt.Orientation.Horizontal)
        self.points.setRange(1000, 500_000)
        self.points.setValue(50_000)
        self.points.setTickInterval(50_000)
        self.points_label = QLabel("50,000")
        self.points.valueChanged.connect(lambda v: self.points_label.setText(f"{v:,}"))
        row = QHBoxLayout(); row.addWidget(self.points, 1); row.addWidget(self.points_label)
        L.addLayout(row)

        bbox_row = QHBoxLayout()
        self.bx, self.by, self.bz = QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox()
        for s, label in zip((self.bx, self.by, self.bz), ("X", "Y", "Z")):
            s.setRange(0.05, 20.0); s.setSingleStep(0.1); s.setValue(1.0); s.setDecimals(2)
            bbox_row.addWidget(QLabel(label)); bbox_row.addWidget(s)
        L.addWidget(QLabel("Bounding box (m)"))
        L.addLayout(bbox_row)

        legs_row = QHBoxLayout()
        legs_row.addWidget(QLabel("Legs / supports"))
        self.legs = QSpinBox(); self.legs.setRange(0, 16); self.legs.setValue(0)
        legs_row.addWidget(self.legs); legs_row.addStretch(1)
        L.addLayout(legs_row)

        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel("Seed"))
        self.seed = QSpinBox(); self.seed.setRange(0, 2_000_000_000); self.seed.setValue(0)
        seed_row.addWidget(self.seed); seed_row.addStretch(1)
        L.addLayout(seed_row)

        L.addWidget(QLabel("Description / details"))
        self.details = QPlainTextEdit()
        self.details.setPlaceholderText(
            "e.g. a four-legged stool with deep scratches across the seat"
        )
        self.details.setMinimumHeight(100)
        L.addWidget(self.details, 1)

        btn_row = QHBoxLayout()
        self.auto_btn = CyberButton("✦ Auto")
        self.gen_btn = CyberButton("▶ Generate", primary=True)
        btn_row.addWidget(self.auto_btn); btn_row.addWidget(self.gen_btn)
        L.addLayout(btn_row)

        self.auto_btn.clicked.connect(self._emit_auto)
        self.gen_btn.clicked.connect(self._emit_generate)

    def _build_request(self, *, force_auto: bool) -> PipelineRequest:
        shape = self.shape.currentText()
        if shape == "(auto)":
            shape = None
        return PipelineRequest(
            user_prompt="" if force_auto else self.details.toPlainText().strip(),
            shape_hint=shape,
            n_points=self.points.value(),
            bbox=(self.bx.value(), self.by.value(), self.bz.value()),
            legs=self.legs.value(),
            details=self.details.toPlainText().strip(),
            seed=self.seed.value(),
        )

    def _emit_auto(self) -> None:
        self.gen_btn.setEnabled(False)
        self.auto_btn.setEnabled(False)
        self.generate_clicked.emit(self._build_request(force_auto=True))

    def _emit_generate(self) -> None:
        self.gen_btn.setEnabled(False)
        self.auto_btn.setEnabled(False)
        self.generate_clicked.emit(self._build_request(force_auto=False))

    def re_enable(self) -> None:
        self.gen_btn.setEnabled(True)
        self.auto_btn.setEnabled(True)
