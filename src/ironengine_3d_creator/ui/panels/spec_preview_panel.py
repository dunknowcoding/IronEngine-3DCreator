"""Collapsible JSON spec viewer/editor.

Hidden by default. When expanded, lets the user view and tweak the
GenerationSpec before re-running the deterministic generator.
"""
from __future__ import annotations

import json

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPlainTextEdit, QSizePolicy, QToolButton, QVBoxLayout, QWidget,
)
from PySide6.QtCore import Qt

from ...alignment.parser import parse_spec
from ...alignment.schema import GenerationSpec
from ...alignment.validator import normalize
from ..widgets.animated_panel import AnimatedPanel
from ..widgets.cyber_button import CyberButton


class SpecPreviewPanel(AnimatedPanel):
    rerun_clicked = Signal(object)   # emits GenerationSpec

    def __init__(self, parent=None) -> None:
        super().__init__(title="Spec preview", parent=parent)
        L = self.panel_layout()

        head = QHBoxLayout()
        self.toggle = QToolButton()
        self.toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle.setCheckable(True)
        self.toggle.setText("Show / edit JSON")
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.toggled.connect(self._on_toggle)
        head.addWidget(self.toggle); head.addStretch(1)
        L.addLayout(head)

        self.body = QWidget()
        body_l = QVBoxLayout(self.body)
        body_l.setContentsMargins(0, 0, 0, 0)
        self.editor = QPlainTextEdit()
        self.editor.setMinimumHeight(140)
        self.editor.setStyleSheet("font-family: 'Cascadia Code', 'Consolas', monospace; font-size: 11px;")
        body_l.addWidget(self.editor)
        self.warn = QLabel(""); self.warn.setObjectName("sectionHint")
        body_l.addWidget(self.warn)
        self.rerun_btn = CyberButton("⟳ Re-run from spec")
        body_l.addWidget(self.rerun_btn)
        self.rerun_btn.clicked.connect(self._emit_rerun)
        self.body.setVisible(False)
        L.addWidget(self.body)

    def _on_toggle(self, checked: bool) -> None:
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        self.body.setVisible(checked)

    def set_spec(self, spec: GenerationSpec) -> None:
        self.editor.setPlainText(json.dumps(spec.to_json(), indent=2))
        self.warn.setText("")

    def _emit_rerun(self) -> None:
        try:
            spec = parse_spec(self.editor.toPlainText())
        except Exception as e:
            self.warn.setText(f"parse error: {e}")
            return
        spec, warns = normalize(spec)
        if warns:
            self.warn.setText("; ".join(warns))
        else:
            self.warn.setText("")
        self.rerun_clicked.emit(spec)
