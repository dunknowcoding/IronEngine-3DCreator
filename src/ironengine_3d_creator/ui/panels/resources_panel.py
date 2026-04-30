"""Resource management panel.

Backend dropdown, RAM/VRAM caps, live RAM/VRAM meters, and the export target
folder picker.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QSlider, QVBoxLayout,
)
from PySide6.QtCore import Qt

from ...core import resources as res
from ...core.settings import default_export_dir
from ..widgets.animated_panel import AnimatedPanel
from ..widgets.cyber_button import CyberButton


class ResourcesPanel(AnimatedPanel):
    target_dir_changed = Signal(str)

    def __init__(self, settings, parent=None) -> None:
        super().__init__(title="Resources", parent=parent)
        self._settings = settings
        L = self.panel_layout()

        L.addWidget(QLabel("Acceleration backend"))
        self.backend = QComboBox()
        for b in res.BACKENDS:
            self.backend.addItem(b)
        self.backend.setCurrentText(settings.get("resources", "backend", default="auto"))
        L.addWidget(self.backend)

        self.detect_label = QLabel("")
        self.detect_label.setObjectName("sectionHint")
        L.addWidget(self.detect_label)

        L.addWidget(QLabel("RAM cap (MB)"))
        ram_row = QHBoxLayout()
        self.ram = QSlider(Qt.Orientation.Horizontal)
        self.ram.setRange(512, 32_768)
        self.ram.setValue(settings.get("resources", "ram_cap_mb", default=4096))
        self.ram_lbl = QLabel(f"{self.ram.value()} MB")
        self.ram.valueChanged.connect(lambda v: self.ram_lbl.setText(f"{v} MB"))
        ram_row.addWidget(self.ram, 1); ram_row.addWidget(self.ram_lbl)
        L.addLayout(ram_row)

        L.addWidget(QLabel("VRAM cap (MB)"))
        vram_row = QHBoxLayout()
        self.vram = QSlider(Qt.Orientation.Horizontal)
        self.vram.setRange(256, 24_576)
        self.vram.setValue(settings.get("resources", "vram_cap_mb", default=2048))
        self.vram_lbl = QLabel(f"{self.vram.value()} MB")
        self.vram.valueChanged.connect(lambda v: self.vram_lbl.setText(f"{v} MB"))
        vram_row.addWidget(self.vram, 1); vram_row.addWidget(self.vram_lbl)
        L.addLayout(vram_row)

        self.usage = QLabel("…")
        self.usage.setObjectName("sectionHint")
        L.addWidget(self.usage)

        L.addWidget(QLabel("Export target folder"))
        target_row = QHBoxLayout()
        target = settings.get("export", "target_dir") or str(default_export_dir())
        self.target = QLineEdit(target)
        self.browse = CyberButton("…")
        self.browse.setMaximumWidth(36)
        target_row.addWidget(self.target, 1); target_row.addWidget(self.browse)
        L.addLayout(target_row)

        self.save_btn = CyberButton("Save", primary=True)
        L.addWidget(self.save_btn)

        # Behaviour wiring -----------------------------------------------------
        self.browse.clicked.connect(self._pick_dir)
        self.save_btn.clicked.connect(self._save)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._refresh_usage)
        self._timer.start()
        self._refresh_detect()

    def _refresh_detect(self) -> None:
        rep = res.detect_backends(prefer_gpu=True)
        bits = []
        if rep.cuda_cupy: bits.append("CuPy")
        if rep.cuda_torch: bits.append("Torch+CUDA")
        if rep.taichi: bits.append("Taichi")
        if not bits: bits.append("CPU only")
        self.detect_label.setText("Detected: " + ", ".join(bits) + f" — auto = {rep.chosen}")

    def _refresh_usage(self) -> None:
        rss = res.process_rss_mb()
        sram_used, sram_total = res.system_ram_mb()
        vram = res.vram_mb()
        bits = [f"proc {rss:.0f}MB", f"sys {sram_used:.0f}/{sram_total:.0f}MB"]
        if vram is not None:
            used, total = vram
            bits.append(f"vram {used:.0f}/{total:.0f}MB")
        self.usage.setText(" · ".join(bits))

    def _pick_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Choose export folder", self.target.text() or str(Path.home()))
        if d:
            self.target.setText(d)

    def _save(self) -> None:
        s = self._settings
        s.set("resources", "backend", value=self.backend.currentText())
        s.set("resources", "ram_cap_mb", value=int(self.ram.value()))
        s.set("resources", "vram_cap_mb", value=int(self.vram.value()))
        s.set("export", "target_dir", value=self.target.text().strip() or None)
        s.save()
        # Re-pin the active backend so the next generation actually uses it.
        active = res.set_active_backend(self.backend.currentText(), prefer_gpu=True)
        self.detect_label.setText(self.detect_label.text().split(" — ")[0] + f" — active = {active}")
        self.target_dir_changed.emit(self.target.text().strip())

    # ------------------------------------------------------------------
    def chosen_backend(self) -> str:
        return res.resolve_backend(self.backend.currentText(), prefer_gpu=True)

    def export_target(self) -> Path:
        t = self.target.text().strip()
        return Path(t) if t else default_export_dir()
