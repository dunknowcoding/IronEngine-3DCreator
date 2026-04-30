"""LLM provider configuration panel.

Provider dropdown, endpoint URL, model dropdown (auto-populated for Ollama and
LMStudio, curated for Anthropic / OpenAI), API key (round-trips through the OS
keychain), code-mode toggle, refresh button, and a 'Test connection' button.
"""
from __future__ import annotations

import logging

from PySide6.QtCore import QThread, Signal, QObject
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QSizePolicy, QVBoxLayout,
)

from ...core import secrets
from ...llm import known_models
from ...llm.lmstudio import LMStudioProvider
from ...llm.ollama import OllamaProvider
from ...llm.registry import PROVIDERS, make_provider
from ..widgets.animated_panel import AnimatedPanel
from ..widgets.cyber_button import CyberButton

_log = logging.getLogger(__name__)


class _ModelProbeWorker(QObject):
    """Probes a local LLM server for its installed models on a worker thread."""

    done = Signal(list)
    failed = Signal(str)

    def __init__(self, provider_name: str, endpoint: str) -> None:
        super().__init__()
        self.provider_name = provider_name
        self.endpoint = endpoint

    def run(self) -> None:
        try:
            if self.provider_name == "ollama":
                models = OllamaProvider(model="probe", endpoint=self.endpoint).list_models()
            elif self.provider_name == "lmstudio":
                models = LMStudioProvider(model="probe", endpoint=self.endpoint).list_models()
            else:
                models = []
            self.done.emit(models)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class LLMConfigPanel(AnimatedPanel):
    config_changed = Signal()

    def __init__(self, settings, parent=None) -> None:
        super().__init__(title="LLM configuration", parent=parent)
        self._settings = settings
        self._probe_thread: QThread | None = None
        self._probe_worker: _ModelProbeWorker | None = None
        L = self.panel_layout()

        L.addWidget(QLabel("Provider"))
        self.provider = QComboBox()
        for p in PROVIDERS:
            self.provider.addItem(p)
        self.provider.setCurrentText(settings.get("llm", "provider", default="ollama"))
        L.addWidget(self.provider)

        L.addWidget(QLabel("Endpoint"))
        self.endpoint = QLineEdit()
        L.addWidget(self.endpoint)

        L.addWidget(QLabel("Model"))
        model_row = QHBoxLayout()
        self.model = QComboBox()
        self.model.setEditable(True)
        self.model.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.model.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.refresh_btn = CyberButton("⟲")
        self.refresh_btn.setToolTip("Refresh model list from server / catalog")
        self.refresh_btn.setMaximumWidth(36)
        model_row.addWidget(self.model, 1); model_row.addWidget(self.refresh_btn)
        L.addLayout(model_row)
        self.model_hint = QLabel("")
        self.model_hint.setObjectName("sectionHint")
        L.addWidget(self.model_hint)

        L.addWidget(QLabel("API key (stored in OS keychain)"))
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        L.addWidget(self.api_key)

        self.think_mode = QCheckBox("Reasoning / thinking mode (slower, sometimes higher quality)")
        self.think_mode.setToolTip(
            "When enabled, reasoning models (qwen3.5, deepseek-r1, …) emit a "
            "chain-of-thought before the answer. Off by default because it's "
            "much slower and the answer JSON is what we actually need."
        )
        self.think_mode.setChecked(bool(settings.get("llm", "think_mode", default=False)))
        L.addWidget(self.think_mode)

        self.code_mode = QCheckBox("Code mode (advanced — runs sandboxed Python)")
        self.code_mode.setChecked(bool(settings.get("llm", "code_mode", default=False)))
        L.addWidget(self.code_mode)

        row = QHBoxLayout()
        self.test_btn = CyberButton("Test connection")
        self.save_btn = CyberButton("Save", primary=True)
        row.addWidget(self.test_btn); row.addWidget(self.save_btn)
        L.addLayout(row)

        self.status = QLabel("")
        self.status.setObjectName("sectionHint")
        self.status.setWordWrap(True)
        L.addWidget(self.status)

        # ------------------------------------------------------------------ wiring
        self.provider.currentTextChanged.connect(self._on_provider_changed)
        self.endpoint.editingFinished.connect(self._on_endpoint_changed)
        self.refresh_btn.clicked.connect(self._refresh_models)
        self.test_btn.clicked.connect(self._on_test)
        self.save_btn.clicked.connect(self._on_save)

        self._on_provider_changed(self.provider.currentText())

    # ------------------------------------------------------------------ helpers
    def _populate_model_dropdown(self, items: list[str], *, keep_current: bool = True) -> None:
        current = self.model.currentText().strip()
        self.model.blockSignals(True)
        self.model.clear()
        self.model.addItems(items)
        if keep_current and current and current not in items:
            # Preserve a user-typed model that the catalog doesn't know.
            self.model.insertItem(0, current)
            self.model.setCurrentIndex(0)
        elif current and current in items:
            self.model.setCurrentText(current)
        elif items:
            self.model.setCurrentIndex(0)
        self.model.blockSignals(False)

    def _on_provider_changed(self, name: str) -> None:
        s = self._settings
        eps = s.get("llm", "endpoints", default={})
        models = s.get("llm", "models", default={})
        defaults = {
            "ollama": "http://localhost:11434",
            "lmstudio": "http://localhost:1234/v1",
            "anthropic": "",
            "openai": "",
        }
        self.endpoint.setText(eps.get(name, defaults.get(name, "")))
        self.api_key.setText(secrets.get_api_key(name) or "")

        saved_model = models.get(name, "")
        if name in ("anthropic", "openai"):
            catalog = list(known_models.for_provider(name))
            if saved_model and saved_model not in catalog:
                catalog.insert(0, saved_model)
            self._populate_model_dropdown(catalog, keep_current=False)
            if saved_model:
                self.model.setCurrentText(saved_model)
            elif catalog:
                self.model.setCurrentText(catalog[0])
            self.model_hint.setText(f"{len(catalog)} curated cloud models — type a custom name to override.")
        else:
            # Local provider: seed with whatever we last saved, then probe.
            if saved_model:
                self._populate_model_dropdown([saved_model], keep_current=False)
                self.model.setCurrentText(saved_model)
            else:
                self._populate_model_dropdown([], keep_current=False)
            self.model_hint.setText("Detecting installed models…")
            self._refresh_models()

    def _on_endpoint_changed(self) -> None:
        if self.provider.currentText() in ("ollama", "lmstudio"):
            self._refresh_models()

    def _refresh_models(self) -> None:
        name = self.provider.currentText()
        if name in ("anthropic", "openai"):
            catalog = list(known_models.for_provider(name))
            self._populate_model_dropdown(catalog, keep_current=True)
            self.model_hint.setText(f"{len(catalog)} curated cloud models")
            return

        if self._probe_thread is not None:
            return  # one probe at a time
        endpoint = self.endpoint.text().strip() or "http://localhost"
        _log.debug("probing %s at %s for installed models", name, endpoint)
        thread = QThread(self)
        worker = _ModelProbeWorker(name, endpoint)
        # Keep references on `self`. Without this Python GCs the local
        # `worker` after _refresh_models returns and the thread runs
        # against a dangling C++ object — the symptom is a probe that
        # never delivers its `done` signal.
        self._probe_thread = thread
        self._probe_worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.done.connect(self._on_probe_done)
        worker.failed.connect(self._on_probe_failed)
        worker.done.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_probe_finished)
        self.refresh_btn.setEnabled(False)
        self.model_hint.setText(f"Probing {name}…")
        thread.start()

    def _on_probe_done(self, models: list) -> None:
        _log.debug("probe returned %d models: %s", len(models), models[:5])
        if not models:
            self.model_hint.setText(
                "No models reported by server — type one manually, or check that the server is running."
            )
            return
        self._populate_model_dropdown(list(models), keep_current=True)
        self.model_hint.setText(f"Found {len(models)} installed model(s).")

    def _on_probe_failed(self, msg: str) -> None:
        _log.warning("model probe failed: %s", msg)
        self.model_hint.setText(f"Probe failed: {msg}")

    def _on_probe_finished(self) -> None:
        self._probe_thread = None
        self._probe_worker = None
        self.refresh_btn.setEnabled(True)

    # ------------------------------------------------------------------ actions
    def _on_save(self) -> None:
        name = self.provider.currentText()
        s = self._settings
        s.set("llm", "provider", value=name)
        s.set("llm", "code_mode", value=bool(self.code_mode.isChecked()))
        s.set("llm", "think_mode", value=bool(self.think_mode.isChecked()))
        eps = dict(s.get("llm", "endpoints", default={}))
        eps[name] = self.endpoint.text().strip()
        s.set("llm", "endpoints", value=eps)
        models = dict(s.get("llm", "models", default={}))
        models[name] = self.model.currentText().strip()
        s.set("llm", "models", value=models)
        if self.api_key.text():
            secrets.set_api_key(name, self.api_key.text().strip())
        s.save()
        self.status.setText("Saved.")
        self.config_changed.emit()

    def _on_test(self) -> None:
        try:
            provider = self.build_provider()
        except Exception as e:
            self.status.setText(f"failed: {type(e).__name__}: {e}")
            return
        ok, msg = provider.test()
        self.status.setText(("ok: " if ok else "failed: ") + msg)

    # ------------------------------------------------------------------
    def build_provider(self):
        # Code mode emits raw Python — json_mode would refuse non-JSON output.
        json_mode = not self.code_mode.isChecked()
        return make_provider(
            self.provider.currentText(),
            model=self.model.currentText().strip() or "default",
            endpoint=self.endpoint.text().strip() or None,
            api_key=secrets.get_api_key(self.provider.currentText()),
            think_mode=self.think_mode.isChecked(),
            json_mode=json_mode,
        )

    def is_code_mode(self) -> bool:
        return bool(self.code_mode.isChecked())
