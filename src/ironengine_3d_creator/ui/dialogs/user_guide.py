"""User guide + shortcuts dialogs.

Both dialogs are built from a Markdown source so they stay in sync with
README / SOUL.md and can be edited without touching widget code.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QTextBrowser, QVBoxLayout,
)

USER_GUIDE_MD = """
# IronEngine 3D Creator — User Guide

## 1. Generate a model

* **Auto** (`Ctrl + Shift + G`) — instant model from a curated template
  (chair, vase, fence, archway, …). No LLM required.
* **Generate** (`Ctrl + G`) — type a description in the *Details* box on
  the left, set point count + bounding box, hit Generate. The active LLM
  provider (Ollama / LMStudio / Anthropic / OpenAI, set on the right
  panel) is consulted; the response is structured into primitives via
  `SOUL.md` principles, repaired by the integrity layer (legs to floor,
  parts to neighbours, fence bars don't collapse), then turned into a
  point cloud.

If the LLM is offline or returns invalid JSON, the system falls back to
the auto template — generation never blocks.

## 2. Viewport controls

| Action | Mouse / key |
|---|---|
| Pan | left-drag |
| Rotate around the cloud's volume centre | right-drag, middle-drag, or shift+left-drag |
| Zoom | mouse wheel |
| Frame all | `F` key, or the `⌂ Frame` toolbar button |
| Render: points / mesh / both | `1` / `2` / `3`, or toolbar buttons |
| Color: textured / plain | toolbar buttons |
| Wireframe (mesh only) | toolbar toggle |
| Increase / decrease point size | `+` / `−` toolbar buttons |
| Help / shortcuts | `F1` |

The rotation pivot is **always** the cloud's volume centre — pans don't
move the rotation axis. Hit `F` to recenter and zoom-fit at any time.

## 3. Mesh preview

Switch render mode to **Mesh** or **Both** to see a triangle mesh of the
cloud. Reconstruction runs on a worker thread (the shimmer progress bar
shows it's busy); your UI stays responsive. Methods tried in order:

1. **Ball-pivot** — preserves the cloud's actual openings (gaps between
   chair legs, fence bars, archway interior).
2. **Poisson** — fallback when ball-pivot can't bridge sparse regions.
   Watertight, but may close openings.
3. **Convex hull** — last-resort fallback if Open3D isn't installed.

The first build can take a few seconds on big clouds (>100k points).
Subsequent toggles are instant — the mesh is cached.

## 4. Vertex editing

In the *Edit* panel (right side), pick one of:

* **Move** — push points along a direction.
* **Warp** — radial scale around the cursor.
* **Paint** — colour points with the picker colour.
* **Smooth** — average each point with its neighbours.

Drag in the viewport to apply. The mesh becomes stale during a drag and
is rebuilt automatically when you release the mouse (only if you're in
Mesh / Both mode).

`Ctrl + Z` / `Ctrl + Y` undo and redo (32 steps).

## 5. Export

`File → Export…` (or `Ctrl + E`) writes one of:

* **PLY / PCD** — the raw point cloud. Consumed by IronEngine-Sim and
  SceneEditor directly.
* **GLB / OBJ** — the reconstructed mesh.

`File → Save session` (`Ctrl + S`) writes a `.iecreator.json` with the
prompt, spec, seed and edit history — open it later to iterate.

## 6. Send to SceneEditor

The toolbar's `⮕ Send to SceneEditor` button writes the current cloud
to the user-models library (`%LOCALAPPDATA%\\IronEngine\\user_models`)
and launches SceneEditor. The asset browser there scans the same
folder, so refreshing it surfaces your new model immediately. Drag it
into a scene to attach it as a `point_cloud` component.

## 7. Resource panel

* **Acceleration backend** — auto-detects CUDA via CuPy / Torch, falls
  back to Taichi (cross-vendor), then NumPy. Override if you want.
* **RAM / VRAM caps** — advisory limits; generation is throttled or
  warns before exceeding them.
* **Export target** — defaults to the SceneEditor library folder.
  Change to a project-local `assets/models/` if you prefer.

## 8. LLM panel

* **Provider** — Ollama / LMStudio (local) or Anthropic / OpenAI (cloud).
* **Endpoint** — server URL. Refresh `⟲` re-probes for installed models.
* **Model** — editable dropdown. Local providers list installed models;
  cloud providers list a curated catalogue.
* **API key** — kept in the OS keychain; never in plain JSON.
* **Reasoning / thinking mode** — when enabled, `<think>…</think>`
  blocks emitted by reasoning models (qwen3.5, deepseek-r1, …) appear
  in the token-stream pane in dim italics. The JSON parser strips them
  before parsing.
* **Code mode** — advanced. The model emits a Python script that's
  executed in a strict AST sandbox to produce points directly.
"""

SHORTCUTS_MD = """
# Keyboard shortcuts

| Action | Shortcut |
|---|---|
| Generate from prompt | `Ctrl + G` |
| Auto-generate (no LLM) | `Ctrl + Shift + G` |
| Save session | `Ctrl + S` |
| Open session | `Ctrl + O` |
| Export | `Ctrl + E` |
| Undo / Redo | `Ctrl + Z` / `Ctrl + Y` |
| Render mode: points | `1` |
| Render mode: mesh | `2` |
| Render mode: both | `3` |
| Frame all | `F` |
| Help (this dialog) | `F1` |
"""


class _MarkdownDialog(QDialog):
    def __init__(self, title: str, markdown: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(720, 640)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12); layout.setSpacing(8)
        view = QTextBrowser()
        view.setOpenExternalLinks(True)
        f = QFont("Segoe UI Variable"); f.setPointSize(10)
        view.setFont(f)
        view.setMarkdown(markdown.strip())
        layout.addWidget(view, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


def show_user_guide(parent=None) -> None:
    dlg = _MarkdownDialog("User Guide", USER_GUIDE_MD, parent)
    dlg.exec()


def show_shortcuts(parent=None) -> None:
    dlg = _MarkdownDialog("Keyboard shortcuts", SHORTCUTS_MD, parent)
    dlg.exec()
