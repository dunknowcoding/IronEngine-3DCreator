"""Theme palettes and a global Qt stylesheet generator.

Six starter palettes — three dark (cyber_neon, deep_space, emerald_matrix) and
three light (paper_white, solar_warm, mint_paper). Widgets pull the active
palette from `current()` and refresh on `set_theme()`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    name: str
    accent: str          # primary brand colour — used for borders, focused inputs
    accent_dim: str      # darker accent for pressed buttons and slider tracks
    secondary: str       # contrast accent (numerics, alt highlights)
    bg_window: str       # page background
    bg_panel: str        # card background
    bg_input: str        # form-field background
    border: str          # divider / frame border
    text_primary: str    # body text
    text_dim: str        # secondary text / hints
    text_on_accent: str  # readable text drawn on top of the accent colour
    danger: str
    success: str
    is_light: bool = False


PALETTES: dict[str, Palette] = {
    # ------------------------------------------------------------------ dark
    "cyber_neon": Palette(
        name="cyber_neon",
        accent="#00f0ff",
        accent_dim="#00b6c4",
        secondary="#ff3aa1",
        bg_window="#0a0e1a",
        bg_panel="#14182c",
        bg_input="#1d2240",
        border="#2c3357",
        text_primary="#e0e8ff",
        text_dim="#7c87b5",
        text_on_accent="#0a0e1a",
        danger="#ff5577",
        success="#00ff88",
    ),
    "deep_space": Palette(
        name="deep_space",
        accent="#7c5cff",
        accent_dim="#5a45c2",
        secondary="#36e0ff",
        bg_window="#070612",
        bg_panel="#10122a",
        bg_input="#1a1c3a",
        border="#2a2d52",
        text_primary="#dcdfff",
        text_dim="#7e80a8",
        text_on_accent="#ffffff",
        danger="#ff5e5e",
        success="#5cffb0",
    ),
    "emerald_matrix": Palette(
        name="emerald_matrix",
        accent="#00ff95",
        accent_dim="#00b96d",
        secondary="#ffd84d",
        bg_window="#06120c",
        bg_panel="#0d1f15",
        bg_input="#15301f",
        border="#1f4a30",
        text_primary="#dffff0",
        text_dim="#67a586",
        text_on_accent="#06120c",
        danger="#ff6f6f",
        success="#9bff7a",
    ),
    # ------------------------------------------------------------------ light
    "paper_white": Palette(
        name="paper_white",
        accent="#0066cc",
        accent_dim="#0050a3",
        secondary="#c91560",
        bg_window="#f6f8fc",
        bg_panel="#ffffff",
        bg_input="#eef2f8",
        border="#cdd5e3",
        text_primary="#1a2030",
        text_dim="#637088",
        text_on_accent="#ffffff",
        danger="#cc2244",
        success="#127a3b",
        is_light=True,
    ),
    "solar_warm": Palette(
        name="solar_warm",
        accent="#d2691e",
        accent_dim="#a8511a",
        secondary="#007a5c",
        bg_window="#faf5ec",
        bg_panel="#fffbf2",
        bg_input="#f4ead4",
        border="#dac8a5",
        text_primary="#2a2418",
        text_dim="#7a6c50",
        text_on_accent="#ffffff",
        danger="#bb3a2c",
        success="#0f7a3e",
        is_light=True,
    ),
    "mint_paper": Palette(
        name="mint_paper",
        accent="#00875a",
        accent_dim="#006946",
        secondary="#6f4dbf",
        bg_window="#f0f5f1",
        bg_panel="#ffffff",
        bg_input="#e7eee8",
        border="#c7d3c9",
        text_primary="#1a2620",
        text_dim="#5d7166",
        text_on_accent="#ffffff",
        danger="#b13a3a",
        success="#1e6b3a",
        is_light=True,
    ),
}

DARK_THEMES = ("cyber_neon", "deep_space", "emerald_matrix")
LIGHT_THEMES = ("paper_white", "solar_warm", "mint_paper")

_current = "cyber_neon"


def current() -> Palette:
    return PALETTES[_current]


def set_theme(name: str) -> Palette:
    global _current
    if name in PALETTES:
        _current = name
    return current()


def names() -> list[str]:
    return list(PALETTES.keys())


def stylesheet(p: Palette | None = None) -> str:
    p = p or current()
    return f"""
    QMainWindow, QWidget {{
        background-color: {p.bg_window};
        color: {p.text_primary};
        font-family: 'Segoe UI Variable', 'Segoe UI', 'Cascadia Code', 'Consolas', monospace;
        font-size: 12px;
    }}
    QFrame#card {{
        background-color: {p.bg_panel};
        border: 1px solid {p.border};
        border-radius: 6px;
    }}
    QLabel#sectionTitle {{
        color: {p.accent};
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        font-size: 11px;
    }}
    QLabel#sectionHint, QLabel.dim {{ color: {p.text_dim}; }}
    QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background-color: {p.bg_input};
        color: {p.text_primary};
        border: 1px solid {p.border};
        border-radius: 4px;
        padding: 4px 6px;
        selection-background-color: {p.accent_dim};
        selection-color: {p.text_on_accent};
    }}
    QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QSpinBox:focus,
    QDoubleSpinBox:focus, QComboBox:focus {{
        border: 1px solid {p.accent};
    }}
    QComboBox QAbstractItemView {{
        background-color: {p.bg_panel};
        color: {p.text_primary};
        border: 1px solid {p.border};
        selection-background-color: {p.accent_dim};
        selection-color: {p.text_on_accent};
    }}
    QPushButton {{
        background-color: {p.bg_input};
        color: {p.text_primary};
        border: 1px solid {p.border};
        border-radius: 4px;
        padding: 6px 12px;
    }}
    QPushButton:hover {{
        border: 1px solid {p.accent};
        color: {p.accent};
    }}
    QPushButton:pressed {{
        background-color: {p.accent_dim};
        color: {p.text_on_accent};
    }}
    QPushButton:disabled {{
        color: {p.text_dim};
        border: 1px solid {p.border};
    }}
    QPushButton#primary {{
        background-color: {p.accent_dim};
        color: {p.text_on_accent};
        border: 1px solid {p.accent};
        font-weight: 600;
    }}
    QPushButton#primary:hover {{
        background-color: {p.accent};
        color: {p.text_on_accent};
    }}
    QPushButton#danger {{
        border: 1px solid {p.danger};
        color: {p.danger};
    }}
    QSlider::groove:horizontal {{
        background: {p.bg_input};
        height: 4px;
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {p.accent};
        width: 14px;
        margin: -6px 0;
        border-radius: 7px;
    }}
    QSlider::sub-page:horizontal {{ background: {p.accent_dim}; border-radius: 2px; }}
    QToolBar {{
        background-color: {p.bg_panel};
        border-bottom: 1px solid {p.border};
        spacing: 4px;
        padding: 4px;
    }}
    QStatusBar {{
        background-color: {p.bg_panel};
        color: {p.text_dim};
        border-top: 1px solid {p.border};
    }}
    QMenuBar {{
        background-color: {p.bg_panel};
        color: {p.text_primary};
        border-bottom: 1px solid {p.border};
    }}
    QMenuBar::item:selected {{ background-color: {p.accent_dim}; color: {p.text_on_accent}; }}
    QMenu {{ background-color: {p.bg_panel}; color: {p.text_primary}; border: 1px solid {p.border}; }}
    QMenu::item:selected {{ background-color: {p.accent_dim}; color: {p.text_on_accent}; }}
    QScrollBar:vertical, QScrollBar:horizontal {{
        background: {p.bg_window}; width: 10px; height: 10px;
    }}
    QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
        background: {p.border}; border-radius: 4px;
    }}
    QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
        background: {p.accent_dim};
    }}
    QGroupBox {{
        border: 1px solid {p.border};
        border-radius: 6px;
        margin-top: 16px;
        padding: 8px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 6px;
        color: {p.accent};
        font-weight: 600;
    }}
    QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {p.border}; border-radius: 3px; background: {p.bg_input}; }}
    QCheckBox::indicator:checked {{ background: {p.accent}; border: 1px solid {p.accent}; }}
    QRadioButton::indicator {{ width: 14px; height: 14px; border-radius: 7px; border: 1px solid {p.border}; background: {p.bg_input}; }}
    QRadioButton::indicator:checked {{ background: {p.accent}; }}
    QToolTip {{
        background-color: {p.bg_panel}; color: {p.accent}; border: 1px solid {p.accent};
        border-radius: 3px; padding: 4px;
    }}
    """
