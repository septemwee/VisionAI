"""Shared visual theme for the trainer application.

Single source of truth for the palette, typography, spacing and the global
stylesheet. Pages build their UI through the helper factories here instead
of carrying private stylesheets, so the look stays consistent everywhere.
The palette is the original VisionAI light theme (white surfaces, blue
primary, Poppins with a system fallback).
"""

from pathlib import Path

from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QFrame, QLabel, QPushButton

from utils.resource_path import resource_path

_LOADED_FONT_COUNT = None


def load_ui_fonts():
    """Load bundled Poppins faces once for both source and frozen builds."""
    global _LOADED_FONT_COUNT
    if _LOADED_FONT_COUNT is not None:
        return _LOADED_FONT_COUNT
    loaded = 0
    for filename in (
        "Poppins-Regular.ttf", "Poppins-Medium.ttf",
        "Poppins-SemiBold.ttf", "Poppins-Bold.ttf",
    ):
        path = Path(resource_path(f"assets/fonts/{filename}"))
        if path.is_file() and QFontDatabase.addApplicationFont(str(path)) >= 0:
            loaded += 1
    _LOADED_FONT_COUNT = loaded
    return loaded

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

BG = "#F5F7FB"
SURFACE = "#F8FAFC"
BORDER = "#E2E8F0"
TEXT = "#0F172A"
TEXT_SECONDARY = "#334155"
MUTED = "#64748B"
DISABLED = "#9CA3AF"

PRIMARY = "#2563EB"
PRIMARY_HOVER = "#1D4ED8"
PRIMARY_PRESSED = "#1E40AF"
PRIMARY_SOFT = "#EFF6FF"
PRIMARY_SOFT_BORDER = "#C7D7F8"
PRIMARY_SOFT_PRESSED = "#DBEAFE"

SUCCESS = "#10B981"
SUCCESS_SOFT = "#D1FAE5"
SUCCESS_TEXT = "#047857"
WARNING = "#F59E0B"
WARNING_SOFT = "#FEF3C7"
WARNING_TEXT = "#B45309"
DANGER = "#EF4444"
DANGER_SOFT = "#FEF2F2"
DANGER_SOFT_BORDER = "#FECACA"

FONT_STACK = "'Poppins','Segoe UI',Arial,sans-serif"
MONO_STACK = "'Consolas','Courier New',monospace"

# ---------------------------------------------------------------------------
# Spacing (8pt grid)
# ---------------------------------------------------------------------------

SPACE_XS = 4
SPACE_S = 8
SPACE_M = 12
SPACE_L = 16
SPACE_XL = 24

# ---------------------------------------------------------------------------
# Global stylesheet (applied once on the TrainerWindow)
# ---------------------------------------------------------------------------

TRAINER_STYLESHEET = f"""
QWidget {{
    background: {BG};
    font-family: {FONT_STACK};
    font-size: 13px;
    color: {TEXT_SECONDARY};
}}

QWidget#stepRail, QWidget#qt_scrollarea_viewport {{ background: transparent; }}
QScrollArea {{ background: transparent; border: none; }}
QStackedWidget#trainerPages {{ background: transparent; }}

QLabel {{
    background: transparent;
}}

QLabel[role="title"] {{ font-size: 24px; font-weight: 700; color: {TEXT}; }}
QLabel[role="section"] {{ font-size: 15px; font-weight: 600; color: {TEXT}; }}
QLabel[role="body"] {{ font-size: 13px; color: {TEXT_SECONDARY}; }}
QLabel[role="caption"] {{ font-size: 11px; color: {MUTED}; }}
QLabel[role="statValue"] {{ font-size: 13px; font-weight: 600; color: {TEXT}; }}

QLabel#contextChip {{
    background: {PRIMARY_SOFT};
    color: {PRIMARY_HOVER};
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 13px;
    font-weight: 600;
}}

QLabel#productEyebrow {{
    color: {PRIMARY}; font-size: 10px; font-weight: 700; letter-spacing: 1px;
}}
QLabel#productSubtitle {{ color: {MUTED}; font-size: 12px; }}

QFrame#workflowCard, QFrame#trainerFooter {{
    background: #FFFFFF;
    border: 1px solid {BORDER};
    border-radius: 12px;
}}

QLabel#preview {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    color: {MUTED};
    font-weight: 600;
}}

QFrame#card {{
    background: #FFFFFF;
    border: 1px solid {BORDER};
    border-radius: 12px;
}}
QFrame#card QLabel {{ border: none; }}

QPushButton {{
    background: {PRIMARY};
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    min-height: 18px;
    padding: 9px 18px;
    font-size: 13px;
    font-weight: 600;
}}
QPushButton:hover {{ background: {PRIMARY_HOVER}; }}
QPushButton:pressed {{ background: {PRIMARY_PRESSED}; }}
QPushButton:disabled {{ background: {BORDER}; color: {DISABLED}; }}

QPushButton[variant="secondary"] {{
    background: {BG};
    color: {PRIMARY};
    border: 1px solid {PRIMARY_SOFT_BORDER};
}}
QPushButton[variant="secondary"]:hover {{
    background: {PRIMARY_SOFT};
    border-color: {PRIMARY};
}}
QPushButton[variant="secondary"]:pressed {{
    background: {PRIMARY_SOFT_PRESSED};
}}
QPushButton[variant="secondary"]:disabled {{
    background: {SURFACE};
    color: {DISABLED};
    border-color: {BORDER};
}}

QPushButton[variant="danger"] {{
    background: {BG};
    color: {DANGER};
    border: 1px solid {DANGER_SOFT_BORDER};
}}
QPushButton[variant="danger"]:hover {{
    background: {DANGER_SOFT};
    border-color: {DANGER};
}}
QPushButton[variant="danger"]:pressed {{ background: #FEE2E2; }}
QPushButton[variant="danger"]:disabled {{
    background: {SURFACE};
    color: {DISABLED};
    border-color: {BORDER};
}}

QPushButton[variant="ghost"] {{
    background: transparent;
    color: {MUTED};
    border: none;
    padding: 9px 12px;
}}
QPushButton[variant="ghost"]:hover {{ background: {SURFACE}; color: {TEXT_SECONDARY}; }}

QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 8px 10px;
    color: {TEXT};
    selection-background-color: {PRIMARY};
    selection-color: #FFFFFF;
}}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ min-height: 20px; }}
QLineEdit[compactField="true"], QComboBox[compactField="true"],
QTextEdit[compactField="true"] {{
    min-height: 20px;
    padding: 3px 8px;
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QComboBox:focus,
QSpinBox:focus, QDoubleSpinBox:focus {{
    background: {BG};
    border: 1px solid {PRIMARY};
}}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {BG};
    border: 1px solid {BORDER};
    selection-background-color: {PRIMARY_SOFT};
    selection-color: {PRIMARY_HOVER};
}}

QListWidget {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 4px;
    color: {TEXT_SECONDARY};
}}
QListWidget::item {{ padding: 6px 8px; border-radius: 6px; }}
QListWidget::item:hover {{ background: {PRIMARY_SOFT}; }}
QListWidget::item:selected {{
    background: {PRIMARY_SOFT_PRESSED};
    color: {PRIMARY_PRESSED};
}}

QProgressBar {{
    background: #F3F4F6;
    border: none;
    border-radius: 7px;
    min-height: 14px;
    max-height: 14px;
    text-align: center;
    color: {TEXT_SECONDARY};
    font-size: 10px;
}}
QProgressBar::chunk {{ background: {PRIMARY}; border-radius: 7px; }}

QSlider::groove:horizontal {{
    height: 6px;
    background: {BORDER};
    border-radius: 3px;
}}
QSlider::sub-page:horizontal {{
    background: {PRIMARY_SOFT_PRESSED};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
    background: {PRIMARY};
}}
QSlider::handle:horizontal:hover {{ background: {PRIMARY_HOVER}; }}

QToolTip {{
    background: {TEXT};
    color: #FFFFFF;
    border: none;
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 12px;
}}

QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px 3px;
}}
QScrollBar::handle:vertical {{
    background: #D1D5DB;
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {DISABLED}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 3px 2px;
}}
QScrollBar::handle:horizontal {{
    background: #D1D5DB;
    border-radius: 4px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{ background: {DISABLED}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

QPlainTextEdit#logView {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 8px;
    color: {TEXT};
    font-family: {MONO_STACK};
    font-size: 12px;
}}

QPlainTextEdit#summaryView {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 10px;
    color: {TEXT_SECONDARY};
    font-size: 13px;
}}
"""

# ---------------------------------------------------------------------------
# Widget factories
# ---------------------------------------------------------------------------


def title_label(text=""):
    """Page title (26px bold)."""
    label = QLabel(text)
    label.setProperty("role", "title")
    return label


def section_label(text=""):
    """Card / group heading (15px semibold)."""
    label = QLabel(text)
    label.setProperty("role", "section")
    return label


def caption_label(text=""):
    """Muted helper text (11px)."""
    label = QLabel(text)
    label.setProperty("role", "caption")
    label.setWordWrap(True)
    return label


def make_button(text="", variant="primary"):
    """Create a themed button.

    Variants: ``primary`` (filled blue), ``secondary`` (outline blue),
    ``danger`` (outline red), ``ghost`` (text only).
    """
    button = QPushButton(text)
    button.setProperty("variant", variant)
    return button


def card_frame():
    """White rounded card with the standard border."""
    frame = QFrame()
    frame.setObjectName("card")
    return frame
