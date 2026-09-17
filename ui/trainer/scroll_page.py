"""Scrollable page wrapper for the trainer wizard."""

from PySide6.QtWidgets import QFrame, QScrollArea, QWidget


def scrollable_page(page: QWidget) -> QScrollArea:
    """Wrap ``page`` in a frameless scroll area.

    Qt layouts cannot shrink below their widgets' minimum sizes; without
    a scroll area the stacked wizard pages clip or overlap their content
    when the window is smaller than the page. Wrapping every page keeps
    the whole wizard usable at any window size.
    """
    scroll = QScrollArea()
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setWidgetResizable(True)
    scroll.setWidget(page)
    return scroll
