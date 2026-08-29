"""Redirects stdout/stderr writes to a Qt signal."""

from PySide6.QtCore import QObject, Signal


class LogRedirector(QObject):
    """File-like object that forwards written text to connected slots."""

    text_written = Signal(str)

    def write(self, text):
        self.text_written.emit(str(text))

    def flush(self):
        pass
