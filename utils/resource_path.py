"""Resolution of paths to bundled resources.

When the application is frozen with PyInstaller, read-only resources live in
the temporary bundle directory (``sys._MEIPASS``); otherwise they live in the
project root.
"""

import sys
from pathlib import Path


def resource_path(relative_path: str) -> Path:
    """Return the absolute path of a bundled resource."""
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / relative_path

    return Path(__file__).resolve().parent.parent / relative_path
