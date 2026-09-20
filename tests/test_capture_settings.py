import os
import subprocess
import sys
from pathlib import Path


def test_capture_settings_survive_process_restart(tmp_path):
    script = '''
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings
import ui.inspection.status_widget as module
from ui.inspection.overlay import OverlayWindow
app = QApplication([])
module.QSettings = lambda *args: QSettings(sys.argv[1], QSettings.IniFormat)
overlay = OverlayWindow()
widget = module.StatusWidget(overlay)
if sys.argv[2] == "write":
    widget.capture_width.setValue(1234)
    widget.capture_height.setValue(987)
    widget.capture_zoom.setValue(43.2)
    widget.capture_x.setValue(-31)
    widget.capture_y.setValue(22)
    widget.capture_service.set_folder(sys.argv[3])
    widget._toggle_capture_geometry(False)
    widget.capture_width.lineEdit().setText("1357")
else:
    assert widget.capture_width.value() == 1357
    assert widget.capture_height.value() == 987
    assert widget.capture_zoom.value() == 43.2
    assert widget.capture_x.value() == -31
    assert widget.capture_y.value() == 22
    assert not widget.capture_size_toggle.isChecked()
    assert str(widget.capture_service.folder) == sys.argv[3]
widget.close()
overlay.close()
'''
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    for mode in ("write", "read"):
        result = subprocess.run([sys.executable, "-c", script, str(tmp_path / "prefs.ini"),
                                 mode, str(tmp_path)], cwd=Path(__file__).resolve().parents[1],
                                env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
