"""Entry point for the recipe trainer application."""

import sys

from PySide6.QtWidgets import QApplication

from ui.trainer.trainer_window import TrainerWindow
from utils.pilot_log import start_session_log


def main():
    start_session_log()

    app = QApplication(sys.argv)

    window = TrainerWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
