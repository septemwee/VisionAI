"""Entry point for the recipe trainer application."""

import sys

from PySide6.QtWidgets import QApplication

from ui.trainer.trainer_window import TrainerWindow


def main():
    app = QApplication(sys.argv)

    window = TrainerWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
