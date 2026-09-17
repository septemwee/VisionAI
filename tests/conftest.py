"""Shared pytest configuration: headless Qt and a temp recipes directory."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def temp_recipes_dir(tmp_path, monkeypatch):
    """Point RecipeService at a temporary recipes directory."""
    monkeypatch.setattr("services.recipe_service.RECIPES_DIR", tmp_path)
    return tmp_path
