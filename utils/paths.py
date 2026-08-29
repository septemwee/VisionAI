"""Centralised filesystem paths for the application.

All paths are resolved relative to the project root so the applications
behave identically regardless of the current working directory.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RECIPES_DIR = PROJECT_ROOT / "recipes"
MODELS_DIR = PROJECT_ROOT / "models"

DETECTION_MODEL_PATH = MODELS_DIR / "detection" / "best.pt"
BACKBONE_WEIGHTS_PATH = MODELS_DIR / "backbone" / "wide_resnet50_2" / "model.safetensors"
