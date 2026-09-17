"""Centralised filesystem paths for the application.

Read-only bundled assets resolve through ``resource_path`` so a frozen
(PyInstaller) build finds them inside the temporary bundle directory.
Writable state (recipes, config, logs) resolves against ``DATA_ROOT``:
the executable's directory when frozen, the project root in a normal
checkout — the bundle directory is wiped on every run, so recipes and
config must never live there.
"""

import sys
from pathlib import Path

from utils.resource_path import resource_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else PROJECT_ROOT
)

RECIPES_DIR = DATA_ROOT / "recipes"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = DATA_ROOT / "config"

# Bundled, read-only model resources resolve through resource_path so a
# frozen (PyInstaller) build finds them inside the temporary bundle
# directory; in a normal checkout this is identical to MODELS_DIR.
DETECTION_MODEL_PATH = resource_path("models/detection/best.pt")
BACKBONE_WEIGHTS_PATH = resource_path(
    "models/backbone/wide_resnet50_2/model.safetensors"
)
MODEL_REGISTRY_PATH = CONFIG_DIR / "model_registry.json"


def resolve_recipe_path(path):
    """Resolve a stored recipe path against the data root.

    Recipes store either absolute paths or paths relative to the data root
    (the project root in a normal checkout). Resolving here keeps behaviour
    independent of the current working directory. Empty values resolve to
    ``None``.
    """
    if not path:
        return None

    path = Path(path)
    if path.is_absolute():
        return path

    return DATA_ROOT / path


def relativize_recipe_path(path):
    """Return ``path`` relative to the data root when it lives inside it.

    Recipe files must stay machine-independent, so writers store data-root
    relative paths. Paths outside the data root (network shares, other
    drives) cannot be relativized and are returned unchanged. Empty values
    return an empty string.
    """
    if not path:
        return ""

    path = Path(path)
    try:
        return str(path.resolve().relative_to(DATA_ROOT.resolve()))
    except ValueError:
        return str(path)
