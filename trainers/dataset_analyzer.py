"""Basic dataset statistics used by training experiments."""

from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]


class DatasetAnalyzer:
    """Counts images and reads the resolution of the first one."""

    def analyze(self, dataset_path):
        dataset_path = Path(dataset_path)

        image_files = []
        for extension in IMAGE_EXTENSIONS:
            image_files.extend(dataset_path.glob(extension))

        result = {
            "image_count": len(image_files),
            "resolution": "Unknown",
        }

        if image_files:
            with Image.open(image_files[0]) as image:
                result["resolution"] = f"{image.width}x{image.height}"

        return result
