"""Basic dataset validation used by the trainer."""

from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]


class DatasetService:
    """Collects simple statistics about a folder of images."""

    def analyze_dataset(self, dataset_path):
        dataset_path = Path(dataset_path)

        image_files = []
        for extension in IMAGE_EXTENSIONS:
            image_files.extend(dataset_path.glob(extension))

        image_count = len(image_files)
        resolution = "-"
        corrupted = 0

        for image_path in image_files:
            try:
                with Image.open(image_path) as image:
                    if resolution == "-":
                        resolution = f"{image.width} x {image.height}"
            except Exception:
                corrupted += 1

        return {
            "image_count": image_count,
            "resolution": resolution,
            "corrupted": corrupted,
            "status": "VALID" if image_count > 0 else "INVALID",
        }
