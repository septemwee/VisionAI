"""Basic dataset validation used by the trainer."""

from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]


class DatasetService:
    """Collects simple statistics about a folder of images."""

    def analyze_dataset(self, dataset_path, progress_cb=None):
        """Collect statistics for ``dataset_path``.

        ``progress_cb(done, total)`` is optional and called after each
        image so callers running this off the UI thread can report
        progress. Scanning opens every image once, which is slow on
        large folders — callers should not run this on the GUI thread.
        """
        dataset_path = Path(dataset_path)

        image_files = []
        for extension in IMAGE_EXTENSIONS:
            image_files.extend(dataset_path.glob(extension))

        image_count = len(image_files)
        resolution = "-"
        corrupted = 0

        total = max(image_count, 1)
        for index, image_path in enumerate(image_files):
            try:
                with Image.open(image_path) as image:
                    if resolution == "-":
                        resolution = f"{image.width} x {image.height}"
            except Exception:
                corrupted += 1
            if progress_cb:
                progress_cb(index + 1, total)

        return {
            "image_count": image_count,
            "resolution": resolution,
            "corrupted": corrupted,
            "status": "VALID" if image_count > 0 else "INVALID",
        }
