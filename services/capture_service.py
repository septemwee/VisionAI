"""Independent still-image capture helpers for the LCmicro image viewport."""

from datetime import datetime
from pathlib import Path

import cv2


class CaptureService:
    """Save cropped frames without involving the inspection pipeline."""

    def __init__(self):
        self.folder = None
        self.area = None  # normalized x, y, width, height
        self.sequence = 0

    def set_folder(self, folder):
        path = Path(folder).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        self.folder = path

    def set_area(self, rect, frame_size):
        x, y, w, h = rect
        fw, fh = frame_size
        if w < 4 or h < 4 or fw <= 0 or fh <= 0:
            raise ValueError("Capture area is too small")
        self.area = (x / fw, y / fh, w / fw, h / fh)

    def save(self, frame):
        if self.folder is None:
            raise ValueError("Select a capture folder first")
        if self.area is None:
            raise ValueError("Set the image area first")
        height, width = frame.shape[:2]
        nx, ny, nw, nh = self.area
        x, y = max(0, int(nx * width)), max(0, int(ny * height))
        w, h = min(int(nw * width), width - x), min(int(nh * height), height - y)
        if w < 4 or h < 4:
            raise ValueError("Capture area is outside the current image")
        image = frame[y:y + h, x:x + w].copy()
        self.sequence += 1
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = self.folder / f"capture_{stamp}_{self.sequence:04d}.jpg"
        if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise IOError("Could not write image")
        return path
