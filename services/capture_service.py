"""Independent still-image capture helpers for the LCmicro image viewport."""

from datetime import datetime
from pathlib import Path

import cv2


def scaled_capture_rect(frame_size, image_size, zoom, offset=(0, 0)):
    """Pixel rectangle centered in the supplied capture surface."""
    fw, fh = frame_size
    iw, ih = image_size
    w, h = round(iw * zoom / 100), round(ih * zoom / 100)
    x = round((fw - w) / 2 + offset[0])
    y = round((fh - h) / 2 + offset[1])
    if min(w, h) < 4 or x < 0 or y < 0 or x + w > fw or y + h > fh:
        raise ValueError("Frame exceeds capture area — reduce zoom or adjust X/Y")
    return x, y, w, h


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
        x, y = max(0, round(nx * width)), max(0, round(ny * height))
        w, h = min(round(nw * width), width - x), min(round(nh * height), height - y)
        if w < 4 or h < 4:
            raise ValueError("Capture area is outside the current image")
        image = frame[y:y + h, x:x + w].copy()
        self.sequence += 1
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = self.folder / f"capture_{stamp}_{self.sequence:04d}.jpg"
        if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise IOError("Could not write image")
        return path
