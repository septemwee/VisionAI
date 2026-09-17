"""Single-window content capture used by the inspection pipeline.

``grab_window_content`` renders one window's own content into a bitmap with
``PrintWindow(PW_RENDERFULLCONTENT)``. Overlapping windows — including this
application's own overlay and status panel — are NOT part of that bitmap,
which keeps the captured frame free of the app's drawn output. A plain screen
region grab would instead composite everything drawn on top of the source
window back into the next inspection frame.
"""

import ctypes

import cv2
import numpy as np

PW_RENDERFULLCONTENT = 0x00000002
DIB_RGB_COLORS = 0
BI_RGB = 0


class _RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", ctypes.c_uint32),
        ("biWidth", ctypes.c_long),
        ("biHeight", ctypes.c_long),
        ("biPlanes", ctypes.c_uint16),
        ("biBitCount", ctypes.c_uint16),
        ("biCompression", ctypes.c_uint32),
        ("biSizeImage", ctypes.c_uint32),
        ("biXPelsPerMeter", ctypes.c_long),
        ("biYPelsPerMeter", ctypes.c_long),
        ("biClrUsed", ctypes.c_uint32),
        ("biClrImportant", ctypes.c_uint32),
    ]


class _BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", _BITMAPINFOHEADER),
        ("bmiColors", ctypes.c_uint32 * 3),
    ]


def grab_window_content(hwnd):
    """Return ``(frame_bgr, (left, top, width, height))`` for a window HWND.

    The frame contains only the target window's rendered content; anything
    overlapping it (the overlay, the status panel, popups) is excluded.
    Returns ``(None, None)`` when the capture is not possible so callers can
    fall back to a screen-region grab.
    """
    if not hwnd or not hasattr(ctypes, "windll"):
        return None, None

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32

    rect = _RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None, None

    width = int(rect.right) - int(rect.left)
    height = int(rect.bottom) - int(rect.top)
    if width <= 0 or height <= 0:
        return None, None

    window_dc = user32.GetWindowDC(hwnd)
    if not window_dc:
        return None, None

    memory_dc = None
    bitmap = None
    old_bitmap = None
    try:
        memory_dc = gdi32.CreateCompatibleDC(window_dc)
        bitmap = gdi32.CreateCompatibleBitmap(window_dc, width, height)
        if not memory_dc or not bitmap:
            return None, None

        old_bitmap = gdi32.SelectObject(memory_dc, bitmap)

        if not user32.PrintWindow(hwnd, memory_dc, PW_RENDERFULLCONTENT):
            return None, None

        # GetDIBits requires the bitmap to be selected out of the DC first.
        gdi32.SelectObject(memory_dc, old_bitmap)
        old_bitmap = None

        header = _BITMAPINFOHEADER()
        header.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
        header.biWidth = width
        header.biHeight = -height  # negative: top-down row order
        header.biPlanes = 1
        header.biBitCount = 32
        header.biCompression = BI_RGB

        info = _BITMAPINFO()
        info.bmiHeader = header

        buffer = ctypes.create_string_buffer(width * height * 4)
        copied = gdi32.GetDIBits(
            memory_dc, bitmap, 0, height, buffer, ctypes.byref(info), DIB_RGB_COLORS
        )
        if copied != height:
            return None, None

        frame = (
            np.frombuffer(buffer.raw, dtype=np.uint8)
            .reshape(height, width, 4)
            .copy()
        )
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame, (int(rect.left), int(rect.top), width, height)
    finally:
        if old_bitmap and memory_dc:
            gdi32.SelectObject(memory_dc, old_bitmap)
        if bitmap:
            gdi32.DeleteObject(bitmap)
        if memory_dc:
            gdi32.DeleteDC(memory_dc)
        user32.ReleaseDC(hwnd, window_dc)
