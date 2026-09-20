"""Locate an explicitly identified image-view child, never the whole window."""

import ctypes
from ctypes import wintypes


def select_viewport(candidates, bounds):
    """Choose the innermost named image view; reject ambiguous layouts."""
    left, top, width, height = bounds
    matches = []
    for name, rect in candidates:
        name = name.lower()
        if not any(token in name for token in ("imageview", "image view", "imagewindow", "image window", "viewport", "live view", "liveview")):
            continue
        x, y, w, h = rect
        if w < 100 or h < 100 or x < left or y < top:
            continue
        if x + w > left + width or y + h > top + height:
            continue
        matches.append(rect)
    matches = list(set(matches))
    inner = [r for r in matches if not any(
        s != r and s[0] >= r[0] and s[1] >= r[1]
        and s[0] + s[2] <= r[0] + r[2] and s[1] + s[3] <= r[1] + r[3]
        for s in matches)]
    return inner[0] if len(inner) == 1 else None


def find_image_viewport(hwnd, bounds):
    if not hwnd or not hasattr(ctypes, "windll"):
        return None
    api = ctypes.windll.user32
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    api.EnumChildWindows.argtypes = [wintypes.HWND, callback_type, wintypes.LPARAM]
    api.IsWindowVisible.argtypes = [wintypes.HWND]
    api.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    api.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    api.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    candidates = []

    @callback_type
    def visit(child, _):
        if api.IsWindowVisible(child):
            name = ctypes.create_unicode_buffer(256)
            api.GetClassNameW(child, name, len(name))
            rect, origin = wintypes.RECT(), wintypes.POINT()
            if api.GetClientRect(child, ctypes.byref(rect)) and api.ClientToScreen(child, ctypes.byref(origin)):
                candidates.append((name.value, (origin.x, origin.y, rect.right, rect.bottom)))
        return True

    api.EnumChildWindows(hwnd, visit, 0)
    return select_viewport(candidates, bounds)
