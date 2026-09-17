"""Shared image-processing helpers used by both applications."""

from PIL.ImageOps import crop
import cv2
import math
import numpy as np

# Default ROI crop settings, shared by the trainer pages and helpers.
DEFAULT_WIDTH_SCALE = 1.0
DEFAULT_HEIGHT_SCALE = 1.0
DEFAULT_PADDING = 10
DEFAULT_ANGLE_OFFSET = 0.0


def letterbox_placement(source_width, source_height, target_width, target_height):
    """Return the letterbox placement for one source into one target size.

    Returns a dict with ``scale``, the scaled ``width``/``height`` and the
    centring ``x_offset``/``y_offset`` — the single source of the placement
    math shared by ``letterbox`` and the heatmap display mapping.
    """
    scale = min(target_width / source_width, target_height / source_height)
    new_width = int(source_width * scale)
    new_height = int(source_height * scale)
    return {
        "scale": scale,
        "width": new_width,
        "height": new_height,
        "x_offset": (target_width - new_width) // 2,
        "y_offset": (target_height - new_height) // 2,
    }


def unwrap_anomaly_map(anomaly_map):
    """Reduce a model anomaly map to a 2-D (H, W) numpy array.

    Anomalib returns ``(1, 1, H, W)`` tensors; every consumer of the map
    (heatmap rendering, the pixel gate, the review-page mask) must unwrap
    the leading dimensions identically so they always evaluate the same
    pixels. Returns ``None`` for ``None`` input.
    """
    if anomaly_map is None:
        return None

    amap = np.asarray(anomaly_map)
    while amap.ndim > 2:
        amap = amap[0]
    return amap


# Fixed raw-distance ceiling for heatmap colouring; matches the PatchCore
# score convention (services.patchcore_service.SCORE_SCALE = 100). Kept
# local to avoid a circular import.
_HEATMAP_RAW_SCALE = 100.0


def heatmap_overlay(
    image: np.ndarray, anomaly_map, opacity: float = 0.45, mask=None
) -> np.ndarray:
    """Blend a JET-coloured anomaly map over ``image`` (BGR in and out).

    Single implementation of the heatmap rendering shared by the live
    inspection status panel and the trainer review page, so both always
    display the same visualisation convention.

    The map is coloured with a fixed 0-100 raw-distance scale so the
    colouring is stable frame-to-frame. ``mask`` (optional) limits the
    blend to its truthy pixels; masked-out pixels keep the original image.
    """
    if anomaly_map is None or getattr(image, "size", 0) == 0:
        return None

    amap = unwrap_anomaly_map(anomaly_map)
    # Fixed scale (raw 0-100 -> 0-255) instead of per-frame min-max
    # normalisation: per-frame rescaling makes the colours wander even on a
    # static scene, and it makes frames incomparable to each other.
    amap = (np.clip(amap / _HEATMAP_RAW_SCALE, 0.0, 1.0) * 255.0).astype(np.uint8)
    heatmap = cv2.applyColorMap(amap, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    blended = cv2.addWeighted(image, 1.0 - opacity, heatmap, opacity, 0)
    if mask is None:
        return blended

    selected = np.asarray(mask).astype(bool)
    result = image.copy()
    result[selected] = blended[selected]
    return result


def letterbox(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Resize an image while keeping its aspect ratio, then centre it on a
    black canvas of the requested size."""
    height, width = image.shape[:2]

    if height <= 0 or width <= 0:
        return cv2.resize(image, (target_width, target_height))

    placement = letterbox_placement(width, height, target_width, target_height)
    resized = cv2.resize(image, (placement["width"], placement["height"]))
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    y_offset = placement["y_offset"]
    x_offset = placement["x_offset"]
    canvas[y_offset:y_offset + placement["height"], x_offset:x_offset + placement["width"]] = resized
    return canvas


def rotated_crop_bounds(image_shape, roi, config):
    """Return the ``(x1, y1, x2, y2)`` window ``crop_rotated_roi`` slices.

    Single source of the crop-window math so other consumers (the heatmap
    display mapping) place content in exactly the same window.
    """
    width_scale = float(config.get("width_scale", DEFAULT_WIDTH_SCALE))
    height_scale = float(config.get("height_scale", DEFAULT_HEIGHT_SCALE))
    padding = int(config.get("padding", DEFAULT_PADDING))

    cx = int(roi["cx"])
    cy = int(roi["cy"])
    half_width = int(roi["width"] * width_scale) // 2
    half_height = int(roi["height"] * height_scale) // 2

    x1 = max(0, cx - half_width - padding)
    y1 = max(0, cy - half_height - padding)
    x2 = min(image_shape[1], cx + half_width + padding)
    y2 = min(image_shape[0], cy + half_height + padding)
    return x1, y1, x2, y2


def crop_rotated_roi(image: np.ndarray, roi: dict, config: dict) -> np.ndarray:
    """Rotate the image so the package is upright and crop around it.

    ``roi`` is an OBB dict (``cx``, ``cy``, ``width``, ``height``, ``angle``
    in radians). ``config`` may provide ``width_scale``, ``height_scale``,
    ``padding`` and ``angle_offset``; missing values fall back to the shared
    defaults. This is the single implementation used by the ROI page, the
    augment worker and the recommendation matcher.
    """
    angle_offset = float(config.get("angle_offset", DEFAULT_ANGLE_OFFSET))

    angle_deg = math.degrees(roi["angle"]) + angle_offset

    matrix = cv2.getRotationMatrix2D((roi["cx"], roi["cy"]), angle_deg, 1.0)
    rotated = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))

    x1, y1, x2, y2 = rotated_crop_bounds(rotated.shape, roi, config)

    return rotated[y1:y2, x1:x2]


def box_aligned_heatmap(
    frame, package_roi, roi_config, crop, anomaly_map, points,
    target_width, target_height, opacity=0.45,
):
    """Build the display heatmap over the box's on-screen view.

    The model runs on the rotated-upright letterboxed crop; this maps its
    anomaly map BACK onto the original frame (inverse of the crop
    transform) and crops to the drawn box's axis-aligned bounds, so the
    panel shows exactly the region the overlay box covers, in the same
    orientation as the screen. Pure display logic — inference and verdict
    keep using the upright crop.
    """
    amap = unwrap_anomaly_map(anomaly_map)
    if amap is None or getattr(frame, "size", 0) == 0:
        return None

    crop_height, crop_width = crop.shape[:2]
    placement = letterbox_placement(crop_width, crop_height, target_width, target_height)
    if placement["width"] <= 0 or placement["height"] <= 0:
        return None

    # 256x256 map -> full letterbox canvas -> crop-sized map.
    upscaled = cv2.resize(amap, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    x_off = placement["x_offset"]
    y_off = placement["y_offset"]
    crop_map = cv2.resize(
        upscaled[y_off:y_off + placement["height"], x_off:x_off + placement["width"]],
        (crop_width, crop_height),
        interpolation=cv2.INTER_LINEAR,
    )

    # Paste into a frame-sized canvas (crop window in ROTATED space), then
    # warp back to screen orientation with the inverse rotation.
    x1, y1, x2, y2 = rotated_crop_bounds(frame.shape, package_roi, roi_config)
    canvas = np.zeros(frame.shape[:2], dtype=np.float32)
    canvas[y1:y2, x1:x2] = crop_map
    coverage = np.zeros(frame.shape[:2], dtype=np.float32)
    coverage[y1:y2, x1:x2] = 1.0

    angle_deg = math.degrees(package_roi["angle"]) + float(
        roi_config.get("angle_offset", DEFAULT_ANGLE_OFFSET)
    )
    inverse = cv2.getRotationMatrix2D(
        (package_roi["cx"], package_roi["cy"]), -angle_deg, 1.0
    )
    screen_size = (frame.shape[1], frame.shape[0])
    screen_map = cv2.warpAffine(canvas, inverse, screen_size)
    screen_coverage = cv2.warpAffine(coverage, inverse, screen_size)
    obb_mask = np.zeros(
        frame.shape[:2],
        dtype=np.uint8
    )

    cv2.fillPoly(
        obb_mask,
        [np.round(points).astype(np.int32)],
        1
)

    xs = points[:, 0]
    ys = points[:, 1]
    pad = 0
    # pad = max(8, int(0.1 * max(xs.max() - xs.min(), ys.max() - ys.min())))
    dx1 = max(0, int(xs.min()) - pad)
    dy1 = max(0, int(ys.min()) - pad)
    dx2 = min(frame.shape[1], int(xs.max()) + pad)
    dy2 = min(frame.shape[0], int(ys.max()) + pad)
    if dx2 - dx1 < 4 or dy2 - dy1 < 4:
        return None

    display_mask = (
        (screen_coverage > 0.5)
        & (obb_mask > 0)
    )

    return heatmap_overlay(
        frame[dy1:dy2, dx1:dx2],
        screen_map[dy1:dy2, dx1:dx2],
        opacity,
        mask=display_mask[dy1:dy2, dx1:dx2],
    )



def order_obb_points(points):
    """
    Convert YOLO OBB points to:

        TL ---- TR
        |       |
        BL ---- BR

    return shape (4,2)
    """

    pts = np.asarray(points, dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array(
        [tl, tr, br, bl],
        dtype=np.float32,
    )


def crop_yolo_obb(frame, points):
    """
    Crop directly from smoothed YOLO OBB.

    Parameters
    ----------
    frame : np.ndarray
        BGR image

    points : np.ndarray
        YOLO OBB points after smoothing
        shape = (4,2)

    Returns
    -------
    np.ndarray
        Straightened crop
    """

    pts = order_obb_points(points)

    width_top = np.linalg.norm(pts[1] - pts[0])
    width_bottom = np.linalg.norm(pts[2] - pts[3])
    width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(pts[3] - pts[0])
    height_right = np.linalg.norm(pts[2] - pts[1])
    height = int(max(height_left, height_right))

    if width < 2 or height < 2:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    dst = np.array(
        [
            [0, 0],                  # TL
            [width - 1, 0],          # TR
            [width - 1, height - 1], # BR
            [0, height - 1],         # BL
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(
        pts,
        dst
    )

    crop = cv2.warpPerspective(
        frame,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
    )

    return crop