"""Inspection-only nested mark comparison and template persistence."""
import copy
import os
import uuid
from pathlib import Path

import cv2
import numpy as np

from services.recipe_service import RecipeService
from utils.paths import resolve_recipe_path, relativize_recipe_path


def save_regions(recipe_path, frame, regions, dirty):
    import json
    path = Path(recipe_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    tx, ty, tw, th = regions["top_mark"]
    for name, (x, y, w, h) in regions.items():
        if name not in ("top_mark", "laser_mark") or min(w, h) < 4:
            raise ValueError("Invalid ROI")
        if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
            raise ValueError("ROI is outside the image")
        if name == "laser_mark" and (x < tx or y < ty or x+w > tx+tw or y+h > ty+th):
            raise ValueError("Laser-mark must be inside Top-mark")
    remove_laser = "laser_mark" in dirty and "laser_mark" not in regions
    old_laser_path = resolve_recipe_path(data.get("laser_mark_template")) if remove_laser and data.get("laser_mark_template") else None
    updates = {}
    # If Top-mark changes, its child's coordinates and image change together.
    names = {name for name in dirty if name in regions}
    if "top_mark" in dirty and "laser_mark" in regions:
        names.add("laser_mark")
    for name in names:
        x, y, w, h = regions[name]
        image = frame[y:y+h, x:x+w]
        if name == "laser_mark" and "top_mark" not in dirty:
            # The editor may enlarge the saved Top-mark for easier drawing.
            # Crop its original pixels, not that enlarged display image.
            top_path = resolve_recipe_path(data["top_mark_template"])
            original_top = cv2.imdecode(np.frombuffer(top_path.read_bytes(), np.uint8),
                                       cv2.IMREAD_COLOR)
            if original_top is None:
                raise ValueError("Cannot read original Top-mark")
            oh, ow = original_top.shape[:2]
            left, upper = round((x-tx)/tw*ow), round((y-ty)/th*oh)
            right, lower = round((x+w-tx)/tw*ow), round((y+h-ty)/th*oh)
            if right-left < 4 or lower-upper < 4:
                raise ValueError("Laser-mark must cover at least 4 original pixels")
            image = original_top[upper:lower, left:right]
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        if not success:
            raise ValueError("Could not encode template")
        target = path.parent / (name + "_template.jpg")
        updates[target] = encoded.tobytes()
        data[name + "_template"] = relativize_recipe_path(target)
        data[name + "_roi"] = (
            {"x": (x-tx)/tw, "y": (y-ty)/th, "w": w/tw, "h": h/th}
            if name == "laser_mark" else {"x": x, "y": y, "w": w, "h": h}
        )
    if "laser_mark" in names:
        data["laser_mark_coordinate_space"] = "top_mark_normalized"
        data.setdefault("laser_mark_threshold", 0.8)
        data.setdefault("laser_mark_position_tolerance", 0.03)
    if remove_laser:
        for key in ("laser_mark_template", "laser_mark_roi", "laser_mark_coordinate_space",
                    "laser_mark_threshold", "laser_mark_position_tolerance"):
            data.pop(key, None)
    originals = {target: target.read_bytes() if target.exists() else None for target in updates}
    staged = []
    try:
        for target, content in updates.items():
            temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
            staged.append(temporary)
            temporary.write_bytes(content)
            os.replace(temporary, target)
        RecipeService.write_json_file(path, data)
        if remove_laser and old_laser_path and old_laser_path.exists():
            old_laser_path.unlink()
    except Exception:
        for target, content in originals.items():
            if content is None:
                target.unlink(missing_ok=True)
            else:
                target.write_bytes(content)
        raise
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)
    return copy.deepcopy(data)


class MarkingService:
    def __init__(self):
        self.cache = {}

    def _read(self, value):
        path = resolve_recipe_path(value)
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        if key not in self.cache:
            image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError("Template cannot be read")
            if len(self.cache) > 8:
                self.cache.clear()
            self.cache[key] = image
        return self.cache[key]

    @staticmethod
    def _anchor(gray, top):
        """Search scale coarsely, then refine around the best geometry.

        Limit correlation work to a 480-pixel view. Include the exact fit
        scale to keep self-matches accurate at arbitrary image sizes.
        """
        factor = min(1.0, 480.0 / max(gray.shape))
        view = cv2.resize(gray, None, fx=factor, fy=factor)
        view = cv2.GaussianBlur(view, (5, 5), 1.0)
        fit = min(gray.shape[1]/top.shape[1], gray.shape[0]/top.shape[0])
        high = min(3.0, fit)
        low = max(0.15, high / 4.0)
        scales = [1.0, fit] + list(np.linspace(low, high, 28))
        best = None
        visited = set()

        def evaluate(scale):
            nonlocal best
            sw, sh = round(top.shape[1]*scale*factor), round(top.shape[0]*scale*factor)
            if min(sw, sh) < 8 or sw > view.shape[1] or sh > view.shape[0]:
                return
            if (sw, sh) in visited:
                return
            visited.add((sw, sh))
            candidate = cv2.resize(top, (sw, sh), interpolation=cv2.INTER_AREA)
            candidate = cv2.GaussianBlur(candidate, (5, 5), 1.0)
            response = cv2.matchTemplate(view, candidate, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(response)
            if np.isfinite(score) and (best is None or score > best[0]):
                best = score, location, sw, sh, scale
        for scale in scales:
            if 0 < scale <= fit + 1e-6:
                evaluate(scale)
                if best is not None and best[0] >= 0.985:
                    break
        if best is None:
            raise ValueError("Top-mark anchor unavailable")
        step = (high-low)/27
        if best[0] < 0.985:
            for scale in np.linspace(max(low,best[4]-step), min(fit,best[4]+step), 13):
                evaluate(scale)
        score, (ax, ay), sw, sh, _ = best
        x, y = round(ax/factor), round(ay/factor)
        width, height = round(sw/factor), round(sh/factor)
        crop = gray[y:min(gray.shape[0],y+height), x:min(gray.shape[1],x+width)]
        return score, cv2.resize(crop, (top.shape[1],top.shape[0]))

    @staticmethod
    def _refine_alignment(aligned, top, region):
        """Refine small crop errors using the surrounding Top-mark.

        Exclude Laser-mark from alignment so a displaced laser cannot pull
        the anchor toward itself and silently pass the position check.
        """
        factor = min(1.0, 400.0/max(top.shape))
        size = (max(8,round(top.shape[1]*factor)), max(8,round(top.shape[0]*factor)))
        reference = cv2.resize(top, size).astype(np.float32)/255
        moving = cv2.resize(aligned, size).astype(np.float32)/255
        mask = np.ones(reference.shape,np.uint8)*255
        x,y,w,h = region
        margin = 0.05
        mask[max(0,int((y-margin)*size[1])):min(size[1],int((y+h+margin)*size[1])+1),
             max(0,int((x-margin)*size[0])):min(size[0],int((x+w+margin)*size[0])+1)] = 0
        if np.count_nonzero(mask) < mask.size*0.25:
            return aligned
        try:
            warp = np.eye(2,3,dtype=np.float32)
            _, warp = cv2.findTransformECC(
                reference, moving, warp, cv2.MOTION_EUCLIDEAN,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4),
                inputMask=mask, gaussFiltSize=5)
            if abs(float(np.arctan2(warp[1,0],warp[0,0]))) > np.deg2rad(3):
                return aligned
            if abs(warp[0,2]) > size[0]*0.04 or abs(warp[1,2]) > size[1]*0.04:
                return aligned
            warp[:,2] /= factor
            return cv2.warpAffine(aligned, warp, (top.shape[1],top.shape[0]),
                                 flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_REPLICATE)
        except cv2.error:
            return aligned

    def check(self, roi, recipe):
        result = {"status": "DISABLED", "score": None, "reason": "Laser-mark not configured"}
        if not recipe.get("laser_mark_template") and not recipe.get("laser_mark_roi"):
            return result
        result.update(status="FAIL", reason="Marking unavailable")
        try:
            top = self._read(recipe.get("top_mark_template"))
            laser = self._read(recipe.get("laser_mark_template"))
            region = recipe["laser_mark_roi"]
            x, y, w, h = [float(region[k]) for k in ("x", "y", "w", "h")]
            if min(x, y) < 0 or min(w, h) <= 0 or x+w > 1.001 or y+h > 1.001:
                raise ValueError("Recapture Laser-mark ROI")
            threshold = float(recipe.get("laser_mark_threshold", 0.8))
            tolerance = float(recipe.get("laser_mark_position_tolerance", 0.03))
            if not 0 < threshold <= 1 or not 0 <= tolerance <= 0.2:
                raise ValueError("Invalid marking settings")
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            if min(float(top.std()), float(laser.std()), float(gray.std())) < 2:
                raise ValueError("Insufficient contrast for marking")
            anchor_score, aligned = self._anchor(gray, top)
            if anchor_score < 0.45:
                raise ValueError("Top-mark anchor unclear")
            aligned = self._refine_alignment(aligned, top, (x,y,w,h))
            result["anchor_score"] = float(anchor_score)
            tw, th = top.shape[1], top.shape[0]
            lx, ly = round(x*tw), round(y*th)
            lw, lh = max(4, round(w*tw)), max(4, round(h*th))
            reference = cv2.resize(laser, (lw, lh))
            # Suppress fine sensor/JPEG grain equally on both images.
            aligned = cv2.GaussianBlur(aligned, (5,5), 1.0)
            reference = cv2.GaussianBlur(reference, (5,5), 1.0)
            # Search the anchor to distinguish a matching mark in the wrong position.
            response = cv2.matchTemplate(aligned, reference, cv2.TM_CCOEFF_NORMED)
            px, py = max(1, round(tolerance*tw)), max(1, round(tolerance*th))
            window = response[max(0,ly-py):min(response.shape[0],ly+py+1),
                              max(0,lx-px):min(response.shape[1],lx+px+1)]
            if not window.size:
                raise ValueError("Expected Laser-mark position unavailable")
            score = float(window.max())
            if not np.isfinite(score):
                raise ValueError("Invalid marking score")
            status = "MATCH" if score >= threshold else (
                "POSITION ERROR" if float(response.max()) >= threshold else "MISMATCH")
            result.update(status=status, score=score,
                          reason=f"Laser-mark {status.lower()} (similarity {score:.3f})")
        except Exception as error:
            result.update(status="FAIL", reason=str(error))
        return result
