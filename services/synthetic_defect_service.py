"""Deterministic synthetic defects used only for model validation."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def generate_synthetic_validation(source_dir, output_dir, *, mask_dir=None):
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if mask_dir is not None:
        mask_dir = Path(mask_dir)
        mask_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    paths = sorted(p for p in source_dir.rglob("*") if p.suffix.lower() in
                   {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
    for source in paths:
        image = cv2.imread(str(source))
        if image is None or min(image.shape[:2]) < 16:
            continue
        seed = int(hashlib.sha256(str(source.relative_to(source_dir)).encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        height, width = image.shape[:2]
        for kind in ("scratch", "spot", "chip", "pin_discoloration"):
            defect = image.copy()
            mask = np.zeros((height, width), np.uint8)
            placement = "body"
            if kind == "scratch":
                start = (int(rng.integers(width // 5, width * 4 // 5)), int(rng.integers(height // 5, height * 4 // 5)))
                end = (max(0, min(width - 1, start[0] + int(rng.integers(-width // 4, width // 4 + 1)))),
                       max(0, min(height - 1, start[1] + int(rng.integers(-height // 4, height // 4 + 1)))))
                thickness = max(1, round(min(width, height) * 0.012))
                cv2.line(mask, start, end, 255, thickness)
            elif kind == "spot":
                center = (int(rng.integers(width // 5, width * 4 // 5)), int(rng.integers(height // 5, height * 4 // 5)))
                radius = max(2, round(min(width, height) * float(rng.uniform(0.025, 0.07))))
                cv2.circle(mask, center, radius, 255, -1)
            elif kind == "chip":
                size = max(3, round(min(width, height) * 0.06))
                x = int(rng.choice([rng.integers(0, max(1, width // 8)), rng.integers(max(1, width * 7 // 8), width)]))
                y = int(rng.integers(height // 5, height * 4 // 5))
                cv2.rectangle(mask, (max(0, x - size), max(0, y - size)),
                              (min(width - 1, x + size), min(height - 1, y + size)), 255, -1)
            else:
                side = int(rng.integers(0, 4))
                placement = ("top_edge", "bottom_edge", "left_edge", "right_edge")[side]
                band = max(2, round(min(width, height) * 0.08))
                position = int(rng.integers(0, (width if side < 2 else height) // 2))
                if side == 0:
                    cv2.rectangle(mask, (position, 0), (min(width - 1, position + band), band * 2), 255, -1)
                elif side == 1:
                    cv2.rectangle(mask, (position, height - band * 2), (min(width - 1, position + band), height - 1), 255, -1)
                elif side == 2:
                    cv2.rectangle(mask, (0, position), (band * 2, min(height - 1, position + band)), 255, -1)
                else:
                    cv2.rectangle(mask, (width - band * 2, position), (width - 1, min(height - 1, position + band)), 255, -1)
            if kind == "pin_discoloration":
                # Warm brown oxidation/tarnish, retained as BGR because cv2 images are BGR.
                color = np.empty_like(defect)
                color[:] = (25, 85, 180)
            else:
                color = np.full_like(defect, int(rng.choice([20, 235])))
            alpha = cv2.GaussianBlur(mask, (0, 0), max(0.8, min(width, height) * 0.006)).astype(np.float32) / 255
            defect = (defect * (1 - alpha[..., None]) + color * alpha[..., None]).astype(np.uint8)
            target = output_dir / f"{source.stem}_{seed:08x}__{kind}.png"
            if not cv2.imwrite(str(target), defect):
                raise OSError(f'Cannot write synthetic image: {target}')
            if mask_dir is not None:
                # Keep masks outside image folders so scoring never treats
                # a ground-truth mask as an inspection sample.
                if not cv2.imwrite(str(mask_dir / target.name), np.rint(alpha * 255).astype(np.uint8)):
                    raise OSError(f'Cannot write synthetic alpha mask: {target.name}')
            manifest.append({"source": str(source.relative_to(source_dir)), "file": target.name,
                             "type": kind, "placement": placement, "seed": seed})
    (output_dir / "synthetic_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
