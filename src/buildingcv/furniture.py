from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def detect_furniture_from_png(
    png_bytes: bytes,
    wall_mask: np.ndarray | None = None,
    min_area_px: int = 200,
    iou_thresh: float = 0.2,
) -> List[list[list[float]]]:
    """Heuristic furniture detector on a rendered plan PNG.

    Returns a list of polygons (outer ring) detected as furniture. This is a
    conservative fallback using bright-fill detection and connected components
    so it works without SAM. When SAM is available you can replace this with
    SAM proposals.
    """
    img = Image.open(BytesIO(png_bytes)).convert("RGB")
    arr = np.array(img)
    H, W = arr.shape[:2]

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    # Threshold for bright-ish fills: furniture in many SVGs is light.
    _, th = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

    # Morphological cleanup to join nearby bits
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    # find contours
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: list[list[list[float]]] = []
    image_area = float(H * W)
    max_area = min(max_area_px, int(image_area * 0.15))
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_px or area > max_area:
            continue
        x, y, w_box, h_box = cv2.boundingRect(c)
        if w_box > W * 0.9 or h_box > H * 0.9:
            continue
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 1, thickness=-1)

        mean_val = float(cv2.mean(gray, mask=mask)[0])
        if mean_val > 248 and area > min_area_px * 4:
            continue

        if wall_mask is not None:
            # resize wall_mask to image size if needed
            if wall_mask.shape != mask.shape:
                wall_resized = cv2.resize((wall_mask > 0).astype('uint8'), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                wall_resized = (wall_mask > 0).astype('uint8')
            # compute IoU with wall mask and skip if heavily overlapping
            iou = _mask_iou(mask > 0, wall_resized > 0)
            if iou > iou_thresh:
                continue

        # simplify polygon
        eps = 3.0  # pixels
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(float).tolist()
        if len(approx) >= 3:
            polys.append(approx)
    return polys
