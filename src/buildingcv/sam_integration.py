from __future__ import annotations

from io import BytesIO
from typing import List

import cv2
import numpy as np
from PIL import Image

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
except Exception:  # pragma: no cover - optional dependency
    sam_model_registry = None
    SamAutomaticMaskGenerator = None


def _mask_to_polygons(mask: np.ndarray, eps: float = 3.0) -> List[list[list[float]]]:
    contours, _ = cv2.findContours((mask > 0).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < 1:
            continue
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(float).tolist()
        if len(approx) >= 3:
            polys.append(approx)
    return polys


def generate_sam_polygons(png_bytes: bytes, checkpoint_path: str | None = None, device: str = "cuda", min_area_px: int = 100, iou_thresh: float = 0.25, max_area_frac: float = 0.15):
    """Run SAM AutomaticMaskGenerator on `png_bytes` and return vectorized polygons.

    Returns list of polygons (outer rings). If SAM isn't available or checkpoint
    missing, raises RuntimeError.
    """
    if sam_model_registry is None or SamAutomaticMaskGenerator is None:
        raise RuntimeError("segment-anything not available")

    img = Image.open(BytesIO(png_bytes)).convert("RGB")
    arr = np.array(img)
    H, W = arr.shape[:2]
    image_area = float(H * W)

    # choose model class by checkpoint name if provided
    model_name = "vit_h"
    if checkpoint_path and "vit_b" in checkpoint_path:
        model_name = "vit_b"
    elif checkpoint_path and "vit_l" in checkpoint_path:
        model_name = "vit_l"

    sam = sam_model_registry[model_name](checkpoint=checkpoint_path)
    try:
        sam.to(device)
    except Exception:
        sam.to("cpu")

    mask_gen = SamAutomaticMaskGenerator(sam)
    masks = mask_gen.generate(arr)

    polys: list[list[list[float]]] = []
    for m in masks:
        seg = m.get("segmentation")
        if seg is None:
            continue
        area = seg.sum()
        if area < min_area_px or area > image_area * max_area_frac:
            continue
        # vectorize
        p = _mask_to_polygons(seg)
        for ring in p:
            polys.append(ring)
    return polys
