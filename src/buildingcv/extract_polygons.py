"""Turn a 4-class segmentation mask into per-class polygons for 3D rendering.

The mask is dense per-pixel labels; the viewer wants vector geometry. We
extract contours per structural class (wall / door / window) and emit a
small JSON describing each polygon as an outer ring plus optional holes.
The viewer extrudes these into 3D walls and slabs.

Pipeline per class:
    1. Binary mask = (pred == class_id).
    2. Morphological closing (3x3) — bridges 1-px gaps so a wall the model
       drew with a hairline break still becomes one polygon, not two.
    3. cv2.findContours(RETR_CCOMP) — outer contours and one level of holes.
       That's all we need: a wall ring with a doorway cut out is exactly
       one outer + one hole.
    4. cv2.approxPolyDP(epsilon = APPROX_EPSILON_PX) — kills the pixel-
       staircase. Without this every horizontal wall edge is ~50 collinear
       points; with it, two.
    5. Drop polygons whose outer-ring area is below MIN_POLYGON_AREA_PX —
       speckle from model noise that would render as floating debris.
"""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from .checkpoint import load_inference_checkpoint
from .data import IMAGENET_MEAN, IMAGENET_STD, _read_svg_dims  # type: ignore[attr-defined]
from .labels import CLASS_COLORS, CLASS_NAMES, CLASS_TO_ID, FLOOR_ID
from .model import build_model
from .svg_render import render_input_png
from .svg_to_mask import svg_to_mask
from .semantic_schema import Object, Polygon, Room

# Classes we extract geometry for. `floor` is implicit — it's the negative
# space, the viewer renders it as a single ground plane and doesn't need
# polygon detail.
EXTRACT_CLASSES: tuple[str, ...] = ("wall", "door", "window")

# Closing kernel size in pixels. 3x3 is enough to seal hairline breaks
# without smoothing away real ~5px door openings.
CLOSING_KERNEL_PX: int = 3

# Douglas-Peucker tolerance in pixels. ~1.5 collapses pixel staircases on
# axis-aligned walls without distorting diagonal walls. Set lower if walls
# look polygonal in the viewer; higher if there are too many vertices.
APPROX_EPSILON_PX: float = 1.5

# Polygons smaller than this are model speckle, not real structure. At
# 512x512 a real door is ~80 px², a window ~150 px², so 30 is a safe floor.
MIN_POLYGON_AREA_PX: float = 30.0


class ExtractionResult(TypedDict):
    canvas_size: list[int]            # [W, H]
    content_rect: list[int]           # [left, top, inner_w, inner_h] for letterboxed renders
    polygons: dict[str, list[Polygon]]
    # Phase 1 (schema only): semantic room/object models aren't wired in yet,
    # so these are optional empty arrays.
    rooms: list[Room]
    objects: list[Object]


def _approx(contour: np.ndarray) -> list[list[float]]:
    """Simplify a cv2 contour and return [[x, y], ...] floats."""
    simplified = cv2.approxPolyDP(contour, APPROX_EPSILON_PX, closed=True)
    return simplified.reshape(-1, 2).astype(float).tolist()


def _polygons_for_class(mask: np.ndarray, class_id: int) -> list[Polygon]:
    """Extract simplified polygons (outer + holes) for one class from a label mask."""
    binary = (mask == class_id).astype(np.uint8)
    if binary.sum() == 0:
        return []

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (CLOSING_KERNEL_PX, CLOSING_KERNEL_PX))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # RETR_CCOMP returns a two-level hierarchy: top-level outer contours
    # and their immediate holes. hierarchy[i] = [next, prev, child, parent].
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]  # shape (N, 4)

    # Group child holes under their parent outer contour.
    holes_by_parent: dict[int, list[int]] = {}
    for i, (_, _, _, parent) in enumerate(hierarchy):
        if parent != -1:
            holes_by_parent.setdefault(parent, []).append(i)

    polygons: list[Polygon] = []
    for i, (_, _, _, parent) in enumerate(hierarchy):
        if parent != -1:
            continue  # this is a hole, processed via its parent
        if cv2.contourArea(contours[i]) < MIN_POLYGON_AREA_PX:
            continue
        outer = _approx(contours[i])
        if len(outer) < 3:
            continue
        holes: list[list[list[float]]] = []
        for j in holes_by_parent.get(i, []):
            if cv2.contourArea(contours[j]) < MIN_POLYGON_AREA_PX:
                continue
            ring = _approx(contours[j])
            if len(ring) >= 3:
                holes.append(ring)
        polygons.append({"outer": outer, "holes": holes})
    return polygons


def mask_to_polygons(mask: np.ndarray) -> dict[str, list[Polygon]]:
    """Run the per-class extraction for every structural class. Pure function on the mask."""
    return {cls: _polygons_for_class(mask, CLASS_TO_ID[cls]) for cls in EXTRACT_CLASSES}


def _render_input_tensor(
    svg_path: Path, image_size: tuple[int, int], letterbox: bool, normalize: bool
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """Render an SVG to the model's input tensor. Returns (tensor, (left, top, inner_w, inner_h)).

    Mirrors CubiCasaDataset._load so model inference here matches training-time
    preprocessing exactly. Returned content rect tells the viewer which pixels
    are real plan vs. letterbox padding.
    """
    H, W = image_size
    if letterbox:
        svg_w, svg_h = _read_svg_dims(svg_path)
        scale = min(W / svg_w, H / svg_h)
        inner_w = max(1, int(round(svg_w * scale)))
        inner_h = max(1, int(round(svg_h * scale)))
    else:
        inner_w, inner_h = W, H

    png = render_input_png(svg_path, out_size=(inner_w, inner_h))
    img = np.array(Image.open(BytesIO(png)).convert("RGB"))
    image_t = torch.from_numpy(img).permute(2, 0, 1).contiguous().float().div_(255.0)
    if normalize:
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        image_t = (image_t - mean) / std

    if letterbox and (inner_h, inner_w) != (H, W):
        top = (H - inner_h) // 2
        left = (W - inner_w) // 2
        if normalize:
            canvas = torch.zeros(3, H, W)
        else:
            canvas = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).expand(3, H, W).clone()
        canvas[:, top:top + inner_h, left:left + inner_w] = image_t
        return canvas, (left, top, inner_w, inner_h)

    return image_t, (0, 0, W, H)


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _draw_contour_overlay(base_rgb: Image.Image, mask: np.ndarray) -> Image.Image:
    """Draw per-class contours on top of an RGB image for debug dumps."""
    out = base_rgb.convert("RGB").copy()
    arr = np.array(out)
    for cls in EXTRACT_CLASSES:
        binary = (mask == CLASS_TO_ID[cls]).astype(np.uint8)
        if binary.sum() == 0:
            continue
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (CLOSING_KERNEL_PX, CLOSING_KERNEL_PX))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        color = CLASS_COLORS[cls]
        cv2.drawContours(arr, contours, -1, color, 1)
    return Image.fromarray(arr)


def dump_extraction_stages(
    out_dir: Path,
    rendered_rgb: Image.Image,
    predicted_mask: np.ndarray,
    cleaned_mask: np.ndarray,
    final_mask: np.ndarray,
    polygons: dict[str, list[Polygon]],
) -> None:
    """Write intermediate debug images for every extraction stage."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered_rgb.save(out_dir / "01_rendered_input.png")
    _colorize_mask(predicted_mask).save(out_dir / "02_predicted_mask.png")
    _colorize_mask(cleaned_mask).save(out_dir / "03_cleaned_mask.png")
    _colorize_mask(final_mask).save(out_dir / "04_final_mask.png")
    _draw_contour_overlay(rendered_rgb, final_mask).save(out_dir / "05_extracted_contours.png")
    overlay = _draw_polygons(rendered_rgb, {k: polygons.get(k, []) for k in EXTRACT_CLASSES})
    overlay.save(out_dir / "06_polygons_overlay.png")
    (out_dir / "stage_counts.json").write_text(
        json.dumps(
            {
                "predicted": {n: int((predicted_mask == i).sum()) for i, n in enumerate(CLASS_NAMES)},
                "cleaned": {n: int((cleaned_mask == i).sum()) for i, n in enumerate(CLASS_NAMES)},
                "final": {n: int((final_mask == i).sum()) for i, n in enumerate(CLASS_NAMES)},
                "polygons": {k: len(v) for k, v in polygons.items()},
            },
            indent=2,
        )
    )


def _should_fallback_to_svg(mask: np.ndarray, svg_mask: np.ndarray) -> bool:
    """Prefer SVG mask when the model clearly lost structural openings."""
    wall_id = CLASS_TO_ID["wall"]
    door_id = CLASS_TO_ID["door"]
    window_id = CLASS_TO_ID["window"]

    model_walls = int(np.count_nonzero(mask == wall_id))
    if mask.sum() == 0 or model_walls < 10:
        return np.count_nonzero(svg_mask == wall_id) >= 10

    # Domain-shift case: walls exist but doors/windows vanished while the SVG
    # still encodes openings. Without this gate, synthetic/non-CubiCasa plans
    # keep a wall-only model mask and never recover door/window polygons.
    model_openings = int(np.count_nonzero(mask == door_id) + np.count_nonzero(mask == window_id))
    svg_openings = int(
        np.count_nonzero(svg_mask == door_id) + np.count_nonzero(svg_mask == window_id)
    )
    if model_openings == 0 and svg_openings >= 10:
        return True
    return False


class PolygonExtractor:
    """Stateful wrapper: load the model once, then extract for many SVGs.

    The server uses this to keep the model warm across requests; the CLI
    uses it for one shot.
    """

    def __init__(self, run_dir: Path, ckpt: str = "best.safetensors", device: str = "auto") -> None:
        cfg_path = run_dir / "config.yaml"
        ckpt_path = run_dir / ckpt
        with cfg_path.open() as f:
            self.cfg = yaml.safe_load(f)

        self.device = _resolve_device(device)
        self.image_size: tuple[int, int] = tuple(self.cfg["data"]["image_size"])  # (H, W)
        self.letterbox: bool = self.cfg["data"].get("letterbox", False)
        self.normalize: bool = self.cfg["data"]["normalize"]

        self.model = build_model(
            encoder_name=self.cfg["model"]["encoder_name"],
            encoder_weights=None,
        ).to(self.device)
        state, self.epoch = load_inference_checkpoint(ckpt_path, self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def predict_mask(self, svg_path: Path) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Run the model on one SVG. Returns (mask H×W uint8, content rect)."""
        image_t, rect = _render_input_tensor(svg_path, self.image_size, self.letterbox, self.normalize)
        logits = self.model(image_t.unsqueeze(0).to(self.device))
        mask = logits.argmax(dim=1).squeeze(0).to("cpu", torch.uint8).numpy()
        return mask, rect

    def extract(self, svg_path: Path, debug_dir: Path | None = None) -> ExtractionResult:
        image_t, rect = _render_input_tensor(
            svg_path, self.image_size, self.letterbox, self.normalize
        )
        with torch.no_grad():
            logits = self.model(image_t.unsqueeze(0).to(self.device))
        predicted_mask = logits.argmax(dim=1).squeeze(0).to("cpu", torch.uint8).numpy()
        left, top, inner_w, inner_h = rect

        # Zero out letterbox padding so we don't extract polygons from the
        # gray border. The model is well-behaved here in practice (predicts
        # floor) but the closing op can otherwise pull stray pixels outward.
        cleaned_mask = predicted_mask
        if (inner_h, inner_w) != predicted_mask.shape:
            cleaned_mask = np.full_like(predicted_mask, FLOOR_ID)
            cleaned_mask[top:top + inner_h, left:left + inner_w] = predicted_mask[
                top:top + inner_h, left:left + inner_w
            ]

        final_mask = cleaned_mask
        # Fall back to SVG-derived labels when the model loses structure or
        # openings (common on non-CubiCasa / synthetic SVGs).
        try:
            svg_mask = svg_to_mask(svg_path, size=(predicted_mask.shape[1], predicted_mask.shape[0]))
            if _should_fallback_to_svg(cleaned_mask, svg_mask):
                final_mask = svg_mask
        except Exception:
            pass

        H, W = final_mask.shape
        polygons = mask_to_polygons(final_mask)
        polygons.setdefault("furniture", [])

        if debug_dir is not None:
            dump_extraction_stages(
                Path(debug_dir),
                _denormalize_to_pil(image_t, self.normalize),
                predicted_mask,
                cleaned_mask,
                final_mask,
                polygons,
            )

        return {
            "canvas_size": [W, H],
            "content_rect": [left, top, inner_w, inner_h],
            "polygons": polygons,
            "rooms": [],
            "objects": [],
        }


def _denormalize_to_pil(image_t: torch.Tensor, normalize: bool) -> Image.Image:
    """Undo ImageNet normalization (if applied) and convert (3,H,W) → PIL RGB."""
    if normalize:
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        img = (image_t * std + mean).clamp(0, 1)
    else:
        img = image_t.clamp(0, 1)
    arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _colorize_mask(mask: np.ndarray) -> Image.Image:
    """Map class-index mask to RGB using the per-class color table."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for idx, name in enumerate(CLASS_NAMES):
        rgb[mask == idx] = CLASS_COLORS[name]
    return Image.fromarray(rgb)


def _draw_polygons(base: Image.Image, polygons: dict[str, list[Polygon]]) -> Image.Image:
    """Render polygons over `base` with semi-transparent fills, bright outlines,
    and vertex dots so simplification artifacts are visible at a glance.
    """
    canvas = base.convert("RGBA")
    fill_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    fdraw = ImageDraw.Draw(fill_layer)

    for cls in EXTRACT_CLASSES:
        r, g, b = CLASS_COLORS[cls]
        fill_rgba = (r, g, b, 90)  # ~35% alpha — visible but doesn't hide the input
        for poly in polygons[cls]:
            outer = [(x, y) for x, y in poly["outer"]]
            if len(outer) >= 3:
                fdraw.polygon(outer, fill=fill_rgba)
            for hole in poly["holes"]:
                # Holes punch back to fully transparent so the original input
                # image shows through where the model said "not this class".
                fdraw.polygon([(x, y) for x, y in hole], fill=(0, 0, 0, 0))

    canvas = Image.alpha_composite(canvas, fill_layer)
    odraw = ImageDraw.Draw(canvas)

    for cls in EXTRACT_CLASSES:
        r, g, b = CLASS_COLORS[cls]
        outline = (r, g, b, 255)
        for poly in polygons[cls]:
            rings = [poly["outer"]] + poly["holes"]
            for ring in rings:
                pts = [(x, y) for x, y in ring]
                if len(pts) >= 2:
                    odraw.line(pts + [pts[0]], fill=outline, width=1)
                # Vertex dots — 2px radius. Small enough not to swamp the image,
                # large enough to see whether approxPolyDP was over- or under-eager.
                for x, y in pts:
                    odraw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=outline)

    return canvas.convert("RGB")


def _hconcat_with_titles(images: list[Image.Image], titles: list[str]) -> Image.Image:
    """Side-by-side panel with a small label strip on top — keeps dimensions
    aligned even if a panel comes from a different source size."""
    label_h = 24
    h = max(im.height for im in images)
    w = sum(im.width for im in images)
    out = Image.new("RGB", (w, h + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    x = 0
    for im, title in zip(images, titles):
        out.paste(im, (x, label_h))
        draw.text((x + 6, 4), title, fill=(20, 20, 20))
        x += im.width
    return out


def render_debug_panel(
    image_t: torch.Tensor,
    mask: np.ndarray,
    result: ExtractionResult,
    normalize: bool,
) -> Image.Image:
    """Three panels: input | predicted mask | polygons overlaid on input."""
    input_img = _denormalize_to_pil(image_t, normalize)
    mask_img = _colorize_mask(mask)
    overlay_img = _draw_polygons(input_img, result["polygons"])
    return _hconcat_with_titles(
        [input_img, mask_img, overlay_img],
        ["input", "predicted mask", "polygons"],
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--svg", required=True, type=Path, help="path to a CubiCasa model.svg")
    p.add_argument("--run-dir", required=True, type=Path, help="trained run dir (with best.pt + config.yaml)")
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--out", required=True, type=Path, help="output JSON path")
    p.add_argument(
        "--debug-png",
        type=Path,
        default=None,
        help="if set, also write a 3-panel debug PNG (input | mask | polygons)",
    )
    p.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="if set, write per-stage debug images (rendered/mask/contours)",
    )
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    extractor = PolygonExtractor(args.run_dir, ckpt=args.ckpt, device=args.device)
    print(f"device: {extractor.device}   epoch: {extractor.epoch}")

    result = extractor.extract(args.svg, debug_dir=args.debug_dir)
    if args.debug_png is not None:
        image_t, _ = _render_input_tensor(
            args.svg, extractor.image_size, extractor.letterbox, extractor.normalize
        )
        # Rebuild a mask for the panel from the final polygons' source isn't
        # stored; re-run predict + same cleaning for the classic triptych.
        mask, rect = extractor.predict_mask(args.svg)
        left, top, inner_w, inner_h = rect
        if (inner_h, inner_w) != mask.shape:
            cleaned = np.full_like(mask, FLOOR_ID)
            cleaned[top:top + inner_h, left:left + inner_w] = mask[
                top:top + inner_h, left:left + inner_w
            ]
            mask = cleaned
        panel = render_debug_panel(image_t, mask, result, extractor.normalize)
        panel.save(args.debug_png)
        print(f"wrote {args.debug_png}")

    args.out.write_text(json.dumps(result))
    n_per_class = {k: len(v) for k, v in result["polygons"].items()}
    print(f"wrote {args.out}   polygons: {n_per_class}")


if __name__ == "__main__":
    main()
