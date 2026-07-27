"""Evaluate the floorplan pipeline on a folder of test plans.

For each plan:
  structural extraction -> room detection -> room refinement (if available)

Writes per-plan visualizations + JSON to tests/results/, plus a summary report.

Example:
    python scripts/evaluate_pipeline.py \\
        --input-dir tests/floorplans \\
        --out-dir tests/results
"""

from __future__ import annotations

import argparse
import json
import math
import os
from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from buildingcv.extract_polygons import PolygonExtractor
from buildingcv.room_detector import detect_rooms_from_png
from buildingcv.svg_render import filtered_svg_bytes, svg2png

try:
    from buildingcv.room_refinement import refine_rooms
except Exception:  # pragma: no cover - optional
    refine_rooms = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_EXTS = {".svg", ".png", ".jpg", ".jpeg", ".webp"}

ROOM_COLORS = {
    "bedroom": (100, 150, 255),
    "kitchen": (255, 180, 90),
    "living room": (120, 200, 130),
    "bathroom": (120, 210, 220),
    "corridor": (200, 190, 160),
    "dining room": (230, 170, 140),
    "storage": (180, 160, 210),
    "office": (170, 150, 230),
    "default": (180, 180, 170),
}


def _discover_cases(input_dir: Path) -> list[tuple[str, Path]]:
    """Return [(case_name, input_path), ...] for SVGs/images under input_dir."""
    cases: list[tuple[str, Path]] = []
    if not input_dir.is_dir():
        return cases

    # Prefer nested folders with model.svg
    for child in sorted(input_dir.iterdir()):
        if child.is_dir():
            preferred = child / "model.svg"
            if preferred.exists():
                cases.append((child.name, preferred))
                continue
            for f in sorted(child.iterdir()):
                if f.suffix.lower() in INPUT_EXTS:
                    cases.append((child.name, f))
                    break
        elif child.is_file() and child.suffix.lower() in INPUT_EXTS:
            cases.append((child.stem, child))
    return cases


def _load_input_png(path: Path, out_size: tuple[int, int] | None = None) -> bytes:
    if path.suffix.lower() == ".svg":
        w, h = out_size or (512, 512)
        return svg2png(
            bytestring=filtered_svg_bytes(path),
            output_width=w,
            output_height=h,
        )
    img = Image.open(path).convert("RGBA")
    if out_size is not None:
        img = img.resize(out_size, Image.Resampling.BILINEAR)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _png_to_rgb(png_bytes: bytes, size: tuple[int, int] | None = None) -> Image.Image:
    img = Image.open(BytesIO(png_bytes)).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.Resampling.BILINEAR)
    return img


def _draw_polygons(
    base: Image.Image,
    polygons: list[dict],
    color: tuple[int, int, int],
    fill_alpha: int = 90,
    outline_width: int = 2,
) -> Image.Image:
    canvas = base.convert("RGBA")
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    r, g, b = color
    for poly in polygons:
        outer = [(float(x), float(y)) for x, y in poly.get("outer", [])]
        if len(outer) >= 3:
            draw.polygon(outer, fill=(r, g, b, fill_alpha), outline=(r, g, b, 255))
            if outline_width > 1:
                draw.line(outer + [outer[0]], fill=(r, g, b, 255), width=outline_width)
        for hole in poly.get("holes", []):
            pts = [(float(x), float(y)) for x, y in hole]
            if len(pts) >= 3:
                draw.polygon(pts, fill=(0, 0, 0, 0))
    return Image.alpha_composite(canvas, layer).convert("RGB")


def _draw_rooms(base: Image.Image, rooms: list[dict], label: bool = True) -> Image.Image:
    canvas = base.convert("RGBA")
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    text_draw = ImageDraw.Draw(canvas)
    for room in rooms:
        room_type = str(room.get("type", "default")).lower()
        color = ROOM_COLORS.get(room_type, ROOM_COLORS["default"])
        poly = room.get("polygon", {})
        outer = [(float(x), float(y)) for x, y in poly.get("outer", [])]
        if len(outer) >= 3:
            r, g, b = color
            draw.polygon(outer, fill=(r, g, b, 100), outline=(r, g, b, 255))
            draw.line(outer + [outer[0]], fill=(r, g, b, 255), width=2)
        if label:
            anchor = room.get("label_anchor")
            if not anchor and outer:
                anchor = [
                    sum(p[0] for p in outer) / len(outer),
                    sum(p[1] for p in outer) / len(outer),
                ]
            if anchor:
                text_draw.text(
                    (float(anchor[0]), float(anchor[1])),
                    room_type.upper(),
                    fill=(20, 20, 20),
                )
    out = Image.alpha_composite(canvas, layer).convert("RGB")
    # Re-draw labels on the composited RGB so they stay crisp.
    if label:
        text_draw = ImageDraw.Draw(out)
        for room in rooms:
            room_type = str(room.get("type", "default")).lower()
            poly = room.get("polygon", {})
            outer = [(float(x), float(y)) for x, y in poly.get("outer", [])]
            anchor = room.get("label_anchor")
            if not anchor and outer:
                anchor = [
                    sum(p[0] for p in outer) / len(outer),
                    sum(p[1] for p in outer) / len(outer),
                ]
            if anchor:
                text_draw.text(
                    (float(anchor[0]), float(anchor[1])),
                    room_type.upper(),
                    fill=(20, 20, 20),
                )
    return out


def _draw_comparison(
    base: Image.Image,
    rooms_before: list[dict],
    rooms_after: list[dict],
) -> Image.Image:
    """Left: before refinement (bbox). Right: after refinement."""
    left = _draw_rooms(base, rooms_before, label=True)
    right = _draw_rooms(base, rooms_after, label=True)
    ImageDraw.Draw(left).text((8, 8), "BEFORE refinement (bbox)", fill=(20, 20, 20))
    ImageDraw.Draw(right).text((8, 8), "AFTER refinement", fill=(20, 20, 20))
    h = max(left.height, right.height)
    out = Image.new("RGB", (left.width + right.width + 8, h), (255, 255, 255))
    out.paste(left, (0, 0))
    out.paste(right, (left.width + 8, 0))
    return out


def _polygon_area(outer: list[list[float]]) -> float:
    if len(outer) < 3:
        return 0.0
    pts = np.array(outer, dtype=np.float64)
    return float(abs(cv2.contourArea(pts.astype(np.float32))))


def _is_invalid_polygon(poly: dict) -> bool:
    outer = poly.get("outer") or []
    if len(outer) < 3:
        return True
    try:
        area = _polygon_area(outer)
    except Exception:
        return True
    if area <= 1.0:
        return True
    # Self-intersecting / NaN check
    for x, y in outer:
        if not (math.isfinite(float(x)) and math.isfinite(float(y))):
            return True
    return False


def compute_metrics(result: dict, rooms_before: list[dict] | None = None) -> dict[str, Any]:
    polys = result.get("polygons", {})
    rooms = result.get("rooms", []) or []

    wall_areas = [_polygon_area(p.get("outer", [])) for p in polys.get("wall", [])]
    room_areas = [_polygon_area(r.get("polygon", {}).get("outer", [])) for r in rooms]
    invalid_walls = sum(1 for p in polys.get("wall", []) if _is_invalid_polygon(p))
    invalid_rooms = sum(
        1 for r in rooms if _is_invalid_polygon(r.get("polygon", {}))
    )

    metrics: dict[str, Any] = {
        "structural": {
            "walls": len(polys.get("wall", [])),
            "doors": len(polys.get("door", [])),
            "windows": len(polys.get("window", [])),
            "furniture": len(polys.get("furniture", [])),
        },
        "rooms": {
            "count": len(rooms),
            "types": sorted({str(r.get("type", "unknown")) for r in rooms}),
            "confidences": [float(r.get("confidence", 0.0)) for r in rooms],
            "mean_confidence": (
                float(np.mean([float(r.get("confidence", 0.0)) for r in rooms]))
                if rooms
                else 0.0
            ),
        },
        "geometry": {
            "wall_areas_px": wall_areas,
            "room_areas_px": room_areas,
            "invalid_wall_polygons": invalid_walls,
            "invalid_room_polygons": invalid_rooms,
            "total_wall_area_px": float(sum(wall_areas)),
            "total_room_area_px": float(sum(room_areas)),
        },
    }

    if rooms_before is not None:
        before_areas = [
            _polygon_area(r.get("polygon", {}).get("outer", [])) for r in rooms_before
        ]
        after_areas = room_areas
        metrics["refinement"] = {
            "rooms_before": len(rooms_before),
            "rooms_after": len(rooms),
            "area_before_px": before_areas,
            "area_after_px": after_areas,
            "mean_area_delta_px": (
                float(np.mean([a - b for a, b in zip(after_areas, before_areas)]))
                if before_areas and after_areas and len(before_areas) == len(after_areas)
                else 0.0
            ),
        }
    return metrics


def evaluate_one(
    extractor: PolygonExtractor,
    case_name: str,
    input_path: Path,
    out_dir: Path,
    skip_rooms: bool = False,
) -> dict[str, Any]:
    case_dir = out_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    # Structural extraction
    if input_path.suffix.lower() == ".svg":
        debug_dir = case_dir / "debug_stages"
        result = extractor.extract(input_path, debug_dir=debug_dir)
    else:
        # Raster inputs: write a temporary SVG-less path via mask-free extract is
        # not supported by PolygonExtractor (expects SVG). Save metrics as failed.
        raise ValueError(f"evaluate_pipeline currently requires SVG inputs: {input_path}")

    left, top, inner_w, inner_h = result["content_rect"]
    canvas_w, canvas_h = result["canvas_size"]

    # Original rendered image (content size), then padded onto canvas for overlays.
    png = _load_input_png(input_path, out_size=(inner_w, inner_h))
    content_img = _png_to_rgb(png)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 240))
    canvas.paste(content_img, (left, top))
    canvas.save(case_dir / "01_original.png")

    # Wall / door / window overlay
    walls_vis = canvas.copy()
    walls_vis = _draw_polygons(walls_vis, result["polygons"].get("wall", []), (40, 40, 45), 120)
    walls_vis = _draw_polygons(walls_vis, result["polygons"].get("door", []), (230, 120, 50), 140)
    walls_vis = _draw_polygons(walls_vis, result["polygons"].get("window", []), (60, 150, 220), 140)
    walls_vis.save(case_dir / "02_walls.png")

    rooms_before: list[dict] = []
    rooms_after: list[dict] = []

    if not skip_rooms:
        try:
            rooms_before = detect_rooms_from_png(png, result["content_rect"])
        except Exception:
            rooms_before = []

        rooms_after = deepcopy(rooms_before)
        if refine_rooms is not None and rooms_before:
            try:
                rooms_after = refine_rooms(
                    rooms_before,
                    result["polygons"].get("wall", []),
                    result["canvas_size"],
                )
            except Exception:
                rooms_after = deepcopy(rooms_before)

    result["rooms"] = rooms_after
    result.setdefault("objects", [])

    rooms_vis = _draw_rooms(canvas, rooms_after, label=True)
    rooms_vis.save(case_dir / "03_rooms.png")

    comparison = _draw_comparison(canvas, rooms_before, rooms_after)
    comparison.save(case_dir / "04_refinement_comparison.png")

    # Final JSON (drop huge optional fields carefully — keep schema)
    out_json = {
        "canvas_size": result["canvas_size"],
        "content_rect": result["content_rect"],
        "polygons": result["polygons"],
        "rooms": rooms_after,
        "rooms_before_refinement": rooms_before,
        "objects": result.get("objects", []),
    }
    (case_dir / "05_result.json").write_text(json.dumps(out_json, indent=2))

    metrics = compute_metrics(result, rooms_before=rooms_before)
    metrics["case"] = case_name
    metrics["input"] = str(input_path.relative_to(REPO_ROOT)) if input_path.is_relative_to(REPO_ROOT) else str(input_path)
    (case_dir / "06_metrics.json").write_text(json.dumps(metrics, indent=2))

    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate floorplan-to-3d pipeline")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "tests" / "floorplans",
        help="Folder of floor-plan cases",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "tests" / "results",
        help="Where to write visualizations and JSON",
    )
    p.add_argument("--run-dir", type=Path, default=None, help="Weights dir (default: weights/)")
    p.add_argument("--ckpt", default="best.safetensors")
    p.add_argument("--device", default="auto")
    p.add_argument("--skip-rooms", action="store_true", help="Skip Roboflow room detection")
    args = p.parse_args()

    run_dir = args.run_dir or Path(os.environ.get("BUILDINGCV_RUN_DIR", REPO_ROOT / "weights"))
    cases = _discover_cases(args.input_dir)
    if not cases:
        raise SystemExit(f"no floor plans found under {args.input_dir}")

    print(f"loading model from {run_dir / args.ckpt}")
    extractor = PolygonExtractor(run_dir=run_dir, ckpt=args.ckpt, device=args.device)
    print(f"device={extractor.device}  cases={len(cases)}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_out = args.out_dir / stamp
    run_out.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    for name, path in cases:
        print(f"  evaluating {name} ({path.name})…")
        try:
            metrics = evaluate_one(
                extractor, name, path, run_out, skip_rooms=args.skip_rooms
            )
            summary.append(metrics)
            s = metrics["structural"]
            r = metrics["rooms"]
            print(
                f"    walls={s['walls']} doors={s['doors']} windows={s['windows']} "
                f"rooms={r['count']} types={r['types']}"
            )
        except Exception as exc:
            print(f"    FAILED: {exc}")
            summary.append({"case": name, "error": str(exc)})

    report = {
        "generated_at": stamp,
        "input_dir": str(args.input_dir),
        "cases": summary,
    }
    (run_out / "summary.json").write_text(json.dumps(report, indent=2))
    # Also write a stable "latest" pointer copy of summary at out-dir root.
    (args.out_dir / "latest_summary.json").write_text(json.dumps(report, indent=2))
    print(f"wrote results to {run_out}")


if __name__ == "__main__":
    main()
