"""Convert a CubiCasa5K `model.svg` into a per-pixel class mask.

The SVG is XML with nested <g> groups carrying class attributes. Architectural
elements appear as polygons (and occasionally rects) whose class — directly or
inherited from an ancestor group — names them. We walk the tree, collect every
fillable shape with a recognized class, and paint them onto a 2D array in the
order defined by PAINT_ORDER.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .labels import CLASS_TO_ID, FLOOR_ID, PAINT_ORDER, TOKEN_TO_CLASS

SVG_NS = "{http://www.w3.org/2000/svg}"
_POLYGON_TAG = f"{SVG_NS}polygon"
_RECT_TAG = f"{SVG_NS}rect"
_PATH_TAG = f"{SVG_NS}path"

# Class tokens that identify non-structural elements which should be
# ignored when painting the mask. Matching is case-insensitive.
IGNORE_TOKENS: frozenset[str] = frozenset(
    {
        "furniture",
        "fixedfurniture",
        "fixedfurnitureset",
        "faucet",
        "hanger",
        "railing",
        "innerdrain",
        "outerdrain",
        "outercircle",
        "direction",
        "textlabel",
        "name",
        "spacedimensionslabel",
        "electricitysign",
        # common generic tokens from user SVGs
        "sofa",
        "bed",
        "table",
        "chair",
        "kitchen",
        "counter",
        "sink",
        "toilet",
        "shower",
    }
)
_PATH_TAG = f"{SVG_NS}path"


def _classify(tokens: frozenset[str]) -> str | None:
    # Match token names case-insensitively so user-generated SVGs with
    # lowercase class attributes (e.g. class="wall") still map correctly.
    tokens_lc = {t.lower() for t in tokens}
    for token, cls in TOKEN_TO_CLASS.items():
        if token.lower() in tokens_lc:
            return cls
    return None


def _is_hidden(elem: ET.Element) -> bool:
    """True if the element (or its style) explicitly hides it from rendering.

    cairosvg honors `display:none` and `visibility:hidden` when rasterizing
    the input image. Without this check, svg_to_mask would still paint the
    hidden subtrees onto the mask — which is exactly the bug behind GT masks
    containing multiple floor plans (Floor 2/3 are display:none) while the
    input image shows only Floor 1.
    """
    if elem.get("display") == "none" or elem.get("visibility") == "hidden":
        return True
    style = elem.get("style") or ""
    return "display:none" in style.replace(" ", "") or "visibility:hidden" in style.replace(" ", "")


def _polygon_points(elem: ET.Element) -> list[tuple[float, float]]:
    raw = (elem.get("points") or "").replace(",", " ").split()
    coords = [float(x) for x in raw]
    return list(zip(coords[0::2], coords[1::2]))


def _rect_points(elem: ET.Element) -> list[tuple[float, float]]:
    x = float(elem.get("x", 0))
    y = float(elem.get("y", 0))
    w = float(elem.get("width", 0))
    h = float(elem.get("height", 0))
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def _path_points(elem: ET.Element) -> list[list[tuple[float, float]]]:
    """Parse simple SVG path `d` data into one or more polygon rings.

    This supports only straight-segment commands (M, L, H, V, Z) which is
    sufficient for most architectural outlines. Curves are ignored — if a
    path contains curves this function will skip those segments.
    Returns a list of rings (each ring is a list of (x,y) tuples).
    """
    d = (elem.get("d") or "").strip()
    if not d:
        return []

    tokens = []
    cur = ""
    # tokenise: separate letters and numbers
    for ch in d:
        if ch.isalpha():
            if cur.strip():
                tokens.append(cur.strip())
            cur = ""
            tokens.append(ch)
        else:
            cur += ch
    if cur.strip():
        tokens.append(cur.strip())

    rings: list[list[tuple[float, float]]] = []
    pts: list[tuple[float, float]] = []
    cx = cy = 0.0
    start_x = start_y = None
    it = iter(tokens)
    for tok in it:
        if not tok:
            continue
        if tok.upper() == "M":
            # Move to — may be absolute or relative. A subsequent M after Z
            # starts a new subpath/ring (required for evenodd wall shells).
            if pts:
                rings.append(pts)
                pts = []
            coords = next(it)
            nums = [float(x) for x in coords.replace(",", " ").split() if x]
            if len(nums) >= 2:
                x, y = nums[0], nums[1]
                if tok == "m":
                    x += cx
                    y += cy
                cx, cy = x, y
                start_x, start_y = x, y
                pts = [(cx, cy)]
                # extra coordinate pairs after M are treated as implicit L
                for i in range(2, len(nums), 2):
                    x, y = nums[i], nums[i + 1]
                    if tok == "m":
                        x += cx
                        y += cy
                    cx, cy = x, y
                    pts.append((cx, cy))
        elif tok.upper() == "L":
            coords = next(it)
            nums = [float(x) for x in coords.replace(",", " ").split() if x]
            for i in range(0, len(nums), 2):
                x, y = nums[i], nums[i + 1]
                if tok == "l":
                    x += cx
                    y += cy
                cx, cy = x, y
                pts.append((cx, cy))
        elif tok.upper() == "H":
            coords = next(it)
            nums = [float(x) for x in coords.replace(",", " ").split() if x]
            for x in nums:
                if tok == "h":
                    x += cx
                cx = x
                pts.append((cx, cy))
        elif tok.upper() == "V":
            coords = next(it)
            nums = [float(x) for x in coords.replace(",", " ").split() if x]
            for y in nums:
                if tok == "v":
                    y += cy
                cy = y
                pts.append((cx, cy))
        elif tok.upper() == "Z":
            # close path
            if pts:
                rings.append(pts)
                pts = []
                start_x = start_y = None
        else:
            # unsupported command (curves etc.) — consume following coordinate token
            try:
                _ = next(it)
            except StopIteration:
                pass

    # if a path didn't explicitly close, treat collected pts as a ring
    if pts:
        rings.append(pts)
    return rings


def _fill_rule_evenodd(elem: ET.Element) -> bool:
    """True when the element (or its style) requests evenodd fill."""
    rule = (elem.get("fill-rule") or "").strip().lower()
    if rule == "evenodd":
        return True
    style = elem.get("style") or ""
    return "fill-rule:evenodd" in style.replace(" ", "").lower()


def _collect(
    elem: ET.Element,
    inherited: frozenset[str],
    out: dict[str, list[list[tuple[float, float]]]],
    holes_out: dict[str, list[list[tuple[float, float]]]] | None = None,
) -> None:
    if holes_out is None:
        holes_out = {}
    if _is_hidden(elem):
        return

    own = elem.get("class")
    tokens = inherited | (frozenset(own.split()) if own else frozenset())

    # If any token marks this subtree as non-structural (furniture, labels,
    # etc), skip the entire subtree so we don't accidentally inherit a
    # parent `Wall` class on child furniture shapes.
    if any(t.lower() in IGNORE_TOKENS for t in tokens):
        return

    points: list[tuple[float, float]] | None = None
    if elem.tag == _POLYGON_TAG:
        points = _polygon_points(elem)
    elif elem.tag == _RECT_TAG:
        points = _rect_points(elem)
    elif elem.tag == _PATH_TAG:
        # Paths can encode filled walls in many CubiCasa files. Parse simple
        # straight-segment paths into polygon rings and add each ring.
        # evenodd compound paths (outer shell + inner cutout) must NOT paint
        # every ring as a solid fill — that turns the whole floor plate into
        # "wall". Treat ring 0 as exterior and later rings as holes.
        rings = _path_points(elem)
        if rings:
            cls = _classify(tokens)
            if cls is not None:
                valid = [ring for ring in rings if len(ring) >= 3]
                if not valid:
                    points = None
                elif _fill_rule_evenodd(elem) and len(valid) >= 2:
                    out.setdefault(cls, []).append(valid[0])
                    for hole in valid[1:]:
                        holes_out.setdefault(cls, []).append(hole)
                else:
                    for ring in valid:
                        out.setdefault(cls, []).append(ring)
        points = None

    if points is not None and len(points) >= 3:
        cls = _classify(tokens)
        if cls is not None:
            out.setdefault(cls, []).append(points)

    for child in elem:
        _collect(child, tokens, out, holes_out)


def svg_to_mask(
    svg_path: str | Path,
    size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Rasterize a CubiCasa SVG into a per-pixel class mask.

    Args:
        svg_path: path to a CubiCasa `model.svg`.
        size: (W, H) in pixels for the output. If None, the SVG's native
            viewBox/width/height is used.

    Returns:
        uint8 ndarray of shape (H, W). Each value is a class index in
        [0, NUM_CLASSES). 0 = floor (default fill).
    """
    root = ET.parse(svg_path).getroot()

    viewbox = (root.get("viewBox") or "").split()

    def _parse_len(s: str | None) -> float | None:
        if s is None:
            return None
        s = s.strip()
        if not s:
            return None
        if s.endswith("%"):
            return None
        if s.endswith("px"):
            try:
                return float(s[:-2])
            except Exception:
                return None
        try:
            return float(s)
        except Exception:
            return None

    w_val = _parse_len(root.get("width"))
    h_val = _parse_len(root.get("height"))
    if (w_val is None or h_val is None) and len(viewbox) == 4:
        try:
            vb_w = float(viewbox[2])
            vb_h = float(viewbox[3])
            if w_val is None:
                w_val = vb_w
            if h_val is None:
                h_val = vb_h
        except Exception:
            pass

    if w_val is None or h_val is None or w_val <= 0 or h_val <= 0:
        raise ValueError(f"could not determine SVG size for {svg_path}")

    svg_w = w_val
    svg_h = h_val

    if size is None:
        out_w, out_h = int(round(svg_w)), int(round(svg_h))
        sx = sy = 1.0
    else:
        out_w, out_h = size
        sx = out_w / svg_w
        sy = out_h / svg_h

    polys_by_class: dict[str, list[list[tuple[float, float]]]] = {}
    holes_by_class: dict[str, list[list[tuple[float, float]]]] = {}
    _collect(root, frozenset(), polys_by_class, holes_by_class)

    mask_img = Image.new("L", (out_w, out_h), color=FLOOR_ID)
    draw = ImageDraw.Draw(mask_img)
    for cls in PAINT_ORDER:
        idx = CLASS_TO_ID[cls]
        for pts in polys_by_class.get(cls, []):
            draw.polygon([(x * sx, y * sy) for x, y in pts], fill=idx)
        # Punch evenodd inner rings back to floor so compound wall shells
        # keep a hollow interior instead of a solid filled plate.
        for pts in holes_by_class.get(cls, []):
            draw.polygon([(x * sx, y * sy) for x, y in pts], fill=FLOOR_ID)

    return np.array(mask_img, dtype=np.uint8)
