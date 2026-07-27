"""Render a CubiCasa SVG to an RGB image, dropping non-structural content.

The mask side of the dataset (svg_to_mask.py) only labels walls/doors/windows;
everything else collapses to `floor`. But cairosvg renders *all* SVG content
into the input image — furniture, fixtures, dimension lines, drains, text —
so the model is shown furniture and told it's floor. That input/label
mismatch is what's currently leaking onto wall predictions: thin black
furniture lines look like walls.

This module strips the non-structural subtrees *before* rendering, so the
input image carries only walls + doors + windows + open floor, matching the
mask's semantics.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import fitz
from PIL import Image
import io

# Default-prefix the SVG namespace on serialization so the round-tripped
# document is plain `<svg xmlns="...">` rather than `<ns0:svg xmlns:ns0="...">`.
# cairosvg accepts both, but the default-namespace form is what the original
# files use and it keeps debug-saved SVGs human-readable.
ET.register_namespace("", "http://www.w3.org/2000/svg")

# Class tokens (space-separated values of the SVG `class` attribute) that
# identify subtrees the model should not see in the input image. A node is
# dropped if any of its own class tokens appears here — children are dropped
# transitively because we remove the whole subtree, not just the tagged node.
DROP_TOKENS: frozenset[str] = frozenset(
    {
        # Furniture and fixtures — the dominant source of wall-IoU damage.
        "FixedFurniture",
        "FixedFurnitureSet",
        # Plumbing/hardware bits that aren't structural:
        "Faucet",
        "Hanger",
        "Railing",
        "InnerDrain",
        "OuterDrain",
        "OuterCircle",
        # Drafting annotations: dimensions, labels, north-arrows:
        "Dimension",
        "DimensionMark",
        "Direction",
        "TextLabel",
        "Name",
        "SpaceDimensionsLabel",
        # Stray legend-style markers:
        "electricitySign",
        # common user tokens
        "furniture",
        "sofa",
        "bed",
        "table",
        "chair",
        "sink",
        "toilet",
        "shower",
    }
)
DROP_TOKENS_LC: frozenset[str] = frozenset(t.lower() for t in DROP_TOKENS)


def _prune(elem: ET.Element) -> None:
    """Remove direct children whose `class` attr contains any DROP token, recursively."""
    for child in list(elem):
        tokens = {(t or "").strip() for t in (child.get("class") or "").split()}
        if any(t.lower() in DROP_TOKENS_LC for t in tokens if t):
            elem.remove(child)
        else:
            _prune(child)


def filtered_svg_bytes(svg_path: str | Path) -> bytes:
    """Parse `svg_path`, drop non-structural subtrees, return serialized bytes."""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    _prune(root)

    # Normalize root width/height when they are percentages or use
    # non-pixel units. Some SVG rasterizers (PyMuPDF/fitz) don't handle
    # percent sizes consistently; prefer explicit pixel sizes derived
    # from the viewBox when available so rasterization is deterministic.
    w_attr = (root.get("width") or "").strip()
    h_attr = (root.get("height") or "").strip()
    viewbox = (root.get("viewBox") or "").split()

    def is_percent_or_unit(s: str) -> bool:
        if not s:
            return True
        if s.endswith("%"):
            return True
        # treat anything that isn't a plain number or a px-suffixed number
        # as a unit-bearing value that we don't trust for pixel sizing.
        try:
            float(s)
            return False
        except Exception:
            if s.endswith("px"):
                try:
                    float(s[:-2])
                    return False
                except Exception:
                    return True
            return True

    if len(viewbox) == 4 and (is_percent_or_unit(w_attr) or is_percent_or_unit(h_attr)):
        try:
            vb_w = float(viewbox[2])
            vb_h = float(viewbox[3])
            # Use integer pixel sizes for rasterizers.
            root.set("width", str(int(round(vb_w))))
            root.set("height", str(int(round(vb_h))))
        except Exception:
            # Fall back to leaving attributes as-is if parsing fails.
            pass
    else:
        # If width/height are px-suffixed, normalize to bare numbers (fitz
        # prefers plain numeric attributes).
        if w_attr.endswith("px"):
            try:
                root.set("width", str(int(round(float(w_attr[:-2])))))
            except Exception:
                pass
        if h_attr.endswith("px"):
            try:
                root.set("height", str(int(round(float(h_attr[:-2])))))
            except Exception:
                pass

    return ET.tostring(root, encoding="utf-8")


# ImageNet mean expressed in 0–255 RGB. Used as the cairosvg background color
# so that pixels outside any drawn polygon match the letterbox padding fill —
# both "non-content" regions look identical to the model after normalization,
# and neither collides with the black used for walls.
_IMAGENET_MEAN_RGB = "rgb(124, 116, 104)"


def svg2png(bytestring: bytes, output_width: int, output_height: int, background_color: str | None = None) -> bytes:
    doc = fitz.open(stream=bytestring, filetype="svg")
    page = doc[0]
    rect = page.rect
    w, h = rect.width, rect.height
    
    zoom_x = output_width / w if w else 1.0
    zoom_y = output_height / h if h else 1.0
    
    mat = fitz.Matrix(zoom_x, zoom_y)
    pix = page.get_pixmap(matrix=mat, alpha=True)
    png_data = pix.tobytes("png")
    
    if background_color:
        if background_color.startswith("rgb"):
            try:
                parts = background_color.replace("rgb(", "").replace(")", "").split(",")
                bg_tuple = tuple(int(x.strip()) for x in parts)
            except Exception:
                bg_tuple = (255, 255, 255)
        else:
            bg_tuple = (255, 255, 255)
            
        img = Image.open(io.BytesIO(png_data))
        bg = Image.new("RGBA", img.size, bg_tuple + (255,))
        bg.alpha_composite(img)
        
        out = io.BytesIO()
        bg.convert("RGB").save(out, format="PNG")
        return out.getvalue()
        
    return png_data


def render_input_png(svg_path: str | Path, out_size: tuple[int, int]) -> bytes:
    """Render the structural-only SVG to PNG bytes at (W, H).

    Output is pixel-aligned with `svg_to_mask.svg_to_mask` at the same size,
    so the (image, mask) pair stays consistent.
    """
    W, H = out_size
    return svg2png(
        bytestring=filtered_svg_bytes(svg_path),
        output_width=W,
        output_height=H,
        background_color=_IMAGENET_MEAN_RGB,
    )
