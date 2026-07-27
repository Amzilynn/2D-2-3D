"""Refine room polygons using structural wall geometry."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .semantic_schema import Polygon, Room

logger = logging.getLogger(__name__)

WALL_DILATE_PX = 3
MIN_REGION_AREA_PX = 200.0
APPROX_EPSILON_PX = 2.0


def _room_center(room: Room) -> tuple[float, float]:
    anchor = room.get("label_anchor")
    if anchor and len(anchor) >= 2:
        return float(anchor[0]), float(anchor[1])
    outer = room["polygon"]["outer"]
    sx = sum(p[0] for p in outer)
    sy = sum(p[1] for p in outer)
    n = max(len(outer), 1)
    return sx / n, sy / n


def _polygon_centroid(polygon: Polygon) -> tuple[float, float]:
    outer = polygon["outer"]
    sx = sum(p[0] for p in outer)
    sy = sum(p[1] for p in outer)
    n = max(len(outer), 1)
    return sx / n, sy / n


def _rasterize_walls(wall_polygons: list[Polygon], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in wall_polygons:
        outer = np.array(poly["outer"], dtype=np.int32)
        if outer.shape[0] < 3:
            continue
        cv2.fillPoly(mask, [outer], 1)
        for hole in poly.get("holes", []):
            hole_arr = np.array(hole, dtype=np.int32)
            if hole_arr.shape[0] >= 3:
                cv2.fillPoly(mask, [hole_arr], 0)
    if WALL_DILATE_PX > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (WALL_DILATE_PX, WALL_DILATE_PX)
        )
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def _mark_exterior(free_mask: np.ndarray) -> np.ndarray:
    """Return a mask of free pixels connected to the image border."""
    h, w = free_mask.shape
    exterior = np.zeros((h, w), dtype=np.uint8)
    queue: list[tuple[int, int]] = []

    for x in range(w):
        if free_mask[0, x]:
            queue.append((x, 0))
        if free_mask[h - 1, x]:
            queue.append((x, h - 1))
    for y in range(h):
        if free_mask[y, 0]:
            queue.append((0, y))
        if free_mask[y, w - 1]:
            queue.append((w - 1, y))

    while queue:
        x, y = queue.pop()
        if exterior[y, x]:
            continue
        if not free_mask[y, x]:
            continue
        exterior[y, x] = 1
        if x > 0:
            queue.append((x - 1, y))
        if x < w - 1:
            queue.append((x + 1, y))
        if y > 0:
            queue.append((x, y - 1))
        if y < h - 1:
            queue.append((x, y + 1))
    return exterior


def _contour_to_polygon(contour: np.ndarray) -> Polygon | None:
    simplified = cv2.approxPolyDP(contour, APPROX_EPSILON_PX, closed=True)
    if simplified.shape[0] < 3:
        return None
    outer = simplified.reshape(-1, 2).astype(float).tolist()
    return {"outer": outer, "holes": []}


def _region_polygon_for_center(
    interior_labels: np.ndarray,
    center_x: float,
    center_y: float,
) -> Polygon | None:
    h, w = interior_labels.shape
    cx = int(round(center_x))
    cy = int(round(center_y))
    if cx < 0 or cy < 0 or cx >= w or cy >= h:
        return None

    label = int(interior_labels[cy, cx])
    if label <= 0:
        return None

    region = (interior_labels == label).astype(np.uint8)
    if float(region.sum()) < MIN_REGION_AREA_PX:
        return None

    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < MIN_REGION_AREA_PX:
        return None
    return _contour_to_polygon(contour)


def refine_room_polygon(
    room: Room,
    wall_polygons: list[Polygon],
    canvas_size: tuple[int, int],
) -> Room:
    """Refine one room polygon; fall back to the input polygon on failure."""
    width, height = canvas_size
    if width <= 0 or height <= 0 or not wall_polygons:
        return room

    try:
        wall_mask = _rasterize_walls(wall_polygons, width, height)
        free_mask = wall_mask == 0
        exterior = _mark_exterior(free_mask)
        interior = free_mask & (exterior == 0)
        if not interior.any():
            return room

        num_labels, labels = cv2.connectedComponents(interior.astype(np.uint8))
        if num_labels <= 1:
            return room

        cx, cy = _room_center(room)
        refined = _region_polygon_for_center(labels, cx, cy)
        if refined is None:
            return room

        refined_room = dict(room)
        refined_room["polygon"] = refined
        refined_room["label_anchor"] = list(_polygon_centroid(refined))
        return refined_room  # type: ignore[return-value]
    except Exception as exc:
        logger.warning("room refinement failed for %s: %s", room.get("type"), exc)
        return room


def refine_rooms(
    rooms: list[Room],
    wall_polygons: list[Polygon],
    canvas_size: tuple[int, int] | list[int],
) -> list[Room]:
    """Refine all room polygons using wall geometry."""
    if not rooms:
        return []
    size = (int(canvas_size[0]), int(canvas_size[1]))
    return [refine_room_polygon(room, wall_polygons, size) for room in rooms]
