from __future__ import annotations

from typing import TypedDict


class Polygon(TypedDict):
    outer: list[list[float]]
    holes: list[list[list[float]]]


class Room(TypedDict):
    type: str
    polygon: Polygon
    # Label anchor in the same 2D image coordinate space as polygon vertices.
    # JSON-safe representation: [x, y]
    label_anchor: list[float]
    confidence: float


class Object(TypedDict):
    type: str
    polygon: Polygon
    confidence: float

