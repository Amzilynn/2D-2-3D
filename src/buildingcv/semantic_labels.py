from __future__ import annotations

# Semantic categories for future room/object models.
# Keep these separate from `src/buildingcv/labels.py` (structural wall/door/window).

ROOM_TYPES: tuple[str, ...] = (
    "bedroom",
    "kitchen",
    "living room",
    "bathroom",
    "corridor",
    "dining room",
    "storage",
    "laundry",
    "office",
    "study",
)

OBJECT_TYPES: tuple[str, ...] = (
    "bed",
    "sofa",
    "chair",
    "table",
    "desk",
    "cabinet",
    "shelf",
    "tv_stand",
    "sink",
    "toilet",
    "shower",
    "bathtub",
)

