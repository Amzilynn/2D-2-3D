"""Quick check for room boundary refinement."""

from __future__ import annotations

import json
from pathlib import Path

from buildingcv.room_refinement import refine_rooms

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = REPO_ROOT / "viewer" / "demos" / "semantic_rooms_example.json"


def _wall_band(x0: float, y0: float, x1: float, y1: float) -> dict:
    return {
        "outer": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        "holes": [],
    }


def synthetic_walls() -> list[dict]:
    """Walls that enclose the three rooms in semantic_rooms_example.json."""
    t = 8.0
    return [
        _wall_band(50, 50, 445, 50 + t),          # top
        _wall_band(50, 475 - t, 445, 475),        # bottom
        _wall_band(50, 50, 50 + t, 475),          # left
        _wall_band(445 - t, 50, 445, 475),        # right
        _wall_band(230, 50, 230 + t, 250),        # bedroom | kitchen
        _wall_band(50, 250, 445, 250 + t),        # top row | living room
    ]


def main() -> None:
    data = json.loads(EXAMPLE.read_text())
    rooms_before = data["rooms"]
    walls = synthetic_walls()
    rooms_after = refine_rooms(rooms_before, walls, data["canvas_size"])

    print("before bedroom:", rooms_before[0]["polygon"]["outer"])
    print("after  bedroom:", rooms_after[0]["polygon"]["outer"])
    print("before kitchen:", rooms_before[1]["polygon"]["outer"])
    print("after  kitchen:", rooms_after[1]["polygon"]["outer"])
    print("before living :", rooms_before[2]["polygon"]["outer"])
    print("after  living :", rooms_after[2]["polygon"]["outer"])


if __name__ == "__main__":
    main()
