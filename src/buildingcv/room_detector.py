"""Roboflow Architectural Blueprint room detection for semantic rooms[]."""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .semantic_schema import Room

logger = logging.getLogger(__name__)

ROBOFLOW_API_URL = "https://serverless.roboflow.com"
DEFAULT_CONFIDENCE = 0.4
REQUEST_TIMEOUT_S = 45.0

# Roboflow class names -> our room schema types.
_CLASS_TO_ROOM_TYPE: dict[str, str] = {
    "space_bedroom": "bedroom",
    "space_kitchen": "kitchen",
    "space_living_room": "living room",
    "space_toilet": "bathroom",
    "space_balconi": "corridor",
    "space_dressroom": "storage",
    "space_elevator": "corridor",
    "space_elevator_hall": "corridor",
    "space_front": "corridor",
    "space_multipurpose_space": "office",
    "space_other": "storage",
    "space_outdoor_room": "corridor",
    "space_staircase": "corridor",
    "bedroom": "bedroom",
    "kitchen": "kitchen",
    "living room": "living room",
    "living_room": "living room",
    "bathroom": "bathroom",
    "toilet": "bathroom",
}

_SKIP_CLASSES = frozenset({"background", "space_background"})


def _normalize_class_name(name: str) -> str:
    return (name or "").strip().lower().replace("-", "_").replace(" ", "_")


def _map_class_to_room_type(class_name: str) -> str | None:
    key = _normalize_class_name(class_name)
    if key in _SKIP_CLASSES:
        return None
    if key in _CLASS_TO_ROOM_TYPE:
        return _CLASS_TO_ROOM_TYPE[key]
    if key.startswith("space_"):
        return key.removeprefix("space_").replace("_", " ")
    return class_name.strip().lower() or None


def _bbox_to_polygon(x: float, y: float, w: float, h: float) -> dict[str, list]:
    """Convert Roboflow center-format bbox to our polygon schema."""
    xmin = x - w / 2.0
    ymin = y - h / 2.0
    xmax = x + w / 2.0
    ymax = y + h / 2.0
    outer = [
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax],
    ]
    return {"outer": outer, "holes": []}


def _offset_polygon(polygon: dict[str, list], dx: float, dy: float) -> dict[str, list]:
    outer = [[x + dx, y + dy] for x, y in polygon["outer"]]
    holes = [[[x + dx, y + dy] for x, y in ring] for ring in polygon.get("holes", [])]
    return {"outer": outer, "holes": holes}


def _prediction_to_room(pred: dict[str, Any], offset_x: float, offset_y: float) -> Room | None:
    class_name = pred.get("class") or pred.get("class_name") or pred.get("label")
    room_type = _map_class_to_room_type(str(class_name or ""))
    if room_type is None:
        return None

    try:
        x = float(pred["x"])
        y = float(pred["y"])
        w = float(pred["width"])
        h = float(pred["height"])
        confidence = float(pred.get("confidence", 0.0))
    except (KeyError, TypeError, ValueError):
        return None

    if w <= 0 or h <= 0:
        return None

    polygon = _offset_polygon(_bbox_to_polygon(x, y, w, h), offset_x, offset_y)
    label_anchor = [x + offset_x, y + offset_y]
    return {
        "type": room_type,
        "polygon": polygon,
        "label_anchor": label_anchor,
        "confidence": round(confidence, 4),
    }


def _roboflow_config() -> tuple[str, str, float] | None:
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    model_id = os.environ.get("ROBOFLOW_MODEL_ID", "").strip()
    if not api_key or not model_id:
        return None
    try:
        confidence = float(os.environ.get("ROBOFLOW_CONFIDENCE", DEFAULT_CONFIDENCE))
    except ValueError:
        confidence = DEFAULT_CONFIDENCE
    return api_key, model_id, confidence


def _infer_url(model_id: str, api_key: str, confidence: float) -> str:
    model_id = model_id.strip("/")
    query = urlencode({"api_key": api_key, "confidence": confidence})
    return f"{ROBOFLOW_API_URL}/{model_id}?{query}"


def _post_image(url: str, png_bytes: bytes) -> dict[str, Any]:
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="floorplan.png"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode("utf-8") + png_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
    req = Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _extract_predictions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get("predictions"), list):
        return payload["predictions"]
    outputs = payload.get("outputs")
    if isinstance(outputs, list):
        preds: list[dict[str, Any]] = []
        for item in outputs:
            if isinstance(item, dict) and isinstance(item.get("predictions"), list):
                preds.extend(item["predictions"])
        if preds:
            return preds
    return []


def detect_rooms_from_png(
    png_bytes: bytes,
    content_rect: list[int] | tuple[int, int, int, int],
) -> list[Room]:
    """Run Roboflow room detection and return normalized rooms[] entries.

    Detections are returned in canvas pixel coordinates by offsetting the
    content-sized inference image with content_rect's left/top padding.
    """
    config = _roboflow_config()
    if config is None:
        return []

    api_key, model_id, confidence = config
    left, top, _, _ = content_rect

    try:
        payload = _post_image(_infer_url(model_id, api_key, confidence), png_bytes)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        logger.warning("room detection failed: %s", exc)
        return []

    rooms: list[Room] = []
    for pred in _extract_predictions(payload):
        room = _prediction_to_room(pred, float(left), float(top))
        if room is not None:
            rooms.append(room)
    return rooms
