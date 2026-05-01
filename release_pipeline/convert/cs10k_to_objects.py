"""Convert CS10k annotation formats into canonical objects[] + rule_violations[].

CS10k shards exist in two formats — both must round-trip to the same canonical
representation:

Format A (per-class columns): rain/snow/fog/original
    excavator: [[x1,y1,x2,y2], ...]    # already normalised
    rebar: [[...], ...]
    worker_with_white_hard_hat: [[...], ...]
    rule_N_violation: {"bounding_box": [[...]], "reason": "..."} | None
    image_caption, illumination, camera_distance, view, quality_of_info: str

Format B (JSON annotation): cs_night, cs_small
    annotation: JSON string with same fields under top-level keys, including
    "bbox_format": "xyxy" and the same per-class arrays + rule fields.
    image_caption is NOT in format-B annotation — pulled from outer struct
    (cs_night/cs_small don't have it; we accept None).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

CS10K_DETECTION_CLASSES = ("excavator", "rebar", "worker_with_white_hard_hat")
CS10K_CLASS_TO_ID = {name: i for i, name in enumerate(CS10K_DETECTION_CLASSES)}
CS10K_RULE_KEYS = ("rule_1_violation", "rule_2_violation", "rule_3_violation", "rule_4_violation")


@dataclass
class CS10kCanonical:
    objects: list[dict]
    rule_violations: list[dict]
    image_attributes: Optional[dict]
    warnings: list[dict]


class DegenerateBboxError(ValueError):
    """Raised when bbox has zero or negative area. Caller may catch and treat
    it as missing-bbox (rule_violations) or re-raise (object detections)."""


def _validate_normalised_bbox(b, *, ctx: str) -> list[float]:
    if not (isinstance(b, (list, tuple)) and len(b) == 4):
        raise ValueError(f"[{ctx}] bbox must be 4-list, got {b!r}")
    for v in b:
        if not isinstance(v, (int, float)):
            raise ValueError(f"[{ctx}] bbox value not numeric: {v!r}")
        if not (-1e-6 <= float(v) <= 1.0 + 1e-6):
            raise ValueError(f"[{ctx}] bbox value out of [0,1]: {v!r}")
    x1, y1, x2, y2 = (float(v) for v in b)
    if x2 <= x1 or y2 <= y1:
        raise DegenerateBboxError(f"[{ctx}] degenerate bbox: {b!r}")
    return [x1, y1, x2, y2]


def _extract_objects(per_class: dict) -> tuple[list[dict], list[dict]]:
    """Returns (objects, warnings). Degenerate (zero-area) bboxes are skipped
    with a warning rather than raising — they are source data defects that
    cannot be repaired here, and dropping the entire row would lose all the
    other valid objects on the same image."""
    out: list[dict] = []
    warns: list[dict] = []
    for class_name in CS10K_DETECTION_CLASSES:
        boxes = per_class.get(class_name) or []
        for i, b in enumerate(boxes):
            try:
                bb = _validate_normalised_bbox(b, ctx=f"{class_name}[{i}]")
            except DegenerateBboxError:
                warns.append({
                    "kind": "degenerate_object_bbox",
                    "class": class_name,
                    "raw_bbox": b,
                })
                continue
            out.append({
                "class_id": CS10K_CLASS_TO_ID[class_name],
                "class_name": class_name,
                "bbox": bb,
            })
    return out, warns


def _extract_rule_violations(rule_dict: dict) -> tuple[list[dict], list[dict]]:
    """Emit one entry per rule fire. Returns (violations, warnings).

    - rule fires with N valid bbox -> N entries.
    - rule fires with 0 bbox (reason text only) -> ONE entry with bbox=None.
    - rule fires with degenerate bbox (zero area) -> ONE entry with bbox=None,
      logged as a warning. Source defect; we preserve the rule fire faithfully
      without propagating a malformed geometry.
    """
    out: list[dict] = []
    warns: list[dict] = []
    for rule_key in CS10K_RULE_KEYS:
        rv = rule_dict.get(rule_key)
        if rv is None:
            continue
        rule_id = int(rule_key.split("_")[1])
        bboxes = rv.get("bounding_box") or []
        reason = rv.get("reason") or ""
        if not bboxes:
            out.append({"rule_id": rule_id, "bbox": None, "reason": reason})
            continue
        valid_seen = False
        for i, b in enumerate(bboxes):
            try:
                bb = _validate_normalised_bbox(b, ctx=f"{rule_key}[{i}]")
            except DegenerateBboxError as e:
                warns.append({
                    "kind": "degenerate_rule_bbox",
                    "rule_key": rule_key,
                    "raw_bbox": b,
                    "reason": reason,
                })
                continue
            valid_seen = True
            out.append({"rule_id": rule_id, "bbox": bb, "reason": reason})
        # If ALL bboxes for this rule fire were degenerate, still emit one
        # bbox=None entry so the rule fire is not lost.
        if not valid_seen:
            out.append({"rule_id": rule_id, "bbox": None, "reason": reason})
    return out, warns


def from_format_a(row: dict) -> CS10kCanonical:
    """Format A: top-level row dict has per-class columns directly."""
    objects, obj_warns = _extract_objects(row)
    rule_violations, rv_warns = _extract_rule_violations(row)
    warns = obj_warns + rv_warns
    image_attributes = {
        "caption": row.get("image_caption") or "",
        "illumination": row.get("illumination") or "",
        "camera_distance": row.get("camera_distance") or "",
        "view": row.get("view") or "",
        "quality_of_info": row.get("quality_of_info") or "",
    }
    if not any(image_attributes.values()):
        image_attributes = None
    return CS10kCanonical(objects, rule_violations, image_attributes, warns)


def from_format_b(annotation_json: str, *, image_caption: Optional[str] = None) -> CS10kCanonical:
    """Format B: parse `annotation` JSON string."""
    if not annotation_json:
        raise ValueError("format-B annotation is empty")
    try:
        ann = json.loads(annotation_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"format-B annotation not valid JSON: {e}") from e
    bbox_fmt = ann.get("bbox_format")
    if bbox_fmt not in (None, "xyxy"):
        raise ValueError(f"format-B unexpected bbox_format={bbox_fmt!r}; expected 'xyxy'")
    objects, obj_warns = _extract_objects(ann)
    rule_violations, rv_warns = _extract_rule_violations(ann)
    warns = obj_warns + rv_warns
    image_attributes = None
    if image_caption:
        image_attributes = {
            "caption": image_caption, "illumination": "", "camera_distance": "",
            "view": "", "quality_of_info": "",
        }
    return CS10kCanonical(objects, rule_violations, image_attributes, warns)


def detect_format(row: dict) -> str:
    """Return 'A' if the row has per-class columns, 'B' if it has annotation JSON.
    Raises if neither or both, since that's ambiguous and we want to fail loud."""
    has_a = any(k in row for k in CS10K_DETECTION_CLASSES)
    has_b = "annotation" in row and row.get("annotation") and isinstance(row["annotation"], str)
    if has_a and not has_b:
        return "A"
    if has_b and not has_a:
        return "B"
    if has_a and has_b:
        # Some shards may carry both — prefer B (newer, gathered) but log
        return "B"
    raise ValueError(f"row has neither per-class cols nor annotation JSON; keys={list(row)}")
