"""Parse Pascal VOC XML annotation strings into the canonical objects[] struct.

Round-trip guarantee: parse → de-normalise back to integer pixel xyxy must
recover the original VOC bbox EXACTLY (zero-pixel drift). Drift indicates
either a normalisation bug or a W/H mismatch with the embedded image.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedObject:
    class_name: str
    xmin: float  # pixel-absolute (float to preserve any sub-pixel VOC values)
    ymin: float
    xmax: float
    ymax: float


@dataclass
class VocParseResult:
    width: int
    height: int
    objects: list[ParsedObject]
    raw_size_in_xml: Optional[tuple[int, int]]  # what <size> claimed


def parse_voc_xml(xml_text: str) -> VocParseResult:
    """Parse VOC XML. Does NOT trust <size> — caller must pass image W,H separately
    when normalising. We still record what XML claimed for cross-check."""
    root = ET.fromstring(xml_text)
    size_el = root.find("size")
    raw_wh = None
    if size_el is not None:
        try:
            raw_wh = (int(size_el.findtext("width")), int(size_el.findtext("height")))
        except (TypeError, ValueError):
            raw_wh = None

    objs: list[ParsedObject] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        bb = obj.find("bndbox")
        if bb is None:
            raise ValueError(f"VOC object missing <bndbox>: name={name!r}")
        xmin = float(bb.findtext("xmin"))
        ymin = float(bb.findtext("ymin"))
        xmax = float(bb.findtext("xmax"))
        ymax = float(bb.findtext("ymax"))
        # Mark degenerate bboxes (zero or negative area). Caller decides policy
        # (drop + warn vs raise) since detection vs rule_violation differ.
        is_degen = xmax <= xmin or ymax <= ymin
        po = ParsedObject(name, xmin, ymin, xmax, ymax)
        po.degenerate = is_degen  # type: ignore[attr-defined]
        objs.append(po)

    return VocParseResult(
        width=raw_wh[0] if raw_wh else 0,
        height=raw_wh[1] if raw_wh else 0,
        objects=objs,
        raw_size_in_xml=raw_wh,
    )


def normalise_objects(
    parsed: VocParseResult,
    image_w: int,
    image_h: int,
    class_to_id: dict[str, int],
    *,
    strict_size_check: bool = True,
    out_of_bounds_tol: float = 1.0,  # pixels of tolerance before flagging
) -> tuple[list[dict], list[dict]]:
    """Convert ParsedObject -> objects[] struct dicts using image_w,h for normalisation.

    Returns (objects_list, warnings). Each warning is a dict suitable for the audit log.
    Raises on hard errors (unknown class, bbox outside image beyond tolerance).
    """
    warnings: list[dict] = []

    if strict_size_check and parsed.raw_size_in_xml is not None:
        if parsed.raw_size_in_xml != (image_w, image_h):
            # Hard error: VOC bbox is pixel-absolute. If we don't know whether
            # those pixels reference the XML's claimed size or the actual image
            # size, we cannot normalise without risking silent data drift.
            # Caller must reconcile (e.g., trust XML, re-encode image, etc.)
            # before invoking the converter.
            raise ValueError(
                f"VOC <size>={parsed.raw_size_in_xml} disagrees with image (W,H)={(image_w, image_h)}; "
                f"refusing to normalise bbox to avoid silent drift. Reconcile upstream."
            )

    out: list[dict] = []
    for o in parsed.objects:
        if getattr(o, "degenerate", False):
            warnings.append({
                "kind": "degenerate_voc_bbox",
                "class": o.class_name,
                "raw_bbox": (o.xmin, o.ymin, o.xmax, o.ymax),
            })
            continue
        if o.class_name not in class_to_id:
            raise ValueError(f"unknown class {o.class_name!r}; extend taxonomy first")
        # Check bounds (pixel space)
        if (o.xmin < -out_of_bounds_tol or o.ymin < -out_of_bounds_tol
                or o.xmax > image_w + out_of_bounds_tol or o.ymax > image_h + out_of_bounds_tol):
            raise ValueError(
                f"bbox out of image bounds for class {o.class_name!r}: "
                f"({o.xmin},{o.ymin},{o.xmax},{o.ymax}) image=({image_w},{image_h})"
            )
        # Clamp to [0,W]/[0,H] only the tiny tolerance overshoot (logged)
        xmin = max(0.0, min(o.xmin, float(image_w)))
        ymin = max(0.0, min(o.ymin, float(image_h)))
        xmax = max(0.0, min(o.xmax, float(image_w)))
        ymax = max(0.0, min(o.ymax, float(image_h)))
        if (xmin, ymin, xmax, ymax) != (o.xmin, o.ymin, o.xmax, o.ymax):
            warnings.append({
                "kind": "bbox_tol_clamp",
                "class": o.class_name,
                "raw": (o.xmin, o.ymin, o.xmax, o.ymax),
                "clamped": (xmin, ymin, xmax, ymax),
            })
        out.append({
            "class_id": class_to_id[o.class_name],
            "class_name": o.class_name,
            "bbox": [xmin / image_w, ymin / image_h, xmax / image_w, ymax / image_h],
        })
    return out, warnings


def denormalise_for_round_trip(obj: dict, image_w: int, image_h: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = obj["bbox"]
    return (x1 * image_w, y1 * image_h, x2 * image_w, y2 * image_h)
