"""Emit COCO-format JSON + image files from canonical release records.

Streaming design: open(out_dir) -> add_record(rec) for each row -> finalize().
Image bytes are written to disk and IMMEDIATELY released by the caller — the
emitter only retains JSON-shaped metadata (small) per image/annotation.

COCO compliance:
  - bbox: [x, y, w, h] in absolute pixels (NOT normalised, NOT xyxy).
  - area: w*h (float).
  - iscrowd: 0.
  - integer image_id and annotation_id (auto-assigned, monotonically increasing).
  - file_name references an image written next to annotations.json.

Custom extras (pycocotools tolerates unknown keys):
  - images[*].consynth_extra: {consynth_image_id, source_id, source_dataset,
        condition, condition_labels, pipeline, quality_scores, quality_alert,
        cs10k_image_attributes?, ktsh_captions?, rule_violations_no_bbox?}
  - annotations[*].consynth_extra: {kind: 'object'|'rule_violation', reason?}
"""
from __future__ import annotations

import json
import os
from typing import Iterable

from .integrity import atomic_write, image_size, sha256_bytes


def _category_table(class_names: list[str], rule_ids: list[int] | None) -> tuple[list[dict], dict]:
    cats: list[dict] = []
    name_to_id: dict[str, int] = {}
    for i, name in enumerate(class_names, start=1):
        cats.append({"id": i, "name": name, "supercategory": "object"})
        name_to_id[name] = i
    if rule_ids:
        for rid in rule_ids:
            name = f"rule_{rid}_violation"
            cid = 1000 + rid
            cats.append({"id": cid, "name": name, "supercategory": "rule_violation"})
            name_to_id[name] = cid
    return cats, name_to_id


class CocoStreamingEmitter:
    """Streaming COCO writer. Image bytes are written and released per record."""

    def __init__(
        self,
        *,
        out_dir: str,
        sub: str,
        condition: str,
        class_names: list[str],
        has_rules: bool = False,
        info_extra: dict | None = None,
    ) -> None:
        self.out_dir = out_dir
        self.images_dir = os.path.join(out_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

        self.sub = sub
        self.condition = condition
        self.info_extra = info_extra or {}

        rule_ids = [1, 2, 3, 4] if has_rules else None
        self.categories, self.name_to_id = _category_table(class_names, rule_ids)

        self.coco_images: list[dict] = []
        self.coco_annots: list[dict] = []
        self._next_img_id = 1
        self._next_ann_id = 1
        self.n_image_bytes_written = 0
        self.image_sha_count = 0

    def add_record(self, rec: dict) -> None:
        img_bytes = rec["image"]
        if not isinstance(img_bytes, (bytes, bytearray)):
            raise TypeError("CocoStreamingEmitter expects rec['image'] as raw bytes")
        if img_bytes[:3] != b"\xff\xd8\xff":
            raise ValueError(f"image {rec['image_id']} is not JPEG")
        w, h = image_size(img_bytes)

        file_name = f"{rec['image_id']}.jpg"
        img_path = os.path.join(self.images_dir, file_name)
        atomic_write(img_path, bytes(img_bytes))
        self.image_sha_count += 1
        self.n_image_bytes_written += len(img_bytes)

        consynth_extra: dict = {
            "consynth_image_id": rec["image_id"],
            "source_id": rec.get("source_id"),
            "source_dataset": rec["source_dataset"],
            "condition": rec["condition"],
            "condition_labels": rec["condition_labels"],
            "pipeline": rec["pipeline"],
            "quality_scores": rec.get("quality_scores"),
            "quality_alert": rec.get("quality_alert"),
        }
        if rec.get("image_attributes") is not None:
            consynth_extra["cs10k_image_attributes"] = rec["image_attributes"]
        if "captions" in rec:
            consynth_extra["ktsh_captions"] = rec["captions"]

        img_meta = {
            "id": self._next_img_id,
            "file_name": file_name,
            "width": w,
            "height": h,
            "consynth_extra": consynth_extra,
        }
        self.coco_images.append(img_meta)

        # Detection annotations
        for obj in rec.get("objects", []) or []:
            x1, y1, x2, y2 = obj["bbox"]
            px, py = x1 * w, y1 * h
            pw, ph = (x2 - x1) * w, (y2 - y1) * h
            cat_id = self.name_to_id.get(obj["class_name"])
            if cat_id is None:
                raise ValueError(f"class {obj['class_name']!r} not in class_names")
            self.coco_annots.append({
                "id": self._next_ann_id,
                "image_id": self._next_img_id,
                "category_id": cat_id,
                "bbox": [px, py, pw, ph],
                "area": pw * ph,
                "iscrowd": 0,
                "consynth_extra": {"kind": "object"},
            })
            self._next_ann_id += 1

        # Rule violations (cs10k only)
        no_bbox_rules: list[dict] = []
        for rv in rec.get("rule_violations", []) or []:
            if rv.get("bbox") is None:
                no_bbox_rules.append({"rule_id": rv["rule_id"], "reason": rv.get("reason", "")})
                continue
            x1, y1, x2, y2 = rv["bbox"]
            px, py = x1 * w, y1 * h
            pw, ph = (x2 - x1) * w, (y2 - y1) * h
            cat_name = f"rule_{rv['rule_id']}_violation"
            cat_id = self.name_to_id[cat_name]
            self.coco_annots.append({
                "id": self._next_ann_id,
                "image_id": self._next_img_id,
                "category_id": cat_id,
                "bbox": [px, py, pw, ph],
                "area": pw * ph,
                "iscrowd": 0,
                "consynth_extra": {"kind": "rule_violation", "reason": rv.get("reason", "")},
            })
            self._next_ann_id += 1
        if no_bbox_rules:
            consynth_extra["rule_violations_no_bbox"] = no_bbox_rules

        self._next_img_id += 1

    def finalize(self) -> dict:
        coco = {
            "info": {
                "description": f"ConSynth-X release v1 — sub={self.sub}, condition={self.condition}",
                "version": "1.0",
                **self.info_extra,
            },
            "licenses": [{"id": 1, "name": "CC BY-NC 4.0",
                           "url": "https://creativecommons.org/licenses/by-nc/4.0/"}],
            "images": self.coco_images,
            "annotations": self.coco_annots,
            "categories": self.categories,
        }
        json_path = os.path.join(self.out_dir, "annotations.json")
        atomic_write(json_path, json.dumps(coco, indent=2, default=str).encode("utf-8"))
        return {
            "n_images": len(self.coco_images),
            "n_annotations": len(self.coco_annots),
            "n_categories": len(self.categories),
            "json_path": json_path,
            "image_sha_count": self.image_sha_count,
            "image_bytes_written": self.n_image_bytes_written,
        }


def emit_coco(
    *,
    out_dir: str,
    sub: str,
    condition: str,
    records: Iterable[dict],
    class_names: list[str],
    has_rules: bool = False,
    info_extra: dict | None = None,
) -> dict:
    """Convenience: in-memory wrapper around CocoStreamingEmitter for small shards."""
    emit = CocoStreamingEmitter(
        out_dir=out_dir, sub=sub, condition=condition,
        class_names=class_names, has_rules=has_rules, info_extra=info_extra,
    )
    for rec in records:
        emit.add_record(rec)
    return emit.finalize()
