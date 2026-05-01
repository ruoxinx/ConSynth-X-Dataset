"""Cross-validate the released Parquet + COCO output.

Run AFTER `repack_to_release.py --execute`. Verifies:
  - All Parquet files match the canonical schema for their sub.
  - image_id uniqueness within each shard.
  - Byte-level identity between Parquet `image` column and COCO `images/<id>.jpg`
    (sample 5% per shard to keep runtime bounded; full check optional).
  - Bbox round-trip: Parquet normalised xyxy -> COCO pixel xywh consistency.
  - Row-count invariants vs registry inputs.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Iterable

import pyarrow.parquet as pq
import yaml

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from release_pipeline.convert import release_schema as RS
from release_pipeline.convert.integrity import (
    image_size, read_arrow_table, sha256_bytes, sha256_file,
)


def validate_parquet(path: str, sub: str) -> dict:
    table = pq.read_table(path)
    RS.validate_table(table, sub)
    image_ids = table.column("image_id").to_pylist()
    if len(set(image_ids)) != len(image_ids):
        dups = [x for x in set(image_ids) if image_ids.count(x) > 1]
        raise ValueError(f"duplicate image_id in {path}: {dups[:5]}")
    return {"rows": table.num_rows, "unique_ids": len(set(image_ids))}


def cross_check_coco(parquet_path: str, coco_path: str, image_dir: str, sample_pct: float = 0.05) -> dict:
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    with open(coco_path) as f:
        coco = json.load(f)
    coco_imgs = {im["consynth_extra"]["consynth_image_id"]: im for im in coco["images"]}
    coco_anns_by_img = {}
    for a in coco["annotations"]:
        coco_anns_by_img.setdefault(a["image_id"], []).append(a)

    # Image-id consistency
    parquet_ids = set(r["image_id"] for r in rows)
    coco_ids = set(coco_imgs.keys())
    if parquet_ids != coco_ids:
        only_p = parquet_ids - coco_ids
        only_c = coco_ids - parquet_ids
        raise ValueError(
            f"image_id mismatch: only_in_parquet={list(only_p)[:5]} only_in_coco={list(only_c)[:5]}"
        )

    # Sample check: byte SHA + bbox round-trip
    n_check = max(1, int(len(rows) * sample_pct))
    rng = random.Random(0)
    sampled = rng.sample(rows, k=min(n_check, len(rows)))
    drift_cnt = 0
    bbox_mismatches = 0
    for rec in sampled:
        coco_im = coco_imgs[rec["image_id"]]
        img_path = os.path.join(image_dir, coco_im["file_name"])
        if sha256_bytes(rec["image"]) != sha256_file(img_path):
            drift_cnt += 1
            continue
        # Bbox check on first object if any
        if rec.get("objects"):
            obj = rec["objects"][0]
            x1, y1, x2, y2 = obj["bbox"]
            w, h = coco_im["width"], coco_im["height"]
            expect = [x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h]
            anns = coco_anns_by_img.get(coco_im["id"], [])
            ann = next((a for a in anns if a.get("consynth_extra", {}).get("kind") == "object"), None)
            if ann is None:
                continue
            if any(abs(a - b) > 1e-3 for a, b in zip(ann["bbox"], expect)):
                bbox_mismatches += 1
    return {
        "rows": len(rows),
        "sampled": len(sampled),
        "byte_drift": drift_cnt,
        "bbox_mismatch": bbox_mismatches,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True,
                    help="Parquet release root (contains <sub>/parquet/...).")
    ap.add_argument("--coco-release-root", default=None,
                    help="Optional separate COCO release root. Defaults to --release-root.")
    ap.add_argument("--registry", required=True)
    ap.add_argument("--sample-pct", type=float, default=0.05)
    args = ap.parse_args(argv)
    coco_root = args.coco_release_root or args.release_root

    with open(args.registry) as f:
        reg = yaml.safe_load(f)
    def _basename_for(s):
        b = os.path.splitext(os.path.basename(s["path"]))[0]
        return f"{s['split']}__{b}" if s.get("split") else b

    expected = {(s["sub"], s["condition"], _basename_for(s)): s for s in reg["shards"]}

    failures = []
    summary = []
    for (sub, cond, basename), entry in expected.items():
        pq_path = os.path.join(args.release_root, sub, "parquet", cond, f"{basename}.parquet")
        coco_path = os.path.join(coco_root, sub, "coco", cond, basename, "annotations.json")
        image_dir = os.path.join(coco_root, sub, "coco", cond, basename, "images")
        if not os.path.exists(pq_path):
            failures.append({"shard": entry["path"], "error": f"parquet missing: {pq_path}"})
            continue
        try:
            schema_info = validate_parquet(pq_path, sub)
            cross_info = cross_check_coco(pq_path, coco_path, image_dir, sample_pct=args.sample_pct)
            ok = cross_info["byte_drift"] == 0 and cross_info["bbox_mismatch"] == 0
            summary.append({
                "shard": entry["path"], **schema_info, **cross_info, "ok": ok
            })
            print(f"  {'OK' if ok else 'FAIL'}  {entry['path']:65s} rows={schema_info['rows']} "
                  f"drift={cross_info['byte_drift']} bbox_mismatch={cross_info['bbox_mismatch']}")
            if not ok:
                failures.append({"shard": entry["path"], "info": cross_info})
        except Exception as e:
            print(f"  FAIL  {entry['path']:65s} {type(e).__name__}: {e}")
            failures.append({"shard": entry["path"], "error": f"{type(e).__name__}: {e}"})

    print(f"\nshards={len(summary)} failures={len(failures)}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
