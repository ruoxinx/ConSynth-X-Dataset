"""Repack ConSynth-X arrow shards into release v1 format.

By default runs in --dry-run: reads everything, validates everything, writes
NOTHING. Caller reviews the audit log, then re-runs with --execute.

Usage:
    python -m release_pipeline.scripts.repack_to_release \
        --registry release_pipeline/metadata/condition_registry.yaml \
        --output-root /fs/scratch/PGS0407/binben14/ConSynth-X-release-v1 \
        --workspace-root /users/PGS0407/binben14/VietHuy/ConSynth-X \
        [--shard PATH ...] [--execute]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from release_pipeline.convert import release_schema as RS
from release_pipeline.convert.cs10k_to_objects import (
    CS10K_DETECTION_CLASSES, from_format_a, from_format_b,
)
from release_pipeline.convert.bbox_join import build_source_index, attach_objects
from release_pipeline.convert.coco_emitter import emit_coco, CocoStreamingEmitter
from release_pipeline.convert.integrity import (
    apply_exif_rotation, atomic_write, coerce_image_bytes, exif_orientation,
    image_size, read_arrow_table, sha256_bytes, write_audit_json,
)
from release_pipeline.convert.voc_to_objects import (
    normalise_objects, parse_voc_xml,
)


# ---------- format-specific row builders ----------

def _empty_quality() -> dict:
    return {"dino_sim": None, "ssim": None, "clip_sim": None, "lpips": None}


def _maybe_attach_dino(q: dict, dino_scores: dict | None, image_id: str) -> dict:
    if dino_scores is None:
        return q
    q = dict(q)
    q["dino_sim"] = dino_scores.get(image_id)
    return q


def _load_quality_csv(path: str) -> dict[str, dict]:
    """Load DINO/SSIM scores CSV. Expects header `image_id,ssim,dino_sim,dino_dist`.
    Returns {image_id: {ssim: float|None, dino_sim: float|None}}.
    Extra columns are tolerated; missing columns leave the field None.
    """
    import csv as _csv
    out: dict[str, dict] = {}
    with open(path, newline="") as f:
        r = _csv.DictReader(f)
        for row in r:
            iid = row.get("image_id")
            if not iid:
                continue
            entry: dict = {}
            for col, key in [("dino_sim", "dino_sim"), ("ssim", "ssim"),
                              ("clip_sim", "clip_sim"), ("lpips", "lpips")]:
                if col in row and row[col] not in (None, "", "nan", "NaN"):
                    try:
                        entry[key] = float(row[col])
                    except ValueError:
                        pass
            out[iid] = entry
    return out


def _voc_class_table_from_shard(rows: list[dict]) -> dict[str, int]:
    """Scan all VOC annotations in a shard, build {class_name: int_id} stable order."""
    names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        ann = row.get("annotation")
        if not ann:
            continue
        parsed = parse_voc_xml(ann)
        for o in parsed.objects:
            if o.class_name not in seen:
                seen.add(o.class_name)
                names.append(o.class_name)
    return {n: i for i, n in enumerate(sorted(names))}, sorted(names)


def _normalize_image_for_voc(row: dict, ann_text: str | None) -> tuple[bytes, int, int, list[dict]]:
    """Read row image, decide if EXIF re-encode is needed (so VOC <size> matches),
    and return (bytes, w, h, warnings).

    Apply re-encode ONLY when raw size != VOC size AND raw size == swapped VOC size,
    confirming an EXIF rotation. Other size disagreements remain hard errors.
    """
    b = coerce_image_bytes(row["image"])
    w, h = image_size(b)
    warnings: list[dict] = []
    if not ann_text or not ann_text.strip():
        return b, w, h, warnings
    try:
        pre = parse_voc_xml(ann_text)
    except Exception:
        return b, w, h, warnings
    if pre.raw_size_in_xml is None:
        return b, w, h, warnings
    voc_w, voc_h = pre.raw_size_in_xml
    if (voc_w, voc_h) == (w, h):
        return b, w, h, warnings
    # Try EXIF-rotated dimensions
    if (voc_w, voc_h) == (h, w):
        orient = exif_orientation(b)
        if orient not in (1, 0):
            old_sha = sha256_bytes(b)
            b = apply_exif_rotation(b)
            w, h = image_size(b)
            warnings.append({
                "image_id": row["image_id"],
                "kind": "exif_reencoded",
                "orientation": orient,
                "old_sha256": old_sha,
                "new_sha256": sha256_bytes(b),
                "new_wh": (w, h),
            })
            return b, w, h, warnings
    # If we get here, size disagreement is unresolved — let the parser raise.
    return b, w, h, warnings


def _build_records_voc_xml(arrow_path: str, entry: dict) -> tuple[list[dict], list[str], list[dict]]:
    """Returns (records, class_names, warnings).

    Handles EXIF orientation: if VOC <size> matches the EXIF-rotated dimensions
    (not the raw dimensions), re-encode the image after applying rotation so the
    stored bytes and bbox normalisation are consistent.
    """
    table = read_arrow_table(arrow_path)
    rows = table.to_pylist()
    class_to_id, class_names = _voc_class_table_from_shard(rows)
    records: list[dict] = []
    warnings: list[dict] = []
    for row in rows:
        ann = row.get("annotation") or ""
        b, w, h, exif_warns = _normalize_image_for_voc(row, ann)
        warnings.extend(exif_warns)
        if not ann.strip():
            objects = []
        else:
            parsed = parse_voc_xml(ann)
            try:
                objects, warns = normalise_objects(parsed, w, h, class_to_id)
            except ValueError as e:
                # Source data inconsistency we can't resolve (e.g. VOC <size>
                # disagrees with image WH and no EXIF rotation explains it).
                # Skip the row with a logged warning rather than crash the shard.
                warnings.append({
                    "image_id": row["image_id"],
                    "kind": "voc_skipped_row",
                    "reason": str(e),
                })
                continue
            for x in warns:
                warnings.append({"image_id": row["image_id"], **x})
        records.append({
            "image": b,
            "image_id": row["image_id"],
            "source_id": row.get("ref_id") or row["image_id"],
            "objects": objects,
        })
    return records, class_names, warnings


def _build_records_cs10k_columns(arrow_path: str, entry: dict) -> tuple[list[dict], list[str], list[dict]]:
    table = read_arrow_table(arrow_path)
    rows = table.to_pylist()
    records: list[dict] = []
    warnings: list[dict] = []
    for row in rows:
        b = coerce_image_bytes(row["image"])
        canon = from_format_a(row)
        for w in canon.warnings:
            warnings.append({"image_id": row["image_id"], **w})
        records.append({
            "image": b,
            "image_id": row["image_id"],
            "source_id": row.get("ref_id") or row["image_id"],
            "objects": canon.objects,
            "rule_violations": canon.rule_violations,
            "image_attributes": canon.image_attributes,
        })
    return records, list(CS10K_DETECTION_CLASSES), warnings


def _build_records_cs10k_json(arrow_path: str, entry: dict) -> tuple[list[dict], list[str], list[dict]]:
    table = read_arrow_table(arrow_path)
    rows = table.to_pylist()
    records: list[dict] = []
    warnings: list[dict] = []
    for row in rows:
        b = coerce_image_bytes(row["image"])
        canon = from_format_b(row.get("annotation") or "", image_caption=None)
        for w in canon.warnings:
            warnings.append({"image_id": row["image_id"], **w})
        records.append({
            "image": b,
            "image_id": row["image_id"],
            "source_id": row.get("ref_id") or row["image_id"],
            "objects": canon.objects,
            "rule_violations": canon.rule_violations,
            "image_attributes": canon.image_attributes,
        })
    return records, list(CS10K_DETECTION_CLASSES), warnings


def _build_records_join_from_source(arrow_path: str, entry: dict, workspace_root: str) -> tuple[list[dict], list[str], list[dict]]:
    """For variant-9 soda_voc shards: read synthetic, build source index, join.

    Both source and synthetic images go through the same EXIF-aware loader so
    that size comparisons happen in a canonical (rotated-when-needed) frame.
    """
    join_src = os.path.join(workspace_root, entry["join_source"])
    if entry["join_source_format"] != "voc_xml":
        raise ValueError(f"join_source_format={entry['join_source_format']!r} unsupported")

    src_table = read_arrow_table(join_src)
    src_rows = src_table.to_pylist()
    class_to_id, class_names = _voc_class_table_from_shard(src_rows)

    join_warns: list[dict] = []

    def parser(row: dict, w: int, h: int) -> list[dict]:
        ann = row.get("annotation") or ""
        if not ann.strip():
            return []
        parsed = parse_voc_xml(ann)
        try:
            objs, _ = normalise_objects(parsed, w, h, class_to_id)
        except ValueError as e:
            # Source row has unresolvable VOC<->image size disagreement.
            # Treat as having no objects; downstream synthetic row will inherit
            # an empty object list with a warning logged below.
            join_warns.append({
                "image_id": row.get("image_id"),
                "kind": "join_source_voc_skipped",
                "reason": str(e),
            })
            return []
        return objs

    # Source loader: use VOC <size> to determine if EXIF rotation is needed.
    def source_loader(row):
        return _normalize_image_for_voc(row, row.get("annotation") or "")

    # Synthetic loader: there is no annotation here. We try to align to source's
    # canonical size by checking the source index later (see align_synthetic).
    src_index = build_source_index(join_src, parser, image_loader=source_loader)

    def synthetic_loader(row):
        b = coerce_image_bytes(row["image"])
        w, h = image_size(b)
        warnings: list[dict] = []
        sid = row.get("ref_id") or row["image_id"]
        target = src_index.get(sid, {}).get("size")
        if target and (w, h) != target and (h, w) == target:
            orient = exif_orientation(b)
            if orient not in (1, 0):
                old_sha = sha256_bytes(b)
                b = apply_exif_rotation(b)
                w, h = image_size(b)
                warnings.append({
                    "image_id": row["image_id"],
                    "kind": "exif_reencoded_synthetic",
                    "orientation": orient,
                    "old_sha256": old_sha,
                    "new_sha256": sha256_bytes(b),
                    "new_wh": (w, h),
                })
        return b, w, h, warnings

    syn_table = read_arrow_table(arrow_path)
    syn_rows = syn_table.to_pylist()
    objects_per_row, image_bytes_per_row, warns = attach_objects(
        syn_rows, src_index, image_loader=synthetic_loader,
    )
    records: list[dict] = []
    for row, objs, b in zip(syn_rows, objects_per_row, image_bytes_per_row):
        records.append({
            "image": b,
            "image_id": row["image_id"],
            "source_id": row.get("ref_id") or row["image_id"],
            "objects": objs,
        })
    return records, class_names, list(warns) + join_warns


def _build_caption_join_map(workspace_root: str, sources: list[str]) -> dict[str, list[str]]:
    """Build {image_id: captions} from one or more arrow shards that DO carry
    `captions` field. Only adds a new entry the first time an image_id is seen
    so caller can pass sources in priority order. Source rows with empty/null
    captions are skipped (we never overwrite a populated entry).
    """
    cmap: dict[str, list[str]] = {}
    for rel in sources:
        p = os.path.join(workspace_root, rel)
        with pa.OSFile(p, "rb") as f:
            try:
                r = pa.ipc.open_stream(f)
            except pa.ArrowInvalid:
                f.seek(0)
                r = pa.ipc.open_file(f)
            t = r.read_all()
        if "captions" not in t.column_names:
            continue
        ids = t.column("image_id").to_pylist()
        caps = t.column("captions").to_pylist()
        for iid, c in zip(ids, caps):
            if c and iid not in cmap:
                cmap[iid] = [str(x) for x in c]
    return cmap


def _build_records_ktsh(arrow_path: str, entry: dict) -> tuple[list[dict], list[str], list[dict]]:
    """ktsh caption-only builder. If `entry.caption_join_sources` is provided,
    captions will be filled from those source arrows for rows whose own
    `captions` field is empty/missing. Existing populated captions are NEVER
    overwritten."""
    table = read_arrow_table(arrow_path)
    rows = table.to_pylist()

    cmap: dict[str, list[str]] = {}
    if entry.get("caption_join_sources"):
        ws = os.environ.get("CONSYNTH_WORKSPACE_ROOT") or "/users/PGS0407/binben14/VietHuy/ConSynth-X"
        cmap = _build_caption_join_map(ws, entry["caption_join_sources"])

    n_filled = 0
    n_kept_existing = 0
    n_still_empty = 0
    warnings: list[dict] = []
    records: list[dict] = []
    for row in rows:
        b = coerce_image_bytes(row["image"])
        captions = row.get("captions") or []
        if not isinstance(captions, list):
            captions = [str(captions)]
        captions = [str(c) for c in captions]
        if captions:
            n_kept_existing += 1
        elif cmap:
            joined = cmap.get(row["image_id"])
            if joined:
                captions = joined
                n_filled += 1
            else:
                n_still_empty += 1
        else:
            n_still_empty += 1
        records.append({
            "image": b,
            "image_id": row["image_id"],
            "source_id": row.get("ref_id") or row["image_id"],
            "objects": [],
            "captions": captions,
        })
    if cmap:
        warnings.append({
            "kind": "caption_join_summary",
            "kept_existing": n_kept_existing,
            "filled_from_join": n_filled,
            "still_empty": n_still_empty,
            "join_sources": entry.get("caption_join_sources"),
        })
    return records, [], warnings


def _drop_empty_or_non_jpeg(records: list[dict]) -> list[dict]:
    """In-place filter: remove records whose image bytes are empty OR not JPEG.
    Returns warnings about every drop. Mutates `records` (slice-replaces).
    """
    warns: list[dict] = []
    keep: list[dict] = []
    for rec in records:
        b = rec.get("image", b"")
        if not isinstance(b, (bytes, bytearray)) or len(b) < 4:
            warns.append({"image_id": rec.get("image_id"), "kind": "image_empty"})
            continue
        if bytes(b[:3]) != b"\xff\xd8\xff":
            warns.append({
                "image_id": rec.get("image_id"),
                "kind": "image_not_jpeg",
                "prefix_hex": bytes(b[:4]).hex(),
            })
            continue
        keep.append(rec)
    records[:] = keep
    return warns


# ---------- per-shard orchestration ----------

FORMAT_DISPATCH = {
    "voc_xml": _build_records_voc_xml,
    "cs10k_columns": _build_records_cs10k_columns,
    "cs10k_json": _build_records_cs10k_json,
    "ktsh_caption_only": _build_records_ktsh,
}


CHUNK_ROWS = 1000  # streaming batch size — caps peak memory at ~1k images


def process_shard(entry: dict, pipelines: dict, workspace_root: str, output_root: str,
                  *, dry_run: bool, coco_output_root: str | None = None) -> dict:
    """Streaming repack: builders produce all records in memory (acceptable —
    builders only retain raw bytes per row, not duplicated copies), then
    Parquet + COCO output is streamed in chunks of CHUNK_ROWS to keep peak RAM
    bounded by the chunk size rather than the whole shard."""
    import gc
    arrow_path = os.path.join(workspace_root, entry["path"])
    sub = entry["sub"]
    fmt = entry["format"]

    t0 = time.time()
    if fmt == "join_from_source":
        records, class_names, warnings = _build_records_join_from_source(arrow_path, entry, workspace_root)
    else:
        builder = FORMAT_DISPATCH[fmt]
        records, class_names, warnings = builder(arrow_path, entry)

    # Drop rows with empty/non-JPEG image bytes — these are upstream data defects
    # we cannot ship. Logged in audit; downstream consumer should not see them.
    drop_warns = _drop_empty_or_non_jpeg(records)
    warnings.extend(drop_warns)

    pipeline_struct = pipelines[entry["pipeline"]]
    quality_table = None
    dino_csv_status = "absent"
    if entry.get("dino_scores_path"):
        csv_full = os.path.join(workspace_root, entry["dino_scores_path"])
        if os.path.exists(csv_full):
            quality_table = _load_quality_csv(csv_full)
            dino_csv_status = "loaded"
        else:
            # User policy: treat missing DINO CSV as "scores not yet computed".
            # quality_scores stays null, quality_alert stays null. Recorded in audit.
            warnings.append({
                "kind": "dino_csv_missing",
                "path": entry["dino_scores_path"],
            })
            dino_csv_status = "missing"

    image_ids: list[str] = []
    n_dino_matched = 0
    n_dino_missing = 0
    for rec in records:
        image_ids.append(rec["image_id"])
        rec["source_dataset"] = sub
        rec["condition"] = entry["condition"]
        rec["condition_labels"] = list(entry["condition_labels"])
        rec["pipeline"] = pipeline_struct
        q = _empty_quality()
        if quality_table is not None:
            row_q = quality_table.get(rec["image_id"])
            if row_q is not None:
                n_dino_matched += 1
                q = dict(q, **row_q)
            else:
                n_dino_missing += 1
        rec["quality_scores"] = q if any(v is not None for v in q.values()) else None
        rec["quality_alert"] = RS.compute_quality_alert(q.get("dino_sim"))

    if len(set(image_ids)) != len(image_ids):
        dups = [x for x in set(image_ids) if image_ids.count(x) > 1]
        raise ValueError(f"duplicate image_ids in {arrow_path}: {dups[:5]}")

    n_objects = sum(len(r.get("objects") or []) for r in records)
    n_rule_violations = sum(len(r.get("rule_violations") or []) for r in records)
    n_alerts_true = sum(1 for r in records if r["quality_alert"] is True)
    n_alerts_null = sum(1 for r in records if r["quality_alert"] is None)

    audit = {
        "shard": entry["path"],
        "sub": sub,
        "format": fmt,
        "condition": entry["condition"],
        "condition_labels": entry["condition_labels"],
        "rows_in": len(records),
        "rows_out": len(records),
        "n_objects": n_objects,
        "n_rule_violations": n_rule_violations,
        "n_quality_alert_true": n_alerts_true,
        "n_quality_alert_null": n_alerts_null,
        "n_dino_matched": n_dino_matched,
        "n_dino_missing": n_dino_missing,
        "dino_scores_path": entry.get("dino_scores_path"),
        "n_warnings": len(warnings),
        "warnings_sample": warnings[:5],
        "class_names": class_names,
        "elapsed_sec": round(time.time() - t0, 2),
    }

    if dry_run:
        # Build and validate the schema with one tiny chunk just to catch type errors.
        if records:
            sample = records[: min(8, len(records))]
            tiny = _records_to_table(sample, sub, has_rules=(sub == "cs10k"))
            RS.validate_table(tiny, sub)
        audit["mode"] = "dry-run"
        return audit

    # ---- STREAMING WRITE PHASE ----
    # Parquet and COCO outputs may live in separate release roots so each can be
    # uploaded as its own HuggingFace dataset / archive.
    pq_root = os.path.join(output_root, sub)
    parquet_dir = os.path.join(pq_root, "parquet", entry["condition"])
    audit_dir = os.path.join(pq_root, "_audit")
    os.makedirs(parquet_dir, exist_ok=True)
    os.makedirs(audit_dir, exist_ok=True)

    coco_root = coco_output_root if coco_output_root else output_root
    coco_dir = os.path.join(coco_root, sub, "coco", entry["condition"])
    coco_audit_dir = os.path.join(coco_root, sub, "_audit")
    os.makedirs(coco_dir, exist_ok=True)
    os.makedirs(coco_audit_dir, exist_ok=True)

    raw_basename = os.path.splitext(os.path.basename(entry["path"]))[0]
    # Prefix with split when present so train/test of the same condition don't
    # collide on identical file basenames (e.g., cs10k rain_heavy.arrow exists
    # in both train/ and test/ folders).
    if entry.get("split"):
        shard_basename = f"{entry['split']}__{raw_basename}"
    else:
        shard_basename = raw_basename
    parquet_path = os.path.join(parquet_dir, f"{shard_basename}.parquet")
    parquet_tmp = parquet_path + ".tmp"

    coco_emit = CocoStreamingEmitter(
        out_dir=os.path.join(coco_dir, shard_basename),
        sub=sub,
        condition=entry["condition"],
        class_names=class_names,
        has_rules=(sub == "cs10k"),
    )

    pq_writer = None
    try:
        sha_in: list[str] = []
        for start in range(0, len(records), CHUNK_ROWS):
            chunk = records[start:start + CHUNK_ROWS]
            chunk_table = _records_to_table(chunk, sub, has_rules=(sub == "cs10k"))
            if pq_writer is None:
                RS.validate_table(chunk_table, sub)
                pq_writer = pq.ParquetWriter(parquet_tmp, chunk_table.schema, compression="zstd")
            pq_writer.write_table(chunk_table)
            for rec in chunk:
                sha_in.append(sha256_bytes(rec["image"]))
                coco_emit.add_record(rec)
            # Free chunk's image bytes from records list to keep memory flat
            for i in range(start, start + len(chunk)):
                records[i]["image"] = b""
            del chunk, chunk_table
            gc.collect()
    finally:
        if pq_writer is not None:
            pq_writer.close()

    os.replace(parquet_tmp, parquet_path)
    coco_stats = coco_emit.finalize()

    # Round-trip verification: stream Parquet back, compare per-row SHA against sha_in.
    rt = pq.read_table(parquet_path, columns=["image_id", "image"])
    rt_ids = rt.column("image_id").to_pylist()
    drift = []
    for i, (iid, img) in enumerate(zip(rt_ids, rt.column("image").to_pylist())):
        if sha256_bytes(img) != sha_in[i]:
            drift.append(iid)
            if len(drift) >= 5:
                break
    if drift:
        raise RuntimeError(f"image bytes drift after Parquet round-trip: {drift}")
    if len(rt_ids) != len(records):
        raise RuntimeError(f"parquet read-back row count mismatch: {len(rt_ids)} vs {len(records)}")

    audit["mode"] = "execute"
    audit["parquet_path"] = parquet_path
    audit["coco_stats"] = coco_stats
    audit["coco_image_sha_count"] = coco_stats.get("image_sha_count", 0)

    audit_path = os.path.join(audit_dir, f"{shard_basename}.json")
    write_audit_json(audit_path, audit)
    if coco_output_root and coco_output_root != output_root:
        # Mirror audit into coco release root for traceability
        write_audit_json(os.path.join(coco_audit_dir, f"{shard_basename}.json"), audit)
    return audit


def _records_to_table(records: list[dict], sub: str, *, has_rules: bool) -> pa.Table:
    """Build a pa.Table that exactly matches SCHEMAS[sub]."""
    schema = RS.SCHEMAS[sub]
    cols: dict[str, list] = {f.name: [] for f in schema}
    for r in records:
        for f in schema:
            name = f.name
            if name == "image":
                cols[name].append(r["image"])
            elif name == "image_id":
                cols[name].append(r["image_id"])
            elif name == "source_id":
                cols[name].append(r.get("source_id"))
            elif name == "source_dataset":
                cols[name].append(r["source_dataset"])
            elif name == "condition":
                cols[name].append(r["condition"])
            elif name == "condition_labels":
                cols[name].append(list(r["condition_labels"]))
            elif name == "objects":
                cols[name].append(list(r.get("objects") or []))
            elif name == "pipeline":
                cols[name].append(dict(r["pipeline"]))
            elif name == "quality_scores":
                cols[name].append(r.get("quality_scores"))
            elif name == "quality_alert":
                cols[name].append(r.get("quality_alert"))
            elif name == "image_attributes":
                cols[name].append(r.get("image_attributes"))
            elif name == "rule_violations":
                cols[name].append(list(r.get("rule_violations") or []))
            elif name == "captions":
                cols[name].append(list(r.get("captions") or []))
            else:
                raise KeyError(f"unhandled schema field {name!r}")
    arrays = []
    for f in schema:
        arrays.append(pa.array(cols[f.name], type=f.type))
    return pa.Table.from_arrays(arrays, schema=schema)


# ---------- CLI ----------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--workspace-root", required=True)
    ap.add_argument("--output-root", required=True,
                    help="Release root for Parquet output (and audit logs).")
    ap.add_argument("--coco-output-root", default=None,
                    help="Optional separate release root for COCO output. If not "
                         "set, COCO is written under --output-root.")
    ap.add_argument("--shard", action="append", default=None,
                    help="Process only these shard paths (repeat). Default: all shards in registry.")
    ap.add_argument("--execute", action="store_true",
                    help="Actually write outputs. Default is dry-run.")
    ap.add_argument("--stop-on-error", action="store_true", default=True)
    args = ap.parse_args(argv)

    with open(args.registry) as f:
        reg = yaml.safe_load(f)
    pipelines = reg["pipelines"]
    shards = reg["shards"]
    if args.shard:
        wanted = set(args.shard)
        shards = [s for s in shards if s["path"] in wanted]
        missing = wanted - {s["path"] for s in shards}
        if missing:
            raise SystemExit(f"--shard not in registry: {missing}")

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"[repack] mode={mode} shards={len(shards)} output_root={args.output_root}")

    summary = []
    failed = []
    for entry in shards:
        try:
            audit = process_shard(entry, pipelines, args.workspace_root, args.output_root,
                                   dry_run=not args.execute,
                                   coco_output_root=args.coco_output_root)
            print(f"  OK  {entry['path']:60s} rows={audit['rows_in']:5d} "
                  f"objs={audit['n_objects']:5d} rules={audit['n_rule_violations']:5d} "
                  f"warn={audit['n_warnings']}")
            summary.append(audit)
        except Exception as e:
            print(f"  FAIL  {entry['path']:60s}  {type(e).__name__}: {e}")
            failed.append({"path": entry["path"], "error": f"{type(e).__name__}: {e}"})
            if args.stop_on_error:
                break

    # Write run-level summary
    run_audit = {
        "mode": mode,
        "registry": args.registry,
        "workspace_root": args.workspace_root,
        "output_root": args.output_root,
        "shards_attempted": len(shards),
        "shards_ok": len(summary),
        "shards_failed": len(failed),
        "failures": failed,
        "shards": summary,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if args.execute:
        os.makedirs(args.output_root, exist_ok=True)
        write_audit_json(os.path.join(args.output_root, f"_run_audit_{int(time.time())}.json"), run_audit)
    else:
        print(json.dumps(run_audit, indent=2, default=str)[:4000])

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
