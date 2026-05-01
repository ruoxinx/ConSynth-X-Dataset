"""Canonical Parquet schemas for ConSynth-X release v1.

Three sub-datasets, three schemas. Do NOT collapse into one — class taxonomies
diverge across cs10k / soda_voc / soda_ktsh and SODA-KTSH carries no bbox.

bbox convention in Parquet: [x1, y1, x2, y2] in normalised coordinates [0,1].
COCO-JSON output uses pixel-absolute [x, y, w, h] — see convert/coco_emitter.py.
"""
import pyarrow as pa

OBJECT_STRUCT = pa.struct([
    pa.field("class_id", pa.int32()),
    pa.field("class_name", pa.string()),
    # Variable-size list (always length 4) for runtime nullability support.
    # Validators enforce len==4 in non-null entries.
    pa.field("bbox", pa.list_(pa.float64())),
])

PIPELINE_STRUCT = pa.struct([
    pa.field("method", pa.string()),
    pa.field("checkpoint", pa.string()),
    pa.field("prompt_template", pa.string()),
    pa.field("params", pa.string()),
])

QUALITY_STRUCT = pa.struct([
    pa.field("dino_sim", pa.float32()),
    pa.field("ssim", pa.float32()),
    pa.field("clip_sim", pa.float32()),
    pa.field("lpips", pa.float32()),
])

_BASE = [
    pa.field("image", pa.binary(), nullable=False),
    pa.field("image_id", pa.string(), nullable=False),
    pa.field("source_id", pa.string(), nullable=True),
    pa.field("source_dataset", pa.string(), nullable=False),
    pa.field("condition", pa.string(), nullable=False),
    pa.field("condition_labels", pa.list_(pa.string()), nullable=False),
    pa.field("objects", pa.list_(OBJECT_STRUCT), nullable=False),
    pa.field("pipeline", PIPELINE_STRUCT, nullable=False),
    pa.field("quality_scores", QUALITY_STRUCT, nullable=True),
    pa.field("quality_alert", pa.bool_(), nullable=True),
]

CS10K_IMAGE_ATTRS = pa.struct([
    pa.field("caption", pa.string()),
    pa.field("illumination", pa.string()),
    pa.field("camera_distance", pa.string()),
    pa.field("view", pa.string()),
    pa.field("quality_of_info", pa.string()),
])

CS10K_RULE_VIOLATION = pa.struct([
    pa.field("rule_id", pa.int32()),
    # bbox nullable: a rule may fire with reason text but no bbox in source data.
    # Variable-size list to allow null entries (PyArrow fixed-size list can't hold nulls).
    pa.field("bbox", pa.list_(pa.float64()), nullable=True),
    pa.field("reason", pa.string()),
])

SCHEMA_CS10K = pa.schema(_BASE + [
    pa.field("image_attributes", CS10K_IMAGE_ATTRS, nullable=True),
    pa.field("rule_violations", pa.list_(CS10K_RULE_VIOLATION), nullable=False),
])

SCHEMA_SODA_VOC = pa.schema(_BASE)

SCHEMA_SODA_KTSH = pa.schema(_BASE + [
    pa.field("captions", pa.list_(pa.string()), nullable=False),
])

SCHEMAS = {
    "cs10k": SCHEMA_CS10K,
    "soda_voc": SCHEMA_SODA_VOC,
    "soda_ktsh": SCHEMA_SODA_KTSH,
}

DINO_ALERT_THRESHOLD = 0.75


def validate_table(table: pa.Table, sub: str) -> None:
    """Hard validate a table against the canonical schema for `sub`. Raises on mismatch.

    Also enforces that every non-null bbox is exactly 4 floats (since we use a
    variable-size list to support nullability).
    """
    expected = SCHEMAS[sub]
    if table.schema.names != expected.names:
        raise ValueError(
            f"[{sub}] column names mismatch.\n  expected: {expected.names}\n  got:      {table.schema.names}"
        )
    for f_exp, f_got in zip(expected, table.schema):
        if not f_exp.type.equals(f_got.type):
            raise ValueError(
                f"[{sub}] field '{f_exp.name}' type mismatch.\n  expected: {f_exp.type}\n  got:      {f_got.type}"
            )
        if f_exp.nullable is False and f_got.nullable is True:
            raise ValueError(f"[{sub}] field '{f_exp.name}' must be non-nullable")

    # bbox length invariant: each non-null bbox must be exactly 4 floats.
    for col_name in ("objects", "rule_violations"):
        if col_name not in table.column_names:
            continue
        for row_idx, lst in enumerate(table.column(col_name).to_pylist()):
            for i, item in enumerate(lst or []):
                bb = item.get("bbox")
                if bb is None:
                    continue
                if len(bb) != 4:
                    raise ValueError(
                        f"[{sub}] {col_name}[{row_idx}][{i}].bbox has len={len(bb)} (expected 4)"
                    )


def compute_quality_alert(dino_sim):
    """Apply DINO-based alert rule. Returns None when dino_sim is None."""
    if dino_sim is None:
        return None
    return bool(dino_sim < DINO_ALERT_THRESHOLD)
