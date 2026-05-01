"""Join bbox annotations from a source shard onto a synthetic shard that has none.

Specifically used for soda_voc diffusion shards (variant 9) which store only
ssim/lpips/status. The synthetic image preserves source geometry — verified by
W,H equality check — so source bbox normalised coords remain valid.

Safety contract:
  - For every synthetic row, image_id must exist in source. Unmatched -> ERROR.
  - For every synthetic row, PIL-derived (W,H) of the synthetic image must
    equal (W,H) of the source image. Mismatch -> ERROR (do not silently skip).
  - The source bbox is taken from source_objects[image_id] verbatim.
"""
from __future__ import annotations

from typing import Optional

from .integrity import coerce_image_bytes, image_size, read_arrow_table


class BboxJoinError(Exception):
    pass


def build_source_index(source_arrow_path: str, parser, *, image_loader=None) -> dict:
    """Build {image_id: {'objects': [...], 'size': (W,H)}}.

    `parser` is a callable that takes (row_dict, image_w, image_h) -> list[dict] objects.
    `image_loader` (optional) is a callable taking (row_dict) -> (bytes, w, h, warnings).
    Defaults to raw image_size(bytes) — used for indexing source shards. Pass a
    custom loader (e.g., one that applies EXIF rotation) when source needs
    canonicalisation matching the parser's expectations.
    """
    table = read_arrow_table(source_arrow_path)
    index: dict[str, dict] = {}
    for row in table.to_pylist():
        iid = row["image_id"]
        if image_loader is not None:
            b, w, h, _ = image_loader(row)
        else:
            b = coerce_image_bytes(row["image"])
            w, h = image_size(b)
        objects = parser(row, w, h)
        index[iid] = {"objects": objects, "size": (w, h)}
    return index


def attach_objects(synthetic_rows: list[dict], source_index: dict, *, image_loader=None) -> tuple[list[list[dict]], list[bytes], list[dict]]:
    """For each synthetic row, look up source objects by image_id (or ref_id).

    Returns (objects_per_row, image_bytes_per_row, warnings). The returned
    image_bytes are the (possibly EXIF-rotated) bytes used for size matching, so
    the caller can persist them consistently with the bbox interpretation.

    Raises BboxJoinError on unmatched or size-mismatched row (no silent skips).
    """
    out_objects: list[list[dict]] = []
    out_bytes: list[bytes] = []
    warnings: list[dict] = []
    for i, row in enumerate(synthetic_rows):
        sid = row.get("ref_id") or row["image_id"]
        if sid not in source_index:
            raise BboxJoinError(f"row {i} image_id={row['image_id']!r} ref_id={sid!r} not in source index")
        if image_loader is not None:
            b, w, h, lw = image_loader(row)
            warnings.extend(lw)
            syn_size = (w, h)
        else:
            b = coerce_image_bytes(row["image"])
            syn_size = image_size(b)
        src = source_index[sid]
        if syn_size != src["size"]:
            raise BboxJoinError(
                f"row {i} image_id={row['image_id']!r}: synthetic size {syn_size} != source size {src['size']} "
                f"(bbox join would be invalid; aborting)"
            )
        out_objects.append([dict(o, bbox=list(o["bbox"])) for o in src["objects"]])
        out_bytes.append(b)
    return out_objects, out_bytes, warnings
