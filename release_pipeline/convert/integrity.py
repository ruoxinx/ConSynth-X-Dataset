"""Integrity helpers for ConSynth-X release pipeline.

All operations are read-only and side-effect-free. Callers are responsible for
writing audit logs.
"""
import hashlib
import json
import os
from io import BytesIO
from typing import Iterable

import pyarrow as pa
import pyarrow.ipc as ipc
from PIL import Image


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def read_arrow_table(path: str) -> pa.Table:
    """Open both stream and file Arrow IPC formats. Raises on failure."""
    with pa.OSFile(path, "rb") as f:
        try:
            return ipc.open_stream(f).read_all()
        except pa.ArrowInvalid:
            f.seek(0)
            return ipc.open_file(f).read_all()


def coerce_image_bytes(field) -> bytes:
    """Existing shards store image either as bytes or as {'bytes': b...} struct.

    Always returns the raw JPEG/PNG bytes — never re-encodes.
    """
    if isinstance(field, (bytes, bytearray)):
        return bytes(field)
    if isinstance(field, dict) and "bytes" in field:
        return bytes(field["bytes"])
    raise TypeError(f"unexpected image field type: {type(field).__name__}")


def image_size(b: bytes) -> tuple[int, int]:
    """Return (width, height) without decoding pixels."""
    with Image.open(BytesIO(b)) as im:
        return im.size


def exif_orientation(b: bytes) -> int:
    """Return EXIF orientation tag (1 = no rotation, 6 = rotate 90 CW, 8 = rotate 90 CCW, etc.).
    Returns 1 if no orientation tag present."""
    with Image.open(BytesIO(b)) as im:
        return im.getexif().get(0x0112, 1) or 1


def apply_exif_rotation(b: bytes, jpeg_quality: int = 95) -> bytes:
    """Re-encode image bytes after applying EXIF rotation.

    Used when bbox annotations are valid for the EXIF-applied view but the raw
    bytes have a non-trivial orientation. Re-encoding loses byte-level fidelity
    on affected images, so the runner logs original sha, new sha, and orientation.
    """
    from PIL import ImageOps
    with Image.open(BytesIO(b)) as im:
        rotated = ImageOps.exif_transpose(im)
        buf = BytesIO()
        # Drop EXIF after rotation to prevent double-application downstream.
        rotated.save(buf, format="JPEG", quality=jpeg_quality, exif=b"")
        return buf.getvalue()


def assert_unique_image_ids(image_ids: Iterable[str], context: str) -> None:
    seen = set()
    dups = []
    for x in image_ids:
        if x in seen:
            dups.append(x)
        seen.add(x)
    if dups:
        raise ValueError(f"[{context}] duplicate image_id values: {dups[:10]}{'...' if len(dups)>10 else ''}")


def atomic_write(path: str, payload: bytes) -> None:
    """Write payload to path atomically: tmp → fsync → rename."""
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, path)


def write_audit_json(path: str, record: dict) -> None:
    atomic_write(path, json.dumps(record, indent=2, default=str).encode("utf-8"))


def diff_sets(a: set, b: set) -> dict:
    return {
        "only_in_a": sorted(a - b),
        "only_in_b": sorted(b - a),
        "common_count": len(a & b),
    }
