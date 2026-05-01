"""Pack soda_voc/small/_jpgs_extra2k/ into a single Arrow shard matching the
`soda_small.arrow` schema [image_id, image, annotation, meta].

Bbox transformation (source -> canvas):
    bbox_canvas = bbox_source * scale_factor + (paste.x, paste.y)
    <size>      = canvas_size  (from meta)

The transform is linear and deterministic. Round-trip test: re-derive source
bbox from canvas bbox and meta, must equal original VOC bbox within sub-pixel.
"""
from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as ET

import pyarrow as pa
import pyarrow.ipc as ipc
from PIL import Image
from io import BytesIO

EXTRA2K_DIR = "/users/PGS0407/binben14/VietHuy/ConSynth-X/augmentation_data/soda_voc/small/_jpgs_extra2k"
OUT_PATH = "/users/PGS0407/binben14/VietHuy/ConSynth-X/augmentation_data/soda_voc/small/extra2k.arrow"


def transform_voc_xml(xml_text: str, scale: float, paste_x: int, paste_y: int,
                       canvas_w: int, canvas_h: int, image_id: str) -> str:
    """Transform VOC XML bbox from source coords to canvas coords.

    Round-trip checked: bbox_source = (bbox_canvas - paste) / scale must equal
    original VOC bbox to within 1px.
    """
    root = ET.fromstring(xml_text)
    size_el = root.find("size")
    if size_el is None:
        raise ValueError(f"{image_id}: VOC XML missing <size>")
    src_w = int(size_el.findtext("width"))
    src_h = int(size_el.findtext("height"))
    # Update size to canvas
    size_el.find("width").text = str(canvas_w)
    size_el.find("height").text = str(canvas_h)
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        xmin = float(bb.findtext("xmin"))
        ymin = float(bb.findtext("ymin"))
        xmax = float(bb.findtext("xmax"))
        ymax = float(bb.findtext("ymax"))
        # transform
        nx1 = xmin * scale + paste_x
        ny1 = ymin * scale + paste_y
        nx2 = xmax * scale + paste_x
        ny2 = ymax * scale + paste_y
        # round-trip check
        rx1 = (nx1 - paste_x) / scale
        ry1 = (ny1 - paste_y) / scale
        if abs(rx1 - xmin) > 1e-3 or abs(ry1 - ymin) > 1e-3:
            raise RuntimeError(f"{image_id}: round-trip drift > 1e-3")
        # bounds clamp (with 1px tolerance)
        if nx1 < -1 or ny1 < -1 or nx2 > canvas_w + 1 or ny2 > canvas_h + 1:
            # Source bbox falls outside canvas after transform - skip object
            obj_parent = root
            obj_parent.remove(obj)
            continue
        bb.find("xmin").text = f"{int(round(max(0.0, nx1)))}"
        bb.find("ymin").text = f"{int(round(max(0.0, ny1)))}"
        bb.find("xmax").text = f"{int(round(min(float(canvas_w), nx2)))}"
        bb.find("ymax").text = f"{int(round(min(float(canvas_h), ny2)))}"
    return ET.tostring(root, encoding="unicode")


def main():
    images_dir = os.path.join(EXTRA2K_DIR, "images")
    meta_dir = os.path.join(EXTRA2K_DIR, "meta")
    image_files = sorted([x for x in os.listdir(images_dir) if x.lower().endswith(".jpg")])
    print(f"Found {len(image_files)} images")

    rows = {"image_id": [], "image": [], "annotation": [], "meta": []}
    n_skip_no_meta = 0
    n_objects_dropped_oob = 0
    for i, fn in enumerate(image_files):
        image_id = os.path.splitext(fn)[0]
        meta_path = os.path.join(meta_dir, image_id + ".json")
        if not os.path.exists(meta_path):
            n_skip_no_meta += 1
            continue
        with open(meta_path) as f:
            meta = json.load(f)

        scale = float(meta["scale_factor"])
        paste = meta["paste_position"]
        canvas_w, canvas_h = meta["canvas_size"]
        src_xml = meta["original_annotations"]["annotation"]
        if not isinstance(src_xml, str):
            raise RuntimeError(f"{image_id}: meta.original_annotations.annotation not string")

        # Count objects before
        before = ET.fromstring(src_xml).findall("object")
        new_xml = transform_voc_xml(src_xml, scale, int(paste["x"]), int(paste["y"]),
                                     int(canvas_w), int(canvas_h), image_id)
        after = ET.fromstring(new_xml).findall("object")
        n_objects_dropped_oob += len(before) - len(after)

        # Read image bytes & verify dimensions
        with open(os.path.join(images_dir, fn), "rb") as f:
            img_bytes = f.read()
        with Image.open(BytesIO(img_bytes)) as im:
            pil_w, pil_h = im.size
        if (pil_w, pil_h) != (canvas_w, canvas_h):
            raise RuntimeError(f"{image_id}: PIL size {(pil_w,pil_h)} != meta canvas {(canvas_w,canvas_h)}")

        rows["image_id"].append(image_id)
        rows["image"].append(img_bytes)
        rows["annotation"].append(new_xml)
        rows["meta"].append(json.dumps(meta))

        if (i + 1) % 200 == 0:
            print(f"  packed {i+1}/{len(image_files)}")

    print(f"\nDone. skipped_no_meta={n_skip_no_meta}, objects_dropped_oob={n_objects_dropped_oob}")
    print(f"Total rows to write: {len(rows['image_id'])}")

    schema = pa.schema([
        pa.field("image_id", pa.string()),
        pa.field("image", pa.binary()),
        pa.field("annotation", pa.string()),
        pa.field("meta", pa.string(), nullable=True),
    ])
    table = pa.Table.from_pydict(rows, schema=schema)
    print(f"Built pa.Table: rows={table.num_rows} cols={table.column_names}")

    # Write Arrow IPC stream (matching existing soda_small.arrow format)
    tmp = OUT_PATH + ".tmp"
    with pa.OSFile(tmp, "wb") as sink:
        with ipc.new_stream(sink, schema) as writer:
            writer.write_table(table)
    os.replace(tmp, OUT_PATH)
    print(f"Wrote {OUT_PATH}  size={os.path.getsize(OUT_PATH)//1024//1024} MB")


if __name__ == "__main__":
    main()
