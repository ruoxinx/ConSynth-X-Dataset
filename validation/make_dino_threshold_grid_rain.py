#!/usr/bin/env python3
"""
Create a comparison grid for IP2P rain (heavy): 5 images above DINO >= 0.75 + 5 below.
"""

import csv
import io
import random
from pathlib import Path
import os as _os
from pathlib import Path as _Path
_DATA_ROOT = _Path(_os.environ.get("CONSYNTH_DATA_ROOT", str(_Path.home() / "consynth_data")))
_REPO_ROOT = _Path(_os.environ.get("CONSYNTH_REPO_ROOT", str(_Path(__file__).resolve().parents[1])))

import pyarrow as pa
from PIL import Image, ImageDraw, ImageFont

CSV_PATH = (_REPO_ROOT / "validation/results/dino_ssim/ip2p_rain.csv")
ORIG_PATH = (_DATA_ROOT / "augmentation_data_arrow/construction_site_test.arrow")
AUG_DIR = (_REPO_ROOT / "augmentation_data/construction_site/rain_snow/diffusion/test/rain")
OUT_PATH = (_REPO_ROOT / "validation/results/rain_heavy_dino075_grid.jpg")

THRESHOLD = 0.75
N_EACH = 5
SEED = 42
TILE_H = 300
LABEL_H = 40


def load_arrow_lookup(path):
    with open(path, "rb") as f:
        try:
            table = pa.ipc.open_file(path).read_all()
        except (pa.ArrowInvalid, Exception):
            f.seek(0)
            table = pa.ipc.open_stream(f).read_all()
    out = {}
    for i in range(len(table)):
        img_id = str(table.column("image_id")[i].as_py())
        b = table.column("image")[i].as_py()
        if isinstance(b, dict):
            b = b["bytes"]
        out[img_id] = Image.open(io.BytesIO(b)).convert("RGB")
    return out


def resize_tile(img, h=TILE_H):
    w = int(img.width * h / img.height)
    return img.resize((w, h), Image.LANCZOS)


def main():
    rows = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            rows.append({
                "id": r["image_id"],
                "ssim": float(r["ssim"]),
                "dino": float(r["dino_sim"]),
            })

    random.seed(SEED)
    above = [r for r in rows if r["dino"] >= THRESHOLD]
    below = [r for r in rows if r["dino"] < THRESHOLD]
    print(f"Above {THRESHOLD}: {len(above)} | Below: {len(below)}")

    sample_above = random.sample(above, N_EACH)
    sample_below = random.sample(below, N_EACH)

    print("Loading originals...")
    orig_lookup = load_arrow_lookup(ORIG_PATH)
    print("Loading augmented...")
    aug_lookup = {}
    for af in sorted(AUG_DIR.glob("*.arrow")):
        aug_lookup.update(load_arrow_lookup(af))

    groups = [("ABOVE", sample_above), ("BELOW", sample_below)]
    all_tiles_rows = []

    for label, samples in groups:
        orig_row = []
        aug_row = []
        captions = []
        for s in samples:
            orig_row.append(resize_tile(orig_lookup[s["id"]]))
            aug_row.append(resize_tile(aug_lookup[s["id"]]))
            captions.append(f"{s['id']}  DINO={s['dino']:.2f}  SSIM={s['ssim']:.2f}")
        all_tiles_rows.append((f"{label}: Original", orig_row, captions))
        all_tiles_rows.append((f"{label}: Augmented (rain heavy)", aug_row, None))

    col_widths = [max(row[1][j].width for row in all_tiles_rows) for j in range(N_EACH)]
    total_w = sum(col_widths) + (N_EACH + 1) * 10
    row_height = TILE_H + LABEL_H
    total_h = row_height * len(all_tiles_rows) + 80

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    y = 10
    for row_label, tiles, captions in all_tiles_rows:
        draw.text((10, y), row_label, fill=(0, 0, 0), font=font)
        y += 25
        x = 10
        for j, tile in enumerate(tiles):
            pad = (col_widths[j] - tile.width) // 2
            canvas.paste(tile, (x + pad, y))
            if captions is not None:
                draw.text((x + 5, y + TILE_H + 2), captions[j],
                          fill=(60, 60, 60), font=small_font)
            x += col_widths[j] + 10
        y += TILE_H + LABEL_H

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT_PATH, quality=92)
    print(f"\nSaved grid to: {OUT_PATH}")
    print(f"Size: {canvas.size}")


if __name__ == "__main__":
    main()
