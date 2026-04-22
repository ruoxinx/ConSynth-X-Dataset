#!/usr/bin/env python3
"""
Create a comparison grid: 5 images above DINO >= 0.75 + 5 below.

Layout (2 rows × 5 cols for each group):
  Row 1: Original (above threshold)
  Row 2: Augmented (above threshold)  → should look realistic
  Row 3: Original (below threshold)
  Row 4: Augmented (below threshold)  → should look problematic
"""

import csv
import io
import random
from pathlib import Path
import os as _os
from pathlib import Path as _Path
_DATA_ROOT = _Path(_os.environ.get("CONSYNTH_DATA_ROOT", str(_Path.home() / "consynth_data")))
_REPO_ROOT = _Path(_os.environ.get("CONSYNTH_REPO_ROOT", str(_Path(__file__).resolve().parents[1])))
_BR_ROOT = _Path(_os.environ.get("CONSYNTH_BENCHMARK_RUNNER", str(_REPO_ROOT.parent / "Benchmark_runner")))

import pyarrow as pa
from PIL import Image, ImageDraw, ImageFont

CSV_PATH = (_REPO_ROOT / "validation/results/snow_strong_dino_ssim.csv")
ORIG_PATH = (_DATA_ROOT / "augmentation_data_arrow/construction_site_test.arrow")
AUG_DIR = (_DATA_ROOT / "output/construction_site_test/diffusion_snow_strong")
OUT_PATH = (_REPO_ROOT / "validation/results/snow_strong_dino075_grid.jpg")

THRESHOLD = 0.75
N_EACH = 5
SEED = 42
TILE_H = 300  # pixel height per tile
LABEL_H = 40


def load_arrow_lookup(path):
    with open(path, "rb") as f:
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
    # Read DINO scores
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

    # Load all needed images
    print("Loading originals...")
    orig_lookup = load_arrow_lookup(ORIG_PATH)
    print("Loading augmented...")
    aug_lookup = {}
    for af in sorted(AUG_DIR.glob("batch_*.arrow")):
        aug_lookup.update(load_arrow_lookup(af))

    # Build rows of resized tiles
    groups = [("ABOVE", sample_above), ("BELOW", sample_below)]
    all_tiles_rows = []  # list of (label_prefix, list_of_PIL)

    for label, samples in groups:
        orig_row = []
        aug_row = []
        captions = []
        for s in samples:
            orig_row.append(resize_tile(orig_lookup[s["id"]]))
            aug_row.append(resize_tile(aug_lookup[s["id"]]))
            captions.append(f"{s['id']}  DINO={s['dino']:.2f}  SSIM={s['ssim']:.2f}")
        all_tiles_rows.append((f"{label}: Original", orig_row, captions))
        all_tiles_rows.append((f"{label}: Augmented", aug_row, None))

    # Compute canvas size (all rows must have same width; use max of each group)
    # Simpler: pad to uniform tile width using centered layout
    max_widths = [max(t.width for t in row[1]) for row in all_tiles_rows]
    col_widths = [max(row[1][j].width for row in all_tiles_rows) for j in range(N_EACH)]
    total_w = sum(col_widths) + (N_EACH + 1) * 10
    row_height = TILE_H + LABEL_H
    total_h = row_height * len(all_tiles_rows) + 80  # extra for row labels + captions

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
        # Row label
        draw.text((10, y), row_label, fill=(0, 0, 0), font=font)
        y += 25
        # Tiles
        x = 10
        for j, tile in enumerate(tiles):
            # Center tile in its column
            pad = (col_widths[j] - tile.width) // 2
            canvas.paste(tile, (x + pad, y))
            if captions is not None:
                # Add small caption below tile
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
