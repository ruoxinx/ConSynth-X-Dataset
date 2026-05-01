"""Draw bounding boxes on a few ConSynth-X v1 samples for visual inspection."""
import io
import os
import random
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/fs/scratch/PGS0407/binben14/ConSynth-X-release-v1")
OUT = Path("/users/PGS0407/binben14/VietHuy/ConSynth-X/propotype/bbox")
OUT.mkdir(parents=True, exist_ok=True)

# (dataset, condition_dir, parquet_file, n_samples)
TARGETS = [
    ("soda_voc", "small",      "soda_small.parquet",       4),
    ("soda_voc", "small",      "extra2k.parquet",          2),
    ("soda_voc", "fog_heavy",  "fog_heavy.parquet",        2),
    ("soda_voc", "night",      "soda_day2night.parquet",   2),
    ("soda_voc", "rain_heavy", "rain_heavy.parquet",       2),
    ("soda_voc", "snow_heavy", "snow_heavy.parquet",       2),
    ("cs10k",    "small",      "test__small_constructionsite_test.parquet",  3),
    ("cs10k",    "small",      "train__small_constructionsite_train.parquet", 2),
    ("cs10k",    "night",      "night_constructionsite_test.parquet", 2),
    ("cs10k",    "rain_night", "rain_night.parquet",       2),
    ("cs10k",    "snow_heavy", "test__snow_heavy.parquet", 2),
    ("cs10k",    "rain_heavy", "test__rain_heavy.parquet", 2),
]

# stable color per class
def color_for(class_id: int):
    random.seed(class_id * 7919 + 31)
    return (random.randint(60, 255), random.randint(60, 255), random.randint(60, 255))

try:
    FONT = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 16)
except Exception:
    FONT = ImageFont.load_default()


def draw(img: Image.Image, objs):
    W, H = img.size
    d = ImageDraw.Draw(img)
    lw = max(2, int(min(W, H) / 400))
    for o in objs:
        x1, y1, x2, y2 = o["bbox"]
        # parquet stores normalized xyxy
        if max(x1, y1, x2, y2) <= 1.5:
            x1, y1, x2, y2 = x1 * W, y1 * H, x2 * W, y2 * H
        c = color_for(int(o["class_id"]))
        d.rectangle([x1, y1, x2, y2], outline=c, width=lw)
        label = f"{o['class_name']}"
        try:
            tb = d.textbbox((0, 0), label, font=FONT)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except Exception:
            tw, th = 8 * len(label), 16
        ty = max(0, y1 - th - 2)
        d.rectangle([x1, ty, x1 + tw + 4, ty + th + 2], fill=c)
        d.text((x1 + 2, ty), label, fill=(0, 0, 0), font=FONT)
    return img


for ds, cond, fname, n in TARGETS:
    pq_path = ROOT / ds / "parquet" / cond / fname
    if not pq_path.exists():
        print(f"[skip] missing {pq_path}")
        continue
    table = pq.read_table(pq_path, columns=["image", "image_id", "objects"])
    total = table.num_rows
    rng = random.Random(f"{ds}/{cond}")
    # prefer rows with at least one object
    chosen, tries = [], 0
    while len(chosen) < n and tries < n * 20:
        i = rng.randrange(total)
        row = table.slice(i, 1).to_pylist()[0]
        if row["objects"]:
            chosen.append(row)
        tries += 1
    if not chosen:
        chosen = table.slice(0, n).to_pylist()
    for k, row in enumerate(chosen, 1):
        img = Image.open(io.BytesIO(row["image"])).convert("RGB")
        img = draw(img, row["objects"])
        out = OUT / f"{ds}__{cond}__{row['image_id']}__bbox.jpg"
        img.save(out, quality=92)
        print(f"[ok] {out.name}  ({len(row['objects'])} boxes, {img.size})")

print(f"\nDone. Output: {OUT}")
