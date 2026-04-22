#!/usr/bin/env python3
"""
Rain heavy = physics-only (no IP2P/diffusion).

Applies `add_natural_rain(image, intensity='heavy_fog')` directly on the
original image, producing the heavy level. Light level remains the existing
IP2P-based `test/rain/` arrow files — unchanged.

Pilot: 10 samples from CS test. 3-col grid [original | light (IP2P) | heavy (physics)].
Runs on CPU; no GPU needed. Uses same SEED=42 indices as earlier pilots for
direct visual comparison.
"""

import os
import sys
import io
import random
from pathlib import Path

_REPO = Path(os.environ.get("CONSYNTH_REPO_ROOT", Path(__file__).resolve().parents[4]))

import numpy as np
import pyarrow as pa
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from physics import add_natural_rain

OUTPUT_DIR = _REPO / "validation" / "results" / "rain_heavy_physics_on_light_pilot"
ARROW_PATH = Path("/users/PGS0407/binben14/VietHuy/ConstructionSite/augmentation_data_arrow/construction_site_test.arrow")
LIGHT_DIR = _REPO / "augmentation_data" / "construction_site" / "rain_snow" / "diffusion" / "test" / "rain"

N_SAMPLES = 10
SEED = 42


def load_arrow(path):
    with open(path, 'rb') as f:
        return pa.ipc.open_stream(f).read_all()


def get_image(table, idx):
    row = table.column('image')[idx].as_py()
    if isinstance(row, dict):
        return Image.open(io.BytesIO(row['bytes'])).convert('RGB')
    return Image.open(io.BytesIO(row)).convert('RGB')


def load_light_by_image_id(image_ids):
    """Build {image_id: PIL} from all light arrow shards."""
    out = {}
    target = set(str(x) for x in image_ids)
    for shard in sorted(LIGHT_DIR.glob('*.arrow')):
        t = load_arrow(shard)
        ids = [str(t.column('image_id')[i].as_py()) for i in range(len(t))]
        for i, iid in enumerate(ids):
            if iid in target:
                row = t.column('image')[i].as_py()
                out[iid] = Image.open(io.BytesIO(row['bytes'] if isinstance(row, dict) else row)).convert('RGB')
        if len(out) == len(target):
            break
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading originals...")
    table = load_arrow(ARROW_PATH)
    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(table), size=N_SAMPLES, replace=False)
    image_ids = [str(table.column('image_id')[int(idx)].as_py()) for idx in indices]
    print(f"Image IDs: {image_ids}")

    originals = []
    orig_dir = OUTPUT_DIR / "original"
    orig_dir.mkdir(exist_ok=True)
    for i, idx in enumerate(indices):
        img = get_image(table, int(idx))
        img.save(orig_dir / f"{i:03d}.jpg")
        originals.append(img)

    print("Loading existing IP2P light outputs...")
    light_map = load_light_by_image_id(image_ids)
    print(f"Found {len(light_map)}/{len(image_ids)} light outputs")

    lights = []
    for iid in image_ids:
        if iid in light_map:
            lights.append(light_map[iid])
        else:
            print(f"  missing light for {iid}, using original as placeholder")
            lights.append(originals[image_ids.index(iid)])

    print("Applying heavy physics ON TOP of light IP2P output...")
    heavies = []
    heavy_dir = OUTPUT_DIR / "heavy"
    heavy_dir.mkdir(exist_ok=True)
    for i, (idx, iid) in enumerate(zip(indices, image_ids)):
        if iid not in light_map:
            heavies.append(originals[i])
            continue
        random.seed(SEED + int(idx))
        aug = add_natural_rain(np.array(light_map[iid]), intensity='heavy_fog')
        out = Image.fromarray(aug)
        out.save(heavy_dir / f"{i:03d}.jpg")
        heavies.append(out)
        print(f"  [{i+1}/{N_SAMPLES}] done (id={iid})")

    # Grid: 3 cols
    print("\nBuilding grid...")
    cell_w = 384
    cell_h_list = [int(im.height * cell_w / im.width) for im in originals]
    label_h = 32
    margin = 4
    cols = ["original", "light (IP2P)", "heavy (physics)"]

    total_w = cell_w * len(cols) + margin * (len(cols) + 1)
    total_h = sum(cell_h_list) + margin * (N_SAMPLES + 1) + label_h

    grid = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    for ci, cname in enumerate(cols):
        x = margin + ci * (cell_w + margin)
        draw.text((x + cell_w // 2 - 50, 6), cname, fill=(0, 0, 0), font=font)
    y = label_h + margin
    for ri in range(N_SAMPLES):
        ch = cell_h_list[ri]
        cells = [originals[ri], lights[ri], heavies[ri]]
        for ci, img in enumerate(cells):
            x = margin + ci * (cell_w + margin)
            grid.paste(img.resize((cell_w, ch), Image.LANCZOS), (x, y))
        y += ch + margin

    grid_path = OUTPUT_DIR / "grid.jpg"
    grid.save(grid_path, quality=92)
    print(f"Grid saved: {grid_path}")


if __name__ == "__main__":
    main()
