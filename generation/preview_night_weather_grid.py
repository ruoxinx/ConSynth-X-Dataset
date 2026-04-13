#!/usr/bin/env python3
"""
Preview Grid: Snow Night & Rain Night combinations
Pipeline: Night images (CycleGAN-Turbo) + Weather Particles (rain/snow overlay)

Creates a comparison grid: Original | Night | Rain+Night | Snow+Night
"""

import sys
import random
import io
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pyarrow as pa

# ============================================================
# Rain & Snow particle generators (from diffusion/physics.py)
# Self-contained: only needs numpy + cv2
# ============================================================

def add_rain_fog(image: np.ndarray, strength=0.25) -> np.ndarray:
    """Add atmospheric haze from rain — reduces contrast, adds grey tone."""
    h, w = image.shape[:2]
    result = image.astype(np.float64)
    fog_color = np.array([170, 175, 185], dtype=np.float64)
    gradient = np.linspace(strength * 1.3, strength * 0.5, h).reshape(h, 1, 1)
    gradient = np.broadcast_to(gradient, (h, w, 3))
    result = result * (1 - gradient) + fog_color * gradient
    return np.clip(result, 0, 255).astype(np.uint8)


def add_natural_rain(image: np.ndarray) -> np.ndarray:
    """Rain streaks at 3 depth layers + atmospheric fog."""
    h, w = image.shape[:2]
    result = add_rain_fog(image, strength=random.uniform(0.15, 0.30))
    result = result.astype(np.float64)

    base_angle = random.choice([-1, 1]) * random.randint(75, 87)
    wind_var = random.uniform(-5, 5)

    layers = [
        {'n': random.randint(1500, 2500), 'len': (10, 22), 'thick': 1,
         'alpha': 0.25, 'y_range': (0, h)},
        {'n': random.randint(1000, 2000), 'len': (18, 35), 'thick': 1,
         'alpha': 0.35, 'y_range': (0, h)},
        {'n': random.randint(300, 700), 'len': (25, 50), 'thick': random.choice([1, 2]),
         'alpha': 0.40, 'y_range': (0, h)},
    ]

    for layer in layers:
        canvas = np.zeros((h, w), dtype=np.float64)
        angle = np.radians(base_angle + random.uniform(-3, 3) + wind_var)
        y_lo, y_hi = layer['y_range']
        for _ in range(layer['n']):
            y = random.randint(y_lo, y_hi - 1)
            x = random.randint(0, w - 1)
            length = random.randint(*layer['len'])
            dx = int(length * np.cos(angle))
            dy = int(length * np.sin(angle))
            cv2.line(canvas, (x, y), (x + dx, y + dy), 255, layer['thick'], cv2.LINE_AA)

        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
        a = (canvas / 255.0) * layer['alpha']
        a = np.expand_dims(a, axis=2)
        color = np.array([200 + random.randint(0, 20),
                          205 + random.randint(0, 15),
                          220 + random.randint(0, 15)], dtype=np.float64)
        result = result * (1 - a) + color * a

    return np.clip(result, 0, 255).astype(np.uint8)


def add_natural_snow(image: np.ndarray) -> np.ndarray:
    """Subtle falling snowflakes — soft, varied sizes."""
    h, w = image.shape[:2]
    result = image.astype(np.float64)

    layers = [
        {'n': random.randint(1500, 3000), 'r': (1, 2), 'alpha': 0.35, 'blur': 3},
        {'n': random.randint(500, 1000), 'r': (2, 4), 'alpha': 0.50, 'blur': 5},
        {'n': random.randint(100, 300), 'r': (4, 6), 'alpha': 0.65, 'blur': 5},
    ]

    for layer in layers:
        canvas = np.zeros((h, w), dtype=np.float64)
        for _ in range(layer['n']):
            y = random.randint(0, h - 1)
            x = random.randint(0, w - 1)
            r = random.randint(*layer['r'])
            cv2.circle(canvas, (x, y), r, 255, -1, cv2.LINE_AA)

        k = layer['blur']
        if k % 2 == 0:
            k += 1
        canvas = cv2.GaussianBlur(canvas, (k, k), 0)

        a = (canvas / 255.0) * layer['alpha']
        a = np.expand_dims(a, axis=2)
        color = np.array([240, 245, 255], dtype=np.float64)
        result = result + color * a * (1 - result / 255.0)

    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================
# Data loading
# ============================================================

def load_arrow(path):
    with open(path, 'rb') as f:
        return pa.ipc.open_stream(f).read_all()


def get_image(table, idx):
    row = table.column('image')[idx].as_py()
    if isinstance(row, dict):
        return Image.open(io.BytesIO(row['bytes'])).convert('RGB')
    elif isinstance(row, bytes):
        return Image.open(io.BytesIO(row)).convert('RGB')
    return None


# ============================================================
# Grid creation
# ============================================================

def create_comparison_grid(samples, cell_w=400, cell_h=300, padding=4, label_h=30):
    """
    samples: list of dicts with keys 'original', 'night', 'rain_night', 'snow_night', 'image_id'
    Each value is a PIL Image.
    """
    n_rows = len(samples)
    n_cols = 4  # original, night, rain_night, snow_night

    grid_w = n_cols * cell_w + (n_cols + 1) * padding
    grid_h = n_rows * (cell_h + label_h) + (n_rows + 1) * padding + label_h  # extra label_h for header

    grid = Image.new('RGB', (grid_w, grid_h), (30, 30, 30))
    draw = ImageDraw.Draw(grid)

    # Try to get a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/liberation/LiberationSans-Bold.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/liberation/LiberationSans-Regular.ttf", 12)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 16)
            font_small = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
            font_small = font

    # Header row
    headers = ['Original', 'Night (CycleGAN)', 'Rain + Night', 'Snow + Night']
    header_colors = [(200, 200, 200), (100, 150, 255), (255, 180, 80), (180, 220, 255)]
    for col, (header, hcolor) in enumerate(zip(headers, header_colors)):
        x = padding + col * (cell_w + padding)
        draw.text((x + cell_w // 2 - len(header) * 4, 8), header, fill=hcolor, font=font)

    keys = ['original', 'night', 'rain_night', 'snow_night']

    for row_idx, sample in enumerate(samples):
        y_offset = label_h + padding + row_idx * (cell_h + label_h + padding)

        # Image ID label
        img_id = sample.get('image_id', f'#{row_idx}')
        draw.text((8, y_offset - 2), f"ID: {img_id}", fill=(150, 150, 150), font=font_small)

        for col_idx, key in enumerate(keys):
            x = padding + col_idx * (cell_w + padding)
            y = y_offset + label_h - 10

            img = sample[key]
            if img is not None:
                img_resized = img.resize((cell_w, cell_h), Image.LANCZOS)
                grid.paste(img_resized, (x, y))

    return grid


# ============================================================
# Main
# ============================================================

def main():
    BASE_DIR = Path('/users/PGS0407/binben14/VietHuy/ConstructionSite/augmentation_data_arrow')
    OUTPUT_DIR = Path('/users/PGS0407/binben14/VietHuy/ConSynth-X/generation/preview_grids')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading arrow files...")
    orig_table = load_arrow(BASE_DIR / 'construction_site_test.arrow')
    night_table = load_arrow(BASE_DIR / 'night.arrow')

    print(f"Original: {len(orig_table)} images, Night: {len(night_table)} images")

    # Build image_id index for night table
    night_ids = {night_table.column('image_id')[i].as_py(): i for i in range(len(night_table))}

    # Sample 5 diverse images
    random.seed(42)
    n_samples = 5
    indices = random.sample(range(len(orig_table)), n_samples)
    indices.sort()

    samples = []
    for i, idx in enumerate(indices):
        image_id = orig_table.column('image_id')[idx].as_py()
        print(f"[{i+1}/{n_samples}] Processing image {image_id}...")

        # Get original
        orig_img = get_image(orig_table, idx)

        # Get night version
        night_idx = night_ids.get(image_id)
        if night_idx is None:
            print(f"  Warning: no night version for {image_id}, skipping")
            continue
        night_img = get_image(night_table, night_idx)

        # Apply rain on night image
        random.seed(42 + idx)  # Reproducible
        night_np = np.array(night_img)
        rain_night_np = add_natural_rain(night_np.copy())
        rain_night_img = Image.fromarray(rain_night_np)

        # Apply snow on night image
        random.seed(99 + idx)  # Different seed for variety
        snow_night_np = add_natural_snow(night_np.copy())
        snow_night_img = Image.fromarray(snow_night_np)

        samples.append({
            'image_id': image_id,
            'original': orig_img,
            'night': night_img,
            'rain_night': rain_night_img,
            'snow_night': snow_night_img,
        })
        print(f"  Done: {orig_img.size}")

    print(f"\nCreating comparison grid ({len(samples)} rows x 4 cols)...")
    grid = create_comparison_grid(samples, cell_w=450, cell_h=320)

    output_path = OUTPUT_DIR / 'preview_night_weather_grid.jpg'
    grid.save(output_path, 'JPEG', quality=95)
    print(f"Grid saved: {output_path}")
    print(f"Grid size: {grid.size[0]}x{grid.size[1]}")

    # Also save individual images for closer inspection
    for sample in samples:
        img_id = sample['image_id']
        for key in ['rain_night', 'snow_night']:
            sample[key].save(OUTPUT_DIR / f'{img_id}_{key}.jpg', 'JPEG', quality=95)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
