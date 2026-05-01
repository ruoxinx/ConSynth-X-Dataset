#!/usr/bin/env python3
"""Build comparison grids for the completed rain_night / snow_night Order B output.

4 columns: Original | Night only | Weather day (IP2P+physics) | Weather_Night (final).
"""

import io
import os
import random
from pathlib import Path

import pyarrow as pa
from PIL import Image, ImageDraw, ImageFont

BASE = Path('/users/PGS0407/binben14/VietHuy')
ORIG = BASE / 'ConstructionSite/augmentation_data_arrow/construction_site_test.arrow'
NIGHT = BASE / 'ConSynth-X/augmentation_data/construction_site/night/test/night_constructionsite_test.arrow'
WEATHER_DAY = {
    'rain': BASE / 'ConSynth-X/augmentation_data/construction_site/rain_snow/diffusion/test/rain',
    'snow': BASE / 'ConSynth-X/augmentation_data/construction_site/rain_snow/diffusion/test/snow_light',
}
WEATHER_NIGHT = {
    'rain': BASE / 'ConSynth-X/augmentation_data/construction_site/night_weather/rain_night',
    'snow': BASE / 'ConSynth-X/augmentation_data/construction_site/night_weather/snow_night',
}
OUT = BASE / 'ConSynth-X/generation/preview_grids'
OUT.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 6
SEED = 42


def load_arrow(path):
    with open(path, 'rb') as f:
        return pa.ipc.open_stream(f).read_all()


def index_by_id(table, id_col='image_id'):
    return {table.column(id_col)[i].as_py(): i for i in range(len(table))}


def index_dir_by_id(dir_path, id_col='image_id'):
    """Merge batch_*.arrow in a dir into {image_id: (table, idx)}."""
    idx = {}
    for p in sorted(dir_path.glob('batch_*.arrow')):
        t = load_arrow(p)
        col = t.column(id_col)
        for i in range(len(t)):
            idx[col[i].as_py()] = (t, i)
    return idx


def get_image(table, i):
    row = table.column('image')[i].as_py()
    if isinstance(row, dict):
        return Image.open(io.BytesIO(row['bytes'])).convert('RGB')
    return Image.open(io.BytesIO(row)).convert('RGB')


def build_grid(samples, weather, cell_w=420, cell_h=300, padding=4, label_h=30):
    n_rows = len(samples)
    n_cols = 4
    grid_w = n_cols * cell_w + (n_cols + 1) * padding
    grid_h = n_rows * (cell_h + label_h) + (n_rows + 1) * padding + label_h
    grid = Image.new('RGB', (grid_w, grid_h), (30, 30, 30))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype('/usr/share/fonts/liberation/LiberationSans-Bold.ttf', 16)
        font_small = ImageFont.truetype('/usr/share/fonts/liberation/LiberationSans-Regular.ttf', 12)
    except Exception:
        try:
            font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf', 16)
            font_small = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf', 12)
        except Exception:
            font = ImageFont.load_default()
            font_small = font

    headers = [
        'Original',
        'Night (CycleGAN)',
        f'{weather.capitalize()} day (IP2P+physics)',
        f'{weather.capitalize()}_Night (Order B final)',
    ]
    colors = [(200, 200, 200), (100, 150, 255), (255, 200, 120), (160, 255, 180)]
    for c, (h_text, hcolor) in enumerate(zip(headers, colors)):
        x = padding + c * (cell_w + padding)
        draw.text((x + 10, 8), h_text, fill=hcolor, font=font)

    keys = ['original', 'night', 'weather_day', 'weather_night']
    for r, s in enumerate(samples):
        y0 = label_h + padding + r * (cell_h + label_h + padding)
        draw.text((8, y0 - 2), f"ID: {s['image_id']}", fill=(150, 150, 150), font=font_small)
        for c, k in enumerate(keys):
            x = padding + c * (cell_w + padding)
            y = y0 + label_h - 10
            img = s[k]
            if img is not None:
                grid.paste(img.resize((cell_w, cell_h), Image.LANCZOS), (x, y))
    return grid


def main():
    print('Loading arrows...')
    orig_t = load_arrow(ORIG)
    night_t = load_arrow(NIGHT)
    orig_idx = index_by_id(orig_t)
    night_idx = index_by_id(night_t)
    print(f'  orig={len(orig_t)}  night={len(night_t)}')

    for weather in ['rain', 'snow']:
        day_idx = index_dir_by_id(WEATHER_DAY[weather])
        night_w_idx = index_dir_by_id(WEATHER_NIGHT[weather])
        common = sorted(set(orig_idx) & set(night_idx) & set(day_idx) & set(night_w_idx))
        print(f'\n{weather}: day={len(day_idx)}  weather_night={len(night_w_idx)}  common={len(common)}')
        if len(common) < N_SAMPLES:
            print(f'  SKIP: only {len(common)} overlap')
            continue
        rng = random.Random(SEED)
        picks = rng.sample(common, N_SAMPLES)
        samples = []
        for image_id in picks:
            t_day, i_day = day_idx[image_id]
            t_wn, i_wn = night_w_idx[image_id]
            samples.append({
                'image_id': image_id,
                'original': get_image(orig_t, orig_idx[image_id]),
                'night': get_image(night_t, night_idx[image_id]),
                'weather_day': get_image(t_day, i_day),
                'weather_night': get_image(t_wn, i_wn),
            })
        grid = build_grid(samples, weather)
        out_path = OUT / f'weather_night_output_{weather}.jpg'
        grid.save(out_path, 'JPEG', quality=92)
        print(f'  -> {out_path}  ({grid.size[0]}x{grid.size[1]})')


if __name__ == '__main__':
    main()
