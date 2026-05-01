"""Build paper-style Figure 2: 4 rows x 4 cols (Clear / Fog / Rain / Snow).

Each row = different sample image. Column labels printed at the bottom.
"""
import io
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow as pa
import pyarrow.ipc as ipc
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/users/PGS0407/binben14/VietHuy/ConSynth-X')
OUT_DIR = ROOT / 'paper'
OUT_DIR.mkdir(exist_ok=True)

CS_TEST_ORIG = '/users/PGS0407/binben14/VietHuy/ConstructionSite/LouisChen15___construction_site/construction_site-test.arrow'

COLUMNS: List[Tuple[str, str]] = [
    ('Clear (GT)', CS_TEST_ORIG),
    ('Fog',   str(ROOT / 'augmentation_data/construction_site/fog/diffusion/test/fog_medium.arrow')),
    ('Rain',  str(ROOT / 'augmentation_data/construction_site/rain_snow/diffusion/test/rain_light/rain_light.arrow')),
    ('Snow',  str(ROOT / 'augmentation_data/construction_site/rain_snow/diffusion/test/snow_light/snow_light.arrow')),
    ('Small', str(ROOT / 'augmentation_data/construction_site/small/test/small_constructionsite_test.arrow')),
]
N_ROWS = 4

# Layout — paper-quality, slightly wider than tall to match screenshot
CELL_W, CELL_H = 320, 200
PAD = 6
LABEL_H = 32
SEED = 11


def load_arrow_index(path: str) -> Dict[str, bytes]:
    src = pa.memory_map(path, 'r'); r = ipc.open_stream(src)
    out: Dict[str, bytes] = {}
    for b in r:
        cols = b.column_names
        if 'image_id' not in cols or 'image' not in cols:
            continue
        ids = b.column('image_id').to_pylist()
        imgs = b.column('image').to_pylist()
        for iid, img in zip(ids, imgs):
            if isinstance(img, dict):
                img = img.get('bytes')
            if iid not in out:
                out[iid] = img
    return out


def get_font(size: int):
    for path in [
        '/usr/share/fonts/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf',
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def fit_resize(img: Image.Image, w: int, h: int) -> Image.Image:
    """Resize keeping aspect ratio, center-crop to (w,h) for clean grid."""
    src_ratio = img.width / img.height
    tgt_ratio = w / h
    if src_ratio > tgt_ratio:
        # source is wider; scale to match height, crop sides
        new_h = h
        new_w = int(h * src_ratio)
    else:
        new_w = w
        new_h = int(w / src_ratio)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (new_w - w) // 2
    top = (new_h - h) // 2
    return img.crop((left, top, left + w, top + h))


def main():
    print('Loading column indices...')
    indices = []
    for label, path in COLUMNS:
        idx = load_arrow_index(path)
        indices.append(idx)
        print(f'  {label}: {len(idx)} ids')

    common = set(indices[0].keys())
    for idx in indices[1:]:
        common &= set(idx.keys())
    print(f'Common ids across all 4: {len(common)}')
    if len(common) < N_ROWS:
        raise SystemExit(f'not enough common ids: {len(common)}')

    rng = random.Random(SEED)
    pick = sorted(rng.sample(sorted(common), N_ROWS))
    print(f'Picked: {pick}')

    n_cols = len(COLUMNS)
    W = n_cols * CELL_W + (n_cols + 1) * PAD
    H = N_ROWS * CELL_H + (N_ROWS + 1) * PAD + LABEL_H
    grid = Image.new('RGB', (W, H), 'white')
    draw = ImageDraw.Draw(grid)

    # paste cells
    for r, iid in enumerate(pick):
        y = PAD + r * (CELL_H + PAD)
        for c, (_, _) in enumerate(COLUMNS):
            x = PAD + c * (CELL_W + PAD)
            data = indices[c].get(iid)
            img = Image.open(io.BytesIO(data)).convert('RGB')
            grid.paste(fit_resize(img, CELL_W, CELL_H), (x, y))

    # column labels at bottom (centered under each column)
    font = get_font(22)
    label_y = H - LABEL_H + 6
    for c, (label, _) in enumerate(COLUMNS):
        x_center = PAD + c * (CELL_W + PAD) + CELL_W // 2
        draw.text((x_center, label_y), label, fill='black', font=font, anchor='mt')

    out_path = OUT_DIR / 'figure2_conditions.png'
    grid.save(out_path, optimize=True)
    out_path_pdf = OUT_DIR / 'figure2_conditions.pdf'
    grid.save(out_path_pdf)
    print(f'-> {out_path}  ({W}x{H})')
    print(f'-> {out_path_pdf}')


if __name__ == '__main__':
    main()
