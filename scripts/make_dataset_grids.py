"""For each dataset (construction_site, soda_voc, soda_ktsh) build a grid:
   rows = 3 sample images, cols = original + every condition.
"""
import io
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow as pa
import pyarrow.ipc as ipc
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/users/PGS0407/binben14/VietHuy/ConSynth-X')
OUT_DIR = ROOT / 'figures'
OUT_DIR.mkdir(exist_ok=True)

CELL = 220
HEADER_H = 36
LABEL_W = 0  # no row labels; image_id printed inside the original cell
PAD = 4
SEED = 7

# ---------------------------------------------------------------------------
# Dataset specs: ordered list of (label, source).
# Source is either ('arrow', path) or ('jpg-dir', dir).
# Original is first.
# ---------------------------------------------------------------------------

CS_TRAIN_ORIG = '/users/PGS0407/binben14/VietHuy/ConstructionSite/LouisChen15___construction_site/construction_site-train-00000-of-00002.arrow'
CS_TEST_ORIG = '/users/PGS0407/binben14/VietHuy/ConstructionSite/LouisChen15___construction_site/construction_site-test.arrow'

DATASETS = {
    'construction_site_train': [
        ('original',    ('arrow', CS_TRAIN_ORIG)),
        ('rain_light',  ('arrow', ROOT / 'augmentation_data/construction_site/rain_snow/diffusion/train/rain_light/rain_light.arrow')),
        ('rain_heavy',  ('arrow', ROOT / 'augmentation_data/construction_site/rain_snow/diffusion/train/rain_heavy/rain_heavy.arrow')),
        ('snow_light',  ('arrow', ROOT / 'augmentation_data/construction_site/rain_snow/diffusion/train/snow_light/snow_light.arrow')),
        ('snow_heavy',  ('arrow', ROOT / 'augmentation_data/construction_site/rain_snow/diffusion/train/snow_heavy/snow_heavy.arrow')),
        ('fog_light',   ('arrow-glob', ROOT / 'augmentation_data/construction_site/fog/diffusion/train/light')),
        ('fog_medium',  ('arrow-glob', ROOT / 'augmentation_data/construction_site/fog/diffusion/train/medium')),
        ('fog_heavy',   ('arrow-glob', ROOT / 'augmentation_data/construction_site/fog/diffusion/train/heavy')),
        ('rain_night',  ('arrow', ROOT / 'augmentation_data/construction_site/night_weather/rain_night/rain_night.arrow')),
        ('snow_night',  ('arrow', ROOT / 'augmentation_data/construction_site/night_weather/snow_night/snow_night.arrow')),
        ('small',       ('arrow', ROOT / 'augmentation_data/construction_site/small/train/small_constructionsite_train.arrow')),
    ],
    'construction_site_test': [
        ('original',    ('arrow', CS_TEST_ORIG)),
        ('rain_light',  ('arrow', ROOT / 'augmentation_data/construction_site/rain_snow/diffusion/test/rain_light/rain_light.arrow')),
        ('rain_heavy',  ('arrow', ROOT / 'augmentation_data/construction_site/rain_snow/diffusion/test/rain_heavy/rain_heavy.arrow')),
        ('snow_light',  ('arrow', ROOT / 'augmentation_data/construction_site/rain_snow/diffusion/test/snow_light/snow_light.arrow')),
        ('snow_heavy',  ('arrow', ROOT / 'augmentation_data/construction_site/rain_snow/diffusion/test/snow_heavy/snow_heavy.arrow')),
        ('fog_light',   ('arrow', ROOT / 'augmentation_data/construction_site/fog/diffusion/test/fog_light.arrow')),
        ('fog_medium',  ('arrow', ROOT / 'augmentation_data/construction_site/fog/diffusion/test/fog_medium.arrow')),
        ('fog_heavy',   ('arrow', ROOT / 'augmentation_data/construction_site/fog/diffusion/test/fog_heavy.arrow')),
        ('night',       ('arrow', ROOT / 'augmentation_data/construction_site/night/test/night_constructionsite_test.arrow')),
        ('small',       ('arrow', ROOT / 'augmentation_data/construction_site/small/test/small_constructionsite_test.arrow')),
    ],
    'soda_voc': [
        ('original',    ('arrow', ROOT / 'augmentation_data/soda_voc/original/soda_voc_original_first3000.arrow')),
        ('rain_light',  ('arrow', ROOT / 'augmentation_data/soda_voc/rain_snow/diffusion/rain_light.arrow')),
        ('rain_heavy',  ('arrow', ROOT / 'augmentation_data/soda_voc/rain_snow/diffusion/rain_heavy.arrow')),
        ('snow_light',  ('arrow', ROOT / 'augmentation_data/soda_voc/rain_snow/diffusion/snow_light.arrow')),
        ('snow_heavy',  ('arrow', ROOT / 'augmentation_data/soda_voc/rain_snow/diffusion/snow_heavy.arrow')),
        ('fog_light',   ('arrow', ROOT / 'augmentation_data/soda_voc/fog/test/fog_light.arrow')),
        ('fog_medium',  ('arrow', ROOT / 'augmentation_data/soda_voc/fog/test/fog_medium.arrow')),
        ('fog_heavy',   ('arrow', ROOT / 'augmentation_data/soda_voc/fog/test/fog_heavy.arrow')),
        ('day2night',   ('arrow', ROOT / 'augmentation_data/soda_voc/night/soda_day2night.arrow')),
        ('small',       ('arrow', ROOT / 'augmentation_data/soda_voc/small/soda_small.arrow')),
    ],
    'soda_ktsh': [
        ('original',    ('jpg-dir', '/users/PGS0407/binben14/VietHuy/ConstructionSite/SODA/data/soda-ktsh/images')),
        ('rain_light',  ('arrow', ROOT / 'augmentation_data/soda_ktsh/rain_snow/diffusion/rain_light.arrow')),
        ('rain_heavy',  ('arrow', ROOT / 'augmentation_data/soda_ktsh/rain_snow/diffusion/rain_heavy.arrow')),
        ('snow_light',  ('arrow', ROOT / 'augmentation_data/soda_ktsh/rain_snow/diffusion/snow_light.arrow')),
        ('snow_heavy',  ('arrow', ROOT / 'augmentation_data/soda_ktsh/rain_snow/diffusion/snow_heavy.arrow')),
        ('fog_light',   ('arrow', ROOT / 'augmentation_data/soda_ktsh/fog/fog_light.arrow')),
        ('fog_medium',  ('arrow', ROOT / 'augmentation_data/soda_ktsh/fog/fog_medium.arrow')),
        ('fog_heavy',   ('arrow', ROOT / 'augmentation_data/soda_ktsh/fog/fog_heavy.arrow')),
        ('day2night',   ('arrow', ROOT / 'augmentation_data/soda_ktsh/night/soda_ktsh_day2night.arrow')),
        ('small',       ('arrow', ROOT / 'augmentation_data/soda_ktsh/small/soda_ktsh_small.arrow')),
    ],
}


def load_arrow_index(path) -> Dict[str, bytes]:
    src = pa.memory_map(str(path), 'r')
    r = ipc.open_stream(src)
    out = {}
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


def load_jpg_index(jpg_dir) -> Dict[str, bytes]:
    out = {}
    for p in Path(jpg_dir).glob('*.jpg'):
        out[p.stem] = p  # store path; lazy-read on use
    return out


def get_image(source, image_id):
    kind, src = source
    if kind == 'arrow':
        idx = source._cache  # type: ignore
        b = idx.get(image_id)
        if b is None:
            return None
        return Image.open(io.BytesIO(b)).convert('RGB')
    if kind == 'jpg-dir':
        idx = source._cache  # type: ignore
        p = idx.get(image_id)
        if p is None:
            return None
        return Image.open(p).convert('RGB')
    return None


def square_resize(img: Image.Image, size: int) -> Image.Image:
    """Resize keeping aspect ratio, pad to square."""
    img = img.copy()
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (size, size), 'white')
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def get_font(size=14):
    for path in ['/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                 '/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf']:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def build_grid(name: str, spec: List[Tuple[str, Tuple[str, str]]], n_rows: int = 3):
    print(f'\n=== {name} ===')
    sources = []
    indices = []
    for label, src in spec:
        kind, path = src
        if kind == 'arrow':
            if not Path(path).exists():
                print(f'  [skip] {label}: missing {path}')
                continue
            idx = load_arrow_index(path)
        elif kind == 'arrow-glob':
            # path is a directory; merge all arrow files in it.
            d = Path(path)
            if not d.exists():
                print(f'  [skip] {label}: dir missing {d}')
                continue
            parts = sorted(d.glob('*.arrow'))
            if not parts:
                print(f'  [skip] {label}: no parts in {d}')
                continue
            idx = {}
            for p in parts:
                idx.update(load_arrow_index(p))
        else:
            idx = load_jpg_index(path)
        # attach for closure
        wrapper = (kind, path)
        # use a small object to bundle cache
        sources.append((label, wrapper, idx))
        print(f'  {label}: {len(idx)} ids')

    # Pick image_ids such that each fog level (if present) is covered by exactly
    # one row, and each row maxes coverage of the remaining (non-exclusive) conditions.
    fog_labels = [lab for lab, _, _ in sources if lab.startswith('fog_')]
    non_fog = [(lab, idx) for lab, _, idx in sources if not lab.startswith('fog_')]
    score_nonfog: Dict[str, int] = {}
    for _, idx in non_fog:
        for iid in idx.keys():
            score_nonfog[iid] = score_nonfog.get(iid, 0) + 1

    rng = random.Random(SEED)
    pick: List[str] = []
    if fog_labels and len(fog_labels) >= n_rows:
        # one row per fog level, ranked by non-fog coverage
        for lab in fog_labels[:n_rows]:
            fog_idx = next(idx for l, _, idx in sources if l == lab)
            ranked = sorted(fog_idx.keys(), key=lambda i: -score_nonfog.get(i, 0))
            for iid in ranked:
                if iid not in pick:
                    pick.append(iid); break
    else:
        # fallback: pure max-coverage pick
        score: Dict[str, int] = {}
        for _, _, idx in sources:
            for iid in idx.keys():
                score[iid] = score.get(iid, 0) + 1
        max_score = max(score.values())
        candidates = [iid for iid, s in score.items() if s == max_score]
        pick = sorted(rng.sample(candidates, min(n_rows, len(candidates))))
    print(f'  picked: {pick}')

    # Build grid
    n_cols = len(sources)
    W = n_cols * CELL + (n_cols + 1) * PAD
    H = HEADER_H + n_rows * CELL + (n_rows + 1) * PAD
    grid = Image.new('RGB', (W, H), 'white')
    draw = ImageDraw.Draw(grid)
    font = get_font(14)
    id_font = get_font(12)

    # Headers
    for c, (label, _, _) in enumerate(sources):
        x = PAD + c * (CELL + PAD)
        draw.text((x + CELL // 2, 4), label, fill='black', font=font, anchor='mt')

    # Cells
    for r, iid in enumerate(pick):
        y = HEADER_H + PAD + r * (CELL + PAD)
        for c, (label, src, idx) in enumerate(sources):
            x = PAD + c * (CELL + PAD)
            kind, path = src
            data = idx.get(iid)
            img = None
            if data is not None:
                if kind in ('arrow', 'arrow-glob'):
                    img = Image.open(io.BytesIO(data)).convert('RGB')
                else:
                    img = Image.open(data).convert('RGB')
            if img is None:
                # blank cell with label
                draw.rectangle((x, y, x + CELL, y + CELL), fill='#eeeeee', outline='#cccccc')
                draw.text((x + CELL // 2, y + CELL // 2), 'missing', fill='#888', font=id_font, anchor='mm')
            else:
                grid.paste(square_resize(img, CELL), (x, y))
        # image_id label on left side of original cell
        draw.text((PAD + 4, y + 4), iid, fill='red', font=id_font)

    out_path = OUT_DIR / f'grid_{name}.png'
    grid.save(out_path, optimize=True)
    print(f'  -> {out_path}  ({W}x{H})')


def main():
    import sys
    only = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    for name, spec in DATASETS.items():
        if only and name not in only:
            continue
        try:
            build_grid(name, spec)
        except Exception as e:
            print(f'[{name}] ERROR: {e}')
            import traceback; traceback.print_exc()


if __name__ == '__main__':
    main()
