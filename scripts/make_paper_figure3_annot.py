"""Build paper-style Figure 3 — annotation showcase.

2x2 panels showing SODA-VOC samples under 4 conditions (Fog / Night / Snow / Rain),
each with bounding-box overlays parsed from the VOC XML annotation column.
"""
import io
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow as pa
import pyarrow.ipc as ipc
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/users/PGS0407/binben14/VietHuy/ConSynth-X')
OUT_DIR = ROOT / 'paper'
OUT_DIR.mkdir(exist_ok=True)

PANELS: List[Tuple[str, str]] = [
    ('Fog',   str(ROOT / 'augmentation_data/soda_voc/fog/test/fog_medium.arrow')),
    ('Night', str(ROOT / 'augmentation_data/soda_voc/night/soda_day2night.arrow')),
    ('Snow',  str(ROOT / 'augmentation_data/soda_voc/rain_snow/diffusion/snow_light.arrow')),
    ('Rain',  str(ROOT / 'augmentation_data/soda_voc/rain_snow/diffusion/rain_light.arrow')),
]

# tab10 / tab20-style palette, RGB
PALETTE = {
    'person':     (255, 99, 71),     # tomato
    'helmet':     (60, 179, 113),    # mediumseagreen
    'vest':       (30, 144, 255),    # dodgerblue
    'slogan':     (255, 215, 0),     # gold
    'hook':       (138, 43, 226),    # blueviolet
    'scaffold':   (0, 206, 209),     # darkturquoise
    'board':      (255, 140, 0),     # darkorange
    'brick':      (210, 105, 30),    # chocolate
    'hopper':     (255, 105, 180),   # hotpink
    'handcart':   (50, 205, 50),     # limegreen
    'rebar':      (220, 20, 60),     # crimson
    'fence':      (147, 112, 219),   # mediumpurple
    'wood':       (160, 82, 45),     # sienna
    'ebox':       (0, 191, 255),     # deepskyblue
    'cutter':     (255, 0, 255),     # magenta
}
DEFAULT_COLOR = (200, 200, 200)

# Layout
PANEL_W, PANEL_H = 640, 400
PAD = 8
LABEL_H = 36
SEED = 23
BBOX_ALPHA = 90  # 0-255


def iter_arrow(path: str, max_rows: int = 200):
    """Yield up to max_rows (image_id, image_bytes, annotation_xml). Lazy: stops early."""
    src = pa.memory_map(path, 'r'); r = ipc.open_stream(src)
    n = 0
    for b in r:
        cols = b.column_names
        if 'image_id' not in cols or 'image' not in cols or 'annotation' not in cols:
            continue
        ids = b.column('image_id').to_pylist()
        anns = b.column('annotation').to_pylist()
        # Only fetch image bytes lazily; first scan annotations to find rows with enough bboxes.
        # Yield (iid, lazy-image-getter, ann) where lazy-image-getter returns bytes.
        for i, (iid, ann) in enumerate(zip(ids, anns)):
            if not ann:
                continue
            img_data = b.column('image')[i].as_py()
            if isinstance(img_data, dict):
                img_data = img_data.get('bytes')
            if not img_data:
                continue
            yield iid, img_data, ann
            n += 1
            if n >= max_rows:
                return


def parse_bboxes(xml_str: str):
    out = []
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return out, (0, 0)
    size = root.find('size')
    src_w = int(size.findtext('width', '0')) if size is not None else 0
    src_h = int(size.findtext('height', '0')) if size is not None else 0
    for o in root.findall('object'):
        name = o.findtext('name', 'object')
        b = o.find('bndbox')
        if b is None:
            continue
        try:
            x1 = float(b.findtext('xmin'))
            y1 = float(b.findtext('ymin'))
            x2 = float(b.findtext('xmax'))
            y2 = float(b.findtext('ymax'))
        except (TypeError, ValueError):
            continue
        out.append((name, x1, y1, x2, y2))
    return out, (src_w, src_h)


def get_font(size: int):
    for path in [
        '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf',
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def fit_resize(img: Image.Image, w: int, h: int):
    """Resize keeping aspect, center-crop to (w,h). Returns image and crop transform.

    Returns (resized_img, scale_x, scale_y, offset_x, offset_y) where bbox in
    original space (orig_w, orig_h) maps to:  x' = x*scale - offset_x
    """
    src_w, src_h = img.width, img.height
    src_ratio = src_w / src_h
    tgt_ratio = w / h
    if src_ratio > tgt_ratio:
        new_h = h
        new_w = int(h * src_ratio)
        scale = h / src_h
    else:
        new_w = w
        new_h = int(w / src_ratio)
        scale = w / src_w
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    off_x = (new_w - w) // 2
    off_y = (new_h - h) // 2
    cropped = img.crop((off_x, off_y, off_x + w, off_y + h))
    return cropped, scale, off_x, off_y


def draw_bboxes(panel: Image.Image, bboxes, src_size, scale, off_x, off_y):
    """Draw filled semi-transparent bboxes + outline + label."""
    overlay = Image.new('RGBA', panel.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    font = get_font(14)
    for name, x1, y1, x2, y2 in bboxes:
        # original->resized->cropped
        rx1 = x1 * scale - off_x
        ry1 = y1 * scale - off_y
        rx2 = x2 * scale - off_x
        ry2 = y2 * scale - off_y
        # clamp to panel
        rx1 = max(0, min(panel.width, rx1))
        ry1 = max(0, min(panel.height, ry1))
        rx2 = max(0, min(panel.width, rx2))
        ry2 = max(0, min(panel.height, ry2))
        if rx2 - rx1 < 4 or ry2 - ry1 < 4:
            continue
        color = PALETTE.get(name, DEFAULT_COLOR)
        fill = color + (BBOX_ALPHA,)
        outline = color + (255,)
        od.rectangle((rx1, ry1, rx2, ry2), fill=fill, outline=outline, width=2)
    return Image.alpha_composite(panel.convert('RGBA'), overlay).convert('RGB')


def pick_sample(path: str, rng, min_boxes=4):
    """Stream rows; pick first with >= min_boxes (after random offset for variety)."""
    skip = rng.randrange(0, 30)
    fallback = None
    for k, (iid, img, ann) in enumerate(iter_arrow(path, max_rows=200)):
        if k < skip:
            continue
        bboxes, src = parse_bboxes(ann)
        if len(bboxes) >= min_boxes:
            return iid, img, ann, bboxes, src
        if fallback is None and bboxes:
            fallback = (iid, img, ann, bboxes, src)
    return fallback


def main():
    rng = random.Random(SEED)
    panels_imgs = []
    for label, path in PANELS:
        print(f'{label}: streaming {path}')
        sel = pick_sample(path, rng, min_boxes=4)
        if sel is None:
            print(f'  WARN no good sample for {label}')
            continue
        iid, img_bytes, ann, bboxes, src_size = sel
        print(f'  picked {iid}, {len(bboxes)} bboxes, src={src_size}')
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        # if VOC XML width/height is 0 or inconsistent, fall back to actual image size
        if src_size[0] == 0 or src_size[1] == 0:
            src_size = img.size
        # rescale src->actual ratio if differs (e.g. augmented img resized)
        scale_to_actual = (img.width / src_size[0], img.height / src_size[1])
        # apply to bboxes
        bboxes = [(n,
                   x1 * scale_to_actual[0],
                   y1 * scale_to_actual[1],
                   x2 * scale_to_actual[0],
                   y2 * scale_to_actual[1]) for (n, x1, y1, x2, y2) in bboxes]
        panel, scale, off_x, off_y = fit_resize(img, PANEL_W, PANEL_H)
        panel = draw_bboxes(panel, bboxes, img.size, scale, off_x, off_y)
        panels_imgs.append((label, panel))

    # 2x2 grid
    cols, rows = 2, 2
    W = cols * PANEL_W + (cols + 1) * PAD
    H = rows * PANEL_H + (rows + 1) * PAD + LABEL_H
    grid = Image.new('RGB', (W, H), 'white')
    draw = ImageDraw.Draw(grid)
    for i, (label, panel) in enumerate(panels_imgs):
        r = i // cols; c = i % cols
        x = PAD + c * (PANEL_W + PAD)
        y = PAD + r * (PANEL_H + PAD)
        grid.paste(panel, (x, y))

    # column labels at bottom
    font = get_font(22)
    label_y = H - LABEL_H + 6
    for c in range(cols):
        labels_in_col = [PANELS[c][0], PANELS[c + cols][0]] if len(PANELS) > c + cols else [PANELS[c][0]]
        # the screenshot shows just one label per column at the bottom; pick the column's label
        # Here we'll instead label PER PANEL inside the panel's bottom-left.
        pass

    # Instead, draw label at bottom CENTER of each panel
    font_label = get_font(20)
    for i, (label, _) in enumerate(panels_imgs):
        r = i // cols; c = i % cols
        x = PAD + c * (PANEL_W + PAD)
        y = PAD + r * (PANEL_H + PAD)
        # bottom-left tag with semi-transparent bg
        text_w = font_label.getlength(label)
        tag_w = int(text_w) + 16
        tag_h = 28
        tag = Image.new('RGBA', (tag_w, tag_h), (0, 0, 0, 160))
        td = ImageDraw.Draw(tag)
        td.text((tag_w // 2, tag_h // 2), label, fill=(255, 255, 255), font=font_label, anchor='mm')
        grid.paste(tag, (x + 8, y + PANEL_H - tag_h - 8), tag)

    # legend at bottom
    legend_y = H - LABEL_H + 4
    cats_used = set()
    for _, panel in panels_imgs:
        pass
    legend_font = get_font(14)
    # collect categories actually used
    # (simpler: just write all palette entries)
    x = PAD
    for cat, color in PALETTE.items():
        sw = 16
        draw.rectangle((x, legend_y + 4, x + sw, legend_y + sw + 4), fill=color, outline=(0, 0, 0))
        draw.text((x + sw + 4, legend_y + 6), cat, fill=(0, 0, 0), font=legend_font)
        x += sw + 4 + int(legend_font.getlength(cat)) + 12
        if x > W - 80:
            break

    out_path = OUT_DIR / 'figure3_annotations.png'
    grid.save(out_path, optimize=True)
    out_path_pdf = OUT_DIR / 'figure3_annotations.pdf'
    grid.save(out_path_pdf)
    print(f'-> {out_path}  ({W}x{H})')
    print(f'-> {out_path_pdf}')


if __name__ == '__main__':
    main()
