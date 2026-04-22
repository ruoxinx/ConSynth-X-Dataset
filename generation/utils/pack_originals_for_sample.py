#!/usr/bin/env python3
"""
Pack "original" (non-augmented) source images into augmentation_data_sample/,
filtered to the union of image_ids referenced in the existing cs_*/soda_*/soda_ktsh_*
sample arrows. This produces paired originals to compare against augmented samples.

Output:
  augmentation_data_sample/cs_original.arrow        (CS train+test, LouisChen15 schema)
  augmentation_data_sample/soda_voc_original.arrow  (SODA VOC JPEG + XML annotation)
  augmentation_data_sample/soda_ktsh_original.arrow (SODA-KTSH JPEG + decoded captions)
"""

import json
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

_GEN_ROOT = next(p for p in Path(__file__).resolve().parents if (p / '_paths.py').exists())
sys.path.insert(0, str(_GEN_ROOT))
from _paths import REPO_ROOT, ARROW_CS_DIR, SODA_VOC_ROOT, SODA_KTSH_IMG, SODA_KTSH_CAP

SAMPLE_DIR = REPO_ROOT / 'augmentation_data_sample'
CS_SRC_DIR = ARROW_CS_DIR
CS_ARROWS = [
    'construction_site-test.arrow',
    'construction_site-train-00000-of-00002.arrow',
    'construction_site-train-00001-of-00002.arrow',
]
VOC_ROOT = SODA_VOC_ROOT
KTSH_IMG_DIR = SODA_KTSH_IMG
KTSH_CAP_DIR = SODA_KTSH_CAP


def read_arrow(path):
    with pa.memory_map(str(path)) as mm:
        try:
            return ipc.open_stream(mm).read_all()
        except Exception:
            return ipc.open_file(mm).read_all()


def collect_ids_by_source():
    cs_ids, voc_ids, ktsh_ids = set(), set(), set()
    for f in sorted(os.listdir(SAMPLE_DIR)):
        if not f.endswith('.arrow'):
            continue
        t = read_arrow(SAMPLE_DIR / f)
        ids = t.column('image_id').to_pylist()
        if f.startswith('cs_'):
            cs_ids.update(ids)
        elif f.startswith('soda_voc_'):
            voc_ids.update(ids)
        elif f.startswith('soda_ktsh_'):
            ktsh_ids.update(ids)
    return cs_ids, voc_ids, ktsh_ids


def pack_cs_original(target_ids):
    tables = [read_arrow(CS_SRC_DIR / f) for f in CS_ARROWS]
    full = pa.concat_tables(tables)
    ids_col = full.column('image_id').to_pylist()
    mask = pa.array([i in target_ids for i in ids_col], type=pa.bool_())
    filtered = full.filter(mask)

    id_to_idx = {v: k for k, v in enumerate(filtered.column('image_id').to_pylist())}
    order = [id_to_idx[i] for i in sorted(target_ids) if i in id_to_idx]
    filtered = filtered.take(pa.array(order, type=pa.int64()))

    out = SAMPLE_DIR / 'cs_original.arrow'
    with ipc.new_stream(str(out), filtered.schema) as w:
        w.write_table(filtered)
    missing = [i for i in target_ids if i not in id_to_idx]
    print(f'[cs_original] rows={filtered.num_rows}/{len(target_ids)} missing={len(missing)} -> {out}')
    if missing:
        print(f'  first_missing={missing[:5]}')


def pack_soda_voc_original(target_ids):
    jpg_dir = VOC_ROOT / 'JPEGImages'
    xml_dir = VOC_ROOT / 'Annotations'
    ids, blobs, anns = [], [], []
    missing = []
    for stem in sorted(target_ids):
        img = jpg_dir / f'{stem}.jpg'
        xml = xml_dir / f'{stem}.xml'
        if not img.exists():
            missing.append(stem)
            continue
        with open(img, 'rb') as f:
            blobs.append(f.read())
        if xml.exists():
            with open(xml, 'r', encoding='utf-8') as f:
                anns.append(f.read())
        else:
            anns.append('')
        ids.append(stem)
    metas = [None] * len(ids)
    schema = pa.schema([
        pa.field('image_id', pa.string()),
        pa.field('image', pa.binary()),
        pa.field('annotation', pa.string()),
        pa.field('meta', pa.string()),
    ])
    table = pa.table({'image_id': ids, 'image': blobs, 'annotation': anns, 'meta': metas}, schema=schema)
    out = SAMPLE_DIR / 'soda_voc_original.arrow'
    with ipc.new_stream(str(out), table.schema) as w:
        w.write_table(table)
    print(f'[soda_voc_original] rows={table.num_rows}/{len(target_ids)} missing={len(missing)} -> {out}')


def pack_soda_ktsh_original(target_ids):
    with open(KTSH_CAP_DIR / 'WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json') as f:
        wordmap = json.load(f)
    rev_map = {v: k for k, v in wordmap.items()}

    all_captions = []
    for split in ['TRAIN', 'VAL', 'TEST']:
        with open(KTSH_CAP_DIR / f'{split}_CAPTIONS_flickr8k_5_cap_per_img_5_min_word_freq.json') as f:
            raw = json.load(f)
        for cap in raw:
            toks = [rev_map.get(t, '') for t in cap if t != 0]
            text = ' '.join(t for t in toks if t not in ('<start>', '<end>', '<pad>', ''))
            all_captions.append(text)

    orig_images = sorted(KTSH_IMG_DIR.glob('*.jpg'))
    n_caps_images = len(all_captions) // 5
    img2cap = {}
    for i, img in enumerate(orig_images[:n_caps_images]):
        img2cap[img.stem] = all_captions[i * 5:(i + 1) * 5]

    ids, blobs, fnames, caps, refs = [], [], [], [], []
    missing = []
    for stem in sorted(target_ids):
        img = KTSH_IMG_DIR / f'{stem}.jpg'
        if not img.exists():
            missing.append(stem)
            continue
        with open(img, 'rb') as f:
            blobs.append(f.read())
        ids.append(stem)
        fnames.append(img.name)
        caps.append(img2cap.get(stem, []))
        refs.append(stem)

    table = pa.table({
        'image': blobs,
        'image_id': ids,
        'filename': fnames,
        'captions': caps,
        'ref_id': refs,
    })
    out = SAMPLE_DIR / 'soda_ktsh_original.arrow'
    with ipc.new_stream(str(out), table.schema) as w:
        w.write_table(table)
    print(f'[soda_ktsh_original] rows={table.num_rows}/{len(target_ids)} missing={len(missing)} -> {out}')


def main():
    cs_ids, voc_ids, ktsh_ids = collect_ids_by_source()
    print(f'IDs to pack: cs={len(cs_ids)}, soda_voc={len(voc_ids)}, soda_ktsh={len(ktsh_ids)}')
    pack_cs_original(cs_ids)
    pack_soda_voc_original(voc_ids)
    pack_soda_ktsh_original(ktsh_ids)


if __name__ == '__main__':
    main()
