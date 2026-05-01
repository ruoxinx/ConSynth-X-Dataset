#!/usr/bin/env python3
"""Re-run CycleGAN-Turbo day_to_night for a specific list of soda_ktsh image_ids
that were corrupted/empty in the original batch run, then patch them into the
existing soda_ktsh/night arrow shard.

Inputs:
  --ids-json: JSON list of image_ids to recover (e.g. ['Ktsh0005', 'Ktsh0006', ...])
  --source-images-dir: directory of clean ktsh JPGs (Ktsh0005.jpg etc.)
  --existing-arrow: current corrupted arrow at
      augmentation_data/soda_ktsh/night/soda_ktsh_day2night.arrow
  --output-arrow: path for repaired arrow (write-side)

Reuses the same model + transforms as `day2night_soda_worker.py` for parity
with the existing 1,135 successful rows.
"""
from __future__ import annotations
import argparse, io, json, sys
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from PIL import Image

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "img2img-turbo" / "src"))
from cyclegan_turbo import CycleGAN_Turbo  # noqa: E402
from my_utils.training_utils import build_transform  # noqa: E402


def load_existing_arrow(path: Path):
    with pa.OSFile(str(path), "rb") as f:
        try:
            r = ipc.open_stream(f)
            batches = list(r)
        except pa.ArrowInvalid:
            f.seek(0)
            r = ipc.open_file(f)
            batches = [r.get_batch(i) for i in range(r.num_record_batches)]
    if not batches:
        raise RuntimeError("empty arrow")
    return pa.Table.from_batches(batches), batches[0].schema


def is_corrupt(b) -> bool:
    if isinstance(b, dict):
        b = b.get("bytes", b"")
    if not b or len(b) < 4:
        return True
    return bytes(b[:3]) != b"\xff\xd8\xff"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-json", required=True)
    ap.add_argument("--source-images-dir", required=True)
    ap.add_argument("--existing-arrow", required=True)
    ap.add_argument("--output-arrow", required=True)
    ap.add_argument("--jpeg-quality", type=int, default=95)
    args = ap.parse_args()

    with open(args.ids_json) as f:
        target_ids = set(json.load(f))
    print(f"target ids to recover: {len(target_ids)}")

    src_dir = Path(args.source_images_dir)

    # Load model once
    print("loading CycleGAN-Turbo day_to_night ...")
    model = CycleGAN_Turbo(pretrained_name="day_to_night")
    model.eval()
    try:
        model.unet.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model.half()
    print(f"device: {device}")

    transform = build_transform("resize_512x512")

    # Generate replacements
    new_bytes_by_id: dict[str, bytes] = {}
    skipped: list[str] = []
    for i, iid in enumerate(sorted(target_ids)):
        src = src_dir / f"{iid}.jpg"
        if not src.exists():
            # try common variants
            cands = list(src_dir.glob(f"{iid}.[jJ][pP][gG]")) + list(src_dir.glob(f"{iid}.[jJ][pP][eE][gG]"))
            if not cands:
                skipped.append(iid)
                continue
            src = cands[0]

        with Image.open(src) as im:
            im = im.convert("RGB")
            orig_w, orig_h = im.size
            x = transform(im).unsqueeze(0).to(device)
            if device == "cuda":
                x = x.half()
            with torch.no_grad():
                out = model(x, direction="a2b", caption="day to night")
            out_pil = out[0].cpu().float().mul(0.5).add(0.5).clamp(0, 1).mul(255).permute(1, 2, 0).byte().numpy()
            out_im = Image.fromarray(out_pil).resize((orig_w, orig_h), Image.LANCZOS)
            buf = io.BytesIO()
            out_im.save(buf, format="JPEG", quality=args.jpeg_quality)
            new_bytes_by_id[iid] = buf.getvalue()

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(target_ids)}] processed; latest={iid}")

    print(f"\nGenerated {len(new_bytes_by_id)} replacement images. Skipped (no source): {len(skipped)}")
    if skipped:
        print(f"  examples skipped: {skipped[:5]}")

    # Patch into existing arrow
    print(f"\nPatching {args.existing_arrow} ...")
    table, schema = load_existing_arrow(Path(args.existing_arrow))
    rows = table.to_pylist()

    n_patched = 0
    n_left_corrupt = 0
    for row in rows:
        iid = row["image_id"]
        if iid in new_bytes_by_id:
            row["image"] = new_bytes_by_id[iid]
            n_patched += 1
        else:
            if is_corrupt(row.get("image")):
                n_left_corrupt += 1

    print(f"  patched: {n_patched}, still corrupt: {n_left_corrupt}, total rows: {len(rows)}")

    # Build new table with same schema (image is binary in new arrow; original schema may
    # have struct{bytes,path}. Check before assuming.)
    out_path = Path(args.output_arrow)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image_field = schema.field("image")
    if pa.types.is_struct(image_field.type):
        # Wrap bytes in {'bytes': ..., 'path': None}
        for row in rows:
            b = row["image"]
            if isinstance(b, (bytes, bytearray)):
                row["image"] = {"bytes": bytes(b), "path": None}
    out_table = pa.Table.from_pylist(rows, schema=schema)
    tmp = str(out_path) + ".tmp"
    with pa.OSFile(tmp, "wb") as sink, ipc.new_stream(sink, schema) as w:
        w.write_table(out_table)
    Path(tmp).replace(out_path)
    print(f"\nWrote {out_path}  rows={out_table.num_rows}")


if __name__ == "__main__":
    main()
