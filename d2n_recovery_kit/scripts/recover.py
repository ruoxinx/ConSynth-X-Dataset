#!/usr/bin/env python3
"""Recover 365 ktsh day2night images using CycleGAN-Turbo (img2img-turbo).

Reads source JPGs from ../src_images/ (365 files), generates day-to-night
counterparts, writes them to ../out_images/ with the same filename.

Run:
    python recover.py

Requires: img2img-turbo cloned next to this script (./img2img-turbo/) OR on PYTHONPATH.
GPU recommended (CUDA). CPU works but slow (~5s/image GPU vs ~60s/image CPU).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT = HERE.parent
SRC_DIR = KIT / "src_images"
OUT_DIR = KIT / "out_images"
IDS_PATH = KIT / "ids.json"

# img2img-turbo path: clone from https://github.com/GaParmar/img2img-turbo
IMG2IMG_DIR = HERE / "img2img-turbo"
if not (IMG2IMG_DIR / "src").exists():
    print(f"ERROR: img2img-turbo missing at {IMG2IMG_DIR}")
    print("Clone it first:")
    print(f"  cd {HERE}")
    print(f"  git clone https://github.com/GaParmar/img2img-turbo")
    sys.exit(1)
sys.path.insert(0, str(IMG2IMG_DIR / "src"))

import torch
from PIL import Image
from cyclegan_turbo import CycleGAN_Turbo
from my_utils.training_utils import build_transform


def main():
    OUT_DIR.mkdir(exist_ok=True)
    ids = json.loads(IDS_PATH.read_text())
    print(f"target ids: {len(ids)}")

    # Skip ids whose output already exists (resume)
    todo = [i for i in ids if not (OUT_DIR / f"{i}.jpg").exists()]
    print(f"already done: {len(ids) - len(todo)}, remaining: {len(todo)}")
    if not todo:
        print("All done.")
        return

    print("Loading CycleGAN-Turbo day_to_night ...")
    model = CycleGAN_Turbo(pretrained_name="day_to_night")
    model.eval()
    try:
        model.unet.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")
    if device == "cuda":
        model.half()
    model.to(device)

    transform = build_transform("resize_512x512")
    t0 = time.time()
    fail = []
    for i, iid in enumerate(todo):
        src = SRC_DIR / f"{iid}.jpg"
        if not src.exists():
            cands = list(SRC_DIR.glob(f"{iid}.[jJ][pP][gG]"))
            if not cands:
                fail.append(iid)
                continue
            src = cands[0]

        try:
            with Image.open(src) as im:
                im = im.convert("RGB")
                W, H = im.size
                x = transform(im).unsqueeze(0).to(device)
                if device == "cuda":
                    x = x.half()
                with torch.no_grad():
                    out = model(x, direction="a2b", caption="day to night")
                arr = out[0].cpu().float().mul(0.5).add(0.5).clamp(0, 1).mul(255).permute(1, 2, 0).byte().numpy()
                out_im = Image.fromarray(arr).resize((W, H), Image.LANCZOS)
                out_im.save(OUT_DIR / f"{iid}.jpg", "JPEG", quality=95)
        except Exception as e:
            fail.append((iid, str(e)))

        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            avg = (time.time() - t0) / (i + 1)
            eta = avg * (len(todo) - i - 1)
            print(f"  [{i+1}/{len(todo)}] avg={avg:.1f}s/img  ETA={eta/60:.1f}m  latest={iid}")

    print(f"\nDone. {len(todo) - len(fail)} succeeded, {len(fail)} failed.")
    if fail:
        print(f"  failures: {fail[:10]}")
        (KIT / "failed_ids.json").write_text(json.dumps(fail, indent=2))
    print(f"\nOutputs: {OUT_DIR}")
    print(f"Next step: zip outputs and ship back to pitzer for arrow patching.")


if __name__ == "__main__":
    main()
