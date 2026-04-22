#!/usr/bin/env python3
"""
Full VLM Jury evaluation on all 3004 snow_strong augmented images.

Output per image: (image_id, dino_sim, ssim, judge_decision, judge_explanation)
Enables post-hoc analysis of VLM Jury acceptance vs DINO/SSIM thresholds.

Usage:
  python validation/vlm_jury/run_full_snow_strong.py --model qwen
  python validation/vlm_jury/run_full_snow_strong.py --model internvl
  python validation/vlm_jury/run_full_snow_strong.py --model phi4
"""

import argparse
import csv
import io
import json
import re
import sys
import tempfile
import time
from pathlib import Path
import os as _os
from pathlib import Path as _Path
_DATA_ROOT = _Path(_os.environ.get("CONSYNTH_DATA_ROOT", str(_Path.home() / "consynth_data")))
_REPO_ROOT = _Path(_os.environ.get("CONSYNTH_REPO_ROOT", str(_Path(__file__).resolve().parents[1])))
_BR_ROOT = _Path(_os.environ.get("CONSYNTH_BENCHMARK_RUNNER", str(_REPO_ROOT.parent / "Benchmark_runner")))

import pyarrow as pa
from PIL import Image

sys.path.insert(0, str(_BR_ROOT))

CSV_PATH = (_REPO_ROOT / "validation/results/snow_strong_dino_ssim.csv")
ORIG_PATH = (_DATA_ROOT / "augmentation_data_arrow/construction_site_test.arrow")
AUG_DIR = (_DATA_ROOT / "output/construction_site_test/diffusion_snow_strong")
OUT_DIR = (_REPO_ROOT / "validation/results/vlm_jury")

PROMPT = """Evaluate this side-by-side image pair of a construction site.

The LEFT half shows the ORIGINAL clear-weather image.
The RIGHT half shows the AUGMENTED image with synthetic snow effects applied.

Assess TWO criteria:

1. **Condition Realism**: Does the snow effect in the right image look realistic?
   Snow augmentation should show: white snow coverage on surfaces, falling snow particles, overcast grey sky, reduced color saturation.

2. **Semantic Preservation**: Apart from the weather change, is the scene content preserved?
   Objects, structures, spatial layout should remain the same.

Both criteria must be satisfied for a positive decision.

Respond with ONLY a JSON object (no other text):
{"explanation": "<brief 1-2 sentence reasoning>", "decision": true/false}"""

SYSTEM = "You are an expert image quality assessor evaluating synthetic weather augmentation for construction site images."

MODELS = {
    "qwen": ("models.qwen2_vl", "Qwen25VL7B"),
    "internvl": ("models.internvl", "InternVL25_8B"),
    "phi4": ("models.phi4", "Phi4Multimodal"),
}


def concat_pair(orig, aug, h=448):
    w1 = int(orig.width * h / orig.height)
    w2 = int(aug.width * h / aug.height)
    o = orig.resize((w1, h), Image.LANCZOS)
    a = aug.resize((w2, h), Image.LANCZOS)
    c = Image.new("RGB", (w1 + w2, h))
    c.paste(o, (0, 0))
    c.paste(a, (w1, 0))
    return c


def load_lookup(path):
    with open(path, "rb") as f:
        table = pa.ipc.open_stream(f).read_all()
    out = {}
    for i in range(len(table)):
        img_id = str(table.column("image_id")[i].as_py())
        b = table.column("image")[i].as_py()
        if isinstance(b, dict):
            b = b["bytes"]
        out[img_id] = Image.open(io.BytesIO(b)).convert("RGB")
    return out


def parse_decision(raw):
    m = re.search(r'"decision"\s*:\s*(true|false)', raw, re.IGNORECASE)
    if m:
        return m.group(1).lower() == "true"
    return None


def parse_explanation(raw):
    m = re.search(r'"explanation"\s*:\s*"([^"]{0,200})', raw)
    return m.group(1) if m else raw[:150]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional: limit to N samples (for testing)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"snow_strong_full_{args.model}.json"
    ckpt_path = OUT_DIR / f"snow_strong_full_{args.model}_checkpoint.json"

    # Load DINO/SSIM metadata
    meta = {}
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            meta[r["image_id"]] = {"ssim": float(r["ssim"]),
                                    "dino_sim": float(r["dino_sim"])}

    # Resume
    results = []
    done_ids = set()
    if args.resume and ckpt_path.exists():
        with open(ckpt_path) as f:
            results = json.load(f)
        done_ids = {r["image_id"] for r in results}
        print(f"Resumed: {len(results)} completed")

    # Load images
    print("Loading originals + augmented...")
    orig_lookup = load_lookup(ORIG_PATH)
    aug_lookup = {}
    for af in sorted(AUG_DIR.glob("batch_*.arrow")):
        aug_lookup.update(load_lookup(af))

    ids_to_eval = [i for i in sorted(meta.keys())
                   if i in aug_lookup and i in orig_lookup and i not in done_ids]
    if args.limit:
        ids_to_eval = ids_to_eval[:args.limit]
    print(f"To evaluate: {len(ids_to_eval)}")

    # Load model
    import importlib
    mod_name, cls_name = MODELS[args.model]
    print(f"Loading {args.model}...")
    mod = importlib.import_module(mod_name)
    model = getattr(mod, cls_name)()
    model.load()

    tmp_dir = Path(tempfile.mkdtemp())
    t0 = time.time()

    for i, img_id in enumerate(ids_to_eval):
        pair = concat_pair(orig_lookup[img_id], aug_lookup[img_id])
        tmp_path = tmp_dir / f"{img_id}.jpg"
        pair.save(tmp_path, "JPEG", quality=90)

        try:
            raw = model.generate(image=str(tmp_path), prompt=PROMPT,
                                 system_prompt=SYSTEM, max_new_tokens=256, temperature=0.0)
        except Exception as e:
            raw = f"ERROR: {e}"

        decision = parse_decision(raw)
        results.append({
            "image_id": img_id,
            "dino_sim": meta[img_id]["dino_sim"],
            "ssim": meta[img_id]["ssim"],
            "decision": decision,
            "explanation": parse_explanation(raw),
        })
        tmp_path.unlink(missing_ok=True)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(ids_to_eval) - i - 1) / rate
            acc = sum(1 for r in results if r["decision"] is True) / len(results)
            print(f"  [{i+1}/{len(ids_to_eval)}] acc={acc:.1%} | "
                  f"{rate:.1f}/s | ETA {eta/60:.0f}m")
            with open(ckpt_path, "w") as f:
                json.dump(results, f)

    # Final save
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    if ckpt_path.exists():
        ckpt_path.unlink()

    # Summary
    total = len(results)
    pass_n = sum(1 for r in results if r["decision"] is True)
    print(f"\n{'='*60}")
    print(f"SUMMARY ({args.model})")
    print(f"{'='*60}")
    print(f"  Total: {total}")
    print(f"  Accepted: {pass_n} ({pass_n/total:.1%})")

    # Breakdown by DINO threshold
    for thr in [0.7, 0.75, 0.8]:
        above = [r for r in results if r["dino_sim"] >= thr]
        below = [r for r in results if r["dino_sim"] < thr]
        a_pass = sum(1 for r in above if r["decision"] is True)
        b_pass = sum(1 for r in below if r["decision"] is True)
        print(f"\n  DINO >= {thr}:")
        print(f"    ABOVE ({len(above)}): {a_pass}/{len(above)} = "
              f"{a_pass/max(len(above),1):.1%}")
        print(f"    BELOW ({len(below)}): {b_pass}/{len(below)} = "
              f"{b_pass/max(len(below),1):.1%}")

    print(f"\nSaved: {out_path}")
    try:
        tmp_dir.rmdir()
    except Exception:
        pass


if __name__ == "__main__":
    main()
