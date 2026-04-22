#!/usr/bin/env python3
"""
Validate DINO threshold 0.75 against VLM Jury.

Takes same 10 samples from the grid (5 ABOVE DINO 0.75, 5 BELOW) and runs
them through a VLM judge. Hypothesis: jury should accept mostly ABOVE samples
and reject mostly BELOW samples.

Usage:
  python validation/vlm_jury/test_dino_threshold_vs_jury.py --model qwen
  python validation/vlm_jury/test_dino_threshold_vs_jury.py --model internvl
  python validation/vlm_jury/test_dino_threshold_vs_jury.py --model phi4
"""

import argparse
import csv
import io
import json
import random
import re
import sys
import tempfile
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

THRESHOLD = 0.75
N_EACH = 5
SEED = 42  # same as grid

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


MODELS = {
    "qwen": ("models.qwen2_vl", "Qwen25VL7B"),
    "internvl": ("models.internvl", "InternVL25_8B"),
    "phi4": ("models.phi4", "Phi4Multimodal"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS))
    args = parser.parse_args()

    # Load CSV and pick same samples as grid
    rows = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            rows.append({"id": r["image_id"],
                         "ssim": float(r["ssim"]),
                         "dino": float(r["dino_sim"])})

    random.seed(SEED)
    above = [r for r in rows if r["dino"] >= THRESHOLD]
    below = [r for r in rows if r["dino"] < THRESHOLD]
    sample_above = random.sample(above, N_EACH)
    sample_below = random.sample(below, N_EACH)

    all_samples = [(s, "ABOVE") for s in sample_above] + [(s, "BELOW") for s in sample_below]

    # Load images
    print("Loading originals + augmented...")
    orig_lookup = load_lookup(ORIG_PATH)
    aug_lookup = {}
    for af in sorted(AUG_DIR.glob("batch_*.arrow")):
        aug_lookup.update(load_lookup(af))

    # Load model
    import importlib
    mod_name, cls_name = MODELS[args.model]
    print(f"Loading {args.model} ({cls_name})...")
    mod = importlib.import_module(mod_name)
    model = getattr(mod, cls_name)()
    model.load()

    # Run evaluation
    tmp_dir = Path(tempfile.mkdtemp())
    results = []
    print(f"\n{'Group':<7} {'ID':<9} {'DINO':<7} {'Decision':<10} {'Explanation'}")
    print("-" * 100)

    for sample, group in all_samples:
        img_id = sample["id"]
        pair = concat_pair(orig_lookup[img_id], aug_lookup[img_id])
        tmp_path = tmp_dir / f"{img_id}.jpg"
        pair.save(tmp_path, "JPEG", quality=90)

        try:
            raw = model.generate(image=str(tmp_path), prompt=PROMPT,
                                 system_prompt=SYSTEM, max_new_tokens=256, temperature=0.0)
        except Exception as e:
            raw = f"ERROR: {e}"

        decision = parse_decision(raw)
        # Extract explanation from JSON
        expl_match = re.search(r'"explanation"\s*:\s*"([^"]{0,150})', raw)
        expl = expl_match.group(1) if expl_match else raw[:80]

        results.append({
            "group": group, "id": img_id, "dino": sample["dino"],
            "decision": decision, "raw": raw[:300],
        })
        dec_str = "PASS" if decision else ("FAIL" if decision is False else "ERR")
        print(f"{group:<7} {img_id:<9} {sample['dino']:.3f}  {dec_str:<10} {expl[:80]}")
        tmp_path.unlink(missing_ok=True)

    # Summary
    above_pass = sum(1 for r in results if r["group"] == "ABOVE" and r["decision"] is True)
    below_pass = sum(1 for r in results if r["group"] == "BELOW" and r["decision"] is True)

    print(f"\n{'='*60}")
    print(f"SUMMARY ({args.model})")
    print(f"{'='*60}")
    print(f"  ABOVE DINO {THRESHOLD}: {above_pass}/{N_EACH} accepted")
    print(f"  BELOW DINO {THRESHOLD}: {below_pass}/{N_EACH} accepted")
    print(f"  Agreement with DINO: "
          f"{(above_pass + (N_EACH - below_pass)) / (2*N_EACH):.0%}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"dino_threshold_jury_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    try:
        tmp_dir.rmdir()
    except Exception:
        pass


if __name__ == "__main__":
    main()
