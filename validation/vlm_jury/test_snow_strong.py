#!/usr/bin/env python3
"""
Quick VLM Jury test on 20 strong snow samples.
Uses side-by-side (original | strong) images.
"""

import sys
import json
import time
import tempfile
from pathlib import Path
import os as _os
from pathlib import Path as _Path
_DATA_ROOT = _Path(_os.environ.get("CONSYNTH_DATA_ROOT", str(_Path.home() / "consynth_data")))
_REPO_ROOT = _Path(_os.environ.get("CONSYNTH_REPO_ROOT", str(_Path(__file__).resolve().parents[1])))
_BR_ROOT = _Path(_os.environ.get("CONSYNTH_BENCHMARK_RUNNER", str(_REPO_ROOT.parent / "Benchmark_runner")))

sys.path.insert(0, str(_BR_ROOT))

from PIL import Image

SNOW_DIR = (_REPO_ROOT / "validation/results/snow_intensity_test")
OUT_DIR = (_REPO_ROOT / "validation/results/vlm_jury")


def concat_pair(orig: Image.Image, aug: Image.Image, h: int = 448) -> Image.Image:
    w1 = int(orig.width * h / orig.height)
    w2 = int(aug.width * h / aug.height)
    o = orig.resize((w1, h), Image.LANCZOS)
    a = aug.resize((w2, h), Image.LANCZOS)
    combined = Image.new("RGB", (w1 + w2, h))
    combined.paste(o, (0, 0))
    combined.paste(a, (w1, 0))
    return combined


PROMPT = """Evaluate this side-by-side image pair of a construction site.

The LEFT half shows the ORIGINAL clear-weather image.
The RIGHT half shows the AUGMENTED image with synthetic snow effects applied.

Assess TWO criteria:

1. **Condition Realism**: Does the snow effect in the right image look realistic?
   Snow augmentation should show: white snow coverage on surfaces, falling snow particles, overcast grey sky, reduced color saturation, frost or ice effects on surfaces.

2. **Semantic Preservation**: Apart from the weather change, is the scene content preserved?
   Objects, structures, spatial layout should remain the same.

Both criteria must be satisfied for a positive decision.

Respond with ONLY a JSON object (no other text):
{"explanation": "<brief 1-2 sentence reasoning>", "decision": true/false}"""

SYSTEM = "You are an expert image quality assessor evaluating synthetic weather augmentation for construction site images."


def run_model(model_name, model_module, model_class):
    import importlib
    mod = importlib.import_module(model_module)
    cls = getattr(mod, model_class)
    model = cls()
    model.load()

    tmp_dir = Path(tempfile.mkdtemp())
    results = []

    for i in range(20):
        orig = Image.open(SNOW_DIR / "original" / f"{i:03d}.jpg").convert("RGB")
        aug = Image.open(SNOW_DIR / "strong" / f"{i:03d}.jpg").convert("RGB")
        combined = concat_pair(orig, aug)
        tmp_path = tmp_dir / f"{i:03d}.jpg"
        combined.save(tmp_path, "JPEG", quality=90)

        try:
            raw = model.generate(image=str(tmp_path), prompt=PROMPT,
                                 system_prompt=SYSTEM, max_new_tokens=256, temperature=0.0)
        except Exception as e:
            raw = f"ERROR: {e}"

        # Parse
        decision = None
        try:
            import re
            m = re.search(r'"decision"\s*:\s*(true|false)', raw, re.IGNORECASE)
            if m:
                decision = m.group(1).lower() == "true"
        except:
            pass

        results.append({"idx": i, "decision": decision, "raw": raw[:300]})
        status = "PASS" if decision else ("FAIL" if decision is False else "ERR")
        print(f"  [{i+1}/20] {status}")
        tmp_path.unlink(missing_ok=True)

    accepted = sum(1 for r in results if r["decision"] is True)
    print(f"\n  {model_name}: {accepted}/20 = {accepted/20:.0%}")

    out_path = OUT_DIR / f"snow_strong_test_{model_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    try:
        tmp_dir.rmdir()
    except:
        pass
    return accepted


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["qwen", "internvl", "phi4"])
    args = parser.parse_args()

    models = {
        "qwen": ("qwen", "models.qwen2_vl", "Qwen25VL7B"),
        "internvl": ("internvl", "models.internvl", "InternVL25_8B"),
        "phi4": ("phi4", "models.phi4", "Phi4Multimodal"),
    }

    name, mod, cls = models[args.model]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== VLM Jury Snow Strong Test: {name} ===")
    run_model(name, mod, cls)


if __name__ == "__main__":
    main()
