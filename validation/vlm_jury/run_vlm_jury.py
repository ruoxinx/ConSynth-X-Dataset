#!/usr/bin/env python3
"""
VLM Jury evaluation: run one model on all conditions.

Following Ruck et al. (2026): 3 VLM judges independently assess whether
synthetic augmentations (1) show realistic weather and (2) preserve scene
semantics. Binary accept/reject decisions aggregated across judges.

Usage:
  python -m validation.vlm_jury.run_vlm_jury --model qwen2.5-vl-7b
  python -m validation.vlm_jury.run_vlm_jury --model internvl2.5-8b --resume
  python -m validation.vlm_jury.run_vlm_jury --model phi-4-multimodal
"""

import argparse
import json
import sys
import time
import os
from pathlib import Path
import os as _os
from pathlib import Path as _Path
_DATA_ROOT = _Path(_os.environ.get("CONSYNTH_DATA_ROOT", str(_Path.home() / "consynth_data")))
_REPO_ROOT = _Path(_os.environ.get("CONSYNTH_REPO_ROOT", str(_Path(__file__).resolve().parents[1])))
_BR_ROOT = _Path(_os.environ.get("CONSYNTH_BENCHMARK_RUNNER", str(_REPO_ROOT.parent / "Benchmark_runner")))

# Add Benchmark_runner to path for model imports
BENCHMARK_DIR = _BR_ROOT
sys.path.insert(0, str(BENCHMARK_DIR))

OUT_DIR = (_REPO_ROOT / "validation/results/vlm_jury")

# Model name → (registry_class, import_path)
MODEL_MAP = {
    "qwen2.5-vl-7b": ("Qwen25VL7B", "models.qwen2_vl"),
    "internvl2.5-8b": ("InternVL25_8B", "models.internvl"),
    "phi-4-multimodal": ("Phi4Multimodal", "models.phi4"),
    "qwen3-vl-8b": ("Qwen3VL8B", "models.qwen3_5"),
    "claude-sonnet-4.6": ("ClaudeSonnet46", "models.claude_api"),
}


def load_model(model_name: str):
    """Load a VLM model by name (direct import, no registry)."""
    if model_name not in MODEL_MAP:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_MAP.keys())}")

    class_name, module_path = MODEL_MAP[model_name]
    print(f"Loading {model_name} ({class_name}) from {module_path}...")

    # Direct import to avoid registry loading all models
    import importlib
    mod = importlib.import_module(module_path)
    model_cls = getattr(mod, class_name)
    model = model_cls()
    model.load()
    return model


def save_checkpoint(results: list, path: Path):
    """Save intermediate results."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_checkpoint(path: Path) -> list:
    """Load checkpoint if exists."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def run_evaluation(model, model_name: str, n_synthetic: int = 50,
                   n_acdc: int = 40, resume: bool = False,
                   dino_threshold: float | None = None,
                   conditions: list | None = None,
                   skip_acdc: bool = False):
    """Run VLM Jury evaluation for one model."""
    from validation.vlm_jury.data_loader import load_all_samples, concat_pair
    from validation.vlm_jury.prompts import (
        get_paired_prompt, get_baseline_prompt, SYSTEM_PROMPT
    )
    from validation.vlm_jury.result_parser import parse_response

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_dino{dino_threshold}" if dino_threshold is not None else ""
    checkpoint_path = out_dir / f"{model_name}{suffix}_checkpoint.json"
    result_path = out_dir / f"{model_name}{suffix}_results.json"

    # Load checkpoint
    results = load_checkpoint(checkpoint_path) if resume else []
    done_keys = {(r["condition"], r["image_id"]) for r in results}
    if results:
        print(f"Resumed: {len(results)} completed evaluations")

    # Load samples
    samples = load_all_samples(n_synthetic=n_synthetic, n_acdc=n_acdc,
                                dino_threshold=dino_threshold,
                                conditions=conditions,
                                skip_acdc=skip_acdc)

    # Filter already-done
    todo = [s for s in samples if (s.condition, s.image_id) not in done_keys]
    print(f"\nTo evaluate: {len(todo)} samples ({len(samples) - len(todo)} already done)")

    # Temporary directory for concatenated images
    import tempfile
    tmp_dir = Path(tempfile.mkdtemp(prefix="vlm_jury_"))

    start_time = time.time()
    for i, sample in enumerate(todo):
        # Prepare image
        if sample.eval_mode == "paired":
            # Concatenate original (left) + augmented (right)
            combined = concat_pair(sample.original_img, sample.augmented_img)
            prompt = get_paired_prompt(sample.condition)
        else:
            # Single ACDC image
            combined = sample.augmented_img
            prompt = get_baseline_prompt(sample.condition)

        # Save temp image (some models need file path)
        tmp_path = tmp_dir / f"eval_{i:04d}.jpg"
        combined.save(tmp_path, "JPEG", quality=90)

        # Run inference
        try:
            raw = model.generate(
                image=str(tmp_path),
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                max_new_tokens=256,
                temperature=0.0,
            )
        except Exception as e:
            raw = f"ERROR: {e}"

        # Parse response
        parsed = parse_response(raw)

        result = {
            "condition": sample.condition,
            "sample_idx": sample.sample_idx,
            "image_id": sample.image_id,
            "eval_mode": sample.eval_mode,
            "raw_response": raw,
            "explanation": parsed["explanation"],
            "decision": parsed["decision"],
            "parse_error": parsed["parse_error"],
        }
        results.append(result)

        # Progress
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (len(todo) - i - 1) / rate if rate > 0 else 0
        if (i + 1) % 10 == 0:
            n_true = sum(1 for r in results if r["decision"] is True)
            n_total = sum(1 for r in results if r["decision"] is not None)
            acc = n_true / n_total if n_total > 0 else 0
            print(f"  [{i+1}/{len(todo)}] acc={acc:.2%} | "
                  f"{rate:.1f} img/s | ETA {eta/60:.0f}m")
            save_checkpoint(results, checkpoint_path)

        # Clean temp
        tmp_path.unlink(missing_ok=True)

    # Final save
    save_checkpoint(results, result_path)
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {model_name}")
    print(f"{'='*60}")
    for cond in sorted(set(r["condition"] for r in results)):
        cond_results = [r for r in results if r["condition"] == cond]
        n = len(cond_results)
        n_true = sum(1 for r in cond_results if r["decision"] is True)
        n_err = sum(1 for r in cond_results if r["parse_error"])
        acc = n_true / n if n > 0 else 0
        print(f"  {cond:<20} {n_true:>3}/{n:>3} = {acc:.1%}  (parse_err={n_err})")

    total = len(results)
    total_true = sum(1 for r in results if r["decision"] is True)
    print(f"  {'TOTAL':<20} {total_true:>3}/{total:>3} = {total_true/total:.1%}")
    print(f"\nResults saved to: {result_path}")

    # Cleanup
    try:
        tmp_dir.rmdir()
    except OSError:
        pass


def main():
    parser = argparse.ArgumentParser(description="VLM Jury evaluation (single model)")
    parser.add_argument("--model", required=True, choices=list(MODEL_MAP.keys()),
                        help="Model to use as judge")
    parser.add_argument("--n-synthetic", type=int, default=50,
                        help="Samples per synthetic condition (default: 50)")
    parser.add_argument("--n-acdc", type=int, default=40,
                        help="ACDC baseline images per condition (default: 40)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--dino-threshold", type=float, default=None,
                        help="Restrict synthetic sampling to augmentations with "
                             "DINOv3 cosine similarity >= threshold (requires "
                             "validation/results/dino_ssim/{condition}.csv). "
                             "Output files get a _dino<t> suffix.")
    parser.add_argument("--conditions", type=str, default=None,
                        help="Comma-separated subset of SYNTHETIC_CONDITIONS to "
                             "evaluate (default: all).")
    parser.add_argument("--skip-acdc", action="store_true",
                        help="Skip ACDC baseline conditions.")
    args = parser.parse_args()

    conds = [c.strip() for c in args.conditions.split(',')] if args.conditions else None
    model = load_model(args.model)
    run_evaluation(model, args.model, args.n_synthetic, args.n_acdc,
                    args.resume, dino_threshold=args.dino_threshold,
                    conditions=conds, skip_acdc=args.skip_acdc)


if __name__ == "__main__":
    main()
