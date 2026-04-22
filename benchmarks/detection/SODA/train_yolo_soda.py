"""
YOLO Training & Evaluation for SODA Dataset (VOC Format)

3 Training configs:
  1. original              - Original trainval only
  2. original_weather_night - Original + Weather + Day2Night
  3. original_weather_night_small - Original + Weather + Day2Night + Small (outpainting)

4 Test subsets (no overlap with train):
  - original: 500 images from test.txt
  - weather:  500 held-out trainval images with weather augmentation (diverse styles)
  - night:    500 day2night versions of test.txt images
  - small:    50 held-out small (outpainted) images

Classes: 15 SODA classes
"""

import os
import sys
import json
import shutil
import random
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import torch
from ultralytics import YOLO

# ============================================================
# SODA 15 Classes
# ============================================================
CLASS_NAMES = [
    "person", "helmet", "vest", "hook", "fence",
    "board", "slogan", "rebar", "handcart", "ebox",
    "hopper", "wood", "scaffold", "brick", "cutter",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(os.environ.get('CONSYNTH_DATA_ROOT', Path.home() / 'consynth_data'))
VOC_ROOT = BASE_DIR / "SODA/data/SODA VOCdevkit/VOCdevkit/VOC2007"
AUG_DIR = BASE_DIR / "augmentation_data/SODA"
OUTPUT_DIR = BASE_DIR / "validation_data/downstream_detection/SODA"


def load_image_list(path):
    with open(path, encoding="latin-1") as f:
        return [l.strip() for l in f if l.strip()]


def parse_voc_xml(xml_path):
    """Parse VOC XML → list of (class_name, xmin, ymin, xmax, ymax) + image size."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.findtext("width"))
    h = int(size.findtext("height"))
    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in CLASS_TO_ID:
            continue
        bb = obj.find("bndbox")
        objects.append((
            name,
            int(float(bb.findtext("xmin"))),
            int(float(bb.findtext("ymin"))),
            int(float(bb.findtext("xmax"))),
            int(float(bb.findtext("ymax"))),
        ))
    return w, h, objects


def voc_to_yolo_label(xml_path):
    """Convert VOC XML to YOLO label lines."""
    w, h, objects = parse_voc_xml(xml_path)
    lines = []
    for name, xmin, ymin, xmax, ymax in objects:
        cid = CLASS_TO_ID[name]
        xc = ((xmin + xmax) / 2.0) / w
        yc = ((ymin + ymax) / 2.0) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        xc = max(0, min(1, xc))
        yc = max(0, min(1, yc))
        bw = max(0, min(1, bw))
        bh = max(0, min(1, bh))
        if bw > 0.001 and bh > 0.001:
            lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


# ============================================================
# Build helpers
# ============================================================

def add_voc_images(image_ids, images_dir, annotations_dir, out_images, out_labels,
                   prefix="", ext=".jpg"):
    """Link/copy VOC images+annotations into YOLO dirs. Returns count."""
    count = 0
    for img_id in tqdm(image_ids, desc=f"  {prefix or 'images'}", leave=False):
        img_file = images_dir / f"{img_id}{ext}"
        xml_file = annotations_dir / f"{img_id}.xml"
        if not img_file.exists():
            continue
        out_name = f"{prefix}_{img_id}" if prefix else img_id
        dst_img = out_images / f"{out_name}{ext}"
        dst_lbl = out_labels / f"{out_name}.txt"
        if dst_img.exists():
            count += 1
            continue
        try:
            os.link(str(img_file), str(dst_img))
        except FileExistsError:
            pass
        except OSError:
            try:
                shutil.copy2(str(img_file), str(dst_img))
            except shutil.SameFileError:
                pass
        if xml_file.exists():
            lines = voc_to_yolo_label(str(xml_file))
            dst_lbl.write_text("\n".join(lines))
        else:
            dst_lbl.write_text("")
        count += 1
    return count


def build_test_sets(held_out_ids, small_test_ids, output_base):
    """Build 4 test subsets."""
    test_ids = load_image_list(str(VOC_ROOT / "ImageSets/Main/test.txt"))
    random.seed(42)

    print("\n=== Building Test Sets ===")
    test_base = output_base / "test_sets"

    # --- 1. Original test: 500 from test.txt ---
    orig_ids = random.sample(test_ids, min(500, len(test_ids)))
    d = test_base / "original"
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "labels").mkdir(parents=True, exist_ok=True)
    n = add_voc_images(orig_ids, VOC_ROOT / "JPEGImages", VOC_ROOT / "Annotations",
                       d / "images", d / "labels", prefix="")
    print(f"  original test: {n} images")

    # --- 2. Night test: 500 day2night versions of test.txt images ---
    d2n_img = AUG_DIR / "day2night/images"
    d2n_ann = AUG_DIR / "day2night/Annotations"
    night_candidates = [i for i in test_ids if (d2n_img / f"{i}.jpg").exists()]
    night_ids = random.sample(night_candidates, min(500, len(night_candidates)))
    d = test_base / "night"
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "labels").mkdir(parents=True, exist_ok=True)
    n = add_voc_images(night_ids, d2n_img, d2n_ann, d / "images", d / "labels", prefix="")
    print(f"  night test: {n} images")

    # --- 3. Weather test: 500 held-out trainval w/ diverse weather styles ---
    weather_base = AUG_DIR / "weather"
    styles = sorted([s for s in os.listdir(str(weather_base))
                     if (weather_base / s / "JPEGImages").is_dir()])
    d = test_base / "weather"
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "labels").mkdir(parents=True, exist_ok=True)
    # Distribute held_out_ids across styles evenly
    weather_count = 0
    per_style = max(1, 500 // len(styles))
    random.shuffle(held_out_ids)
    idx = 0
    for si, style in enumerate(styles):
        take = per_style if si < len(styles) - 1 else 500 - weather_count
        take = min(take, len(held_out_ids) - idx)
        style_ids = held_out_ids[idx:idx + take]
        idx += take
        imgs_dir = weather_base / style / "JPEGImages"
        anns_dir = weather_base / style / "Annotations"
        n = add_voc_images(style_ids, imgs_dir, anns_dir,
                           d / "images", d / "labels", prefix=style)
        weather_count += n
        if weather_count >= 500:
            break
    print(f"  weather test: {weather_count} images (from {len(styles)} styles)")

    # --- 4. Small test: 50 held-out outpainted images ---
    small_img = AUG_DIR / "small/JPEGImages"
    small_ann = AUG_DIR / "small/Annotations"
    d = test_base / "small"
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "labels").mkdir(parents=True, exist_ok=True)
    n = add_voc_images(small_test_ids, small_img, small_ann,
                       d / "images", d / "labels", prefix="")
    print(f"  small test: {n} images")

    # Create dataset.yaml for each test set (val-only, reusing train path placeholder)
    for subset in ["original", "night", "weather", "small"]:
        yaml_path = test_base / subset / "dataset.yaml"
        yaml_content = f"""# SODA Test Set - {subset}
path: {test_base / subset}
train: images
val: images

names:
{chr(10).join(f'  {i}: {n}' for i, n in enumerate(CLASS_NAMES))}

nc: {NUM_CLASSES}
"""
        yaml_path.write_text(yaml_content)

    return {
        "original": len(orig_ids),
        "night": len(night_ids),
        "weather": weather_count,
        "small": len(small_test_ids),
    }


def build_train_val(config_name, train_ids, held_out_ids, small_train_ids, output_base):
    """Build YOLO train+val dataset for a config."""
    dataset_dir = output_base / f"yolo_data_{config_name}"
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    train_img = dataset_dir / "images/train"
    train_lbl = dataset_dir / "labels/train"
    val_img = dataset_dir / "images/val"
    val_lbl = dataset_dir / "labels/val"
    for d in [train_img, train_lbl, val_img, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Building dataset: {config_name} ===")

    # --- Training data ---
    # 1. Original
    n = add_voc_images(train_ids, VOC_ROOT / "JPEGImages", VOC_ROOT / "Annotations",
                       train_img, train_lbl, prefix="orig")
    print(f"  Original train: {n}")
    total = n

    if "weather" in config_name or "night" in config_name:
        # 2. Weather (all 6 styles, only train_ids)
        weather_base = AUG_DIR / "weather"
        for style in sorted(os.listdir(str(weather_base))):
            sd = weather_base / style
            if not (sd / "JPEGImages").is_dir():
                continue
            n = add_voc_images(train_ids, sd / "JPEGImages", sd / "Annotations",
                               train_img, train_lbl, prefix=style)
            print(f"  Weather {style}: {n}")
            total += n

        # 3. Day2Night (only train_ids)
        d2n_img = AUG_DIR / "day2night/images"
        d2n_ann = AUG_DIR / "day2night/Annotations"
        n = add_voc_images(train_ids, d2n_img, d2n_ann,
                           train_img, train_lbl, prefix="night")
        print(f"  Day2Night train: {n}")
        total += n

    if "small" in config_name:
        # 4. Small (outpainted) - only train portion
        n = add_voc_images(small_train_ids, AUG_DIR / "small/JPEGImages",
                           AUG_DIR / "small/Annotations",
                           train_img, train_lbl, prefix="small")
        print(f"  Small train: {n}")
        total += n

    print(f"  TOTAL train: {total}")

    # --- Validation data: 500 from test.txt (same as original test) ---
    test_ids = load_image_list(str(VOC_ROOT / "ImageSets/Main/test.txt"))
    random.seed(42)
    val_ids = random.sample(test_ids, min(500, len(test_ids)))
    nv = add_voc_images(val_ids, VOC_ROOT / "JPEGImages", VOC_ROOT / "Annotations",
                        val_img, val_lbl, prefix="")
    print(f"  Validation: {nv}")

    # dataset.yaml
    yaml_content = f"""# SODA YOLOv8 - {config_name}
path: {dataset_dir}
train: images/train
val: images/val

names:
{chr(10).join(f'  {i}: {n}' for i, n in enumerate(CLASS_NAMES))}

nc: {NUM_CLASSES}
"""
    (dataset_dir / "dataset.yaml").write_text(yaml_content)
    return dataset_dir, total, nv


def train_and_evaluate(config_name, dataset_dir, test_base, output_base,
                       epochs=50, batch_size=16, imgsz=640, model_size="s"):
    """Train YOLO and evaluate on all 4 test subsets."""
    yaml_path = dataset_dir / "dataset.yaml"

    print(f"\n{'='*60}")
    print(f"Training YOLOv8{model_size}: {config_name}")
    print(f"{'='*60}")

    model = YOLO(f"yolov8{model_size}.pt")
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        project=str(output_base / "runs"),
        name=config_name,
        exist_ok=True,
        verbose=True,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=4,
        patience=15,
        save=True,
        plots=True,
    )

    best_model = output_base / "runs" / config_name / "weights" / "best.pt"
    model = YOLO(str(best_model))

    # Evaluate on val
    val_results = model.val(data=str(yaml_path), split="val")
    all_metrics = {
        "config": config_name,
        "val": {
            "mAP50": float(val_results.box.map50),
            "mAP50-95": float(val_results.box.map),
            "precision": float(val_results.box.mp),
            "recall": float(val_results.box.mr),
        },
    }

    # Evaluate on each test subset
    for subset in ["original", "night", "weather", "small"]:
        test_yaml = test_base / subset / "dataset.yaml"
        if not test_yaml.exists():
            continue
        print(f"\n--- Evaluating on {subset} test ---")
        try:
            r = model.val(data=str(test_yaml), split="val")
            all_metrics[f"test_{subset}"] = {
                "mAP50": float(r.box.map50),
                "mAP50-95": float(r.box.map),
                "precision": float(r.box.mp),
                "recall": float(r.box.mr),
            }
            # Per-class
            for i, cn in enumerate(CLASS_NAMES):
                if i < len(r.box.ap50):
                    all_metrics[f"test_{subset}"][f"{cn}_AP50"] = float(r.box.ap50[i])
            print(f"  mAP50={all_metrics[f'test_{subset}']['mAP50']:.4f}  "
                  f"mAP50-95={all_metrics[f'test_{subset}']['mAP50-95']:.4f}")
        except Exception as e:
            print(f"  Error evaluating {subset}: {e}")
            all_metrics[f"test_{subset}"] = {"error": str(e)}

    # Save metrics
    metrics_file = output_base / f"metrics_{config_name}.json"
    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved: {metrics_file}")

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="YOLO Training for SODA")
    parser.add_argument("--config", required=True,
                        choices=["original", "original_weather_night",
                                 "original_weather_night_small"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model-size", type=str, default="s",
                        choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    print(f"Device: {'cuda - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print(f"Config: {args.config}")

    # --- Determine held-out IDs ---
    trainval_ids = load_image_list(str(VOC_ROOT / "ImageSets/Main/trainval.txt"))
    random.seed(42)

    # Find trainval IDs that exist in ALL 6 weather styles (for weather test)
    weather_base = AUG_DIR / "weather"
    styles = sorted([s for s in os.listdir(str(weather_base))
                     if (weather_base / s / "JPEGImages").is_dir()])
    style_sets = []
    for s in styles:
        d = weather_base / s / "JPEGImages"
        style_sets.append(set(os.path.splitext(f)[0] for f in os.listdir(str(d))))
    common_weather = style_sets[0]
    for ss in style_sets[1:]:
        common_weather &= ss

    # Pick 500 held-out for weather test from common_weather
    common_list = sorted(common_weather & set(trainval_ids))
    random.shuffle(common_list)
    held_out_ids = common_list[:500]
    held_out_set = set(held_out_ids)

    # Small: hold out last 50 for test
    small_all = sorted(os.path.splitext(f)[0] for f in
                       os.listdir(str(AUG_DIR / "small/JPEGImages")) if f.endswith(".jpg"))
    small_test_ids = small_all[-50:]
    small_train_ids = small_all[:-50]
    small_test_set = set(small_test_ids)

    # Train IDs = trainval minus held-out weather test IDs minus small test IDs
    train_ids = [i for i in trainval_ids if i not in held_out_set and i not in small_test_set]
    print(f"Trainval: {len(trainval_ids)}, Held-out weather: {len(held_out_ids)}, "
          f"Small test: {len(small_test_ids)}, Train: {len(train_ids)}")

    # --- Build test sets (only once, shared by all configs) ---
    test_base = output_base / "test_sets"
    lock_file = output_base / ".test_sets_building.lock"
    done_file = output_base / ".test_sets_done"
    if not done_file.exists():
        import fcntl, time
        try:
            fd = open(lock_file, "w")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # We got the lock, build test sets
            if not done_file.exists():
                build_test_sets(held_out_ids, small_test_ids, output_base)
                done_file.write_text("done")
            fcntl.flock(fd, fcntl.LOCK_UN)
            fd.close()
        except BlockingIOError:
            # Another job is building, wait for it
            print("Waiting for another job to finish building test sets...")
            while not done_file.exists():
                time.sleep(5)
            print("Test sets ready.")
    else:
        print("Test sets already exist, skipping build.")

    # --- Build training dataset ---
    dataset_dir, n_train, n_val = build_train_val(
        args.config, train_ids, held_out_ids, small_train_ids, output_base
    )

    # --- Train & Evaluate ---
    metrics = train_and_evaluate(
        config_name=args.config,
        dataset_dir=dataset_dir,
        test_base=test_base,
        output_base=output_base,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        model_size=args.model_size,
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.config}")
    print(f"{'='*60}")
    for key, val in metrics.items():
        if isinstance(val, dict) and "mAP50" in val:
            print(f"  {key:25s}  mAP50={val['mAP50']:.4f}  mAP50-95={val['mAP50-95']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
