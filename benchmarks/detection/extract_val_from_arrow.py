"""
Extract validation data from Arrow file to create separate val dataset.
"""

import os
import json
from pathlib import Path
from datasets import Dataset
from tqdm import tqdm
from PIL import Image


def extract_validation_data(
    arrow_file: str, 
    output_dir: str, 
    num_samples: int = 2000,
    seed: int = 42
):
    """
    Extract validation data from Arrow file.
    
    Args:
        arrow_file: Path to the Arrow file
        output_dir: Output directory for extracted data
        num_samples: Number of samples to extract
        seed: Random seed for reproducibility
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    annotations_dir = output_dir / "annotations"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading Arrow file: {arrow_file}")
    ds = Dataset.from_file(arrow_file)
    print(f"Total samples in file: {len(ds)}")
    
    # Shuffle and select samples
    ds_shuffled = ds.shuffle(seed=seed)
    actual_samples = min(num_samples, len(ds_shuffled))
    print(f"Extracting {actual_samples} samples...")
    
    success_count = 0
    
    for idx in tqdm(range(actual_samples), desc="Extracting"):
        try:
            sample = ds_shuffled[idx]
            image_id = sample.get("image_id", f"{idx:07d}")
            
            # Save image
            image = sample["image"]
            if image.mode == "RGBA":
                image = image.convert("RGB")
            
            image_path = images_dir / f"{image_id}.jpg"
            image.save(image_path, "JPEG", quality=95)
            
            # Build annotation dictionary
            annotation = {
                "image_id": image_id,
                "ref_idx": idx,
                "image_caption": sample.get("image_caption", ""),
                "illumination": sample.get("illumination", ""),
                "camera_distance": sample.get("camera_distance", ""),
                "view": sample.get("view", ""),
                "quality_of_info": sample.get("quality_of_info", ""),
                "rule_1_violation": sample.get("rule_1_violation"),
                "rule_2_violation": sample.get("rule_2_violation"),
                "rule_3_violation": sample.get("rule_3_violation"),
                "rule_4_violation": sample.get("rule_4_violation"),
                "excavator": sample.get("excavator", []),
                "rebar": sample.get("rebar", []),
                "worker_with_white_hard_hat": sample.get("worker_with_white_hard_hat", []),
            }
            
            annotation_path = annotations_dir / f"{image_id}.json"
            with open(annotation_path, "w") as f:
                json.dump(annotation, f, indent=2)
            
            success_count += 1
            
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            continue
    
    print(f"\nExtraction complete!")
    print(f"  Total extracted: {success_count}")
    print(f"  Images saved to: {images_dir}")
    print(f"  Annotations saved to: {annotations_dir}")
    
    return success_count


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract validation data from Arrow file')
    parser.add_argument('--arrow-file', type=str, 
                        default="/users/PGS0407/binben14/VietHuy/ConstructionSite/LouisChen15___construction_site/construction_site-train-00000-of-00002.arrow",
                        help='Path to Arrow file')
    parser.add_argument('--output-dir', type=str,
                        default="/users/PGS0407/binben14/VietHuy/ConstructionSite/augmentation_data/val_from_train",
                        help='Output directory')
    parser.add_argument('--num-samples', type=int, default=2000,
                        help='Number of samples to extract')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    extract_validation_data(
        arrow_file=args.arrow_file,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        seed=args.seed
    )
