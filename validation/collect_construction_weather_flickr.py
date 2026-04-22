#!/usr/bin/env python3
"""
Collect construction site images under adverse weather conditions from Flickr.

Downloads Creative Commons licensed images to build a construction-specific
reference dataset for Relative Mahalanobis Distance evaluation, replacing
cross-domain ACDC (driving) references.

Target: ~200-300 images per condition (rain, snow, fog, night).
License: CC-BY, CC-BY-SA, CC-BY-ND, CC0 (research-compatible).

Usage:
  # First: set env vars
  export FLICKR_API_KEY="your_key"
  export FLICKR_API_SECRET="your_secret"

  # Run collection
  python validation/collect_construction_weather_flickr.py

  # Run for specific condition
  python validation/collect_construction_weather_flickr.py --condition rain

  # Dry run (no download)
  python validation/collect_construction_weather_flickr.py --dry-run
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.request import urlretrieve

import flickrapi

# ── Config ──────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "reference_data" / "construction_weather"

# Creative Commons licenses (research-compatible)
# 1=CC-BY-NC-SA, 2=CC-BY-NC, 3=CC-BY-NC-ND, 4=CC-BY, 5=CC-BY-SA, 6=CC-BY-ND,
# 9=CC0, 10=PDM
CC_LICENSES = "4,5,6,9,10"  # Only fully open licenses (no NC)

TARGET_PER_CONDITION = 300
MIN_SIZE = 640  # minimum dimension in pixels

# Search queries per condition — multiple queries to increase diversity
CONDITION_QUERIES = {
    "rain": [
        "construction site rain",
        "building construction rainy",
        "construction rain weather",
        "construction site wet rain",
        "building site rain storm",
        "crane rain construction",
        "road construction rain",
        "construction worker rain",
    ],
    "snow": [
        "construction site snow",
        "building construction snow",
        "construction site winter snow",
        "construction snow weather",
        "crane snow construction",
        "road construction snow",
        "building site snow winter",
        "construction worker snow",
    ],
    "fog": [
        "construction site fog",
        "building construction fog",
        "construction site foggy",
        "construction fog mist",
        "crane fog construction",
        "construction site haze",
        "building site misty fog",
        "road construction fog",
    ],
    "night": [
        "construction site night",
        "building construction night",
        "construction site dark night",
        "construction night lighting",
        "crane night construction",
        "road construction night",
        "construction site evening",
        "building site night lights",
    ],
}


def get_flickr_client():
    """Initialize Flickr API client from environment variables."""
    api_key = os.environ.get("FLICKR_API_KEY")
    api_secret = os.environ.get("FLICKR_API_SECRET")
    if not api_key or not api_secret:
        raise ValueError(
            "Set FLICKR_API_KEY and FLICKR_API_SECRET environment variables.\n"
            "Get keys at: https://www.flickr.com/services/apps/create/apply/"
        )
    return flickrapi.FlickrAPI(api_key, api_secret, format="parsed-json")


def search_photos(flickr, query, per_page=100, pages=3):
    """Search Flickr for photos matching query with CC license."""
    all_photos = []
    for page in range(1, pages + 1):
        try:
            result = flickr.photos.search(
                text=query,
                license=CC_LICENSES,
                media="photos",
                content_type=1,  # photos only
                sort="relevance",
                extras="url_l,url_c,url_z,url_o,license,owner_name,date_taken",
                per_page=per_page,
                page=page,
            )
            photos = result.get("photos", {}).get("photo", [])
            if not photos:
                break
            all_photos.extend(photos)
            time.sleep(1)  # rate limit
        except Exception as e:
            print(f"    API error on page {page}: {e}")
            time.sleep(3)
    return all_photos


def get_best_url(photo):
    """Get the best available URL (prefer url_l > url_c > url_z)."""
    for key in ["url_l", "url_c", "url_z", "url_o"]:
        url = photo.get(key)
        if url:
            return url, key
    return None, None


def download_image(url, save_path, timeout=30):
    """Download image with timeout."""
    try:
        urlretrieve(url, save_path)
        return True
    except Exception as e:
        print(f"    Download failed: {e}")
        return False


LICENSE_NAMES = {
    "0": "All Rights Reserved",
    "1": "CC-BY-NC-SA-2.0",
    "2": "CC-BY-NC-2.0",
    "3": "CC-BY-NC-ND-2.0",
    "4": "CC-BY-2.0",
    "5": "CC-BY-SA-2.0",
    "6": "CC-BY-ND-2.0",
    "9": "CC0-1.0",
    "10": "Public Domain Mark",
}


def collect_condition(flickr, condition, queries, out_dir, target, dry_run=False):
    """Collect images for a single weather condition."""
    cond_dir = out_dir / condition
    cond_dir.mkdir(parents=True, exist_ok=True)

    # Track downloaded photo IDs to avoid duplicates
    metadata_path = cond_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {"condition": condition, "images": {}, "queries_used": []}

    existing_ids = set(metadata["images"].keys())
    downloaded = len(existing_ids)
    print(f"\n{'='*60}")
    print(f"  Condition: {condition} | Existing: {downloaded} | Target: {target}")
    print(f"{'='*60}")

    if downloaded >= target:
        print(f"  Already have {downloaded} images, skipping.")
        return downloaded

    for query in queries:
        if downloaded >= target:
            break

        print(f"\n  Query: '{query}'")
        photos = search_photos(flickr, query)
        print(f"  Found {len(photos)} results")

        new_count = 0
        for photo in photos:
            if downloaded >= target:
                break

            photo_id = photo["id"]
            if photo_id in existing_ids:
                continue

            url, url_key = get_best_url(photo)
            if not url:
                continue

            # Generate filename
            ext = url.rsplit(".", 1)[-1].lower()
            if ext not in ("jpg", "jpeg", "png"):
                ext = "jpg"
            filename = f"{condition}_{photo_id}.{ext}"
            save_path = cond_dir / filename

            if dry_run:
                print(f"    [DRY RUN] Would download: {photo_id} ({url_key})")
                downloaded += 1
                existing_ids.add(photo_id)
                new_count += 1
                continue

            # Download
            if download_image(url, save_path):
                # Verify file size (skip tiny images)
                fsize = save_path.stat().st_size
                if fsize < 10_000:  # < 10KB probably broken
                    save_path.unlink()
                    continue

                license_id = str(photo.get("license", ""))
                metadata["images"][photo_id] = {
                    "filename": filename,
                    "url": url,
                    "url_size": url_key,
                    "license": LICENSE_NAMES.get(license_id, license_id),
                    "owner": photo.get("ownername", ""),
                    "title": photo.get("title", ""),
                    "date_taken": photo.get("datetaken", ""),
                    "flickr_url": f"https://www.flickr.com/photos/{photo.get('owner', '')}/{photo_id}",
                }
                existing_ids.add(photo_id)
                downloaded += 1
                new_count += 1

                if downloaded % 20 == 0:
                    print(f"    Progress: {downloaded}/{target}")

                time.sleep(0.5)  # rate limit

        if query not in metadata["queries_used"]:
            metadata["queries_used"].append(query)
        print(f"    New from this query: {new_count}")

    # Save metadata
    if not dry_run:
        metadata["total_images"] = len(metadata["images"])
        metadata["license_note"] = (
            "All images are Creative Commons licensed (CC-BY, CC-BY-SA, CC-BY-ND, or CC0). "
            "See individual image entries for specific licenses."
        )
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n  Final count for {condition}: {downloaded}")
    return downloaded


def main():
    parser = argparse.ArgumentParser(
        description="Collect construction weather images from Flickr (CC licensed)"
    )
    parser.add_argument("--condition", choices=list(CONDITION_QUERIES.keys()),
                        help="Collect only this condition (default: all)")
    parser.add_argument("--target", type=int, default=TARGET_PER_CONDITION,
                        help=f"Target images per condition (default: {TARGET_PER_CONDITION})")
    parser.add_argument("--output-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--dry-run", action="store_true",
                        help="Search but don't download")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Flickr Construction Weather Image Collector")
    print(f"Output: {out_dir}")
    print(f"Target: {args.target} images/condition")
    print(f"Licenses: {CC_LICENSES} (CC-BY, CC-BY-SA, CC-BY-ND, CC0, PDM)")

    flickr = get_flickr_client()

    conditions = [args.condition] if args.condition else list(CONDITION_QUERIES.keys())
    summary = {}

    for condition in conditions:
        queries = CONDITION_QUERIES[condition]
        count = collect_condition(flickr, condition, queries, out_dir, args.target, args.dry_run)
        summary[condition] = count

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total = 0
    for cond, count in summary.items():
        status = "OK" if count >= args.target else "BELOW TARGET"
        print(f"  {cond:<10} {count:>5} / {args.target}  [{status}]")
        total += count
    print(f"  {'TOTAL':<10} {total:>5}")
    print(f"\nData saved to: {out_dir}")
    print("Each condition folder contains metadata.json with license info.")


if __name__ == "__main__":
    main()
