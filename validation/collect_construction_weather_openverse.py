#!/usr/bin/env python3
"""
Collect construction site images under adverse weather from Openverse API.

Openverse indexes 800M+ Creative Commons images. No API key required.
Downloads CC-licensed images to build a construction-specific reference
dataset for Relative Mahalanobis Distance evaluation.

Target: ~200-300 images per condition (rain, snow, fog, night).

Usage:
  python validation/collect_construction_weather_openverse.py
  python validation/collect_construction_weather_openverse.py --condition rain
  python validation/collect_construction_weather_openverse.py --dry-run
  python validation/collect_construction_weather_openverse.py --target 200
"""

import argparse
import json
import time
from pathlib import Path
from urllib.request import urlretrieve, Request, urlopen
from urllib.error import HTTPError, URLError

# ── Config ──────────────────────────────────────────────────────
API_BASE = "https://api.openverse.org/v1"
OUT_DIR = Path(__file__).parent / "reference_data" / "construction_weather"

TARGET_PER_CONDITION = 300
PAGE_SIZE = 20  # Openverse max per page

# CC licenses suitable for research (no NC restriction)
LICENSES = "by,by-sa,cc0,pdm,by-nd"

# Multiple search queries per condition for diversity
CONDITION_QUERIES = {
    "rain": [
        '"construction site" rain weather',
        '"construction site" rain',
        '"building site" rain',
        '"building construction" rain',
        '"road construction" rain',
        '"construction work" rain',
        'construction crane rain wet',
        'scaffolding rain construction',
        'excavator rain site',
        '"construction zone" rain',
    ],
    "snow": [
        '"construction site" snow weather',
        '"construction site" snow',
        '"building site" snow winter',
        '"building construction" snow',
        '"road construction" snow',
        '"construction work" snow',
        'construction crane snow winter',
        'scaffolding snow construction',
        'excavator snow site',
        '"construction zone" snow',
    ],
    "fog": [
        '"construction site" fog',
        '"construction site" foggy',
        '"construction site" mist',
        '"building site" fog',
        '"building construction" fog',
        'construction crane fog mist',
        'scaffolding fog construction',
        '"road construction" fog',
        '"construction zone" fog',
        '"construction site" haze morning',
    ],
    "night": [
        '"construction site" night',
        '"construction site" dark',
        '"building site" night',
        '"building construction" night',
        '"road construction" night',
        'construction crane night lights',
        'construction floodlight night',
        '"construction work" night',
        '"construction zone" night',
        'scaffolding night construction lights',
    ],
}


def api_search(query, page=1, page_size=PAGE_SIZE):
    """Search Openverse API for CC-licensed images."""
    import urllib.parse
    params = urllib.parse.urlencode({
        "q": query,
        "license": LICENSES,
        "page": page,
        "page_size": page_size,
    })
    url = f"{API_BASE}/images/?{params}"

    req = Request(url, headers={"User-Agent": "ConSynth-X/1.0 (research; vduong1@kent.edu)"})
    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        return data
    except HTTPError as e:
        if e.code == 429:
            print(f"    Rate limited, waiting 60s...")
            time.sleep(60)
            return api_search(query, page, page_size)
        print(f"    API error {e.code}: {e.reason}")
        return None
    except (URLError, Exception) as e:
        print(f"    Request error: {e}")
        return None


def download_image(url, save_path, timeout=30):
    """Download image file."""
    try:
        req = Request(url, headers={"User-Agent": "ConSynth-X/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            with open(save_path, "wb") as f:
                f.write(resp.read())
        return True
    except Exception as e:
        print(f"    Download failed ({type(e).__name__}): {url[:80]}")
        return False


def is_valid_image(path, min_size_kb=10):
    """Check if downloaded file is a valid image above minimum size."""
    if not path.exists():
        return False
    size_kb = path.stat().st_size / 1024
    if size_kb < min_size_kb:
        path.unlink()
        return False
    # Quick header check
    try:
        with open(path, "rb") as f:
            header = f.read(4)
        # JPEG, PNG, or WebP
        if header[:2] == b'\xff\xd8':  # JPEG
            return True
        if header[:4] == b'\x89PNG':  # PNG
            return True
        if header[:4] == b'RIFF':  # WebP
            return True
        path.unlink()
        return False
    except Exception:
        path.unlink()
        return False


def collect_condition(condition, queries, out_dir, target, dry_run=False):
    """Collect images for one weather condition."""
    cond_dir = out_dir / condition
    cond_dir.mkdir(parents=True, exist_ok=True)

    # Load existing metadata
    metadata_path = cond_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {"condition": condition, "images": {}, "queries_used": []}

    existing_ids = set(metadata["images"].keys())
    downloaded = len(existing_ids)

    print(f"\n{'='*60}")
    print(f"  {condition.upper()} | Existing: {downloaded} | Target: {target}")
    print(f"{'='*60}")

    if downloaded >= target:
        print(f"  Already have {downloaded} images, skipping.")
        return downloaded

    for query in queries:
        if downloaded >= target:
            break

        print(f"\n  Query: '{query}'")

        # Paginate through results
        for page in range(1, 13):  # max 12 pages
            if downloaded >= target:
                break

            result = api_search(query, page=page)
            if not result:
                break

            images = result.get("results", [])
            if not images:
                break

            total = result.get("result_count", 0)
            if page == 1:
                print(f"    Found {total} results, fetching pages...")

            new_this_page = 0
            for img in images:
                if downloaded >= target:
                    break

                img_id = img.get("id", "")
                if img_id in existing_ids:
                    continue

                img_url = img.get("url", "")
                if not img_url:
                    continue

                # Skip very small images
                w = img.get("width") or 0
                h = img.get("height") or 0
                if w > 0 and h > 0 and max(w, h) < 400:
                    continue

                if dry_run:
                    title = img.get("title", "untitled")[:50]
                    lic = img.get("license", "?")
                    print(f"    [DRY] {img_id[:8]} | {lic} | {w}x{h} | {title}")
                    downloaded += 1
                    existing_ids.add(img_id)
                    new_this_page += 1
                    continue

                # Determine extension
                filetype = img.get("filetype", "")
                if filetype in ("jpg", "jpeg"):
                    ext = "jpg"
                elif filetype == "png":
                    ext = "png"
                else:
                    ext = img_url.rsplit(".", 1)[-1].lower()[:4]
                    if ext not in ("jpg", "jpeg", "png", "webp"):
                        ext = "jpg"

                filename = f"{condition}_{img_id}.{ext}"
                save_path = cond_dir / filename

                if download_image(img_url, save_path):
                    if not is_valid_image(save_path):
                        continue

                    metadata["images"][img_id] = {
                        "filename": filename,
                        "source_url": img_url,
                        "source_page": img.get("foreign_landing_url", ""),
                        "license": img.get("license", ""),
                        "license_version": img.get("license_version", ""),
                        "license_url": img.get("license_url", ""),
                        "creator": img.get("creator", ""),
                        "creator_url": img.get("creator_url", ""),
                        "title": img.get("title", ""),
                        "provider": img.get("provider", ""),
                        "width": w,
                        "height": h,
                        "query": query,
                    }
                    existing_ids.add(img_id)
                    downloaded += 1
                    new_this_page += 1

                    if downloaded % 25 == 0:
                        print(f"    Progress: {downloaded}/{target}")
                        # Save intermediate metadata
                        _save_metadata(metadata, metadata_path)

                time.sleep(0.3)  # polite rate limiting

            if new_this_page == 0 and page > 1:
                break  # no new images, move to next query

            time.sleep(1)  # between pages

        if query not in metadata["queries_used"]:
            metadata["queries_used"].append(query)

    # Final save
    if not dry_run:
        _save_metadata(metadata, metadata_path)

    print(f"\n  Final count for {condition}: {downloaded}")
    return downloaded


def _save_metadata(metadata, path):
    """Save metadata JSON."""
    metadata["total_images"] = len(metadata["images"])
    metadata["license_note"] = (
        "All images are Creative Commons licensed (CC-BY, CC-BY-SA, CC-BY-ND, CC0, or PDM). "
        "See individual image entries for specific licenses and attribution requirements."
    )
    metadata["source"] = "Openverse API (https://api.openverse.org)"
    metadata["collected_for"] = (
        "ConSynth-X: construction-specific weather reference for "
        "Relative Mahalanobis Distance evaluation"
    )
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def verify_images(out_dir):
    """Post-collection verification: check all downloaded images."""
    from PIL import Image

    print(f"\n{'='*60}")
    print("POST-COLLECTION VERIFICATION")
    print(f"{'='*60}")

    for cond_dir in sorted(out_dir.iterdir()):
        if not cond_dir.is_dir():
            continue
        condition = cond_dir.name
        valid = 0
        invalid = 0
        sizes = []

        for img_path in cond_dir.iterdir():
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                continue
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
                    if min(w, h) < 200:
                        img_path.unlink()
                        invalid += 1
                    else:
                        valid += 1
                        sizes.append((w, h))
            except Exception:
                img_path.unlink()
                invalid += 1

        if sizes:
            avg_w = sum(s[0] for s in sizes) / len(sizes)
            avg_h = sum(s[1] for s in sizes) / len(sizes)
            print(f"  {condition:<10} valid={valid:>4}  removed={invalid:>3}  avg={avg_w:.0f}x{avg_h:.0f}")
        else:
            print(f"  {condition:<10} no images")


def main():
    parser = argparse.ArgumentParser(
        description="Collect construction weather images from Openverse (CC licensed, no API key)"
    )
    parser.add_argument("--condition", choices=list(CONDITION_QUERIES.keys()),
                        help="Collect only this condition")
    parser.add_argument("--target", type=int, default=TARGET_PER_CONDITION,
                        help=f"Target images per condition (default: {TARGET_PER_CONDITION})")
    parser.add_argument("--output-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--dry-run", action="store_true",
                        help="Search but don't download")
    parser.add_argument("--verify", action="store_true",
                        help="Verify downloaded images (requires Pillow)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Openverse Construction Weather Image Collector")
    print("=" * 60)
    print(f"Output:   {out_dir}")
    print(f"Target:   {args.target} images/condition")
    print(f"License:  {LICENSES}")
    print(f"API:      {API_BASE} (no key required)")

    if args.verify:
        verify_images(out_dir)
        return

    conditions = [args.condition] if args.condition else list(CONDITION_QUERIES.keys())
    summary = {}

    for condition in conditions:
        queries = CONDITION_QUERIES[condition]
        count = collect_condition(condition, queries, out_dir, args.target, args.dry_run)
        summary[condition] = count

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total = 0
    for cond, count in summary.items():
        status = "OK" if count >= args.target else f"NEED {args.target - count} MORE"
        print(f"  {cond:<10} {count:>5} / {args.target}  [{status}]")
        total += count
    print(f"  {'TOTAL':<10} {total:>5}")
    print(f"\nData: {out_dir}")
    print(f"Each folder has metadata.json with license/attribution info.")
    print(f"\nNext steps:")
    print(f"  1. python {__file__} --verify  (check image quality)")
    print(f"  2. Manual review: remove non-construction images")
    print(f"  3. Update compute_relative_mahalanobis.py to use this reference")


if __name__ == "__main__":
    main()
