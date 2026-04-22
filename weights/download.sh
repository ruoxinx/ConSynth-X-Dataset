#!/usr/bin/env bash
# Fetch pretrained weights from their ORIGINAL upstream hosts.
#
# These weights are NOT produced by ConSynth-X; they are redistributed
# from third parties under their respective licenses (see weights/README.md).
# The script just automates the download the upstream authors already publish.
#
# Files fetched:
#   - day2night.pkl        (1.6 GB) — CMU img2img-turbo host (public HTTPS URL)
#   - rain_vgg_512.pth     (535 MB) } Weather_Effect_Generator Google Drive folder
#   - snow_vgg_512.pth     (535 MB) }   (also contains fog_vgg_512.pth — ignored)
#
# Optional env overrides (for cluster mirrors, offline hosts):
#   CONSYNTH_WEIGHTS_SRC   local directory holding the 3 files
#   CONSYNTH_WEIGHTS_URL   override base URL (files must be named as above)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Upstream sources ------------------------------------------------------
DAY2NIGHT_URL="https://www.cs.cmu.edu/~img2img-turbo/models/day2night.pkl"
VGG_GDRIVE_FOLDER="https://drive.google.com/drive/folders/1MEVMLVhrv4t7efwAfCSk13yie8G-XcIB"
VGG_FILES=(rain_vgg_512.pth snow_vgg_512.pth)

# ---- Helpers ---------------------------------------------------------------
have_cmd() { command -v "$1" >/dev/null 2>&1; }

copy_from_mirror_if_set() {
  local f="$1"
  if [[ -n "${CONSYNTH_WEIGHTS_SRC:-}" && -f "$CONSYNTH_WEIGHTS_SRC/$f" ]]; then
    echo "[copy] $CONSYNTH_WEIGHTS_SRC/$f -> $f"
    cp "$CONSYNTH_WEIGHTS_SRC/$f" "$f"
    return 0
  fi
  return 1
}

fetch_override_if_set() {
  local f="$1"
  if [[ -n "${CONSYNTH_WEIGHTS_URL:-}" ]]; then
    echo "[curl] $CONSYNTH_WEIGHTS_URL/$f -> $f"
    curl -L --fail --retry 3 -o "$f" "$CONSYNTH_WEIGHTS_URL/$f"
    return 0
  fi
  return 1
}

# ---- day2night.pkl ---------------------------------------------------------
if [[ ! -f day2night.pkl ]]; then
  copy_from_mirror_if_set "day2night.pkl" \
    || fetch_override_if_set "day2night.pkl" \
    || { echo "[curl] $DAY2NIGHT_URL -> day2night.pkl"; curl -L --fail --retry 3 -o day2night.pkl "$DAY2NIGHT_URL"; }
else
  echo "[skip] day2night.pkl already exists"
fi

# ---- VGG weights (Google Drive folder) ------------------------------------
need_vgg=0
for f in "${VGG_FILES[@]}"; do
  [[ -f "$f" ]] || need_vgg=1
done

if (( need_vgg )); then
  # First try mirror / override for each file
  for f in "${VGG_FILES[@]}"; do
    [[ -f "$f" ]] && continue
    copy_from_mirror_if_set "$f" || fetch_override_if_set "$f" || true
  done
  # Any still missing → fall back to gdown folder download
  still_missing=0
  for f in "${VGG_FILES[@]}"; do
    [[ -f "$f" ]] || still_missing=1
  done
  if (( still_missing )); then
    if ! have_cmd gdown; then
      cat >&2 <<EOF
ERROR: gdown is not installed. Install with:
  pip install gdown
Or download manually from:
  $VGG_GDRIVE_FOLDER
and place rain_vgg_512.pth and snow_vgg_512.pth into:
  $SCRIPT_DIR/
EOF
      exit 1
    fi
    echo "[gdown] folder $VGG_GDRIVE_FOLDER -> weather_effect_generator_weights/"
    rm -rf weather_effect_generator_weights
    gdown --folder "$VGG_GDRIVE_FOLDER" -O weather_effect_generator_weights
    for f in "${VGG_FILES[@]}"; do
      if [[ ! -f "$f" ]]; then
        # Find the file in the downloaded folder (any subdirectory)
        found=$(find weather_effect_generator_weights -type f -name "$f" | head -1 || true)
        if [[ -z "$found" ]]; then
          echo "ERROR: gdown succeeded but $f not found in download." >&2
          echo "Folder listing:" >&2
          find weather_effect_generator_weights -type f >&2 || true
          exit 1
        fi
        mv "$found" "$f"
      fi
    done
    rm -rf weather_effect_generator_weights
  fi
else
  echo "[skip] rain_vgg_512.pth and snow_vgg_512.pth already exist"
fi

# ---- Verify + symlink ------------------------------------------------------
echo
echo "Verifying SHA256…"
sha256sum -c checksums.sha256

echo
echo "Linking into pipeline-expected locations…"
D2N_DIR="$SCRIPT_DIR/../generation/day2night/checkpoints"
VGG_DIR="$SCRIPT_DIR/../generation/weather/libs/Weather_Effect_Generator/VGG"
mkdir -p "$D2N_DIR" "$VGG_DIR"
ln -sf "$SCRIPT_DIR/day2night.pkl"    "$D2N_DIR/day2night.pkl"
ln -sf "$SCRIPT_DIR/rain_vgg_512.pth" "$VGG_DIR/rain_vgg_512.pth"
ln -sf "$SCRIPT_DIR/snow_vgg_512.pth" "$VGG_DIR/snow_vgg_512.pth"

echo "Done."
