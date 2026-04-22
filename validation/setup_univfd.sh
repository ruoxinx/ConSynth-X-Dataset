#!/bin/bash
# Setup UniversalFakeDetect (UnivFD) pretrained weights.
#
# Downloads fc_weights.pth from the official GitHub release.
# Repo: https://github.com/WisconsinAIVision/UniversalFakeDetect (MIT license)
#
# Usage:
#   bash validation/setup_univfd.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WEIGHTS_DIR="${SCRIPT_DIR}/weights"
WEIGHTS_FILE="${WEIGHTS_DIR}/fc_weights.pth"

mkdir -p "${WEIGHTS_DIR}"

if [ -f "${WEIGHTS_FILE}" ]; then
    echo "[OK] Weights already exist: ${WEIGHTS_FILE}"
    echo "     Size: $(du -h "${WEIGHTS_FILE}" | cut -f1)"
    exit 0
fi

echo "=================================================="
echo "Downloading UnivFD pretrained weights..."
echo "=================================================="
echo ""
echo "Source: UniversalFakeDetect (Ojha et al., CVPR 2023)"
echo "License: MIT"
echo ""

# The official repo hosts weights via Google Drive.
# Try direct download first, then give manual instructions.
GDRIVE_ID="1As_kZbGATjPHPhxNLnO8aTF3hQ6yjDVl"
GDRIVE_URL="https://drive.google.com/uc?export=download&id=${GDRIVE_ID}"

echo "Attempting download from Google Drive..."
if command -v gdown &> /dev/null; then
    gdown "${GDRIVE_ID}" -O "${WEIGHTS_FILE}"
elif command -v wget &> /dev/null; then
    wget --no-check-certificate "${GDRIVE_URL}" -O "${WEIGHTS_FILE}" 2>/dev/null || true
fi

if [ -f "${WEIGHTS_FILE}" ] && [ -s "${WEIGHTS_FILE}" ]; then
    echo ""
    echo "[OK] Downloaded: ${WEIGHTS_FILE}"
    echo "     Size: $(du -h "${WEIGHTS_FILE}" | cut -f1)"
else
    echo ""
    echo "============================================"
    echo "AUTO-DOWNLOAD FAILED — Manual steps:"
    echo "============================================"
    echo ""
    echo "1. Clone the UnivFD repo:"
    echo "   git clone https://github.com/WisconsinAIVision/UniversalFakeDetect.git /tmp/UnivFD"
    echo ""
    echo "2. Download pretrained_weights/fc_weights.pth following their README"
    echo "   (Google Drive link in their repo)"
    echo ""
    echo "3. Copy the weights file:"
    echo "   cp /tmp/UnivFD/pretrained_weights/fc_weights.pth ${WEIGHTS_FILE}"
    echo ""
    echo "OR run without weights (CLIP zero-shot fallback):"
    echo "   python validation/run_realism_validation.py --mode clip_zeroshot"
    echo ""
    # Clean up empty file
    rm -f "${WEIGHTS_FILE}"
    exit 1
fi

# Verify CLIP dependency
echo ""
echo "Checking CLIP dependency..."
python -c "import clip; print(f'CLIP OK: {clip.__file__}')" 2>/dev/null || {
    echo "CLIP not installed. Install with:"
    echo "  pip install git+https://github.com/openai/CLIP.git"
    exit 1
}

echo ""
echo "Setup complete! Run validation with:"
echo "  python validation/run_realism_validation.py --weights ${WEIGHTS_FILE}"
