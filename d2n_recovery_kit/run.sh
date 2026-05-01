#!/bin/bash
# One-shot setup + run for the day2night ktsh recovery kit.
# Idempotent: skips clone/install if already done; recover.py resumes per-id.
set -e
cd "$(dirname "$0")"

# 1) Clone img2img-turbo if missing
if [ ! -d scripts/img2img-turbo ]; then
  echo "[1/3] cloning img2img-turbo ..."
  git clone --depth=1 https://github.com/GaParmar/img2img-turbo scripts/img2img-turbo
else
  echo "[1/3] img2img-turbo already present, skip"
fi

# 2) Install Python deps (in current env — recommend a fresh venv first)
echo "[2/3] installing requirements ..."
pip install -q -r requirements.txt

# 3) Run recovery
echo "[3/3] running recovery ..."
python3 scripts/recover.py

echo
echo "When done, zip up out_images/ and send back so I can patch the arrow:"
echo "  cd $(pwd) && zip -r out_images.zip out_images/"
