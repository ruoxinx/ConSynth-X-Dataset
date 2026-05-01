#!/bin/bash
# Refresh the QC funnel report + figures after new DINO/SSIM CSVs land in
# validation/results/dino_ssim/.  Idempotent — safe to run any time.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "[$(date +%H:%M:%S)] rebuilding funnel counts ..."
python3 validation/build_qc_funnel.py
echo "[$(date +%H:%M:%S)] re-plotting figures ..."
python3 validation/plot_qc_funnel.py
echo "[$(date +%H:%M:%S)] done."
