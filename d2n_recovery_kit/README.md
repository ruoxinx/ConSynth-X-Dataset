# Day2Night ktsh Recovery Kit

Recovers the **365 corrupted images** in
`soda_ktsh/night/soda_ktsh_day2night.arrow` (upstream pipeline produced
empty/garbled bytes) by re-running CycleGAN-Turbo `day_to_night` on the
original SODA-KTSH source JPGs.

## Contents

```
d2n_recovery_kit/
├── README.md              ← this file
├── ids.json               ← list of 365 image_ids to recover
├── src_images/            ← 365 source JPGs (Ktsh*.jpg, ~225 MB total)
├── scripts/
│   └── recover.py         ← inference loop
├── requirements.txt
├── run.sh                 ← one-shot setup + run
└── out_images/            ← (created by recover.py) 365 night JPGs go here
```

## Run

GPU recommended (CUDA / Apple MPS works too — slower).

```bash
# Optional: fresh virtualenv
python3 -m venv .venv && source .venv/bin/activate

bash run.sh
```

`run.sh` does:
1. Clone `img2img-turbo` into `scripts/img2img-turbo/` (one-time)
2. `pip install -r requirements.txt`
3. Run `scripts/recover.py` — generates 365 night JPGs into `out_images/`

`recover.py` is **resumable**: re-running skips ids whose output already exists.
Per-image cost: ~3–8 s on RTX 4090 / A100, ~30 s on MPS (M-series Mac), ~60 s on CPU.

Total ETA: **~5 min on A100**, **~20 min on consumer GPU**, **~3–4 h on Mac MPS**.

## Send results back

```bash
cd /path/to/d2n_recovery_kit
zip -r out_images.zip out_images/
# upload out_images.zip back to pitzer (scp / Drive / Kaggle)
```

I will then patch the arrow (`soda_ktsh_day2night.arrow` → 1500/1500 valid rows),
re-stamp DINO, and re-push Kaggle/GDrive.

## Verify locally before shipping

```bash
ls out_images/ | wc -l        # expect 365
file out_images/Ktsh0005.jpg  # expect "JPEG image data, ..."
```

If `failed_ids.json` exists in the kit root, those ids could not be processed
locally — ship the rest, we'll handle the failed ones separately.
