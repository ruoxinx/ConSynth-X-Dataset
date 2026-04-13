#!/bin/bash
#SBATCH --job-name=chk_wnet
#SBATCH --account=pgs0407
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --output=validation/logs/check_weathernet_%j.out
#SBATCH --error=validation/logs/check_weathernet_%j.err

module load miniconda3/24.1.2-py310
conda activate VLM
cd /users/PGS0407/binben14/VietHuy/ConSynth-X

python3 -c "
from datasets import load_dataset
ds = load_dataset('prithivMLmods/WeatherNet-05-18039', split='train', streaming=True)

# Get first example to see structure
ex = next(iter(ds))
print('Columns:', list(ex.keys()))
for k, v in ex.items():
    if hasattr(v, 'size'):
        print(f'  {k}: Image {v.size}')
    else:
        print(f'  {k}: {repr(v)}')

# Get label names from features
ds_full = load_dataset('prithivMLmods/WeatherNet-05-18039', split='train')
label_feat = ds_full.features.get('label')
print(f'\nLabel feature: {label_feat}')
if hasattr(label_feat, 'names'):
    print(f'Class names: {label_feat.names}')

# Count per class
from collections import Counter
labels = [ex['label'] for ex in ds_full]
counts = Counter(labels)
if hasattr(label_feat, 'names'):
    for idx, name in enumerate(label_feat.names):
        print(f'  {name}: {counts.get(idx, 0)}')
else:
    for k, v in sorted(counts.items()):
        print(f'  class {k}: {v}')
print(f'Total: {len(labels)}')
"
