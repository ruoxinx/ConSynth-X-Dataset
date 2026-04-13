# Infrastructure & Compute

---

## SLURM Cluster

- **Cluster**: OSC, account `pgs0407`, partition `batch`
- **Modules**: cuda/12.8.1, miniconda3/24.1.2-py310
- **Submit scripts**: tạo `.sh` files trong `jobs/`, logs vào `logs/`

## Conda Environments

| Env | Transformers | Dùng cho |
|---|---|---|
| `VLM` | 4.46.x | VLM inference (LLaVA, InternVL, Gemma, Phi) |
| `vlm-new` | 5.x | Qwen2.5-VL, Qwen3-VL (cần transformers mới) |

## GPU Requirements

| Model Size | GPU |
|---|---|
| 7-8B (Qwen2.5-7B, LLaVA-7B, InternVL-8B) | 1x A100 |
| 26-27B (InternVL-26B, Gemma-27B) | 3-4x A100 |
| 72B (Qwen2.5-72B) | 4x A100 |
| Generation pipelines (IP2P, FLUX, day2night) | 1x A100 |
| Sensitivity YOLOv8 training (6 configs) | 1x A100 per job, 8h, 32GB |

## Data Locations (External, not copied)

Dữ liệu nằm ngoài repo, tham chiếu bằng absolute path:

| Data | Path | Format |
|---|---|---|
| Construction Site images | `~/LMUData/` | HuggingFace Arrow |
| SODA dataset | `/users/PGS0407/binben14/VietHuy/ConstructionSite/SODA/` | Pascal VOC |
| Augmented data | `/users/PGS0407/binben14/VietHuy/ConstructionSite/augmentation_data/` | Arrow + images |
| Annotations | `/users/PGS0407/binben14/VietHuy/ConstructionSite-10k-Implementation/Annotations/` | — |
| Day2night checkpoint | `ConstructionSite/day2night/checkpoints/day2night.pkl` | PyTorch |
