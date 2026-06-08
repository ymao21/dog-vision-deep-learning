# Results

## Multi-Task Classification

| Task | Classes | Held-Out Test Accuracy |
|---|---:|---:|
| Breed | 120 | 86.1% |
| Breed Group | 8 | 94.1% |
| Size | 5 | 91.5% |
| Coat Length | 3 | 95.1% |
| Coat Color | image-derived pseudo-labels | 71.3% |

Training configuration:

| Setting | Value |
|---|---:|
| Epochs | 8 |
| Backbone learning rate | 1e-4 |
| Head learning rate | 1e-3 |
| Best validation breed accuracy | 87.2% |
| Training time | about 25.3 minutes |

## Dog Replacement Ablation Results

The replacement benchmark is a small controlled pilot over three image pairs. Scores are normalized within each comparison group, so they are best interpreted as relative rankings rather than absolute production metrics.

| Group | Method | Score | Runtime | Boundary MAE | LPIPS | Background MAE |
|---|---|---:|---:|---:|---:|---:|
| Segmentation | SAM using YOLO bbox prior | 0.738 | 3.20 | 44.57 | 0.2347 | 2.05 |
| Segmentation | YOLO + SAM consensus mask | 0.582 | 3.43 | 42.61 | 0.3022 | 2.54 |
| Segmentation | YOLO segmentation only | 0.530 | 3.13 | 41.93 | 0.3134 | 2.54 |
| Segmentation | Conservative refined mask pipeline | 0.359 | 3.63 | 42.72 | 0.2925 | 2.96 |
| Pose | Face-DragGAN face only | 0.929 | 20.66 | 35.43 | 0.2332 | 1.31 |
| Pose | Face-DragGAN face + body pose warp | 0.316 | 24.40 | 43.56 | 0.2355 | 1.99 |
| Pose | Keypoint-guided micro piecewise warp | 0.157 | 3.45 | 42.72 | 0.2925 | 2.96 |
| Diffusion | RealVisXL V4 inpainting | 0.988 | 3.55 | 38.09 | 0.2164 | 1.54 |
| Diffusion | SDXL inpainting | 0.576 | 3.80 | 40.07 | 0.2616 | 1.96 |
| Diffusion | RealisticVision v5.1 inpainting | 0.276 | 3.39 | 42.72 | 0.2925 | 2.96 |
| Face-DragGAN | Fast PTI setting | 0.882 | 13.65 | 42.64 | 0.2344 | 1.86 |
| Face-DragGAN | Lower support movement | 0.657 | 19.92 | 42.83 | 0.2320 | 1.91 |
| Face-DragGAN | High inversion steps | 0.111 | 33.83 | 44.23 | 0.2461 | 2.13 |

## Inpainting Results

| Method | L1 ↓ | MSE ↓ | PSNR ↑ | SSIM ↑ | Interpretation |
|---|---:|---:|---:|---:|---|
| U-Net baseline | 0.0142 | 0.0035 | 24.61 | - | Reliable pixel baseline, limited texture realism |
| Edge-guided CNN | 0.0119 | 0.0026 | - | - | Better synthetic-mask reconstruction and boundaries |
| Stable Diffusion inpainting | 0.0369 | 0.0058 | 23.31 | 0.756 | Best perceptual realism, less pixel-aligned |

## Key Takeaways

- Multi-task classification improved portfolio value because it demonstrates representation learning beyond single-label classification.
- Replacement quality depends heavily on mask quality and pose compatibility; no single component solves the full realism problem alone.
- Diffusion inpainting is strongest for perceptual completion, while CNN baselines remain useful for fast, deterministic reconstruction benchmarks.
- Region-aware metrics are more honest than global image metrics because most pixels outside the dog region remain unchanged.
