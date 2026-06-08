# Inpainting Source Package

Recommended future modules:

- `dataset.py`: synthetic mask generation and image completion datasets.
- `unet.py`: baseline U-Net.
- `edge_guided.py`: edge generator and completion network.
- `diffusion.py`: Stable Diffusion / SDXL inpainting wrappers.
- `sam_masks.py`: SAM-guided mask selection and refinement.
- `evaluate.py`: L1, MSE, PSNR, SSIM, and visual grids.

The current implementation lives in `notebooks/inpainting/`.
