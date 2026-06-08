# Models

Model checkpoints are intentionally excluded from Git because they are large and may have third-party license restrictions.

Expected artifacts include:

- `multitask_resnet50_best.pt`
- fine-tuned YOLO dog pose checkpoint
- YOLO/SAM segmentation dependencies
- StyleGAN2-ADA AFHQ dog generator for DragGAN-style face editing
- optional diffusion model caches from Hugging Face
- CNN inpainting checkpoints such as `completion_generator_edge_guided.pt`

Recommended public release strategy:

1. Keep small configuration files in Git.
2. Store large checkpoints in GitHub Releases, Hugging Face, Google Drive, or another model registry.
3. Document exact checkpoint names, source URLs, and licenses.
4. Add checksum hashes for any artifact required to reproduce reported results.
