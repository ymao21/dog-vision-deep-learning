# Project Summary

## Classifying and Reimagining Dogs with Deep Learning

This project builds a multi-part computer vision system for dog image understanding and editing. It combines supervised classification, object detection, segmentation, pose estimation, geometric warping, GAN latent editing, and diffusion-based inpainting.

The work is organized around three applied problems:

1. **Multi-task dog classification:** infer breed and related visual attributes from one dog image.
2. **Pose-aware dog replacement:** insert a replacement dog into an original scene while preserving pose, background, and visual realism.
3. **Dog image inpainting:** reconstruct missing or occluded regions using CNN baselines and pretrained diffusion models.

## Portfolio Positioning

This repository is strongest when framed as an end-to-end applied ML project rather than a single-model notebook. The project demonstrates:

- Deep learning model fine-tuning with PyTorch and torchvision.
- Multi-task learning with shared visual representations.
- Computer vision pipeline design across detection, segmentation, warping, and generation.
- Generative AI experimentation with Stable Diffusion, SDXL, RealVisXL, and GAN inversion.
- Region-aware model evaluation beyond global accuracy or global image similarity.
- Practical ML engineering decisions around data layout, reproducibility, and artifact management.

## Main Results

The multi-task classifier reached 86.1% breed accuracy and above 91% accuracy on group, size, and coat-length prediction on a held-out Stanford Dogs test split. The dog replacement system was evaluated with a small cycle-consistency benchmark across three image pairs, where RealVisXL and SAM with a YOLO bounding-box prior performed best among tested component variants. For inpainting, CNN methods achieved stronger pixel reconstruction scores, while diffusion produced more perceptually realistic completions.

## Deployment Concept

The proposed deployment flow validates user uploads, detects dogs, estimates pose, segments target regions, performs pose alignment or inpainting, and returns a downloadable edited image. Maintenance relies on controlled ablations and metric comparisons before replacing production modules.
