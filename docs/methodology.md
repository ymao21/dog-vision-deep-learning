# Methodology

## Data Preparation

The main dataset is Stanford Dogs, with 20,580 images across 120 breeds. The classification workflow builds a clean index, drops images with a shorter side below 100 px, applies resize/crop preprocessing, and normalizes images with ImageNet channel statistics. The final indexed classification split contains 20,579 images:

| Split | Images | Share |
|---|---:|---:|
| Train | 14,405 | 70% |
| Validation | 3,087 | 15% |
| Test | 3,087 | 15% |

Augmentations include random crop, horizontal flip, color jitter, and rotation.

## Problem 0: Multi-Task Classification

The classifier uses a ResNet-50 backbone pretrained on ImageNet. The final representation is shared across multiple lightweight heads:

- Breed classification.
- Breed group classification.
- Size classification.
- Coat length classification.
- Coat color prediction from HSV-derived pseudo-labels.

Training uses AdamW with a lower learning rate for the backbone and a higher learning rate for the heads. The multi-task objective emphasizes breed classification while using related attributes as regularizers.

## Problem 1: Dog Replacement

The dog replacement system is an end-to-end image editing pipeline:

1. **Detection:** YOLO identifies dog bounding boxes in original and replacement images.
2. **Pose estimation:** a fine-tuned YOLO-v11n pose model predicts 24 dog keypoints with visibility flags.
3. **Candidate search:** query images are matched to replacement candidates by pose and detectability.
4. **Feasibility gating:** pairs are accepted or rejected based on keypoint overlap, pose similarity, and expected warp error.
5. **Segmentation:** YOLO segmentation and SAM/SAM2-style masks isolate the original and replacement dogs.
6. **Pose alignment:** replacement cutouts are scaled and transformed with similarity or piecewise affine warping.
7. **Face alignment:** the Face-DragGAN variant uses StyleGAN2-ADA dog inversion, PTI, and latent-space dragging for local face adjustment.
8. **Diffusion refinement:** local inpainting repairs seams, old-dog remnants, and transition artifacts.
9. **Post-processing:** harmonization adjusts local color, boundary softness, and sharpness.

## Problem 2: Dog Inpainting

The inpainting workflow uses synthetic masks and real occlusion demos.

The U-Net baseline reconstructs RGB pixels directly from masked images. The edge-guided pipeline first predicts edges for the missing region, then conditions an image completion network on the masked RGB image, mask, and edge information. Diffusion inpainting uses pretrained generative priors and prompt guidance for more realistic semantic completion.

SAM-guided real-world occlusion removal uses segmentation masks to isolate occluders, then dilates and smooths the mask before diffusion inference.

## Evaluation

Classification is evaluated with held-out test accuracy per task.

Replacement is evaluated with cycle consistency:

1. Replace dog A with dog B.
2. Replace dog B back with dog A.
3. Compare the reconstructed image with the original.

Primary metrics include boundary-ring MAE, edited-region LPIPS, edited-region SSIM, background MAE, dog-region SSIM, and runtime.

Inpainting is evaluated with L1, MSE, PSNR, SSIM, and visual inspection. Pixel metrics are appropriate for synthetic masks but incomplete for diffusion outputs because generative completion may be plausible without being pixel-identical to the original.
