# Code Quality Review

## Highest-Impact Improvements

1. **Move reusable logic out of notebooks.** The replacement notebooks contain repeated YOLO, SAM, pose, warp, mask, and visualization helpers. Convert these into modules under `src/dog_replacement/`.
2. **Replace hardcoded paths.** Several notebooks reference Google Drive, Windows temp paths, local `data/outputs`, or environment-specific filenames. Use a `configs/` directory with YAML files and environment-variable overrides.
3. **Separate experiments from library code.** Keep notebooks for narrative experiments, but put deterministic training, inference, and evaluation code in Python modules or CLI scripts.
4. **Standardize output contracts.** Each pipeline stage should read from and write to documented folders with predictable filenames.
5. **Add smoke tests.** Small tests should verify image loading, transform shapes, metric functions, mask generation, and pipeline config parsing.
6. **Remove stale design constraints.** Existing notes say not to maintain pipeline logic through scripts, but the portfolio version should move toward reusable modules and scripts for reproducibility.

## Specific Issues To Address

- Duplicate segmentation and cutout logic appears in both `main_pipeline/01_enhanced_dog_replacement.ipynb` and `face_draggan_pipeline/01_cutout.ipynb`.
- Pose helpers are duplicated across feasibility, query, warp, and face-DragGAN notebooks.
- Some notebooks mix data download, training, evaluation, visualization, and report writing in one file.
- Notebook outputs contain machine-specific warnings and paths; clear or curate outputs before public release.
- Model files and generated artifacts should stay out of Git unless they are tiny demo examples.
- Add docstrings for model classes, metric functions, mask generation helpers, and pipeline-stage functions.
- Create a consistent naming scheme: `classification`, `dog_replacement`, and `inpainting` are clearer than final-project question labels.

## Recommended Refactor Roadmap

1. Create `src/classification/dataset.py`, `model.py`, `train.py`, `evaluate.py`.
2. Create `src/inpainting/dataset.py`, `models.py`, `diffusion.py`, `evaluate.py`.
3. Create `src/dog_replacement/detect.py`, `segment.py`, `pose.py`, `warp.py`, `diffusion.py`, `harmonize.py`, `evaluate.py`.
4. Add `configs/classification.yaml`, `configs/replacement.yaml`, and `configs/inpainting.yaml`.
5. Add `python -m src...` command examples once modules exist.
6. Add GitHub Actions for linting and lightweight import checks.
