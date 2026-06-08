# Data

Raw datasets are not committed to this repository.

## Stanford Dogs

Download the Stanford Dogs Dataset from Kaggle:

https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset

Suggested layout:

```text
data/raw/stanford-dogs-dataset/
├── images/
└── annotations/
```

The classification and inpainting notebooks use the image folders by breed. The classification workflow drops images below 100 px on the shorter side, resizes/crops images, normalizes with ImageNet statistics, and splits the valid index into 70% train, 15% validation, and 15% test.

## Replacement Examples

Dog replacement experiments expect a small set of original/replacement image pairs under:

```text
data/example/
├── pair1/
├── pair2/
└── face_pair/
```

Do not commit private photos, full datasets, or generated intermediate artifacts. Commit only small, permission-safe examples in `outputs/examples/`.
