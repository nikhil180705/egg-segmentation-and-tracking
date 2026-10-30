# Dataset Information

## Source

- **Primary**: Roboflow Universe — `fire-bf7qe/egg-vzhtr` Dataset v1
  - URL: https://universe.roboflow.com/fire-bf7qe/egg-vzhtr/dataset/1
  - License: CC BY 4.0
- **Alternative**: Google Drive ZIP — original images

## Dataset Statistics

| Property | Value |
|----------|-------|
| Total images (v1) | 110 |
| Annotation type | Classification (egg class labels) |
| Image format | JPEG |
| Train split | 77 images (70%) |
| Validation split | 22 images (20%) |
| Test split | 11 images (10%) |

> **Note**: No pixel-level segmentation masks are provided in the original dataset.
> Task 1 uses SAM to generate pseudo-ground-truth segmentation masks.
> Task 2 classical algorithm predictions are evaluated against these SAM masks.

## Dataset Characteristics (from inspection)

- Single object class: `egg`
- Variable backgrounds (hand-held, surface, colorful cloth, white/dark backgrounds)
- Variable lighting conditions
- Mix of single and multiple eggs per image
- No augmentations applied in v1

## How to Download

### Option A: Google Drive (Recommended for raw images)
```bash
python scripts/download_dataset.py --source gdrive
```

### Option B: Roboflow API
```bash
python scripts/download_dataset.py --source roboflow --api-key YOUR_API_KEY
```

## Directory Structure After Download

```
data/
├── raw/              # Original downloaded images (gitignored)
│   ├── train/
│   ├── valid/
│   └── test/
├── processed/        # Preprocessed images
├── dev/              # 80% development split (for training/tuning)
└── eval/             # 20% evaluation split (held-out, untouched)
```
