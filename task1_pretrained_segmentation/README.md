# Task 1 — Pretrained Segmentation (SAM)

## Overview

Uses Meta's **Segment Anything Model (SAM)** to generate egg segmentation masks.

### Model Selection

| Variant | Size | VRAM (RTX 2050) | Quality | Decision |
|---------|------|-----------------|---------|----------|
| vit_h | 2.4 GB | Exceeds 4 GB | Best | ❌ Excluded |
| vit_l | 1.2 GB | ~3.8 GB (borderline) | High | ❌ Excluded |
| **vit_b** | **375 MB** | **~2.1 GB** | Good | ✅ **Selected** |

### Prompting Strategies

Two strategies are implemented and experimentally compared:

1. **Auto (AMG)** — Automatic Mask Generation. SAM proposes all masks in the image. Post-filtered by area ratio, circularity, and solidity to isolate egg regions.

2. **Bbox** — Classical CV detects approximate egg bounding boxes (HSV thresholding + morphology + contour detection), which are then fed to SAM as prompts for precise masks.

## Directory Structure

```
task1_pretrained_segmentation/
├── src/
│   ├── model_loader.py    # SAM download and initialization
│   ├── egg_segmenter.py   # Main orchestrator
│   ├── mask_generator.py  # AMG + bbox-prompted generation
│   └── visualization.py   # Overlays, grids, failure analysis
├── scripts/
│   └── run_segmentation.py
└── outputs/
    ├── masks/             # Binary PNG masks (gitignored)
    ├── overlays/          # Visualization images (gitignored)
    └── reports/           # CSV/JSON summaries (version controlled)
```

## How to Run

```bash
# Process development set (default)
python task1_pretrained_segmentation/scripts/run_segmentation.py

# Process evaluation set
python task1_pretrained_segmentation/scripts/run_segmentation.py --split eval

# Use bbox prompting instead of auto
python task1_pretrained_segmentation/scripts/run_segmentation.py --strategy bbox

# Quick test on 5 images
python task1_pretrained_segmentation/scripts/run_segmentation.py --max-images 5
```

## Expected Outputs

- `outputs/masks/{stem}_mask_00.png` — Binary mask per egg
- `outputs/masks/{stem}_combined_mask.png` — Union of all egg masks
- `outputs/overlays/{stem}_overlay.jpg` — Colored mask overlay
- `outputs/overlays/{stem}_comparison.jpg` — Side-by-side comparison
- `outputs/reports/segmentation_results_dev.csv` — Per-image statistics
- `outputs/reports/segmentation_summary_dev.json` — Aggregate summary
- `outputs/reports/segmentation_grid.png` — Sample grid visualization
- `outputs/reports/failure_cases.png` — Failed segmentation cases
