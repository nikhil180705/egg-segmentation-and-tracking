"""
task2_classical_segmentation/src/visualization.py
==================================================
Visualization utilities for classical segmentation results.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from shared.image_utils import load_image_rgb, save_image
from shared.logging_utils import get_logger

logger = get_logger(__name__)

_RNG = np.random.default_rng(123)


def _random_color():
    hue = _RNG.integers(0, 180)
    color_hsv = np.array([[[hue, 220, 200]]], dtype=np.uint8)
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(c) for c in color_bgr[::-1])  # RGB


def create_overlay(
    image_rgb: np.ndarray,
    masks: List[np.ndarray],
    alpha: float = 0.4,
) -> np.ndarray:
    """Create colored mask overlay on image."""
    overlay = image_rgb.copy().astype(np.float32)
    for mask in masks:
        color = _random_color()
        colored = np.zeros_like(image_rgb, dtype=np.float32)
        for c_idx, c_val in enumerate(color):
            colored[:, :, c_idx] = c_val
        mask_bool = mask > 127
        overlay[mask_bool] = overlay[mask_bool] * (1 - alpha) + colored[mask_bool] * alpha
        mask_u8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay_u8 = overlay.astype(np.uint8)
        cv2.drawContours(overlay_u8, contours, -1, color, 2)
        overlay = overlay_u8.astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_overlay(
    image_path: Path,
    masks: List[np.ndarray],
    output_dir: Path,
    alpha: float = 0.4,
) -> Path:
    """Generate and save overlay visualization."""
    output_dir = Path(output_dir)
    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    img_rgb = load_image_rgb(image_path)

    if not masks:
        overlay = img_rgb.copy()
        cv2.putText(overlay, "NO EGGS DETECTED", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 50, 50), 2)
    else:
        overlay = create_overlay(img_rgb, masks, alpha=alpha)

    overlay_path = overlays_dir / f"{image_path.stem}_overlay.jpg"
    save_image(overlay, overlay_path, is_rgb=True)

    # Side-by-side
    h, w = img_rgb.shape[:2]
    sbs = np.zeros((h + 30, w * 2 + 5, 3), dtype=np.uint8)
    sbs[:] = 25
    sbs[30:, :w] = img_rgb
    if masks:
        sbs[30:, w + 5:] = overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(sbs, "Original", (5, 22), font, 0.6, (200, 200, 200), 1)
    cv2.putText(sbs, f"Classical ({len(masks)} eggs)", (w + 10, 22), font, 0.6, (100, 255, 150), 1)
    sbs_path = overlays_dir / f"{image_path.stem}_comparison.jpg"
    save_image(sbs, sbs_path, is_rgb=True)
    return overlay_path


def save_summary_grid(
    results: List[Dict[str, Any]],
    output_dir: Path,
    max_samples: int = 12,
    grid_cols: int = 4,
) -> None:
    """Save summary grid of segmented images."""
    successful = [r for r in results if not r.get("error") and r["num_eggs"] > 0]
    if not successful:
        logger.warning("No successful segmentations to show in grid.")
        return

    n = min(len(successful), max_samples)
    rows = (n + grid_cols - 1) // grid_cols
    fig, axes = plt.subplots(rows, grid_cols, figsize=(grid_cols * 4, rows * 4))
    axes_flat = np.array(axes).flatten() if rows > 1 or grid_cols > 1 else [axes]

    for i, result in enumerate(successful[:n]):
        ax = axes_flat[i]
        img_rgb = load_image_rgb(result["image_path"])
        overlay = create_overlay(img_rgb, result["masks"], alpha=0.4)
        ax.imshow(overlay)
        ax.set_title(
            f"{result['image_name'][:20]}\n{result['num_eggs']} eggs | {result['processing_time_s']:.2f}s",
            fontsize=7,
        )
        ax.axis("off")

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    reports_dir = Path(output_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plt.suptitle("Classical Egg Segmentation Results", fontsize=12, fontweight="bold")
    plt.tight_layout()
    grid_path = reports_dir / "classical_segmentation_grid.png"
    plt.savefig(str(grid_path), dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"Summary grid saved: {grid_path}")
