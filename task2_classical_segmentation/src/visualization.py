"""
task2_classical_segmentation/src/visualization.py
==================================================
Visualization utilities for classical segmentation results.

Provides:
- Distinct high-contrast colors for each individual egg instance.
- Semi-transparent mask overlays with crisp contour outlines.
- Side-by-side comparison images showing Original vs Detected Egg Masks + Contours.
- Comprehensive summary grids displaying image names, detected egg counts, and processing times.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from shared.image_utils import load_image_rgb, save_image
from shared.logging_utils import get_logger

logger = get_logger(__name__)

# Curated high-contrast vibrant color palette for instance distinction (RGB)
DISTINCT_COLORS: List[Tuple[int, int, int]] = [
    (0, 255, 128),    # Vibrant Lime Green
    (255, 60, 180),   # Vivid Magenta / Pink
    (0, 215, 255),    # Cyan / Sky Blue
    (255, 180, 0),    # Amber / Golden Yellow
    (255, 80, 40),    # Bright Coral / Orange
    (170, 70, 255),   # Purple / Violet
    (50, 255, 220),   # Aquamarine
    (255, 225, 50),   # Bright Lemon
    (255, 120, 190),  # Rose
    (100, 240, 80),   # Chartreuse
    (70, 160, 255),   # Cornflower Blue
    (255, 140, 90),   # Peach
]


def get_instance_color(instance_idx: int) -> Tuple[int, int, int]:
    """Return a deterministic, distinct RGB color for the given instance index."""
    return DISTINCT_COLORS[instance_idx % len(DISTINCT_COLORS)]


def create_overlay(
    image_rgb: np.ndarray,
    masks: List[np.ndarray],
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Create a colored mask and contour overlay on the image.

    Parameters
    ----------
    image_rgb : np.ndarray
        Original image in RGB format (H, W, 3).
    masks : list of np.ndarray
        Binary masks (H, W) uint8.
    alpha : float
        Opacity of the mask overlay.

    Returns
    -------
    np.ndarray
        Image with semi-transparent mask fills and crisp outer contour borders.
    """
    overlay = image_rgb.copy().astype(np.float32)

    for idx, mask in enumerate(masks):
        color_rgb = get_instance_color(idx)
        mask_bool = (mask > 127)

        # Blend mask fill
        for c_idx in range(3):
            overlay[mask_bool, c_idx] = (
                overlay[mask_bool, c_idx] * (1.0 - alpha) + color_rgb[c_idx] * alpha
            )

        # Draw crisp contour boundary
        mask_u8 = mask.astype(np.uint8)
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay_u8 = np.clip(overlay, 0, 255).astype(np.uint8)
        cv2.drawContours(overlay_u8, cnts, -1, color_rgb, 2, cv2.LINE_AA)
        overlay = overlay_u8.astype(np.float32)

    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_overlay(
    image_path: Path,
    masks: List[np.ndarray],
    output_dir: Path,
    alpha: float = 0.45,
    processing_time_s: float = 0.0,
) -> Path:
    """
    Generate and save overlay and side-by-side comparison visualization.

    Parameters
    ----------
    image_path : Path
        Source image path.
    masks : list of np.ndarray
        Detected egg binary masks.
    output_dir : Path
        Output directory root.
    alpha : float
        Overlay transparency.
    processing_time_s : float
        Processing time in seconds for HUD display.

    Returns
    -------
    Path
        Path to the saved overlay file.
    """
    output_dir = Path(output_dir)
    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    img_rgb = load_image_rgb(image_path)
    h, w = img_rgb.shape[:2]

    if not masks:
        overlay = img_rgb.copy()
        cv2.putText(
            overlay, "NO EGGS DETECTED", (15, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 60, 60), 2, cv2.LINE_AA
        )
    else:
        overlay = create_overlay(img_rgb, masks, alpha=alpha)

    # Save standalone overlay
    overlay_path = overlays_dir / f"{image_path.stem}_overlay.jpg"
    save_image(overlay, overlay_path, is_rgb=True)

    # Create publication-quality Side-by-Side Comparison
    header_h = 36
    sbs = np.zeros((h + header_h, w * 2 + 6, 3), dtype=np.uint8)
    sbs[:] = 30  # Dark slate background header

    # Embed original on left, overlay on right
    sbs[header_h:, :w] = img_rgb
    sbs[header_h:, w + 6:] = overlay

    # Draw divider line
    sbs[:, w:w + 6] = [20, 20, 20]

    # Header text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(sbs, f"Original ({image_path.name[:25]})", (12, 24), font, 0.65, (220, 220, 220), 1, cv2.LINE_AA)

    status_str = f"Classical CV: {len(masks)} eggs | {processing_time_s:.3f}s"
    text_color = (100, 255, 150) if masks else (255, 100, 100)
    cv2.putText(sbs, status_str, (w + 16, 24), font, 0.65, text_color, 2, cv2.LINE_AA)

    sbs_path = overlays_dir / f"{image_path.stem}_comparison.jpg"
    save_image(sbs, sbs_path, is_rgb=True)

    return overlay_path


def save_summary_grid(
    results: List[Dict[str, Any]],
    output_dir: Path,
    max_samples: int = 36,
    grid_cols: int = 6,
) -> None:
    """Save summary grid of segmented images showing detections, counts, and timing."""
    # Prioritize images with detections first, then hard negative / clean background rejection scenes
    with_eggs = [r for r in results if not r.get("error") and r.get("num_eggs", 0) > 0]
    without_eggs = [r for r in results if not r.get("error") and r.get("num_eggs", 0) == 0]
    ordered = with_eggs + without_eggs
    if not ordered:
        logger.warning("No segmentations to show in grid.")
        return

    n = min(len(ordered), max_samples)
    grid_cols = min(grid_cols, n)
    rows = (n + grid_cols - 1) // grid_cols

    fig, axes = plt.subplots(rows, grid_cols, figsize=(grid_cols * 3.4, rows * 3.4), dpi=150)
    axes_flat = np.array(axes).flatten() if (rows > 1 or grid_cols > 1) else [axes]

    for i, result in enumerate(ordered[:n]):
        ax = axes_flat[i]
        img_rgb = load_image_rgb(result["image_path"])
        overlay = create_overlay(img_rgb, result["masks"], alpha=0.45)
        ax.imshow(overlay)
        t_sec = result.get("processing_time_s", 0.0)
        n_eggs = result["num_eggs"]
        ax.set_title(
            f"{result['image_name'][:20]}\n{n_eggs} eggs | {t_sec:.3f}s",
            fontsize=7.5,
            fontweight="bold",
            pad=3,
        )
        ax.axis("off")

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()
    reports_dir = Path(output_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    grid_path = reports_dir / "classical_segmentation_grid.png"
    plt.savefig(grid_path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info(f"Saved classical summary grid: {grid_path}")
