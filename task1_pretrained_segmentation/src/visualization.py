"""
task1_pretrained_segmentation/src/visualization.py
===================================================
Visualization utilities for Task 1 SAM segmentation results.

Generates:
- Overlay images (original + colored masks)
- Side-by-side comparisons (original | overlay)
- Grid summaries of multiple segmented images
- Failure case visualizations
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for script execution
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from shared.image_utils import load_image_rgb, save_image
from shared.logging_utils import get_logger
from shared.mask_utils import combine_masks

logger = get_logger(__name__)

# Color palette for instance masks (seed ensures reproducibility)
_RNG = np.random.default_rng(42)


def _random_color() -> Tuple[int, int, int]:
    """Generate a random vivid RGB color."""
    hue = _RNG.integers(0, 180)
    color_hsv = np.array([[[hue, 220, 220]]], dtype=np.uint8)
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(c) for c in color_bgr[::-1])  # Return as RGB


def create_overlay(
    image_rgb: np.ndarray,
    masks: List[np.ndarray],
    alpha: float = 0.40,
) -> np.ndarray:
    """
    Create a segmentation overlay image.

    Each egg mask is filled with a distinct random color and blended
    with the original image.

    Parameters
    ----------
    image_rgb : np.ndarray
        Original image (H, W, 3) RGB uint8.
    masks : list of np.ndarray
        Binary masks (H, W) uint8.
    alpha : float
        Mask opacity (0=transparent, 1=opaque).

    Returns
    -------
    np.ndarray
        Overlay image (H, W, 3) RGB uint8.
    """
    overlay = image_rgb.copy().astype(np.float32)

    for mask in masks:
        color = _random_color()
        colored = np.zeros_like(image_rgb, dtype=np.float32)
        for c_idx, c_val in enumerate(color):
            colored[:, :, c_idx] = c_val
        mask_bool = mask > 127
        overlay[mask_bool] = (
            overlay[mask_bool] * (1 - alpha) + colored[mask_bool] * alpha
        )

        # Draw contour outline
        mask_u8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay_u8 = overlay.astype(np.uint8)
        cv2.drawContours(overlay_u8, contours, -1, color, 2)
        overlay = overlay_u8.astype(np.float32)

    return np.clip(overlay, 0, 255).astype(np.uint8)


def create_side_by_side(
    image_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    title: str = "",
    num_eggs: int = 0,
) -> np.ndarray:
    """
    Create a side-by-side comparison: original | segmented overlay.

    Parameters
    ----------
    image_rgb : np.ndarray
        Original image.
    overlay_rgb : np.ndarray
        Overlay image.
    title : str
        Image filename title.
    num_eggs : int
        Number of detected eggs (for annotation).

    Returns
    -------
    np.ndarray
        Side-by-side image.
    """
    h, w = image_rgb.shape[:2]
    canvas = np.zeros((h + 30, w * 2 + 10, 3), dtype=np.uint8)
    canvas[:] = 30  # Dark grey background

    # Place images
    canvas[30:30 + h, :w] = image_rgb
    canvas[30:30 + h, w + 10:w * 2 + 10] = overlay_rgb

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Original", (5, 22), font, 0.6, (200, 200, 200), 1)
    cv2.putText(
        canvas,
        f"SAM Segmentation ({num_eggs} eggs)",
        (w + 15, 22),
        font, 0.6, (100, 255, 150), 1
    )
    return canvas


def save_overlay(
    image_path: Path,
    masks: List[np.ndarray],
    output_dir: Path,
    alpha: float = 0.40,
) -> Path:
    """
    Generate and save overlay + side-by-side visualization for one image.

    Returns path to saved overlay.
    """
    output_dir = Path(output_dir)
    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    img_rgb = load_image_rgb(image_path)

    if not masks:
        # No eggs detected — save warning overlay
        overlay = img_rgb.copy()
        cv2.putText(
            overlay, "NO EGGS DETECTED", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 50, 50), 2
        )
        overlay_path = overlays_dir / f"{image_path.stem}_overlay.jpg"
        save_image(overlay, overlay_path, is_rgb=True)
        return overlay_path

    overlay = create_overlay(img_rgb, masks, alpha=alpha)
    side_by_side = create_side_by_side(img_rgb, overlay, image_path.name, len(masks))

    # Save overlay
    overlay_path = overlays_dir / f"{image_path.stem}_overlay.jpg"
    save_image(overlay, overlay_path, is_rgb=True)

    # Save side-by-side
    sbs_path = overlays_dir / f"{image_path.stem}_comparison.jpg"
    save_image(side_by_side, sbs_path, is_rgb=True)

    return overlay_path


def save_summary_grid(
    results: List[Dict[str, Any]],
    output_dir: Path,
    max_samples: int = 16,
    grid_cols: int = 4,
) -> None:
    """
    Save a grid showing sample segmented images.

    Parameters
    ----------
    results : list of segmentation result dicts
    output_dir : Path
    max_samples : int
        Maximum number of images to include in grid.
    grid_cols : int
        Number of columns in grid.
    """
    output_dir = Path(output_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Select representative samples: include edge cases
    successful = [r for r in results if not r.get("error") and r["num_eggs"] > 0]
    failed = [r for r in results if r.get("error") or r["num_eggs"] == 0]

    samples = successful[:max_samples]

    if not samples:
        logger.warning("No successful segmentations to show in grid.")
        return

    n = len(samples)
    rows = (n + grid_cols - 1) // grid_cols
    fig, axes = plt.subplots(rows, grid_cols, figsize=(grid_cols * 4, rows * 4))
    axes = np.array(axes).flatten() if rows > 1 or grid_cols > 1 else [axes]

    for i, result in enumerate(samples):
        ax = axes[i]
        img_rgb = load_image_rgb(result["image_path"])
        overlay = create_overlay(img_rgb, result["masks"], alpha=0.4)
        ax.imshow(overlay)
        ax.set_title(
            f"{result['image_name'][:20]}\n{result['num_eggs']} eggs | {result['processing_time_s']:.1f}s",
            fontsize=8,
        )
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        f"SAM Egg Segmentation — {n} samples ({len(failed)} failures)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    grid_path = reports_dir / "segmentation_grid.png"
    plt.savefig(str(grid_path), dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Summary grid saved: {grid_path}")


def save_failure_analysis(
    results: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Save visualizations of failed/zero-egg segmentation cases."""
    output_dir = Path(output_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    failures = [r for r in results if r.get("error") or r["num_eggs"] == 0]
    if not failures:
        logger.info("No failure cases to visualize.")
        return

    n = min(len(failures), 9)
    rows = (n + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(12, rows * 4))
    axes = np.array(axes).flatten()

    for i, result in enumerate(failures[:n]):
        ax = axes[i]
        try:
            img_rgb = load_image_rgb(result["image_path"])
            ax.imshow(img_rgb)
        except Exception:
            ax.set_facecolor("black")

        reason = result.get("error") or "0 eggs detected"
        ax.set_title(f"{result['image_name'][:25]}\n{reason[:40]}", fontsize=7, color="red")
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"SAM Failure Cases ({len(failures)} total)", fontsize=11, color="red")
    plt.tight_layout()
    fail_path = reports_dir / "failure_cases.png"
    plt.savefig(str(fail_path), dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"Failure analysis saved: {fail_path}")
