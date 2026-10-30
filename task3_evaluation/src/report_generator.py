"""
task3_evaluation/src/report_generator.py
=========================================
Generates rich visual evaluation reports for Task 3.

Outputs:
- Comparative visualization: SAM mask | Classical mask | Overlay diff
- Metric distribution plots (IoU histogram, precision-recall curve)
- Summary table with best/worst performing images
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from shared.image_utils import load_image_rgb
from shared.logging_utils import get_logger

logger = get_logger(__name__)


def create_mask_comparison(
    image_rgb: np.ndarray,
    sam_mask: Optional[np.ndarray],
    classical_mask: Optional[np.ndarray],
    iou: float = 0.0,
    title: str = "",
) -> np.ndarray:
    """
    Create a 4-panel comparison: Original | SAM | Classical | Diff.

    Parameters
    ----------
    image_rgb : np.ndarray
        Original image RGB.
    sam_mask : np.ndarray or None
        SAM combined binary mask.
    classical_mask : np.ndarray or None
        Classical combined binary mask.
    iou : float
        Pre-computed IoU for annotation.
    title : str
        Image name for title.

    Returns
    -------
    np.ndarray
        Horizontal composite of 4 panels.
    """
    h, w = image_rgb.shape[:2]

    def mask_to_overlay(img, mask, color=(0, 200, 100)):
        if mask is None:
            return img.copy()
        overlay = img.copy().astype(np.float32)
        m = (mask > 127) if mask.dtype != bool else mask
        overlay[m] = overlay[m] * 0.5 + np.array(color, dtype=np.float32) * 0.5
        return overlay.astype(np.uint8)

    sam_overlay = mask_to_overlay(image_rgb, sam_mask, color=(100, 180, 255))
    cls_overlay = mask_to_overlay(image_rgb, classical_mask, color=(255, 160, 50))

    # Diff panel: TP=green, FP=red, FN=blue
    diff = np.zeros((h, w, 3), dtype=np.uint8)
    diff[:] = image_rgb // 3  # Dark background
    if sam_mask is not None and classical_mask is not None:
        gt = (sam_mask > 127)
        pred = (classical_mask > 127)
        diff[gt & pred] = [0, 220, 0]    # TP = green
        diff[pred & ~gt] = [220, 0, 0]   # FP = red
        diff[gt & ~pred] = [0, 0, 220]   # FN = blue

    panels = [image_rgb, sam_overlay, cls_overlay, diff]
    labels = ["Original", f"SAM (GT)", f"Classical", f"Diff (IoU={iou:.3f})"]
    colors = [(200, 200, 200), (100, 180, 255), (255, 160, 50), (200, 200, 200)]

    canvas_h = h + 25
    canvas = np.zeros((canvas_h, w * 4 + 9, 3), dtype=np.uint8)
    canvas[:] = 20

    for i, (panel, label, color) in enumerate(zip(panels, labels, colors)):
        x0 = i * (w + 3)
        canvas[25:25 + h, x0:x0 + w] = panel
        cv2.putText(canvas, label, (x0 + 4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return canvas


def save_comparison_grid(
    per_image_results: List[Dict[str, Any]],
    image_dir: Path,
    sam_masks_dir: Path,
    classical_masks_dir: Path,
    output_dir: Path,
    max_samples: int = 12,
    sort_by: str = "iou",
) -> None:
    """
    Save a grid of comparison panels sorted by IoU.

    Parameters
    ----------
    per_image_results : list
        Per-image evaluation result dicts.
    image_dir : Path
        Directory with original images.
    sam_masks_dir : Path
    classical_masks_dir : Path
    output_dir : Path
    max_samples : int
    sort_by : str
        "iou" (ascending = worst cases) or "iou_desc" (descending = best cases)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort results
    valid = [r for r in per_image_results if r.get("pixel_metrics")]
    descending = sort_by.endswith("_desc")
    sorted_results = sorted(
        valid,
        key=lambda r: r["pixel_metrics"].get("iou", 0),
        reverse=descending,
    )[:max_samples]

    if not sorted_results:
        logger.warning("No valid results to plot comparison grid")
        return

    n = len(sorted_results)
    fig, axes = plt.subplots(n, 1, figsize=(20, n * 2.5))
    if n == 1:
        axes = [axes]

    for ax, result in zip(axes, sorted_results):
        stem = result["image_name"]
        iou = result["pixel_metrics"].get("iou", 0) if result.get("pixel_metrics") else 0

        # Load images
        # Search for image in dev and eval dirs
        img_rgb = None
        for search_dir in [image_dir, image_dir.parent / "dev", image_dir.parent / "eval"]:
            candidates = list(search_dir.glob(f"{stem}.*")) if search_dir.exists() else []
            if candidates:
                img_rgb = load_image_rgb(candidates[0])
                break

        if img_rgb is None:
            img_rgb = np.zeros((100, 100, 3), dtype=np.uint8)

        sam_mask = _load_combined(sam_masks_dir, stem)
        cls_mask = _load_combined(classical_masks_dir, stem)

        panel = create_mask_comparison(img_rgb, sam_mask, cls_mask, iou, stem)
        ax.imshow(panel)
        ax.axis("off")
        ax.set_title(f"{stem[:40]}  |  IoU={iou:.3f}", fontsize=7)

    title = "Best Results (IoU Descending)" if descending else "Worst Results (IoU Ascending)"
    plt.suptitle(f"Classical vs SAM Comparison — {title}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    suffix = "best" if descending else "worst"
    out_path = output_dir / f"comparison_grid_{suffix}.png"
    plt.savefig(str(out_path), dpi=90, bbox_inches="tight")
    plt.close()
    logger.info(f"Comparison grid saved: {out_path}")


def _load_combined(masks_dir: Path, stem: str) -> Optional[np.ndarray]:
    """Load combined mask for an image stem."""
    p = masks_dir / f"{stem}_combined_mask.png"
    if p.exists():
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        return m
    return None


def save_metrics_plots(
    per_image_results: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """
    Save metric distribution plots: IoU histogram, Dice histogram, scatter.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid = [r for r in per_image_results if r.get("pixel_metrics")]
    if not valid:
        logger.warning("No valid results for metric plots")
        return

    ious = [r["pixel_metrics"]["iou"] for r in valid]
    dices = [r["pixel_metrics"]["dice"] for r in valid]
    precisions = [r["pixel_metrics"]["precision"] for r in valid]
    recalls = [r["pixel_metrics"]["recall"] for r in valid]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#1a1a2e")
    colors = {"iou": "#4fc3f7", "dice": "#81c784", "precision": "#ffb74d", "recall": "#f06292"}

    # IoU histogram
    axes[0][0].set_facecolor("#16213e")
    axes[0][0].hist(ious, bins=20, color=colors["iou"], edgecolor="white", alpha=0.85)
    axes[0][0].axvline(np.mean(ious), color="white", linestyle="--", label=f"Mean={np.mean(ious):.3f}")
    axes[0][0].set_title("IoU Distribution", color="white", fontsize=12)
    axes[0][0].set_xlabel("IoU", color="white")
    axes[0][0].set_ylabel("Count", color="white")
    axes[0][0].tick_params(colors="white")
    axes[0][0].legend(facecolor="#16213e", labelcolor="white")

    # Dice histogram
    axes[0][1].set_facecolor("#16213e")
    axes[0][1].hist(dices, bins=20, color=colors["dice"], edgecolor="white", alpha=0.85)
    axes[0][1].axvline(np.mean(dices), color="white", linestyle="--", label=f"Mean={np.mean(dices):.3f}")
    axes[0][1].set_title("Dice Distribution", color="white", fontsize=12)
    axes[0][1].set_xlabel("Dice", color="white")
    axes[0][1].set_ylabel("Count", color="white")
    axes[0][1].tick_params(colors="white")
    axes[0][1].legend(facecolor="#16213e", labelcolor="white")

    # Precision vs Recall scatter
    axes[1][0].set_facecolor("#16213e")
    sc = axes[1][0].scatter(recalls, precisions, c=ious, cmap="viridis", s=40, alpha=0.8)
    plt.colorbar(sc, ax=axes[1][0], label="IoU").ax.yaxis.label.set_color("white")
    axes[1][0].set_title("Precision vs Recall", color="white", fontsize=12)
    axes[1][0].set_xlabel("Recall", color="white")
    axes[1][0].set_ylabel("Precision", color="white")
    axes[1][0].tick_params(colors="white")
    axes[1][0].set_xlim(0, 1); axes[1][0].set_ylim(0, 1)

    # Instance count comparison
    instance_valid = [r for r in per_image_results if r.get("instance_metrics")]
    if instance_valid:
        pred_counts = [r["instance_metrics"]["count_pred"] for r in instance_valid]
        gt_counts = [r["instance_metrics"]["count_gt"] for r in instance_valid]
        axes[1][1].set_facecolor("#16213e")
        axes[1][1].scatter(gt_counts, pred_counts, color=colors["precision"], alpha=0.7, s=30)
        max_count = max(max(pred_counts, default=0), max(gt_counts, default=0))
        axes[1][1].plot([0, max_count], [0, max_count], "w--", alpha=0.5, label="Perfect agreement")
        axes[1][1].set_title("Egg Count: GT vs Predicted", color="white", fontsize=12)
        axes[1][1].set_xlabel("SAM GT Count", color="white")
        axes[1][1].set_ylabel("Classical Predicted Count", color="white")
        axes[1][1].tick_params(colors="white")
        axes[1][1].legend(facecolor="#16213e", labelcolor="white")

    plt.suptitle("Task 3: Classical vs SAM Evaluation Metrics", fontsize=14,
                 color="white", fontweight="bold")
    plt.tight_layout()
    out_path = output_dir / "metrics_plots.png"
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    logger.info(f"Metrics plots saved: {out_path}")
