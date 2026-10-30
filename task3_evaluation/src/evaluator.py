"""
task3_evaluation/src/evaluator.py
==================================
Main evaluation orchestrator for Task 3.

Compares classical segmentation (Task 2) against SAM pseudo-ground truth (Task 1)
on the held-out evaluation set.

Evaluation Design:
- SAM outputs serve as pseudo-GT (no hand-annotated masks available)
- Both combined masks (union of all egg instances) are compared pixel-wise
- Instance-level comparison uses Hungarian assignment with IoU >= 0.25 threshold
- Results are reported per-image and aggregated across the eval set
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from shared.image_utils import collect_images
from shared.logging_utils import get_logger
from task3_evaluation.src.metrics import (
    aggregate_metrics,
    compute_instance_metrics,
    compute_pixel_metrics,
)

logger = get_logger(__name__)

# Output directories
SAM_MASKS_DIR = Path("task1_pretrained_segmentation/outputs/masks")
CLASSICAL_MASKS_DIR = Path("task2_classical_segmentation/outputs/masks")
EVAL_DIR = Path("task3_evaluation/outputs")


def load_combined_mask(masks_dir: Path, image_stem: str) -> Optional[np.ndarray]:
    """
    Load the combined mask (union of all egg instances) for an image.

    Falls back to binary-ORing individual masks if combined not found.

    Parameters
    ----------
    masks_dir : Path
        Directory containing mask files.
    image_stem : str
        Image filename without extension.

    Returns
    -------
    np.ndarray or None
        Binary mask (H, W) uint8, or None if not found.
    """
    combined_path = masks_dir / f"{image_stem}_combined_mask.png"
    if combined_path.exists():
        mask = cv2.imread(str(combined_path), cv2.IMREAD_GRAYSCALE)
        return (mask > 127).astype(np.uint8) * 255 if mask is not None else None

    # Fallback: OR all individual masks
    individual = sorted(masks_dir.glob(f"{image_stem}_mask_*.png"))
    individual = [p for p in individual if "combined" not in p.name]
    if not individual:
        return None

    first = cv2.imread(str(individual[0]), cv2.IMREAD_GRAYSCALE)
    if first is None:
        return None
    combined = (first > 127).astype(np.uint8) * 255
    for p in individual[1:]:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            combined = cv2.bitwise_or(combined, (m > 127).astype(np.uint8) * 255)
    return combined


def load_instance_masks(masks_dir: Path, image_stem: str) -> List[np.ndarray]:
    """
    Load all individual instance masks for an image.

    Returns
    -------
    list of np.ndarray
        Binary masks (H, W) uint8 for each egg instance.
    """
    individual = sorted(masks_dir.glob(f"{image_stem}_mask_*.png"))
    individual = [p for p in individual if "combined" not in p.name]
    masks = []
    for p in individual:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            masks.append((m > 127).astype(np.uint8) * 255)
    return masks


def evaluate_image(
    image_stem: str,
    sam_masks_dir: Path,
    classical_masks_dir: Path,
    iou_threshold: float = 0.25,
) -> Dict[str, Any]:
    """
    Evaluate classical vs SAM segmentation for a single image.

    Parameters
    ----------
    image_stem : str
        Image filename stem (without extension).
    sam_masks_dir : Path
        Directory with SAM masks.
    classical_masks_dir : Path
        Directory with classical masks.
    iou_threshold : float
        Minimum IoU for instance match.

    Returns
    -------
    dict with pixel metrics, instance metrics, and metadata.
    """
    result = {
        "image_name": image_stem,
        "sam_mask_found": False,
        "classical_mask_found": False,
        "pixel_metrics": None,
        "instance_metrics": None,
        "error": None,
    }

    try:
        # Load combined masks for pixel-level comparison
        sam_combined = load_combined_mask(sam_masks_dir, image_stem)
        classical_combined = load_combined_mask(classical_masks_dir, image_stem)

        result["sam_mask_found"] = sam_combined is not None
        result["classical_mask_found"] = classical_combined is not None

        if sam_combined is None:
            result["error"] = "SAM mask not found"
            logger.warning(f"SAM mask missing for {image_stem}")
            return result

        if classical_combined is None:
            # Classical detected 0 eggs — create empty mask
            h, w = sam_combined.shape[:2]
            classical_combined = np.zeros((h, w), dtype=np.uint8)
            logger.debug(f"Classical mask missing for {image_stem} — using empty mask")

        # Pixel-level metrics
        result["pixel_metrics"] = compute_pixel_metrics(
            pred_mask=classical_combined,
            gt_mask=sam_combined,
        )

        # Instance-level metrics
        sam_instances = load_instance_masks(sam_masks_dir, image_stem)
        classical_instances = load_instance_masks(classical_masks_dir, image_stem)

        result["instance_metrics"] = compute_instance_metrics(
            pred_masks=classical_instances,
            gt_masks=sam_instances,
            iou_threshold=iou_threshold,
        )

    except Exception as e:
        logger.error(f"Error evaluating {image_stem}: {e}", exc_info=True)
        result["error"] = str(e)

    return result


def run_evaluation(
    split_manifest_path: str = "data/split_manifest.json",
    sam_masks_dir: Optional[str] = None,
    classical_masks_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    iou_threshold: float = 0.25,
    on_eval_set: bool = True,
) -> Dict[str, Any]:
    """
    Run full evaluation on development or evaluation set.

    Parameters
    ----------
    split_manifest_path : str
        Path to dataset split manifest JSON.
    sam_masks_dir : str, optional
        Override SAM masks directory.
    classical_masks_dir : str, optional
        Override classical masks directory.
    output_dir : str, optional
        Where to save results.
    iou_threshold : float
        Minimum IoU for instance matching.
    on_eval_set : bool
        If True, evaluate on eval set; else dev set.

    Returns
    -------
    dict with per-image results and aggregate statistics.
    """
    sam_dir = Path(sam_masks_dir) if sam_masks_dir else SAM_MASKS_DIR
    classical_dir = Path(classical_masks_dir) if classical_masks_dir else CLASSICAL_MASKS_DIR
    out_dir = Path(output_dir) if output_dir else EVAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load split manifest
    with open(split_manifest_path, "r") as f:
        manifest = json.load(f)

    split_key = "eval" if on_eval_set else "dev"
    image_paths = manifest.get(split_key, [])
    if not image_paths:
        logger.error(f"No images in {split_key} split of manifest")
        return {}

    image_stems = [Path(p).stem for p in image_paths]
    logger.info(f"Evaluating {len(image_stems)} images from {split_key} split")

    # Evaluate each image
    per_image_results = []
    for stem in image_stems:
        result = evaluate_image(stem, sam_dir, classical_dir, iou_threshold)
        per_image_results.append(result)

    # Collect per-image pixel/instance metrics for aggregation
    valid_pixel = [
        r["pixel_metrics"] for r in per_image_results
        if r["pixel_metrics"] is not None
    ]
    valid_instance = [
        r["instance_metrics"] for r in per_image_results
        if r["instance_metrics"] is not None
    ]

    pixel_aggregate = aggregate_metrics(valid_pixel)
    instance_aggregate = aggregate_metrics(valid_instance)

    # Summary
    n_evaluated = len(per_image_results)
    n_sam_found = sum(1 for r in per_image_results if r["sam_mask_found"])
    n_classical_found = sum(1 for r in per_image_results if r["classical_mask_found"])
    n_errors = sum(1 for r in per_image_results if r["error"])

    summary = {
        "split": split_key,
        "total_images": n_evaluated,
        "sam_masks_found": n_sam_found,
        "classical_masks_found": n_classical_found,
        "errors": n_errors,
        "iou_threshold": iou_threshold,
        "pixel_metrics": pixel_aggregate,
        "instance_metrics": instance_aggregate,
    }

    # Save results
    _save_per_image_csv(per_image_results, out_dir / f"per_image_metrics_{split_key}.csv")
    _save_summary_json(summary, out_dir / f"evaluation_summary_{split_key}.json")

    # Print summary
    _print_summary(summary)

    return {"summary": summary, "per_image": per_image_results}


def _save_per_image_csv(results: List[Dict], csv_path: Path) -> None:
    """Save per-image metrics to CSV."""
    fieldnames = [
        "image_name",
        "sam_found", "classical_found", "error",
        # Pixel metrics
        "iou", "dice", "precision", "recall", "f1",
        # Instance metrics
        "count_pred", "count_gt", "matched",
        "instance_precision", "instance_recall", "instance_f1", "mean_matched_iou",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            pm = r.get("pixel_metrics") or {}
            im = r.get("instance_metrics") or {}
            writer.writerow({
                "image_name": r["image_name"],
                "sam_found": r["sam_mask_found"],
                "classical_found": r["classical_mask_found"],
                "error": r.get("error", ""),
                "iou": pm.get("iou", ""),
                "dice": pm.get("dice", ""),
                "precision": pm.get("precision", ""),
                "recall": pm.get("recall", ""),
                "f1": pm.get("f1", ""),
                "count_pred": im.get("count_pred", ""),
                "count_gt": im.get("count_gt", ""),
                "matched": im.get("matched", ""),
                "instance_precision": im.get("instance_precision", ""),
                "instance_recall": im.get("instance_recall", ""),
                "instance_f1": im.get("instance_f1", ""),
                "mean_matched_iou": im.get("mean_matched_iou", ""),
            })
    logger.info(f"Per-image metrics saved: {csv_path}")


def _save_summary_json(summary: Dict, json_path: Path) -> None:
    """Save evaluation summary to JSON."""
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Evaluation summary saved: {json_path}")


def _print_summary(summary: Dict) -> None:
    """Print formatted evaluation summary."""
    pm = summary.get("pixel_metrics", {})
    im = summary.get("instance_metrics", {})

    print("\n" + "=" * 65)
    print(f"TASK 3 -- EVALUATION SUMMARY ({summary['split'].upper()} SET)")
    print("=" * 65)
    print(f"Images evaluated:    {summary['total_images']}")
    print(f"SAM masks found:     {summary['sam_masks_found']}")
    print(f"Classical masks:     {summary['classical_masks_found']}")
    print(f"Errors:              {summary['errors']}")
    print(f"IoU threshold:       {summary['iou_threshold']}")

    print("\nPIXEL-LEVEL METRICS (Classical vs SAM):")
    for metric in ["iou", "dice", "precision", "recall", "f1"]:
        if metric in pm:
            m = pm[metric]
            print(f"  {metric:10s}: mean={m['mean']:.4f} ± {m['std']:.4f}  "
                  f"(median={m['median']:.4f}, min={m['min']:.4f}, max={m['max']:.4f})")

    print("\nINSTANCE-LEVEL METRICS:")
    print(f"  Total predicted eggs: {im.get('total_count_pred', 'N/A')}")
    print(f"  Total GT eggs (SAM):  {im.get('total_count_gt', 'N/A')}")
    print(f"  Total matched:        {im.get('total_matched', 'N/A')}")
    for metric in ["instance_precision", "instance_recall", "instance_f1", "mean_matched_iou"]:
        if metric in im:
            m = im[metric]
            print(f"  {metric:22s}: mean={m['mean']:.4f} ± {m['std']:.4f}")
    print("=" * 65)
