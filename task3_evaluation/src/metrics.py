"""
task3_evaluation/src/metrics.py
================================
Segmentation evaluation metrics.

Implements pixel-level and instance-level metrics for comparing
classical segmentation against SAM pseudo-ground truth.

Metric Definitions:
-------------------

PIXEL-LEVEL METRICS (mask comparison):
  - IoU (Jaccard Index): |A ∩ B| / |A ∪ B|
    Range [0, 1]; 1 = perfect overlap
  - Dice (F1): 2|A ∩ B| / (|A| + |B|)
    More sensitive to small regions than IoU
  - Precision: True Positives / (TP + FP)
    What fraction of predicted pixels are correct?
  - Recall: True Positives / (TP + FN)
    What fraction of ground truth pixels are found?
  - F1 Score: Harmonic mean of Precision and Recall

INSTANCE-LEVEL METRICS:
  - Mean IoU per egg instance (after Hungarian matching)
  - Count accuracy: |predicted_count - gt_count| / gt_count
  - Detection rate: fraction of GT eggs with IoU > threshold

AGGREGATION:
  - Per-image metrics averaged across all images
  - Distribution statistics (mean, std, median, min, max)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from shared.logging_utils import get_logger

logger = get_logger(__name__)


def _ensure_binary(mask: np.ndarray) -> np.ndarray:
    """Convert any mask to binary boolean array."""
    return mask > 0 if mask.dtype != bool else mask


def compute_pixel_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Compute pixel-level segmentation metrics between prediction and GT.

    Parameters
    ----------
    pred_mask : np.ndarray
        Predicted binary mask (H, W). Any dtype, thresholded at >0.
    gt_mask : np.ndarray
        Ground truth binary mask (H, W).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    dict with keys: iou, dice, precision, recall, f1, tp, fp, fn, tn
    """
    pred = _ensure_binary(pred_mask).astype(bool)
    gt = _ensure_binary(gt_mask).astype(bool)

    tp = float(np.sum(pred & gt))
    fp = float(np.sum(pred & ~gt))
    fn = float(np.sum(~pred & gt))
    tn = float(np.sum(~pred & ~gt))

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)

    return {
        "iou": round(iou, 6),
        "dice": round(dice, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute IoU between two binary masks. Lightweight version for instance matching.

    Parameters
    ----------
    mask_a, mask_b : np.ndarray
        Binary masks (H, W).

    Returns
    -------
    float
        IoU in [0, 1].
    """
    a = _ensure_binary(mask_a)
    b = _ensure_binary(mask_b)
    intersection = np.sum(a & b)
    union = np.sum(a | b)
    return float(intersection) / (float(union) + eps)


def hungarian_match_instances(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    iou_threshold: float = 0.25,
) -> Tuple[List[Tuple[int, int]], List[int], List[int], np.ndarray]:
    """
    Match predicted instance masks to ground truth using Hungarian algorithm.

    The cost matrix is the negative IoU between each pair of masks.
    A match is valid only if the IoU exceeds iou_threshold.

    Parameters
    ----------
    pred_masks : list of np.ndarray
        Predicted instance masks.
    gt_masks : list of np.ndarray
        Ground truth instance masks.
    iou_threshold : float
        Minimum IoU for a valid match.

    Returns
    -------
    matches : list of (pred_idx, gt_idx) tuples
        Valid matches above the IoU threshold.
    unmatched_preds : list of int
        Prediction indices with no valid GT match (false positives).
    unmatched_gts : list of int
        GT indices with no valid prediction match (false negatives).
    iou_matrix : np.ndarray
        (n_pred, n_gt) IoU matrix.
    """
    from scipy.optimize import linear_sum_assignment

    if not pred_masks or not gt_masks:
        return [], list(range(len(pred_masks))), list(range(len(gt_masks))), np.zeros((0, 0))

    n_pred = len(pred_masks)
    n_gt = len(gt_masks)

    # Build IoU matrix
    iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float32)
    for i, pm in enumerate(pred_masks):
        for j, gm in enumerate(gt_masks):
            iou_matrix[i, j] = compute_mask_iou(pm, gm)

    # Hungarian assignment (minimizes cost = maximizes IoU)
    cost_matrix = 1.0 - iou_matrix
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

    matches = []
    matched_preds = set()
    matched_gts = set()
    for pi, gi in zip(pred_indices, gt_indices):
        if iou_matrix[pi, gi] >= iou_threshold:
            matches.append((pi, gi))
            matched_preds.add(pi)
            matched_gts.add(gi)

    unmatched_preds = [i for i in range(n_pred) if i not in matched_preds]
    unmatched_gts = [j for j in range(n_gt) if j not in matched_gts]

    return matches, unmatched_preds, unmatched_gts, iou_matrix


def compute_instance_metrics(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    iou_threshold: float = 0.25,
) -> Dict[str, float]:
    """
    Compute instance-level detection and segmentation metrics.

    Parameters
    ----------
    pred_masks : list of np.ndarray
        Predicted instance masks.
    gt_masks : list of np.ndarray
        Ground truth instance masks (SAM pseudo-GT).
    iou_threshold : float
        Minimum IoU for a valid detection.

    Returns
    -------
    dict with keys:
        matched, unmatched_pred, unmatched_gt,
        instance_precision, instance_recall, instance_f1,
        mean_matched_iou, count_pred, count_gt
    """
    matches, unmatched_preds, unmatched_gts, iou_matrix = hungarian_match_instances(
        pred_masks, gt_masks, iou_threshold
    )

    n_matches = len(matches)
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)

    inst_precision = n_matches / (n_pred + 1e-8) if n_pred > 0 else 0.0
    inst_recall = n_matches / (n_gt + 1e-8) if n_gt > 0 else 0.0
    inst_f1 = (
        2 * inst_precision * inst_recall / (inst_precision + inst_recall + 1e-8)
        if (inst_precision + inst_recall) > 0 else 0.0
    )

    # Mean IoU over matched pairs
    matched_ious = [float(iou_matrix[pi, gi]) for pi, gi in matches]
    mean_matched_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    return {
        "matched": n_matches,
        "unmatched_pred": len(unmatched_preds),
        "unmatched_gt": len(unmatched_gts),
        "instance_precision": round(inst_precision, 6),
        "instance_recall": round(inst_recall, 6),
        "instance_f1": round(inst_f1, 6),
        "mean_matched_iou": round(mean_matched_iou, 6),
        "count_pred": n_pred,
        "count_gt": n_gt,
    }


def aggregate_metrics(
    per_image_metrics: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate per-image metrics into dataset-level statistics.

    Parameters
    ----------
    per_image_metrics : list of per-image metric dicts

    Returns
    -------
    dict mapping metric_name -> {mean, std, median, min, max}
    """
    if not per_image_metrics:
        return {}

    scalar_keys = [
        k for k, v in per_image_metrics[0].items()
        if isinstance(v, (int, float)) and k not in ("tp", "fp", "fn", "tn", "matched", "unmatched_pred", "unmatched_gt", "count_pred", "count_gt")
    ]

    aggregated = {}
    for key in scalar_keys:
        values = [m[key] for m in per_image_metrics if key in m and m[key] is not None]
        if not values:
            continue
        arr = np.array(values, dtype=float)
        aggregated[key] = {
            "mean": round(float(np.mean(arr)), 6),
            "std": round(float(np.std(arr)), 6),
            "median": round(float(np.median(arr)), 6),
            "min": round(float(np.min(arr)), 6),
            "max": round(float(np.max(arr)), 6),
        }

    # Also aggregate counts
    for key in ("matched", "unmatched_pred", "unmatched_gt", "count_pred", "count_gt", "tp", "fp", "fn", "tn"):
        values = [m.get(key, 0) for m in per_image_metrics]
        aggregated[f"total_{key}"] = int(sum(values))

    return aggregated
