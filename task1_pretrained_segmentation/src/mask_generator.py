"""
task1_pretrained_segmentation/src/mask_generator.py
====================================================
SAM-based mask generation with two strategies:

1. AUTOMATIC MASK GENERATION (AMG):
   - SAM generates all possible masks in the image automatically
   - We then filter masks by area, circularity, and solidity to keep egg masks
   - Advantage: No manual prompts needed
   - Disadvantage: May produce many non-egg masks; filtering required

2. BOUNDING BOX PROMPTING:
   - First detect egg-like regions via classical CV (Otsu + morphology + contours)
   - Feed bounding boxes as prompts to SAM for precise mask generation
   - Advantage: More accurate, fewer false positives
   - Disadvantage: Depends on quality of initial bounding box detection

Both strategies are implemented; selection is config-driven.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from shared.logging_utils import get_logger
from shared.mask_utils import ensure_binary_uint8, mask_circularity

logger = get_logger(__name__)


def generate_masks_auto(
    image_rgb: np.ndarray,
    predictor_or_generator,
    amg_config: dict,
    egg_filter_config: dict,
) -> List[Dict[str, Any]]:
    """
    Generate egg masks using SAM's Automatic Mask Generator (AMG).

    Generates all candidate masks, then filters for egg-like regions.

    Parameters
    ----------
    image_rgb : np.ndarray
        Input image (H, W, 3) in RGB format.
    predictor_or_generator : SamAutomaticMaskGenerator
        Initialized AMG instance.
    amg_config : dict
        AMG hyperparameters.
    egg_filter_config : dict
        Egg-specific post-filtering thresholds.

    Returns
    -------
    list of dict
        Each dict has: 'segmentation' (bool array), 'area', 'bbox',
        'predicted_iou', 'stability_score', 'is_egg_candidate'.
    """
    try:
        from segment_anything import SamAutomaticMaskGenerator
    except ImportError:
        raise ImportError("segment_anything not installed.")

    logger.debug(f"Running AMG on image {image_rgb.shape}")
    raw_masks = predictor_or_generator.generate(image_rgb)
    logger.debug(f"AMG produced {len(raw_masks)} raw masks")

    img_area = image_rgb.shape[0] * image_rgb.shape[1]
    filtered_masks = []

    for mask_data in raw_masks:
        seg = mask_data["segmentation"].astype(np.uint8) * 255
        area = int(np.sum(mask_data["segmentation"]))
        area_ratio = area / img_area

        # Area filter
        if not (egg_filter_config["min_area_ratio"] <= area_ratio <= egg_filter_config["max_area_ratio"]):
            continue

        # Circularity filter — eggs are elliptical
        circ = mask_circularity(seg)
        if not (egg_filter_config["min_circularity"] <= circ <= egg_filter_config["max_circularity"]):
            continue

        # Solidity filter — eggs are convex
        contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            contour_area = cv2.contourArea(contour)
            solidity = contour_area / hull_area if hull_area > 0 else 0.0
            if solidity < egg_filter_config["min_solidity"]:
                continue
        else:
            continue

        mask_data["circularity"] = circ
        mask_data["area_ratio"] = area_ratio
        mask_data["solidity"] = solidity
        mask_data["is_egg_candidate"] = True
        filtered_masks.append(mask_data)

    logger.info(f"AMG: {len(raw_masks)} raw masks -> {len(filtered_masks)} egg candidates")
    return filtered_masks


def detect_egg_bboxes_classical(
    image_bgr: np.ndarray,
    min_area: int = 2000,
    max_area: int = 250000,
) -> List[Tuple[int, int, int, int]]:
    """
    Use classical CV to detect egg bounding boxes for SAM bbox prompting.

    Pipeline: Gaussian blur → HSV threshold (low-saturation bright objects)
              → Otsu fallback → morphological clean → contour filtering.

    Parameters
    ----------
    image_bgr : np.ndarray
        Input image (H, W, 3) BGR format.
    min_area : int
        Minimum contour area to keep.
    max_area : int
        Maximum contour area to keep.

    Returns
    -------
    list of (x1, y1, x2, y2) bounding boxes in image coordinates.
    """
    h, w = image_bgr.shape[:2]

    # Gaussian blur
    blurred = cv2.GaussianBlur(image_bgr, (5, 5), 0)

    # Try HSV thresholding first (whitish/brownish eggs = low saturation, high value)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask_hsv = cv2.inRange(hsv, np.array([0, 0, 120]), np.array([179, 100, 255]))

    # Try Otsu on grayscale
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    _, mask_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Combine: take union then morphological clean
    combined = cv2.bitwise_or(mask_hsv, mask_otsu)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=3)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel_open, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (min_area <= area <= max_area):
            continue

        x, y, cw, ch = cv2.boundingRect(contour)
        # Add 5% padding around each bbox for better SAM coverage
        pad_x = int(cw * 0.05)
        pad_y = int(ch * 0.05)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + cw + pad_x)
        y2 = min(h, y + ch + pad_y)
        bboxes.append((x1, y1, x2, y2))

    logger.debug(f"Classical detection found {len(bboxes)} egg bboxes")
    return bboxes


def generate_masks_bbox(
    image_rgb: np.ndarray,
    sam_predictor,
    bboxes: List[Tuple[int, int, int, int]],
) -> List[Dict[str, Any]]:
    """
    Generate SAM masks using bounding box prompts.

    Parameters
    ----------
    image_rgb : np.ndarray
        Input image (H, W, 3) RGB.
    sam_predictor : SamPredictor
        SAM predictor with image already set.
    bboxes : list of (x1, y1, x2, y2)
        Bounding box prompts.

    Returns
    -------
    list of dicts with 'segmentation', 'bbox', 'predicted_iou', 'score'
    """
    import torch

    masks_out = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        bbox_tensor = np.array([[x1, y1, x2, y2]])
        bbox_transformed = sam_predictor.transform.apply_boxes_torch(
            torch.tensor(bbox_tensor, dtype=torch.float),
            image_rgb.shape[:2]
        ).to(sam_predictor.device)

        with torch.no_grad():
            masks, iou_preds, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=bbox_transformed,
                multimask_output=False,  # Single best mask per bbox
            )

        # masks shape: (1, 1, H, W)
        mask_np = masks[0, 0].cpu().numpy()
        iou = float(iou_preds[0, 0].cpu().numpy())

        masks_out.append({
            "segmentation": mask_np,
            "bbox": bbox,
            "predicted_iou": iou,
            "area": int(np.sum(mask_np)),
        })

    logger.debug(f"BBox prompting: generated {len(masks_out)} masks from {len(bboxes)} bboxes")
    return masks_out


def select_best_masks(
    masks: List[Dict[str, Any]],
    iou_threshold: float = 0.70,
    apply_nms: bool = True,
    nms_iou_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Filter and deduplicate masks.

    Steps:
    1. Filter by predicted IoU threshold.
    2. Apply NMS to remove highly overlapping duplicates.
    3. Sort by predicted IoU descending.

    Parameters
    ----------
    masks : list of mask dicts
    iou_threshold : float
        Minimum predicted IoU to keep a mask.
    apply_nms : bool
        Whether to apply non-maximum suppression.
    nms_iou_threshold : float
        IoU threshold above which two masks are considered duplicates.

    Returns
    -------
    list of filtered mask dicts
    """
    if not masks:
        return []

    # Filter by predicted IoU
    filtered = [m for m in masks if m.get("predicted_iou", 1.0) >= iou_threshold]

    if not filtered:
        # Relax threshold and keep top candidate
        logger.warning(f"No masks pass IoU threshold {iou_threshold}. Taking top candidate.")
        filtered = sorted(masks, key=lambda x: x.get("predicted_iou", 0), reverse=True)[:1]

    if apply_nms and len(filtered) > 1:
        filtered = _apply_mask_nms(filtered, nms_iou_threshold)

    # Sort by predicted IoU descending
    filtered.sort(key=lambda x: x.get("predicted_iou", 0), reverse=True)
    logger.debug(f"After filtering: {len(masks)} -> {len(filtered)} masks")
    return filtered


def _apply_mask_nms(
    masks: List[Dict[str, Any]], iou_threshold: float
) -> List[Dict[str, Any]]:
    """
    Remove overlapping masks using a greedy NMS strategy.

    Masks are sorted by predicted IoU; the highest-scoring mask is kept,
    and all masks with overlap > iou_threshold are removed.
    """
    sorted_masks = sorted(masks, key=lambda x: x.get("predicted_iou", 0), reverse=True)
    kept = []

    for candidate in sorted_masks:
        seg_c = candidate["segmentation"].astype(bool)
        overlap = False
        for kept_mask in kept:
            seg_k = kept_mask["segmentation"].astype(bool)
            intersection = np.logical_and(seg_c, seg_k).sum()
            union = np.logical_or(seg_c, seg_k).sum()
            if union > 0 and intersection / union > iou_threshold:
                overlap = True
                break
        if not overlap:
            kept.append(candidate)

    return kept
