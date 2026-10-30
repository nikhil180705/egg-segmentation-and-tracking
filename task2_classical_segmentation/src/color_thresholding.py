"""
task2_classical_segmentation/src/color_thresholding.py
=======================================================
Color-space-based thresholding for egg segmentation.

Implements and compares multiple thresholding strategies:

1. HSV thresholding — Hue-invariant; isolates bright, low-saturation objects.
   Best for white/brown eggs on variable-color backgrounds.

2. Grayscale Otsu — Global automatic threshold; works well when egg-background
   contrast is strong.

3. Adaptive thresholding — Local threshold; handles non-uniform lighting.

4. RGB channel thresholding — Direct per-channel thresholds; simple but brittle.

5. YCrCb thresholding — Separates luminance from chrominance; sometimes better
   for skin-toned backgrounds.

Each function returns a binary mask (uint8, 0/255).
"""

from typing import Optional, Tuple, List

import cv2
import numpy as np

from shared.logging_utils import get_logger

logger = get_logger(__name__)


def threshold_hsv(
    hsv: np.ndarray,
    lower: Tuple[int, int, int] = (0, 0, 120),
    upper: Tuple[int, int, int] = (179, 100, 255),
) -> np.ndarray:
    """
    Threshold image in HSV color space.

    Default range captures bright (V>120), low-saturation (S<100) objects
    regardless of hue — matches white/beige/light-brown eggs.

    Parameters
    ----------
    hsv : np.ndarray
        HSV image (H: 0-179, S: 0-255, V: 0-255 in OpenCV).
    lower, upper : (H, S, V) tuples

    Returns
    -------
    np.ndarray
        Binary mask uint8.
    """
    lower_arr = np.array(lower, dtype=np.uint8)
    upper_arr = np.array(upper, dtype=np.uint8)
    return cv2.inRange(hsv, lower_arr, upper_arr)


def threshold_hsv_dual(
    hsv: np.ndarray,
    lower1: Tuple[int, int, int] = (0, 0, 100),
    upper1: Tuple[int, int, int] = (30, 60, 255),
    lower2: Tuple[int, int, int] = (140, 0, 100),
    upper2: Tuple[int, int, int] = (179, 60, 255),
) -> np.ndarray:
    """
    Dual-range HSV thresholding for capturing brown/reddish eggs.

    Uses two HSV ranges and combines them. Useful when egg hues wrap around
    the hue circle (e.g., red-brown eggs: H near 0 and 179).

    Returns
    -------
    np.ndarray
        Union of both ranges.
    """
    m1 = cv2.inRange(hsv, np.array(lower1, dtype=np.uint8), np.array(upper1, dtype=np.uint8))
    m2 = cv2.inRange(hsv, np.array(lower2, dtype=np.uint8), np.array(upper2, dtype=np.uint8))
    return cv2.bitwise_or(m1, m2)


def threshold_otsu(gray: np.ndarray, invert: bool = False) -> Tuple[np.ndarray, float]:
    """
    Apply Otsu's automatic global thresholding.

    Otsu's method selects the threshold that minimizes intra-class variance
    (equivalently, maximizes inter-class variance) between foreground and
    background pixels. Works well when the histogram is bimodal.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image.
    invert : bool
        If True, invert the result (foreground = dark objects).

    Returns
    -------
    mask : np.ndarray
        Binary mask uint8.
    threshold_value : float
        The automatically chosen threshold.
    """
    flags = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    if invert:
        flags = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    thresh_val, mask = cv2.threshold(gray, 0, 255, flags)
    logger.debug(f"Otsu threshold: {thresh_val:.1f}")
    return mask, float(thresh_val)


def threshold_adaptive(
    gray: np.ndarray,
    block_size: int = 21,
    C: int = 5,
    method: str = "gaussian",
    invert: bool = False,
) -> np.ndarray:
    """
    Apply adaptive thresholding.

    Unlike global Otsu, adaptive thresholding computes a local threshold for
    each pixel neighborhood. Better for images with spatially varying lighting.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image.
    block_size : int
        Neighborhood size for local threshold (must be odd).
    C : int
        Constant subtracted from computed local mean/weighted mean.
    method : str
        "gaussian" (weighted mean) or "mean" (simple mean).
    invert : bool
        If True, foreground = dark objects.

    Returns
    -------
    np.ndarray
        Binary mask uint8.
    """
    if block_size % 2 == 0:
        block_size += 1
    adaptive_method = (
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == "gaussian"
        else cv2.ADAPTIVE_THRESH_MEAN_C
    )
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(gray, 255, adaptive_method, thresh_type, block_size, C)


def threshold_rgb(
    img_bgr: np.ndarray,
    r_range: Tuple[int, int] = (140, 255),
    g_range: Tuple[int, int] = (130, 255),
    b_range: Tuple[int, int] = (120, 255),
) -> np.ndarray:
    """
    Threshold in RGB space with per-channel ranges.

    Simple but effective when egg color is relatively consistent.
    Note: OpenCV stores images as BGR.

    Parameters
    ----------
    img_bgr : np.ndarray
        BGR image.
    r_range, g_range, b_range : (min, max) tuples

    Returns
    -------
    np.ndarray
        Binary mask.
    """
    b, g, r = cv2.split(img_bgr)
    mask_r = cv2.inRange(r, r_range[0], r_range[1])
    mask_g = cv2.inRange(g, g_range[0], g_range[1])
    mask_b = cv2.inRange(b, b_range[0], b_range[1])
    return cv2.bitwise_and(mask_r, cv2.bitwise_and(mask_g, mask_b))


def combine_threshold_masks(
    masks: List[np.ndarray],
    mode: str = "union",
) -> np.ndarray:
    """
    Combine multiple binary masks.

    Parameters
    ----------
    masks : list of np.ndarray
    mode : str
        "union" (OR), "intersection" (AND), or "majority" (voting).

    Returns
    -------
    np.ndarray
        Combined binary mask.
    """
    if not masks:
        raise ValueError("No masks to combine.")
    result = masks[0].copy()
    for m in masks[1:]:
        if mode == "union":
            result = cv2.bitwise_or(result, m)
        elif mode == "intersection":
            result = cv2.bitwise_and(result, m)

    if mode == "majority":
        # Count how many masks agree on each pixel
        stack = np.stack([m > 0 for m in masks], axis=0).astype(np.uint8)
        vote_count = stack.sum(axis=0)
        threshold = len(masks) // 2 + 1
        result = (vote_count >= threshold).astype(np.uint8) * 255

    return result
