"""
task2_classical_segmentation/src/color_thresholding.py
=======================================================
Color-space-based thresholding for egg segmentation.

Implements:
1. Non-egg chromaticity rejection (green vegetation, blue/cyan tableware, dark cast shadows).
2. Dual-band egg chromaticity segmentation (white/cream and brown/tan eggs).
3. Hierarchical multi-level Otsu thresholding for separating eggs from bright plates and pans.
4. Adaptive and global Otsu thresholding.
5. Standard HSV, RGB, and mask combination utilities.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from shared.logging_utils import get_logger

logger = get_logger(__name__)


def get_non_egg_rejection_mask(
    hsv: np.ndarray,
    lab: np.ndarray,
    suppress_green: bool = True,
    suppress_blue: bool = True,
    suppress_shadow: bool = True,
) -> np.ndarray:
    """
    Construct a boolean mask of pixels that are definitively NOT eggs based on chromaticity.

    - Green vegetation (herbs, salad, garnishes, green bowls):
      In CIELAB, a* < 120 (negative a* is green).
      In HSV, Hue in [35, 85] with Saturation > 35.
    - Blue/cyan plates, cookware, plastic containers:
      In CIELAB, b* < 118 (negative b* is blue).
      In HSV, Hue in [90, 140] with Saturation > 35.
    - Deep shadow / black pans:
      L < 28 and V < 32.

    Returns
    -------
    np.ndarray
        Boolean mask (H, W), True where pixels should be suppressed.
    """
    h, w = hsv.shape[:2]
    rejection_mask = np.zeros((h, w), dtype=bool)

    if suppress_green:
        green_lab = lab[:, :, 1] < 125
        green_hsv = (hsv[:, :, 0] >= 28) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] > 28)
        rejection_mask |= (green_lab | green_hsv)

    if suppress_blue:
        blue_lab = lab[:, :, 2] < 118
        blue_hsv = (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 140) & (hsv[:, :, 1] > 35)
        rejection_mask |= (blue_lab | blue_hsv)

    if suppress_shadow:
        shadow_mask = (lab[:, :, 0] < 28) & (hsv[:, :, 2] < 32)
        rejection_mask |= shadow_mask

    return rejection_mask


def threshold_egg_chromaticity(
    hsv: np.ndarray,
    lab: np.ndarray,
    non_egg_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Segment pixels whose color matches plausible egg shells (white, cream, beige, brown).

    Dual-stream color gating:
    - Stream 1: White/cream eggs (high L, low saturation, neutral a* and b*).
    - Stream 2: Brown/tan eggs (warm hue, moderate saturation, warm a* and b*).

    Returns
    -------
    np.ndarray
        Binary mask (uint8 0/255).
    """
    L = lab[:, :, 0]
    a_star = lab[:, :, 1]
    b_star = lab[:, :, 2]

    # White eggs: high lightness, low saturation, neutral chromaticity
    white_eggs = (
        (L >= 135) &
        (hsv[:, :, 1] <= 75) &
        (a_star >= 118) & (a_star <= 142) &
        (b_star >= 120) & (b_star <= 160)
    )

    # Brown / tan eggs: warm reddish/yellowish hue, moderate saturation
    brown_eggs = (
        ((hsv[:, :, 0] <= 32) | (hsv[:, :, 0] >= 165)) &
        (hsv[:, :, 1] >= 25) & (hsv[:, :, 1] <= 175) &
        (L >= 60) & (L <= 215) &
        (a_star >= 124) & (a_star <= 165) &
        (b_star >= 128) & (b_star <= 195)
    )

    egg_pixels = white_eggs | brown_eggs
    if non_egg_mask is not None:
        egg_pixels &= ~non_egg_mask

    return egg_pixels.astype(np.uint8) * 255


def threshold_hierarchical_otsu(
    image_gray: np.ndarray,
    base_mask: np.ndarray,
    min_component_area: int = 75000,
    non_egg_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply hierarchical sub-thresholding inside oversized connected components.

    When eggs sit on white plates, ceramic dishes, or stainless cookware,
    global Otsu groups the dish and eggs together into a single massive component.
    Sub-thresholding within that component cleanly separates the eggs (brightest)
    from the dish/pan floor.

    Parameters
    ----------
    image_gray : np.ndarray
        Grayscale intensity image.
    base_mask : np.ndarray
        Initial thresholded binary mask.
    min_component_area : int
        Components larger than this are treated as background surfaces and sub-thresholded.
    non_egg_mask : np.ndarray, optional
        Pixels to exclude from foreground.

    Returns
    -------
    np.ndarray
        Refined binary mask.
    """
    cnts, _ = cv2.findContours(base_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refined = np.zeros_like(base_mask)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_component_area:
            c_mask = np.zeros_like(base_mask)
            cv2.drawContours(c_mask, [c], -1, 255, -1)
            pix = image_gray[c_mask > 0]
            if len(pix) > 0:
                t_sub, _ = cv2.threshold(pix, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                sub_mask = np.zeros_like(base_mask)
                sub_mask[(image_gray > t_sub) & (c_mask > 0)] = 255
                if non_egg_mask is not None:
                    sub_mask[non_egg_mask] = 0
                sub_clean = cv2.morphologyEx(sub_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                sub_clean = cv2.morphologyEx(sub_clean, cv2.MORPH_CLOSE, close_kernel)
                refined = cv2.bitwise_or(refined, sub_clean)
        elif area > 1200:
            cv2.drawContours(refined, [c], -1, 255, -1)

    if non_egg_mask is not None:
        refined[non_egg_mask] = 0

    return refined


def threshold_hsv(
    hsv: np.ndarray,
    lower: Tuple[int, int, int] = (0, 0, 120),
    upper: Tuple[int, int, int] = (179, 100, 255),
) -> np.ndarray:
    """Threshold image in HSV color space."""
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
    """Dual-range HSV thresholding for capturing brown/reddish eggs."""
    m1 = cv2.inRange(hsv, np.array(lower1, dtype=np.uint8), np.array(upper1, dtype=np.uint8))
    m2 = cv2.inRange(hsv, np.array(lower2, dtype=np.uint8), np.array(upper2, dtype=np.uint8))
    return cv2.bitwise_or(m1, m2)


def threshold_otsu(gray: np.ndarray, invert: bool = False) -> Tuple[np.ndarray, float]:
    """Apply Otsu's automatic global thresholding."""
    flags = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    if invert:
        flags = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    thresh_val, mask = cv2.threshold(gray, 0, 255, flags)
    return mask, float(thresh_val)


def threshold_adaptive(
    gray: np.ndarray,
    block_size: int = 51,
    C: int = -6,
    method: str = "gaussian",
    invert: bool = False,
) -> np.ndarray:
    """Apply adaptive local thresholding."""
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
    """Threshold in RGB space with per-channel ranges."""
    b, g, r = cv2.split(img_bgr)
    mask_r = cv2.inRange(r, r_range[0], r_range[1])
    mask_g = cv2.inRange(g, g_range[0], g_range[1])
    mask_b = cv2.inRange(b, b_range[0], b_range[1])
    return cv2.bitwise_and(mask_r, cv2.bitwise_and(mask_g, mask_b))


def combine_threshold_masks(
    masks: List[np.ndarray],
    mode: str = "union",
) -> np.ndarray:
    """Combine multiple binary masks (union, intersection, majority)."""
    if not masks:
        raise ValueError("No masks to combine.")
    result = masks[0].copy()
    for m in masks[1:]:
        if mode == "union":
            result = cv2.bitwise_or(result, m)
        elif mode == "intersection":
            result = cv2.bitwise_and(result, m)

    if mode == "majority":
        stack = np.stack([m > 0 for m in masks], axis=0).astype(np.uint8)
        vote_count = stack.sum(axis=0)
        threshold = len(masks) // 2 + 1
        result = (vote_count >= threshold).astype(np.uint8) * 255

    return result
