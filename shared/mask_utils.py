"""
shared/mask_utils.py
====================
Binary mask operations shared across Tasks 1, 2, and 3.

All masks are represented as:
  - dtype: uint8
  - values: 0 (background) or 255 (foreground)
  - shape: (H, W) — single channel

Combined/labeled masks use dtype int32 with sequential labels starting at 1.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from shared.logging_utils import get_logger

logger = get_logger(__name__)


# ── Mask I/O ─────────────────────────────────────────────────────────────────

def save_binary_mask(mask: np.ndarray, path: Union[str, Path]) -> None:
    """
    Save a binary mask (0/255 uint8) as a PNG.

    PNG is lossless and preserves exact binary values.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask, shape (H, W), uint8, values 0 or 255.
    path : str or Path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_out = ensure_binary_uint8(mask)
    cv2.imwrite(str(path), mask_out)
    logger.debug(f"Saved binary mask: {path}")


def load_binary_mask(path: Union[str, Path]) -> np.ndarray:
    """
    Load a binary mask from PNG.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    np.ndarray
        Binary mask, shape (H, W), uint8, values 0 or 255.
    """
    path = Path(path)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot load mask: {path}")
    return ensure_binary_uint8(mask)


# ── Mask Validation & Conversion ─────────────────────────────────────────────

def ensure_binary_uint8(mask: np.ndarray, threshold: int = 127) -> np.ndarray:
    """
    Ensure mask is uint8 with values strictly 0 or 255.

    Handles boolean masks, float masks [0,1], and grayscale [0,255].

    Parameters
    ----------
    mask : np.ndarray
        Input mask of any dtype.
    threshold : int
        Binarization threshold for grayscale inputs.

    Returns
    -------
    np.ndarray
        uint8 mask with values 0 or 255.
    """
    if mask.dtype == bool:
        return (mask.astype(np.uint8)) * 255
    if mask.dtype in (np.float32, np.float64):
        # Assume values in [0, 1]
        return ((mask > 0.5).astype(np.uint8)) * 255
    # Grayscale uint8
    _, binary = cv2.threshold(mask.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)
    return binary


def is_valid_binary_mask(mask: np.ndarray) -> bool:
    """
    Check that a mask is a 2D uint8 array with only values 0 or 255.

    Parameters
    ----------
    mask : np.ndarray

    Returns
    -------
    bool
    """
    if mask.ndim != 2:
        return False
    if mask.dtype != np.uint8:
        return False
    unique_vals = set(np.unique(mask).tolist())
    return unique_vals.issubset({0, 255})


def combine_masks(masks: List[np.ndarray]) -> np.ndarray:
    """
    Combine multiple binary instance masks into a single union mask.

    Parameters
    ----------
    masks : list of np.ndarray
        Each mask is (H, W) uint8 with values 0/255.

    Returns
    -------
    np.ndarray
        Union mask (H, W) uint8.
    """
    if not masks:
        raise ValueError("Cannot combine empty list of masks.")
    union = np.zeros_like(masks[0], dtype=np.uint8)
    for m in masks:
        union = cv2.bitwise_or(union, ensure_binary_uint8(m))
    return union


def label_connected_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Label connected components in a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (H, W) uint8.

    Returns
    -------
    labeled : np.ndarray
        int32 array where each component has a unique integer label (1, 2, ..., N).
        Background = 0.
    num_components : int
        Number of connected components found.
    """
    binary = ensure_binary_uint8(mask)
    num_labels, labeled = cv2.connectedComponents(binary, connectivity=8)
    num_components = num_labels - 1  # Subtract background label 0
    return labeled.astype(np.int32), num_components


def extract_instance_masks(mask: np.ndarray) -> List[np.ndarray]:
    """
    Extract individual instance masks from a labeled or combined mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary union mask (H, W) uint8.

    Returns
    -------
    list of np.ndarray
        One binary uint8 mask per connected component.
    """
    labeled, num_components = label_connected_components(mask)
    instance_masks = []
    for label_id in range(1, num_components + 1):
        instance = np.zeros_like(mask, dtype=np.uint8)
        instance[labeled == label_id] = 255
        instance_masks.append(instance)
    return instance_masks


def mask_to_bounding_box(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Return the bounding box of the foreground region in a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (H, W) uint8.

    Returns
    -------
    (x, y, w, h) or None if mask is empty.
    """
    pts = cv2.findNonZero(mask)
    if pts is None:
        return None
    x, y, w, h = cv2.boundingRect(pts)
    return (x, y, w, h)


def mask_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Compute the centroid of a binary mask via image moments.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (H, W) uint8.

    Returns
    -------
    (cx, cy) float tuple, or None if mask is empty.
    """
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return (cx, cy)


def mask_area(mask: np.ndarray) -> int:
    """
    Return the foreground pixel count (area) of a binary mask.

    Parameters
    ----------
    mask : np.ndarray

    Returns
    -------
    int
    """
    return int(np.count_nonzero(mask))


def mask_circularity(mask: np.ndarray) -> float:
    """
    Compute circularity = 4π·area / perimeter².

    Perfect circle = 1.0, elongated shape < 1.0.

    Parameters
    ----------
    mask : np.ndarray

    Returns
    -------
    float, or 0.0 if contour cannot be found.
    """
    binary = ensure_binary_uint8(mask)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)
    if perimeter == 0:
        return 0.0
    return float(4 * np.pi * area / (perimeter ** 2))
