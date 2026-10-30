"""
task2_classical_segmentation/src/morphology.py
===============================================
Morphological operations for mask cleanup.

Morphological operations modify binary masks based on neighborhood analysis.
For egg segmentation, the key operations are:

OPENING (erosion → dilation):
  - Removes small isolated foreground blobs (noise, debris)
  - Does NOT affect large regions (eggs)
  - Parameter: kernel_size controls minimum feature size preserved

CLOSING (dilation → erosion):
  - Fills small holes and gaps inside foreground regions (egg interiors)
  - Connects nearby regions separated by small gaps
  - Parameter: kernel_size controls maximum hole size filled

DILATION (expand foreground):
  - Recovers edges lost during erosion
  - Connects nearly-touching regions
  - Used after opening/closing to restore boundary accuracy

Kernel shape selection:
  - ELLIPSE: Preferred for eggs (circular structuring element)
  - RECT: Simpler, faster, but produces blockier artifacts
  - CROSS: Better for preserving thin structures

Iterations:
  - More iterations = stronger effect
  - 2-3 closing iterations typically needed for egg interiors
"""

from typing import Tuple

import cv2
import numpy as np

from shared.logging_utils import get_logger

logger = get_logger(__name__)

KERNEL_SHAPES = {
    "ellipse": cv2.MORPH_ELLIPSE,
    "rect": cv2.MORPH_RECT,
    "cross": cv2.MORPH_CROSS,
}


def make_kernel(size: int, shape: str = "ellipse") -> np.ndarray:
    """
    Create a morphological structuring element.

    Parameters
    ----------
    size : int
        Kernel size (diameter). Larger = stronger effect.
    shape : str
        "ellipse", "rect", or "cross".

    Returns
    -------
    np.ndarray
        Structuring element.
    """
    morph_shape = KERNEL_SHAPES.get(shape, cv2.MORPH_ELLIPSE)
    return cv2.getStructuringElement(morph_shape, (size, size))


def apply_opening(
    mask: np.ndarray,
    kernel_size: int = 5,
    iterations: int = 2,
    kernel_shape: str = "ellipse",
) -> np.ndarray:
    """
    Apply morphological opening (erosion followed by dilation).

    Removes small spurious foreground blobs. Kernel size determines
    the minimum blob size that will be REMOVED.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask uint8.
    kernel_size : int
    iterations : int
    kernel_shape : str

    Returns
    -------
    np.ndarray
        Cleaned mask.
    """
    kernel = make_kernel(kernel_size, kernel_shape)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)


def apply_closing(
    mask: np.ndarray,
    kernel_size: int = 15,
    iterations: int = 3,
    kernel_shape: str = "ellipse",
) -> np.ndarray:
    """
    Apply morphological closing (dilation followed by erosion).

    Fills small holes and connects nearby regions. Kernel size determines
    the maximum hole size that will be FILLED.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask uint8.
    kernel_size : int
    iterations : int
    kernel_shape : str

    Returns
    -------
    np.ndarray
        Filled mask.
    """
    kernel = make_kernel(kernel_size, kernel_shape)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def apply_dilation(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
    kernel_shape: str = "ellipse",
) -> np.ndarray:
    """
    Apply morphological dilation (expand foreground).

    Parameters
    ----------
    mask : np.ndarray
    kernel_size : int
    iterations : int
    kernel_shape : str

    Returns
    -------
    np.ndarray
    """
    kernel = make_kernel(kernel_size, kernel_shape)
    return cv2.dilate(mask, kernel, iterations=iterations)


def apply_erosion(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
    kernel_shape: str = "ellipse",
) -> np.ndarray:
    """Apply morphological erosion (shrink foreground)."""
    kernel = make_kernel(kernel_size, kernel_shape)
    return cv2.erode(mask, kernel, iterations=iterations)


def clean_mask(
    mask: np.ndarray,
    opening_kernel: int = 5,
    opening_iter: int = 2,
    closing_kernel: int = 15,
    closing_iter: int = 3,
    dilation_kernel: int = 3,
    dilation_iter: int = 1,
    kernel_shape: str = "ellipse",
) -> np.ndarray:
    """
    Apply the full morphological cleaning pipeline:
    1. Opening — remove noise
    2. Closing — fill holes
    3. Dilation — recover boundaries

    Parameters correspond to segmentation_config.yaml morphology section.
    """
    result = apply_opening(mask, opening_kernel, opening_iter, kernel_shape)
    result = apply_closing(result, closing_kernel, closing_iter, kernel_shape)
    result = apply_dilation(result, dilation_kernel, dilation_iter, kernel_shape)
    return result


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill enclosed holes in a binary mask using flood fill.

    More aggressive than morphological closing — fills all enclosed regions
    regardless of their size.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask uint8.

    Returns
    -------
    np.ndarray
        Hole-filled mask.
    """
    filled = mask.copy()
    h, w = filled.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(filled, flood_mask, (0, 0), 255)
    filled_inv = cv2.bitwise_not(filled)
    return cv2.bitwise_or(mask, filled_inv)
