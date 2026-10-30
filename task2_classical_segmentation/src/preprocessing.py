"""
task2_classical_segmentation/src/preprocessing.py
===================================================
Image preprocessing utilities specific to classical segmentation.

Provides:
- Gaussian, median, and edge-preserving bilateral blur for noise reduction
- Color space conversions with proper channel ordering (HSV, CIELAB, Grayscale)
- CLAHE (Contrast Limited Adaptive Histogram Equalization) for lighting normalization
- Shadow and uneven illumination mitigation
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from shared.logging_utils import get_logger

logger = get_logger(__name__)


def apply_gaussian_blur(
    img: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 0,
) -> np.ndarray:
    """
    Apply Gaussian blur for noise reduction.

    Parameters
    ----------
    img : np.ndarray
        Input image (any channels).
    kernel_size : int
        Must be odd. Larger = more smoothing.
    sigma : float
        Standard deviation. 0 = auto-computed from kernel size.

    Returns
    -------
    np.ndarray
        Blurred image.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)


def apply_median_blur(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply median blur (effective for salt-and-pepper noise).

    Parameters
    ----------
    img : np.ndarray
    kernel_size : int
        Must be odd.

    Returns
    -------
    np.ndarray
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(img, kernel_size)


def apply_bilateral_filter(
    img: np.ndarray,
    d: int = 7,
    sigma_color: float = 50.0,
    sigma_space: float = 50.0,
) -> np.ndarray:
    """
    Apply bilateral filter for edge-preserving noise reduction.

    Smooths flat egg shell texture and image sensor noise while preserving
    sharp outer egg contours.

    Parameters
    ----------
    img : np.ndarray
        Input image (grayscale or color).
    d : int
        Diameter of each pixel neighborhood.
    sigma_color : float
        Filter sigma in the color space.
    sigma_space : float
        Filter sigma in the coordinate space.

    Returns
    -------
    np.ndarray
        Edge-preserved filtered image.
    """
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def to_hsv(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image to HSV color space."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)


def to_grayscale(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def to_lab(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to CIELAB color space.

    LAB is perceptually uniform:
    - L: Luminance (0-255 in uint8)
    - a*: Green (-128) to Red (+127)
    - b*: Blue (-128) to Yellow (+127)
    """
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)


def apply_clahe(
    gray: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Improves local contrast, especially useful for images with uneven lighting,
    cast shadows, and subtle egg-to-background contrast.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image (H, W) uint8.
    clip_limit : float
        Contrast limiting factor.
    tile_grid_size : (int, int)
        Grid size for local histogram computation.

    Returns
    -------
    np.ndarray
        CLAHE-enhanced grayscale image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def preprocess_image(
    img_bgr: np.ndarray,
    gaussian_kernel: int = 5,
    gaussian_sigma: float = 0,
    use_clahe: bool = False,
    clahe_clip: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standard preprocessing pipeline returning three representations.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input BGR image.
    gaussian_kernel : int
    gaussian_sigma : float
    use_clahe : bool
    clahe_clip : float

    Returns
    -------
    blurred_bgr : np.ndarray
        Gaussian-blurred BGR image.
    hsv : np.ndarray
        HSV color space image.
    gray : np.ndarray
        Grayscale image (CLAHE-enhanced if requested).
    """
    blurred = apply_gaussian_blur(img_bgr, gaussian_kernel, gaussian_sigma)
    hsv = to_hsv(blurred)
    gray = to_grayscale(blurred)
    if use_clahe:
        gray = apply_clahe(gray, clip_limit=clahe_clip)
    return blurred, hsv, gray
