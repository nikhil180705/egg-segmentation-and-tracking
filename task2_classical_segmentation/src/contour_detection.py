"""
task2_classical_segmentation/src/contour_detection.py
=====================================================
Contour detection and geometric filtering for egg identification.

Filters candidate regions using:
- Area (eliminates small noise artifacts and massive dishes/cookware/tables)
- Aspect ratio (eggs have natural elongation; eliminates thin lines)
- Circularity (eggs are elliptical; eliminates jagged, sprawling background blobs)
- Solidity (eggs are convex; eliminates concave dishes, bowls, and hollow shadows)
- Convexity & Bounding-box dimensions
- Border contact safeguards (rejects background frames touching 3 or 4 borders)
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from shared.logging_utils import get_logger
from shared.mask_utils import ensure_binary_uint8

logger = get_logger(__name__)


def find_contours(mask: np.ndarray) -> List[np.ndarray]:
    """Find external contours in a binary mask."""
    binary = ensure_binary_uint8(mask)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


def compute_contour_features(
    contour: np.ndarray,
    image_shape: Optional[Tuple[int, int]] = (640, 640),
) -> Dict:
    """
    Compute geometric features of a contour for egg filtering.

    Parameters
    ----------
    contour : np.ndarray
        Single contour array from cv2.findContours.
    image_shape : (H, W), optional
        Dimensions for border contact evaluation.

    Returns
    -------
    dict with keys:
        area, perimeter, circularity, solidity, aspect_ratio,
        bounding_box (x, y, w, h), centroid (cx, cy),
        border_touches, touches_left, touches_top, touches_right, touches_bottom
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)

    # Circularity: 4π·area / perimeter² = 1 for perfect circle
    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

    # Solidity: area / convex_hull_area
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = (area / hull_area) if hull_area > 0 else 0.0

    # Bounding box
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0.0

    # Centroid via spatial moments
    M = cv2.moments(contour)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx, cy = x + w / 2, y + h / 2

    # Border touches
    img_h, img_w = (image_shape[0], image_shape[1]) if image_shape else (640, 640)
    touches_left = (x <= 2)
    touches_top = (y <= 2)
    touches_right = (x + w >= img_w - 2)
    touches_bottom = (y + h >= img_h - 2)
    border_touches = int(touches_left + touches_top + touches_right + touches_bottom)

    return {
        "area": area,
        "perimeter": perimeter,
        "circularity": circularity,
        "solidity": solidity,
        "aspect_ratio": aspect_ratio,
        "bounding_box": (x, y, w, h),
        "centroid": (cx, cy),
        "border_touches": border_touches,
        "touches_left": touches_left,
        "touches_top": touches_top,
        "touches_right": touches_right,
        "touches_bottom": touches_bottom,
    }


def is_egg_contour(
    features: Dict,
    min_area: float = 1200,
    max_area: float = 95000,
    min_circularity: float = 0.35,
    min_solidity: float = 0.78,
    min_aspect_ratio: float = 0.45,
    max_aspect_ratio: float = 2.30,
    max_border_touch_count: int = 2,
    max_border_touch_area: float = 45000,
    **kwargs,
) -> Tuple[bool, str]:
    """
    Determine if a contour is a plausible egg based on geometric criteria.

    Parameters
    ----------
    features : dict
        From compute_contour_features().
    min_area, max_area : float
        Area bounds in pixels².
    min_circularity : float
        Minimum circularity (0-1).
    min_solidity : float
        Minimum solidity (0-1).
    min_aspect_ratio, max_aspect_ratio : float
        Bounding box aspect ratio range.
    max_border_touch_count : int
        Maximum number of image borders touched before rejecting as background.

    Returns
    -------
    is_egg : bool
    reason : str
        Reason for rejection (empty if accepted).
    """
    area = features["area"]
    if area < min_area:
        return False, f"area_too_small({area:.0f}<{min_area})"
    if area > max_area:
        return False, f"area_too_large({area:.0f}>{max_area})"

    circ = features["circularity"]
    if circ < min_circularity:
        return False, f"circularity_too_low({circ:.3f}<{min_circularity})"

    solid = features["solidity"]
    if solid < min_solidity:
        return False, f"solidity_too_low({solid:.3f}<{min_solidity})"

    ar = features["aspect_ratio"]
    if ar < min_aspect_ratio or ar > max_aspect_ratio:
        return False, f"aspect_ratio_out_of_range({ar:.2f})"

    # Border touch check
    bt = features.get("border_touches", 0)
    if bt >= 3:
        return False, f"border_touches_too_high({bt}>=3)"
    if bt >= max_border_touch_count and area > max_border_touch_area:
        return False, f"large_border_touch({bt} borders, area={area:.0f})"

    return True, ""


def filter_egg_contours(
    contours: List[np.ndarray],
    filter_params: Optional[Dict] = None,
    image_shape: Optional[Tuple[int, int]] = (640, 640),
) -> Tuple[List[np.ndarray], List[Dict], List[Dict]]:
    """
    Filter contours to keep only egg candidates.

    Parameters
    ----------
    contours : list of np.ndarray
    filter_params : dict, optional
        Overrides for is_egg_contour() parameters.
    image_shape : (H, W), optional

    Returns
    -------
    egg_contours : list of np.ndarray
        Accepted contours.
    egg_features : list of dict
        Features for each accepted contour.
    rejected : list of dict
        Info about rejected contours.
    """
    params = filter_params or {}
    egg_contours = []
    egg_features = []
    rejected = []

    for contour in contours:
        features = compute_contour_features(contour, image_shape=image_shape)
        accepted, reason = is_egg_contour(features, **params)
        if accepted:
            egg_contours.append(contour)
            egg_features.append(features)
        else:
            rejected.append({**features, "rejection_reason": reason})

    return egg_contours, egg_features, rejected


def contours_to_masks(
    contours: List[np.ndarray],
    image_shape: Tuple[int, int],
) -> List[np.ndarray]:
    """Convert contour list to individual binary masks."""
    h, w = image_shape[:2]
    masks = []
    for contour in contours:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        masks.append(mask)
    return masks


def contours_to_combined_mask(
    contours: List[np.ndarray],
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Draw all contours onto a single binary mask."""
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if contours:
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    return mask
