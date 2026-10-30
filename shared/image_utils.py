"""
shared/image_utils.py
=====================
Image I/O and preprocessing utilities shared across all tasks.

Design decisions:
- All functions return numpy arrays (BGR for OpenCV, RGB for matplotlib/SAM).
- Explicit color-space conversion to prevent subtle BGR/RGB bugs.
- Pathlib-based paths throughout.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from shared.logging_utils import get_logger

logger = get_logger(__name__)

# Supported image extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def load_image_bgr(path: Union[str, Path]) -> np.ndarray:
    """
    Load an image as BGR numpy array (OpenCV default).

    Parameters
    ----------
    path : str or Path
        Path to the image file.

    Returns
    -------
    np.ndarray
        Image array of shape (H, W, 3), dtype uint8, BGR.

    Raises
    ------
    FileNotFoundError
        If the image file does not exist.
    ValueError
        If the file cannot be decoded as an image.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot decode image: {path}")
    return img


def load_image_rgb(path: Union[str, Path]) -> np.ndarray:
    """
    Load an image as RGB numpy array (required by SAM and matplotlib).

    Parameters
    ----------
    path : str or Path
        Path to the image file.

    Returns
    -------
    np.ndarray
        Image array of shape (H, W, 3), dtype uint8, RGB.
    """
    img_bgr = load_image_bgr(path)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def save_image(img: np.ndarray, path: Union[str, Path], is_rgb: bool = False) -> None:
    """
    Save an image to disk.

    Parameters
    ----------
    img : np.ndarray
        Image to save. Can be BGR or RGB (set is_rgb=True for the latter).
    path : str or Path
        Destination path; parent directories are created if needed.
    is_rgb : bool
        If True, convert RGB → BGR before saving.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if is_rgb and img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    success = cv2.imwrite(str(path), img)
    if not success:
        raise IOError(f"Failed to write image to {path}")
    logger.debug(f"Saved image: {path}")


def resize_image(
    img: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    max_side: Optional[int] = None,
    keep_aspect: bool = True,
) -> np.ndarray:
    """
    Resize an image.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    target_size : (width, height), optional
        Exact target size.
    max_side : int, optional
        Scale so the longer side equals max_side (aspect-preserving).
    keep_aspect : bool
        Maintain aspect ratio when using target_size.

    Returns
    -------
    np.ndarray
        Resized image.
    """
    h, w = img.shape[:2]

    if max_side is not None:
        scale = max_side / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if target_size is not None:
        tw, th = target_size
        if keep_aspect:
            scale = min(tw / w, th / h)
            new_w, new_h = int(w * scale), int(h * scale)
        else:
            new_w, new_h = tw, th
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img


def collect_images(directory: Union[str, Path], recursive: bool = False) -> list[Path]:
    """
    Return a sorted list of image paths in a directory.

    Parameters
    ----------
    directory : str or Path
    recursive : bool
        If True, search subdirectories.

    Returns
    -------
    list[Path]
        Sorted list of image file paths.
    """
    directory = Path(directory)
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []

    pattern = "**/*" if recursive else "*"
    paths = [
        p for p in directory.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(paths)


def get_image_info(path: Union[str, Path]) -> dict:
    """
    Return basic statistics about an image without fully loading it into memory.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    dict with keys: path, width, height, channels, file_size_bytes
    """
    path = Path(path)
    img = load_image_bgr(path)
    h, w = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1
    return {
        "path": str(path),
        "filename": path.name,
        "width": w,
        "height": h,
        "channels": channels,
        "file_size_bytes": path.stat().st_size,
        "aspect_ratio": round(w / h, 3),
    }


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert BGR array to RGB."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    """Convert RGB array to BGR."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
