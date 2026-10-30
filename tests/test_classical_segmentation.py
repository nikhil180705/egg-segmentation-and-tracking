"""
tests/test_classical_segmentation.py
====================================
Unit tests for Task 2 classical computer vision segmentation pipeline.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from task2_classical_segmentation.src.contour_detection import filter_egg_contours, find_contours
from task2_classical_segmentation.src.egg_segmenter import ClassicalEggSegmenter
from task2_classical_segmentation.src.morphology import clean_mask, fill_holes
from task2_classical_segmentation.src.preprocessing import preprocess_image


def test_preprocessing():
    """Verify preprocessing outputs proper shapes and channels."""
    img = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    blurred, hsv, gray = preprocess_image(img, gaussian_kernel=5, gaussian_sigma=0.0)

    assert blurred.shape == img.shape
    assert hsv.shape == img.shape
    assert gray.shape == (200, 200)
    assert gray.dtype == np.uint8


def test_morphology_clean_mask():
    """Verify clean_mask removes isolated pixel noise and retains connected regions."""
    # Create mask with one big egg-like circle and isolated salt noise
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (50, 50), 20, 255, -1)  # Area ~ 1256 px
    mask[5, 5] = 255  # Isolated noise
    mask[10, 80] = 255  # Isolated noise

    cleaned = clean_mask(
        mask,
        opening_kernel=5,
        opening_iter=1,
        closing_kernel=5,
        closing_iter=1,
    )

    # Isolated noise should be removed
    assert cleaned[5, 5] == 0
    assert cleaned[10, 80] == 0
    # Core circle should remain intact
    assert cleaned[50, 50] == 255


def test_filter_egg_contours():
    """Verify contour filtering correctly accepts egg shapes and rejects tiny artifacts."""
    img_h, img_w = 640, 640
    test_mask = np.zeros((img_h, img_w), dtype=np.uint8)

    # Valid egg-like ellipse: center (200, 200), axes (50, 35) -> area ~ pi*50*35 ~ 5497
    cv2.ellipse(test_mask, (200, 200), (50, 35), 0, 0, 360, 255, -1)

    # Tiny artifact: 3x3 square -> area 9
    test_mask[10:13, 10:13] = 255

    contours = find_contours(test_mask)
    assert len(contours) >= 2

    valid_contours, _, _ = filter_egg_contours(
        contours=contours,
        filter_params={
            "min_area": 500.0,
            "max_area": 150000.0,
            "min_circularity": 0.2,
            "min_aspect_ratio": 0.3,
            "max_aspect_ratio": 3.0,
        },
    )

    # Only the egg-like ellipse should pass
    assert len(valid_contours) == 1


def test_classical_segmenter_on_real_image():
    """Verify ClassicalEggSegmenter runs on an actual dataset image without error."""
    dev_images = list(Path("data/dev").glob("*.jpg"))
    if not dev_images:
        pytest.skip("No dev images available in data/dev")

    sample_path = dev_images[0]
    segmenter = ClassicalEggSegmenter(config_path="configs/segmentation_config.yaml")
    result = segmenter.segment_image(sample_path)

    assert result["error"] is None
    assert result["image_name"] == sample_path.name
    assert result["processing_time_s"] > 0
    assert "masks" in result
    assert isinstance(result["masks"], list)
