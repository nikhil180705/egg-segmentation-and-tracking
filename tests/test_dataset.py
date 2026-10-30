"""
tests/test_dataset.py
=====================
Unit tests for dataset loading, manifest validation, and split integrity.
"""

import json
from pathlib import Path

import pytest
import cv2
import numpy as np

from shared.image_utils import collect_images, load_image_bgr


def test_manifest_exists():
    """Verify data/split_manifest.json exists and is valid JSON."""
    manifest_path = Path("data/split_manifest.json")
    assert manifest_path.exists(), "data/split_manifest.json must exist"
    
    with open(manifest_path, "r") as f:
        data = json.load(f)
    
    assert "dev" in data, "Manifest must contain 'dev' split"
    assert "eval" in data, "Manifest must contain 'eval' split"
    assert len(data["dev"]) > 0, "Dev split must not be empty"
    assert len(data["eval"]) > 0, "Eval split must not be empty"


def test_split_directories_exist():
    """Check that dev and eval directories exist and contain images."""
    dev_dir = Path("data/dev")
    eval_dir = Path("data/eval")

    assert dev_dir.exists(), "data/dev directory must exist"
    assert eval_dir.exists(), "data/eval directory must exist"

    dev_images = collect_images(dev_dir)
    eval_images = collect_images(eval_dir)

    assert len(dev_images) == 88, f"Expected 88 dev images, found {len(dev_images)}"
    assert len(eval_images) == 22, f"Expected 22 eval images, found {len(eval_images)}"


def test_data_leakage_absence():
    """Ensure no image filename is shared between dev and eval sets."""
    dev_images = set(p.name for p in collect_images(Path("data/dev")))
    eval_images = set(p.name for p in collect_images(Path("data/eval")))

    intersection = dev_images.intersection(eval_images)
    assert len(intersection) == 0, f"Data leakage detected! Shared files: {intersection}"


def test_image_loading_and_shape():
    """Test loading a sample image and verify it has valid 3-channel dimensions."""
    dev_images = collect_images(Path("data/dev"))
    assert len(dev_images) > 0

    sample_img = load_image_bgr(dev_images[0])
    assert sample_img is not None, "Failed to load sample image"
    assert isinstance(sample_img, np.ndarray), "Image must be numpy array"
    assert sample_img.ndim == 3, "Image must have 3 dimensions (H, W, C)"
    assert sample_img.shape[2] == 3, "Image must have 3 color channels"
    assert sample_img.dtype == np.uint8, "Image dtype must be uint8"
