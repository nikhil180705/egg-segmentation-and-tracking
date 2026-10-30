"""
tests/test_evaluation_metrics.py
================================
Unit tests for segmentation evaluation metrics and Hungarian matching.
"""

import numpy as np
import pytest

from task3_evaluation.src.metrics import (
    compute_instance_metrics,
    compute_mask_iou,
    compute_pixel_metrics,
    hungarian_match_instances,
)


def test_pixel_metrics_perfect_overlap():
    """Identical masks should produce IoU=1.0 and Dice/F1=1.0."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 1

    metrics = compute_pixel_metrics(mask, mask)
    assert pytest.approx(metrics["iou"], abs=1e-4) == 1.0
    assert pytest.approx(metrics["dice"], abs=1e-4) == 1.0
    assert pytest.approx(metrics["precision"], abs=1e-4) == 1.0
    assert pytest.approx(metrics["recall"], abs=1e-4) == 1.0
    assert pytest.approx(metrics["f1"], abs=1e-4) == 1.0
    assert metrics["tp"] == 3600
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0


def test_pixel_metrics_disjoint_masks():
    """Disjoint masks should produce IoU=0.0 and Precision=Recall=F1=0.0."""
    pred = np.zeros((100, 100), dtype=np.uint8)
    pred[0:20, 0:20] = 1

    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[50:70, 50:70] = 1

    metrics = compute_pixel_metrics(pred, gt)
    assert pytest.approx(metrics["iou"], abs=1e-4) == 0.0
    assert pytest.approx(metrics["dice"], abs=1e-4) == 0.0
    assert pytest.approx(metrics["precision"], abs=1e-4) == 0.0
    assert pytest.approx(metrics["recall"], abs=1e-4) == 0.0
    assert metrics["tp"] == 0
    assert metrics["fp"] == 400
    assert metrics["fn"] == 400


def test_pixel_metrics_partial_overlap():
    """Verify known 50% intersection math."""
    # A has area 20x20 = 400 (cols 0 to 20)
    # B has area 20x20 = 400 (cols 10 to 30)
    # Intersection = 20x10 = 200
    # Union = 400 + 400 - 200 = 600
    # IoU = 200 / 600 = 1/3 ~ 0.333333
    # Dice = 2*200 / (400 + 400) = 400/800 = 0.5
    pred = np.zeros((100, 100), dtype=np.uint8)
    pred[0:20, 0:20] = 1

    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[0:20, 10:30] = 1

    metrics = compute_pixel_metrics(pred, gt)
    assert pytest.approx(metrics["iou"], abs=1e-3) == 1.0 / 3.0
    assert pytest.approx(metrics["dice"], abs=1e-3) == 0.5


def test_hungarian_match_instances():
    """Test instance matching with Hungarian assignment."""
    # Create 2 GT instances
    gt1 = np.zeros((100, 100), dtype=np.uint8)
    gt1[10:30, 10:30] = 1

    gt2 = np.zeros((100, 100), dtype=np.uint8)
    gt2[60:80, 60:80] = 1

    # Create 2 Pred instances: one near gt1, one near gt2
    pred1 = np.zeros((100, 100), dtype=np.uint8)
    pred1[12:32, 12:32] = 1  # strong overlap with gt1

    pred2 = np.zeros((100, 100), dtype=np.uint8)
    pred2[58:78, 58:78] = 1  # strong overlap with gt2

    matches, unmatched_p, unmatched_g, iou_matrix = hungarian_match_instances(
        [pred1, pred2], [gt1, gt2], iou_threshold=0.25
    )

    assert len(matches) == 2
    assert len(unmatched_p) == 0
    assert len(unmatched_g) == 0
    assert iou_matrix.shape == (2, 2)


def test_hungarian_match_with_unmatched():
    """Test matching with an extra false positive and false negative."""
    gt1 = np.zeros((100, 100), dtype=np.uint8)
    gt1[10:30, 10:30] = 1

    gt_missed = np.zeros((100, 100), dtype=np.uint8)
    gt_missed[70:90, 70:90] = 1

    pred1 = np.zeros((100, 100), dtype=np.uint8)
    pred1[10:30, 10:30] = 1  # matches gt1 perfectly

    pred_false = np.zeros((100, 100), dtype=np.uint8)
    pred_false[40:50, 40:50] = 1  # in the middle, no overlap

    matches, unmatched_p, unmatched_g, _ = hungarian_match_instances(
        [pred1, pred_false], [gt1, gt_missed], iou_threshold=0.25
    )

    assert len(matches) == 1
    assert matches[0] == (0, 0)
    assert unmatched_p == [1]
    assert unmatched_g == [1]
