"""
task2_classical_segmentation/experiments/experiment_runner.py
=============================================================
Experiment tracking framework for classical segmentation.

Every experiment is:
1. Defined by a configuration override dict
2. Run on the development set
3. Scored by a proxy metric (since we don't have GT masks yet, we use
   detection rate and avg egg count consistency as proxy)
4. Logged to experiment_log.csv with all parameters and results

After Task 1 SAM masks are generated, Task 3 will compute IoU/Dice
between classical predictions and SAM reference masks.

Usage:
    python task2_classical_segmentation/experiments/experiment_runner.py
"""

import csv
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.image_utils import collect_images
from shared.logging_utils import get_logger
from task2_classical_segmentation.src.egg_segmenter import ClassicalEggSegmenter

logger = get_logger(__name__)

EXPERIMENT_LOG_PATH = Path("task2_classical_segmentation/experiments/experiment_log.csv")
RESULTS_DIR = Path("task2_classical_segmentation/experiments/results")
DEV_DIR = Path("data/dev")

CSV_FIELDS = [
    "exp_id", "timestamp", "technique", "description",
    "gaussian_kernel", "use_hsv", "hsv_lower", "hsv_upper",
    "use_otsu", "use_adaptive", "adaptive_block_size",
    "opening_kernel", "opening_iter", "closing_kernel", "closing_iter",
    "min_area", "max_area", "min_circularity", "min_solidity",
    "use_watershed", "watershed_dt_threshold",
    "detection_rate", "avg_eggs_per_image", "total_eggs",
    "zero_egg_images", "error_images", "avg_processing_time_s",
    "decision", "reason",
]


def run_experiment(
    exp_id: str,
    technique: str,
    description: str,
    config_override: Dict,
    max_images: Optional[int] = None,
    decision: str = "pending",
    reason: str = "",
) -> Dict[str, Any]:
    """
    Run a single experiment with the given config override.

    Parameters
    ----------
    exp_id : str
        Unique experiment identifier (e.g., "EXP001")
    technique : str
        Short name of the technique tested
    description : str
        Human-readable description of what was changed
    config_override : dict
        Config parameters to override from defaults
    max_images : int, optional
        Limit images for fast testing
    decision : str
        "accepted", "rejected", or "pending"
    reason : str
        Reason for the decision

    Returns
    -------
    dict
        Experiment results including metrics and metadata
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Experiment: {exp_id} — {technique}")
    logger.info(f"Description: {description}")
    logger.info(f"Config override: {json.dumps(config_override, indent=2)}")

    segmenter = ClassicalEggSegmenter(config_override=config_override)

    images = collect_images(DEV_DIR)
    if max_images:
        images = images[:max_images]

    results = []
    for img_path in images:
        result = segmenter.segment_image(img_path)
        results.append(result)

    # Compute metrics
    successful = [r for r in results if not r.get("error")]
    total_eggs = sum(r["num_eggs"] for r in successful)
    zero_egg = [r for r in successful if r["num_eggs"] == 0]
    error_images = [r for r in results if r.get("error")]
    detection_rate = len(successful) / len(results) if results else 0
    avg_eggs = total_eggs / len(successful) if successful else 0
    avg_time = sum(r["processing_time_s"] for r in results) / len(results) if results else 0

    # Extract config for logging
    cfg = segmenter.config
    prep = cfg.get("preprocessing", {})
    cs = cfg.get("color_space", {})
    morph = cfg.get("morphology", {})
    contour = cfg.get("contour_detection", {})
    ws = cfg.get("watershed", {})

    record = {
        "exp_id": exp_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "technique": technique,
        "description": description,
        "gaussian_kernel": prep.get("gaussian_blur_kernel", ""),
        "use_hsv": cs.get("use_hsv", ""),
        "hsv_lower": str(cs.get("hsv_lower", "")),
        "hsv_upper": str(cs.get("hsv_upper", "")),
        "use_otsu": cs.get("use_otsu_fallback", ""),
        "use_adaptive": cs.get("use_adaptive", ""),
        "adaptive_block_size": cs.get("adaptive_block_size", ""),
        "opening_kernel": morph.get("opening_kernel_size", ""),
        "opening_iter": morph.get("opening_iterations", ""),
        "closing_kernel": morph.get("closing_kernel_size", ""),
        "closing_iter": morph.get("closing_iterations", ""),
        "min_area": contour.get("min_contour_area", ""),
        "max_area": contour.get("max_contour_area", ""),
        "min_circularity": contour.get("min_circularity", ""),
        "min_solidity": contour.get("min_solidity", ""),
        "use_watershed": ws.get("use_watershed", ""),
        "watershed_dt_threshold": ws.get("distance_transform_threshold", ""),
        "detection_rate": round(detection_rate, 4),
        "avg_eggs_per_image": round(avg_eggs, 2),
        "total_eggs": total_eggs,
        "zero_egg_images": len(zero_egg),
        "error_images": len(error_images),
        "avg_processing_time_s": round(avg_time, 3),
        "decision": decision,
        "reason": reason,
    }

    # Append to CSV
    _append_to_csv(record)

    logger.info(
        f"EXP {exp_id} complete: detection_rate={detection_rate:.2%}, "
        f"avg_eggs={avg_eggs:.1f}, zero_egg={len(zero_egg)}, errors={len(error_images)}"
    )
    return {"record": record, "results": results}


def _append_to_csv(record: Dict) -> None:
    """Append experiment record to CSV log."""
    EXPERIMENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not EXPERIMENT_LOG_PATH.exists()
    with open(EXPERIMENT_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(record)
    logger.info(f"Experiment logged: {EXPERIMENT_LOG_PATH}")


# ── Experiment Suite ─────────────────────────────────────────────────────────

def run_all_experiments(max_images: Optional[int] = None) -> None:
    """
    Run the full suite of development experiments.

    Each experiment tests a specific technique or parameter variation.
    Results are logged to experiment_log.csv.
    """
    logger.info("Starting classical segmentation experiment suite")

    # EXP001: Baseline — HSV only, default params
    run_experiment(
        exp_id="EXP001",
        technique="HSV_only_baseline",
        description="Baseline: HSV thresholding only, no Otsu, default morphology",
        config_override={
            "color_space": {"use_hsv": True, "use_otsu_fallback": False},
            "watershed": {"use_watershed": False},
        },
        max_images=max_images,
        decision="baseline",
        reason="Establishing baseline detection capability",
    )

    # EXP002: Otsu only
    run_experiment(
        exp_id="EXP002",
        technique="Otsu_only",
        description="Otsu global thresholding only, no HSV",
        config_override={
            "color_space": {"use_hsv": False, "use_otsu_fallback": True},
            "watershed": {"use_watershed": False},
        },
        max_images=max_images,
        decision="compare",
        reason="Compare Otsu vs HSV on diverse backgrounds",
    )

    # EXP003: HSV + Otsu union (proposed approach)
    run_experiment(
        exp_id="EXP003",
        technique="HSV_plus_Otsu_union",
        description="Union of HSV and Otsu masks — catches eggs missed by either alone",
        config_override={
            "color_space": {"use_hsv": True, "use_otsu_fallback": True},
            "watershed": {"use_watershed": False},
        },
        max_images=max_images,
        decision="compare",
        reason="Union should improve recall",
    )

    # EXP004: HSV + Otsu + Watershed
    run_experiment(
        exp_id="EXP004",
        technique="HSV_Otsu_Watershed",
        description="Full pipeline: HSV+Otsu + watershed for touching eggs",
        config_override={
            "color_space": {"use_hsv": True, "use_otsu_fallback": True},
            "watershed": {"use_watershed": True, "distance_transform_threshold": 0.4},
        },
        max_images=max_images,
        decision="compare",
        reason="Watershed should separate touching egg clusters",
    )

    # EXP005: Tighter HSV (pure white eggs)
    run_experiment(
        exp_id="EXP005",
        technique="HSV_tight_white_eggs",
        description="Narrower HSV range for white eggs only (high Value, very low Sat)",
        config_override={
            "color_space": {
                "use_hsv": True,
                "use_otsu_fallback": True,
                "hsv_lower": [0, 0, 150],
                "hsv_upper": [179, 60, 255],
            },
        },
        max_images=max_images,
        decision="compare",
        reason="Test if tighter range reduces false positives on bright backgrounds",
    )

    # EXP006: Looser HSV (include brown eggs)
    run_experiment(
        exp_id="EXP006",
        technique="HSV_loose_brown_eggs",
        description="Wider HSV range to capture brown/dark eggs",
        config_override={
            "color_space": {
                "use_hsv": True,
                "use_otsu_fallback": True,
                "hsv_lower": [0, 0, 80],
                "hsv_upper": [179, 150, 255],
            },
        },
        max_images=max_images,
        decision="compare",
        reason="Check if wider range helps with darker eggs but increases false positives",
    )

    # EXP007: Larger morphological closing
    run_experiment(
        exp_id="EXP007",
        technique="Large_closing",
        description="Larger closing kernel (25x25) to fill bigger holes",
        config_override={
            "morphology": {
                "closing_kernel_size": 25,
                "closing_iterations": 4,
            },
        },
        max_images=max_images,
        decision="compare",
        reason="Some egg interiors have large holes due to shadows",
    )

    # EXP008: Smaller min_area filter
    run_experiment(
        exp_id="EXP008",
        technique="Small_min_area",
        description="Reduce min_area to 500 to catch smaller eggs in frame",
        config_override={
            "contour_detection": {"min_contour_area": 500},
        },
        max_images=max_images,
        decision="compare",
        reason="Some images have smaller eggs (farther from camera)",
    )

    # EXP009: Relaxed circularity
    run_experiment(
        exp_id="EXP009",
        technique="Relaxed_circularity",
        description="Lower circularity threshold (0.15) to catch more elongated eggs",
        config_override={
            "contour_detection": {
                "min_circularity": 0.15,
                "min_solidity": 0.55,
            },
        },
        max_images=max_images,
        decision="compare",
        reason="Some eggs are quite elongated; may be rejected by circularity filter",
    )

    # EXP010: Adaptive thresholding
    run_experiment(
        exp_id="EXP010",
        technique="Adaptive_threshold",
        description="Adaptive Gaussian thresholding for non-uniform lighting",
        config_override={
            "color_space": {
                "use_hsv": True,
                "use_otsu_fallback": False,
                "use_adaptive": True,
                "adaptive_block_size": 21,
                "adaptive_C": 5,
            },
        },
        max_images=max_images,
        decision="compare",
        reason="Test adaptive threshold for images with strong shadows",
    )

    logger.info("All experiments complete. See experiment_log.csv for results.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run classical segmentation experiments")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit images per experiment (for fast testing)")
    parser.add_argument("--exp-id", default=None,
                        help="Run only a specific experiment ID")
    args = parser.parse_args()

    run_all_experiments(max_images=args.max_images)
