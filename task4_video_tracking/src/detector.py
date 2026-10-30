"""
task4_video_tracking/src/detector.py
=====================================
Frame-by-frame egg detector for video.

Uses the same classical pipeline from Task 2 (Otsu + morphology + contours)
applied to each video frame. Classical detection is preferred over SAM for
video because:
- Speed: ~0.07s/frame vs ~22s/frame with SAM
- Real-time: classical can process video at meaningful frame rates
- Robustness: consistent parameters across frames

Also provides a SAM-based detector as an option for higher accuracy
on static video shots (slower but more precise).
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from shared.logging_utils import get_logger
from task2_classical_segmentation.src.egg_segmenter import ClassicalEggSegmenter

logger = get_logger(__name__)


class EggDetector:
    """
    Egg detector for video frames.

    Uses classical segmentation to detect eggs in each frame and returns
    bounding boxes and centroids for tracking.
    """

    def __init__(
        self,
        config_path: str = "configs/segmentation_config.yaml",
        config_override: Optional[Dict] = None,
    ) -> None:
        self._segmenter = ClassicalEggSegmenter(
            config_path=config_path,
            config_override=config_override or {},
        )
        self._frame_count = 0

    def detect(
        self,
        frame_bgr: np.ndarray,
    ) -> List[Dict]:
        """
        Detect eggs in a single video frame.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Video frame in BGR format.

        Returns
        -------
        list of dicts, each with:
            - 'centroid': (cx, cy) float tuple
            - 'bbox': (x, y, w, h) int tuple
            - 'area': float
            - 'circularity': float
            - 'mask': np.ndarray (H, W) binary uint8
        """
        self._frame_count += 1

        # Run segmenter on this frame directly in-memory
        result = self._segmenter.segment_image(frame_bgr)

        detections = []
        egg_features = result.get("egg_features", [])
        masks = result.get("masks", [])

        for i, (feat, mask) in enumerate(zip(egg_features, masks)):
            cx, cy = feat["centroid"]
            x, y, w, h = feat["bounding_box"]
            detections.append({
                "centroid": (float(cx), float(cy)),
                "bbox": (int(x), int(y), int(w), int(h)),
                "area": float(feat["area"]),
                "circularity": float(feat["circularity"]),
                "mask": mask,
            })

        return detections
