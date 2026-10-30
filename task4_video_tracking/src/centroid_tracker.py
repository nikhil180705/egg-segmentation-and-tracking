"""
task4_video_tracking/src/centroid_tracker.py
=============================================
Centroid-based egg tracker using the Hungarian algorithm.

Algorithm Overview:
-------------------
The Hungarian algorithm (also called the Munkres algorithm) finds the
optimal assignment between predicted detections and existing tracked objects
by minimizing a cost matrix (Euclidean distance between centroids).

Tracking State Machine:
  NEW -> ACTIVE: Object confirmed on first detection
  ACTIVE -> LOST: Object not detected for max_missing_frames consecutive frames
  LOST -> ACTIVE: Object re-detected within max_missing_frames

Track ID Assignment:
  Each track gets a unique monotonically increasing integer ID.
  IDs are never reused, so a track's history can always be reconstructed.

Parameters (from tracking_config.yaml):
  max_distance: Maximum centroid distance (pixels) for assignment.
                Higher value = allows more object movement between frames.
                640x640 image: 100px recommended.
  max_missing_frames: Number of consecutive frames a track can be absent
                      before being declared lost and removed.
  min_hits: Minimum consecutive frames a track must be seen before being
            considered reliable (filters single-frame false positives).

References:
  - Hungarian algorithm: scipy.optimize.linear_sum_assignment
  - Multi-object tracking: Bewley et al., "Simple Online and Realtime Tracking" (SORT)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from shared.logging_utils import get_logger

logger = get_logger(__name__)


class Track:
    """Single tracked object."""

    def __init__(
        self,
        track_id: int,
        centroid: Tuple[float, float],
        bbox: Tuple[int, int, int, int],
        frame_idx: int,
    ) -> None:
        self.track_id = track_id
        self.centroid = centroid
        self.bbox = bbox
        self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.consecutive_hits = 1
        self.total_hits = 1
        self.missing_frames = 0
        self.is_confirmed = False
        # Full centroid history for trajectory visualization
        self.centroid_history: List[Tuple[float, float]] = [centroid]
        self.frame_history: List[int] = [frame_idx]

    def update(
        self,
        centroid: Tuple[float, float],
        bbox: Tuple[int, int, int, int],
        frame_idx: int,
    ) -> None:
        """Update track with new detection."""
        self.centroid = centroid
        self.bbox = bbox
        self.last_frame = frame_idx
        self.consecutive_hits += 1
        self.total_hits += 1
        self.missing_frames = 0
        self.centroid_history.append(centroid)
        self.frame_history.append(frame_idx)

    def mark_missing(self) -> None:
        """Increment missing frames counter."""
        self.missing_frames += 1
        self.consecutive_hits = 0

    @property
    def age(self) -> int:
        return self.last_frame - self.first_frame + 1

    def to_dict(self) -> Dict:
        return {
            "track_id": self.track_id,
            "centroid": self.centroid,
            "bbox": self.bbox,
            "age": self.age,
            "hits": self.total_hits,
            "missing": self.missing_frames,
            "is_confirmed": self.is_confirmed,
        }


class CentroidTracker:
    """
    Hungarian algorithm-based centroid tracker.

    Parameters
    ----------
    max_distance : float
        Maximum centroid displacement (pixels) between frames to allow assignment.
    max_missing_frames : int
        Frames a track can be absent before being removed.
    min_hits : int
        Minimum detections before a track is confirmed (reduces false positives).
    """

    def __init__(
        self,
        max_distance: float = 100.0,
        max_missing_frames: int = 5,
        min_hits: int = 2,
    ) -> None:
        self.max_distance = max_distance
        self.max_missing_frames = max_missing_frames
        self.min_hits = min_hits
        self._next_id = 1
        self._tracks: Dict[int, Track] = {}

    @property
    def active_tracks(self) -> List[Track]:
        """Return currently active (not lost) tracks."""
        return [t for t in self._tracks.values() if t.missing_frames == 0]

    @property
    def confirmed_tracks(self) -> List[Track]:
        """Return confirmed active tracks (above min_hits threshold)."""
        return [t for t in self.active_tracks if t.is_confirmed]

    def update(
        self,
        detections: List[Dict],
        frame_idx: int,
    ) -> List[Dict]:
        """
        Update tracker with new detections for a frame.

        Parameters
        ----------
        detections : list of dicts (must have 'centroid' key)
        frame_idx : int

        Returns
        -------
        list of active track dicts (including confirmed status)
        """
        detection_centroids = [d["centroid"] for d in detections]
        detection_bboxes = [d.get("bbox", (0, 0, 0, 0)) for d in detections]

        # Case 1: No existing tracks — create new track for each detection
        if not self._tracks:
            for centroid, bbox in zip(detection_centroids, detection_bboxes):
                self._create_track(centroid, bbox, frame_idx)
            self._finalize_update()
            return [t.to_dict() for t in self.active_tracks]

        # Case 2: No new detections — mark all tracks as missing
        if not detection_centroids:
            for track in self._tracks.values():
                track.mark_missing()
            self._remove_lost_tracks()
            self._finalize_update()
            return [t.to_dict() for t in self.active_tracks]

        # Case 3: Both detections and tracks exist — Hungarian matching
        track_ids = list(self._tracks.keys())
        track_centroids = [self._tracks[tid].centroid for tid in track_ids]

        # Build cost matrix (Euclidean distance)
        cost_matrix = self._build_cost_matrix(track_centroids, detection_centroids)

        # Apply Hungarian algorithm
        track_idxs, det_idxs = linear_sum_assignment(cost_matrix)

        # Process assignments
        assigned_tracks = set()
        assigned_dets = set()

        for ti, di in zip(track_idxs, det_idxs):
            if cost_matrix[ti, di] <= self.max_distance:
                tid = track_ids[ti]
                self._tracks[tid].update(
                    detection_centroids[di],
                    detection_bboxes[di],
                    frame_idx,
                )
                assigned_tracks.add(ti)
                assigned_dets.add(di)

        # Mark unmatched tracks as missing
        for ti, tid in enumerate(track_ids):
            if ti not in assigned_tracks:
                self._tracks[tid].mark_missing()

        # Create new tracks for unmatched detections
        for di in range(len(detection_centroids)):
            if di not in assigned_dets:
                self._create_track(detection_centroids[di], detection_bboxes[di], frame_idx)

        self._remove_lost_tracks()
        self._finalize_update()

        return [t.to_dict() for t in self.active_tracks]

    def _build_cost_matrix(
        self,
        track_centroids: List[Tuple],
        det_centroids: List[Tuple],
    ) -> np.ndarray:
        """Build pairwise Euclidean distance cost matrix."""
        n_tracks = len(track_centroids)
        n_dets = len(det_centroids)
        cost = np.zeros((n_tracks, n_dets), dtype=np.float32)
        for i, tc in enumerate(track_centroids):
            for j, dc in enumerate(det_centroids):
                cost[i, j] = np.sqrt((tc[0] - dc[0]) ** 2 + (tc[1] - dc[1]) ** 2)
        return cost

    def _create_track(
        self,
        centroid: Tuple[float, float],
        bbox: Tuple[int, int, int, int],
        frame_idx: int,
    ) -> Track:
        """Create and register a new track."""
        track = Track(self._next_id, centroid, bbox, frame_idx)
        self._tracks[self._next_id] = track
        self._next_id += 1
        return track

    def _remove_lost_tracks(self) -> None:
        """Remove tracks that have been missing too long."""
        to_remove = [
            tid for tid, t in self._tracks.items()
            if t.missing_frames > self.max_missing_frames
        ]
        for tid in to_remove:
            logger.debug(f"Track {tid} lost after {self._tracks[tid].total_hits} total hits")
            del self._tracks[tid]

    def _finalize_update(self) -> None:
        """Update confirmation status for all tracks."""
        for track in self._tracks.values():
            if track.total_hits >= self.min_hits:
                track.is_confirmed = True

    def get_all_tracks(self) -> Dict[int, Track]:
        """Return all currently tracked objects."""
        return self._tracks.copy()
