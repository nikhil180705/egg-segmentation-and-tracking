"""
task4_video_tracking/src/visualizer.py
=======================================
Video annotation and visualization for tracked eggs.

Renders:
- Per-track bounding boxes with unique colors
- Track ID labels
- Centroid dots
- Trajectory tails (last N frames of centroid path)
- Frame counter overlay
- Track count summary
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from shared.logging_utils import get_logger

logger = get_logger(__name__)

# Deterministic color palette for track IDs (max 50 distinct colors)
_COLOR_PALETTE = [
    (255, 80, 80), (80, 255, 80), (80, 80, 255), (255, 255, 80),
    (255, 80, 255), (80, 255, 255), (255, 160, 80), (160, 80, 255),
    (80, 255, 160), (255, 80, 160), (160, 255, 80), (80, 160, 255),
    (255, 200, 100), (100, 255, 200), (200, 100, 255), (255, 100, 200),
    (200, 255, 100), (100, 200, 255), (255, 140, 0), (0, 255, 140),
    (140, 0, 255), (255, 0, 140), (0, 140, 255), (140, 255, 0),
    (255, 220, 0), (0, 255, 220), (220, 0, 255), (255, 0, 220),
    (0, 220, 255), (220, 255, 0), (180, 100, 80), (80, 180, 100),
    (100, 80, 180), (180, 80, 100), (100, 180, 80), (80, 100, 180),
    (255, 60, 120), (60, 255, 120), (120, 60, 255), (255, 120, 60),
    (120, 255, 60), (60, 120, 255), (200, 200, 80), (80, 200, 200),
    (200, 80, 200), (255, 190, 190), (190, 255, 190), (190, 190, 255),
    (255, 255, 190), (190, 255, 255),
]


def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """Get a consistent BGR color for a given track ID."""
    # Note: OpenCV uses BGR, so we flip the RGB palette
    rgb = _COLOR_PALETTE[track_id % len(_COLOR_PALETTE)]
    return (rgb[2], rgb[1], rgb[0])  # Convert to BGR


def draw_tracks(
    frame: np.ndarray,
    active_tracks: List[Dict],
    trajectory_history: Dict[int, List[Tuple[float, float]]],
    frame_idx: int,
    trajectory_length: int = 30,
    show_unconfirmed: bool = False,
) -> np.ndarray:
    """
    Draw tracking annotations on a video frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR video frame (modified in-place and returned).
    active_tracks : list of track dicts
        Output from CentroidTracker.update().
    trajectory_history : dict
        Maps track_id -> list of (cx, cy) centroid positions.
    frame_idx : int
        Current frame index (for display).
    trajectory_length : int
        How many past frames to draw trajectory for.
    show_unconfirmed : bool
        Whether to draw unconfirmed (tentative) tracks.

    Returns
    -------
    np.ndarray
        Annotated frame.
    """
    annotated = frame.copy()
    confirmed_count = 0

    for track in active_tracks:
        tid = track["track_id"]
        confirmed = track.get("is_confirmed", False)

        if not confirmed and not show_unconfirmed:
            continue

        confirmed_count += 1
        color = get_track_color(tid)
        cx, cy = track["centroid"]
        cx, cy = int(cx), int(cy)
        x, y, w, h = [int(v) for v in track["bbox"]]

        # Draw bounding box
        thickness = 2 if confirmed else 1
        style = cv2.LINE_AA
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness, style)

        # Draw centroid dot
        cv2.circle(annotated, (cx, cy), 5, color, -1, style)
        cv2.circle(annotated, (cx, cy), 7, (255, 255, 255), 1, style)  # White ring

        # Draw track ID label
        label = f"#{tid}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        lw, lh = label_size
        cv2.rectangle(annotated, (x, y - lh - 6), (x + lw + 4, y), color, -1)
        cv2.putText(
            annotated, label, (x + 2, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
        )

        # Draw trajectory tail
        if tid in trajectory_history:
            history = trajectory_history[tid][-trajectory_length:]
            if len(history) > 1:
                for i in range(1, len(history)):
                    alpha = i / len(history)  # Fade older points
                    pt1 = (int(history[i - 1][0]), int(history[i - 1][1]))
                    pt2 = (int(history[i][0]), int(history[i][1]))
                    faded_color = tuple(int(c * alpha) for c in color)
                    cv2.line(annotated, pt1, pt2, faded_color, 2, cv2.LINE_AA)

    # Draw HUD overlay
    _draw_hud(annotated, frame_idx, confirmed_count)

    return annotated


def _draw_hud(
    frame: np.ndarray,
    frame_idx: int,
    egg_count: int,
) -> None:
    """Draw heads-up display overlay (frame counter and egg count)."""
    h, w = frame.shape[:2]
    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(
        frame,
        f"Frame: {frame_idx:04d}",
        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA
    )
    cv2.putText(
        frame,
        f"Eggs tracked: {egg_count}",
        (160, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 255, 150), 2, cv2.LINE_AA
    )
    cv2.putText(
        frame,
        "Nosh Robotics CV — Egg Tracker",
        (w - 300, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 200), 1, cv2.LINE_AA
    )
