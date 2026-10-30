"""
tests/test_tracker.py
=====================
Unit tests for CentroidTracker with Hungarian algorithm assignment.
"""

import pytest

from task4_video_tracking.src.centroid_tracker import CentroidTracker, Track


def test_track_creation_and_update():
    """Test single Track object update and missing frame logic."""
    track = Track(track_id=1, centroid=(100.0, 100.0), bbox=(90, 90, 20, 20), frame_idx=0)
    assert track.track_id == 1
    assert track.total_hits == 1
    assert track.consecutive_hits == 1
    assert track.missing_frames == 0
    assert not track.is_confirmed

    # Update track at frame 1
    track.update(centroid=(102.0, 101.0), bbox=(91, 91, 20, 20), frame_idx=1)
    assert track.total_hits == 2
    assert track.consecutive_hits == 2
    assert len(track.centroid_history) == 2

    # Mark missing
    track.mark_missing()
    assert track.missing_frames == 1
    assert track.consecutive_hits == 0


def test_tracker_id_persistence():
    """Verify that a moving object retains the same track ID across frames."""
    tracker = CentroidTracker(max_distance=50.0, max_missing_frames=3, min_hits=2)

    # Frame 0: detection at (50, 50)
    det_f0 = [{"centroid": (50.0, 50.0), "bbox": (40, 40, 20, 20)}]
    tracks_f0 = tracker.update(det_f0, frame_idx=0)
    assert len(tracks_f0) == 1
    assigned_id = tracks_f0[0]["track_id"]

    # Frame 1: object moves slightly to (55, 53)
    det_f1 = [{"centroid": (55.0, 53.0), "bbox": (45, 43, 20, 20)}]
    tracks_f1 = tracker.update(det_f1, frame_idx=1)
    assert len(tracks_f1) == 1
    assert tracks_f1[0]["track_id"] == assigned_id
    assert tracks_f1[0]["is_confirmed"] is True  # Reached min_hits=2

    # Frame 2: object moves to (60, 56)
    det_f2 = [{"centroid": (60.0, 56.0), "bbox": (50, 46, 20, 20)}]
    tracks_f2 = tracker.update(det_f2, frame_idx=2)
    assert len(tracks_f2) == 1
    assert tracks_f2[0]["track_id"] == assigned_id


def test_tracker_multiple_objects_disambiguation():
    """Verify two distinct objects get separate IDs and don't swap."""
    tracker = CentroidTracker(max_distance=50.0, max_missing_frames=3, min_hits=1)

    # Frame 0: two objects far apart
    dets_f0 = [
        {"centroid": (50.0, 50.0), "bbox": (40, 40, 20, 20)},
        {"centroid": (300.0, 300.0), "bbox": (290, 290, 20, 20)},
    ]
    tracks_f0 = tracker.update(dets_f0, frame_idx=0)
    assert len(tracks_f0) == 2
    id_a = tracks_f0[0]["track_id"]
    id_b = tracks_f0[1]["track_id"]
    assert id_a != id_b

    # Frame 1: both objects move slightly
    dets_f1 = [
        {"centroid": (52.0, 51.0), "bbox": (42, 41, 20, 20)},
        {"centroid": (303.0, 301.0), "bbox": (293, 291, 20, 20)},
    ]
    tracks_f1 = tracker.update(dets_f1, frame_idx=1)
    assert len(tracks_f1) == 2

    # Find the tracks closest to (50, 50) and (300, 300)
    track_for_a = [t for t in tracks_f1 if t["track_id"] == id_a][0]
    track_for_b = [t for t in tracks_f1 if t["track_id"] == id_b][0]

    assert abs(track_for_a["centroid"][0] - 52.0) < 1.0
    assert abs(track_for_b["centroid"][0] - 303.0) < 1.0


def test_tracker_missing_and_deregistration():
    """Verify track is marked missing and eventually deregistered."""
    max_missing = 3
    tracker = CentroidTracker(max_distance=50.0, max_missing_frames=max_missing, min_hits=1)

    # Frame 0: detection
    tracker.update([{"centroid": (100.0, 100.0), "bbox": (90, 90, 20, 20)}], frame_idx=0)
    assert len(tracker.active_tracks) == 1

    # Frames 1, 2, 3: no detections (missing)
    for f in range(1, max_missing + 1):
        tracker.update([], frame_idx=f)
        assert len(tracker.active_tracks) == 0  # Not active because missing_frames > 0

    # Frame 4: exceeds max_missing_frames, track should be deleted from _tracks
    tracker.update([], frame_idx=max_missing + 1)
    assert len(tracker._tracks) == 0
