"""
task4_video_tracking/scripts/run_tracking.py
=============================================
Run egg tracking on the Nosh Robotics YouTube video.

Pipeline:
1. Download video with yt-dlp (best quality ≤ 720p)
2. For each frame: detect eggs (classical pipeline)
3. Update tracker (Hungarian centroid matching)
4. Draw annotations and write to output MP4
5. Export tracking log CSV (frame, track_id, cx, cy, bbox)

Usage:
    python task4_video_tracking/scripts/run_tracking.py

    # Custom video / options
    python task4_video_tracking/scripts/run_tracking.py \\
        --video data/videos/raw/nosh_eggs.mp4 \\
        --output task4_video_tracking/outputs/tracked.mp4 \\
        --max-frames 300
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.logging_utils import get_logger
from task4_video_tracking.src.centroid_tracker import CentroidTracker
from task4_video_tracking.src.detector import EggDetector
from task4_video_tracking.src.visualizer import draw_tracks

logger = get_logger(__name__)

VIDEO_URL = "https://www.youtube.com/watch?v=Y4JyV94y2_A"
DEFAULT_VIDEO_DIR = Path("data/videos/raw")
DEFAULT_OUTPUT_DIR = Path("task4_video_tracking/outputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run egg tracking on video.")
    parser.add_argument("--url", default=VIDEO_URL, help="YouTube URL")
    parser.add_argument("--video", default=None, help="Local video path (skips download)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--config", default="configs/tracking_config.yaml")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--skip-frames", type=int, default=1,
                        help="Process every N-th frame (1=all frames)")
    parser.add_argument("--no-download", action="store_true",
                        help="Use cached video without downloading")
    return parser.parse_args()


def download_video(url: str, output_dir: Path) -> Optional[Path]:
    """
    Download YouTube video using yt-dlp.

    Returns path to downloaded video or None on failure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "nosh_eggs.%(ext)s")

    # Check if already downloaded
    existing = list(output_dir.glob("nosh_eggs.*"))
    if existing:
        video_path = existing[0]
        logger.info(f"Video already downloaded: {video_path} ({video_path.stat().st_size/1e6:.1f}MB)")
        return video_path

    logger.info(f"Downloading video from: {url}")
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--extractor-args", "youtube:player_client=android",
                "--format", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]/best",
                "--merge-output-format", "mp4",
                "--output", output_template,
                "--no-playlist",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            logger.error(f"yt-dlp failed: {result.stderr}")
            return None

        downloaded = list(output_dir.glob("nosh_eggs.*"))
        if downloaded:
            logger.info(f"Downloaded: {downloaded[0]}")
            return downloaded[0]
        return None
    except Exception as e:
        logger.error(f"Download error: {e}")
        return None


# Type alias for clarity
Optional_Path = type(None)  # replaced below


def process_video(
    video_path: Path,
    output_dir: Path,
    tracking_config: dict,
    max_frames: Optional[int] = None,
    skip_frames: int = 1,
) -> dict:
    """
    Process video with egg detection and tracking.

    Parameters
    ----------
    video_path : Path
    output_dir : Path
    tracking_config : dict
    max_frames : int, optional
    skip_frames : int

    Returns
    -------
    dict with tracking statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"Video: {video_path.name}")
    logger.info(f"Resolution: {orig_w}x{orig_h} | FPS: {fps:.1f} | Frames: {total_frames}")

    # Resize for processing if large
    max_dim = tracking_config.get("preprocessing", {}).get("max_dimension", 640)
    scale = min(max_dim / orig_w, max_dim / orig_h, 1.0)
    proc_w = int(orig_w * scale)
    proc_h = int(orig_h * scale)

    # Output video writer
    output_fps = fps / skip_frames
    output_path = output_dir / "tracked_eggs.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (orig_w, orig_h))

    # Initialize detector and tracker
    detector = EggDetector(config_override=tracking_config.get("detector_override", {}))
    tracker_cfg = tracking_config.get("tracker", {})
    tracker = CentroidTracker(
        max_distance=float(tracker_cfg.get("max_distance", tracker_cfg.get("max_centroid_distance", 80.0))),
        max_missing_frames=int(tracker_cfg.get("max_missing_frames", tracker_cfg.get("max_disappeared", 15))),
        min_hits=int(tracker_cfg.get("min_hits", tracker_cfg.get("min_confirm_frames", 2))),
    )

    # Trajectory history for visualization
    trajectory_history = {}  # track_id -> list of (cx, cy)

    # CSV log writer
    csv_path = output_dir / "tracking_log.csv"
    csv_fields = ["frame_idx", "track_id", "cx", "cy", "x", "y", "w", "h", "confirmed", "age"]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    # Processing loop
    frame_idx = 0
    processed_frames = 0
    total_detections = 0
    max_simultaneous_eggs = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames and frame_idx >= max_frames:
            break

        frame_idx += 1

        # Skip frames
        if frame_idx % skip_frames != 0:
            writer.write(frame)  # Write unprocessed frame as-is
            continue

        processed_frames += 1

        # Resize for processing
        if scale < 1.0:
            small_frame = cv2.resize(frame, (proc_w, proc_h))
        else:
            small_frame = frame

        # Detect eggs
        try:
            detections = detector.detect(small_frame)
        except Exception as e:
            logger.warning(f"Frame {frame_idx}: detection error: {e}")
            detections = []

        # Scale detections back to original size
        if scale < 1.0:
            for det in detections:
                cx, cy = det["centroid"]
                det["centroid"] = (cx / scale, cy / scale)
                x, y, w, h = det["bbox"]
                det["bbox"] = (int(x / scale), int(y / scale), int(w / scale), int(h / scale))

        total_detections += len(detections)

        # Update tracker
        active_tracks = tracker.update(detections, frame_idx)
        confirmed = [t for t in active_tracks if t.get("is_confirmed")]
        max_simultaneous_eggs = max(max_simultaneous_eggs, len(confirmed))

        # Update trajectory history
        for track in active_tracks:
            tid = track["track_id"]
            if tid not in trajectory_history:
                trajectory_history[tid] = []
            trajectory_history[tid].append(track["centroid"])

        # Draw annotations on original-size frame
        annotated = draw_tracks(
            frame,
            active_tracks,
            trajectory_history,
            frame_idx,
            trajectory_length=tracking_config.get("visualization", {}).get("trajectory_length", 30),
        )

        # Write annotated frame
        writer.write(annotated)

        # Log tracks to CSV
        for track in active_tracks:
            cx, cy = track["centroid"]
            x, y, w, h = track["bbox"]
            csv_writer.writerow({
                "frame_idx": frame_idx,
                "track_id": track["track_id"],
                "cx": round(cx, 1),
                "cy": round(cy, 1),
                "x": x, "y": y, "w": w, "h": h,
                "confirmed": track.get("is_confirmed", False),
                "age": track.get("age", 0),
            })

        if processed_frames % 50 == 0:
            elapsed = time.time() - start_time
            fps_actual = processed_frames / elapsed
            logger.info(
                f"Frame {frame_idx}/{total_frames} | "
                f"Detections: {len(detections)} | "
                f"Tracks: {len(confirmed)} | "
                f"FPS: {fps_actual:.1f}"
            )

    cap.release()
    writer.release()
    csv_file.close()

    elapsed = time.time() - start_time
    total_unique_tracks = tracker._next_id - 1

    stats = {
        "video": str(video_path),
        "output": str(output_path),
        "total_frames": frame_idx,
        "processed_frames": processed_frames,
        "total_detections": total_detections,
        "avg_detections_per_frame": round(total_detections / max(processed_frames, 1), 2),
        "total_unique_tracks": total_unique_tracks,
        "max_simultaneous_eggs": max_simultaneous_eggs,
        "processing_time_s": round(elapsed, 1),
        "fps_achieved": round(processed_frames / elapsed, 2),
    }

    # Save stats JSON
    stats_path = output_dir / "tracking_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    # Load tracking config
    with open(args.config, "r") as f:
        tracking_config = yaml.safe_load(f)

    # Determine video path
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            sys.exit(1)
    else:
        video_dir = DEFAULT_VIDEO_DIR
        if not args.no_download:
            video_path = download_video(args.url, video_dir)
        else:
            existing = list(video_dir.glob("nosh_eggs.*"))
            video_path = existing[0] if existing else None

        if video_path is None:
            logger.error("No video available. Download failed or no cached video found.")
            sys.exit(1)

    logger.info("Task 4: Egg Tracking")
    logger.info(f"Video: {video_path}")
    logger.info(f"Output: {output_dir}")

    stats = process_video(
        video_path=video_path,
        output_dir=output_dir,
        tracking_config=tracking_config,
        max_frames=args.max_frames,
        skip_frames=args.skip_frames,
    )

    print("\n" + "=" * 60)
    print("TASK 4 -- EGG TRACKING SUMMARY")
    print("=" * 60)
    print(f"Total frames:          {stats['total_frames']}")
    print(f"Processed frames:      {stats['processed_frames']}")
    print(f"Total detections:      {stats['total_detections']}")
    print(f"Avg detections/frame:  {stats['avg_detections_per_frame']}")
    print(f"Total unique tracks:   {stats['total_unique_tracks']}")
    print(f"Max simultaneous eggs: {stats['max_simultaneous_eggs']}")
    print(f"Processing speed:      {stats['fps_achieved']} fps")
    print(f"Total time:            {stats['processing_time_s']}s")
    print(f"Output video:          {stats['output']}")
    print("=" * 60)


if __name__ == "__main__":
    # Fix type hint forward reference
    Optional_Path = type(None)
    main()
