"""
scripts/download_dataset.py
===========================
Downloads the egg dataset from Google Drive or Roboflow, then
runs initial inspection and creates the 80/20 dev/eval split.

Usage:
    python scripts/download_dataset.py --source gdrive
    python scripts/download_dataset.py --source roboflow --api-key YOUR_KEY
"""

import argparse
import sys
import zipfile
from pathlib import Path

# Ensure repository root is on path for shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yaml

from shared.dataset_loader import create_split, inspect_dataset
from shared.logging_utils import get_logger

logger = get_logger(__name__)

GDRIVE_FILE_ID = "1a58rACcCFnaj0ZiDoNyGbDWXunRyWziv"
RAW_DIR = Path("data/raw")
DEV_DIR = Path("data/dev")
EVAL_DIR = Path("data/eval")
ZIP_PATH = Path("data/egg_dataset.zip")


def download_gdrive(file_id: str, dest: Path) -> None:
    """Download a file from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        logger.error("gdown not installed. Run: pip install gdown")
        sys.exit(1)

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    logger.info(f"Downloading from Google Drive: {url}")
    gdown.download(url, str(dest), quiet=False)
    logger.info(f"Download complete: {dest}")


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract a ZIP archive, handling nested directory structures."""
    logger.info(f"Extracting {zip_path} -> {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        logger.info(f"  Archive contains {len(members)} entries")
        zf.extractall(str(dest_dir))
    logger.info("Extraction complete.")


def download_roboflow(api_key: str, workspace: str, project: str, version: int) -> None:
    """Download dataset via Roboflow API."""
    try:
        from roboflow import Roboflow
    except ImportError:
        logger.error("roboflow package not installed. Run: pip install roboflow")
        sys.exit(1)

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download("folder", location=str(RAW_DIR))
    logger.info(f"Roboflow dataset downloaded to: {dataset.location}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare the Nosh Robotics egg dataset."
    )
    parser.add_argument(
        "--source",
        choices=["gdrive", "roboflow"],
        default="gdrive",
        help="Data source (default: gdrive)",
    )
    parser.add_argument("--api-key", default="", help="Roboflow API key (required for roboflow source)")
    parser.add_argument("--skip-download", action="store_true", help="Skip download; use existing raw data")
    parser.add_argument("--overwrite-split", action="store_true", help="Re-create split even if manifest exists")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split (default: 42)")
    args = parser.parse_args()

    # ── Step 1: Download ──────────────────────────────────────────────────
    if not args.skip_download:
        if args.source == "gdrive":
            if not ZIP_PATH.exists():
                download_gdrive(GDRIVE_FILE_ID, ZIP_PATH)
            else:
                logger.info(f"ZIP already exists: {ZIP_PATH}. Skipping download.")
            extract_zip(ZIP_PATH, RAW_DIR)
        elif args.source == "roboflow":
            if not args.api_key:
                logger.error("--api-key required for Roboflow source.")
                sys.exit(1)
            download_roboflow(args.api_key, "fire-bf7qe", "egg-vzhtr", 1)
    else:
        logger.info("Skipping download (--skip-download specified).")

    # Verify images exist
    from shared.image_utils import collect_images
    images = collect_images(RAW_DIR, recursive=True)
    if not images:
        logger.error(f"No images found in {RAW_DIR} after extraction. Check archive contents.")
        sys.exit(1)
    logger.info(f"Found {len(images)} images in {RAW_DIR}")

    # ── Step 2: Dataset Inspection ────────────────────────────────────────
    logger.info("Running dataset inspection...")
    df = inspect_dataset(str(RAW_DIR))

    inspection_path = Path("data/dataset_inspection.csv")
    df.to_csv(inspection_path, index=False)
    logger.info(f"Inspection results saved: {inspection_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("DATASET INSPECTION SUMMARY")
    print("=" * 60)
    print(f"Total images: {len(df)}")
    if not df.empty:
        print(f"Width  — min: {df['width'].min()}, max: {df['width'].max()}, mean: {df['width'].mean():.0f}")
        print(f"Height — min: {df['height'].min()}, max: {df['height'].max()}, mean: {df['height'].mean():.0f}")
        print(f"Aspect — min: {df['aspect_ratio'].min():.2f}, max: {df['aspect_ratio'].max():.2f}")
        print(f"Size   — min: {df['file_size_bytes'].min()/1024:.1f}KB, max: {df['file_size_bytes'].max()/1024:.1f}KB")

    # ── Step 3: Create Dev/Eval Split ─────────────────────────────────────
    logger.info("Creating 80/20 development/evaluation split...")
    split = create_split(
        raw_dir=str(RAW_DIR),
        dev_dir=str(DEV_DIR),
        eval_dir=str(EVAL_DIR),
        manifest_path="data/split_manifest.json",
        dev_ratio=0.80,
        seed=args.seed,
        overwrite=args.overwrite_split,
    )

    print("\n" + "=" * 60)
    print("SPLIT SUMMARY")
    print("=" * 60)
    print(f"Development set: {len(split['dev'])} images -> {DEV_DIR}")
    print(f"Evaluation set:  {len(split['eval'])} images -> {EVAL_DIR}")
    print("Split manifest:  data/split_manifest.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
