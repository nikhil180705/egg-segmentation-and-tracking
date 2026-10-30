"""
shared/dataset_loader.py
========================
Dataset loading, inspection, and development/evaluation splitting.

Splitting strategy:
- If images appear to come from distinct video sessions (near-duplicate detection),
  group-based splitting is used to prevent data leakage.
- Otherwise, random splitting with fixed seed 42 is applied.
- The split is recorded in data/split_manifest.json for full reproducibility.
"""

import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from shared.image_utils import collect_images, get_image_info
from shared.logging_utils import get_logger

logger = get_logger(__name__)

RANDOM_SEED = 42
DEV_RATIO = 0.80
EVAL_RATIO = 0.20


# ── Configuration Loading ────────────────────────────────────────────────────

def load_dataset_config(config_path: str = "configs/dataset.yaml") -> dict:
    """Load the dataset YAML configuration."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Dataset Inspection ───────────────────────────────────────────────────────

def inspect_dataset(raw_dir: str = "data/raw") -> pd.DataFrame:
    """
    Inspect all images in the raw directory and return a summary DataFrame.

    Records filename, dimensions, file size, aspect ratio for each image.
    Also performs near-duplicate detection based on file size and perceptual hash.

    Parameters
    ----------
    raw_dir : str or Path
        Directory containing raw images (may include subdirectories).

    Returns
    -------
    pd.DataFrame
        One row per image with inspection statistics.
    """
    raw_dir = Path(raw_dir)
    image_paths = collect_images(raw_dir, recursive=True)

    logger.info(f"Found {len(image_paths)} images in {raw_dir}")

    records = []
    for path in image_paths:
        try:
            info = get_image_info(path)
            info["phash"] = _compute_phash(path)
            info["relative_path"] = str(path.relative_to(raw_dir))
            records.append(info)
        except Exception as e:
            logger.warning(f"Skipping {path.name}: {e}")

    df = pd.DataFrame(records)
    if not df.empty:
        logger.info(f"Image stats:\n{df[['width', 'height', 'file_size_bytes']].describe().to_string()}")
        _report_near_duplicates(df)
    return df


def _compute_phash(path: Path, hash_size: int = 8) -> str:
    """
    Compute a perceptual hash (pHash) for near-duplicate detection.

    Reduces image to hash_size×hash_size DCT representation.
    Images with identical or near-identical content will have the same hash.

    Parameters
    ----------
    path : Path
    hash_size : int
        Hash output dimension.

    Returns
    -------
    str
        Hex string representing the hash.
    """
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return ""
    # Resize to (hash_size*4) × (hash_size*4) for DCT computation
    dct_size = hash_size * 4
    resized = cv2.resize(img, (dct_size, dct_size), interpolation=cv2.INTER_AREA)
    # Apply DCT
    dct = cv2.dct(resized.astype(np.float32))
    dct_low = dct[:hash_size, :hash_size]
    median_val = np.median(dct_low)
    bits = (dct_low > median_val).flatten()
    # Pack bits into integer
    hash_int = int("".join(str(int(b)) for b in bits), 2)
    return format(hash_int, "016x")


def _report_near_duplicates(df: pd.DataFrame) -> None:
    """Log any detected near-duplicate groups."""
    if "phash" not in df.columns:
        return
    dupes = df.groupby("phash").filter(lambda g: len(g) > 1)
    if not dupes.empty:
        n_groups = dupes["phash"].nunique()
        n_images = len(dupes)
        logger.warning(
            f"Detected {n_groups} near-duplicate groups involving {n_images} images. "
            "Group-based splitting will be used to prevent data leakage."
        )
    else:
        logger.info("No near-duplicate images detected. Random splitting will be used.")


# ── Development / Evaluation Split ──────────────────────────────────────────

def create_split(
    raw_dir: str = "data/raw",
    dev_dir: str = "data/dev",
    eval_dir: str = "data/eval",
    manifest_path: str = "data/split_manifest.json",
    dev_ratio: float = DEV_RATIO,
    seed: int = RANDOM_SEED,
    overwrite: bool = False,
) -> Dict[str, List[str]]:
    """
    Split raw images into development (80%) and evaluation (20%) sets.

    Uses group-based splitting if near-duplicates are found; random otherwise.
    Copies images to dev_dir and eval_dir.
    Records the split in a JSON manifest for reproducibility.

    Parameters
    ----------
    raw_dir : str or Path
    dev_dir : str or Path
    eval_dir : str or Path
    manifest_path : str or Path
    dev_ratio : float
    seed : int
    overwrite : bool
        If False, raises error if split already exists.

    Returns
    -------
    dict
        {"dev": [filenames...], "eval": [filenames...]}
    """
    raw_dir = Path(raw_dir)
    dev_dir = Path(dev_dir)
    eval_dir = Path(eval_dir)
    manifest_path = Path(manifest_path)

    if manifest_path.exists() and not overwrite:
        logger.info(f"Split manifest already exists at {manifest_path}. Loading existing split.")
        return _load_manifest(manifest_path)

    # Collect all images
    all_images = collect_images(raw_dir, recursive=True)
    if not all_images:
        raise RuntimeError(f"No images found in {raw_dir}")

    logger.info(f"Splitting {len(all_images)} images (dev={dev_ratio:.0%}, eval={1-dev_ratio:.0%}, seed={seed})")

    # Detect near-duplicates for group-based splitting
    df = inspect_dataset(str(raw_dir))
    strategy, groups = _determine_split_strategy(df, dev_ratio, seed)
    logger.info(f"Split strategy: {strategy}")

    dev_files = groups["dev"]
    eval_files = groups["eval"]

    # Copy files to split directories
    _copy_files(raw_dir, dev_dir, dev_files)
    _copy_files(raw_dir, eval_dir, eval_files)

    # Save manifest
    manifest = {
        "seed": seed,
        "dev_ratio": dev_ratio,
        "strategy": strategy,
        "total_images": len(all_images),
        "dev_count": len(dev_files),
        "eval_count": len(eval_files),
        "dev": dev_files,
        "eval": eval_files,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Split manifest saved: {manifest_path}")
    logger.info(f"Dev: {len(dev_files)} images | Eval: {len(eval_files)} images")

    # Verify no leakage
    _verify_no_leakage(dev_files, eval_files)

    return {"dev": dev_files, "eval": eval_files}


def _determine_split_strategy(
    df: pd.DataFrame, dev_ratio: float, seed: int
) -> Tuple[str, Dict[str, List[str]]]:
    """
    Determine and execute the split strategy.

    Returns strategy name and {'dev': [...], 'eval': [...]}.
    """
    rng = random.Random(seed)

    # Check for near-duplicate groups
    has_duplicates = (
        "phash" in df.columns
        and df["phash"].duplicated().any()
        and df["phash"].ne("").all()
    )

    filenames = df["relative_path"].tolist() if "relative_path" in df.columns else df["filename"].tolist()

    if has_duplicates:
        # Group-based split: keep duplicate groups together in one split
        strategy = "group_based"
        groups = df.groupby("phash")["relative_path"].apply(list).to_dict()
        group_keys = list(groups.keys())
        rng.shuffle(group_keys)

        split_idx = max(1, int(len(group_keys) * dev_ratio))
        dev_keys = group_keys[:split_idx]
        eval_keys = group_keys[split_idx:]

        dev_files = [f for k in dev_keys for f in groups[k]]
        eval_files = [f for k in eval_keys for f in groups[k]]
    else:
        # Simple random split
        strategy = "random"
        shuffled = filenames.copy()
        rng.shuffle(shuffled)
        split_idx = max(1, int(len(shuffled) * dev_ratio))
        dev_files = shuffled[:split_idx]
        eval_files = shuffled[split_idx:]

    return strategy, {"dev": dev_files, "eval": eval_files}


def _copy_files(src_dir: Path, dst_dir: Path, relative_paths: List[str]) -> None:
    """Copy files from src_dir to dst_dir preserving relative paths."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for rel_path in relative_paths:
        src = src_dir / rel_path
        dst = dst_dir / Path(rel_path).name  # Flatten into single directory
        if src.exists():
            shutil.copy2(str(src), str(dst))
        else:
            logger.warning(f"Source file not found during copy: {src}")


def _verify_no_leakage(dev_files: List[str], eval_files: List[str]) -> None:
    """Assert no filename appears in both splits."""
    dev_names = {Path(f).name for f in dev_files}
    eval_names = {Path(f).name for f in eval_files}
    overlap = dev_names & eval_names
    if overlap:
        raise RuntimeError(
            f"DATA LEAKAGE DETECTED: {len(overlap)} files appear in both dev and eval splits: "
            f"{list(overlap)[:5]}"
        )
    logger.info("Leakage check passed: no files shared between dev and eval splits.")


def _load_manifest(manifest_path: Path) -> Dict[str, List[str]]:
    """Load an existing split manifest."""
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    logger.info(
        f"Loaded existing split: dev={manifest['dev_count']}, eval={manifest['eval_count']}, "
        f"strategy={manifest['strategy']}, seed={manifest['seed']}"
    )
    return {"dev": manifest["dev"], "eval": manifest["eval"]}


# ── Convenience Loaders ──────────────────────────────────────────────────────

def get_dev_images(dev_dir: str = "data/dev") -> List[Path]:
    """Return sorted list of development set image paths."""
    return collect_images(dev_dir)


def get_eval_images(eval_dir: str = "data/eval") -> List[Path]:
    """Return sorted list of evaluation set image paths."""
    return collect_images(eval_dir)
