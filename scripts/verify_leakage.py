"""
scripts/verify_leakage.py
=========================
Rigorously verify data leakage between dev (88 images) and eval (22 images):
1. Exact hash matching via SHA-256
2. Perceptual difference hashing (dHash) and average hashing (aHash) for near-duplicates
3. Roboflow base filename prefix analysis (tracking original unaugmented source IDs)
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def compute_sha256(filepath: Path) -> str:
    """Compute exact SHA-256 hash of file bytes."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_dhash(image_path: Path, hash_size: int = 8) -> int:
    """
    Compute 64-bit difference hash (dHash) of an image.
    Robust to slight brightness changes, compression, and scaling.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read {image_path}")
    resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    # Pack boolean array into integer
    hash_int = 0
    for bit in diff.flatten():
        hash_int = (hash_int << 1) | int(bit)
    return hash_int


def hamming_distance(h1: int, h2: int) -> int:
    """Compute Hamming distance between two 64-bit integer hashes."""
    return bin(h1 ^ h2).count("1")


def extract_base_id(filename: str) -> str:
    """
    Extract base identifier from Roboflow export filename.
    Example: 'not_damaged_100_jpg.rf.5d780a6024004fa2f68741fa56ddf4c9.jpg' -> 'not_damaged_100'
    """
    parts = filename.split(".rf.")
    return parts[0].replace("_jpg", "").replace("_png", "")


def main():
    dev_dir = Path("data/dev")
    eval_dir = Path("data/eval")

    dev_files = sorted(list(dev_dir.glob("*.jpg")) + list(dev_dir.glob("*.png")))
    eval_files = sorted(list(eval_dir.glob("*.jpg")) + list(eval_dir.glob("*.png")))

    print(f"Loaded {len(dev_files)} dev images and {len(eval_files)} eval images.")

    # 1. Exact SHA-256 Hash Matching
    dev_sha256 = {f.name: compute_sha256(f) for f in dev_files}
    eval_sha256 = {f.name: compute_sha256(f) for f in eval_files}

    dev_hash_set = set(dev_sha256.values())
    exact_duplicates = []
    for fname, h in eval_sha256.items():
        if h in dev_hash_set:
            matching_dev = [k for k, v in dev_sha256.items() if v == h]
            exact_duplicates.append({"eval_file": fname, "matching_dev": matching_dev, "sha256": h})

    print(f"Exact SHA-256 Duplicates between dev and eval: {len(exact_duplicates)}")

    # 2. Perceptual Difference Hash (dHash) & Hamming Distance Matrix
    dev_dhash = {f.name: compute_dhash(f) for f in dev_files}
    eval_dhash = {f.name: compute_dhash(f) for f in eval_files}

    near_duplicates = []
    min_distances = []
    for eval_name, eh in eval_dhash.items():
        best_dist = 64
        best_match = None
        for dev_name, dh in dev_dhash.items():
            dist = hamming_distance(eh, dh)
            if dist < best_dist:
                best_dist = dist
                best_match = dev_name
        min_distances.append(best_dist)
        # Threshold: distance <= 5 bits out of 64 indicates strong perceptual similarity / near-duplicate
        if best_dist <= 5:
            near_duplicates.append({
                "eval_file": eval_name,
                "dev_file": best_match,
                "hamming_distance": best_dist,
                "similarity_pct": round((1.0 - best_dist / 64.0) * 100, 2),
            })

    print(f"Perceptual Near-Duplicates (Hamming distance <= 5 / 64): {len(near_duplicates)}")
    print(f"Minimum pairwise Hamming distance observed: {min(min_distances)} bits")
    print(f"Average nearest-neighbor Hamming distance: {np.mean(min_distances):.2f} bits")

    # 3. Base Image ID Prefix Analysis
    dev_base_ids = {extract_base_id(f.name): f.name for f in dev_files}
    eval_base_ids = {extract_base_id(f.name): f.name for f in eval_files}

    shared_base_ids = set(dev_base_ids.keys()).intersection(set(eval_base_ids.keys()))
    print(f"Unique base IDs in dev: {len(dev_base_ids)}")
    print(f"Unique base IDs in eval: {len(eval_base_ids)}")
    print(f"Shared base IDs (potential augmentation leakage): {len(shared_base_ids)}")
    if shared_base_ids:
        print(f"Shared IDs: {shared_base_ids}")

    # Generate Report Dict
    report = {
        "dev_image_count": len(dev_files),
        "eval_image_count": len(eval_files),
        "exact_sha256_duplicates_count": len(exact_duplicates),
        "exact_duplicates": exact_duplicates,
        "perceptual_near_duplicates_count": len(near_duplicates),
        "near_duplicates": near_duplicates,
        "min_hamming_distance_bits": int(min(min_distances)),
        "mean_nearest_hamming_distance_bits": round(float(np.mean(min_distances)), 2),
        "shared_base_ids_count": len(shared_base_ids),
        "shared_base_ids": list(shared_base_ids),
        "verdict": "NO_LEAKAGE" if len(exact_duplicates) == 0 and len(near_duplicates) == 0 and len(shared_base_ids) == 0 else "INVESTIGATION_NEEDED",
    }

    report_path = Path("data/leakage_analysis_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Leakage analysis report saved to: {report_path}")
    print(f"Final Verdict: {report['verdict']}")


if __name__ == "__main__":
    main()
