"""
task2_classical_segmentation/scripts/run_segmentation.py
=========================================================
Run classical egg segmentation on the development or evaluation dataset.

Usage:
    python task2_classical_segmentation/scripts/run_segmentation.py
    python task2_classical_segmentation/scripts/run_segmentation.py --split eval
    python task2_classical_segmentation/scripts/run_segmentation.py --max-images 10
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.image_utils import collect_images
from shared.logging_utils import get_logger
from task2_classical_segmentation.src.egg_segmenter import ClassicalEggSegmenter
from task2_classical_segmentation.src.visualization import save_overlay, save_summary_grid

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classical egg segmentation.")
    parser.add_argument("--config", default="configs/segmentation_config.yaml")
    parser.add_argument("--split", choices=["dev", "eval", "all"], default="dev")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default="task2_classical_segmentation/outputs")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--skip-overlays", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_total = time.time()

    if args.input_dir:
        input_dir = Path(args.input_dir)
    elif args.split == "dev":
        input_dir = Path("data/dev")
    elif args.split == "eval":
        input_dir = Path("data/eval")
    else:
        input_dir = Path("data/raw")

    output_dir = Path(args.output_dir)

    logger.info(f"Task 2: Classical Egg Segmentation")
    logger.info(f"Input: {input_dir} | Output: {output_dir}")

    segmenter = ClassicalEggSegmenter(config_path=args.config)
    results = segmenter.segment_directory(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        max_images=args.max_images,
    )

    if not results:
        logger.error("No results. Check input directory.")
        sys.exit(1)

    # Generate overlays
    if not args.skip_overlays:
        logger.info("Generating overlays...")
        for result in results:
            if not result.get("error"):
                try:
                    save_overlay(
                        image_path=Path(result["image_path"]),
                        masks=result["masks"],
                        output_dir=output_dir,
                    )
                except Exception as e:
                    logger.warning(f"Overlay failed for {result['image_name']}: {e}")

    # Summary grid
    save_summary_grid(results, output_dir)

    # Save CSV report
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / f"classical_results_{args.split}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "num_eggs", "processing_time_s", "error"])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "image_name": r["image_name"],
                "num_eggs": r["num_eggs"],
                "processing_time_s": r["processing_time_s"],
                "error": r.get("error", ""),
            })

    # JSON summary
    successful = [r for r in results if not r.get("error")]
    total_eggs = sum(r["num_eggs"] for r in successful)
    summary = {
        "split": args.split,
        "total_images": len(results),
        "successful": len(successful),
        "zero_egg_images": len([r for r in successful if r["num_eggs"] == 0]),
        "total_eggs_detected": total_eggs,
        "avg_eggs_per_image": round(total_eggs / len(successful), 2) if successful else 0,
        "avg_processing_time_s": round(sum(r["processing_time_s"] for r in results) / len(results), 4),
    }
    json_path = reports_dir / f"classical_summary_{args.split}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - start_total
    print("\n" + "=" * 60)
    print("TASK 2 -- CLASSICAL SEGMENTATION SUMMARY")
    print("=" * 60)
    print(f"Images processed: {summary['total_images']}")
    print(f"Successful:       {summary['successful']}")
    print(f"Zero-egg images:  {summary['zero_egg_images']}")
    print(f"Total eggs found: {summary['total_eggs_detected']}")
    print(f"Avg eggs/image:   {summary['avg_eggs_per_image']}")
    print(f"Avg time/image:   {summary['avg_processing_time_s']:.3f}s")
    print(f"Total elapsed:    {elapsed:.1f}s")
    print(f"Results saved:    {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
