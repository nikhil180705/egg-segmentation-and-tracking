"""
task1_pretrained_segmentation/scripts/run_segmentation.py
==========================================================
Run SAM egg segmentation on the development or evaluation dataset.

Usage:
    # Run on development set with default config:
    python task1_pretrained_segmentation/scripts/run_segmentation.py

    # Run on evaluation set:
    python task1_pretrained_segmentation/scripts/run_segmentation.py --split eval

    # Override config and limit images:
    python task1_pretrained_segmentation/scripts/run_segmentation.py \\
        --config configs/sam_config.yaml --max-images 10 --split dev

    # Use bbox prompting strategy:
    python task1_pretrained_segmentation/scripts/run_segmentation.py \\
        --strategy bbox
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import yaml

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.logging_utils import get_logger
from task1_pretrained_segmentation.src.egg_segmenter import SAMEggSegmenter
from task1_pretrained_segmentation.src.visualization import (
    save_failure_analysis,
    save_overlay,
    save_summary_grid,
)

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM-based egg segmentation."
    )
    parser.add_argument(
        "--config", default="configs/sam_config.yaml",
        help="SAM configuration YAML path"
    )
    parser.add_argument(
        "--split", choices=["dev", "eval", "all"], default="dev",
        help="Which data split to process (default: dev)"
    )
    parser.add_argument(
        "--input-dir", default=None,
        help="Override input directory (overrides --split)"
    )
    parser.add_argument(
        "--output-dir",
        default="task1_pretrained_segmentation/outputs",
        help="Output directory for masks and overlays"
    )
    parser.add_argument(
        "--strategy", choices=["auto", "bbox"], default=None,
        help="Override prompting strategy from config"
    )
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Process only first N images (for testing)"
    )
    parser.add_argument(
        "--skip-overlays", action="store_true",
        help="Skip generating overlay images"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_total = time.time()

    # Determine input directory
    if args.input_dir:
        input_dir = Path(args.input_dir)
    elif args.split == "dev":
        input_dir = Path("data/dev")
    elif args.split == "eval":
        input_dir = Path("data/eval")
    else:
        input_dir = Path("data/raw")

    output_dir = Path(args.output_dir)

    logger.info(f"Task 1: SAM Egg Segmentation")
    logger.info(f"Input: {input_dir} | Output: {output_dir}")
    logger.info(f"Config: {args.config}")

    # Load config and optionally override strategy
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if args.strategy:
        config["inference"]["prompting_strategy"] = args.strategy
        # Write back to config (temporary override)
        with open(args.config, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Strategy overridden to: {args.strategy}")

    # Initialize and load model
    segmenter = SAMEggSegmenter(config_path=args.config)
    segmenter.load_model()

    # Run segmentation
    results = segmenter.segment_directory(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        max_images=args.max_images,
    )

    if not results:
        logger.error("No results generated. Check input directory.")
        sys.exit(1)

    # Generate overlays
    if not args.skip_overlays:
        logger.info("Generating overlay visualizations...")
        from shared.image_utils import collect_images
        for result in results:
            if result["masks"] and not result.get("error"):
                try:
                    save_overlay(
                        image_path=Path(result["image_path"]),
                        masks=result["masks"],
                        output_dir=output_dir,
                        alpha=config["output"]["overlay_alpha"],
                    )
                except Exception as e:
                    logger.warning(f"Overlay failed for {result['image_name']}: {e}")

    # Summary grid
    save_summary_grid(results, output_dir)

    # Failure analysis
    save_failure_analysis(results, output_dir)

    # Save per-image CSV report
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / f"segmentation_results_{args.split}.csv"
    _save_csv_report(results, csv_path)

    # Save JSON summary
    json_path = reports_dir / f"segmentation_summary_{args.split}.json"
    _save_json_summary(results, json_path, config, args)

    # Print final summary
    elapsed = time.time() - start_total
    successful = [r for r in results if not r.get("error")]
    total_eggs = sum(r["num_eggs"] for r in successful)
    zero_egg = [r for r in successful if r["num_eggs"] == 0]
    errors = [r for r in results if r.get("error")]

    print("\n" + "=" * 60)
    print("TASK 1 — SAM SEGMENTATION SUMMARY")
    print("=" * 60)
    print(f"Images processed:    {len(results)}")
    print(f"Successful:          {len(successful)}")
    print(f"Errors:              {len(errors)}")
    print(f"Zero eggs detected:  {len(zero_egg)}")
    print(f"Total eggs found:    {total_eggs}")
    if successful:
        avg_eggs = total_eggs / len(successful)
        avg_time = sum(r["processing_time_s"] for r in results) / len(results)
        print(f"Avg eggs/image:      {avg_eggs:.2f}")
        print(f"Avg time/image:      {avg_time:.2f}s")
    print(f"Total elapsed:       {elapsed:.1f}s")
    print(f"Results saved:       {output_dir}")
    print("=" * 60)


def _save_csv_report(results: list, csv_path: Path) -> None:
    """Save per-image results as CSV."""
    fieldnames = [
        "image_name", "num_eggs", "processing_time_s",
        "strategy_used", "image_shape", "error"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "image_name": r["image_name"],
                "num_eggs": r["num_eggs"],
                "processing_time_s": r["processing_time_s"],
                "strategy_used": r["strategy_used"],
                "image_shape": str(r["image_shape"]),
                "error": r.get("error", ""),
            })
    logger.info(f"CSV report saved: {csv_path}")


def _save_json_summary(results: list, json_path: Path, config: dict, args) -> None:
    """Save summary statistics as JSON."""
    successful = [r for r in results if not r.get("error")]
    summary = {
        "model": config["model"]["name"],
        "variant": config["model"]["variant"],
        "strategy": config["inference"]["prompting_strategy"],
        "split": args.split,
        "total_images": len(results),
        "successful": len(successful),
        "errors": len(results) - len(successful),
        "zero_egg_images": len([r for r in successful if r["num_eggs"] == 0]),
        "total_eggs_detected": sum(r["num_eggs"] for r in successful),
        "avg_processing_time_s": round(
            sum(r["processing_time_s"] for r in results) / len(results), 3
        ) if results else 0,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"JSON summary saved: {json_path}")


if __name__ == "__main__":
    main()
