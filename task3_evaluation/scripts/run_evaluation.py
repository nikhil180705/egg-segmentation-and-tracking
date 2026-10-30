"""
task3_evaluation/scripts/run_evaluation.py
==========================================
Run Task 3 evaluation: classical vs SAM segmentation comparison.

Usage:
    python task3_evaluation/scripts/run_evaluation.py --split dev
    python task3_evaluation/scripts/run_evaluation.py --split eval
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.logging_utils import get_logger
from task3_evaluation.src.evaluator import run_evaluation
from task3_evaluation.src.report_generator import save_comparison_grid, save_metrics_plots

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task 3 evaluation")
    parser.add_argument("--split", choices=["dev", "eval"], default="dev")
    parser.add_argument("--manifest", default="data/split_manifest.json")
    parser.add_argument("--sam-masks", default="task1_pretrained_segmentation/outputs/masks")
    parser.add_argument("--classical-masks", default="task2_classical_segmentation/outputs/masks")
    parser.add_argument("--output-dir", default="task3_evaluation/outputs")
    parser.add_argument("--iou-threshold", type=float, default=0.25)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    on_eval = args.split == "eval"

    logger.info(f"Task 3: Evaluation on {args.split} set")

    results = run_evaluation(
        split_manifest_path=args.manifest,
        sam_masks_dir=args.sam_masks,
        classical_masks_dir=args.classical_masks,
        output_dir=args.output_dir,
        iou_threshold=args.iou_threshold,
        on_eval_set=on_eval,
    )

    if not results or not results.get("per_image"):
        logger.error("Evaluation failed — no results produced")
        sys.exit(1)

    per_image = results["per_image"]

    if not args.skip_plots:
        output_dir = Path(args.output_dir)
        image_dir = Path("data/dev") if not on_eval else Path("data/eval")
        sam_dir = Path(args.sam_masks)
        cls_dir = Path(args.classical_masks)

        # Comparison grids: best and worst cases
        save_comparison_grid(
            per_image, image_dir, sam_dir, cls_dir, output_dir,
            max_samples=8, sort_by="iou_desc"
        )
        save_comparison_grid(
            per_image, image_dir, sam_dir, cls_dir, output_dir,
            max_samples=8, sort_by="iou"
        )
        # Metric distribution plots
        save_metrics_plots(per_image, output_dir)

    logger.info(f"Task 3 evaluation complete. Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
