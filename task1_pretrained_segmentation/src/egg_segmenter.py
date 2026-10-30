"""
task1_pretrained_segmentation/src/egg_segmenter.py
===================================================
Main orchestrator for SAM-based egg segmentation.

Processing pipeline per image:
  1. Load image (RGB for SAM)
  2. Select prompting strategy from config
  3. Generate masks (AMG or bbox-prompted)
  4. Filter masks for egg-specific criteria
  5. Apply NMS to remove duplicates
  6. Save binary masks + return structured results
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from shared.image_utils import collect_images, load_image_bgr, load_image_rgb, save_image
from shared.logging_utils import get_logger
from shared.mask_utils import ensure_binary_uint8, save_binary_mask
from task1_pretrained_segmentation.src.mask_generator import (
    detect_egg_bboxes_classical,
    generate_masks_auto,
    generate_masks_bbox,
    select_best_masks,
)
from task1_pretrained_segmentation.src.model_loader import get_device, load_sam_model

logger = get_logger(__name__)


class SAMEggSegmenter:
    """
    SAM-based egg segmenter.

    Supports two prompting strategies:
    - 'auto': Automatic Mask Generation (AMG) with egg filtering
    - 'bbox': Classical CV bounding box detection → SAM refinement

    Parameters
    ----------
    config_path : str
        Path to sam_config.yaml.
    """

    def __init__(self, config_path: str = "configs/sam_config.yaml") -> None:
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.device = get_device(self.config["inference"]["device"])
        self.sam = None
        self.predictor = None
        self.mask_generator = None
        self._strategy = self.config["inference"]["prompting_strategy"]

    def _load_config(self) -> dict:
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def load_model(self) -> None:
        """Load SAM model and initialize predictor/generator."""
        from segment_anything import SamAutomaticMaskGenerator, SamPredictor

        model_cfg = self.config["model"]
        self.sam = load_sam_model(
            variant=model_cfg["variant"],
            checkpoints_dir=model_cfg["checkpoints_dir"],
            device=self.device,
        )

        # Initialize both predictor (for bbox) and generator (for auto)
        self.predictor = SamPredictor(self.sam)

        amg_cfg = self.config["inference"]["amg"]
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=amg_cfg["points_per_side"],
            pred_iou_thresh=amg_cfg["pred_iou_thresh"],
            stability_score_thresh=amg_cfg["stability_score_thresh"],
            crop_n_layers=amg_cfg["crop_n_layers"],
            crop_n_points_downscale_factor=amg_cfg["crop_n_points_downscale_factor"],
            min_mask_region_area=amg_cfg["min_mask_region_area"],
        )
        logger.info(f"SAM model loaded. Strategy: {self._strategy}")

    def segment_image(
        self,
        image_path: Path,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Segment eggs in a single image.

        Parameters
        ----------
        image_path : Path
            Path to input image.
        output_dir : Path, optional
            If provided, saves binary masks and overlays here.

        Returns
        -------
        dict with keys:
            - 'image_name': str
            - 'image_path': str
            - 'image_shape': (H, W, C)
            - 'num_eggs': int
            - 'masks': list of np.ndarray (binary uint8)
            - 'mask_metadata': list of dicts (area, iou, circularity, ...)
            - 'processing_time_s': float
            - 'strategy_used': str
            - 'error': str or None
        """
        if self.sam is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.perf_counter()
        result = {
            "image_name": image_path.name,
            "image_path": str(image_path),
            "image_shape": None,
            "num_eggs": 0,
            "masks": [],
            "mask_metadata": [],
            "processing_time_s": 0.0,
            "strategy_used": self._strategy,
            "error": None,
        }

        try:
            img_rgb = load_image_rgb(image_path)
            img_bgr = load_image_bgr(image_path)
            result["image_shape"] = img_rgb.shape

            if self._strategy == "auto":
                raw_masks = generate_masks_auto(
                    img_rgb,
                    self.mask_generator,
                    self.config["inference"]["amg"],
                    self.config["inference"]["egg_filter"],
                )
                filtered_masks = select_best_masks(
                    raw_masks,
                    iou_threshold=self.config["inference"]["amg"]["pred_iou_thresh"],
                    apply_nms=True,
                )

            elif self._strategy == "bbox":
                # Step 1: Classical CV detection for bboxes
                bboxes = detect_egg_bboxes_classical(
                    img_bgr,
                    min_area=2000,
                    max_area=int(img_rgb.shape[0] * img_rgb.shape[1] * 0.9),
                )

                if not bboxes:
                    logger.warning(f"No egg bboxes detected classically for {image_path.name}; falling back to AMG.")
                    raw_masks = generate_masks_auto(
                        img_rgb,
                        self.mask_generator,
                        self.config["inference"]["amg"],
                        self.config["inference"]["egg_filter"],
                    )
                    filtered_masks = select_best_masks(raw_masks, iou_threshold=0.80)
                    result["strategy_used"] = "auto_fallback"
                else:
                    # Step 2: Set SAM image once, then prompt with each bbox
                    self.predictor.set_image(img_rgb)
                    raw_masks = generate_masks_bbox(img_rgb, self.predictor, bboxes)
                    filtered_masks = select_best_masks(raw_masks, iou_threshold=0.70)
            else:
                raise ValueError(f"Unknown prompting strategy: {self._strategy}")

            # Convert to binary uint8 masks
            binary_masks = []
            metadata = []
            for i, mask_data in enumerate(filtered_masks):
                seg = mask_data["segmentation"]
                binary = ensure_binary_uint8(seg)
                binary_masks.append(binary)
                metadata.append({
                    "mask_id": i,
                    "area": int(np.sum(seg.astype(bool))),
                    "predicted_iou": round(mask_data.get("predicted_iou", 0.0), 4),
                    "circularity": round(mask_data.get("circularity", 0.0), 4),
                    "area_ratio": round(mask_data.get("area_ratio", 0.0), 4),
                })

            result["num_eggs"] = len(binary_masks)
            result["masks"] = binary_masks
            result["mask_metadata"] = metadata

            # Save outputs if directory specified
            if output_dir is not None and binary_masks:
                self._save_outputs(image_path, img_bgr, binary_masks, output_dir)

        except Exception as e:
            logger.error(f"Error segmenting {image_path.name}: {e}", exc_info=True)
            result["error"] = str(e)

        result["processing_time_s"] = round(time.perf_counter() - start_time, 3)
        logger.info(
            f"{image_path.name}: {result['num_eggs']} eggs detected "
            f"({result['processing_time_s']:.2f}s) [{result['strategy_used']}]"
        )
        return result

    def _save_outputs(
        self,
        image_path: Path,
        img_bgr: np.ndarray,
        binary_masks: List[np.ndarray],
        output_dir: Path,
    ) -> None:
        """Save individual binary masks for each egg."""
        stem = image_path.stem
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        for i, mask in enumerate(binary_masks):
            mask_path = masks_dir / f"{stem}_mask_{i:02d}.png"
            save_binary_mask(mask, mask_path)

        # Save combined mask (union of all egg masks)
        from shared.mask_utils import combine_masks
        if len(binary_masks) > 1:
            combined = combine_masks(binary_masks)
        else:
            combined = binary_masks[0]
        combined_path = masks_dir / f"{stem}_combined_mask.png"
        save_binary_mask(combined, combined_path)

    def segment_directory(
        self,
        input_dir: str,
        output_dir: str,
        max_images: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Segment all images in a directory.

        Parameters
        ----------
        input_dir : str
        output_dir : str
        max_images : int, optional
            Limit number of images (for testing).

        Returns
        -------
        list of per-image result dicts
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = collect_images(input_dir)
        if max_images is not None:
            image_paths = image_paths[:max_images]

        if not image_paths:
            logger.warning(f"No images found in {input_dir}")
            return []

        logger.info(f"Segmenting {len(image_paths)} images from {input_dir}")
        results = []
        for i, path in enumerate(image_paths, 1):
            logger.info(f"[{i}/{len(image_paths)}] Processing: {path.name}")
            result = self.segment_image(path, output_dir)
            results.append(result)

        return results
