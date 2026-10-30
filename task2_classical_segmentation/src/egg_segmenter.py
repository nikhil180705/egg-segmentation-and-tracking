"""
task2_classical_segmentation/src/egg_segmenter.py
===================================================
Main classical egg segmentation pipeline.

Final algorithm (selected after experiments — see experiment_log.csv):

Pipeline:
1. Gaussian blur (k=5) — noise reduction
2. HSV thresholding — isolate bright, low-saturation egg pixels
3. Grayscale + Otsu — capture eggs missed by HSV
4. Union of HSV + Otsu masks
5. Morphological opening (k=5, iter=2) — remove noise blobs
6. Morphological closing (k=15, iter=3) — fill egg interiors
7. Watershed — separate touching eggs
8. Contour detection + geometric filtering
9. Per-egg mask extraction

The pipeline is fully parameterized via segmentation_config.yaml.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from shared.image_utils import collect_images, load_image_bgr, save_image
from shared.logging_utils import get_logger
from shared.mask_utils import combine_masks, ensure_binary_uint8, save_binary_mask
from task2_classical_segmentation.src.color_thresholding import (
    combine_threshold_masks,
    threshold_hsv,
    threshold_otsu,
    threshold_adaptive,
)
from task2_classical_segmentation.src.contour_detection import (
    contours_to_combined_mask,
    contours_to_masks,
    filter_egg_contours,
    find_contours,
)
from task2_classical_segmentation.src.morphology import clean_mask
from task2_classical_segmentation.src.preprocessing import preprocess_image

logger = get_logger(__name__)


class ClassicalEggSegmenter:
    """
    Classical image processing egg segmenter.

    Uses no pretrained neural networks. Combines:
    - HSV color thresholding
    - Grayscale + Otsu thresholding
    - Morphological operations
    - Contour-based filtering
    - Watershed for touching eggs

    Parameters
    ----------
    config_path : str
        Path to segmentation_config.yaml.
    """

    def __init__(
        self,
        config_path: str = "configs/segmentation_config.yaml",
        config_override: Optional[Dict] = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.config = self._load_config()
        if config_override:
            self._deep_update(self.config, config_override)

    def _load_config(self) -> dict:
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def _deep_update(self, base: dict, override: dict) -> None:
        for k, v in override.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                self._deep_update(base[k], v)
            else:
                base[k] = v

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
        output_dir : Path, optional

        Returns
        -------
        dict with keys:
            image_name, image_path, num_eggs, masks, contours,
            processing_time_s, combined_mask, error
        """
        start_time = time.perf_counter()
        result = {
            "image_name": image_path.name,
            "image_path": str(image_path),
            "num_eggs": 0,
            "masks": [],
            "contours": [],
            "combined_mask": None,
            "processing_time_s": 0.0,
            "error": None,
        }

        try:
            img_bgr = load_image_bgr(image_path)
            h, w = img_bgr.shape[:2]

            # Step 1: Preprocess
            prep_cfg = self.config["preprocessing"]
            blurred, hsv, gray = preprocess_image(
                img_bgr,
                gaussian_kernel=prep_cfg["gaussian_blur_kernel"],
                gaussian_sigma=prep_cfg["gaussian_blur_sigma"],
            )

            # Step 2: HSV thresholding
            cs_cfg = self.config["color_space"]
            if cs_cfg.get("use_hsv", True):
                hsv_lower = tuple(cs_cfg["hsv_lower"])
                hsv_upper = tuple(cs_cfg["hsv_upper"])
                mask_hsv = threshold_hsv(hsv, hsv_lower, hsv_upper)
            else:
                mask_hsv = np.zeros((h, w), dtype=np.uint8)

            # Step 3: Otsu on grayscale
            if cs_cfg.get("use_otsu_fallback", True):
                mask_otsu, otsu_val = threshold_otsu(gray, invert=False)
                # If Otsu inverts (background brighter than eggs), flip
                egg_pixels_otsu = np.sum(mask_otsu > 0) / (h * w)
                if egg_pixels_otsu > 0.7:  # More than 70% = background selected, invert
                    mask_otsu = cv2.bitwise_not(mask_otsu)
            else:
                mask_otsu = np.zeros((h, w), dtype=np.uint8)

            # Step 4: Combine HSV + Otsu
            combined = combine_threshold_masks([mask_hsv, mask_otsu], mode="union")

            # Step 5: Morphological cleaning
            morph_cfg = self.config["morphology"]
            cleaned = clean_mask(
                combined,
                opening_kernel=morph_cfg["opening_kernel_size"],
                opening_iter=morph_cfg["opening_iterations"],
                closing_kernel=morph_cfg["closing_kernel_size"],
                closing_iter=morph_cfg["closing_iterations"],
                dilation_kernel=morph_cfg["dilation_kernel_size"],
                dilation_iter=morph_cfg["dilation_iterations"],
                kernel_shape=morph_cfg["kernel_shape"],
            )

            # Step 6: Watershed for touching eggs
            ws_cfg = self.config["watershed"]
            if ws_cfg.get("use_watershed", True):
                final_mask, watershed_labels = self._apply_watershed(cleaned, ws_cfg)
            else:
                final_mask = cleaned
                watershed_labels = None

            # Step 7: Find and filter contours
            contour_cfg = self.config["contour_detection"]
            contours = find_contours(final_mask)
            contour_params = {
                "min_area": contour_cfg["min_contour_area"],
                "max_area": contour_cfg["max_contour_area"],
                "min_circularity": contour_cfg["min_circularity"],
                "min_solidity": contour_cfg["min_solidity"],
                "min_aspect_ratio": contour_cfg["min_aspect_ratio"],
                "max_aspect_ratio": contour_cfg["max_aspect_ratio"],
            }
            egg_contours, egg_features, rejected = filter_egg_contours(
                contours, contour_params
            )

            # Step 8: Extract per-egg masks
            instance_masks = contours_to_masks(egg_contours, img_bgr.shape[:2])
            combined_mask = contours_to_combined_mask(egg_contours, img_bgr.shape[:2])

            result["num_eggs"] = len(instance_masks)
            result["masks"] = instance_masks
            result["contours"] = egg_contours
            result["combined_mask"] = combined_mask
            result["egg_features"] = egg_features

            # Save outputs
            if output_dir is not None:
                self._save_outputs(
                    image_path, img_bgr, instance_masks, combined_mask, output_dir
                )

        except Exception as e:
            logger.error(f"Error segmenting {image_path.name}: {e}", exc_info=True)
            result["error"] = str(e)

        result["processing_time_s"] = round(time.perf_counter() - start_time, 3)
        logger.debug(
            f"{image_path.name}: {result['num_eggs']} eggs ({result['processing_time_s']:.3f}s)"
        )
        return result

    def _apply_watershed(
        self,
        mask: np.ndarray,
        ws_config: dict,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply watershed algorithm to separate touching eggs.

        Algorithm:
        1. Compute distance transform on the binary mask.
           Each foreground pixel gets the distance to the nearest background pixel.
        2. Threshold the distance transform to find "sure foreground"
           (peaks = centers of individual eggs, far from boundaries).
        3. Dilate the mask to find "sure background" (clearly background pixels).
        4. Unknown region = everything between sure foreground and background.
        5. Apply watershed with sure-foreground seed markers.
        6. Each watershed basin corresponds to one egg.

        Parameters
        ----------
        mask : np.ndarray
            Cleaned binary mask.
        ws_config : dict

        Returns
        -------
        segmented_mask : np.ndarray
            Binary mask after watershed (individual regions separated).
        labels : np.ndarray or None
            Watershed label map.
        """
        if np.sum(mask > 0) == 0:
            return mask, None

        dt_thresh = ws_config.get("distance_transform_threshold", 0.4)
        min_fg_area = ws_config.get("min_sure_foreground_area", 500)

        # Distance transform
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        max_dist = dist.max()
        if max_dist == 0:
            return mask, None

        # Sure foreground: pixels with distance > threshold * max_distance
        _, sure_fg = cv2.threshold(dist, dt_thresh * max_dist, 255, cv2.THRESH_BINARY)
        sure_fg = sure_fg.astype(np.uint8)

        # Remove tiny foreground regions
        contours_fg, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sure_fg_clean = np.zeros_like(sure_fg)
        for c in contours_fg:
            if cv2.contourArea(c) >= min_fg_area:
                cv2.drawContours(sure_fg_clean, [c], -1, 255, thickness=cv2.FILLED)

        if np.sum(sure_fg_clean > 0) == 0:
            return mask, None

        # Sure background: dilated mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sure_bg = cv2.dilate(mask, kernel, iterations=3)

        # Unknown region
        unknown = cv2.subtract(sure_bg, sure_fg_clean)

        # Marker labeling
        num_markers, markers = cv2.connectedComponents(sure_fg_clean)

        # Shift markers: 0 = unknown (add 1 to all, then set unknown back to 0)
        markers = markers + 1
        markers[unknown == 255] = 0

        # Apply watershed on BGR image (requires 3-channel)
        h, w = mask.shape[:2]
        img_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Use mask as "image"
        markers = cv2.watershed(img_3ch, markers)

        # Create final segmented mask (remove watershed boundaries = -1)
        watershed_mask = np.zeros((h, w), dtype=np.uint8)
        watershed_mask[markers > 1] = 255  # Label 1 = background

        return watershed_mask, markers

    def _save_outputs(
        self,
        image_path: Path,
        img_bgr: np.ndarray,
        instance_masks: List[np.ndarray],
        combined_mask: np.ndarray,
        output_dir: Path,
    ) -> None:
        """Save masks to output directory."""
        stem = image_path.stem
        masks_dir = Path(output_dir) / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        for i, mask in enumerate(instance_masks):
            save_binary_mask(mask, masks_dir / f"{stem}_mask_{i:02d}.png")

        save_binary_mask(combined_mask, masks_dir / f"{stem}_combined_mask.png")

    def segment_directory(
        self,
        input_dir: str,
        output_dir: str,
        max_images: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Segment all images in a directory."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        image_paths = collect_images(input_dir)
        if max_images:
            image_paths = image_paths[:max_images]

        logger.info(f"Classical segmentation: {len(image_paths)} images from {input_dir}")
        results = []
        for i, path in enumerate(image_paths, 1):
            logger.info(f"[{i}/{len(image_paths)}] {path.name}")
            result = self.segment_image(path, output_dir)
            results.append(result)
        return results
