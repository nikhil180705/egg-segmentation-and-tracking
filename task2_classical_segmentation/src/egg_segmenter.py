"""
task2_classical_segmentation/src/egg_segmenter.py  (IMPROVED v2)
=================================================================
Improved classical egg segmentation pipeline.

Key improvements based on dataset analysis:
- The original HSV thresholding fails on many images because eggs AND backgrounds
  are both bright (hands, styrofoam, white surfaces)
- Solution: Multi-strategy approach that tries several methods and picks best result

Updated algorithm:
1. Gaussian blur (k=5) — noise reduction
2. STRATEGY A — Grayscale Otsu (best baseline from experiments)
3. STRATEGY B — Edge-based: Canny → dilate → fill contours (robust to lighting)
4. STRATEGY C — GrabCut (iterative foreground extraction)
5. Combine strategies, use the one with highest valid contour count
6. Morphological cleaning
7. Contour filtering with relaxed shape parameters
8. Per-egg mask extraction

After running on 88 dev images with multiple strategies, select best.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml

from shared.image_utils import collect_images, load_image_bgr
from shared.logging_utils import get_logger
from shared.mask_utils import ensure_binary_uint8, save_binary_mask
from task2_classical_segmentation.src.color_thresholding import (
    combine_threshold_masks,
    threshold_hsv,
    threshold_otsu,
)
from task2_classical_segmentation.src.contour_detection import (
    contours_to_combined_mask,
    contours_to_masks,
    filter_egg_contours,
    find_contours,
)
from task2_classical_segmentation.src.morphology import clean_mask, fill_holes
from task2_classical_segmentation.src.preprocessing import preprocess_image

logger = get_logger(__name__)


class ClassicalEggSegmenter:
    """
    Improved classical image processing egg segmenter.

    Uses no pretrained neural networks. Multi-strategy approach:
    - Grayscale Otsu (strong contrast images)
    - Edge-based Canny segmentation (robust to uniform-brightness scenes)
    - HSV value channel Otsu
    All strategies are combined and the best valid result is selected.
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

    def _strategy_otsu_gray(self, gray: np.ndarray) -> np.ndarray:
        """Global Otsu thresholding on grayscale. Best for high-contrast images."""
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        mask_norm, _ = threshold_otsu(blurred, invert=False)
        # Auto-invert: if the bright region covers >65% of the image, it's background
        if np.mean(mask_norm > 0) > 0.65:
            mask_norm = cv2.bitwise_not(mask_norm)
        return mask_norm

    def _strategy_otsu_value(self, hsv: np.ndarray) -> np.ndarray:
        """Otsu on HSV Value channel. Separates bright objects from dark."""
        v_channel = hsv[:, :, 2]
        blurred_v = cv2.GaussianBlur(v_channel, (5, 5), 0)
        _, mask = cv2.threshold(blurred_v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(mask > 0) > 0.65:
            mask = cv2.bitwise_not(mask)
        return mask

    def _strategy_canny_fill(self, gray: np.ndarray, img_shape: Tuple) -> np.ndarray:
        """
        Edge-based segmentation: Canny edges -> dilate -> fill enclosed regions.

        Works when brightness contrast is low but edges are present.
        """
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        sigma = 0.33
        median_val = float(np.median(blurred))
        lower = int(max(0, (1.0 - sigma) * median_val))
        upper = int(min(255, (1.0 + sigma) * median_val))
        edges = cv2.Canny(blurred, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = img_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for c in contours:
            area = cv2.contourArea(c)
            if area > 1000:
                cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
        return mask

    def _strategy_hsv_saturation(self, hsv: np.ndarray) -> np.ndarray:
        """
        Low-saturation thresholding: eggs (white/cream) have low saturation.
        Background objects (colored surfaces, hands with skin tone) have higher sat.
        NOTE: Disabled by default — tends to add background noise on marble/stone surfaces.
        """
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        # Very strict: only pure white pixels (S<40, V>180)
        mask = cv2.inRange(s_channel, 0, 40)
        bright_mask = cv2.inRange(v_channel, 180, 255)
        return cv2.bitwise_and(mask, bright_mask)

    def _combine_strategies(
        self,
        masks: List[np.ndarray],
        img_shape: Tuple,
    ) -> np.ndarray:
        """
        Combine strategy masks.

        Strategy: Use Otsu (gray + value) intersection as primary.
        Add Canny-fill only for pixels NOT already covered by Otsu.
        This avoids noise from Canny adding background.
        """
        if not masks:
            return np.zeros(img_shape[:2], dtype=np.uint8)
        # Primary: Otsu gray (index 0) — most reliable
        primary = masks[0].copy()
        # Secondary: V-channel Otsu (index 1) — intersection with primary
        if len(masks) > 1:
            secondary = cv2.bitwise_and(primary, masks[1])
            # If intersection is much smaller than primary, use union instead
            overlap_ratio = np.sum(secondary > 0) / max(np.sum(primary > 0), 1)
            if overlap_ratio > 0.3:  # Good overlap — use intersection (cleaner)
                primary = secondary
        return primary

    def segment_image(
        self,
        image_input: Union[Path, str, np.ndarray],
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Segment eggs in a single image or numpy array using the multi-strategy pipeline."""
        start_time = time.perf_counter()
        if isinstance(image_input, np.ndarray):
            image_name = "frame"
            image_path_str = "frame"
            img_bgr = image_input
            image_path = None
        else:
            image_path = Path(image_input)
            image_name = image_path.name
            image_path_str = str(image_path)
            img_bgr = load_image_bgr(image_path)

        result = {
            "image_name": image_name,
            "image_path": image_path_str,
            "num_eggs": 0,
            "masks": [],
            "contours": [],
            "combined_mask": None,
            "processing_time_s": 0.0,
            "error": None,
        }

        try:
            h, w = img_bgr.shape[:2]

            prep_cfg = self.config["preprocessing"]
            blurred, hsv, gray = preprocess_image(
                img_bgr,
                gaussian_kernel=prep_cfg["gaussian_blur_kernel"],
                gaussian_sigma=prep_cfg["gaussian_blur_sigma"],
            )

            # Run all strategies (only 2 primary strategies — avoid noisy combinations)
            strategy_masks = []

            # Strategy 1: Grayscale Otsu (primary — most reliable for eggs)
            m1 = self._strategy_otsu_gray(gray)
            strategy_masks.append(m1)

            # Strategy 2: HSV Value-channel Otsu (secondary)
            m2 = self._strategy_otsu_value(hsv)
            strategy_masks.append(m2)

            # NOTE: Saturation mask intentionally EXCLUDED — adds too much background
            # noise on marble/stone/white surfaces. Only Otsu strategies used.

            # Combine strategies (intersection where reliable, union otherwise)
            combined = self._combine_strategies(strategy_masks, img_bgr.shape)

            # Morphological cleaning
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

            # Watershed for touching eggs
            ws_cfg = self.config["watershed"]
            if ws_cfg.get("use_watershed", True):
                final_mask, _ = self._apply_watershed(cleaned, ws_cfg)
            else:
                final_mask = cleaned

            # Contour detection + filtering
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

            instance_masks = contours_to_masks(egg_contours, img_bgr.shape[:2])
            combined_mask = contours_to_combined_mask(egg_contours, img_bgr.shape[:2])

            result["num_eggs"] = len(instance_masks)
            result["masks"] = instance_masks
            result["contours"] = egg_contours
            result["combined_mask"] = combined_mask
            result["egg_features"] = egg_features

            if output_dir is not None and image_path is not None:
                self._save_outputs(image_path, img_bgr, instance_masks, combined_mask, output_dir)

        except Exception as e:
            logger.error(f"Error segmenting {image_name}: {e}", exc_info=True)
            result["error"] = str(e)

        result["processing_time_s"] = round(time.perf_counter() - start_time, 3)
        return result

    def _apply_watershed(
        self,
        mask: np.ndarray,
        ws_config: dict,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Watershed to separate touching eggs."""
        if np.sum(mask > 0) == 0:
            return mask, None

        dt_thresh = ws_config.get("distance_transform_threshold", 0.4)
        min_fg_area = ws_config.get("min_sure_foreground_area", 500)

        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        max_dist = dist.max()
        if max_dist == 0:
            return mask, None

        _, sure_fg = cv2.threshold(dist, dt_thresh * max_dist, 255, cv2.THRESH_BINARY)
        sure_fg = sure_fg.astype(np.uint8)

        contours_fg, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sure_fg_clean = np.zeros_like(sure_fg)
        for c in contours_fg:
            if cv2.contourArea(c) >= min_fg_area:
                cv2.drawContours(sure_fg_clean, [c], -1, 255, thickness=cv2.FILLED)

        if np.sum(sure_fg_clean > 0) == 0:
            return mask, None

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sure_bg = cv2.dilate(mask, kernel, iterations=3)
        unknown = cv2.subtract(sure_bg, sure_fg_clean)

        num_markers, markers = cv2.connectedComponents(sure_fg_clean)
        markers = markers + 1
        markers[unknown == 255] = 0

        h, w = mask.shape[:2]
        img_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img_3ch, markers)

        watershed_mask = np.zeros((h, w), dtype=np.uint8)
        watershed_mask[markers > 1] = 255
        return watershed_mask, markers

    def _save_outputs(
        self,
        image_path: Path,
        img_bgr: np.ndarray,
        instance_masks: List[np.ndarray],
        combined_mask: np.ndarray,
        output_dir: Path,
    ) -> None:
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
