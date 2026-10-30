"""
task2_classical_segmentation/src/egg_segmenter.py (Improved Classical Vision v3)
=================================================================================
Robust classical computer vision egg segmentation pipeline.

Key algorithmic improvements:
1. Preprocessing:
   - Bilateral filtering for edge-preserving noise reduction on egg shells.
   - Perceptual CIELAB & HSV color-space transformations.
   - CLAHE (Contrast Limited Adaptive Histogram Equalization) on lightness L
     to normalize uneven illumination, shadows, and subtle white-on-white edges.
2. Non-Egg Chromaticity Rejection:
   - Strictly suppresses green vegetation (herbs, salads, garnishes) and blue/cyan
     dishes/cookware based on LAB (a* < 120, b* < 118) and HSV hue bounds.
   - Suppresses deep black cast shadows.
3. Hierarchical Multi-Level Otsu for Dishes & Cookware:
   - When eggs sit on white plates, ceramic dishes, or stainless pans, global Otsu
     groups the dish and eggs together into one massive component.
   - Hierarchical sub-thresholding within components > 75,000 px cleanly isolates
     the egg core from the dish surface.
4. Dual-Band Egg Chromaticity Segmentation:
   - Dynamic thresholding for white/cream eggs (high L, low saturation, neutral hue)
     and brown/tan eggs (warm hue, moderate saturation, positive a* and b*).
5. Per-Component Distance Transform & Marker-Controlled Watershed:
   - Evaluates Euclidean distance transform on each connected component independently.
   - Applies watershed segmentation only when multiple seeds exist, cleanly
     separating touching eggs in cartons without corrupting isolated eggs.
6. Convexity & Boundary Refinement:
   - Filled binary masks with flood-fill hole closure to handle specular reflections.
   - Geometric contour filtering (area, circularity >= 0.35, solidity >= 0.78,
     aspect ratio [0.45, 2.30], and border contact safeguards).
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
    get_non_egg_rejection_mask,
    threshold_egg_chromaticity,
    threshold_hierarchical_otsu,
    threshold_otsu,
)
from task2_classical_segmentation.src.contour_detection import (
    contours_to_combined_mask,
    contours_to_masks,
    filter_egg_contours,
    find_contours,
)
from task2_classical_segmentation.src.morphology import clean_mask, fill_holes
from task2_classical_segmentation.src.preprocessing import (
    apply_bilateral_filter,
    apply_clahe,
    apply_gaussian_blur,
    to_hsv,
    to_lab,
    to_grayscale,
)

logger = get_logger(__name__)


class ClassicalEggSegmenter:
    """
    Production-grade classical computer vision egg segmenter.

    Uses strictly traditional computer vision algorithms (OpenCV, morphology,
    color-space geometry, hierarchical thresholding, and watershed).
    Contains zero neural networks or learned weights.
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
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        return {}

    def _deep_update(self, base: dict, override: dict) -> None:
        for k, v in override.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                self._deep_update(base[k], v)
            else:
                base[k] = v

    def _apply_watershed_per_component(
        self,
        cleaned_mask: np.ndarray,
        dt_thresh: float = 0.40,
        min_seed_area: int = 350,
    ) -> List[np.ndarray]:
        """
        Apply distance transform watershed independently to each connected component.

        Separates touching eggs in clusters while leaving clean, isolated eggs untouched.
        """
        cnts, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        separated_masks = []

        for comp in cnts:
            area = cv2.contourArea(comp)
            if area < 1000:
                continue

            comp_img = np.zeros_like(cleaned_mask)
            cv2.drawContours(comp_img, [comp], -1, 255, -1)

            # Compact, convex shapes with single peak need no watershed
            p = cv2.arcLength(comp, True)
            circ = 4 * np.pi * area / (p * p) if p > 0 else 0
            hull = cv2.convexHull(comp)
            ha = cv2.contourArea(hull)
            sol = area / ha if ha > 0 else 0

            if area < 30000 and circ >= 0.55 and sol >= 0.88:
                separated_masks.append(comp_img)
                continue

            # Compute Euclidean distance transform
            dist = cv2.distanceTransform(comp_img, cv2.DIST_L2, 5)
            max_d = dist.max()
            if max_d < 12:
                separated_masks.append(comp_img)
                continue

            # Threshold distance transform to locate distinct centers of mass
            _, sure_fg = cv2.threshold(dist, dt_thresh * max_d, 255, cv2.THRESH_BINARY)
            sure_fg = sure_fg.astype(np.uint8)
            sure_fg = cv2.morphologyEx(
                sure_fg,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            )

            num_seeds, markers = cv2.connectedComponents(sure_fg)

            if num_seeds <= 2:
                # 0 background + 1 seed = single egg
                separated_masks.append(comp_img)
            else:
                # Multiple seeds -> watershed separation
                sure_bg = cv2.dilate(
                    comp_img,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                    iterations=2,
                )
                unknown = cv2.subtract(sure_bg, sure_fg)
                markers = markers + 1
                markers[unknown == 255] = 0

                comp_3ch = cv2.cvtColor(comp_img, cv2.COLOR_GRAY2BGR)
                markers = cv2.watershed(comp_3ch, markers)

                for m_id in range(2, num_seeds + 2):
                    egg_m = (markers == m_id).astype(np.uint8) * 255
                    egg_cnts, _ = cv2.findContours(egg_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if egg_cnts and cv2.contourArea(egg_cnts[0]) > 800:
                        separated_masks.append(egg_m)

        return separated_masks

    def segment_image(
        self,
        image_input: Union[Path, str, np.ndarray],
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Segment eggs in a single image or in-memory BGR numpy array frame.

        Parameters
        ----------
        image_input : Path, str, or np.ndarray
        output_dir : Path, optional

        Returns
        -------
        dict with keys:
            image_name, image_path, num_eggs, masks, contours,
            combined_mask, egg_features, processing_time_s, error
        """
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
            "egg_features": [],
            "processing_time_s": 0.0,
            "error": None,
        }

        try:
            h, w = img_bgr.shape[:2]

            # 1. Color-space transformations
            lab = to_lab(img_bgr)
            hsv = to_hsv(img_bgr)
            gray = to_grayscale(img_bgr)

            L = lab[:, :, 0]

            # 2. Non-egg chromaticity rejection mask (vegetation, blue dishes, dark shadows)
            cs_cfg = self.config.get("color_space", {})
            suppress_green = cs_cfg.get("suppress_green", True)
            suppress_blue = cs_cfg.get("suppress_blue", True)
            suppress_shadow = cs_cfg.get("suppress_deep_shadow", True)

            non_egg_mask = get_non_egg_rejection_mask(
                hsv=hsv,
                lab=lab,
                suppress_green=suppress_green,
                suppress_blue=suppress_blue,
                suppress_shadow=suppress_shadow,
            )

            # 3. Preprocessing: Bilateral filtering + CLAHE local contrast
            prep_cfg = self.config.get("preprocessing", {})
            use_bilateral = prep_cfg.get("use_bilateral", True)

            if use_bilateral:
                d = prep_cfg.get("bilateral_d", 7)
                sc = prep_cfg.get("bilateral_sigma_color", 50.0)
                ss = prep_cfg.get("bilateral_sigma_space", 50.0)
                L_filtered = apply_bilateral_filter(L, d=d, sigma_color=sc, sigma_space=ss)
            else:
                k = prep_cfg.get("gaussian_blur_kernel", 5)
                L_filtered = apply_gaussian_blur(L, kernel_size=k)

            use_clahe = prep_cfg.get("use_clahe", True)
            if use_clahe:
                clip_limit = prep_cfg.get("clahe_clip_limit", 2.5)
                tile_size = tuple(prep_cfg.get("clahe_tile_grid_size", [8, 8]))
                L_enhanced = apply_clahe(L_filtered, clip_limit=clip_limit, tile_grid_size=tile_size)
            else:
                L_enhanced = L_filtered

            # 4. Strategy 1: Global Otsu with Hierarchical sub-thresholding for plates/pans
            t1, m_otsu = cv2.threshold(L_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            m_otsu[non_egg_mask] = 0

            use_hierarchical = cs_cfg.get("use_hierarchical_otsu", True)
            hierarchical_area = cs_cfg.get("hierarchical_area_threshold", 75000)

            if use_hierarchical:
                m_otsu_refined = threshold_hierarchical_otsu(
                    image_gray=L_enhanced,
                    base_mask=m_otsu,
                    min_component_area=hierarchical_area,
                    non_egg_mask=non_egg_mask,
                )
            else:
                m_otsu_refined = m_otsu

            # 5. Strategy 2: Dual-band egg chromaticity segmentation
            m_color = threshold_egg_chromaticity(
                hsv=hsv,
                lab=lab,
                non_egg_mask=non_egg_mask,
            )

            # Filter oversized components from m_color to prevent dishes/tabletops from bridging
            color_cnts, _ = cv2.findContours(m_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            m_color_filtered = np.zeros_like(m_color)
            for cc in color_cnts:
                if cv2.contourArea(cc) < hierarchical_area:
                    cv2.drawContours(m_color_filtered, [cc], -1, 255, -1)

            # 6. Candidate combination
            combined = cv2.bitwise_or(m_otsu_refined, m_color_filtered)
            combined[non_egg_mask] = 0

            # 7. Morphological cleaning & void closure
            morph_cfg = self.config.get("morphology", {})
            open_k = morph_cfg.get("opening_kernel_size", 5)
            open_iter = morph_cfg.get("opening_iterations", 1)
            close_k = morph_cfg.get("closing_kernel_size", 15)
            close_iter = morph_cfg.get("closing_iterations", 2)

            open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
            close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))

            # Close first to bridge specular highlights and interior shading voids
            cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_kernel, iterations=close_iter)
            # Open to remove small extraneous debris
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel, iterations=open_iter)

            # 8. Watershed separation for touching egg clusters
            ws_cfg = self.config.get("watershed", {})
            use_ws = ws_cfg.get("use_watershed", True)

            if use_ws:
                dt_thresh = ws_cfg.get("distance_transform_threshold", 0.40)
                min_seed_a = ws_cfg.get("min_sure_foreground_area", 400)
                separated_masks = self._apply_watershed_per_component(
                    cleaned_mask=cleaned,
                    dt_thresh=dt_thresh,
                    min_seed_area=min_seed_a,
                )
            else:
                comp_cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                separated_masks = []
                for cc in comp_cnts:
                    m = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(m, [cc], -1, 255, -1)
                    separated_masks.append(m)

            # 9. Contour filtering & feature extraction
            contour_cfg = self.config.get("contour_detection", {})
            filter_params = {
                "min_area": contour_cfg.get("min_contour_area", 1200),
                "max_area": contour_cfg.get("max_contour_area", 95000),
                "min_circularity": contour_cfg.get("min_circularity", 0.35),
                "min_solidity": contour_cfg.get("min_solidity", 0.78),
                "min_aspect_ratio": contour_cfg.get("min_aspect_ratio", 0.45),
                "max_aspect_ratio": contour_cfg.get("max_aspect_ratio", 2.30),
                "max_border_touch_count": contour_cfg.get("max_border_touch_count", 2),
                "max_border_touch_area": contour_cfg.get("max_border_touch_area", 45000),
            }

            candidate_contours = []
            for m in separated_masks:
                cnts_m, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts_m:
                    candidate_contours.append(max(cnts_m, key=cv2.contourArea))

            egg_contours, egg_features, rejected = filter_egg_contours(
                contours=candidate_contours,
                filter_params=filter_params,
                image_shape=(h, w),
            )

            # Secondary non-egg internal color verification
            verified_contours = []
            verified_features = []
            verified_masks = []

            for contour, feat in zip(egg_contours, egg_features):
                m_single = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(m_single, [contour], -1, 255, thickness=cv2.FILLED)

                # Flood-fill hole closure to ensure solid masks
                if morph_cfg.get("fill_holes", True):
                    m_single = fill_holes(m_single)

                # Re-check contours of filled mask
                cnts_filled, _ = cv2.findContours(m_single, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts_filled:
                    continue
                final_c = max(cnts_filled, key=cv2.contourArea)

                # Internal non-egg color check (reject if green or blue > 12%)
                non_egg_frac = np.mean(non_egg_mask[m_single > 0]) if np.any(m_single > 0) else 0.0
                if non_egg_frac > 0.12:
                    continue

                verified_contours.append(final_c)
                verified_features.append(feat)
                verified_masks.append(m_single)

            combined_mask = contours_to_combined_mask(verified_contours, (h, w))

            result["num_eggs"] = len(verified_masks)
            result["masks"] = verified_masks
            result["contours"] = verified_contours
            result["combined_mask"] = combined_mask
            result["egg_features"] = verified_features

            if output_dir is not None and image_path is not None:
                self._save_outputs(image_path, img_bgr, verified_masks, combined_mask, output_dir)

        except Exception as e:
            logger.error(f"Error segmenting {image_name}: {e}", exc_info=True)
            result["error"] = str(e)

        result["processing_time_s"] = round(time.perf_counter() - start_time, 3)
        return result

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
        # Remove any stale individual instance masks from previous runs
        for old_mask in masks_dir.glob(f"{stem}_mask_*.png"):
            if "combined" not in old_mask.name:
                try:
                    old_mask.unlink()
                except OSError:
                    pass
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
