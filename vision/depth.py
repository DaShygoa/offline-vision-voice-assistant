"""Depth-guided class-agnostic object localization (零样本物体定位).

Uses monocular depth estimation (Depth Anything V2 Small, ONNX Runtime CPU) to
locate the nearest object region — completely category-free, no pretrained labels.

Pipeline (replaces YOLO's role in identify/remember):
  frame → DA-V2 depth map → normalize → near-threshold mask
        → connected components → score (area + centrality + depth proximity)
        → best region bbox → crop → ResNet18 retrieval

References:
  - Depth Anything V2 (Yang et al., NeurIPS 2024)
  - DepSeg: depth-guided point prompts + foundation models (arXiv 2512.05529, 2025)
  - ZISVFM: depth-colorized SAM zero-shot instance segmentation (arXiv 2502.03266, 2025)

Model prep (one-time):
    python tools/prepare_models.py --depth
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Depth Anything V2 preprocessing (ImageNet stats, 518×518 default)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Intersection-over-union for two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / float(union) if union > 0 else 0.0


@dataclass
class DepthRegion:
    """A candidate object region found via depth analysis."""
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2
    score: float
    mean_depth: float          # normalized 0=near, 1=far
    area_frac: float           # fraction of frame area
    mask: Optional[np.ndarray] = None  # per-pixel binary mask (H×W), 1=object
    skin_separated: bool = False  # hand already removed while selecting component

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class DepthEstimator:
    """Monocular depth estimation + nearest-object localization (CPU only)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.input_size = int(cfg.get("input_size", 518))
        self.near_percentile = float(cfg.get("near_percentile", 80.0))
        self.min_area_frac = float(cfg.get("min_area_frac", 0.005))
        self.max_area_frac = float(cfg.get("max_area_frac", 0.45))
        self.expand_ratio = float(cfg.get("expand_ratio", 0.06))
        self.model_path = cfg.get("model_path", "models/depth_anything_v2/depth_anything_v2_vits.onnx")
        # optional MobileSAM mask refinement (pixel-precise bbox instead of rectangular)
        self.use_mobilesam = bool(cfg.get("use_mobilesam", False))
        self._sam_refiner = None
        self.ready = False
        self.device_used = "none"
        self._sess = None
        self._low_sess = None
        self.low_input_size = int(cfg.get("low_input_size", 320))
        self._input_name = None

    def load(self) -> None:
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"Depth model not found: {self.model_path}\n"
                "  run: python tools/prepare_models.py --depth"
            )
        import onnxruntime as ort

        # Cap intra-op threads: DA-V2 inference otherwise saturates all cores
        # and starves the UI/streaming threads, dropping frame rate.
        n_threads = int(self.cfg.get("onnx_threads", 4))
        opts = ort.SessionOptions()
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        opts.intra_op_num_threads = max(1, n_threads)
        opts.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(
            self.model_path, opts, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._sess.get_inputs()[0].name
        self.device_used = f"ONNXRuntime:CPU(threads={n_threads})"

        # Low-res session for the live heatmap loop (same model, faster input).
        # ONNX models with dynamic spatial dims accept any H×W; 320² runs ~3.7x
        # faster than 518² (449ms vs 1688ms) which lifts the heatmap from 0.6Hz
        # to ~2.2Hz while identify/remember keep full resolution.
        try:
            opts2 = ort.SessionOptions()
            opts2.intra_op_num_threads = max(1, n_threads)
            opts2.inter_op_num_threads = 1
            opts2.add_session_config_entry("session.intra_op.allow_spinning", "0")
            opts2.add_session_config_entry("session.inter_op.allow_spinning", "0")
            self._low_sess = ort.InferenceSession(
                self.model_path, opts2, providers=["CPUExecutionProvider"]
            )
            log.info("Depth low-res session ready (%dpx for heatmap)", self.low_input_size)
        except Exception as e:
            log.warning("Low-res depth session unavailable (%s); heatmap uses full res", e)
            self._low_sess = None

        self.ready = True
        log.info("Depth Anything V2 loaded (%s, threads=%d)", self.model_path, n_threads)

        # optional MobileSAM refinement
        if self.use_mobilesam:
            try:
                from .mobilesam import MobileSAMRefiner
                self._sam_refiner = MobileSAMRefiner(self.cfg)
                self._sam_refiner.load()
                log.info("MobileSAM refinement enabled (%s)", self._sam_refiner.device_used)
            except Exception as e:
                log.warning("MobileSAM load failed (depth bbox used as-is): %s", e)
                self._sam_refiner = None
                self.use_mobilesam = False

    # ------------------------------------------------------------- infer
    def estimate_depth(self, frame_bgr: np.ndarray, fast: bool = False) -> Optional[np.ndarray]:
        """Return relative depth map (H×W float32, 0=near … 1=far), or None.

        fast=True uses the low-resolution session (input ~320²) for the live
        heatmap loop — measured 449ms vs 1688ms at full 518² — while identify /
        remember keep the full-resolution path for accuracy.
        """
        if not self.ready or frame_bgr is None or frame_bgr.size == 0:
            return None
        t0 = time.perf_counter()
        h, w = frame_bgr.shape[:2]

        if fast and self._low_sess is not None:
            sess, s = self._low_sess, self.low_input_size
        else:
            sess, s = self._sess, self.input_size

        img = cv2.resize(frame_bgr, (s, s), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        blob = img.transpose(2, 0, 1)[None]
        blob = np.ascontiguousarray(blob, dtype=np.float32)

        try:
            out = sess.run(None, {self._input_name: blob})[0]
        except Exception as e:
            log.error("Depth inference failed: %s", e)
            return None

        depth = np.asarray(out, dtype=np.float32).squeeze()
        # resize back to frame resolution
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        # normalize to [0, 1]
        d_min, d_max = float(depth.min()), float(depth.max())
        if d_max - d_min > 1e-8:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)
        # Depth Anything V2 outputs INVERSE depth (disparity): larger = nearer.
        # Invert so 0 = nearest, 1 = farthest (our convention everywhere downstream).
        depth = 1.0 - depth
        dt_ms = (time.perf_counter() - t0) * 1000.0
        log.debug("Depth estimated in %.1f ms (%dx%d)", dt_ms, w, h)
        return depth

    def find_nearest_object(self, frame_bgr: np.ndarray, depth: Optional[np.ndarray] = None) -> Optional[DepthRegion]:
        """Locate the best object region (top-1) via depth saliency."""
        rs = self.find_top_regions(frame_bgr, depth, k=1)
        return rs[0] if rs else None

    def find_top_regions(self, frame_bgr: np.ndarray, depth: Optional[np.ndarray] = None,
                         k: int = 4) -> List[DepthRegion]:
        """Return top-k candidate object regions ranked by score.

        Depth alone is ambiguous in cluttered scenes (e.g. a hand-held box vs the
        person holding it vs nearby furniture). Returning several candidates lets
        the feature matcher pick the true target instead of trusting one guess.
        """
        return self._compute_regions(frame_bgr, depth, k=k)

    def _compute_regions(self, frame_bgr: np.ndarray,
                         depth: Optional[np.ndarray] = None, k: int = 4) -> List[DepthRegion]:
        if depth is None:
            depth = self.estimate_depth(frame_bgr)
        if depth is None:
            return []

        h, w = depth.shape[:2]
        frame_area = float(h * w)

        # 1) depth CONTRAST saliency
        blur_k = max(15, min(h, w) // 12) | 1
        bg_depth = cv2.GaussianBlur(depth, (blur_k, blur_k), 0)
        contrast = bg_depth - depth

        contrast_thresh = np.percentile(contrast, 75.0)
        candidate_mask = (contrast >= max(contrast_thresh, 0.01)).astype(np.uint8)

        # 2) morphological cleanup
        ksize = max(7, min(h, w) // 80)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # 3) connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)

        regions: List[DepthRegion] = []
        for i in range(1, num_labels):
            x, y, cw, ch, area = stats[i]
            area_frac = area / frame_area
            if area_frac < 0.002 or area_frac > self.max_area_frac:
                continue

            comp_depth = depth[labels == i]
            mean_d = float(comp_depth.mean())
            comp_contrast = float(contrast[labels == i].mean())

            cx, cy = centroids[i]
            dist = np.hypot(cx - w / 2, cy - h / 2)
            max_dist = 0.5 * np.hypot(w, h)
            centrality = 1.0 - min(1.0, dist / max_dist)

            depth_prox = 1.0 - mean_d
            contrast_score = min(1.0, comp_contrast * 8.0)
            # Gaussian size preference: a hand-held object is typically 2-10% of the
            # frame. Linear area scoring made large background blobs (person + room)
            # win over the small object being shown — the main detection failure mode.
            area_score = float(np.exp(-((area_frac - 0.05) / 0.12) ** 2))

            score = (0.35 * centrality
                     + 0.25 * depth_prox
                     + 0.25 * contrast_score
                     + 0.15 * area_score)

            comp_mask = (labels == i).astype(np.uint8)
            regions.append(DepthRegion(
                box=(int(x), int(y), int(x + cw), int(y + ch)),
                score=score,
                mean_depth=mean_d,
                area_frac=area_frac,
                mask=comp_mask,
            ))

        # 4) skin-free pass — a hand-held object merges with the hand/arm/body in
        # the depth mask, so the combined blob gets rejected as a person and the
        # object is lost with it. Subtract skin pixels and look for a separate
        # component: the object normally survives on its own.
        # Original candidates are kept so skin-toned objects (wood, bread) still work.
        regions.extend(self._skin_free_regions(frame_bgr, depth, contrast,
                                              candidate_mask, regions, frame_area))

        # Contrast detects boundaries, not whole flat surfaces. A tissue packet
        # can otherwise become a 20px-high strip while the person wins the rank.
        regions.extend(self._surface_regions(frame_bgr, depth))
        regions = [r for r in regions if self._plausible_region(r, w, h)]
        regions.sort(key=lambda r: r.score, reverse=True)
        return regions[:max(1, k)]

    @staticmethod
    def _plausible_region(region: DepthRegion, w: int, h: int) -> bool:
        x1, y1, x2, y2 = region.box
        bw, bh = x2 - x1, y2 - y1
        if min(bw, bh) <= 0 or max(bw, bh) / min(bw, bh) > 4.5:
            return False
        box_area = bw * bh
        # Sparse edge chains spanning a person/room aren't a discrete object.
        return (box_area <= 0.5 * w * h
                and region.area_frac * w * h / box_area >= 0.20
                and bw < 0.95 * w and bh < 0.95 * h)

    def _surface_regions(self, frame: np.ndarray, depth: np.ndarray) -> List[DepthRegion]:
        """Filled near-depth components, with and without hand subtraction.

        Work at preview resolution to keep multiple thresholds inexpensive. Use
        the central field for thresholds so a near wall at the edge cannot hide
        the object being shown. Keep original-colour candidates for skin-toned
        objects; the caller still applies face rejection to raw crops.
        """
        if float(np.ptp(depth)) < 0.02:
            return []
        h, w = depth.shape
        scale = min(1.0, 640.0 / max(h, w))
        sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
        small = cv2.resize(depth, (sw, sh), interpolation=cv2.INTER_AREA)
        rgb = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        central = small[sh // 6:5 * sh // 6, sw // 5:4 * sw // 5]
        if central.size == 0:
            return []
        ycc = cv2.cvtColor(rgb, cv2.COLOR_BGR2YCrCb)
        cr, cb = ycc[:, :, 1], ycc[:, :, 2]
        skin = ((cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin = cv2.dilate(skin, kernel)
        proposals: List[Tuple[DepthRegion, bool]] = []
        for threshold in np.unique(np.percentile(central, [10, 20, 30, 40, 50])):
            near = (small <= threshold).astype(np.uint8)
            for strip_skin in (False, True):
                mask = near.copy()
                if strip_skin:
                    mask[skin > 0] = 0
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
                for i in range(1, n):
                    x, y, bw, bh, area = stats[i]
                    area_frac = float(area) / (sw * sh)
                    if not self.min_area_frac <= area_frac <= self.max_area_frac:
                        continue
                    box = (round(x * w / sw), round(y * h / sh),
                           round((x + bw) * w / sw), round((y + bh) * h / sh))
                    region = DepthRegion(box, 0.0, 0.0, area_frac)
                    if not self._plausible_region(region, w, h):
                        continue
                    component = (labels == i).astype(np.uint8)
                    mean_d = float(small[component > 0].mean())
                    cx, cy = centroids[i]
                    centrality = 1.0 - min(1.0, np.hypot(cx - sw / 2, cy - sh / 2)
                                           / (0.5 * np.hypot(sw, sh)))
                    compactness = float(area) / (bw * bh)
                    score = (0.30 * centrality + 0.25 * (1.0 - mean_d)
                             + 0.20 * compactness + 0.25 * min(1.0, area_frac / 0.05))
                    region.score, region.mean_depth = score, mean_d
                    region.skin_separated = strip_skin
                    region.mask = cv2.resize(component, (w, h), interpolation=cv2.INTER_NEAREST)
                    proposals.append((region, strip_skin))

        # Nested thresholds often trace progressively more of the same surface.
        # Keep the complete component, rather than filling top-k with fragments.
        proposals.sort(key=lambda item: item[0].area_frac, reverse=True)
        unique: List[Tuple[DepthRegion, bool]] = []
        for region, strip_skin in proposals:
            x1, y1, x2, y2 = region.box
            area = (x2 - x1) * (y2 - y1)
            contained = False
            for other, other_stripped in unique:
                # A larger object+hand component must not suppress the separate
                # object recovered by skin subtraction.
                if strip_skin != other_stripped:
                    continue
                ox1, oy1, ox2, oy2 = other.box
                inter = max(0, min(x2, ox2) - max(x1, ox1)) * max(0, min(y2, oy2) - max(y1, oy1))
                other_area = (ox2 - ox1) * (oy2 - oy1)
                if inter / area > 0.85 and other_area <= 3.0 * area:
                    contained = True
                    break
            if not contained:
                unique.append((region, strip_skin))
        return [region for region, _ in unique]

    def _skin_free_regions(self, frame_bgr: np.ndarray, depth: np.ndarray,
                           contrast: np.ndarray, candidate_mask: np.ndarray,
                           existing: List[DepthRegion], frame_area: float) -> List[DepthRegion]:
        """Candidates from the depth mask with skin pixels removed."""
        try:
            h, w = depth.shape[:2]
            ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
            cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
            skin = ((cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)).astype(np.uint8)
            if skin.mean() > 0.6:
                return []  # scene is mostly skin — subtraction would destroy it
            skin = cv2.dilate(skin, np.ones((9, 9), np.uint8), iterations=1)

            mask = candidate_mask.copy()
            mask[skin > 0] = 0
            ksize = max(7, min(h, w) // 80)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            out: List[DepthRegion] = []
            for i in range(1, num):
                x, y, cw, ch, area = stats[i]
                area_frac = area / frame_area
                if area_frac < 0.002 or area_frac > self.max_area_frac:
                    continue
                box = (int(x), int(y), int(x + cw), int(y + ch))
                if any(_iou(box, r.box) > 0.5 for r in existing):
                    continue  # already covered by a normal candidate
                comp_depth = depth[labels == i]
                mean_d = float(comp_depth.mean())
                comp_contrast = float(contrast[labels == i].mean())
                cx, cy = centroids[i]
                dist = np.hypot(cx - w / 2, cy - h / 2)
                centrality = 1.0 - min(1.0, dist / (0.5 * np.hypot(w, h)))
                depth_prox = 1.0 - mean_d
                contrast_score = min(1.0, comp_contrast * 8.0)
                area_score = float(np.exp(-((area_frac - 0.05) / 0.12) ** 2))
                score = (0.35 * centrality + 0.25 * depth_prox
                         + 0.25 * contrast_score + 0.15 * area_score)
                out.append(DepthRegion(
                    box=box, score=score, mean_depth=mean_d, area_frac=area_frac,
                    mask=(labels == i).astype(np.uint8),
                    skin_separated=True,
                ))
            return out
        except Exception as e:
            log.debug("skin-free candidate pass failed: %s", e)
            return []

    def crop_region(self, frame_bgr: np.ndarray, region: DepthRegion,
                    use_mask: bool = True) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Expand + crop one region, optionally cleaning the background via its mask.

        Returns (crop, box) or None.
        """
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = region.box

        bw, bh = x2 - x1, y2 - y1
        dx, dy = bw * self.expand_ratio, bh * self.expand_ratio
        x1 = int(max(0, x1 - dx))
        y1 = int(max(0, y1 - dy))
        x2 = int(min(w, x2 + dx))
        y2 = int(min(h, y2 + dy))

        crop = frame_bgr[y1:y2, x1:x2].copy()
        if crop.size == 0:
            return None

        # Mask out background: convex hull fills the object shape, mild dilation
        # keeps its edges, everything else becomes the object's median colour.
        if use_mask and region.mask is not None:
            crop_mask = region.mask[y1:y2, x1:x2]
            if crop_mask.size > 0 and crop_mask.shape[:2] == crop.shape[:2]:
                mask_uint8 = (crop_mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    hull = np.zeros_like(mask_uint8)
                    all_pts = np.vstack(contours)
                    hull_pts = cv2.convexHull(all_pts)
                    cv2.fillConvexPoly(hull, hull_pts, 255)
                    dilate_k = max(3, min(crop.shape[:2]) // 24) | 1
                    hull = cv2.dilate(hull, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k)))
                    if (hull > 0).any():
                        med = np.median(crop[hull > 0].reshape(-1, 3), axis=0).astype(np.uint8)
                        crop[hull == 0] = med

        # Colour alone cannot distinguish a hand from skin-coloured printing on
        # the object. Preserve all pixels inside the recovered foreground hull;
        # skin-based segmentation is used only while proposing candidate regions.

        return crop, (x1, y1, x2, y2)

    @staticmethod
    def remove_skin(crop: np.ndarray) -> np.ndarray:
        """Replace skin-tone pixels (fingers/hand) with the object's median colour.

        Only acts when skin covers < 55% of the crop — if the object itself is
        skin-toned the median fill would wipe it out, so we leave it alone.
        """
        if crop is None or crop.size == 0:
            return crop
        try:
            ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
            cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
            skin = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
            ratio = float(skin.mean())
            if ratio < 0.02 or ratio > 0.55:
                return crop  # little skin, or object is itself skin-toned
            non_skin = crop[~skin]
            if non_skin.size == 0:
                return crop
            med = np.median(non_skin.reshape(-1, 3), axis=0).astype(np.uint8)
            out = crop.copy()
            out[skin] = med
            return out
        except Exception:
            return crop

    def crop_nearest(self, frame_bgr: np.ndarray) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int], DepthRegion]]:
        """Full pipeline: depth → nearest object → [MobileSAM refine] → mask-clean crop.

        Returns (crop, box, region) or None.
        """
        region = self.find_nearest_object(frame_bgr)
        if region is None:
            return None

        # MobileSAM refinement
        if self._sam_refiner is not None and self._sam_refiner.ready:
            try:
                mask_result = self._sam_refiner.refine_region(frame_bgr, region.box)
                if mask_result is not None:
                    region.box = mask_result.box
                    region.mask = mask_result.mask
            except Exception as e:
                log.debug("MobileSAM refine failed: %s", e)

        r = self.crop_region(frame_bgr, region)
        if r is None:
            return None
        crop, box = r
        return crop, box, region

    # -------------------------------------------------------- visualization
    @staticmethod
    def depth_to_colormap(depth: np.ndarray) -> np.ndarray:
        """Convert normalized depth (0=near, 1=far) to a BGR heatmap for display.

        Uses TURBO colormap: near=red/warm, far=blue/cool (intuitive for demos).
        """
        if depth is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        # invert so near=high value → warm colors in TURBO
        vis = ((1.0 - depth) * 255).astype(np.uint8)
        colored = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
        return colored
