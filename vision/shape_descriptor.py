"""Classical shape descriptor — pure OpenCV, no model, fully offline.

Produces a compact L2-normalized vector from a crop + optional YOLO box:
  * silhouette cues: contours, Hu moments, solidity, extent, circularity
  * bbox geometry: aspect, fill ratio, orientation
  * optional Fourier descriptors of the largest contour

Used as a side-channel next to DINOv2 — never concatenated into the 384-d matrix.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Hu moments are log-scaled (typical range roughly [-10, 10] after -log10)
_HU_SCALE = 8.0
_FOURIER_K = 16  # complex Fourier magnitudes (exclude DC)


class ShapeDescriptor:
    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.use_fourier = bool(cfg.get("use_fourier", True))
        self.use_hu = bool(cfg.get("use_hu", True))
        self.min_side = int(cfg.get("min_side", 32))
        # fixed dim so matrices stay aligned: 7 Hu + 4 geom + 4 extra + Fourier(16)
        self.dim = 0
        self._parts: list[str] = []
        if self.use_hu:
            self._parts.append("hu")
            self.dim += 7
        self._parts.append("geom")
        self.dim += 5  # aspect, fill, circularity, extent, orientation_sin
        if self.use_fourier:
            self._parts.append("fourier")
            self.dim += _FOURIER_K
        self._parts.append("extra")
        self.dim += 4  # contour_count_log, hull_ratio, elongation, solidity_std

    # ------------------------------------------------------------------ util
    @staticmethod
    def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = np.linalg.norm(x)
        return (x / n).astype(np.float32) if n > eps else x.astype(np.float32)

    def _foreground_mask(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Heuristic object mask: Otsu / adaptive on grayscale, largest CC fill."""
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
        h, w = gray.shape[:2]
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Otsu — object often darker or lighter than bg in center ROI
        _, th_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # adaptive fallback
        th_adp = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
        # prefer the mask with more "object-like" fill (5–70% of pixels)
        def fill(m: np.ndarray) -> float:
            return float((m > 0).mean())

        f1, f2 = fill(th_otsu), fill(th_adp)
        mask = th_otsu if 0.05 <= f1 <= 0.70 else (th_adp if 0.05 <= f2 <= 0.70 else th_otsu)
        # morphological close
        k = max(3, min(h, w) // 40)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k | 1, k | 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        # largest connected component
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n > 1:
            # skip background label 0
            areas = stats[1:, cv2.CC_STAT_AREA]
            best = 1 + int(np.argmax(areas))
            mask = np.where(labels == best, 255, 0).astype(np.uint8)
        return mask

    @staticmethod
    def _largest_contour(mask: np.ndarray):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None, 0
        return max(contours, key=cv2.contourArea), len(contours)

    @staticmethod
    def _fourier_desc(contour: np.ndarray, k: int = _FOURIER_K) -> np.ndarray:
        """Rotation/scale-ish invariant Fourier magnitude descriptors (excl. DC)."""
        pts = contour.reshape(-1, 2).astype(np.float64)
        if pts.shape[0] < 8:
            return np.zeros(k, np.float32)
        # resample to fixed length along arc
        N = 128
        idx = np.linspace(0, pts.shape[0] - 1, N).astype(int)
        z = pts[idx, 0] + 1j * pts[idx, 1]
        F = np.fft.fft(z)
        # drop DC, take magnitudes of first k harmonics
        mags = np.abs(F[1 : k + 1])
        if mags[0] > 1e-9:
            mags = mags / mags[0]
        return mags.astype(np.float32)

    # ---------------------------------------------------------------- extract
    def extract(
        self,
        crop_bgr: np.ndarray,
        box: Optional[tuple[int, int, int, int]] = None,
    ) -> Optional[np.ndarray]:
        """Return L2-normalized shape vector (self.dim,) or None if too small/blank."""
        if not self.enabled or crop_bgr is None or crop_bgr.size == 0:
            return None
        if min(crop_bgr.shape[:2]) < self.min_side:
            return None

        h, w = crop_bgr.shape[:2]
        mask = self._foreground_mask(crop_bgr)
        contour, n_contours = self._largest_contour(mask)

        feats: list[np.ndarray] = []

        # --- Hu moments (log-scaled) ---
        hu = np.zeros(7, np.float32)
        hull_ratio = 0.0
        solidity = 0.0
        extent = 0.0
        circularity = 0.0
        elongation = 0.0
        orientation = 0.0
        if contour is not None and cv2.contourArea(contour) > 16:
            m = cv2.moments(contour)
            hu_raw = cv2.HuMoments(m).reshape(-1)
            # log scale, clamp
            hu = (-np.sign(hu_raw) * np.log10(np.abs(hu_raw) + 1e-30)).astype(np.float32)
            hu = np.clip(hu, -_HU_SCALE, _HU_SCALE) / _HU_SCALE

            area = float(cv2.contourArea(contour))
            peri = float(cv2.arcLength(contour, True)) + 1e-9
            hull = cv2.convexHull(contour)
            hull_a = float(cv2.contourArea(hull)) + 1e-9
            x, y, bw, bh = cv2.boundingRect(contour)
            bbox_a = float(bw * bh) + 1e-9
            extent = area / bbox_a
            solidity = area / hull_a
            hull_ratio = hull_a / bbox_a
            circularity = (4.0 * math.pi * area) / (peri * peri)
            circularity = float(np.clip(circularity, 0.0, 1.5))

            # second moments → elongation / orientation
            if m["m00"] > 1e-9:
                mu20 = m["mu20"] / m["m00"]
                mu02 = m["mu02"] / m["m00"]
                mu11 = m["mu11"] / m["m00"]
                tmp = math.sqrt((mu20 - mu02) ** 2 + 4.0 * mu11 * mu11)
                l1 = (mu20 + mu02 + tmp) / 2.0
                l2 = (mu20 + mu02 - tmp) / 2.0
                l2 = max(l2, 1e-9)
                elongation = float(np.clip(math.sqrt(l1 / l2), 1.0, 50.0))
                orientation = float(np.arctan2(2.0 * mu11, mu20 - mu02))

        if self.use_hu:
            feats.append(hu)

        # --- geometry from box / mask ---
        aspect = float(w) / float(h)
        aspect_feat = math.log(max(aspect, 1e-3))  # log-aspect symmetric
        fill = float((mask > 0).mean())
        circ = float(np.clip(circularity, 0.0, 1.5)) / 1.5
        ext = float(np.clip(extent, 0.0, 1.0))
        ori_sin = math.sin(orientation) * 0.5 + 0.5  # keep sign via sin, map later
        feats.append(
            np.array(
                [
                    np.clip(aspect_feat, -4, 4) / 4.0,
                    fill,
                    circ,
                    ext,
                    ori_sin,
                ],
                np.float32,
            )
        )

        if self.use_fourier:
            if contour is not None and cv2.contourArea(contour) > 16:
                feats.append(self._fourier_desc(contour, _FOURIER_K))
            else:
                feats.append(np.zeros(_FOURIER_K, np.float32))

        # --- extras ---
        # aspect from outer box if provided overrides crop aspect slightly
        box_aspect = 1.0
        if box is not None:
            bx1, by1, bx2, by2 = box
            bw, bh = max(1, bx2 - bx1), max(1, by2 - by1)
            box_aspect = bw / bh
        extras = np.array(
            [
                math.log1p(max(n_contours, 0)) / 4.0,  # contour complexity
                float(np.clip(hull_ratio, 0.0, 1.0)),
                float(np.clip(math.log(elongation) / math.log(50.0), 0.0, 1.0)),
                float(np.clip(solidity, 0.0, 1.0)),
            ],
            np.float32,
        )
        # note: box_aspect folded into first geom already via crop; keep extras pure
        _ = box_aspect
        feats.append(extras)

        vec = np.concatenate(feats, axis=0).astype(np.float32)
        if vec.shape[0] != self.dim:
            # pad / truncate defensively
            if vec.shape[0] < self.dim:
                vec = np.pad(vec, (0, self.dim - vec.shape[0]))
            else:
                vec = vec[: self.dim]
        if not np.all(np.isfinite(vec)):
            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        # shape with zero energy → reject (flat / failed mask)
        if float(np.linalg.norm(vec)) < 1e-6:
            return None
        return self.l2_normalize(vec)

    def timed_extract(
        self, crop_bgr: np.ndarray, box: Optional[tuple[int, int, int, int]] = None
    ) -> tuple[Optional[np.ndarray], float]:
        import time as _t

        t0 = _t.perf_counter()
        v = self.extract(crop_bgr, box)
        return v, (_t.perf_counter() - t0) * 1000.0
