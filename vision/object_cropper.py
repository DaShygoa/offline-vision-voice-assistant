"""Object cropping for depth-guided zero-shot localization.

The classroom release keeps only the central-ROI fallback and quality gating.
The YOLO box path was removed: Depth Anything V2 is the sole locator, so the
identity pipeline contains no pretrained category knowledge at all.
"""

from __future__ import annotations

import cv2
import numpy as np


def expand_box(x1: float, y1: float, x2: float, y2: float, frame_w: int, frame_h: int, ratio: float = 0.06) -> tuple[int, int, int, int]:
    w = x2 - x1
    h = y2 - y1
    dx = w * ratio
    dy = h * ratio
    nx1 = int(max(0, x1 - dx))
    ny1 = int(max(0, y1 - dy))
    nx2 = int(min(frame_w - 1, x2 + dx))
    ny2 = int(min(frame_h - 1, y2 + dy))
    if nx2 - nx1 < 8 or ny2 - ny1 < 8:
        return int(max(0, x1)), int(max(0, y1)), int(min(frame_w, x2)), int(min(frame_h, y2))
    return nx1, ny1, nx2, ny2


class ObjectCropper:
    def __init__(self, center_roi_frac: float = 0.45, min_crop_px: int = 64):
        self.center_roi_frac = float(center_roi_frac)
        self.min_crop_px = int(min_crop_px)

    def center_roi(self, frame: np.ndarray) -> tuple[int, int, int, int]:
        """Fallback region when depth estimation is unavailable: centre 45%."""
        h, w = frame.shape[:2]
        rw = int(w * self.center_roi_frac)
        rh = int(h * self.center_roi_frac)
        cx, cy = w // 2, h // 2
        x1 = max(0, cx - rw // 2)
        y1 = max(0, cy - rh // 2)
        x2 = min(w, x1 + rw)
        y2 = min(h, y1 + rh)
        return x1, y1, x2, y2

    def crop_box(self, frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = box
        x1 = int(max(0, x1))
        y1 = int(max(0, y1))
        x2 = int(min(frame.shape[1], x2))
        y2 = int(min(frame.shape[0], y2))
        if x2 - x1 < self.min_crop_px or y2 - y1 < self.min_crop_px:
            # pad to min size around center
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            half = self.min_crop_px // 2
            x1, x2 = cx - half, cx + half
            y1, y2 = cy - half, cy + half
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            x1, y1, x2, y2 = self.center_roi(frame)
            return frame[y1:y2, x1:x2]
        return crop

    def quality_ok(self, crop: np.ndarray) -> bool:
        """Reject blank / too-dark / too-flat crops before embedding."""
        if crop is None or crop.size == 0:
            return False
        if min(crop.shape[:2]) < self.min_crop_px:
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        mean = float(gray.mean())
        std = float(gray.std())
        if mean < 12 or mean > 245:
            return False
        if std < 8:
            return False
        return True
