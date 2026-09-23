"""Object cropping: YOLO bbox first, else central ROI (MobileSAM is post-MVP)."""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from .detector import Detection


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
            return frame[self.center_roi(frame)[1] : self.center_roi(frame)[3], self.center_roi(frame)[0] : self.center_roi(frame)[2]]
        return crop

    def best_center_detection(
        self,
        dets: List[Detection],
        frame: np.ndarray,
        class_id: Optional[int] = None,
        min_iou: float = 0.15,
    ) -> Optional[Detection]:
        """Pick detection whose box overlaps the central ROI most (optionally filtered by class)."""
        h, w = frame.shape[:2]
        roi = np.array(self.center_roi(frame), dtype=np.float32)
        best: Optional[Detection] = None
        best_score = 0.0
        for d in dets:
            if class_id is not None and d.cls_id != class_id:
                continue
            # IoU of det with central ROI
            x1 = max(float(d.xyxy[0]), float(roi[0]))
            y1 = max(float(d.xyxy[1]), float(roi[1]))
            x2 = min(float(d.xyxy[2]), float(roi[2]))
            y2 = min(float(d.xyxy[3]), float(roi[3]))
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            roi_a = (float(roi[2] - roi[0])) * (float(roi[3] - roi[1]))
            score = inter / roi_a if roi_a > 0 else 0.0
            # prefer detections near center even if partially outside ROI
            cx, cy = d.center
            dist = np.hypot(cx - w / 2, cy - h / 2)
            proximity = 1.0 - min(1.0, dist / (0.5 * np.hypot(w, h)))
            total = 0.6 * score + 0.4 * proximity
            if total > best_score:
                best_score = total
                best = d
        if best is None or best_score < min_iou:
            return None
        return best

    def crop_for_learning(
        self,
        frame: np.ndarray,
        dets: List[Detection],
        generic_class: Optional[str] = None,
    ) -> tuple[np.ndarray, tuple[int, int, int, int], str]:
        """Return (crop, box, source) where source is 'yolo' or 'center_roi'."""
        class_id = None
        if generic_class:
            for i, name in enumerate(
                [
                    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
                    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
                    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
                    "toothbrush",
                ]
            ):
                if name == generic_class:
                    class_id = i
                    break
        det = self.best_center_detection(dets, frame, class_id=class_id)
        if det is None and class_id is None:
            det = self.best_center_detection(dets, frame, class_id=None)
        if det is not None:
            box = expand_box(*det.xyxy, frame.shape[1], frame.shape[0])
            return self.crop_box(frame, box), box, "yolo"
        box = self.center_roi(frame)
        return self.crop_box(frame, box), box, "center_roi"

    def crop_detections(self, frame: np.ndarray, dets: List[Detection]) -> List[tuple[np.ndarray, tuple[int, int, int, int], Detection]]:
        out = []
        h, w = frame.shape[:2]
        for d in dets:
            box = expand_box(*d.xyxy, w, h)
            out.append((self.crop_box(frame, box), box, d))
        return out

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
