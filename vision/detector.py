"""YOLOv8n object detector.

Deployment rules (CPU only, NO OpenVINO, NO CUDA):
  * Primary backend = ONNX Runtime CPU
  * Fallback backend = OpenCV DNN (readNetFromONNX)
  * NEVER assumes CUDA / TensorRT / OpenVINO.

Detection is intended to run every N frames (not every frame) from a worker thread.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

# COCO class names (YOLOv8n default)
COCO_CLASSES = [
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


@dataclass
class Detection:
    xyxy: np.ndarray          # (4,) int or float — x1 y1 x2 y2
    conf: float
    cls_id: int
    label: str

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (float(x1 + x2) * 0.5, float(y1 + y2) * 0.5)

    @property
    def wh(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (float(x2 - x1), float(y2 - y1))

    def area(self) -> float:
        w, h = self.wh
        return max(0.0, w) * max(0.0, h)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class YoloDetector:
    """Thread-safe-ish: infer() is called from a single worker thread."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.backend = str(cfg.get("backend", "onnxruntime")).lower()
        self.device_pref = str(cfg.get("device", "GPU")).upper()
        self.precision = str(cfg.get("precision", "FP16")).upper()
        self.conf_th = float(cfg.get("conf_threshold", 0.40))
        self.nms_th = float(cfg.get("nms_threshold", 0.45))
        self.input_size = int(cfg.get("input_size", 640))
        self.class_filter: Optional[set[int]] = None
        if cfg.get("class_filter"):
            self.class_filter = set(int(c) for c in cfg["class_filter"])
        self.labels = COCO_CLASSES
        self.device_used = "none"
        self.ready = False
        self._backend_impl = None

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        if self.backend == "onnxruntime":
            try:
                self._load_onnxruntime()
                self.ready = True
                return
            except Exception as e:
                if not self.cfg.get("allow_cpu_fallback", True):
                    raise
                log.warning("ONNX Runtime load failed (%s), falling back to OpenCV DNN", e)
                self.backend = "opencv_dnn"

        if self.backend == "opencv_dnn":
            self._load_opencv_dnn()
            self.ready = True
            return

        raise RuntimeError(f"Unknown YOLO backend: {self.backend}")

    def _resolve_model_path(self) -> str:
        path = self.cfg.get("onnx_path") or self.cfg.get("model_path", "")
        if path and os.path.isfile(path):
            return path
        raise FileNotFoundError(
            f"YOLO ONNX model not found: {path}. Run: python tools/prepare_models.py --yolo"
        )

    def _load_onnxruntime(self) -> None:
        import onnxruntime as ort

        path = self._resolve_model_path()
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self._backend_impl = ("onnxruntime", sess, sess.get_inputs()[0].name)
        self.device_used = "ONNXRuntime:CPU"
        self.backend = "onnxruntime"
        log.info("YOLO loaded via ONNX Runtime CPU (%s)", path)

    def _load_opencv_dnn(self) -> None:
        onnx = self.cfg.get("onnx_path") or self.cfg.get("model_path", "")
        if not os.path.isfile(onnx):
            raise FileNotFoundError(f"YOLO ONNX not found: {onnx}")
        net = cv2.dnn.readNetFromONNX(onnx)
        # Prefer OpenCV's OpenVINO / TIMVX / CPU backends when build supports them.
        preferred = []
        for be in ("DNN_BACKEND_OPENVINO", "DNN_BACKEND_CANN", "DNN_BACKEND_TIMVX", "DNN_BACKEND_DEFAULT"):
            if hasattr(cv2.dnn, be):
                preferred.append(getattr(cv2.dnn, be))
        for be in preferred:
            try:
                if be == getattr(cv2.dnn, "DNN_BACKEND_OPENVINO", None):
                    target = cv2.dnn.DNN_TARGET_OPENVINO if hasattr(cv2.dnn, "DNN_TARGET_OPENVINO") else cv2.dnn.DNN_TARGET_CPU
                else:
                    target = cv2.dnn.DNN_TARGET_CPU
                net.setPreferableBackend(be)
                net.setPreferableTarget(target)
                # probe
                blob = cv2.dnn.blobFromImage(np.zeros((64, 64, 3), np.uint8), 1 / 255.0, (self.input_size, self.input_size), swapRB=True, crop=False)
                net.forward(net.getUnconnectedOutLayersNames())
                net.forward(blob)
                self.device_used = f"OpenCV-DNN:backend={be},target={target}"
                self._backend_impl = ("opencv_dnn", net)
                log.info("YOLO loaded via OpenCV DNN (%s)", self.device_used)
                return
            except Exception:
                continue
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.device_used = "OpenCV-DNN:CPU"
        self._backend_impl = ("opencv_dnn", net)
        log.info("YOLO loaded via OpenCV DNN CPU")

    # ---------------------------------------------------------------- infer
    def infer(self, frame_bgr: np.ndarray) -> tuple[List[Detection], float]:
        if not self.ready:
            return [], 0.0
        t0 = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        s = self.input_size
        img = cv2.resize(frame_bgr, (s, s), interpolation=cv2.INTER_LINEAR)
        blob = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        blob = np.ascontiguousarray(blob, dtype=np.float32)[None] / 255.0

        kind, impl = self._backend_impl[0], self._backend_impl[1]
        if kind == "onnxruntime":
            inp_name = self._backend_impl[2]
            pred = impl.run(None, {inp_name: blob})[0]  # (1, 84, 8400)
        else:
            impl.setInput(blob)
            names = impl.getUnconnectedOutLayersNames()
            outs = impl.forward(names)
            pred = outs[0] if len(outs) == 1 else np.concatenate(outs, axis=0)

        detections = self._postprocess(pred, w, h, s)
        dt = (time.perf_counter() - t0) * 1000.0
        return detections, dt

    def _postprocess(self, pred: np.ndarray, frame_w: int, frame_h: int, net_s: int) -> List[Detection]:
        p = np.asarray(pred)
        if p.ndim == 3:
            p = p[0]
        # YOLOv8 export layout: (84, 8400) = (C, N) → convert to (N, C)
        if p.shape[0] < p.shape[1] and p.shape[0] <= 512:
            p = p.T  # (N, C)
        if p.shape[1] < 5:
            return []
        boxes_cxcywh = p[:, :4]
        scores_all = p[:, 4:]           # (N, nc)
        cls_ids = np.argmax(scores_all, axis=1)
        confs = scores_all[np.arange(len(cls_ids)), cls_ids]

        mask = confs >= self.conf_th
        if self.class_filter is not None:
            mask &= np.isin(cls_ids, list(self.class_filter))
        if not mask.any():
            return []
        boxes_cxcywh = boxes_cxcywh[mask]
        confs = confs[mask]
        cls_ids = cls_ids[mask]

        # absolute pixel xyxy on network input
        cx, cy, bw, bh = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # scale to frame
        sx = frame_w / net_s
        sy = frame_h / net_s
        x1 = np.clip(x1 * sx, 0, frame_w - 1)
        y1 = np.clip(y1 * sy, 0, frame_h - 1)
        x2 = np.clip(x2 * sx, 0, frame_w - 1)
        y2 = np.clip(y2 * sy, 0, frame_h - 1)

        # NMS per-class
        idxs = cv2.dnn.NMSBoxes(
            bboxes=np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist(),
            scores=confs.tolist(),
            score_threshold=self.conf_th,
            nms_threshold=self.nms_th,
        )
        if idxs is None or len(idxs) == 0:
            return []
        idxs = np.array(idxs).reshape(-1)

        out: List[Detection] = []
        for i in idxs:
            c = int(cls_ids[i])
            label = self.labels[c] if 0 <= c < len(self.labels) else str(c)
            out.append(
                Detection(
                    xyxy=np.array([x1[i], y1[i], x2[i], y2[i]], dtype=np.float32),
                    conf=float(confs[i]),
                    cls_id=c,
                    label=label,
                )
            )
        # sort by confidence
        out.sort(key=lambda d: d.conf, reverse=True)
        return out

    # ------------------------------------------------------------------ util
    @staticmethod
    def center_iou_with_box(det: Detection, box: np.ndarray) -> float:
        return _iou(det.xyxy, box)
