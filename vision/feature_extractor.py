"""ImageNet-pretrained ResNet18 visual feature extractor (选题一).

Backend (CPU only, NO PyTorch, NO OpenVINO):
  * ONNX Runtime CPU (primary and only)

Feature = avgpool output (512-d, before fc), L2-normalized for cosine retrieval.
The model only runs on identify / remember / search — never in the camera hot path.

Model prep (one-time, needs torch once):
    python tools/prepare_models.py --resnet18
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional, Sequence

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ImageNet stats (torchvision ResNet preprocessing)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ResNet18Extractor:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.backend = str(cfg.get("backend", "onnxruntime")).lower()
        self.device_pref = str(cfg.get("device", "GPU")).upper()
        self.precision = str(cfg.get("precision", "FP16")).upper()
        self.input_size = int(cfg.get("input_size", 224))
        self.embed_dim = int(cfg.get("embed_dim", 512))
        self.num_threads = int(cfg.get("num_threads", 4))
        self.device_used = "none"
        self.ready = False
        self._impl = None

    def load(self) -> None:
        path = self.cfg.get("onnx_path") or self.cfg.get("model_path", "")
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(
                "resnet18.onnx_path not configured or file missing — run: python tools/prepare_models.py --resnet18"
            )
        self._load_ort(path)
        self.ready = True
        return

    def _load_ort(self, path: str) -> None:
        import onnxruntime as ort

        # Without explicit options ORT uses every core and busy-waits between
        # runs, which kept several CPU cores hot even while recognition was
        # paused. Pin the thread count and disable spin-waiting.
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, int(self.num_threads))
        so.inter_op_num_threads = 1
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        so.add_session_config_entry("session.inter_op.allow_spinning", "0")

        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self._impl = ("onnxruntime", sess, sess.get_inputs()[0].name, None)
        self.device_used = "ONNXRuntime:CPU"
        self.backend = "onnxruntime"
        log.info("ResNet18 loaded via ONNX Runtime CPU (%s, threads=%d)", path, self.num_threads)

    # ------------------------------------------------------------------ util
    def preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        s = self.input_size
        img = cv2.resize(crop_bgr, (s, s), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        img = img.transpose(2, 0, 1)[None]  # NCHW
        return np.ascontiguousarray(img, dtype=np.float32)

    @staticmethod
    def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
        n = np.linalg.norm(x, axis=axis, keepdims=True)
        return x / np.maximum(n, eps)

    # ---------------------------------------------------------------- infer
    def embed(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Return L2-normalized 512-d feature or None."""
        if not self.ready or crop_bgr is None or crop_bgr.size == 0:
            return None
        if min(crop_bgr.shape[:2]) < 16:
            return None
        blob = self.preprocess(crop_bgr)
        try:
            sess, inp_name = self._impl[1], self._impl[2]
            feat = sess.run(None, {inp_name: blob})[0]
        except Exception as e:
            log.error("ResNet18 infer failed: %s", e)
            return None
        feat = np.asarray(feat, dtype=np.float32).reshape(-1)
        if feat.shape[0] != self.embed_dim:
            if feat.shape[0] > self.embed_dim:
                feat = feat[: self.embed_dim]
            else:
                log.error("Unexpected ResNet18 dim %d != %d", feat.shape[0], self.embed_dim)
                return None
        return self.l2_normalize(feat)

    def embed_many(self, crops: Sequence[np.ndarray]) -> List[np.ndarray]:
        out = []
        for c in crops:
            e = self.embed(c)
            if e is not None:
                out.append(e)
        return out

    def timed_embed(self, crop_bgr: np.ndarray) -> tuple[Optional[np.ndarray], float]:
        t0 = time.perf_counter()
        e = self.embed(crop_bgr)
        return e, (time.perf_counter() - t0) * 1000.0
