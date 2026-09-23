"""Legacy monocular depth feature extractor — NOT wired in 选题一 (ResNet18 only).

Kept for reference; core/app.py no longer imports this module.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ImageNet-style normalize is NOT used; MiDaS small expects RGB /255 with
# optional mean/std — we follow the common MiDaS v2.1 small preprocessing.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_HIST_BINS = 8
_STAT_DIM = 6  # mean, median, std, grad_x, grad_y, range
_DIM = _STAT_DIM + _HIST_BINS + 4  # + contrast mean/std/box_near/surround_near


class DepthExtractor:
    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", True))
        self.backend = str(cfg.get("backend", "openvino")).lower()
        self.device_pref = str(cfg.get("device", "GPU")).upper()
        self.precision = str(cfg.get("precision", "FP16")).upper()
        self.input_size = int(cfg.get("input_size", 256))
        self.allow_cpu_fallback = bool(cfg.get("allow_cpu_fallback", True))
        self.dim = _DIM
        self.device_used = "none"
        self.ready = False
        self._impl = None
        self._error = ""

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        if not self.enabled:
            self.ready = False
            return
        path = str(self.cfg.get("model_path", "") or "")
        if not path or not os.path.isfile(path):
            self._error = f"depth model not found: {path or '(unset)'}"
            log.warning("Depth disabled: %s", self._error)
            self.ready = False
            return
        if self.backend == "openvino":
            try:
                self._load_openvino(path)
                self.ready = True
                return
            except Exception as e:
                if not self.allow_cpu_fallback:
                    raise
                log.warning("Depth OpenVINO GPU failed (%s), CPU fallback", e)
                try:
                    self.device_pref = "CPU"
                    self._load_openvino(path)
                    self.ready = True
                    return
                except Exception as e2:
                    log.warning("Depth OpenVINO failed (%s), ONNX Runtime fallback", e2)
                    self.backend = "onnxruntime"
        if self.backend == "onnxruntime":
            self._load_ort(path)
            self.ready = True
            return
        raise RuntimeError(f"Unknown depth backend: {self.backend}")

    def _load_openvino(self, path: str) -> None:
        import openvino as ov

        core = ov.Core()
        available = list(core.available_devices)
        model = core.read_model(path)
        device = "CPU"
        if self.device_pref in ("GPU", "AUTO") and any(
            d == "GPU" or d.startswith("GPU.") for d in available
        ):
            device = "GPU"
        config = {"PERFORMANCE_HINT": "LATENCY"}
        try:
            compiled = core.compile_model(model, device, config)
        except Exception as e:
            if device != "CPU":
                log.warning("Depth compile GPU failed (%s), CPU", e)
                device = "CPU"
                compiled = core.compile_model(model, "CPU", config)
            else:
                raise
        self._impl = ("openvino", compiled, compiled.input(0), compiled.outputs[0])
        self.device_used = f"OpenVINO:{device}:{self.precision}"
        self.backend = "openvino"
        log.info("Depth model loaded on %s (%s)", device, path)

    def _load_ort(self, path: str) -> None:
        import onnxruntime as ort

        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self._impl = ("onnxruntime", sess, sess.get_inputs()[0].name, None)
        self.device_used = "ONNXRuntime:CPU"
        self.backend = "onnxruntime"
        log.info("Depth model loaded via ONNX Runtime (%s)", path)

    # ------------------------------------------------------------ preprocess
    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        s = self.input_size
        img = crop_bgr
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        img = np.ascontiguousarray(img, dtype=np.uint8)
        img = cv2.resize(img, (s, s), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        img = img.transpose(2, 0, 1)[None]
        return np.ascontiguousarray(img, dtype=np.float32)

    def _infer_map(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Return raw depth map (H, W) float32 or None."""
        if not self.ready or crop_bgr is None or crop_bgr.size == 0:
            return None
        blob = self._preprocess(crop_bgr)
        kind = self._impl[0]
        try:
            if kind == "openvino":
                compiled, inp, out = self._impl[1], self._impl[2], self._impl[3]
                out_blob = compiled({inp: blob})
                feat = out_blob[out]
            else:
                sess, inp_name = self._impl[1], self._impl[2]
                feat = sess.run(None, {inp_name: blob})[0]
        except Exception as e:
            log.error("Depth infer failed: %s", e)
            return None
        m = np.asarray(feat, dtype=np.float32)
        # squeeze batch / channel
        while m.ndim > 2:
            m = m[0] if m.shape[0] == 1 else m.squeeze(0)
        if m.ndim != 2:
            m = m.reshape(self.input_size, self.input_size)
        return m

    # --------------------------------------------------------------- extract
    @staticmethod
    def _robust_norm(d: np.ndarray) -> np.ndarray:
        """Min-max with percentile clip so outliers don't dominate."""
        d = np.asarray(d, dtype=np.float32)
        if not np.all(np.isfinite(d)):
            d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = np.percentile(d, (2.0, 98.0))
        if hi - lo < 1e-9:
            return np.zeros_like(d, np.float32)
        x = (d - float(lo)) / float(hi - lo)
        return np.clip(x, 0.0, 1.0).astype(np.float32)

    def extract(
        self,
        crop_bgr: np.ndarray,
        frame_bgr: Optional[np.ndarray] = None,
        box: Optional[tuple[int, int, int, int]] = None,
    ) -> Optional[np.ndarray]:
        """Compact depth signature for object crop (optional frame for contrast)."""
        if not self.enabled or not self.ready:
            return None
        if crop_bgr is None or crop_bgr.size == 0 or min(crop_bgr.shape[:2]) < 16:
            return None

        depth = self._infer_map(crop_bgr)
        if depth is None:
            return None
        d = self._robust_norm(depth)

        # spatial stats on object depth (np.gradient — avoids OpenCV AVX2 Sobel bug)
        mean = float(d.mean())
        median = float(np.median(d))
        std = float(d.std())
        if d.ndim == 2 and d.shape[0] >= 3 and d.shape[1] >= 3:
            gy_g, gx_g = np.gradient(d.astype(np.float32))
            grad_x = float(np.mean(np.abs(gx_g)))
            grad_y = float(np.mean(np.abs(gy_g)))
        else:
            grad_x = grad_y = 0.0
        rng = float(d.max() - d.min())

        hist, _ = np.histogram(d, bins=_HIST_BINS, range=(0.0, 1.0), density=True)
        # density → probability
        hs = hist.astype(np.float32)
        if hs.sum() > 0:
            hs = hs / hs.sum()

        # contrast vs surrounding ring on the SAME map (approx) when full map
        # is just the crop: use border band of crop as "surround"
        h, w = d.shape
        bh, bw = max(2, h // 8), max(2, w // 8)
        border = d.copy()
        border[bh : h - bh, bw : w - bw] = np.nan
        border_vals = border[~np.isnan(border)]
        if border_vals.size == 0:
            border_vals = d.reshape(-1)
        contrast_mean = float(mean - float(np.mean(border_vals)))
        contrast_std = float(std - float(np.std(border_vals)))
        box_near = float(median)  # object nearness proxy
        surround_near = float(np.mean(border_vals))

        vec = np.concatenate(
            [
                np.array(
                    [mean, median, std, grad_x, grad_y, rng],
                    np.float32,
                ),
                hs,
                np.array(
                    [contrast_mean, contrast_std, box_near, surround_near],
                    np.float32,
                ),
            ]
        ).astype(np.float32)
        if not np.all(np.isfinite(vec)):
            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        if vec.shape[0] != self.dim:
            if vec.shape[0] < self.dim:
                vec = np.pad(vec, (0, self.dim - vec.shape[0]))
            else:
                vec = vec[: self.dim]
        n = float(np.linalg.norm(vec))
        if n < 1e-6:
            return None
        return (vec / n).astype(np.float32)

    def timed_extract(
        self,
        crop_bgr: np.ndarray,
        frame_bgr: Optional[np.ndarray] = None,
        box: Optional[tuple[int, int, int, int]] = None,
    ) -> tuple[Optional[np.ndarray], float]:
        t0 = time.perf_counter()
        v = self.extract(crop_bgr, frame_bgr, box)
        return v, (time.perf_counter() - t0) * 1000.0
