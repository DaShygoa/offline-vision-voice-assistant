"""MobileSAM mask refinement for depth-guided zero-shot localization.

Takes the depth-derived nearest-object region, uses its centroid as a point
prompt to MobileSAM, and returns a pixel-precise mask → tight bbox.

MobileSAM (Zhang et al., arXiv 2306.14289) replaces SAM's ViT-H encoder (632M
params) with TinyViT (5M params) while keeping SAM's mask decoder — 38× faster,
near-equivalent mask quality, CPU-deployable.

Architecture (ONNX Runtime CPU, no PyTorch at runtime):
  image ──► [encoder.onnx: TinyViT] ──► image_embedding (1,256,64,64)
  image_embedding + point_coords + point_labels
          ──► [decoder.onnx: prompt encoder + mask decoder] ──► mask (1,256,256)
  mask ──► threshold + largest component ──► tight bbox

Model prep (one-time, needs torch once):
    python tools/prepare_models.py --mobilesam

References:
  - MobileSAM (arXiv 2306.14289)
  - Segment Anything (Kirillov et al., ICCV 2023, arXiv 2304.02643)
  - DepSeg: depth-guided point prompts (arXiv 2512.05529, 2025)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

# SAM-family preprocessing: 1024×1024, ImageNet stats, pixel-normalized
_IMG_SIZE = 1024
_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


@dataclass
class MaskResult:
    mask: np.ndarray                  # (H, W) uint8, 1 = object
    box: Tuple[int, int, int, int]    # tight bbox from mask
    score: float                      # predicted IoU from decoder (if available)
    latency_ms: float


class MobileSAMRefiner:
    """Point-prompted mask refinement using MobileSAM ONNX (encoder + decoder)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.encoder_path = cfg.get("encoder_path", "models/mobile_sam/mobilesam_encoder.onnx")
        self.decoder_path = cfg.get("decoder_path", "models/mobile_sam/mobilesam_decoder.onnx")
        self.mask_threshold = float(cfg.get("mask_threshold", 0.5))
        self.min_mask_area_frac = float(cfg.get("min_mask_area_frac", 0.002))
        self.ready = False
        self.device_used = "none"
        self._enc_sess = None
        self._dec_sess = None
        self._enc_input = None
        self._dec_inputs = {}

    def load(self) -> None:
        for p in (self.encoder_path, self.decoder_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f"MobileSAM ONNX not found: {p}\n"
                    "  run: python tools/prepare_models.py --mobilesam"
                )
        import onnxruntime as ort

        # TinyViT encoder (1024²) is thread-sensitive: measured 8 threads fastest
        # (940ms vs 1477ms@4, 1464ms@16 — oversubscription hurts).
        opts = ort.SessionOptions()
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        opts.intra_op_num_threads = int(self.cfg.get("onnx_threads", 8))
        opts.inter_op_num_threads = 1

        self._enc_sess = ort.InferenceSession(self.encoder_path, opts, providers=["CPUExecutionProvider"])
        self._dec_sess = ort.InferenceSession(self.decoder_path, opts, providers=["CPUExecutionProvider"])

        self._enc_input = self._enc_sess.get_inputs()[0].name
        # map decoder inputs by name (order varies across exports)
        self._dec_inputs = {inp.name: inp for inp in self._dec_sess.get_inputs()}

        self.device_used = "ONNXRuntime:CPU"
        self.ready = True
        log.info("MobileSAM loaded (encoder: %s, decoder: %s)", self.encoder_path, self.decoder_path)

    # ---------------------------------------------------------- preprocess
    def _preprocess_image(self, frame_bgr: np.ndarray) -> np.ndarray:
        """BGR frame → (1,3,1024,1024) float32 blob (SAM normalization)."""
        h, w = frame_bgr.shape[:2]
        # resize longest side to 1024 preserving aspect, then pad to 1024×1024
        scale = _IMG_SIZE / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((_IMG_SIZE, _IMG_SIZE, 3), 0, dtype=np.uint8)
        canvas[:nh, :nw] = resized

        img = canvas.astype(np.float32) - _MEAN
        img = img / _STD
        blob = img.transpose(2, 0, 1)[None]
        return np.ascontiguousarray(blob, dtype=np.float32), (nh, nw, scale)

    # ------------------------------------------------------------- encode
    def _encode(self, blob: np.ndarray) -> np.ndarray:
        emb = self._enc_sess.run(None, {self._enc_input: blob})[0]
        return emb

    # ------------------------------------------------------------- decode
    def _decode(self, emb: np.ndarray, point_xy: Tuple[float, float],
                frame_shape: Tuple[int, int]) -> Optional[MaskResult]:
        """Run decoder with a single positive point prompt. Returns MaskResult or None."""
        t0 = time.perf_counter()
        H, W = frame_shape

        # point in 1024-space
        nh, nw, scale = self._last_resize_info if hasattr(self, "_last_resize_info") else (H, W, 1.0)
        px = point_xy[0] * scale
        py = point_xy[1] * scale

        point_coords = np.array([[[px, py]]], dtype=np.float32)   # (1, 1, 2)
        point_labels = np.array([[1]], dtype=np.float32)          # (1, 1) = foreground

        # build decoder feed based on available input names
        feed = {}
        for name, inp in self._dec_inputs.items():
            lname = name.lower()
            if "embed" in lname:
                feed[name] = emb.astype(np.float32)
            elif "coord" in lname:
                feed[name] = point_coords
            elif "label" in lname:
                feed[name] = point_labels
            elif "size" in lname or "orig" in lname:
                # orig_im_size: (H, W) of ORIGINAL image, int32 or float32
                if inp.type == "tensor(int32)":
                    feed[name] = np.array([[H, W]], dtype=np.int32)
                else:
                    feed[name] = np.array([[H, W]], dtype=np.float32)

        if not feed:
            log.error("MobileSAM decoder: no recognized inputs (%s)", list(self._dec_inputs))
            return None

        try:
            outputs = self._dec_sess.run(None, feed)
        except Exception as e:
            log.error("MobileSAM decoder failed: %s", e)
            return None

        # find mask output: typically (1, N, 256, 256) or (1, 1, H, W)
        masks = None
        scores = None
        for out in outputs:
            arr = np.asarray(out)
            if arr.ndim >= 3 and arr.shape[-1] > 16:
                masks = arr
            elif arr.ndim <= 2 or (arr.ndim == 3 and arr.shape[-1] <= 16):
                scores = arr

        if masks is None:
            log.error("MobileSAM decoder: no mask output found")
            return None

        # take first (best) mask
        mask = masks[0]
        if mask.ndim == 3:
            mask = mask[0]  # (N, H, W) → (H, W)  or (1, H, W) → (H, W)
        score = float(scores.flat[0]) if scores is not None else 0.0

        # decoder outputs LOGITS — must sigmoid before thresholding
        prob = 1.0 / (1.0 + np.exp(-np.clip(mask, -30, 30)))
        binary = (prob > self.mask_threshold).astype(np.uint8)

        # resize mask to original frame size if needed
        if binary.shape != (H, W):
            binary = cv2.resize(binary, (W, H), interpolation=cv2.INTER_NEAREST)

        # keep largest connected component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            return None
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        binary = (labels == largest).astype(np.uint8)

        area_frac = binary.sum() / float(H * W)
        if area_frac < self.min_mask_area_frac:
            return None

        # tight bbox
        ys, xs = np.where(binary > 0)
        if len(xs) == 0:
            return None
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())

        dt = (time.perf_counter() - t0) * 1000.0
        return MaskResult(mask=binary, box=(x1, y1, x2, y2), score=score, latency_ms=dt)

    # ------------------------------------------------------------- public
    def encode_image(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Run the image encoder once (the expensive part) for reuse across prompts."""
        if not self.ready or frame_bgr is None:
            return None
        blob, resize_info = self._preprocess_image(frame_bgr)
        self._last_resize_info = resize_info
        try:
            return self._encode(blob)
        except Exception as e:
            log.error("MobileSAM encode failed: %s", e)
            return None

    def refine_with_point(self, frame_bgr: np.ndarray,
                          point_xy: Tuple[float, float],
                          embedding: Optional[np.ndarray] = None) -> Optional[MaskResult]:
        """Point prompt → mask → tight bbox.

        Pass a pre-computed `embedding` to skip re-encoding when trying several
        prompt points on the same image.
        """
        if not self.ready or frame_bgr is None:
            return None

        t0 = time.perf_counter()
        h, w = frame_bgr.shape[:2]

        if embedding is None:
            blob, resize_info = self._preprocess_image(frame_bgr)
            self._last_resize_info = resize_info
            embedding = self._encode(blob)
            if embedding is None:
                return None

        result = self._decode(embedding, point_xy, (h, w))
        if result is not None:
            result.latency_ms = (time.perf_counter() - t0) * 1000.0
        return result

    def refine_region(self, frame_bgr: np.ndarray,
                      region_box: Tuple[int, int, int, int],
                      depth: Optional[np.ndarray] = None,
                      mask: Optional[np.ndarray] = None) -> Optional[MaskResult]:
        """Refine a region into a precise mask.

        Point prompt selection matters a lot: the bbox centre of a loose depth
        region often lands on background. Prefer the nearest (min-depth) pixel
        inside the object mask — that is on the object itself — and fall back to
        the mask centroid, then the bbox centre.

        The image is encoded ONCE and shared across candidate prompt points.
        """
        x1, y1, x2, y2 = region_box
        pts: List[Tuple[float, float]] = []

        if mask is not None and depth is not None:
            ys, xs = np.where(mask > 0)
            if len(xs) > 0:
                dvals = depth[ys, xs]
                idx = int(np.argmin(dvals))
                pts.append((float(xs[idx]), float(ys[idx])))
                pts.append((float(xs.mean()), float(ys.mean())))

        pts.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

        emb = self.encode_image(frame_bgr)
        if emb is None:
            return None

        best: Optional[MaskResult] = None
        for px, py in pts:
            r = self.refine_with_point(frame_bgr, (px, py), embedding=emb)
            if r is not None and (best is None or r.score > best.score):
                best = r
        return best
