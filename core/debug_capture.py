"""Debug capture for the identify pipeline.

When enabled, every "这是什么" / identify command dumps a self-contained
session folder so the decision can be inspected offline:

    data/debug/20261026_143519_这是什么/
        frame_annotated.jpg   original frame with all candidate boxes drawn
        frame_raw.jpg         unmodified camera frame
        cand_0_crop.jpg       masked crop fed to ResNet18 (candidate 0)
        cand_0_mask.jpg       binary mask used for background removal
        cand_1_crop.jpg       …
        report.json           structured trace: scores, matches, decision

Only the identify path is captured — remember/search do not dump images.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import List, Optional

import cv2
import numpy as np

from vision.image_io import imwrite_unicode

log = logging.getLogger(__name__)


class DebugCapture:
    def __init__(self, enabled: bool = False, base_dir: str = "data/debug"):
        self.enabled = bool(enabled)
        self.base_dir = base_dir
        self._dir: Optional[str] = None
        self._n = 0
        self._report: dict = {}
        self._t0 = 0.0

    # ---------------------------------------------------------------- session
    def start(self, trigger_text: str, frame: Optional[np.ndarray] = None) -> Optional[str]:
        """Open a new session folder. Returns the folder path (or None if disabled)."""
        if not self.enabled:
            return None
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe = re.sub(r"[^\w一-鿿\-]", "_", trigger_text or "identify")[:24]
        self._dir = os.path.join(self.base_dir, f"{ts}_{safe}")
        os.makedirs(self._dir, exist_ok=True)
        self._n = 0
        self._t0 = time.perf_counter()
        self._report = {
            "trigger": trigger_text,
            "timestamp": ts,
            "dir": self._dir,
            "candidates": [],
            "decision": None,
            "latency_ms": {},
        }
        if frame is not None:
            self._save(frame, "frame_raw.jpg")
        log.info("Debug capture started: %s", self._dir)
        return self._dir

    # ---------------------------------------------------------------- helpers
    def _save(self, img: np.ndarray, name: str) -> None:
        if not self._dir or img is None:
            return
        try:
            # unicode-safe: cv2.imwrite silently fails on non-ASCII paths
            imwrite_unicode(os.path.join(self._dir, name), img)
        except Exception as e:
            log.debug("debug save failed %s: %s", name, e)

    def _mark(self, key: str) -> None:
        """Record elapsed ms for a pipeline stage."""
        if self._dir:
            self._report["latency_ms"][key] = round((time.perf_counter() - self._t0) * 1000, 1)

    # ---------------------------------------------------------------- entries
    def add_candidate(self,
                      crop: Optional[np.ndarray],
                      box: tuple,
                      region_score: float,
                      area_frac: float,
                      source: str,
                      mask: Optional[np.ndarray] = None,
                      match_name: str = "",
                      match_score: float = 0.0,
                      top_matches: Optional[list] = None,
                      is_best: bool = False) -> None:
        """Record one candidate region and its match result."""
        if not self._dir:
            return
        idx = self._n
        self._n += 1

        if crop is not None:
            self._save(crop, f"cand_{idx}_crop.jpg")
        if mask is not None:
            # mask may be frame-sized; store a thumbnail
            m = (mask > 0).astype(np.uint8) * 255
            self._save(cv2.cvtColor(m, cv2.COLOR_GRAY2BGR), f"cand_{idx}_mask.jpg")

        entry = {
            "index": idx,
            "box": [int(v) for v in box],
            "source": source,
            "region_score": round(float(region_score), 4),
            "area_frac": round(float(area_frac), 4),
            "crop_shape": list(crop.shape) if crop is not None else None,
            "match_name": match_name,
            "match_score": round(float(match_score), 4),
            "top_matches": top_matches or [],
            "selected": bool(is_best),
        }
        self._report["candidates"].append(entry)

    def add_stage(self, name: str, **kv) -> None:
        """Record an arbitrary pipeline event (e.g. depth candidate filter)."""
        if not self._dir:
            return
        self._report.setdefault("stages", []).append({"stage": name, **kv})
        self._mark(name)

    def add_frame_annotated(self, frame: np.ndarray, boxes: List[tuple],
                            labels: List[str], best_idx: int = -1) -> None:
        """Save the frame with all candidate boxes drawn (best highlighted)."""
        if not self._dir or frame is None:
            return
        vis = frame.copy()
        for i, (box, label) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = [int(v) for v in box]
            is_best = (i == best_idx)
            color = (80, 220, 80) if is_best else (255, 180, 60)
            thickness = 3 if is_best else 2
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            tag = f"{'BEST ' if is_best else ''}{label}"
            cv2.putText(vis, tag, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        self._save(vis, "frame_annotated.jpg")

    def finish(self, result_text: str = "", decision: str = "",
               asr_text: str = "", intent: str = "") -> Optional[str]:
        """Write report.json and return the session folder path."""
        if not self._dir:
            return None
        self._report["decision"] = decision
        self._report["result_text"] = result_text
        self._report["asr_text"] = asr_text
        self._report["intent"] = intent
        self._report["latency_ms"]["total"] = round((time.perf_counter() - self._t0) * 1000, 1)
        try:
            with open(os.path.join(self._dir, "report.json"), "w", encoding="utf-8") as f:
                json.dump(self._report, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.warning("debug report write failed: %s", e)
        path = self._dir
        self._dir = None
        log.info("Debug capture finished: %s (%d candidates)", path, self._n)
        return path
