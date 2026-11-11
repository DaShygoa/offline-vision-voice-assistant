"""OpenCV camera capture. Runs in its own thread; UI reads latest frame only."""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


class Camera:
    def __init__(self, index: int = 0, width: int = 1920, height: int = 1080, fps: int = 30):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_id = 0
        self._fps_ema = 0.0

    def open(self) -> None:
        backend = getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)
        cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self._cap = cap
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.width
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height

    def start(self) -> None:
        if self._cap is None:
            self.open()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        assert self._cap is not None
        last = time.perf_counter()
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            now = time.perf_counter()
            dt = now - last
            last = now
            if dt > 0:
                inst = 1.0 / dt
                self._fps_ema = inst if self._fps_ema == 0 else 0.9 * self._fps_ema + 0.1 * inst
            with self._lock:
                self._frame = frame
                self._frame_id += 1

    def latest(self) -> tuple[Optional[np.ndarray], int]:
        with self._lock:
            if self._frame is None:
                return None, self._frame_id
            return self._frame.copy(), self._frame_id

    @property
    def capture_fps(self) -> float:
        return self._fps_ema

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        # Drop the last frame so a device switch never serves the old camera's
        # frozen picture to the MJPEG stream / depth worker.
        with self._lock:
            self._frame = None
            self._frame_id += 1

    def reopen(self, index: int) -> None:
        """Switch to another device index, keeping the running state.

        On failure the old index is restored (and reopened if it was running)
        before the exception is re-raised, so the caller can simply report the
        error instead of repairing the capture pipeline.
        """
        was_running = self._running
        old_index = self.index
        self.stop()
        self.index = int(index)
        try:
            if was_running:
                self.start()
        except Exception:
            self.index = old_index
            if was_running:
                try:
                    self.start()
                except Exception as restore_err:
                    log.warning("failed to restore camera %s: %s", old_index, restore_err)
            raise

    def __enter__(self) -> "Camera":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()
