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
    def __init__(self, index: int = 0, width: int = 0, height: int = 0, fps: int = 30):
        self.index = index
        self.requested_width = int(width or 0)
        self.requested_height = int(height or 0)
        self.width = 0
        self.height = 0
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
            cap.release()
            cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open camera index {self.index}")
        if self.requested_width > 0 and self.requested_height > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self._cap = cap
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        log.info("Camera %s opened; reported resolution %sx%s", self.index, self.width, self.height)

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
                self.height, self.width = frame.shape[:2]
                self._frame_id += 1

    def latest(self) -> tuple[Optional[np.ndarray], int]:
        with self._lock:
            if self._frame is None:
                return None, self._frame_id
            return self._frame.copy(), self._frame_id

    @property
    def capture_fps(self) -> float:
        return self._fps_ema

    @property
    def frame_size(self) -> tuple[int, int]:
        """Actual dimensions of the latest frame; (0, 0) before first frame."""
        with self._lock:
            return (self._frame.shape[1], self._frame.shape[0]) if self._frame is not None else (0, 0)

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
