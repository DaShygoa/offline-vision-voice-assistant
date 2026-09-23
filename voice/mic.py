"""Microphone capture → float32 mono PCM @ sample_rate.

Runs its own thread; consumers read from an internal queue (never blocks vision).
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class MicStream:
    def __init__(self, sample_rate: int = 16000, frame_ms: int = 30, channels: int = 1, max_queue_s: float = 5.0):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.channels = channels
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=int(max_queue_s * 1000 / frame_ms))
        self._stream = None
        self._running = False
        self.device_name = ""
        self.available = False

    def list_devices(self):
        try:
            import sounddevice as sd

            return sd.query_devices()
        except Exception as e:
            log.warning("Cannot list audio devices: %s", e)
            return []

    def start(self, device: Optional[int | str] = None) -> bool:
        try:
            import sounddevice as sd
        except Exception as e:
            log.error("sounddevice not installed: %s", e)
            self.available = False
            return False

        def callback(indata, frames, time_info, status):
            if status:
                log.debug("mic status: %s", status)
            mono = indata.mean(axis=1) if indata.ndim > 1 else indata
            data = np.ascontiguousarray(mono, dtype=np.float32)
            if self._q.full():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._q.put_nowait(data)
            except queue.Full:
                pass

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=self.frame_samples,
                device=device,
                callback=callback,
            )
            self._stream.start()
            info = sd.query_devices(device) if device is not None else sd.query_devices(kind="input")
            self.device_name = str(info.get("name", "")) if isinstance(info, dict) else ""
            self._running = True
            self.available = True
            log.info("Mic started: %s @ %d Hz", self.device_name or "default", self.sample_rate)
            return True
        except Exception as e:
            log.error("Mic open failed: %s", e)
            self.available = False
            return False

    def read(self, timeout: float = 0.2) -> Optional[np.ndarray]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
