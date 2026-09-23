"""Silero VAD via sherpa-onnx (offline, CPU).

API note (sherpa-onnx >= 1.10):
  VoiceActivityDetector.accept_waveform(samples)  — no sample_rate arg
  is_speech_detected() / is_speech_finished() / front / pop / flush
Falls back to energy VAD if unavailable.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class Vad:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sample_rate = int(cfg.get("sample_rate", 16000))
        self.vad_model = cfg.get("vad_model", "")
        self.max_end_silence_ms = float(cfg.get("max_end_silence_ms", 800))
        self.min_speech_ms = float(cfg.get("min_speech_ms", 250))
        self._vad = None
        self.backend = "energy"
        self._energy_threshold = 0.01

    def load(self) -> None:
        try:
            import sherpa_onnx

            if not self.vad_model:
                raise FileNotFoundError("asr.vad_model not set")
            silero = sherpa_onnx.SileroVadModelConfig(
                model=self.vad_model,
                threshold=0.5,
                min_silence_duration=self.max_end_silence_ms / 1000.0,
                min_speech_duration=self.min_speech_ms / 1000.0,
                window_size=512,
            )
            vad_cfg = sherpa_onnx.VadModelConfig(
                silero_vad=silero,
                sample_rate=self.sample_rate,
                num_threads=1,
                provider="cpu",
            )
            self._vad = sherpa_onnx.VoiceActivityDetector(vad_cfg, buffer_size_in_seconds=60.0)
            self.backend = "silero_sherpa"
            log.info("VAD loaded: silero (%s)", self.vad_model)
        except Exception as e:
            log.warning("Sherpa VAD unavailable (%s), energy VAD fallback", e)
            self._vad = None
            self.backend = "energy"
            self._energy_threshold = 0.01

    # ---------------------------------------------------------------- feed
    def feed(self, pcm: np.ndarray) -> bool:
        """Feed chunk (float32/int16 mono). Returns True if speech currently detected."""
        x = np.asarray(pcm)
        if x.dtype != np.float32:
            if x.dtype == np.int16:
                x = x.astype(np.float32) / 32768.0
            else:
                x = x.astype(np.float32)
        if x.size == 0:
            return False

        if self._vad is not None:
            try:
                # sherpa expects Sequence[float]; ndarray works
                self._vad.accept_waveform(x.tolist())
                return self.speech_present
            except Exception as e:
                log.debug("vad feed error: %s", e)
                return False

        rms = float(np.sqrt(np.mean(x**2)))
        return rms > self._energy_threshold

    def is_speech_finished(self) -> bool:
        if self._vad is None:
            return False
        try:
            return bool(self._vad.is_speech_finished())
        except Exception:
            return False

    def pop(self) -> np.ndarray:
        if self._vad is None:
            return np.zeros(0, dtype=np.float32)
        try:
            s = self._vad.front
            self._vad.pop()
            return np.asarray(s.samples, dtype=np.float32)
        except Exception:
            return np.zeros(0, dtype=np.float32)

    def flush(self) -> None:
        if self._vad is not None:
            try:
                self._vad.flush()
            except Exception:
                pass

    @property
    def speech_present(self) -> bool:
        if self._vad is None:
            return False
        try:
            if hasattr(self._vad, "is_speech_detected"):
                return bool(self._vad.is_speech_detected())
            if hasattr(self._vad, "is_detected"):
                return bool(self._vad.is_detected())
            return False
        except Exception:
            return False
