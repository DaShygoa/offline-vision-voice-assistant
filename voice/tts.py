"""TTS client — talks to an INDEPENDENT GPT-SoVITS server process.

Design rule (spec §7 / deployment §7):
  The vision / UI thread must NEVER wait for TTS.
  call() is only invoked from a dedicated TTS worker thread.
  If the server is down or slow, we degrade to mock/skip without blocking vision.

Modes:
  gpt_sovits : HTTP POST to local GPT-SoVITS-style server (voice/tts_server.py)
  mock       : offline beep / silent (for tests without GPT-SoVITS installed)
"""

from __future__ import annotations

import logging
import os
import queue
import tempfile
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class TtsClient:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.mode = str(cfg.get("mode", "gpt_sovits")).lower()
        self.server_url = str(cfg.get("server_url", "http://127.0.0.1:19880")).rstrip("/")
        self.timeout = float(cfg.get("timeout_s", 20))
        self.play_audio = bool(cfg.get("play_audio", True))
        self.enabled = bool(cfg.get("enabled", True))
        self._q: "queue.Queue[str]" = queue.Queue(maxsize=16)
        self._worker: Optional[threading.Thread] = None
        self._running = False
        self._sd = None
        self.ready = False
        self.last_rtf = 0.0
        self.stats = {"requests": 0, "ok": 0, "fail": 0, "avg_synth_ms": 0.0}

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if not self.enabled:
            self.ready = True
            return
        try:
            import sounddevice as sd  # noqa: F401

            self._sd = sd
        except Exception as e:
            log.warning("sounddevice unavailable (%s) — speech will be file-only", e)
        if self.mode == "mock":
            self._ensure_mock_wav()
        self._running = True
        self._worker = threading.Thread(target=self._loop, name="tts-worker", daemon=True)
        self._worker.start()
        self.ready = True
        log.info("TTS worker started mode=%s", self.mode)

    def stop(self) -> None:
        self._running = False
        # unblock queue
        try:
            self._q.put_nowait("")
        except Exception:
            pass
        if self._worker:
            self._worker.join(timeout=3.0)
            self._worker = None

    # ------------------------------------------------------------- public
    def say_async(self, text: str) -> bool:
        """Non-blocking enqueue. Vision/UI never waits here."""
        if not self.enabled or not text:
            return False
        try:
            # drop oldest if congested (fresh reply is more relevant)
            if self._q.full():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
            self._q.put_nowait(text)
            return True
        except Exception as e:
            log.warning("tts enqueue failed: %s", e)
            return False

    def speak_blocking(self, text: str, max_wait: Optional[float] = None) -> bool:
        """Only for scripts/benchmarks — never call from vision thread."""
        if not self.say_async(text):
            return False
        deadline = time.time() + (max_wait if max_wait is not None else self.timeout + 5)
        # wait until queue drains roughly
        while time.time() < deadline:
            if self._q.empty():
                time.sleep(0.05)
                if self._q.empty():
                    return True
            else:
                time.sleep(0.05)
        return False

    # ------------------------------------------------------------- worker
    def _loop(self) -> None:
        while self._running:
            try:
                text = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if not text:
                continue
            try:
                wav_path = self.synthesize(text)
                if wav_path and self.play_audio:
                    self._play(wav_path)
            except Exception as e:
                self.stats["fail"] += 1
                log.warning("TTS failed: %s", e)
            finally:
                self._q.task_done()

    def synthesize(self, text: str) -> Optional[str]:
        self.stats["requests"] += 1
        t0 = time.perf_counter()
        try:
            if self.mode == "mock":
                path = self._mock_synth(text)
            else:
                path = self._gpt_sovits_synth(text)
            dt = (time.perf_counter() - t0) * 1000.0
            audio_dur = self._wav_duration(path) if path else 0.0
            if audio_dur > 0:
                self.last_rtf = (dt / 1000.0) / audio_dur
            n = self.stats["ok"]
            self.stats["avg_synth_ms"] = (self.stats["avg_synth_ms"] * n + dt) / (n + 1)
            self.stats["ok"] += 1
            return path
        except Exception as e:
            self.stats["fail"] += 1
            log.warning("synth error: %s", e)
            return None

    def _gpt_sovits_synth(self, text: str) -> Optional[str]:
        """POST {text} to local server; expects WAV bytes or JSON {\"path\": ...}."""
        payload = {
            "text": text,
            "ref_audio": self.cfg.get("ref_audio", ""),
            "prompt_text": self.cfg.get("prompt_text", self.cfg.get("ref_text", "")),
            "prompt_lang": self.cfg.get("prompt_lang", "zh"),
            "text_lang": "zh",
        }
        import json

        req = urllib.request.Request(
            f"{self.server_url}/tts",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = resp.read()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise RuntimeError(f"GPT-SoVITS server unreachable at {self.server_url}: {e}") from e

        content_type = resp.headers.get("Content-Type", "") if hasattr(resp, "headers") else ""
        out = os.path.join(tempfile.gettempdir(), f"tts_out_{int(time.time()*1000)}.wav")
        if "application/json" in content_type:
            meta = json.loads(data.decode("utf-8"))
            src = meta.get("path") or meta.get("audio")
            if src and os.path.isfile(src):
                return src
            if src and str(src).startswith("data:"):
                import base64

                b64 = str(src).split(",", 1)[-1]
                with open(out, "wb") as f:
                    f.write(base64.b64decode(b64))
                return out
            raise RuntimeError(f"Bad TTS JSON response: {meta}")
        with open(out, "wb") as f:
            f.write(data)
        return out

    def _ensure_mock_wav(self) -> None:
        path = self.cfg.get("mock_wav", "data/tts_beep.wav")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if os.path.isfile(path):
            return
        sr = 16000
        t = np.linspace(0, 0.15, int(sr * 0.15), endpoint=False)
        tone = (0.2 * np.sin(2 * np.pi * 660 * t) * (1 - t / 0.15)).astype(np.float32)
        self._write_wav(path, tone, sr)

    def _mock_synth(self, text: str) -> str:
        # duration ~ speech rate 4.5 chars/s, clamp 0.3..4s
        dur = max(0.3, min(4.0, len(text) / 4.5))
        sr = 16000
        n = int(sr * dur)
        t = np.linspace(0, dur, n, endpoint=False)
        # soft square-ish beep envelope pattern so duration is audible
        env = 0.15 * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
        audio = (env * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        path = self.cfg.get("mock_wav", "data/tts_beep.wav")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._write_wav(path, audio, sr)
        return path

    @staticmethod
    def _write_wav(path: str, audio: np.ndarray, sr: int) -> None:
        try:
            import soundfile as sf

            sf.write(path, audio, sr)
        except Exception:
            import wave

            pcm = np.clip(audio, -1, 1)
            pcm = (pcm * 32767).astype(np.int16)
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes(pcm.tobytes())

    @staticmethod
    def _wav_duration(path: str) -> float:
        try:
            import soundfile as sf

            info = sf.info(path)
            return float(info.frames) / float(info.samplerate or 1)
        except Exception:
            try:
                import wave

                with wave.open(path, "rb") as w:
                    return w.getnframes() / float(w.getframerate())
            except Exception:
                return 0.0

    def _play(self, path: str) -> None:
        if self._sd is None:
            try:
                import sounddevice as sd
                import soundfile as sf

                data, sr = sf.read(path, dtype="float32", always_2d=False)
                sd.play(data, sr)
                sd.wait()
            except Exception as e:
                log.debug("audio playback failed: %s", e)
            return
        try:
            import soundfile as sf

            data, sr = sf.read(path, dtype="float32", always_2d=False)
            self._sd.play(data, sr)
            self._sd.wait()
        except Exception as e:
            log.debug("audio playback failed: %s", e)

    def health(self) -> str:
        if not self.enabled:
            return "disabled"
        if self.mode == "mock":
            return "mock"
        try:
            req = urllib.request.Request(f"{self.server_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                return f"ok:{resp.status}"
        except Exception:
            return "down"
