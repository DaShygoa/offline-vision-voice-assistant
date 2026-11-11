"""Asynchronous speech synthesis with an offline Chinese voice.

Design rule: the vision / web thread must NEVER wait for TTS. Synthesis runs
on a dedicated worker thread; if a backend fails we degrade to mock/skip
without blocking recognition.

Modes:
  offline: sherpa-onnx VITS (local model, no network)
  edge  : Microsoft Edge TTS (natural Chinese voice; needs internet)
  mock  : offline beep whose length tracks the reply (no network, for fallback)

On an Edge error, use the local voice instead of a mock beep.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class TtsClient:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.mode = str(cfg.get("mode", "offline")).lower()
        self.timeout = float(cfg.get("timeout_s", 20))
        self.play_audio = bool(cfg.get("play_audio", True))
        self.enabled = bool(cfg.get("enabled", True))
        self._q: "queue.Queue[str]" = queue.Queue(maxsize=16)
        self._worker: Optional[threading.Thread] = None
        self._running = False
        self._sd = None
        self.ready = False
        self._offline_tts = None
        self._offline_error = None
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
            wav_path = None
            try:
                wav_path = self.synthesize(text)
                if wav_path and self.play_audio:
                    self._play(wav_path)
            except Exception as e:
                self.stats["fail"] += 1
                log.warning("TTS failed: %s", e)
            finally:
                if wav_path and self.mode != "mock":
                    try:
                        os.remove(wav_path)
                    except OSError:
                        pass
                self._q.task_done()

    def synthesize(self, text: str) -> Optional[str]:
        self.stats["requests"] += 1
        t0 = time.perf_counter()
        path = None
        try:
            if self.mode == "mock":
                path = self._mock_synth(text)
            elif self.mode == "offline":
                path = self._offline_synth(text)
            elif self.mode == "edge":
                try:
                    path = self._edge_synth(text)
                    if not path:
                        raise RuntimeError("edge TTS returned no audio")
                except Exception as e:
                    log.warning("edge TTS failed (%s); using offline voice", e)
                    path = self._offline_synth(text)
            else:
                raise ValueError(f"unknown TTS mode: {self.mode}")
        except Exception as e:
            self.stats["fail"] += 1
            log.warning("synth error: %s", e)
            return None
        dt = (time.perf_counter() - t0) * 1000.0
        audio_dur = self._wav_duration(path) if path else 0.0
        if audio_dur > 0:
            self.last_rtf = (dt / 1000.0) / audio_dur
        n = self.stats["ok"]
        self.stats["avg_synth_ms"] = (self.stats["avg_synth_ms"] * n + dt) / (n + 1)
        self.stats["ok"] += 1
        return path

    def _load_offline(self):
        if self._offline_tts is not None:
            return self._offline_tts
        import sherpa_onnx

        model_dir = self.cfg.get("offline_model_dir", "models/sherpa-onnx-vits-zh-ll")
        required = ["model.onnx", "tokens.txt", "lexicon.txt", "phone.fst", "date.fst", "number.fst"]
        missing = [name for name in required if not os.path.isfile(os.path.join(model_dir, name))]
        if missing:
            raise FileNotFoundError(f"offline TTS model missing in {model_dir}: {', '.join(missing)}; run python tools/prepare_models.py --tts")
        vits = sherpa_onnx.OfflineTtsVitsModelConfig(
            model=os.path.join(model_dir, "model.onnx"),
            lexicon=os.path.join(model_dir, "lexicon.txt"),
            tokens=os.path.join(model_dir, "tokens.txt"),
        )
        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(vits=vits, num_threads=int(self.cfg.get("offline_threads", 2))),
            rule_fsts=','.join(os.path.join(model_dir, f) for f in ("phone.fst", "date.fst", "number.fst")),
        )
        if not config.validate():
            raise ValueError("invalid offline TTS model configuration")
        self._offline_tts = sherpa_onnx.OfflineTts(config)
        self._offline_error = None
        return self._offline_tts

    def _offline_synth(self, text: str) -> str:
        import tempfile
        try:
            audio = self._load_offline().generate(text, sid=int(self.cfg.get("offline_speaker", 0)),
                                                   speed=float(self.cfg.get("offline_speed", 1.0)))
            samples = np.asarray(audio.samples, dtype=np.float32)
            if not samples.size or not np.isfinite(samples).all() or np.max(np.abs(samples)) < 1e-4:
                raise RuntimeError("offline TTS returned empty or silent audio")
            with tempfile.NamedTemporaryFile(prefix="tts_offline_", suffix=".wav", delete=False) as f:
                path = f.name
            self._write_wav(path, samples, audio.sample_rate)
            return path
        except Exception as e:
            self._offline_error = str(e)
            raise

    # Available zh-CN voices (edge-tts 7.x, verified by list_voices() on this machine)
    EDGE_VOICES = (
        "zh-CN-YunxiNeural",# 男声，年轻，最贴合"语音助手"调性
        "zh-CN-YunjianNeural",    # 男声，播音腔，正式
        "zh-CN-YunyangNeural",    # 男声，解说/新闻腔
        "zh-CN-YunxiaNeural",     # 男声，童声偏成熟
        "zh-CN-XiaoxiaoNeural",   # 女声，通用自然
        "zh-CN-XiaoyiNeural",     # 女声，年轻活泼
        "zh-CN-liaoning-XiaobeiNeural",   # 女声，东北口音
        "zh-CN-shaanxi-XiaoniNeural",     # 女声，陕西口音
    )

    def _edge_synth(self, text: str) -> Optional[str]:
        """Microsoft Edge TTS. Needs network; returns a WAV path.

        edge-tts streams MP3. soundfile/libsndfile>=0.12 handles MP3,
        but we decode to float32 here so downstream _play() is uniform.
        """
        import asyncio
        import tempfile as _tf

        voice = str(self.cfg.get("edge_voice", "zh-CN-YunxiNeural"))
        if voice not in self.EDGE_VOICES:
            log.warning("unknown edge_voice=%s, falling back to Yunxi", voice)
            voice = "zh-CN-YunxiNeural"
        rate = str(self.cfg.get("edge_rate", "+0%"))
        pitch = str(self.cfg.get("edge_pitch", "+0Hz"))

        mp3 = os.path.join(_tf.gettempdir(),
                           f"tts_edge_{int(time.time()*1000)}.mp3")
        wav = mp3[:-4] + ".wav"

        async def _run() -> None:
            import edge_tts

            comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            await comm.save(mp3)

        asyncio.run(_run())

        # MP3 -> float32 WAV for a uniform playback path
        try:
            import soundfile as sf

            data, sr = sf.read(mp3, dtype="float32", always_2d=False)
        except Exception as e:
            raise RuntimeError(f"edge-tts produced unreadable mp3: {e}") from e
        try:
            os.remove(mp3)
        except OSError:
            pass

        if data.ndim > 1:
            data = data.mean(axis=1)
        # Edge occasionally returns near-silent output; surface it instead of
        # silently "succeeding" with an empty wav.
        if float(np.abs(data).max()) < 1e-4:
            raise RuntimeError("edge-tts returned silence (check text/voice)")

        with open(wav, "wb") as f:
            pass
        self._write_wav(wav, data.astype(np.float32), sr)
        return wav

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
        if self.mode == "offline":
            if self._offline_error:
                return "down:offline"
            model_dir = self.cfg.get("offline_model_dir", "models/sherpa-onnx-vits-zh-ll")
            return "ok:offline" if all(os.path.isfile(os.path.join(model_dir, f)) for f in
                ("model.onnx", "tokens.txt", "lexicon.txt", "phone.fst", "date.fst", "number.fst")) else "down:offline-model"
        # edge — no local process to probe; report whether the SDK is importable.
        try:
            import edge_tts  # noqa: F401

            return "ok:edge"
        except Exception as e:
            return f"down:{type(e).__name__}"
