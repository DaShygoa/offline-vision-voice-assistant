"""Standalone GPT-SoVITS TTS server (INDEPENDENT process).

Run this in a separate terminal (never inside the vision process):

    python -m voice.tts_server --gpt-sovits-path D:/GPT-SoVITS --port 9880

It exposes:
    GET  /health
    POST /tts   {"text": "...", "ref_audio": "...", "prompt_text": "...", "prompt_lang": "zh"}
         → audio/wav bytes  (or JSON {"path": "/abs/out.wav"} if --return-path)

If GPT-SoVITS is not installed, start with --mock to serve synthetic audio
so the rest of the pipeline can be tested offline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _mock_wav(text: str, path: str, sample_rate: int = 32000) -> str:
    import numpy as np
    import struct
    import wave

    dur = max(0.4, min(6.0, len(text) / 4.5))
    n = int(sample_rate * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    env = 0.18 * (0.55 + 0.45 * np.sin(2 * np.pi * 2.5 * t))
    audio = (env * np.sin(2 * np.pi * 392 * t)).astype(np.float32)
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return path


class TTSHandler(BaseHTTPRequestHandler):
    mode = "mock"
    gpt_path = ""
    lock = threading.Lock()
    ref_audio = ""
    prompt_text = ""
    return_path = False
    _engine = None

    def log_message(self, fmt, *args):  # quieter
        sys.stderr.write("[tts] " + (fmt % args) + "\n")

    def do_GET(self):
        if self.path.startswith("/health"):
            body = json.dumps({"ok": True, "mode": self.mode}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self):
        if not self.path.startswith("/tts"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self.send_error(400, "invalid json")
            return
        text = str(payload.get("text", "")).strip()
        if not text:
            self.send_error(400, "empty text")
            return
        out = os.path.join(tempfile.gettempdir(), f"gpt_sovits_{int(time.time()*1000)}.wav")
        try:
            with self.lock:
                if self.mode == "mock":
                    _mock_wav(text, out)
                else:
                    self._synth_gpt_sovits(text, payload, out)
            if self.return_path:
                body = json.dumps({"path": out}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                with open(out, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except Exception as e:
            self.send_error(500, str(e))

    def _synth_gpt_sovits(self, text: str, payload: dict, out: str) -> None:
        """Integrate GPT-SoVITS TTS inference API.

        Supports the common GPT-SoVITS repo layout:
          - GPT_SoVITS/TTS_infer_pack/TTS.py  (API v2)
        Adjust paths here if your fork differs.
        """
        if TTSHandler._engine is None:
            gpt = self.gpt_path
            if not gpt or not os.path.isdir(gpt):
                raise RuntimeError(f"GPT-SoVITS path not found: {gpt}")
            if gpt not in sys.path:
                sys.path.insert(0, gpt)
            try:
                from TTS_infer_pack.TTS import TTS, TTS_Config  # type: ignore

                device = "cpu"
                config = TTS_Config()
                config.device = device
                # keep memory modest on 16GB machine
                if hasattr(config, "is_half"):
                    config.is_half = False
                TTSHandler._engine = ("v2", TTS(config))
            except Exception as e:
                raise RuntimeError(f"Failed to init GPT-SoVITS: {e}") from e

        kind, engine = TTSHandler._engine
        ref_audio = payload.get("ref_audio") or self.ref_audio
        prompt_text = payload.get("prompt_text") or self.prompt_text
        text_lang = payload.get("text_lang", "zh")
        prompt_lang = payload.get("prompt_lang", "zh")
        inputs = {
            "text": text,
            "text_lang": text_lang,
            "ref_audio_path": ref_audio,
            "prompt_text": prompt_text if prompt_lang.startswith("zh") or prompt_lang == "zh" else prompt_text,
            "prompt_lang": prompt_lang,
            "media_type": "wav",
            "streaming_mode": "offline",
        }
        # common optional knobs — ignore if unsupported
        for k, v in (
            ("prompt_sample_rate", 32000),
            ("text_split_method", "cut5"),
            ("top_k", 5),
            ("top_p", 1.0),
            ("temperature", 1.0),
            ("speed_factor", 1.0),
        ):
            inputs.setdefault(k, v)
        result = engine(inputs)
        # result: (sample_rate, ndarray) or generator / dict depending on version
        import numpy as np

        if isinstance(result, tuple) and len(result) == 2:
            sr, audio = result
        elif isinstance(result, dict):
            sr = int(result.get("sample_rate", 32000))
            audio = result.get("audio")
        else:
            # offline generator — concat chunks
            chunks = list(result)
            if not chunks:
                raise RuntimeError("GPT-SoVITS returned no audio")
            if isinstance(chunks[0], tuple):
                sr = chunks[0][0]
                audio = np.concatenate([c[1] for c in chunks])
            else:
                sr = 32000
                audio = np.concatenate(chunks)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        import wave

        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        with wave.open(out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(int(sr))
            w.writeframes(pcm.tobytes())


def main(argv=None):
    p = argparse.ArgumentParser(description="Independent GPT-SoVITS TTS server (CPU)")
    p.add_argument("--port", type=int, default=19880)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--gpt-sovits-path", default=os.environ.get("GPT_SOVITS_PATH", ""))
    p.add_argument("--ref-audio", default="")
    p.add_argument("--prompt-text", default="")
    p.add_argument("--mock", action="store_true", help="serve synthetic tones (no GPT-SoVITS)")
    p.add_argument("--return-path", action="store_true", help="return JSON path instead of wav bytes")
    args = p.parse_args(argv)

    TTSHandler.mode = "mock" if args.mock else "gpt_sovits"
    TTSHandler.gpt_path = args.gpt_sovits_path
    TTSHandler.ref_audio = args.ref_audio
    TTSHandler.prompt_text = args.prompt_text
    TTSHandler.return_path = args.return_path

    server = ThreadingHTTPServer((args.host, args.port), TTSHandler)
    print(f"[tts_server] mode={TTSHandler.mode} listening on http://{args.host}:{args.port}")
    print(f"[tts_server] health: GET /health   synth: POST /tts")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[tts_server] bye")
        server.server_close()


if __name__ == "__main__":
    main()
