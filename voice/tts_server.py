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
    device = "cpu"          # cpu | cuda | mps
    fp16 = False
    vits_weights = ""       # optional override, e.g. .../s2Gv3.pth
    t2s_weights = ""        # optional override, e.g. .../s1v3.ckpt
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

    @staticmethod
    def _synth_gpt_sovits(text: str, payload: dict, out: str) -> None:
        """Integrate GPT-SoVITS TTS inference API.

        Repo layout (v2+, verified on E:\\GPT-SoVITS):
          <root>/GPT_SoVITS/TTS_infer_pack/TTS.py   → `from GPT_SoVITS.TTS_infer_pack.TTS import TTS`
          <root>/GPT_SoVITS/sv.py→ needs <root> AND <root>/GPT_SoVITS on sys.path
                                                   (it builds paths from os.getcwd())
          <root>/GPT_SoVITS/eres2net/            → ERes2NetV2, kaldi (voice-similarity)
        NOTE: the server process must be started with CWD = <root> for sv.py to resolve.
        """
        if TTSHandler._engine is None:
            gpt = TTSHandler.gpt_path
            if not gpt or not os.path.isdir(gpt):
                raise RuntimeError(f"GPT-SoVITS path not found: {gpt}")
            gpt = os.path.abspath(gpt)
            # sv.py does `sys.path.append(f"{os.getcwd()}/GPT_SoVITS/eres2net")`,
            # so both the root and root/GPT_SoVITS must be importable, and CWD must be root.
            if os.getcwd() != gpt:
                os.chdir(gpt)
            for p in (gpt, os.path.join(gpt, "GPT_SoVITS")):
                if p not in sys.path:
                    sys.path.insert(0, p)
            try:
                from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # type: ignore
            except Exception:
                # older layout: TTS_infer_pack sits directly under root
                from TTS_infer_pack.TTS import TTS, TTS_Config  # type: ignore

            device = TTSHandler.device or "cpu"
            # 必须显式传 yaml 路径。TTS_Config() 无参调用走的是
            # deepcopty(default_configs)["v2"]，压根不读 tts_infer.yaml
            # （TTS.py:300-307），于是 t2s/vits 权重会指向已裁剪掉的
            # gsv-v2final-pretrained，报 FileNotFoundError。
            cfg_path = os.path.join(gpt, "GPT_SoVITS", "configs", "tts_infer.yaml")
            config = TTS_Config(cfg_path if os.path.isfile(cfg_path) else None)
            # 先存原始文本：TTS_Config.__init__ 内部会 save_configs() 重写该文件。
            if os.path.isfile(cfg_path):
                with open(cfg_path, encoding="utf-8") as f:
                    TTSHandler._yaml_backup = f.read()
            config.device = device
            # CPU / low-memory: FP32 is required for correctness; FP16 only on CUDA
            if hasattr(config, "is_half"):
                config.is_half = bool(device == "cuda" and TTSHandler.fp16)
            # TTS_Config.__init__ 会把解析结果 save_configs() 写回
            # configs/tts_infer.yaml，所以命令行传入的路径会被持久化到磁盘 ——
            # 下次不带参数启动就会沿用这次的路径（实测踩过：yaml 里残留了
            # Temp/e2e_export_xxx/xxx.pth）。记下原始版本，引擎起来后还原。
            yaml_version = config.version
            yaml_cfg_path = config.configs_path
            if TTSHandler.vits_weights and os.path.isfile(TTSHandler.vits_weights):
                # version 保持 yaml 里的值不变：TTS.py:515-522 里 version 决定
                # 是否走 v3/v4 分支（BigVGAN vocoder / sample_steps），
                # 与 vits 权重路径是两份独立配置，不该被 --vits-weights 改写。
                config.vits_weights_path = TTSHandler.vits_weights
            if TTSHandler.t2s_weights and os.path.isfile(TTSHandler.t2s_weights):
                config.t2s_weights_path = TTSHandler.t2s_weights
            log_msg = (f"device={device} is_half={config.is_half} "
                       f"version={config.version} "
                       f"vits={os.path.basename(config.vits_weights_path)}")
            try:
                TTSHandler._engine = ("v2", TTS(config))
            except Exception as e:
                raise RuntimeError(f"Failed to init GPT-SoVITS ({log_msg}): {e}") from e
            print(f"[tts] engine ready: {log_msg}", file=sys.stderr)
            # 还原 yaml：引擎已在内存里跑起来了，磁盘配置没必要留着本次的临时路径。
            # 不还原的话，下次裸跑 start_tts_server.bat 会去加载一个已删除的临时文件
            # → FileNotFoundError。这里直接用启动前的原始文本回写，最保险。
            if TTSHandler.vits_weights or TTSHandler.t2s_weights:
                src_yaml = getattr(TTSHandler, "_yaml_backup", None)
                if src_yaml:
                    with open(yaml_cfg_path, "w", encoding="utf-8") as f:
                        f.write(src_yaml)
                    print(f"[tts] config restored: {yaml_cfg_path}", file=sys.stderr)
                else:
                    print(f"[tts] WARN: 没有 yaml 备份可还原", file=sys.stderr)

        kind, engine = TTSHandler._engine
        ref_audio = payload.get("ref_audio") or payload.get("ref_audio_path") or TTSHandler.ref_audio
        prompt_text = payload.get("prompt_text") or TTSHandler.prompt_text
        text_lang = payload.get("text_lang", "zh")
        prompt_lang = payload.get("prompt_lang", "zh")

        if not ref_audio or not os.path.isfile(ref_audio):
            raise RuntimeError(
                f"reference audio not found: {ref_audio!r} — "
                "start the server with --ref-audio <5-10s wav> (GPT-SoVITS clones the voice from it)"
            )

        inputs = {
            "text": text,
            "text_lang": text_lang,
            "ref_audio_path": ref_audio,
            "prompt_text": prompt_text,
            "prompt_lang": prompt_lang,
            "text_split_method": "cut5",
            "top_k": 15,
            "top_p": 1.0,
            "temperature": 1.0,
            "speed_factor": 1.0,
            "fragment_interval": 0.3,
            "seed": -1,
        }
        # TTS exposes run(); older forks defined __call__ — support both.
        runner = getattr(engine, "run", None) or engine
        result = runner(inputs)
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
        # GPT-SoVITS 的 BigVGAN 声码器输出**是 int16 量级**（实测 peak≈18601），
        # 不是 [-1,1] 浮点。直接 clip(-1,1)*32767 会把 70%+ 样本削平成方波 → 刺耳电子杂音。
        # 这里按实际峰值归一化到 [-0.95, 0.95]，对两种量级都成立。
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.5:                     # int16 量级（或任何超出 [-1,1] 的输出）
            audio = audio / peak * 0.95
        else:
            # 已是浮点量级：轻推一下，静音/极轻样本会让人误以为没出声
            rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) if audio.size else 0.0
            if rms > 1e-6 and rms < 0.02:
                audio = audio * (0.05 / rms)
            elif peak > 0:
                audio = audio / peak * 0.95
        audio = np.clip(audio, -1.0, 1.0)

        import wave

        pcm = (audio * 32767.0).astype(np.int16)
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
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"],
                   help="inference device (default cpu; FP32 is used unless --fp16 + cuda)")
    p.add_argument("--fp16", action="store_true", help="allow FP16 (cuda only; CPU must stay FP32)")
    p.add_argument("--vits-weights", default=os.environ.get("GPT_SOVITS_VITS", ""),
                   help="override VITS weight path (e.g. s2Gv3.pth)")
    p.add_argument("--t2s-weights", default=os.environ.get("GPT_SOVITS_T2S", ""),
                   help="override t2s weight path (e.g. s1v3.ckpt)")
    p.add_argument("--preload", action="store_true",
                   help="load GPT-SoVITS at startup and report failures immediately")
    args = p.parse_args(argv)

    TTSHandler.mode = "mock" if args.mock else "gpt_sovits"
    TTSHandler.gpt_path = args.gpt_sovits_path
    TTSHandler.ref_audio = args.ref_audio
    TTSHandler.prompt_text = args.prompt_text
    TTSHandler.return_path = args.return_path
    TTSHandler.device = args.device
    TTSHandler.fp16 = args.fp16
    TTSHandler.vits_weights = args.vits_weights
    TTSHandler.t2s_weights = args.t2s_weights

    if not args.mock and not args.gpt_sovits_path:
        print("[tts_server] ERROR: --gpt-sovits-path is required unless --mock is given.\n"
              f"           Example: python -m voice.tts_server --gpt-sovits-path E:/GPT-SoVITS --port 19880",
              file=sys.stderr)
        return 2

    # fail fast: a missing repo / ref-audio / weight should not surface at the first user request
    if not args.mock and args.preload:
        tmp = os.path.join(tempfile.gettempdir(), "gpt_sovits_preload.wav")
        try:
            TTSHandler._synth_gpt_sovits(
                "预热",
                {"ref_audio": TTSHandler.ref_audio, "prompt_text": TTSHandler.prompt_text},
                tmp,
            )
            print(f"[tts] preload OK -> {tmp} ({os.path.getsize(tmp)} bytes)", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"[tts] preload FAILED: {e}", file=sys.stderr)
            print("       若报 ref_audio 相关错误，请用 --ref-audio 指定 5-10s 参考 wav。", file=sys.stderr)
            return 3

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
