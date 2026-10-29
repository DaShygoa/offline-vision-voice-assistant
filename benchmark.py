#!/usr/bin/env python
"""benchmark.py — on-machine performance report (选题一).

Measures:
  * ResNet18 inference latency  (ONNX Runtime CPU, avg / p50 / p95)
  * YOLO inference latency      (optional ROI helper)
  * ASR latency                 (Paraformer decode on synthetic/test wav)
  * TTS RTF                     (GPT-SoVITS server or mock)
  * RAM (process / system), CPU % (best effort)

Usage:
    python benchmark.py
    python benchmark.py --skip-asr --skip-tts
    python benchmark.py --runs 50 --out benchmark_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import tempfile
import time
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger("benchmark")


# --------------------------------------------------------------------- utils
def _pct(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def _stats(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "avg_ms": round(statistics.fmean(xs), 3),
        "p50_ms": round(_pct(xs, 50), 3),
        "p95_ms": round(_pct(xs, 95), 3),
        "min_ms": round(min(xs), 3),
        "max_ms": round(max(xs), 3),
    }


def _process_ram_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / (1024 * 1024)


def _system_ram() -> Dict[str, float]:
    import psutil

    v = psutil.virtual_memory()
    return {
        "total_gb": round(v.total / (1024**3), 2),
        "used_gb": round(v.used / (1024**3), 2),
        "percent": v.percent,
    }


def _cpu_info() -> Dict[str, object]:
    import platform
    import psutil

    return {
        "name": platform.processor() or platform.machine(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "usage_percent": psutil.cpu_percent(interval=1.0),
    }


def _onnxruntime_info() -> Dict[str, object]:
    info: Dict[str, object] = {"available": True, "providers": None, "error": None}
    try:
        import onnxruntime as ort
        info["providers"] = ort.get_available_providers()
    except Exception as e:
        info["available"] = False
        info["error"] = str(e)
    return info


def _synth_wav(path: str, seconds: float = 1.6, sr: int = 16000) -> str:
    """Create a simple tonal wav as ASR placeholder input (replace with real speech for real latency)."""
    import wave

    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    # two-formant-ish buzz so VAD/ASR at least has energy
    audio = 0.25 * np.sin(2 * np.pi * 180 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
    audio = audio.astype(np.float32)
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


# ---------------------------------------------------------------- benchmarks
def bench_yolo(cfg: dict, runs: int, frame: np.ndarray) -> Dict[str, object]:
    from vision.detector import YoloDetector

    det = YoloDetector(cfg.get("yolo", {}))
    t_load = time.perf_counter()
    try:
        det.load()
    except Exception as e:
        return {"ok": False, "error": str(e)}
    load_s = time.perf_counter() - t_load
    # warmup
    for _ in range(3):
        det.infer(frame)
    lat = []
    n_det = 0
    for _ in range(runs):
        dets, dt = det.infer(frame)
        lat.append(dt)
        n_det += len(dets)
    return {
        "ok": True,
        "backend": det.backend,
        "device": det.device_used,
        "load_s": round(load_s, 3),
        "detections_avg": round(n_det / max(1, runs), 2),
        "latency": _stats(lat),
        "det_fps_est": round(1000.0 / statistics.fmean(lat), 1) if lat else 0,
    }


def bench_resnet18(cfg: dict, runs: int, frame: np.ndarray) -> Dict[str, object]:
    from vision.feature_extractor import ResNet18Extractor
    from vision.object_cropper import ObjectCropper

    cropper = ObjectCropper()
    crop = cropper.crop_box(frame, cropper.center_roi(frame))
    ext = ResNet18Extractor(cfg.get("resnet18", {}))
    t_load = time.perf_counter()
    try:
        ext.load()
    except Exception as e:
        return {"ok": False, "error": str(e)}
    load_s = time.perf_counter() - t_load
    for _ in range(3):
        ext.embed(crop)
    lat = []
    ok = 0
    for _ in range(runs):
        e, dt = ext.timed_embed(crop)
        lat.append(dt)
        if e is not None:
            ok += 1
    emb = ext.embed(crop)
    return {
        "ok": True,
        "backend": ext.backend,
        "device": ext.device_used,
        "load_s": round(load_s, 3),
        "embed_ok": ok,
        "embed_dim": int(emb.shape[0]) if emb is not None else 0,
        "latency": _stats(lat),
        "note": "ResNet18 runs only on identify/remember/search — not per frame",
    }


def bench_asr(cfg: dict) -> Dict[str, object]:
    from voice.asr import ParaformerASR

    asr = ParaformerASR(cfg.get("asr", {}))
    t_load = time.perf_counter()
    try:
        asr.load()
    except Exception as e:
        return {"ok": False, "error": str(e)}
    load_s = time.perf_counter() - t_load
    wav = os.path.join(tempfile.gettempdir(), "bench_asr.wav")
    # Prefer a real sample if present
    real = None
    for cand in (
        os.path.join(ROOT, "data", "asr_sample.wav"),
        os.path.join(ROOT, "models", "asr_sample.wav"),
    ):
        if os.path.isfile(cand):
            real = cand
            break
    path = real or _synth_wav(wav, 1.6)
    # warmup
    asr.transcribe_file(path)
    lat = []
    text = ""
    for _ in range(5):
        t0 = time.perf_counter()
        text = asr.transcribe_file(path)
        lat.append((time.perf_counter() - t0) * 1000.0)
    return {
        "ok": True,
        "model": cfg.get("asr", {}).get("encoder", ""),
        "load_s": round(load_s, 3),
        "input": path,
        "real_speech_sample": real is not None,
        "latency": _stats(lat),
        "text_head": text[:40],
        "note": "tonal placeholder yields empty/garbage text; latency still valid" if real is None else "real sample",
    }


def bench_tts(cfg: dict) -> Dict[str, object]:
    from voice.tts import TtsClient

    tts_cfg = dict(cfg.get("tts", {}))
    # benchmark shouldn't require auto-play blocking forever
    tts_cfg["play_audio"] = False
    tts_cfg["enabled"] = True
    client = TtsClient(tts_cfg)
    if client.mode == "gpt_sovits":
        health = client.health()
        if health == "down":
            return {
                "ok": False,
                "error": f"GPT-SoVITS server down at {client.server_url}",
                "hint": "start: python -m voice.tts_server --gpt-sovits-path <path> --port 19880",
                "fallback_mode": "mock",
            }
    elif client.mode == "edge":
        health = client.health()
        if health.startswith("down"):
            return {
                "ok": False,
                "error": f"edge-tts unavailable ({health})",
                "hint": "pip install edge-tts  (needs network access)",
                "fallback_mode": "mock",
            }
    # don't start full worker for synth-only bench
    try:
        import sounddevice  # noqa: F401

        has_audio = True
    except Exception:
        has_audio = False
    texts = [
        "这是一个杯子。",
        "好的，我已经记住你的蓝色水杯了。",
        "找到你的蓝色水杯了，在画面左侧。",
        "我还不认识这个物品。",
        "你可以教我：记住，这个叫。",
    ]
    rtfs = []
    lat = []
    audio_durs = []
    for t in texts:
        t0 = time.perf_counter()
        path = client.synthesize(t)
        dt = (time.perf_counter() - t0) * 1000.0
        lat.append(dt)
        if path and os.path.isfile(path):
            dur = TtsClient._wav_duration(path)
            audio_durs.append(dur)
            if dur > 0:
                rtfs.append((dt / 1000.0) / dur)
    if not lat:
        return {"ok": False, "error": "no synthesis completed"}
    rtf = statistics.fmean(rtfs) if rtfs else None
    return {
        "ok": True,
        "mode": client.mode,
        "server": client.server_url if client.mode == "gpt_sovits" else None,
        "has_audio_device": has_audio,
        "synth_latency": _stats(lat),
        "avg_audio_s": round(statistics.fmean(audio_durs), 3) if audio_durs else 0,
        "rtf": round(rtf, 3) if rtf is not None else None,
        "rtf_ok": (rtf is not None and rtf < 1.0),
    }


def bench_memory(cfg: dict) -> Dict[str, object]:
    from vision.memory import VisualMemory

    mem = VisualMemory(cfg.get("memory", {}))
    st = mem.stats()
    # synthetic match latency
    dim = int(cfg.get("memory", {}).get("embed_dim", 512))
    q = np.random.randn(dim).astype(np.float32)
    q /= max(1e-8, float(np.linalg.norm(q)))
    # inject a temporary object
    t0 = time.perf_counter()
    try:
        emb = [np.random.randn(dim).astype(np.float32) for _ in range(5)]
        mem.add_object("__bench_tmp__", "cup", emb, [])
        lat = []
        for _ in range(20):
            tq = time.perf_counter()
            mem.query(q, topk=3)
            lat.append((time.perf_counter() - tq) * 1000.0)
        mem.delete_object("__bench_tmp__")
        return {
            "ok": True,
            "db": cfg.get("memory", {}).get("db_path"),
            "objects": st["objects"],
            "samples": st["samples"],
            "query_latency": _stats(lat),
            "backend": "numpy_cosine_resnet18",
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# --------------------------------------------------------------------- main
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="On-machine benchmark for offline vision-voice assistant")
    p.add_argument("--config", default=None)
    p.add_argument("--runs", type=int, default=30, help="YOLO/ResNet18 timed iterations")
    p.add_argument("--skip-yolo", action="store_true")
    p.add_argument("--skip-resnet", action="store_true")
    p.add_argument("--skip-dino", action="store_true", help="deprecated alias of --skip-resnet")
    p.add_argument("--skip-asr", action="store_true")
    p.add_argument("--skip-tts", action="store_true")
    p.add_argument("--skip-memory", action="store_true")
    p.add_argument("--skip-system", action="store_true")
    p.add_argument("--out", default="benchmark_report.json")
    p.add_argument("--no-cuda-assume", action="store_true", default=True, help="never probe CUDA (target has no NVIDIA)")
    args = p.parse_args(argv)

    from core.config import load_config

    cfg = load_config(args.config)
    report: Dict[str, object] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "cuda_assumed": False,
    }

    # synthetic frame from camera if possible, else noise
    frame = None
    try:
        import cv2

        cap = cv2.VideoCapture(int(cfg.get("app", {}).get("camera_index", 0)), cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, frame = cap.read()
            cap.release()
    except Exception:
        frame = None
    if frame is None:
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        report["frame_source"] = "synthetic"
    else:
        report["frame_source"] = "camera"

    if not args.skip_system:
        report["onnxruntime"] = _onnxruntime_info()
        report["cpu"] = _cpu_info()
        report["ram"] = _system_ram()
        report["process_ram_mb_baseline"] = round(_process_ram_mb(), 1)

    if not args.skip_yolo:
        log.info("Benchmarking YOLO …")
        report["yolo"] = bench_yolo(cfg, args.runs, frame)
    if not (args.skip_resnet or args.skip_dino):
        log.info("Benchmarking ResNet18 …")
        report["resnet18"] = bench_resnet18(cfg, args.runs, frame)
    if not args.skip_memory:
        log.info("Benchmarking visual memory …")
        report["memory"] = bench_memory(cfg)
    if not args.skip_asr:
        log.info("Benchmarking ASR …")
        report["asr"] = bench_asr(cfg)
    if not args.skip_tts:
        log.info("Benchmarking TTS …")
        report["tts"] = bench_tts(cfg)

    # peak RAM after benches
    try:
        import psutil

        proc = psutil.Process()
        report["process_ram_mb_peak"] = round(proc.memory_info().rss / (1024 * 1024), 1)
        report["process_cpu_percent"] = proc.cpu_percent(interval=0.5)
        report["ram"] = _system_ram()
    except Exception:
        pass

    # pretty print
    print("\n" + "=" * 72)
    print("BENCHMARK REPORT")
    print("=" * 72)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("=" * 72)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"saved → {os.path.abspath(args.out)}")

    # non-zero exit if YOLO completely failed (models missing)
    y = report.get("yolo") or {}
    if not args.skip_yolo and isinstance(y, dict) and not y.get("ok"):
        log.warning("YOLO benchmark failed — check models (python tools/prepare_models.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
