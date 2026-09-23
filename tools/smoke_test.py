#!/usr/bin/env python
"""Offline smoke test — no camera/voice required for basic checks.

    python tools/smoke_test.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)


def main() -> int:
    ok = True

    # 1) config
    from core.config import load_config

    cfg = load_config()
    print("[1] config OK", list(cfg.keys()))

    # 2) NLU
    from nlu.intent import IntentType, parse_intent

    cases = [
        ("这是什么", IntentType.IDENTIFY),
        ("记住，这个叫我的蓝色水杯", IntentType.REMEMBER),
        ("帮我找我的蓝色水杯", IntentType.SEARCH),
        ("你记住了什么", IntentType.LIST),
        ("你能做什么", IntentType.HELP),
        ("忘记蓝色水杯", IntentType.FORGET),
    ]
    for text, want in cases:
        got = parse_intent(text)
        status = "OK" if got.type == want else f"FAIL got={got.type}"
        if got.type != want:
            ok = False
        print(f"[2] '{text}' → {got.type.value} label={got.label!r} {status}")

    # 3) memory roundtrip
    from vision.memory import VisualMemory

    mem = VisualMemory(
        {
            "db_path": os.path.join(tempfile.gettempdir(), "smoke_mem.db"),
            "samples_dir": os.path.join(tempfile.gettempdir(), "smoke_samples"),
            "embeddings_path": os.path.join(tempfile.gettempdir(), "smoke_emb.npz"),
            "match_threshold": 0.5,
        }
    )
    dim = 384
    base = np.random.randn(dim).astype(np.float32)
    base /= np.linalg.norm(base)
    emb_near = base + 0.01 * np.random.randn(dim).astype(np.float32)
    emb_near /= np.linalg.norm(emb_near)
    emb_far = np.random.randn(dim).astype(np.float32)
    emb_far /= np.linalg.norm(emb_far)
    mem.add_object("测试杯子", "cup", [base, emb_near], [])
    m = mem.best_match(emb_near)
    if m is None or "测试杯子" not in m.user_name:
        ok = False
        print("[3] memory match FAIL", m)
    else:
        print(f"[3] memory match OK score={m.score:.3f} → {m.user_name}")
    m2 = mem.best_match(emb_far)
    print(f"[3] far query score={(m2.score if m2 else 0):.3f} (may still match at low thr)")
    mem.delete_object("测试杯子")
    mem.close()

    # 4) cropper / position / visualizer (headless draw)
    import cv2

    from vision.object_cropper import ObjectCropper
    from vision.position import describe_position
    from vision.visualizer import OverlayState, Visualizer

    frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    cropper = ObjectCropper()
    box = cropper.center_roi(frame)
    crop = cropper.crop_box(frame, box)
    assert crop.size > 0
    pos = describe_position((50, 50, 200, 200), 640, 480)
    viz = Visualizer()
    st = OverlayState(status_line="smoke", learning=True, learning_label="测试")
    out = viz.draw(frame, st)
    assert out.shape == frame.shape
    print(f"[4] cropper/position/visualizer OK pos={pos}")

    # 5) YOLO / DINO load if models present
    from vision.detector import YoloDetector
    from vision.feature_extractor import DinoExtractor

    yolo = YoloDetector(cfg.get("yolo", {}))
    try:
        yolo.load()
        dets, dt = yolo.infer(frame)
        print(f"[5] YOLO OK device={yolo.device_used} dets={len(dets)} {dt:.1f}ms")
    except Exception as e:
        print(f"[5] YOLO SKIP/FAIL: {e}")

    dino = DinoExtractor(cfg.get("dino", {}))
    try:
        dino.load()
        e, dt = dino.timed_embed(crop)
        print(f"[6] DINO OK device={dino.device_used} dim={0 if e is None else e.shape[0]} {dt:.1f}ms")
    except Exception as e:
        print(f"[6] DINO SKIP/FAIL: {e}")

    # 7) ASR load if present
    try:
        from voice.asr import ParaformerASR

        asr = ParaformerASR(cfg.get("asr", {}))
        asr.load()
        print("[7] ASR OK")
    except Exception as e:
        print(f"[7] ASR SKIP: {e}")

    # 8) TTS mock synth
    from voice.tts import TtsClient

    tts_cfg = dict(cfg.get("tts", {}))
    tts_cfg["mode"] = "mock"
    tts_cfg["play_audio"] = False
    tts_cfg["enabled"] = True
    client = TtsClient(tts_cfg)
    path = client.synthesize("测试语音")
    print(f"[8] TTS mock OK → {path}")

    print("\n" + ("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
