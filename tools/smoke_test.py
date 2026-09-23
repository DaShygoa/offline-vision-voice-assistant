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

    # 3) memory roundtrip — multi-channel (dino + shape + depth fusion)
    from vision.memory import VisualMemory

    mem = VisualMemory(
        {
            "db_path": os.path.join(tempfile.gettempdir(), "smoke_mem.db"),
            "samples_dir": os.path.join(tempfile.gettempdir(), "smoke_samples"),
            "embeddings_path": os.path.join(tempfile.gettempdir(), "smoke_emb.npz"),
            "match_threshold": 0.5,
            "use_shape": True,
            "use_depth": True,
            "weight_dino": 0.55,
            "weight_shape": 0.30,
            "weight_depth": 0.15,
        }
    )
    dim = 384
    base = np.random.randn(dim).astype(np.float32)
    base /= np.linalg.norm(base)
    emb_near = base + 0.01 * np.random.randn(dim).astype(np.float32)
    emb_near /= np.linalg.norm(emb_near)
    emb_far = np.random.randn(dim).astype(np.float32)
    emb_far /= np.linalg.norm(emb_far)
    shape_base = np.random.randn(32).astype(np.float32)
    shape_base /= np.linalg.norm(shape_base)
    shape_near = shape_base + 0.02 * np.random.randn(32).astype(np.float32)
    shape_near /= np.linalg.norm(shape_near)
    depth_base = np.random.randn(18).astype(np.float32)
    depth_base /= np.linalg.norm(depth_base)
    depth_near = depth_base + 0.05 * np.random.randn(18).astype(np.float32)
    depth_near /= np.linalg.norm(depth_near)
    mem.add_object(
        "测试杯子",
        "cup",
        [base, emb_near],
        [],
        shape_embeddings=[shape_base, shape_near],
        depth_embeddings=[depth_base, depth_near],
    )
    m = mem.best_match(emb_near, shape_embedding=shape_near, depth_embedding=depth_near)
    if m is None or "测试杯子" not in m.user_name:
        ok = False
        print("[3] memory match FAIL", m)
    else:
        print(
            f"[3] fused match OK score={m.score:.3f} "
            f"dino={m.score_dino if m.score_dino is not None else 0:.3f} "
            f"shape={m.score_shape if m.score_shape is not None else 0:.3f} "
            f"depth={m.score_depth if m.score_depth is not None else 0:.3f} "
            f"→ {m.user_name}"
        )
        st = mem.stats()
        if st["shape_shape"][0] < 2 or st["depth_shape"][0] < 2:
            ok = False
            print("[3] multi-channel index FAIL", st)
        else:
            print(f"[3] multi-channel OK shape={st['shape_shape']} depth={st['depth_shape']}")
    m2 = mem.best_match(emb_far)
    print(f"[3] far query score={(m2.score if m2 else 0):.3f} (may still match at low thr)")
    mem.delete_object("测试杯子")
    mem.close()

    # 3b) shape descriptor + depth extractor
    from vision.depth import DepthExtractor
    from vision.shape_descriptor import ShapeDescriptor

    sh = ShapeDescriptor(cfg.get("shape", {}))
    # synthetic: dark square on light bg
    syn = np.full((160, 160, 3), 220, dtype=np.uint8)
    syn[40:120, 40:120] = 30
    sv = sh.extract(syn, box=(40, 40, 120, 120))
    if sv is None or sv.shape[0] != sh.dim:
        ok = False
        print(f"[3b] shape FAIL dim={sh.dim} sv={None if sv is None else sv.shape}")
    else:
        print(f"[3b] shape OK dim={sh.dim} L2={float(np.linalg.norm(sv)):.3f}")

    depth = DepthExtractor(cfg.get("depth", {}))
    try:
        depth.load()
        if depth.ready:
            dv, ms = depth.timed_extract(syn, frame_bgr=syn, box=(40, 40, 120, 120))
            if dv is None or dv.shape[0] != depth.dim:
                ok = False
                print(f"[3b] depth FAIL dv={None if dv is None else dv.shape}")
            else:
                print(f"[3b] depth OK device={depth.device_used} dim={depth.dim} {ms:.1f}ms")
        else:
            print(f"[3b] depth SKIP ({depth._error or 'disabled'})")
    except Exception as e:
        print(f"[3b] depth SKIP/FAIL: {e}")

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
    # input bar draw (no-mic path)
    st2 = OverlayState(status_line="smoke", input_visible=True, input_focused=True, input_text="这是什么")
    out2 = viz.draw(frame, st2)
    assert out2.shape == frame.shape
    from ui.text_input import TextInputWindow

    ti = TextInputWindow(on_submit=lambda t: None)
    assert ti.display_text == "" and not ti.is_typing
    ti.set_echo("记住这个")
    assert ti.display_text == "记住这个"
    print(f"[4] cropper/position/visualizer+input OK pos={pos}")

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
