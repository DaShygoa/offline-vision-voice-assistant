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
    assert "resnet18" in cfg, "config must contain resnet18 section"
    assert "dino" not in cfg, "dino section should be removed for 选题一"

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

    # 3) feature bank roundtrip — single-channel ResNet18 cosine retrieval
    from vision.memory import VisualMemory

    mem = VisualMemory(
        {
            "db_path": os.path.join(tempfile.gettempdir(), "smoke_mem.db"),
            "samples_dir": os.path.join(tempfile.gettempdir(), "smoke_samples"),
            "embeddings_path": os.path.join(tempfile.gettempdir(), "smoke_emb.npz"),
            "match_threshold": 0.5,
            "embed_dim": 512,
        }
    )
    dim = 512
    base = np.random.randn(dim).astype(np.float32)
    base /= np.linalg.norm(base)
    emb_near = base + 0.01 * np.random.randn(dim).astype(np.float32)
    emb_near /= np.linalg.norm(emb_near)
    emb_far = np.random.randn(dim).astype(np.float32)
    emb_far /= np.linalg.norm(emb_far)
    # second object for ranking
    base2 = np.random.randn(dim).astype(np.float32)
    base2 /= np.linalg.norm(base2)
    mem.add_object("测试杯子", "cup", [base, emb_near], [])
    mem.add_object("测试书本", "book", [base2], [])
    m = mem.best_match(emb_near)
    if m is None or "测试杯子" not in m.user_name:
        ok = False
        print("[3] memory match FAIL", m)
    else:
        print(f"[3] cosine match OK score={m.score:.3f} → {m.user_name}")
        st = mem.stats()
        if st["matrix_shape"][0] < 3 or st["matrix_shape"][1] != 512:
            ok = False
            print("[3] feature bank FAIL", st)
        else:
            print(f"[3] feature bank OK matrix={st['matrix_shape']}")
    # unknown judgment: low-score query must be below threshold
    m2 = mem.best_match(emb_far)
    if m2 is not None and m2.score >= mem.threshold:
        # random high-dim vectors are usually ~0 cosine; if not, still report
        print(f"[3] far query score={m2.score:.3f} >= thr (random vectors rare case)")
    else:
        sc = m2.score if m2 else 0.0
        print(f"[3] unknown judgment OK score={sc:.3f} < thr={mem.threshold}")
    mem.delete_object("测试杯子")
    mem.close()

    # 4) cropper / position / overlay state (no GUI in the classroom release)
    import cv2

    from vision.object_cropper import ObjectCropper
    from vision.position import describe_position
    from vision.visualizer import OverlayState

    frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    cropper = ObjectCropper()
    box = cropper.center_roi(frame)
    crop = cropper.crop_box(frame, box)
    assert crop.size > 0
    pos = describe_position((50, 50, 200, 200), 640, 480)
    st = OverlayState(learning=True, learning_label="测试")
    assert st.learning and st.learning_label == "测试"
    assert not cropper.quality_ok(np.zeros((80, 80, 3), dtype=np.uint8))  # blank rejected
    print(f"[4] cropper/position/overlay OK pos={pos}")

    # 5) depth model if present (zero-shot localization)
    from vision.depth import DepthEstimator

    depth = DepthEstimator(cfg.get("depth", {}))
    try:
        depth.load()
        d = depth.estimate_depth(frame)
        print(f"[5] depth OK device={depth.device_used} map={'yes' if d is not None else 'no'}")
    except Exception as e:
        print(f"[5] depth SKIP/FAIL: {e}")

    # 6) ResNet18 load if model present (选题一 core)
    from vision.feature_extractor import ResNet18Extractor

    resnet = ResNet18Extractor(cfg.get("resnet18", {}))
    try:
        resnet.load()
        e, dt = resnet.timed_embed(crop)
        dim_ok = e is not None and e.shape[0] == 512
        if not dim_ok:
            ok = False
        nrm = float(np.linalg.norm(e)) if e is not None else 0.0
        print(
            f"[6] ResNet18 OK device={resnet.device_used} "
            f"dim={0 if e is None else e.shape[0]} L2={nrm:.3f} {dt:.1f}ms"
        )
    except Exception as e:
        print(f"[6] ResNet18 SKIP/FAIL: {e}")
        print("     run: python tools/prepare_models.py --resnet18")

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
