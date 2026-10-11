#!/usr/bin/env python
"""驱动真实 AssistantApp 跑完整链路，并把 UI 叠加层渲染结果存成图片。

走的是 core/app.py 的真实代码路径：
    inject_text → 意图队列 → _intent_loop → _dispatch → _do_identify → Visualizer.draw
不是另写一套模拟，用来产出「程序实际长什么样」的证据。

关键点：帧由本脚本合成后注入 camera.latest()，由app 自己的检测线程消费并推理，
本脚本不再并行调 infer —— 同一个 OpenVINO compiled model 并发 infer 会抛
"Infer Request is busy"。
"""

from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from core.app import AssistantApp      # noqa: E402
from core.config import load_config     # noqa: E402

OUT = "docs/screenshots"
os.makedirs(OUT, exist_ok=True)

H, W = 720, 1280


def compose_scene(obj_path: str, rng) -> np.ndarray:
    """把物体图贴到杂乱背景，模拟摄像头看到的一帧。"""
    bg = np.zeros((H, W, 3), np.uint8)
    top, bot = int(rng.integers(70, 130)), int(rng.integers(40, 90))
    for y in range(H):
        t = y / H
        bg[y, :] = (int(top * (1 - t) + bot * t),
                    int(top * 0.95 * (1 - t) + bot * 1.1 * t),
                    int(top * 0.9 * (1 - t) + bot * 1.25 * t))
    for _ in range(int(rng.integers(6, 14))):
        x0, y0 = int(rng.integers(0, W - 120)), int(rng.integers(0, H - 120))
        ww, hh = int(rng.integers(60, 300)), int(rng.integers(40, 200))
        c = tuple(int(v) for v in rng.integers(30, 190, 3))
        cv2.rectangle(bg, (x0, y0), (x0 + ww, y0 + hh), c, -1)
    bg = np.clip(bg.astype(np.int16) + rng.integers(-12, 12, bg.shape, dtype=np.int16),
                 0, 255).astype(np.uint8)

    obj = cv2.imread(obj_path)
    if obj is None:
        return bg
    oh, ow = obj.shape[:2]
    th = int(H * 0.45)
    sc = th / max(oh, ow)
    obj = cv2.resize(obj, (max(8, int(ow * sc)), max(8, int(oh * sc))),
                     interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
    nh, nw = obj.shape[:2]
    x0 = int(np.clip((W - nw) // 2 + rng.integers(-40, 41), 0, W - nw))
    y0 = int(np.clip((H - nh) // 2 + rng.integers(-30, 31), 0, H - nh))
    bg[y0:y0 + nh, x0:x0 + nw] = obj
    return bg


def main() -> int:
    cfg = load_config()
    cfg["yolo"]["device"] = "CPU"
    cfg["tts"]["mode"] = "mock"        # 无 GPT-SoVITS 进程，用 mock 补全播报环节
    cfg["asr"]["enabled"] = False      # 本机无麦克风

    app = AssistantApp(cfg, headless=True, use_voice=False)

    # 合成帧源：检测线程与 UI 都从这里取帧，保证只有一个 infer 在跑
    rng = np.random.default_rng(11)
    jobs = [
        ("bottle", "data/dataset/enroll/bottle_enroll_00.jpg", "这是什么"),
        ("cup", "data/dataset/enroll/cup_enroll_00.jpg", "这是什么"),
        ("bowl", "data/dataset/unknown/unknown_bowl_00.jpg", "这是什么"),
    ]
    frames = {name: compose_scene(src, rng) for name, src, _ in jobs}
    state = {"cur": "bottle", "id": 0}

    def fake_latest():
        state["id"] += 1
        return frames[state["cur"]].copy(), state["id"]

    app.camera.latest = fake_latest

    # 不开真实摄像头（headless + 合成帧）
    app.camera.start = lambda: None

    app.start()
    time.sleep(2.0)   # 等模型与线程就绪

    print("=" * 70)
    for name, src, text in jobs:
        state["cur"] = name
        app.stats.last_reply = ""
        time.sleep(1.5)                       # 让检测线程跑一两轮，填充 _detections
        n_det = len(app._current_detections())

        app.inject_text(text)
        deadline = time.time() + 25
        while time.time() < deadline and not app.stats.last_reply:
            time.sleep(0.05)
        reply = app.stats.last_reply
        sim = app.stats.last_similarity
        print(f"[{name:7s}] {text} -> {reply}")
        print(f"          sim={sim:.1%}  yolo_boxes={n_det}")

        vis = app._viz.draw(frames[name].copy(), app._overlay)
        path = os.path.join(OUT, f"live_{name}.png")
        cv2.imwrite(path, vis)
        print(f"          saved -> {path}")

    print("=" * 70)
    print(f"bank: {app.memory.stats()}")
    app.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
