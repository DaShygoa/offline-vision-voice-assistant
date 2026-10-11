#!/usr/bin/env python
"""真实场景 A/B：把物体裁剪图 贴到「摄像头画面」大小的背景上，模拟真实取景。

为什么要造场景图
----------------
上一轮直接把YOLO 裁剪图当输入，关掉 YOLO 走 center ROI 相当于「在已经裁紧的图上
再取中心 45%」，把物体边缘切掉了——这不是真实用法。真实链路里摄像头看到的是
完整场景，物体只占画面一部分：

    yolo_on : YOLO 框 → 紧贴物体 → 特征干净
    yolo_off: 画面中心 45% → 必然带背景 → 特征被背景稀释

这才是「关掉 YOLO 到底损失什么」的真实答案。
"""

from __future__ import annotations

import glob
import json
import os
import statistics
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from core.config import load_config                      # noqa: E402
from vision.feature_extractor import ResNet18Extractor# noqa: E402
from vision.memory import VisualMemory                  # noqa: E402
from vision.object_cropper import ObjectCropper          # noqa: E402
from vision.detector import YoloDetector                 # noqa: E402

W, H = 1280, 720
KEYS = ["cup", "bottle", "keyboard", "scissors", "clock", "laptop"]
UNK = ["banana", "bowl", "chair", "umbrella"]


def make_background(rng: np.random.Generator) -> np.ndarray:
    """生成桌面/地面风格的杂乱背景，避免纯色导致特征虚高。"""
    bg = np.zeros((H, W, 3), np.uint8)
    # 渐变底色
    top = rng.integers(70, 130)
    bot = rng.integers(40, 90)
    for y in range(H):
        t = y / H
        bg[y, :] = (int(top * (1 - t) + bot * t),
                    int(top * 0.95 * (1 - t) + bot * 1.1 * t),
                    int(top * 0.9 * (1 - t) + bot * 1.25 * t))
    # 随机杂物色块，增加背景多样性
    for _ in range(rng.integers(6, 14)):
        x0, y0 = rng.integers(0, W - 120), rng.integers(0, H - 120)
        ww, hh = rng.integers(60, 300), rng.integers(40, 200)
        c = tuple(int(v) for v in rng.integers(30, 190, 3))
        cv2.rectangle(bg, (x0, y0), (x0 + ww, y0 + hh), c, -1)
    # 轻微噪声
    noise = rng.integers(-12, 12, bg.shape, dtype=np.int16)
    return np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def compose(rng, obj_img: np.ndarray, centered: bool) -> np.ndarray:
    """把物体图缩放后贴到背景上。centered=True 放在画面中央（模拟用户摆正）。"""
    bg = make_background(rng)
    oh, ow = obj_img.shape[:2]
    target_h = int(H * rng.uniform(0.34, 0.52))
    scale = target_h / max(oh, ow)
    nw, nh = max(8, int(ow * scale)), max(8, int(oh * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    obj = cv2.resize(obj_img, (nw, nh), interpolation=interp)
    if centered:
        x0 = (W - nw) // 2 + int(rng.integers(-30, 31))
        y0 = (H - nh) // 2 + int(rng.integers(-25, 26))
    else:
        x0 = int(rng.integers(int(W * 0.12), int(W * 0.62)))
        y0 = int(rng.integers(int(H * 0.12), int(H * 0.55)))
    x0 = int(np.clip(x0, 0, W - nw))
    y0 = int(np.clip(y0, 0, H - nh))
    bg[y0:y0 + nh, x0:x0 + nw] = obj
    return bg


def identify(frame, dets, cropper, ex, mem, use_yolo: bool):
    best = None
    best_box = None
    if use_yolo and dets:
        for crop, box, _d in cropper.crop_detections(frame, dets):
            if not cropper.quality_ok(crop):
                continue
            emb = ex.embed(crop)
            if emb is None:
                continue
            m = mem.best_match(emb)
            if m and (best is None or m.score > best.score):
                best, best_box = m, box
    if best is None:
        box = cropper.center_roi(frame)
        crop = cropper.crop_box(frame, box)
        if not cropper.quality_ok(crop):
            return None, None
        emb = ex.embed(crop)
        if emb is None:
            return None, None
        best = mem.best_match(emb)
        best_box = box
    return best, best_box


def build_scenes(seed=7):
    """为每个测试/未知样本生成 1 张居中场景图 + 1 张偏置场景图。"""
    out = []
    rng = np.random.default_rng(seed)
    for p in sorted(glob.glob("data/dataset/test/*_test_*.jpg")):
        stem = os.path.splitext(os.path.basename(p))[0].split("_test_")[0]
        img = cv2.imread(p)
        if img is None:
            continue
        for centered in (True, False):
            out.append({"kind": "known", "want": stem,
                        "img": compose(rng, img, centered), "centered": centered})
    for p in sorted(glob.glob("data/dataset/unknown/*.jpg")):
        stem = os.path.splitext(os.path.basename(p))[0].split("_")[1]
        img = cv2.imread(p)
        if img is None:
            continue
        for centered in (True, False):
            out.append({"kind": "unknown", "want": None,
                        "img": compose(rng, img, centered), "centered": centered})
    return out


def run(scenes, ex, mem, cropper, det, thr):
    res = {"on": [], "off": []}
    tdet = []
    for sc in scenes:
        frame = sc["img"]
        t0 = time.perf_counter()
        dets, _ = det.infer(frame)
        tdet.append((time.perf_counter() - t0) * 1000)
        for mode, use in (("on", True), ("off", False)):
            t1 = time.perf_counter()
            m, _b = identify(frame, dets, cropper, ex, mem, use)
            ms = (time.perf_counter() - t1) * 1000
            got = m.user_name if m else None
            score = m.score if m else 0.0
            rejected = (m is None) or (score < thr)
            if sc["kind"] == "known":
                rank_ok = (got == sc["want"])
                correct = rank_ok and not rejected
            else:
                rank_ok = None
                correct = rejected
            res[mode].append({
                "kind": sc["kind"], "want": sc["want"], "got": got,
                "centered": sc["centered"], "n_det": len(dets),
                "score": round(score, 4), "rank_ok": rank_ok,
                "correct": bool(correct), "ms": round(ms, 1),
            })
    return res, tdet


def summarize(res, thr):
    out = {}
    for mode, rows in res.items():
        kn = [r for r in rows if r["kind"] == "known"]
        un = [r for r in rows if r["kind"] == "unknown"]
        kn_c = [r for r in kn if r["centered"]]
        kn_off = [r for r in kn if not r["centered"]]
        rank = sum(1 for r in kn if r["rank_ok"])
        dec = sum(1 for r in kn if r["correct"])
        urec = sum(1 for r in un if r["correct"])
        rank_c = sum(1 for r in kn_c if r["rank_ok"])
        rank_o = sum(1 for r in kn_off if r["rank_ok"])
        out[mode] = {
            "known_n": len(kn), "rank_top1": rank,
            "rank_top1_acc": round(rank / len(kn), 4),
            "rank_top1_centered": round(rank_c / len(kn_c), 4) if kn_c else None,
            "rank_top1_offset": round(rank_o / len(kn_off), 4) if kn_off else None,
            "decision_acc": round(dec / len(kn), 4),
            "unknown_n": len(un), "unknown_acc": round(urec / len(un), 4),
            "mean_score": round(statistics.fmean([r["score"] for r in rows]), 4),
            "pos_min": round(min([r["score"] for r in kn]), 4),
            "neg_max": round(max([r["score"] for r in un]), 4),
            "mean_ms": round(statistics.fmean([r["ms"] for r in rows]), 1),
        }
    return out


def report(tag, summ, tdet):
    print(f"\n===== {tag} =====")
    for mode in ("on", "off"):
        s = summ[mode]
        print(f" YOLO {mode:3s}: 排名Top1={s['rank_top1']}/{s['known_n']}"
              f"({s['rank_top1_acc']*100:5.1f}%)"
              f" | 居中={s['rank_top1_centered']*100:5.1f}%"
              f" 偏置={s['rank_top1_offset']*100:5.1f}%"
              f" | 未知拒绝={s['unknown_acc']*100:5.1f}%"
              f" | 决策={s['decision_acc']*100:5.1f}%"
              f" | {s['mean_ms']}ms")
        print(f"          正样本min={s['pos_min']} 负样本max={s['neg_max']}"
              f" 均值={s['mean_score']}")
    a, b = summ["on"], summ["off"]
    print(" 差异 (off - on):")
    print(f"   排名Top1{(b['rank_top1_acc']-a['rank_top1_acc'])*100:+.1f}pp"
          f" | 居中{(b['rank_top1_centered']-a['rank_top1_centered'])*100:+.1f}pp"
          f" | 偏置{(b['rank_top1_offset']-a['rank_top1_offset'])*100:+.1f}pp")
    print(f"   未知拒绝{(b['unknown_acc']-a['unknown_acc'])*100:+.1f}pp"
          f" | 相似度{b['mean_score']-a['mean_score']:+.4f}"
          f" | 耗时{b['mean_ms']-a['mean_ms']:+.1f}ms")
    print(f"   YOLO 检测耗时均值 {statistics.fmean(tdet):.0f}ms "
          f"(p95 {sorted(tdet)[int(len(tdet)*0.95)]:.0f}ms)")


def main() -> int:
    cfg = load_config()
    cfg["yolo"]["device"] = "CPU"
    ex = ResNet18Extractor(cfg["resnet18"]); ex.load()
    mem = VisualMemory(cfg["memory"])
    cropper = ObjectCropper(center_roi_frac=float(cfg["memory"]["center_roi_frac"]),
                            min_crop_px=int(cfg["memory"]["min_crop_px"]))
    det = YoloDetector(cfg["yolo"]); det.load()

    scenes = build_scenes()
    print(f"ResNet18 {ex.device_used} | YOLO {det.device_used}")
    print(f"bank {mem.stats()}")
    print(f"场景数 {len(scenes)}（每物体 居中/偏置 各一张）")

    # 场景图预览（供人工核对）
    os.makedirs("data/bench/scenes", exist_ok=True)
    for i in (0, 1, 2, len(scenes)//2):
        cv2.imwrite(f"data/bench/scenes/scene_{i:03d}.jpg", scenes[i]["img"])

    for thr, tag in ((mem.threshold, f"当前阈值 {mem.threshold}"),
                     (0.6323, "重标定阈值 0.6323")):
        res, tdet = run(scenes, ex, mem, cropper, det, thr)
        report(tag, summarize(res, thr), tdet)
        with open(f"data/bench/scene_ab_thr{thr}.json", "w", encoding="utf-8") as f:
            json.dump({"threshold": thr, "summary": summarize(res, thr), "rows": res},
                      f, ensure_ascii=False, indent=2)

    mem.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
