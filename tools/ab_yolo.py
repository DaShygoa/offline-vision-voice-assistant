#!/usr/bin/env python
"""A/B 实测：YOLO 开启 vs 关闭，对识别准确率与速度的影响。

严格复刻 core/app.py::_do_identify 与 _do_remember 的取框逻辑，
唯一变量是 YOLO 是否提供候选框：

  分支 A (yolo_enabled=True) : 有YOLO 框 → crop_detections → 全部 embed → 取最高分
                               无框 → 回退中心 ROI（同分支 B）
  分支 B (yolo_enabled=False): dets 为空 → 直接走中心 ROI

输出：Top-1 命中率、平均相似度、已知/未知判定正确率、耗时。
"""

from __future__ import annotations

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
from vision.feature_extractor import ResNet18Extractor   # noqa: E402
from vision.memory import VisualMemory                  # noqa: E402
from vision.object_cropper import ObjectCropper          # noqa: E402
from vision.detector import YoloDetector                 # noqa: E402


def collect_images(d: str) -> list[str]:
    if not os.path.isdir(d):
        return []
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return [os.path.join(d, f) for f in sorted(os.listdir(d))
            if f.lower().endswith(exts)]


def identify(frame, dets, cropper, ex, mem, use_yolo: bool):
    """复刻 app._do_identify 的取框分支，返回 (best_match, box)。"""
    best = None
    best_box = None

    # 分支 A：YOLO 候选框
    if use_yolo and dets:
        for crop, box, _det in cropper.crop_detections(frame, dets):
            if not cropper.quality_ok(crop):
                continue
            emb = ex.embed(crop)
            if emb is None:
                continue
            m = mem.best_match(emb)
            if m and (best is None or m.score > best.score):
                best, best_box = m, box

    # 分支 B / 回退：中心 ROI
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


def main() -> int:
    cfg = load_config()
    cfg["yolo"]["device"] = "CPU"

    ex = ResNet18Extractor(cfg.get("resnet18", {}))
    ex.load()
    mem = VisualMemory(cfg.get("memory", {}))
    cropper = ObjectCropper(
        center_roi_frac=float(cfg.get("memory", {}).get("center_roi_frac", 0.45)),
        min_crop_px=int(cfg.get("memory", {}).get("min_crop_px", 64)),
    )
    det = YoloDetector(cfg.get("yolo", {}))
    det.load()

    print(f"ResNet18: {ex.device_used}")
    print(f"YOLO    : {det.device_used}")
    print(f"bank    : {mem.stats()}")
    print(f"threshold: {mem.threshold}\n")

    enroll_root = "data/dataset/enroll"
    test_root = "data/dataset/test"
    unknown_root = "data/dataset/unknown"

    test_imgs = collect_images(test_root)
    unknown_imgs = collect_images(unknown_root)
    if not test_imgs and not unknown_imgs:
        print("[err] no dataset found under data/dataset/", file=sys.stderr)
        return 2

    # 数据集是平铺命名：<key>_test_NN.jpg / unknown_<key>_NN.jpg
    def truth_label(p: str) -> str | None:
        stem = os.path.splitext(os.path.basename(p))[0]
        return stem.split("_test_")[0]

    results = {"yolo_on": [], "yolo_off": []}
    det_time = []

    for mode, use_yolo in (("yolo_on", True), ("yolo_off", False)):
        print(f"===== {mode} =====")
        for p in test_imgs + unknown_imgs:
            img = cv2.imread(p)
            if img is None:
                continue
            # 检测只跑一次，两组共用同一批框（隔离变量：只改变「用不用」）
            t0 = time.perf_counter()
            dets, _ms = det.infer(img)
            if use_yolo:
                det_time.append((time.perf_counter() - t0) * 1000)

            t1 = time.perf_counter()
            m, _box = identify(img, dets, cropper, ex, mem, use_yolo)
            dt = (time.perf_counter() - t1) * 1000

            is_unknown_set = p.startswith(unknown_root.replace("\\", "/"))
            want = truth_label(p)
            got = m.user_name if m else None
            score = m.score if m else 0.0
            thr = mem.threshold
            decided_unknown = (m is None) or (score < thr)

            if is_unknown_set:
                correct = decided_unknown
                kind = "unknown"
            else:
                correct = (not decided_unknown) and (got == want)
                kind = "known"

            results[mode].append({
                "path": os.path.relpath(p).replace("\\", "/"),
                "kind": kind, "want": want, "got": got,
                "score": round(score, 4), "correct": bool(correct),
                "ms": round(dt, 1),
            })

    # ---- 汇总 ----
    summary = {}
    for mode, rows in results.items():
        known = [r for r in rows if r["kind"] == "known"]
        unk = [r for r in rows if r["kind"] == "unknown"]

        # (1) 排名型Top-1：与阈值无关，纯比「正确物体是否排在第 1」
        rank_hit = sum(1 for r in known if r["got"] == r["want"])
        # (2) 决策型正确：排名对 且 分数过阈值
        dec_hit = sum(1 for r in known if r["correct"])
        # (3) 未知拒绝率
        urec = sum(1 for r in unk if r["correct"])

        summary[mode] = {
            "n_total": len(rows),
            "known_n": len(known),
            "rank_top1": rank_hit,
            "rank_top1_acc": round(rank_hit / len(known), 4) if known else None,
            "decision_hit": dec_hit,
            "decision_acc": round(dec_hit / len(known), 4) if known else None,
            "unknown_n": len(unk), "unknown_correct": urec,
            "unknown_acc": round(urec / len(unk), 4) if unk else None,
            "mean_score": round(statistics.fmean([r["score"] for r in rows]), 4),
            "pos_min": round(min([r["score"] for r in known]), 4) if known else None,
            "neg_max": round(max([r["score"] for r in unk]), 4) if unk else None,
            "mean_ms": round(statistics.fmean([r["ms"] for r in rows]), 1),
        }
        s = summary[mode]
        print(f"{mode:9s} 排名Top1={s['rank_top1']}/{s['known_n']}"
              f"({s['rank_top1_acc']*100:.1f}%)  决策命中={s['decision_hit']}/{s['known_n']}"
              f"({s['decision_acc']*100:.1f}%)  未知拒绝={s['unknown_correct']}/{s['unknown_n']}"
              f"({s['unknown_acc']*100:.1f}%)")
        print(f"{'':9s} 正样本min={s['pos_min']} 负样本max={s['neg_max']}"
              f" 均值={s['mean_score']} 耗时={s['mean_ms']}ms")

    if det_time:
        print(f"\nYOLO detect mean: {statistics.fmean(det_time):.1f}ms "
              f"(p95 {sorted(det_time)[int(len(det_time)*0.95)]:.1f}ms)")

    # ---- 差异 ----
    a, b = summary["yolo_on"], summary["yolo_off"]
    print("\n===== YOLO 开 vs 关 差异=====")
    for label, key, fmt in (
        ("排名Top-1 准确率", "rank_top1_acc", "pct"),
        ("决策命中准确率", "decision_acc", "pct"),
        ("未知拒绝率", "unknown_acc", "pct"),
    ):
        d = (b[key] - a[key]) * 100
        print(f"{label:14s} on={a[key]*100:5.1f}%  off={b[key]*100:5.1f}%  Δ={d:+.1f}pp")
    print(f"{'平均相似度':14s} on={a['mean_score']:.4f}   off={b['mean_score']:.4f}   "
          f"Δ={b['mean_score']-a['mean_score']:+.4f}")
    print(f"{'单次识别耗时':14s} on={a['mean_ms']:.1f}ms     off={b['mean_ms']:.1f}ms     "
          f"Δ={b['mean_ms']-a['mean_ms']:+.1f}ms")

    os.makedirs("data/bench", exist_ok=True)
    with open("data/bench/ab_yolo.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": results}, f,
                  ensure_ascii=False, indent=2)
    print("\n[ok] 明细写入 data/bench/ab_yolo.json")

    mem.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
