#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""端到端功能测试：用**真实场景图**跑完整项目链路，逐项断言。

为什么要这个工具
----------------
`tools/eval_dataset.py` 只在**单物体裁剪图**上做评测，`main.py --recognize` 也是
直接把整图喂给 ResNet18 —— 两者都**跳过了 YOLO 检测**。但项目的实际能力是：

    场景帧 → YOLO 检测 → 选框 → 裁剪 ROI → ResNet18 → 特征库检索
           → 未知判断(阈值) → 位置播报 → (NLU/TTS)

本工具用 `tools/fetch_test_scenes.py` 抓到的**完整场景照**跑这条链路，
每张图断言 5 个环节：

    1. 检测   — YOLO 是否在场景里找到目标类
    2. 裁剪   — ROI 是否有效（尺寸/非空）
    3. 识别   — 特征库检索 top-1 是否是正确类别
    4. 未知   — 未录入类别是否被拒绝（score < threshold）
    5. 位置   — 位置播报文案是否非空且合法

特征库从 `data/dataset/enroll/` 现场重建（用临时 DB，不污染演示用的
`data/visual_memory.db`），保证测试自洽、可重复。

用法
----
    python tools/run_feature_tests.py
    python tools/run_feature_tests.py --scenes data/test_scenes
    python tools/run_feature_tests.py --report data/feature_test_report.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

ENROLL_DIR = os.path.join("data", "dataset", "enroll")
SCENE_DIR = os.path.join("data", "test_scenes")

# key -> 中文名（与 fetch 工具一致，仅用于报告展示）
CN = {
    "cup": "马克杯", "bottle": "保温瓶", "keyboard": "键盘",
    "scissors": "剪刀", "clock": "闹钟", "laptop": "笔记本电脑",
    "unknown_bowl": "碗", "unknown_banana": "香蕉",
    "unknown_umbrella": "雨伞", "unknown_chair": "椅子",
    "empty_scene": "无目标场景",
}
ENROLLED_KEYS = {"cup", "bottle", "keyboard", "scissors", "clock", "laptop"}

# 场景图里目标类在 COCO 的类名（用于匹配 YOLO 检测）
CLS = {
    "cup": "cup", "bottle": "bottle", "keyboard": "keyboard",
    "scissors": "scissors", "clock": "clock", "laptop": "laptop",
    "unknown_bowl": "bowl", "unknown_banana": "banana",
    "unknown_umbrella": "umbrella", "unknown_chair": "chair",
    "empty_scene": None,
}


def _key_of_enroll(fname: str) -> Optional[str]:
    m = re.match(r"(.+)_enroll_\d+\.(jpg|jpeg|png)$", fname)
    return m.group(1) if m else None


def _key_of_scene(fname: str) -> Optional[str]:
    m = re.match(r"(.+)_scene_\d+\.(jpg|jpeg|png)$", fname)
    return m.group(1) if m else None


def build_bank(cfg: dict, tmpdir: str):
    """从 data/dataset/enroll/ 重建特征库到临时 DB，返回 (memory, resnet, bank_keys)。"""
    from vision.feature_extractor import ResNet18Extractor
    from vision.memory import VisualMemory

    mcfg = dict(cfg.get("memory", {}) or {})
    mcfg["db_path"] = os.path.join(tmpdir, "test_memory.db")
    mcfg["samples_dir"] = os.path.join(tmpdir, "samples")
    mcfg["embeddings_path"] = os.path.join(tmpdir, "emb.npz")

    resnet = ResNet18Extractor(cfg.get("resnet18", {}))
    resnet.load()
    mem = VisualMemory(mcfg)

    import cv2

    groups: Dict[str, List[str]] = {}
    for fn in sorted(os.listdir(ENROLL_DIR)):
        k = _key_of_enroll(fn)
        if k:
            groups.setdefault(k, []).append(os.path.join(ENROLL_DIR, fn))

    print("[bank] 从 %s 重建特征库" % ENROLL_DIR)
    for key, paths in groups.items():
        embs = []
        used = []
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            e = resnet.embed(img)
            if e is not None:
                embs.append(e)
                used.append(p)
        if not embs:
            print("  [%s] 无有效 embedding，跳过" % key)
            continue
        # user_name 用 ASCII key，避免中文名模糊匹配干扰断言
        mem.add_object(key, CLS.get(key), embs, used)
        print("  [%-8s] %-6s %d 张样本" % (key, CN.get(key, ""), len(embs)))
    print("[bank] 共 %d 个对象，阈值 %.4f\n" % (len(mem.list_objects()), mem.threshold))
    return mem, resnet, set(groups)


def run_unknown_crop_tests(cfg: dict, mem, resnet) -> dict:
    """第二路测试：用 data/dataset/unknown/ 的**已裁剪**未录入类图测拒绝能力。

    为什么单独一路：场景图依赖 Commons 下载，批量抓取时会被限流；
    而这批裁剪图本地已有（就是当初标定阈值用的负样本），是验证
    「未知物体被拒绝」最直接、最可复现的材料。
    """
    unk_dir = os.path.join("data", "dataset", "unknown")
    if not os.path.isdir(unk_dir):
        return {"available": False, "reason": f"{unk_dir} 不存在"}

    import cv2

    thr = float(cfg.get("memory", {}).get("match_threshold", 0.6769))
    rows = []
    n_ok = 0
    for fn in sorted(os.listdir(unk_dir)):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        # 文件名形如 unknown_bowl_00.jpg
        m = re.match(r"unknown_(.+?)_\d+\.(jpg|jpeg|png)$", fn)
        cls_key = m.group(1) if m else "?"
        img = cv2.imread(os.path.join(unk_dir, fn))
        if img is None:
            continue
        emb = resnet.embed(img)
        if emb is None:
            continue
        mm = mem.best_match(emb)
        if mm is None:
            continue
        score = float(mm.score)
        rejected = score < thr
        n_ok += int(rejected)
        rows.append({
            "file": fn, "class": cls_key,
            "score": round(score, 4),
            "best_match": mm.user_name,
            "rejected": rejected,
        })
    return {
        "available": True,
        "threshold": thr,
        "n": len(rows),
        "n_rejected": n_ok,
        "reject_rate": round(n_ok / len(rows) * 100, 1) if rows else None,
        "max_score": round(max((r["score"] for r in rows), default=0.0), 4),
        "rows": rows,
    }


def run_tests(cfg: dict, scenes_dir: str) -> dict:
    from vision.detector import YoloDetector, COCO_CLASSES
    from vision.object_cropper import ObjectCropper
    from vision.position import describe_position

    ycfg = dict(cfg.get("yolo", {}) or {})
    ycfg["conf_threshold"] = 0.25
    det = YoloDetector(ycfg)
    det.load()
    cropper = ObjectCropper()
    thr = float(cfg.get("memory", {}).get("match_threshold", 0.6769))

    tmpdir = tempfile.mkdtemp(prefix="feat_test_")
    try:
        mem, resnet, bank_keys = build_bank(cfg, tmpdir)

        results = []
        counts = {"detect": 0, "crop": 0, "recognize": 0, "unknown": 0, "position": 0}
        totals = {"detect": 0, "crop": 0, "recognize": 0, "unknown": 0, "position": 0}

        import cv2

        for fn in sorted(os.listdir(scenes_dir)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            key = _key_of_scene(fn)
            if not key:
                continue
            path = os.path.join(scenes_dir, fn)
            img = cv2.imread(path)
            if img is None:
                continue
            h, w = img.shape[:2]
            is_known = key in ENROLLED_KEYS
            want_cls = CLS.get(key)

            r = {"file": fn, "key": key, "cn": CN.get(key, key),
                 "expect": "known" if is_known else "unknown",
                 "checks": {}}

            # ---- 1. 检测 ----
            dets, _ = det.infer(img)
            if want_cls is None:
                cid = None
                target = [d for d in dets if d.cls_id == cid] if cid is not None else []
                totals["detect"] += 1
                ok = True  # 反例：不强制要求检出
                r["checks"]["detect"] = ("skip", "反例场景，不要求检出目标类")
            else:
                cid = COCO_CLASSES.index(want_cls)
                target = [d for d in dets if d.cls_id == cid]
                totals["detect"] += 1
                ok = len(target) > 0
                r["checks"]["detect"] = (
                    "pass" if ok else "fail",
                    f"YOLO {len(target)} 个 {want_cls} 检出"
                    + (f"，最高 conf={max(d.conf for d in target):.2f}" if target else ""),
                )
            counts["detect"] += int(ok)
            r["n_detections"] = len(dets)

            if not target:
                r["verdict"] = "未检出目标，后续环节跳过"
                results.append(r)
                continue

            d = max(target, key=lambda x: x.conf)
            box = tuple(map(int, d.xyxy))

            # ---- 2. 裁剪 ----
            crop = cropper.crop_box(img, box)
            totals["crop"] += 1
            ok_crop = crop is not None and min(crop.shape[:2]) >= 64
            counts["crop"] += int(ok_crop)
            r["checks"]["crop"] = (
                "pass" if ok_crop else "fail",
                f"ROI {crop.shape[1]}x{crop.shape[0]}" if crop is not None else "裁剪为空",
            )
            if not ok_crop:
                r["verdict"] = "ROI 无效"
                results.append(r)
                continue

            # ---- 3+4. 识别 / 未知判断 ----
            emb = resnet.embed(crop)
            totals["recognize"] += 1
            totals["unknown"] += 1
            if emb is None:
                r["checks"]["recognize"] = ("fail", "特征提取失败")
                r["checks"]["unknown"] = ("fail", "无特征，无法判断")
                results.append(r)
                continue

            m = mem.best_match(emb)
            if m is None:
                r["checks"]["recognize"] = ("fail", "特征库为空")
                r["checks"]["unknown"] = ("pass", "库空即未知")
                results.append(r)
                continue

            score = float(m.score)
            predicted = m.user_name
            r["best_match"] = {"name": predicted, "score": round(score, 4)}
            r["threshold"] = thr

            if is_known:
                ok_rec = (predicted == key) and (score >= thr)
                counts["recognize"] += int(ok_rec)
                r["checks"]["recognize"] = (
                    "pass" if ok_rec else "fail",
                    f"检索得 {predicted} (期望 {key})，sim={score:.3f} vs 阈值 {thr}",
                )
                # 已知类不该被拒
                ok_unk = score >= thr
                counts["unknown"] += int(ok_unk)
                r["checks"]["unknown"] = (
                    "pass" if ok_unk else "fail",
                    f"已知类应判定为 known，sim={score:.3f}",
                )
                r["verdict"] = "识别正确" if ok_rec else "识别错误"
            else:
                ok_unk = score < thr
                counts["unknown"] += int(ok_unk)
                r["checks"]["unknown"] = (
                    "pass" if ok_unk else "fail",
                    f"未录入类应被拒绝，sim={score:.3f} vs 阈值 {thr}"
                    + (f"（误判为 {predicted}）" if score >= thr else ""),
                )
                counts["recognize"] += int(ok_unk)  # 对未录入类，"识别正确"= 拒绝
                r["checks"]["recognize"] = (
                    "pass" if ok_unk else "fail",
                    "未录入类的正确行为是拒绝",
                )
                r["verdict"] = "正确拒绝" if ok_unk else "误识别为已知类"

            # ---- 5. 位置播报 ----
            pos = describe_position(box, w, h)
            totals["position"] += 1
            ok_pos = isinstance(pos, str) and pos.startswith("画面") and len(pos) >= 3
            counts["position"] += int(ok_pos)
            r["checks"]["position"] = ("pass" if ok_pos else "fail", f"播报「{pos}」")
            r["position"] = pos
            r["box"] = list(box)

            results.append(r)

        return {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scenes_dir": scenes_dir,
            "threshold": thr,
            "bank_size": len(bank_keys),
            "n_images": len(results),
            "summary": {
                k: {"pass": counts[k], "total": totals[k],
                    "rate": round(counts[k] / totals[k] * 100, 1) if totals[k] else None}
                for k in counts
            },
            "unknown_crops": run_unknown_crop_tests(cfg, mem, resnet),
            "results": results,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def print_report(rep: dict) -> None:
    print("=" * 72)
    print("功能测试报告 — 场景图端到端链路")
    print("=" * 72)
    print("阈值 %.4f | 特征库 %d 类 | 场景图 %d 张\n"
          % (rep["threshold"], rep["bank_size"], rep["n_images"]))

    order = ["detect", "crop", "recognize", "unknown", "position"]
    title = {"detect": "① 检测", "crop": "② 裁剪", "recognize": "③ 识别",
             "unknown": "④ 未知判断", "position": "⑤ 位置播报"}
    for k in order:
        s = rep["summary"][k]
        rate = s["rate"]
        bar = "".join("█" for _ in range(int((rate or 0) / 10)))
        print("%-12s %3d/%-3d  %5.1f%%  %s"
              % (title[k], s["pass"], s["total"], rate or 0, bar))

    print("\n逐图结果：" + "-" * 58)
    for r in rep["results"]:
        flags = "".join(
            {"pass": "✓", "fail": "✗", "skip": "·"}[c[0]] if k in r["checks"] else "·"
            for k, c in r["checks"].items()
        )
        bm = r.get("best_match")
        extra = f"  {bm['name']}={bm['score']:.3f}" if bm else ""
        print("  %-28s %-10s [%s] %-10s%s"
              % (r["file"], r["cn"], flags, r.get("verdict", ""), extra))

    uc = rep.get("unknown_crops") or {}
    if uc.get("available"):
        print("\n未录入类拒绝测试（data/dataset/unknown/ 裁剪图，负样本集）" + "-" * 14)
        print("  %d 张未录入类图，阈值 %.4f，拒绝 %d 张 = %.1f%%，最高分 %.4f"
              % (uc["n"], uc["threshold"], uc["n_rejected"],
                 uc["reject_rate"] or 0, uc["max_score"]))
        by_cls = {}
        for r in uc["rows"]:
            by_cls.setdefault(r["class"], []).append(r)
        for cls, rs in sorted(by_cls.items()):
            rej = sum(1 for r in rs if r["rejected"])
            worst = max(rs, key=lambda r: r["score"])
            print("    %-10s 拒绝 %d/%d   最高分 %.3f → 误判为 %s"
                  % (cls, rej, len(rs), worst["score"], worst["best_match"]))

    fails = [(r, k, c) for r in rep["results"] for k, c in r["checks"].items()
             if c[0] == "fail"]
    uc_fail = [r for r in (uc.get("rows") or []) if not r["rejected"]]

    print("\n" + "=" * 72)
    if fails or uc_fail:
        if fails:
            print("场景图失败项 %d 个：" % len(fails))
            for r, k, c in fails:
                print("  %-28s [%s] %s" % (r["file"], k, c[1]))
        if uc_fail:
            print("未录入类未被拒绝 %d 个（应为未知却被认出）：" % len(uc_fail))
            for r in uc_fail:
                print("  %-28s sim=%.3f → 误判为 %s"
                      % (r["file"], r["score"], r["best_match"]))
    else:
        print("全部通过。")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="场景图端到端功能测试")
    p.add_argument("--scenes", default=SCENE_DIR, help="场景图目录")
    p.add_argument("--report", default=os.path.join("data", "feature_test_report.json"),
                   help="JSON 报告输出路径")
    args = p.parse_args(argv)

    if not os.path.isdir(args.scenes):
        print(f"[err] 场景图目录不存在: {args.scenes}\n"
              f"      先跑: python tools/fetch_test_scenes.py --per-class 3",
              file=sys.stderr)
        return 2

    from core.config import load_config

    cfg = load_config()
    rep = run_tests(cfg, args.scenes)
    print_report(rep)

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print("\n报告已写入 %s" % args.report)

    total_fail = sum(1 for r in rep["results"] for c in r["checks"].values()
                     if c[0] == "fail")
    uc = rep.get("unknown_crops") or {}
    total_fail += sum(1 for r in (uc.get("rows") or []) if not r["rejected"])
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main())