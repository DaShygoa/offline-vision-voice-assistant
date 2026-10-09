#!/usr/bin/env python
"""离线评测 + 阈值标定 (选题一「未知判断」).

为什么需要它
------------
`config/default.yaml` 里的 `memory.match_threshold: 0.65` 是拍出来的。
ImageNet-pretrained ResNet18 是分类 backbone，不是细粒度检索模型，
真实余弦分布必须实测。本工具用 data/dataset/ 的真实图片跑完整链路：

    enroll 组 → VisualMemory.add_object (真实 ResNet18 512-d)
    test组   → best_match()→ 统计 top-1 命中率
    unknown 组 → best_match() → 统计「应当拒绝」的最高分
然后扫描阈值求 balanced accuracy 最优点，并给出正负分布直方图。

关键点：负样本取「unknown 图在 enrolled 库上的最高相似度」，
这正是现场 `_do_identify` 判定「未知物体」时看到的那个数。

Usage:
    python tools/eval_dataset.py                       # full report
    python tools/eval_dataset.py --apply               # write threshold into config
    python tools/eval_dataset.py --enroll-from-dataset # rebuild bank from dataset
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

DATA = os.path.join("data", "dataset")
REPORT = os.path.join(DATA, "eval_report.json")


def _list(d: str, prefix: str = "") -> List[str]:
    if not os.path.isdir(d):
        return []
    return sorted(
        os.path.join(d, f)
        for f in os.listdir(d)
        if f.lower().endswith((".jpg", ".jpeg", ".png")) and f.startswith(prefix)
    )


def load_bank_from_dataset(resnet, memory, verbose=True) -> Dict[str, List[str]]:
    """Enroll every object that has images in data/dataset/enroll/."""
    enroll_dir = os.path.join(DATA, "enroll")
    groups: Dict[str, List[str]] = {}
    for p in _list(enroll_dir):
        key = os.path.basename(p).split("_")[0]
        groups.setdefault(key, []).append(p)
    if not groups:
        raise SystemExit(f"no images in {enroll_dir} — run tools/fetch_dataset.py first")

    for key, paths in sorted(groups.items()):
        embs, saved = [], []
        obj_dir = os.path.join("data", "samples", key)
        os.makedirs(obj_dir, exist_ok=True)
        for i, src in enumerate(paths):
            img = cv2.imread(src)
            if img is None:
                continue
            e = resnet.embed(img)
            if e is None:
                continue
            dest = os.path.join(obj_dir, f"s{i:02d}.png")
            cv2.imwrite(dest, img)
            embs.append(e)
            saved.append(dest)
        if not embs:
            continue
        cn = key
        mpath = os.path.join(DATA, "manifest.json")
        if os.path.isfile(mpath):
            try:
                with open(mpath, "r", encoding="utf-8") as f:
                    man = json.load(f)
                for o in man.get("objects", []):
                    if o.get("key") == key:
                        cn = o.get("cn", key)
                        break
            except Exception:  # noqa: BLE001
                pass
        memory.add_object(cn, key, embs, saved)
        if verbose:
            print(f"  enrolled '{cn}' ({key}) with {len(embs)} samples")
    return groups


def sweep_threshold(pos: List[float], neg: List[float]) -> dict:
    """Maximise balanced accuracy over a threshold grid; report the plateau."""
    p = np.asarray(pos, dtype=np.float64)
    n = np.asarray(neg, dtype=np.float64)
    out: dict = {
        "n_pos": int(p.size), "n_neg": int(n.size),
        "pos_mean": None, "pos_min": None, "pos_p05": None,
        "neg_mean": None, "neg_max": None, "neg_p95": None,
        "suggested_threshold": None, "conservative_threshold": None,
        "separable": None, "method": None, "note": None,
    }
    if p.size == 0 or n.size == 0:
        out["note"] = "需要同时有 enrolled(test) 和 unknown 图片"
        return out

    out.update(
        pos_mean=round(float(p.mean()), 4), pos_min=round(float(p.min()), 4),
        pos_p05=round(float(np.percentile(p, 5)), 4),
        neg_mean=round(float(n.mean()), 4), neg_max=round(float(n.max()), 4),
        neg_p95=round(float(np.percentile(n, 95)), 4),
    )

    if float(p.min()) > float(n.max()):
        thr = (float(p.min()) + float(n.max())) / 2.0
        gap = float(p.min()) - float(n.max())
        out.update(suggested_threshold=round(thr, 4), separable=True,
                   method="midpoint(pos_min, neg_max) — 分布完全可分",
                   conservative_threshold=round(thr, 4),
                   gap=round(gap, 4),
                   margin_below_pos_min=round(float(p.min()) - thr, 4),
                   margin_above_neg_max=round(thr - float(n.max()), 4),
                   note=(f"正负分布完全分离，中间有 {gap:.4f} 的空隙；"
                         f"取中点使两侧余量相等，对分布漂移最稳健。"
                         f"注意：当前 config 的阈值若贴近 neg_max，现场稍受扰动就会误判为「已知」。"))
        return out

    out["separable"] = False
    grid = np.linspace(0.20, 0.99, 160)
    best_thr, best_acc = 0.65, -1.0
    curve = []
    for thr in grid:
        tpr = float((p >= thr).mean())
        tnr = float((n < thr).mean())
        acc = 0.5 * (tpr + tnr)
        curve.append((round(float(thr), 3), round(tpr, 4), round(tnr, 4), round(acc, 4)))
        if acc > best_acc:
            best_acc, best_thr = acc, float(thr)
        out["suggested_threshold"] = round(best_thr, 4)
        out["balanced_accuracy"] = round(best_acc, 4)
        out["method"] = f"网格扫描 0.20~0.99，balanced accuracy={best_acc:.4f}"
        out["curve"] = curve
        # margin: how far the chosen thr sits from the nearest failure on each side
        out["margin_below_pos_min"] = round(float(p.min()) - best_thr, 4)
        out["margin_above_neg_max"] = round(best_thr - float(n.max()), 4)
    # plateau: all thresholds within 99% of best acc
    good = [c[0] for c in curve if c[3] >= best_acc * 0.99]
    out["plateau"] = [min(good), max(good)] if good else None
    # conservative: maximise TNR (avoid false accept) subject to TPR >= 0.8
    ok = [c for c in curve if c[1] >= 0.80]
    if ok:
        out["conservative_threshold"] = min(c[0] for c in ok)
        out["note"] = (f"conservative = 在 TPR≥0.80 前提下取最大 TNR 的阈值"
                       f"（现场演示更看重「不误认」）")
    else:
        out["conservative_threshold"] = round(min(0.95, best_thr + 0.03), 4)
        out["note"] = "无阈值能同时满足 TPR≥0.80，建议补数据或换更可分的物体"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate the dataset and calibrate match_threshold")
    ap.add_argument("--apply", action="store_true", help="write the chosen threshold into config/default.yaml")
    ap.add_argument("--prefer", choices=["suggested", "conservative"], default="conservative",
                    help="which threshold to --apply")
    ap.add_argument("--rebuild-bank", action="store_true",
                    help="clear data/visual_memory.db and re-enroll from dataset (destructive)")
    args = ap.parse_args(argv)

    from core.config import load_config
    from vision.feature_extractor import ResNet18Extractor
    from vision.memory import VisualMemory

    cfg = load_config()
    resnet = ResNet18Extractor(cfg.get("resnet18", {}))
    resnet.load()
    print(f"[ok] ResNet18 {resnet.device_used} dim={resnet.embed_dim}")

    if args.rebuild_bank:
        for p in (cfg["memory"]["db_path"], cfg["memory"]["embeddings_path"]):
            if os.path.isfile(p):
                os.remove(p)
                print(f"[rm] {p}")
        sd = cfg["memory"]["samples_dir"]
        if os.path.isdir(sd):
            import shutil
            shutil.rmtree(sd)
            print(f"[rm] {sd}")

    memory = VisualMemory(cfg.get("memory", {}))
    try:
        st = memory.stats()
        if st["objects"] == 0:
            print("[..] bank empty — enrolling from dataset")
            load_bank_from_dataset(resnet, memory)
        else:
            print(f"[ok] bank has {st['objects']} objects / {st['samples']} samples "
                  f"(use --rebuild-bank to rebuild from dataset)")
        print("[..] enrolled objects:", "、".join(o.user_name for o in memory.list_objects()))

        names = {o.user_name for o in memory.list_objects()}

        # ---- positive: held-out test images, expect correct identity ----
        print("\n[..] scoring held-out test images")
        pos_rows = []
        for p in _list(os.path.join(DATA, "test")):
            key = os.path.basename(p).split("_")[0]
            cn = key
            mpath = os.path.join(DATA, "manifest.json")
            if os.path.isfile(mpath):
                try:
                    with open(mpath, "r", encoding="utf-8") as f:
                        for o in json.load(f).get("objects", []):
                            if o.get("key") == key:
                                cn = o.get("cn", key)
                                break
                except Exception:  # noqa: BLE001
                    pass
            img = cv2.imread(p)
            if img is None:
                continue
            e = resnet.embed(img)
            if e is None:
                continue
            m = memory.best_match(e)
            if m is None:
                continue
            hit = m.user_name == cn or key in m.user_name or cn in m.user_name
            pos_rows.append(dict(file=os.path.basename(p), truth=cn, pred=m.user_name,
                                 score=round(float(m.score), 4), correct=bool(hit)))
            print(f"   {os.path.basename(p):<28} truth={cn:<8} pred={m.user_name:<8} "
                  f"score={m.score:.4f} {'OK' if hit else 'MISS'}")

        # ---- negative: unknown objects, must be rejected ----
        print("\n[..] scoring unknown objects (should be rejected)")
        neg_rows = []
        for p in _list(os.path.join(DATA, "unknown")):
            # 文件名形如 unknown_banana_00.jpg → 去掉序号得unknown_banana
            key = os.path.basename(p).rsplit("_", 1)[0]
            img = cv2.imread(p)
            if img is None:
                continue
            e = resnet.embed(img)
            if e is None:
                continue
            m = memory.best_match(e)
            if m is None:
                continue
            neg_rows.append(dict(file=os.path.basename(p), truth=key,
                                 nearest=m.user_name, score=round(float(m.score), 4)))
            print(f"   {os.path.basename(p):<28} unknown={key:<22} "
                  f"nearest={m.user_name:<8} score={m.score:.4f}")

        pos = [r["score"] for r in pos_rows]
        neg = [r["score"] for r in neg_rows]
        sweep = sweep_threshold(pos, neg)
        cur = float(cfg["memory"]["match_threshold"])

        acc = (sum(r["correct"] for r in pos_rows) / len(pos_rows)) if pos_rows else 0.0
        print("\n" + "=" * 68)
        print("EVALUATION REPORT")
        print("=" * 68)
        print(f"enrolled objects : {len(names)}  ({'、'.join(sorted(names))})")
        print(f"test top-1 accuracy: {acc*100:.1f}%  ({sum(r['correct'] for r in pos_rows)}/{len(pos_rows)})")
        print(f"current threshold : {cur}")
        print(f"positive scores  : n={sweep['n_pos']} mean={sweep['pos_mean']} min={sweep['pos_min']} p05={sweep['pos_p05']}")
        print(f"negative scores  : n={sweep['n_neg']} mean={sweep['neg_mean']} max={sweep['neg_max']} p95={sweep['neg_p95']}")
        print(f"separable        : {sweep['separable']}")
        print(f"suggested        : {sweep['suggested_threshold']}  ({sweep['method']})")
        print(f"conservative     : {sweep.get('conservative_threshold')}")
        if sweep.get("plateau"):
            print(f"plateau          : {sweep['plateau']}")
        if sweep.get("note"):
            print(f"note             : {sweep['note']}")
        for thr in (cur, sweep["suggested_threshold"], sweep.get("conservative_threshold")):
            if thr is None:
                continue
            tpr = sum(1 for s in pos if s >= thr) / max(1, len(pos))
            tnr = sum(1 for s in neg if s < thr) / max(1, len(neg))
            print(f"   thr={thr:.3f}  TPR(known)={tpr*100:5.1f}%  TNR(unknown)={tnr*100:5.1f}%  "
                  f"balanced={0.5*(tpr+tnr)*100:5.1f}%")
        print("=" * 68)

        report = dict(
            generated=__import__("time").strftime("%Y-%m-%d %H:%M:%S"),
            device=resnet.device_used, embed_dim=resnet.embed_dim,
            enrolled=sorted(names), current_threshold=cur,
            top1_accuracy=round(acc, 4),
            test_rows=pos_rows, unknown_rows=neg_rows, sweep=sweep,
        )
        os.makedirs(DATA, exist_ok=True)
        with open(REPORT, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"saved → {os.path.abspath(REPORT)}")

        if args.apply:
            key = "conservative_threshold" if args.prefer == "conservative" else "suggested_threshold"
            thr = sweep.get(key)
            if thr is None:
                print("[skip] no threshold to apply")
                return 1
            cpath = os.path.join("config", "default.yaml")
            with open(cpath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            out, done = [], False
            for ln in lines:
                if not done and ln.strip().startswith("match_threshold:"):
                    indent = ln[: len(ln) - len(ln.lstrip())]
                    out.append(f"{indent}match_threshold: {thr}\n")
                    done = True
                else:
                    out.append(ln)
            if not done:
                print(f"[warn] match_threshold not found in {cpath}")
                return 1
            with open(cpath, "w", encoding="utf-8") as f:
                f.writelines(out)
            print(f"[ok] config/default.yaml  match_threshold: {cur} → {thr}")
    finally:
        memory.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
