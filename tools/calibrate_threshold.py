#!/usr/bin/env python
"""Calibrate memory.match_threshold from real similarity data (选题一 unknown judgment).

Collects cosine scores for:
  * same object, different views
  * different objects
  * (optional) screen photos / lighting variants — add samples under data/calib/

Then prints a suggested threshold (e.g. midway between distributions,
or maximizing balanced accuracy).

Usage:
    # 1) teach several objects through the main app first
    # 2) capture extra query crops (optional):
    python tools/calibrate_threshold.py --capture data/calib/query
    # 3) compute:
    python tools/calibrate_threshold.py --compute
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)


def load_pairs_from_memory() -> Tuple[List[float], List[float]]:
    """positive: query a stored sample vs its object (leave-one-out max).
    negative: query sample vs all OTHER objects' best score."""
    from core.config import load_config
    from vision.memory import VisualMemory

    cfg = load_config()
    mem = VisualMemory(cfg["memory"])
    objs = mem.list_objects()
    if len(objs) < 2:
        print("Need at least 2 remembered objects to calibrate (and ideally 3+).")
        return [], []

    positives: List[float] = []
    negatives: List[float] = []

    for obj in objs:
        # gather this object's ResNet18 embeddings from disk
        rows = mem._conn.execute(
            "SELECT image_path FROM object_samples WHERE object_id=?", (obj.object_id,)
        ).fetchall()
        embs = []
        for r in rows:
            p = r["image_path"]
            if p and os.path.isfile(p + ".npy"):
                e = np.load(p + ".npy").astype(np.float32)
                e /= max(1e-8, float(np.linalg.norm(e)))
                if e.shape[0] == mem.embed_dim:
                    embs.append(e)
        if not embs:
            continue
        for e in embs:
            # score vs own object (excluding self via all samples max is fine for pos approx)
            own = [s for s in mem.query(e, topk=20) if s.object_id == obj.object_id]
            if own:
                positives.append(own[0].score)
            others = [s.score for s in mem.query(e, topk=20) if s.object_id != obj.object_id]
            if others:
                negatives.append(max(others))
    mem.close()
    return positives, negatives


def capture_crops(out_dir: str, n: int = 20) -> None:
    """Grab n frames from camera, save center-ROI crops for manual labeling later."""
    import cv2

    from core.config import load_config
    from vision.object_cropper import ObjectCropper

    cfg = load_config()
    cropper = ObjectCropper()
    cap = cv2.VideoCapture(int(cfg.get("app", {}).get("camera_index", 0)))
    os.makedirs(out_dir, exist_ok=True)
    i = 0
    print(f"Capturing {n} crops → {out_dir}  (Esc to stop)")
    while i < n:
        ok, frame = cap.read()
        if not ok:
            break
        box = cropper.center_roi(frame)
        crop = cropper.crop_box(frame, box)
        path = os.path.join(out_dir, f"crop_{i:03d}.png")
        cv2.imwrite(path, crop)
        cv2.imshow("calib capture", frame)
        if cv2.waitKey(200) & 0xFF == 27:
            break
        i += 1
    cap.release()
    cv2.destroyAllWindows()
    print(f"saved {i} crops")


def suggest_threshold(pos: List[float], neg: List[float]) -> dict:
    pos_a = np.array(pos, dtype=np.float64)
    neg_a = np.array(neg, dtype=np.float64)
    result = {
        "n_pos": int(pos_a.size),
        "n_neg": int(neg_a.size),
        "pos_mean": float(pos_a.mean()) if pos_a.size else None,
        "pos_min": float(pos_a.min()) if pos_a.size else None,
        "neg_mean": float(neg_a.mean()) if neg_a.size else None,
        "neg_max": float(neg_a.max()) if neg_a.size else None,
        "suggested_threshold": None,
        "method": None,
    }
    if pos_a.size == 0 or neg_a.size == 0:
        return result

    # midpoint between pos_min and neg_max if separable
    if pos_a.min() > neg_a.max():
        thr = float((pos_a.min() + neg_a.max()) / 2)
        result["suggested_threshold"] = round(thr, 4)
        result["method"] = "midpoint(pos_min, neg_max) — perfectly separable"
        return result

    # else sweep for best balanced accuracy
    grid = np.linspace(0.3, 0.99, 70)
    best_thr, best_acc = 0.75, -1.0
    for thr in grid:
        tpr = float((pos_a >= thr).mean())
        tnr = float((neg_a < thr).mean())
        acc = 0.5 * (tpr + tnr)
        if acc > best_acc:
            best_acc, best_thr = acc, float(thr)
    result["suggested_threshold"] = round(best_thr, 4)
    result["method"] = f"best balanced accuracy={best_acc:.3f} on sweep"
    # safety margin: require a bit above midpoint of overlap
    if neg_a.max() >= pos_a.min():
        overlap_hi = float(neg_a.max())
        conservative = round(min(0.95, max(best_thr, overlap_hi + 0.03)), 4)
        result["conservative_threshold"] = conservative
        result["note"] = "overlap detected — prefer conservative_threshold to reduce false accepts"
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--capture", metavar="DIR", help="capture center-ROI crops from camera")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--compute", action="store_true", help="compute threshold from memory leave-one-out")
    args = p.parse_args(argv)

    if args.capture:
        capture_crops(args.capture, args.n)
        return 0

    if args.compute or not args.capture:
        pos, neg = load_pairs_from_memory()
        rep = suggest_threshold(pos, neg)
        import json

        print(json.dumps(rep, indent=2, ensure_ascii=False))
        thr = rep.get("conservative_threshold") or rep.get("suggested_threshold")
        if thr:
            print(
                f"\n→ update config/default.yaml  memory.match_threshold: {thr}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
