#!/usr/bin/env python
"""Test depth-guided zero-shot localization on real images.

Runs Depth Anything V2 + nearest-object detection on test scene images,
saves: original with bbox overlay + depth heatmap side-by-side.

Usage:
    python tools/test_depth_localization.py
    python tools/test_depth_localization.py --image data/test_scenes/cup_scene_00.jpg
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=None, help="single image path")
    p.add_argument("--dir", default="data/test_scenes", help="directory of test images")
    p.add_argument("--out", default="data/depth_test_results", help="output directory")
    args = p.parse_args()

    # load config
    import yaml
    with open("config/default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    depth_cfg = cfg.get("depth", {})

    # init depth estimator (without MobileSAM for this test)
    from vision.depth import DepthEstimator
    depth_cfg["use_mobilesam"] = False  # test depth-only path
    est = DepthEstimator(depth_cfg)
    print(f"Loading depth model: {depth_cfg.get('model_path')}")
    est.load()
    print(f"Loaded: {est.device_used}")

    # collect images
    if args.image:
        images = [args.image]
    else:
        exts = (".jpg", ".jpeg", ".png")
        images = sorted(
            os.path.join(args.dir, f)
            for f in os.listdir(args.dir)
            if f.lower().endswith(exts)
        )
    print(f"Testing {len(images)} images\n")

    os.makedirs(args.out, exist_ok=True)

    for img_path in images:
        name = os.path.splitext(os.path.basename(img_path))[0]
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[skip] cannot read {img_path}")
            continue

        h, w = frame.shape[:2]
        t0 = time.perf_counter()

        # 1) depth estimation
        depth = est.estimate_depth(frame)
        t_depth = (time.perf_counter() - t0) * 1000

        # 2) nearest object localization
        t1 = time.perf_counter()
        region = est.find_nearest_object(frame, depth)
        t_locate = (time.perf_counter() - t1) * 1000

        # 3) full crop pipeline
        t2 = time.perf_counter()
        result = est.crop_nearest(frame)
        t_crop = (time.perf_counter() - t2) * 1000

        total = (time.perf_counter() - t0) * 1000

        # build visualization
        heatmap = DepthEstimator.depth_to_colormap(depth)
        vis = frame.copy()

        if region is not None:
            x1, y1, x2, y2 = region.box
            # draw on original
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"score={region.score:.2f} depth={region.mean_depth:.2f} area={region.area_frac:.1%}"
            cv2.putText(vis, label, (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
            # draw on heatmap too
            cv2.rectangle(heatmap, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.putText(heatmap, "nearest object", (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # resize for side-by-side
        scale = 800 / w
        nh, nw = int(h * scale), int(w * scale)
        vis_s = cv2.resize(vis, (nw, nh))
        heat_s = cv2.resize(heatmap, (nw, nh))
        combo = np.hstack([vis_s, heat_s])

        # timing text
        info = f"{name}: depth={t_depth:.0f}ms locate={t_locate:.0f}ms total={total:.0f}ms"
        cv2.putText(combo, info, (10, nh - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        out_path = os.path.join(args.out, f"{name}_result.jpg")
        cv2.imwrite(out_path, combo)
        print(f"[ok] {name}: {info}")
        print(f"     saved → {out_path}")

        if region is not None:
            print(f"     box={region.box} score={region.score:.3f} "
                  f"mean_depth={region.mean_depth:.3f} area={region.area_frac:.1%}")
        else:
            print(f"     NO OBJECT FOUND")
        print()

    print(f"\nAll results in: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
