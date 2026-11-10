#!/usr/bin/env python
"""Batch entry point — enroll / recognize local images (classroom release).

The live camera demo runs through the Web UI:

    python web/server.py            # then open http://127.0.0.1:5000

Batch mode (no camera, no UI, no microphone):

    python main.py --enroll 水杯 --images data/enroll/cup
    python main.py --recognize data/query/test.png
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from vision.image_io import imread_unicode, imwrite_unicode


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ("PIL", "urllib3", "matplotlib"):
        logging.getLogger(name).setLevel(logging.WARNING)


_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _collect_images(directory: str) -> list[str]:
    paths = []
    for fn in sorted(os.listdir(directory)):
        if fn.lower().endswith(_IMG_EXTS):
            paths.append(os.path.join(directory, fn))
    return paths


def _run_batch_enroll(cfg: dict, name: str, images_dir: str) -> int:
    """Enroll object from a folder of local images, then exit (no camera/UI)."""
    from vision.feature_extractor import ResNet18Extractor
    from vision.memory import VisualMemory

    if not os.path.isdir(images_dir):
        print(f"[err] image directory not found: {images_dir}", file=sys.stderr)
        return 2
    paths = _collect_images(images_dir)
    if not paths:
        print(f"[err] no images in {images_dir}", file=sys.stderr)
        return 2

    resnet = ResNet18Extractor(cfg.get("resnet18", {}))
    resnet.load()
    memory = VisualMemory(cfg.get("memory", {}))
    try:
        min_samples = int(cfg.get("memory", {}).get("min_samples", 3))
        samples_dir = cfg.get("memory", {}).get("samples_dir", "data/samples")
        from core.app import _safe_name

        obj_dir = os.path.join(samples_dir, _safe_name(name))
        os.makedirs(obj_dir, exist_ok=True)
        embeds = []
        saved = []
        for src in paths:
            img = imread_unicode(src)
            if img is None:
                print(f"[skip] unreadable: {src}")
                continue
            emb = resnet.embed(img)
            if emb is None:
                print(f"[skip] embed failed: {src}")
                continue
            dest = os.path.join(obj_dir, f"s{len(embeds):02d}.png")
            imwrite_unicode(dest, img)
            embeds.append(emb)
            saved.append(dest)
        if len(embeds) < min_samples:
            print(
                f"[err] only {len(embeds)} valid embeddings (need >= {min_samples})",
                file=sys.stderr,
            )
            return 1
        memory.add_object(name, None, embeds, saved)
        st = memory.stats()
        print(f"[ok] enrolled '{name}' with {len(embeds)} images | bank={st}")
        return 0
    finally:
        memory.close()


def _run_batch_recognize(cfg: dict, image_path: str) -> int:
    """Recognize one local image against the feature bank, then exit."""
    from vision.feature_extractor import ResNet18Extractor
    from vision.memory import VisualMemory

    if not os.path.isfile(image_path):
        print(f"[err] image not found: {image_path}", file=sys.stderr)
        return 2

    img = imread_unicode(image_path)
    if img is None:
        print(f"[err] unreadable image: {image_path}", file=sys.stderr)
        return 2

    resnet = ResNet18Extractor(cfg.get("resnet18", {}))
    resnet.load()
    memory = VisualMemory(cfg.get("memory", {}))
    try:
        emb = resnet.embed(img)
        if emb is None:
            print("[err] feature extraction failed", file=sys.stderr)
            return 1
        m = memory.best_match(emb)
        thr = memory.threshold
        if m is None:
            print("result: 未知物体 (feature bank empty)")
            return 0
        pct = m.score * 100
        if m.score >= thr:
            print(f"result: {m.user_name}  similarity={pct:.1f}%  threshold={thr*100:.0f}%  [known]")
            return 0
        print(
            f"result: 未知物体  best_candidate={m.user_name}  "
            f"similarity={pct:.1f}%  threshold={thr*100:.0f}%  [unknown]"
        )
        return 0
    finally:
        memory.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="ResNet18 批量录入 / 识别（课堂展示版；实时演示请用 python web/server.py）"
    )
    p.add_argument("--config", default=None, help="path to YAML config (default: config/demo.yaml)")
    p.add_argument(
        "--enroll",
        default=None,
        metavar="NAME",
        help="batch: enroll object NAME from --images directory, then exit",
    )
    p.add_argument("--images", default=None, metavar="DIR", help="image dir for --enroll")
    p.add_argument(
        "--recognize",
        default=None,
        metavar="IMAGE",
        help="batch: recognize one local image, then exit",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(root)

    _setup_logging(args.verbose)

    missing = []
    for mod in ("cv2", "yaml", "onnxruntime"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"[env] 缺少依赖: {', '.join(missing)}；请先 pip install -r requirements.txt", file=sys.stderr)

    from core.config import load_config

    cfg = load_config(args.config, None)

    if args.enroll:
        if not args.images:
            print("[err] --enroll requires --images DIR", file=sys.stderr)
            return 2
        return _run_batch_enroll(cfg, args.enroll, args.images)
    if args.recognize:
        return _run_batch_recognize(cfg, args.recognize)

    p.print_help()
    print("\n实时摄像头演示请运行: python web/server.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
