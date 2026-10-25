#!/usr/bin/env python
"""Entry point — ResNet18 + 语音交互物体识别 Demo (选题一).

Usage:
    python main.py                              # full UI + voice
    python main.py --no-voice                   # UI + text/keyboard commands
    python main.py --headless                   # no OpenCV window (for CI smoke)
    python main.py --camera 0 --config config/default.yaml

Batch (no camera / no UI):
    python main.py --enroll 水杯 --images data/enroll/cup      # 录入：名称 + 图片目录
    python main.py --recognize data/query/test.png             # 识别：单张本地图片

Windows (recommended — uses vision_ai conda env):
    run.bat --camera 2 --no-voice
    .\run.ps1 --camera 2 --no-voice
"""

from __future__ import annotations

import argparse
import logging
import os
import sys


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
            import cv2

            img = cv2.imread(src)
            if img is None:
                print(f"[skip] unreadable: {src}")
                continue
            emb = resnet.embed(img)
            if emb is None:
                print(f"[skip] embed failed: {src}")
                continue
            dest = os.path.join(obj_dir, f"s{len(embeds):02d}.png")
            cv2.imwrite(dest, img)
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
    import cv2

    img = cv2.imread(image_path)
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
        description="ResNet18 + 语音交互物体识别 Demo (选题一, Intel OpenVINO, no CUDA)"
    )
    p.add_argument("--config", default=None, help="path to YAML config")
    p.add_argument("--camera", type=int, default=None, help="camera index")
    p.add_argument("--no-voice", action="store_true", help="disable mic/ASR (keyboard/text input)")
    p.add_argument("--text-input", action="store_true", help="force text input window (IME) even with mic")
    p.add_argument("--headless", action="store_true", help="no GUI window")
    p.add_argument("--yolo-device", choices=["GPU", "CPU"], default=None)
    p.add_argument("--resnet-device", choices=["GPU", "CPU"], default=None)
    p.add_argument("--precision", choices=["FP32", "FP16", "INT8"], default=None)
    p.add_argument("--tts-mode", choices=["edge", "gpt_sovits", "mock"], default=None,
                   help="TTS 引擎：edge=微软在线(快,需联网) / gpt_sovits=本地离线(慢) / mock=测试")
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

    # project root on sys.path
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(root)

    _setup_logging(args.verbose)

    # wrong env → confusing "No module named openvino/onnxruntime"
    missing = []
    for mod in ("cv2", "yaml"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    try:
        __import__("openvino")
    except ImportError:
        missing.append("openvino")
    if missing:
        py = sys.executable
        if "vision_ai" not in py.replace("\\", "/"):
            print(
                f"[env] 缺少依赖: {', '.join(missing)}\n"
                f"      当前: {py}\n"
                f"      请切换环境后重试:\n"
                f"        conda activate vision_ai\n"
                f"        python main.py --camera {args.camera or 0}",
                file=sys.stderr,
            )
        else:
            print(f"[env] vision_ai 中仍缺少: {', '.join(missing)}", file=sys.stderr)

    from core.config import load_config

    overrides = {}
    if args.camera is not None:
        overrides.setdefault("app", {})["camera_index"] = args.camera
    if args.yolo_device:
        overrides.setdefault("yolo", {})["device"] = args.yolo_device
    if args.resnet_device:
        overrides.setdefault("resnet18", {})["device"] = args.resnet_device
    if args.precision:
        overrides.setdefault("yolo", {})["precision"] = args.precision
        overrides.setdefault("resnet18", {})["precision"] = args.precision
    if args.tts_mode:
        overrides.setdefault("tts", {})["mode"] = args.tts_mode
    if args.text_input:
        overrides.setdefault("app", {})["text_input"] = True
        overrides.setdefault("app", {})["text_input_always"] = True

    cfg = load_config(args.config, overrides or None)

    # ---- batch modes (no camera / no voice / no UI) ----
    if args.enroll:
        if not args.images:
            print("[err] --enroll requires --images DIR", file=sys.stderr)
            return 2
        return _run_batch_enroll(cfg, args.enroll, args.images)
    if args.recognize:
        return _run_batch_recognize(cfg, args.recognize)

    from core.app import AssistantApp

    app = AssistantApp(cfg, headless=args.headless, use_voice=not args.no_voice)
    try:
        app.start()
        app.run_ui()
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        app.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
