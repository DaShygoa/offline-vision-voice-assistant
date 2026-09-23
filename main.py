#!/usr/bin/env python
"""Entry point — offline vision voice assistant.

Usage:
    python main.py                 # full UI + voice
    python main.py --no-voice      # UI + text/keyboard commands
    python main.py --headless      # no OpenCV window (for CI smoke)
    python main.py --camera 0 --config config/default.yaml

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
    # quiet noisy libs
    for name in ("PIL", "urllib3", "matplotlib"):
        logging.getLogger(name).setLevel(logging.WARNING)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Offline OpenCV vision + voice assistant (Intel OpenVINO, no CUDA)")
    p.add_argument("--config", default=None, help="path to YAML config")
    p.add_argument("--camera", type=int, default=None, help="camera index")
    p.add_argument("--no-voice", action="store_true", help="disable mic/ASR (keyboard/text input)")
    p.add_argument("--text-input", action="store_true", help="force text input window (IME) even with mic")
    p.add_argument("--headless", action="store_true", help="no GUI window")
    p.add_argument("--yolo-device", choices=["GPU", "CPU"], default=None)
    p.add_argument("--dino-device", choices=["GPU", "CPU"], default=None)
    p.add_argument("--precision", choices=["FP32", "FP16", "INT8"], default=None)
    p.add_argument("--tts-mode", choices=["gpt_sovits", "mock"], default=None)
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
    from core.app import AssistantApp

    overrides = {}
    if args.camera is not None:
        overrides.setdefault("app", {})["camera_index"] = args.camera
    if args.yolo_device:
        overrides.setdefault("yolo", {})["device"] = args.yolo_device
    if args.dino_device:
        overrides.setdefault("dino", {})["device"] = args.dino_device
    if args.precision:
        overrides.setdefault("yolo", {})["precision"] = args.precision
    if args.tts_mode:
        overrides.setdefault("tts", {})["mode"] = args.tts_mode
    if args.text_input:
        overrides.setdefault("app", {})["text_input"] = True
        overrides.setdefault("app", {})["text_input_always"] = True

    cfg = load_config(args.config, overrides or None)
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
