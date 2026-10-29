"""Web UI for the offline vision + voice assistant.

Flask + MJPEG streaming: camera feed with overlays + depth heatmap in browser,
text command input, live status display.

Usage:
    python web/server.py --config config/demo.yaml
    Then open http://localhost:5000
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from vision.image_io import imread_unicode

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
log = logging.getLogger("web")
# The browser polls /status + /chat a few times per second — logging every
# request filled the log with ~1MB of noise and cost real I/O. Keep errors only.
logging.getLogger("werkzeug").setLevel(logging.WARNING)

from flask import Flask, Response, jsonify, render_template, request, send_file

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))

# global app instance (set by main)
_assistant = None
_latest_jpeg: Optional[bytes] = None
_latest_depth_jpeg: Optional[bytes] = None
_jpeg_lock = threading.Lock()
_status_cache: dict = {}


def _encode_jpeg(img: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def _frame_loop():
    """Continuously stream raw camera frame + depth heatmap to JPEG buffers.

    Uses cv2 drawing only (no PIL) for minimal latency.
    """
    global _latest_jpeg, _latest_depth_jpeg

    a = _assistant
    if a is None:
        return

    fps_ema = 0.0
    last_t = time.perf_counter()
    # stream resolution (encode cost drops ~6x vs 1080p; browser scales it back up)
    VW, VH = 960, 540
    DW, DH = 640, 360
    # depth heatmap only changes at ~2 Hz — cache its JPEG, never re-encode per frame
    last_depth_obj = None
    last_depth_jpeg = None
    target_dt = 1.0 / float(os.environ.get("WEB_FPS", "30"))

    while a._running:
        loop_t0 = time.perf_counter()
        # Camera released → stop grabbing + JPEG encoding entirely. Encoding the
        # same frozen frame 30x/s would burn CPU for nothing.
        if not a.camera_enabled:
            time.sleep(0.05)
            continue
        frame, _ = a.camera.latest()
        if frame is None:
            time.sleep(0.01)
            continue

        # track FPS for status display
        now_t = time.perf_counter()
        dt = now_t - last_t
        last_t = now_t
        if dt > 0:
            inst = 1.0 / dt
            fps_ema = inst if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst
        a.stats.ui_fps = fps_ema

        # downscale FIRST, then draw + encode (1080p JPEG encode costs ~12ms vs ~2ms)
        fh, fw = frame.shape[:2]
        if (fw, fh) != (VW, VH):
            display = cv2.resize(frame, (VW, VH), interpolation=cv2.INTER_AREA)
            sx, sy = VW / fw, VH / fh
        else:
            display = frame
            sx, sy = 1.0, 1.0

        # draw detections (cv2 only, fast)
        with a._det_lock:
            dets = list(a._detections)
        for d in dets:
            bx1, by1, bx2, by2 = d.xyxy
            x1, y1 = int(bx1 * sx), int(by1 * sy)
            x2, y2 = int(bx2 * sx), int(by2 * sy)
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 180, 60), 2)
            cv2.putText(display, d.label, (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 60), 1, cv2.LINE_AA)

        # draw highlight box (match result)
        if a._overlay.highlight and time.time() < a._flash_until:
            for box, label, conf in a._overlay.highlight:
                x1 = int(box[0] * sx); y1 = int(box[1] * sy)
                x2 = int(box[2] * sx); y2 = int(box[3] * sy)
                cv2.rectangle(display, (x1, y1), (x2, y2), (80, 220, 80), 2)
                cv2.putText(display, f"{label} {conf:.2f}", (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 80), 2, cv2.LINE_AA)

        # subtitle bar at top
        sub = a._last_subtitle if time.time() < a._subtitle_until else ""
        if sub:
            cv2.rectangle(display, (0, 0), (VW, 30), (30, 30, 30), -1)
            cv2.putText(display, sub[:70], (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

        with _jpeg_lock:
            _latest_jpeg = _encode_jpeg(display, quality=72)

        # depth heatmap — encode ONLY when it actually changed (2 Hz, not 30 Hz)
        if a.depth_est is not None and a.depth_est.ready:
            with a._depth_lock:
                heatmap = a._depth_heatmap
                box = a._depth_box
            if heatmap is not None and heatmap is not last_depth_obj:
                h2, w2 = heatmap.shape[:2]
                if (w2, h2) != (DW, DH):
                    d_disp = cv2.resize(heatmap, (DW, DH), interpolation=cv2.INTER_AREA)
                    dsx, dsy = DW / w2, DH / h2
                else:
                    d_disp = heatmap
                    dsx, dsy = 1.0, 1.0
                if box is not None:
                    x1 = int(box[0] * dsx); y1 = int(box[1] * dsy)
                    x2 = int(box[2] * dsx); y2 = int(box[3] * dsy)
                    cv2.rectangle(d_disp, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.putText(d_disp, "nearest", (x1, max(18, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                last_depth_jpeg = _encode_jpeg(d_disp, quality=72)
                last_depth_obj = heatmap
                with _jpeg_lock:
                    _latest_depth_jpeg = last_depth_jpeg

        # adaptive pacing to target FPS instead of a fixed sleep
        spent = time.perf_counter() - loop_t0
        time.sleep(max(0.002, target_dt - spent))


def _mjpeg_gen(jpeg_attr: str):
    """MJPEG stream generator."""
    while True:
        with _jpeg_lock:
            data = globals()[jpeg_attr]
        if data:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
        time.sleep(0.033)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(_mjpeg_gen("_latest_jpeg"), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/depth")
def depth():
    return Response(_mjpeg_gen("_latest_depth_jpeg"), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/command", methods=["POST"])
def command():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "empty"}), 400
    if _assistant is None:
        return jsonify({"error": "not ready"}), 503
    _assistant.inject_text(text)
    return jsonify({"ok": True, "sent": text})


@app.route("/status")
def status():
    if _assistant is None:
        return jsonify({"ready": False})
    s = _assistant.stats
    return jsonify({
        "ready": True,
        "fps": round(s.ui_fps, 1),
        "mode": s.mode,
        "last_intent": s.last_intent,
        "last_reply": s.last_reply,
        "last_similarity": round(s.last_similarity, 3),
        "mem_objects": s.mem_objects,
        "mem_info": s.mem_info,
        "det_info": s.det_info,
        "subtitle": _assistant._last_subtitle if time.time() < _assistant._subtitle_until else "",
        "resnet18": _assistant.resnet.device_used if _assistant.resnet else "off",
        "depth": _assistant.depth_est.device_used if _assistant.depth_est else "off",
        "asr_latency_ms": round(s.asr_latency_ms, 0),
        "voice_enabled": _assistant.use_voice,
        "recognition_enabled": _assistant.recognition_enabled,
        "camera_enabled": _assistant.camera_enabled,
        "mic_enabled": _assistant.mic_enabled,
    })


@app.route("/recognition", methods=["POST"])
def recognition_toggle():
    """Start or pause the recognition pipeline (frees CPU-heavy resources)."""
    if _assistant is None:
        return jsonify({"ok": False, "error": "assistant not ready"}), 503
    data = request.get_json(silent=True) or {}
    enabled = data.get("enabled")
    if enabled is None:
        enabled = not _assistant.recognition_enabled
    state = _assistant.set_recognition_enabled(bool(enabled))
    return jsonify({"ok": True, "enabled": state,
                    "camera_enabled": _assistant.camera_enabled,
                    "mic_enabled": _assistant.mic_enabled})


@app.route("/camera", methods=["POST"])
def camera_toggle():
    """Start or release the camera."""
    if _assistant is None:
        return jsonify({"ok": False, "error": "assistant not ready"}), 503
    data = request.get_json(silent=True) or {}
    enabled = data.get("enabled")
    if enabled is None:
        enabled = not _assistant.camera_enabled
    state = _assistant.set_camera_enabled(bool(enabled))
    return jsonify({"ok": True, "camera_enabled": state})


@app.route("/mic", methods=["POST"])
def mic_toggle():
    """Start or release the microphone."""
    if _assistant is None:
        return jsonify({"ok": False, "error": "assistant not ready"}), 503
    data = request.get_json(silent=True) or {}
    enabled = data.get("enabled")
    if enabled is None:
        enabled = not _assistant.mic_enabled
    state = _assistant.set_mic_enabled(bool(enabled))
    return jsonify({"ok": True, "mic_enabled": state})


@app.route("/chat")
def chat():
    """Return new chat messages since the given timestamp."""
    since = float(request.args.get("since", 0))
    if _assistant is None:
        return jsonify({"messages": [], "now": time.time()})
    msgs = _assistant.get_chat_since(since)
    return jsonify({"messages": msgs, "now": time.time()})


@app.route("/objects")
def objects():
    if _assistant is None:
        return jsonify({"objects": []})
    objs = _assistant.memory.list_objects()
    items = []
    for o in objs:
        n = len(_assistant.memory.sample_images(o.user_name))
        items.append({"name": o.user_name, "samples": getattr(o, "sample_count", 0),
                      "images": n})
    return jsonify({"objects": items})


@app.route("/objects/sample")
def objects_sample():
    """Serve one enrolled sample image (for verifying what was recorded)."""
    if _assistant is None:
        return jsonify({"ok": False}), 503
    name = (request.args.get("name") or "").strip()
    idx = int(request.args.get("index") or 0)
    if not name:
        return jsonify({"ok": False}), 400
    paths = _assistant.memory.sample_images(name)
    if not paths:
        return jsonify({"ok": False, "error": "no sample images"}), 404
    path = paths[min(max(idx, 0), len(paths) - 1)]
    img = imread_unicode(path)
    if img is None:
        return jsonify({"ok": False, "error": "unreadable"}), 404
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        return jsonify({"ok": False}), 500
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")


@app.route("/objects/delete", methods=["POST"])
def objects_delete():
    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()
    if not name or _assistant is None:
        return jsonify({"ok": False}), 400
    ok = _assistant.memory.delete_object(name)
    st = _assistant.memory.stats()
    _assistant.stats.mem_objects = st["objects"]
    _assistant._refresh_mem_info()
    return jsonify({"ok": ok})


@app.route("/objects/rename", methods=["POST"])
def objects_rename():
    data = request.get_json(force=True)
    old = (data.get("old_name") or "").strip()
    new = (data.get("new_name") or "").strip()
    if not old or not new or _assistant is None:
        return jsonify({"ok": False, "error": "empty name"}), 400
    ok = _assistant.memory.rename_object(old, new)
    if ok:
        _assistant.push_chat("sys", f"已重命名「{old}」→「{new}」")
        _assistant._refresh_mem_info()
    else:
        return jsonify({"ok": False, "error": "name missing or already taken"})
    return jsonify({"ok": True})


def main():
    global _assistant
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--no-voice", action="store_true")
    p.add_argument("--no-camera", action="store_true",
                   help="不打开摄像头（稍后可在网页上开启）")
    p.add_argument("--no-mic", action="store_true",
                   help="不打开麦克风（稍后可在网页上开启）")
    args = p.parse_args()

    import yaml
    cfg_path = args.config or os.path.join(ROOT, "config", "default.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # web mode: no OpenCV windows
    from core.app import AssistantApp
    assistant = AssistantApp(cfg, headless=True, use_voice=not args.no_voice,
                             start_camera=not args.no_camera,
                             start_mic=not args.no_mic)
    _assistant = assistant

    assistant.start()

    # start frame rendering thread
    threading.Thread(target=_frame_loop, name="web-frames", daemon=True).start()

    log.info("Web UI ready → http://%s:%d", args.host, args.port)
    try:
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        assistant.stop()


if __name__ == "__main__":
    main()
