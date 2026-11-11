"""Web UI for the offline vision + voice assistant.

Flask + MJPEG streaming: camera feed with overlays + depth heatmap in browser,
text command input, live status display, host camera/mic device picker.

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
import tempfile
import threading
import time
from typing import Optional

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from vision.image_io import imread_unicode
from core import devices

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


def _mjpeg_part(jpeg: bytes) -> bytes:
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"


_MJPEG_MIME = "multipart/x-mixed-replace; boundary=frame"


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

        # draw highlight box (match result / depth candidate)
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
            yield _mjpeg_part(data)
        time.sleep(0.033)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(_mjpeg_gen("_latest_jpeg"), mimetype=_MJPEG_MIME)


@app.route("/depth")
def depth():
    return Response(_mjpeg_gen("_latest_depth_jpeg"), mimetype=_MJPEG_MIME)


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
    expires, width, height, boxes = _assistant._overlay.candidate_snapshot
    return jsonify({
        "ready": True,
        **_assistant.current_devices,
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
        "candidate_boxes": {"width": width, "height": height,
                            "boxes": [{"box": box, "name": name, "score": score, "reason": reason}
                                      for box, name, score, reason in boxes]}
        if time.time() < expires and _assistant.camera_enabled and _assistant.recognition_enabled else None,
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


# ------------------------------------------------------------- device picker
@app.route("/devices", methods=["GET"])
def devices_list():
    """Enumerate devices attached to the machine running the server.

    Camera probing is slow (each DirectShow open() ~1 s), so it runs in a
    background thread; the browser polls until ``scanning`` becomes false.
    ``?refresh=1`` forces a fresh scan; ``?all_mics=1`` returns every host API
    of every microphone instead of the de-duplicated WASAPI view.
    """
    if _assistant is None:
        return jsonify({"ready": False, "cameras": [], "microphones": [], "scanning": False})
    cur = _assistant.current_devices
    force = request.args.get("refresh") in ("1", "true", "yes")
    all_mics = request.args.get("all_mics") in ("1", "true", "yes")
    devices.start_camera_scan(cur["camera_index"], force=force)
    snap = devices.camera_snapshot()
    mic = devices.mic_catalog(cur["mic_device"], all_hostapis=all_mics)
    return jsonify({
        "ready": True,
        "cameras": snap["cameras"],
        "camera_info": {"named": snap["named"], "unavailable": snap["unavailable"]},
        "microphones": mic["devices"],
        "mic_info": {k: mic[k] for k in ("preferred_host", "total", "hidden", "compact")},
        "scanning": snap["scanning"],
        "scan_error": snap["scan_error"],
        "camera_enabled": _assistant.camera_enabled,
        **cur,
    })


@app.route("/devices", methods=["POST"])
def devices_apply():
    """Apply the selected camera index and/or microphone device."""
    global _latest_jpeg, _latest_depth_jpeg
    if _assistant is None:
        return jsonify({"ok": False, "errors": {"app": "assistant not ready"}}), 503
    data = request.get_json(silent=True) or {}
    errors: dict = {}

    if "camera" in data:
        # A live preview holds an exclusive capture on the candidate index;
        # free it before the real camera tries to open the same device.
        devices.stop_previews()
        try:
            idx = int(data["camera"])
        except (TypeError, ValueError):
            errors["camera"] = "摄像头索引无效"
        else:
            if not 0 <= idx <= 64:
                errors["camera"] = "摄像头索引超出范围（0–64）"
            elif not _assistant.set_camera_device(idx):
                label = devices.camera_name(idx)
                errors["camera"] = f"无法打开 {label}（索引 {idx}），已恢复原设备"
            else:
                # Old camera's frozen frame must not linger in the MJPEG stream.
                with _jpeg_lock:
                    _latest_jpeg = None
                    _latest_depth_jpeg = None

    if "mic" in data:
        raw = data["mic"]
        if raw is None or raw == "" or str(raw).lower() == "default":
            chosen = None                      # system default input
        else:
            try:
                chosen = int(raw)              # sounddevice index
            except (TypeError, ValueError):
                chosen = str(raw)[:120]        # device name substring
        if not _assistant.set_mic_device(chosen):
            errors["mic"] = "无法打开该麦克风，已恢复原设备"

    return jsonify({
        "ok": not errors,
        "errors": errors,
        **_assistant.current_devices,
        "camera_enabled": _assistant.camera_enabled,
        "mic_enabled": _assistant.mic_enabled,
    })


def _parse_camera_index() -> Optional[int]:
    try:
        idx = int(request.args.get("index", 0))
    except (TypeError, ValueError):
        return None
    return idx if 0 <= idx <= 64 else None


@app.route("/devices/preview/stream")
def devices_preview_stream():
    """Live MJPEG preview for the device picker (not a frozen snapshot).

    For the camera already in use we reuse the main frame buffer. For a
    candidate index a dedicated capture is opened and streamed until the
    browser disconnects, another preview starts, or /preview/stop is called.
    """
    idx = _parse_camera_index()
    if idx is None:
        return jsonify({"error": "摄像头索引无效"}), 400

    # Reuse the main frame buffer for the active camera (a second DSHOW open()
    # on the same device usually fails). If no frame is available -- e.g. the
    # camera is switched off, or it was just re-opened -- fall through to a
    # dedicated capture rather than leaving the browser waiting forever.
    use_live = False
    if (_assistant is not None and idx == _assistant.camera.index
            and _assistant.camera_enabled):
        for _ in range(10):
            with _jpeg_lock:
                data = _latest_jpeg
            if data:
                use_live = True
                break
            time.sleep(0.06)

    if use_live:
        devices.stop_previews()
        token = devices.current_preview_token()

        def live_gen():
            while True:
                if devices.current_preview_token() != token:
                    return
                with _jpeg_lock:
                    data = _latest_jpeg
                if data:
                    yield _mjpeg_part(data)
                time.sleep(0.033)

        return Response(live_gen(), mimetype=_MJPEG_MIME,
                        headers={"Cache-Control": "no-store"})

    try:
        frames = devices.begin_preview(idx)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503

    def preview_gen():
        try:
            for jpeg in frames:
                yield _mjpeg_part(jpeg)
        finally:
            frames.close()   # releases the capture even on early disconnect

    return Response(preview_gen(), mimetype=_MJPEG_MIME,
                    headers={"Cache-Control": "no-store"})


@app.route("/devices/preview/stop", methods=["POST"])
def devices_preview_stop():
    """Release the preview capture when the modal is closed."""
    devices.stop_previews()
    return jsonify({"ok": True})


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
    cfg = _assistant.cfg.get("memory", {})
    return jsonify({"objects": items, "min_samples": int(cfg.get("min_samples", 3)),
                    "max_samples": int(cfg.get("max_samples", 10))})


@app.route("/objects/import", methods=["POST"])
def objects_import():
    if _assistant is None:
        return jsonify({"ok": False, "error": "系统尚未就绪。"}), 503
    if request.content_length and request.content_length > 50 * 1024 * 1024:
        return jsonify({"ok": False, "error": "图片总大小不能超过 50 MB。"}), 413
    label = (request.form.get("name") or "").strip()
    files = request.files.getlist("images")
    cfg = _assistant.cfg.get("memory", {})
    minimum, maximum = int(cfg.get("min_samples", 3)), int(cfg.get("max_samples", 10))
    if not label or len(label) > 80:
        return jsonify({"ok": False, "error": "请输入不超过80个字符的物体名称。"}), 400
    if not minimum <= len(files) <= maximum:
        return jsonify({"ok": False, "error": f"请选择 {minimum} 到 {maximum} 张图片。"}), 400
    try:
        with tempfile.TemporaryDirectory(prefix="object-upload-") as temp_dir:
            paths = []
            for i, upload in enumerate(files):
                content = upload.stream.read(10 * 1024 * 1024 + 1)
                if len(content) > 10 * 1024 * 1024:
                    return jsonify({"ok": False, "error": "单张图片不能超过 10 MB。"}), 413
                path = os.path.join(temp_dir, f"{i}.image")
                with open(path, "wb") as f:
                    f.write(content)
                paths.append(path)
            message = _assistant.enroll_image_files(label, paths)
        return jsonify({"ok": True, "message": message})
    except FileExistsError as e:
        return jsonify({"ok": False, "error": str(e)}), 409
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 503
    except Exception:
        log.exception("Image enrollment failed")
        return jsonify({"ok": False, "error": "图片录入失败，请检查服务日志。"}), 500


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
    p.add_argument("--camera-index", type=int, default=None,
                   help="初始摄像头索引（也可在网页「设备设置」里切换）")
    p.add_argument("--mic-device", default=None,
                   help="初始麦克风：sounddevice 索引或名称；不填=系统默认")
    args = p.parse_args()

    from core.config import load_config

    cfg = load_config(args.config)
    if args.camera_index is not None:
        cfg.setdefault("app", {})["camera_index"] = args.camera_index
    if args.mic_device is not None:
        try:
            mic = int(args.mic_device)
        except ValueError:
            mic = args.mic_device
        cfg.setdefault("asr", {})["mic_device"] = mic

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
