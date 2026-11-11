"""Host device discovery for the Web UI device picker.

The camera and microphone are opened by the Python process, not by the browser,
so the settings panel needs *server-side* enumeration:

* cameras     -- the authoritative device list comes from the Windows
                 DirectShow device enumerator (friendly names, index order
                 matching cv2's CAP_DSHOW) without opening any device. OpenCV
                 probing only annotates resolution / current availability, so
                 an exclusive webcam being busy can no longer make a device
                 disappear from the list. Scanning runs in a background thread
                 and is cached; the picker can also open a live MJPEG capture.
* microphones -- sounddevice index + real name, collapsed to the preferred
                 host API so one physical mic is not listed four times.

Every helper degrades gracefully when a backend is unavailable (headless CI,
``--no-camera`` / ``--no-mic``): it returns an empty list instead of raising.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import uuid
from typing import Any, Optional

import cv2

log = logging.getLogger(__name__)

# Most machines have 1-3 cameras. Probing further rarely finds anything and
# each DSHOW open() is slow, so keep the scan bounded.
MAX_CAMERA_INDEX = 8
SCAN_CACHE_SEC = 120.0

_scan_lock = threading.Lock()
_scan: dict[str, Any] = {
    "cameras": [],
    "scanning": False,
    "ts": 0.0,
    "error": "",
}

# Only one live preview capture may exist at a time: DirectShow webcams are
# usually exclusive, and "apply" needs the target device to be free. A token
# invalidates older streams; each generator releases its own capture in
# ``finally`` and signals an Event so callers can wait instead of sleeping.
_preview_lock = threading.Lock()
_preview_token = 0
_preview_released: Optional[threading.Event] = None

_name_lock = threading.Lock()
_camera_names: list[str] = []


def directshow_camera_names() -> list[str]:
    """Friendly names of Windows DirectShow video inputs, in cv2 index order.

    cv2's CAP_DSHOW backend builds its index list by walking
    ICreateDevEnum / IEnumMoniker over CLSID_VideoInputDeviceCategory, so
    ``names[i]`` is the device behind ``cv2.VideoCapture(i, CAP_DSHOW)``.
    This pure-ctypes walk never opens a device, so an exclusive webcam being
    busy cannot hide itself. Returns [] on non-Windows or any COM failure.
    """
    if sys.platform != "win32":
        return []
    try:
        import ctypes
        from ctypes import POINTER, byref, c_void_p, c_wchar_p, wintypes

        class _GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                ("Data3", wintypes.WORD), ("Data4", ctypes.c_ubyte * 8),
            ]

        class _VARIANT(ctypes.Structure):
            _fields_ = [
                ("vt", wintypes.WORD), ("r1", wintypes.WORD),
                ("r2", wintypes.WORD), ("r3", wintypes.WORD),
                ("bstrVal", c_wchar_p),
            ]

        def guid(text: str) -> _GUID:
            return _GUID.from_buffer_copy(uuid.UUID(text).bytes_le)

        def vfunc(ptr, slot, restype, *argtypes):
            table = ctypes.cast(ptr, POINTER(POINTER(c_void_p))).contents
            proto = ctypes.WINFUNCTYPE(restype, c_void_p, *argtypes)
            return ctypes.cast(table[slot], proto)

        clsid_devenum = guid("62BE5D10-60EB-11D0-BD3B-00A0C911CE86")
        iid_create = guid("29840822-5B84-11d0-BD3B-00A0C911CE86")
        clsid_video = guid("860BB310-5D01-11d0-BD3B-00A0C911CE86")
        iid_propbag = guid("55272A00-42CB-11CE-8135-00AA004BB851")

        ole32 = ctypes.windll.ole32
        oleaut = ctypes.windll.oleaut32
        ole32.CoInitializeEx.argtypes = [c_void_p, wintypes.DWORD]
        ole32.CoInitializeEx.restype = ctypes.c_long
        ole32.CoCreateInstance.argtypes = [
            POINTER(_GUID), c_void_p, wintypes.DWORD,
            POINTER(_GUID), POINTER(c_void_p),
        ]
        ole32.CoCreateInstance.restype = ctypes.c_long
        oleaut.VariantClear.argtypes = [POINTER(_VARIANT)]

        hr = ole32.CoInitializeEx(None, 0)      # COINIT_MULTITHREADED
        owns_com = hr in (0, 1)                 # S_OK / S_FALSE
        out: list[str] = []
        try:
            devenum = c_void_p()
            hr = ole32.CoCreateInstance(
                byref(clsid_devenum), None, 1,   # CLSCTX_INPROC_SERVER
                byref(iid_create), byref(devenum),
            )
            if hr != 0 or not devenum.value:
                return out
            try:
                enum = c_void_p()
                create_enum = vfunc(
                    devenum, 3, ctypes.c_long,
                    POINTER(_GUID), POINTER(c_void_p), wintypes.DWORD,
                )
                if (create_enum(devenum, byref(clsid_video), byref(enum), 0) != 0
                        or not enum.value):
                    return out
                try:
                    next_moniker = vfunc(
                        enum, 3, ctypes.c_long, wintypes.ULONG,
                        POINTER(c_void_p), POINTER(wintypes.ULONG),
                    )
                    while True:
                        moniker = c_void_p()
                        fetched = wintypes.ULONG()
                        if (next_moniker(enum, 1, byref(moniker), byref(fetched)) != 0
                                or not fetched.value or not moniker.value):
                            break
                        try:
                            bag = c_void_p()
                            bind = vfunc(
                                moniker, 9, ctypes.c_long, c_void_p, c_void_p,
                                POINTER(_GUID), POINTER(c_void_p),
                            )
                            name = ""
                            if (bind(moniker, None, None, byref(iid_propbag), byref(bag)) == 0
                                    and bag.value):
                                try:
                                    read = vfunc(
                                        bag, 3, ctypes.c_long, c_wchar_p,
                                        POINTER(_VARIANT), c_void_p,
                                    )
                                    var = _VARIANT()
                                    if read(bag, "FriendlyName", byref(var), None) == 0:
                                        # VT_BSTR == 8; never dereference a
                                        # non-string VARIANT as a wchar*.
                                        if var.vt == 8:
                                            name = var.bstrVal or ""
                                        oleaut.VariantClear(byref(var))
                                finally:
                                    vfunc(bag, 2, ctypes.c_ulong)(bag)
                            out.append(name.strip() or f"摄像头 {len(out)}")
                        finally:
                            vfunc(moniker, 2, ctypes.c_ulong)(moniker)
                finally:
                    vfunc(enum, 2, ctypes.c_ulong)(enum)
            finally:
                vfunc(devenum, 2, ctypes.c_ulong)(devenum)
        finally:
            if owns_com:
                ole32.CoUninitialize()
        return out
    except Exception as e:
        log.debug("DirectShow device enumeration failed: %s", e)
        return []


def camera_name(index: int) -> str:
    """Friendly name for an index, populated on first use and refreshed by scans.

    /status is polled twice a second, so the COM walk is cached rather than
    repeated; a scan replaces it when devices are (re)enumerated.
    """
    global _camera_names
    with _name_lock:
        if not _camera_names:
            fresh = directshow_camera_names()
            if fresh:
                _camera_names = fresh
        if 0 <= int(index) < len(_camera_names):
            return _camera_names[int(index)] or f"摄像头 {index}"
    return f"摄像头 {index}"


def _backend() -> int:
    """DirectShow on Windows, OpenCV's default elsewhere."""
    return int(getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY))


def open_camera(index: int):
    """Open a capture for a short probe. Caller must release it."""
    cap = cv2.VideoCapture(int(index), _backend())
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(int(index))
    return cap


def probe_camera(index: int, want_frame: bool = False, name: str = ""):
    """Briefly open index, grab one frame, release.

    Returns ``(info | None, frame | None)``. ``info`` is the JSON payload for
    the dropdown; ``frame`` is only returned when ``want_frame`` is set.
    A failed probe no longer means "no device" -- the DirectShow name list is
    authoritative, and a busy exclusive webcam simply cannot be probed.
    """
    cap = None
    try:
        cap = open_camera(index)
        if not cap.isOpened():
            return None, None
        ok, frame = False, None
        # The first DSHOW read often returns an empty frame while the sensor
        # warms up, so give it a second chance.
        for _ in range(2):
            ok, frame = cap.read()
            if ok and frame is not None:
                break
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if ok and frame is not None:
            w, h = int(frame.shape[1]), int(frame.shape[0])
        info = {
            "id": int(index),
            "name": name or f"摄像头 {int(index)}",
            "detail": f"{w}×{h}" if w and h else "分辨率未知",
            "readable": bool(ok and frame is not None),
            "available": True,
            "in_use": False,
        }
        return info, (frame if want_frame and ok else None)
    except Exception as e:  # broken driver / backend missing
        log.debug("camera %s probe failed: %s", index, e)
        return None, None
    finally:
        if cap is not None:
            cap.release()


def current_preview_token() -> int:
    with _preview_lock:
        return _preview_token


def stop_previews() -> None:
    """Invalidate every preview stream and wait for its capture to be freed."""
    global _preview_token
    with _preview_lock:
        _preview_token += 1
        released = _preview_released
    if released is not None:
        released.wait(timeout=1.0)


def begin_preview(index: int, max_width: int = 480, quality: int = 68):
    """Return a generator yielding *live* JPEG frames from ``index``.

    Any previous preview is stopped first. Raises RuntimeError when the device
    cannot be opened (busy / unplugged), so the HTTP layer can report it
    instead of serving a broken stream.
    """
    global _preview_token, _preview_released
    stop_previews()
    label = f"{camera_name(index)}（索引 {index}）"
    cap = open_camera(index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"{label} 当前无法打开（可能被占用）")
    # Some drivers open successfully but never deliver a frame. Verify once
    # here so the HTTP stream can never hang with zero frames; keep that first
    # frame and send it as the initial picture.
    first = None
    for _ in range(2):
        ok, candidate = cap.read()
        if ok and candidate is not None:
            first = candidate
            break
        time.sleep(0.05)
    if first is None:
        cap.release()
        raise RuntimeError(f"{label} 打开但没有画面")

    released = threading.Event()
    with _preview_lock:
        _preview_token += 1
        token = _preview_token
        _preview_released = released

    def frames():
        try:
            pending = first
            last_frame_t = time.time()
            while True:
                with _preview_lock:
                    if token != _preview_token:
                        return
                if time.time() - last_frame_t > 5.0:
                    return                      # stalled device: end cleanly
                if pending is not None:
                    frame, pending = pending, None
                else:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        time.sleep(0.03)
                        continue
                last_frame_t = time.time()
                h, w = frame.shape[:2]
                if w > max_width:
                    frame = cv2.resize(
                        frame, (max_width, max(1, int(h * max_width / w))),
                        interpolation=cv2.INTER_AREA,
                    )
                ok2, buf = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
                )
                if not ok2:
                    continue
                with _preview_lock:
                    if token != _preview_token:
                        return
                yield buf.tobytes()
        finally:
            cap.release()
            released.set()

    return frames()


def _scan_cameras(current_index: Optional[int]) -> list[dict]:
    """Stable device list = DirectShow names, annotated by OpenCV probing."""
    global _camera_names
    fresh = directshow_camera_names()
    with _name_lock:
        if fresh:
            _camera_names = fresh          # keep the last good list on a COM hiccup
        names = list(_camera_names)

    cur = int(current_index) if current_index is not None else None
    # When DirectShow gave us the authoritative device count, probing indices
    # far beyond it only produces driver warnings. Keep a small margin for
    # devices the enumerator might have missed; fall back to the wide range
    # when COM is unavailable.
    limit = max(2, len(names) + 2) if names else MAX_CAMERA_INDEX
    found: list[dict] = []
    for i in range(limit):
        named = i < len(names)
        label = names[i] if named else f"摄像头 {i}"
        if cur is not None and i == cur:
            # Never probe the device the capture thread is holding: a second
            # open() on an exclusive webcam would fail or steal frames.
            found.append({
                "id": i, "name": label, "detail": f"索引 {i} · 正在使用",
                "readable": True, "available": True, "in_use": True,
                "named": named,
            })
            continue
        info, _ = probe_camera(i, name=label)
        if info is not None:
            info["detail"] = f"索引 {i} · {info['detail']}"
            info["named"] = named
            found.append(info)
        elif named:
            # A real DirectShow device that currently refuses a second open()
            # (busy, or a virtual camera that is not running) stays in the list.
            found.append({
                "id": i, "name": label, "detail": f"索引 {i} · 当前无法打开",
                "readable": False, "available": False, "in_use": False,
                "named": True,
            })
    # A device selected from config may sit above the probe range.
    if cur is not None and not any(c["id"] == cur for c in found):
        found.insert(0, {
            "id": cur, "name": camera_name(cur), "detail": f"索引 {cur} · 正在使用",
            "readable": True, "available": True, "in_use": True,
            "named": False,
        })
    return found


def _scan_worker(current_index: Optional[int]) -> None:
    try:
        cams = _scan_cameras(current_index)
        err = ""
    except Exception as e:  # pragma: no cover - defensive
        log.warning("camera scan failed: %s", e)
        cams, err = [], f"扫描失败：{e}"
    with _scan_lock:
        _scan["cameras"] = cams
        _scan["ts"] = time.time()
        _scan["scanning"] = False
        _scan["error"] = err


def start_camera_scan(current_index: Optional[int] = None, force: bool = False) -> None:
    """Kick off a non-blocking camera scan. No-op while one is already running."""
    with _scan_lock:
        if _scan["scanning"]:
            return
        if not force and _scan["ts"] and time.time() - _scan["ts"] < SCAN_CACHE_SEC:
            return
        _scan["scanning"] = True
        _scan["error"] = ""
    threading.Thread(
        target=_scan_worker, args=(current_index,), name="camera-scan", daemon=True
    ).start()


def camera_snapshot() -> dict:
    """Last scan result (copy, safe to serialize)."""
    with _scan_lock:
        cameras = list(_scan["cameras"])
        scanning = bool(_scan["scanning"])
        scan_error = _scan["error"]
        scanned = bool(_scan["ts"])
    with _name_lock:
        known_names = len(_camera_names)
    return {
        "cameras": cameras,
        # Before the first scan finishes, /status has already primed the name
        # cache -- report it so the UI hint does not flicker.
        "named": (sum(1 for c in cameras if c.get("named"))
                  if cameras else known_names),
        "unavailable": sum(1 for c in cameras if not c.get("available", True)),
        "scanning": scanning,
        "scan_error": scan_error,
        "scanned": scanned,
    }


# sounddevice reports every physical microphone once per host API (MME,
# DirectSound, WASAPI, WDM-KS): this machine showed 13 entries for 3 real
# devices. Collapse them by normalized name, preferring WASAPI per device but
# still keeping mics that only exist on WDM-KS. "Show all" lists everything.
HOSTAPI_PRIORITY = ("Windows WASAPI", "Windows DirectSound", "MME", "Windows WDM-KS")
_ALIAS_MARKERS = (
    "sound mapper", "primary sound capture", "stereo mix", "pc speaker",
    "wave microphone", "steam streaming",
)


def _norm_mic_name(name: str) -> str:
    """'Microphone Array (AMD Audio Device)' -> 'microphone array'."""
    return " ".join(name.split("(")[0].strip().lower().split())


def _host_rank(host: str) -> int:
    low = host.lower()
    for i, p in enumerate(HOSTAPI_PRIORITY):
        if p.lower() in low:
            return i
    return len(HOSTAPI_PRIORITY)


def _enumerate_mics() -> list[dict]:
    """Raw input devices across all host APIs."""
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        host_names = {
            i: str(h.get("name", ""))
            for i, h in enumerate(sd.query_hostapis())
        }
        default_in: Optional[int] = None
        try:
            dflt = sd.default.device
            if isinstance(dflt, (list, tuple)):
                default_in = int(dflt[0]) if dflt[0] is not None else None
            elif isinstance(dflt, int):
                default_in = dflt
        except Exception:
            pass

        out: list[dict] = []
        for i, d in enumerate(devices):
            if int(d.get("max_input_channels", 0) or 0) <= 0:
                continue
            host = host_names.get(int(d.get("hostapi", 0)), "")
            name = str(d.get("name", "")).strip() or f"输入设备 {i}"
            sr = float(d.get("default_samplerate", 0) or 0)
            low = name.lower()
            out.append({
                "id": i,
                "name": name,
                "detail": f"{host} · {sr / 1000:g} kHz" if sr else host,
                "host": host,
                "is_alias": any(m in low for m in _ALIAS_MARKERS),
                "is_default": i == default_in,
                "is_current": False,
            })
        return out
    except Exception as e:
        log.warning("audio device enumeration failed: %s", e)
        return []


def mic_catalog(current: Any = None, all_hostapis: bool = False) -> dict:
    """De-duplicated microphone list plus statistics for the UI hint."""
    raw = _enumerate_mics()
    for d in raw:
        d["is_current"] = current is not None and str(current) == str(d["id"])
    real = [d for d in raw if not d["is_alias"]]
    hosts = {d["host"] for d in real}
    preferred = ""
    for candidate in HOSTAPI_PRIORITY:
        preferred = next((h for h in hosts if candidate.lower() in h.lower()), "")
        if preferred:
            break
    if not preferred and hosts:
        preferred = sorted(hosts)[0]

    if all_hostapis:
        shown = raw
    else:
        # One physical mic appears on several host APIs; keep the best-ranked
        # entry per normalized name. Restricting to a single API would hide
        # devices that only exist on WDM-KS (e.g. a Bluetooth headset).
        dedup: list[dict] = []
        seen: set[str] = set()
        for d in sorted(real, key=lambda x: (_host_rank(x["host"]), x["id"])):
            key = _norm_mic_name(d["name"])
            if key in seen:
                continue
            seen.add(key)
            dedup.append(d)
        shown = dedup
        # Never hide the device that is actually in use.
        cur = next((d for d in raw if d["is_current"]), None)
        if cur is not None and cur["id"] not in {d["id"] for d in shown}:
            shown.append(cur)

    return {
        "devices": shown,
        "preferred_host": preferred,
        "total": len(raw),
        "hidden": len(raw) - len(shown),
        "compact": not all_hostapis,
    }


def list_microphones(current: Any = None, all_hostapis: bool = False) -> list[dict]:
    """Compact input-device list (kept for callers that only need options)."""
    return mic_catalog(current, all_hostapis)["devices"]


def mic_name(device: Any) -> str:
    """Friendly label for a stored device choice, even while the stream is closed."""
    if device is None or device == "":
        return "系统默认输入设备"
    if isinstance(device, int):
        for m in _enumerate_mics():
            if m["id"] == device:
                return m["name"]
        return f"输入设备 {device}"
    return str(device)
