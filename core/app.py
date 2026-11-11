"""Main application orchestrator — classroom release (Web UI only).

Threading model:
  ├ camera thread    : frame capture → latest() buffer (overwrite, no queue)
  ├ depth thread     : Depth Anything V2 heatmap @ ~2 Hz (paused on demand)
  ├ voice thread     : mic → VAD → Paraformer ASR → intent queue
  ├ intent worker    : identify / remember / search (ResNet18 + cosine bank)
  └ tts worker       : edge / mock synthesis (async; never blocks the worker)

The web layer (web/server.py) owns its own frame-encoding thread; AssistantApp
itself owns no GUI. All heavy inference runs off the Flask request path.

Core flow:
  locate   : depth nearest object → top-k regions → filters → crop
  identify : crop → ResNet18 512-d → centred cosine → name / 不太确定 / 未知
  remember : same crop pipeline → multi-view samples → SQLite + NumPy bank
  voice    : ASR text → rule NLU → intent queue → TTS announces the result
"""

from __future__ import annotations

import collections
import logging
import os
import queue
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from core.debug_capture import DebugCapture
from nlu.intent import Intent, IntentType, parse_intent
from vision.camera import Camera
from vision.depth import DepthEstimator
from vision.feature_extractor import ResNet18Extractor
from vision.image_io import imread_unicode, imwrite_unicode
from vision.memory import MemoryMatch, VisualMemory
from vision.object_cropper import ObjectCropper
from vision.position import describe_position
from vision.visualizer import OverlayState
from voice.asr import ParaformerASR
from voice.mic import MicStream
from voice.tts import TtsClient
from voice.vad import Vad

log = logging.getLogger(__name__)


@dataclass
class AppStats:
    ui_fps: float = 0.0
    det_fps: float = 0.0
    det_latency_ms: float = 0.0
    embed_latency_ms: float = 0.0
    asr_latency_ms: float = 0.0
    last_intent: str = ""
    last_reply: str = ""
    last_similarity: float = 0.0
    mem_objects: int = 0
    mode: str = "idle"
    tts_health: str = ""
    det_info: str = ""
    mem_info: str = ""
    recognition_enabled: bool = True
    camera_enabled: bool = True
    mic_enabled: bool = True


class AssistantApp:
    def __init__(self, cfg: dict, headless: bool = False, use_voice: bool = True,
                 start_camera: bool = True, start_mic: bool = True):
        self.cfg = cfg
        self.headless = headless
        self.use_voice = use_voice

        self.stats = AppStats()
        self._overlay = OverlayState()

        # components
        self.camera = Camera(
            index=int(cfg.get("app", {}).get("camera_index", 0)),
            width=int(cfg.get("app", {}).get("width", 1920)),
            height=int(cfg.get("app", {}).get("height", 1080)),
        )
        self.resnet = ResNet18Extractor(cfg.get("resnet18", {}))
        self.memory = VisualMemory(cfg.get("memory", {}))
        self.cropper = ObjectCropper(
            center_roi_frac=float(cfg.get("memory", {}).get("center_roi_frac", 0.45)),
            min_crop_px=int(cfg.get("memory", {}).get("min_crop_px", 64)),
        )
        # depth-guided zero-shot localization (replaces YOLO for identify/remember)
        depth_cfg = cfg.get("depth", {})
        # Depth-distance gate for candidate filtering: object held near the camera
        # is much nearer than the person's face/background, so dropping candidates
        # that sit far behind the nearest one rejects faces without a face model.
        self.depth_gate = float(depth_cfg.get("candidate_depth_gate", 0.30))
        # Optional skin-based fallback for near-camera faces (cosmetic only).
        self._face_cascade = None
        try:
            cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            if os.path.isfile(cascade_path):
                self._face_cascade = cv2.CascadeClassifier(cascade_path)
                log.info("Face rejection enabled (%s)", cascade_path)
        except Exception as e:
            log.warning("Face rejection unavailable: %s", e)
        self.depth_enabled = bool(depth_cfg.get("enabled", False))
        self.depth_est = DepthEstimator(depth_cfg) if self.depth_enabled else None
        self._depth_heatmap: Optional[np.ndarray] = None
        self._depth_box: Optional[tuple] = None
        self._depth_lock = threading.Lock()
        self._depth_interval = float(depth_cfg.get("refresh_interval_sec", 0.5))
        self._depth_min_idle = float(depth_cfg.get("min_idle_sec", 0.12))
        # Visual recognition can be paused from the web UI. While paused the depth
        # loop stops doing inference (the expensive part) and identify/remember
        # commands are refused; the camera preview keeps running so the user can
        # still frame the shot and resume instantly.
        self._recognition_enabled = True
        self._recognition_lock = threading.Lock()
        # Camera and microphone are released independently — the master pause switch
        # frees both (the three big CPU hogs are depth inference, camera capture
        # and MJPEG encoding), and either can be toggled on its own.
        self._camera_enabled = True
        self._mic_enabled = True
        self._resource_lock = threading.RLock()
        # Honour "don't grab the device yet" requests: the camera/mic are left
        # closed at startup and can be opened later from the web UI.
        if not start_camera:
            self._camera_enabled = False
        if not start_mic:
            self._mic_enabled = False
            self.use_voice = False
        self.tts = TtsClient(cfg.get("tts", {}))
        self.vad = Vad(cfg.get("asr", {}))
        self.asr = ParaformerASR(cfg.get("asr", {}))
        self.mic = MicStream(sample_rate=int(cfg.get("asr", {}).get("sample_rate", 16000)))
        # None = system default input; an int index or a name substring otherwise.
        # The web settings panel can change this at runtime.
        self.mic.device = cfg.get("asr", {}).get("mic_device", None)
        if self.mic.device is not None:
            from core import devices as _dev
            self.mic.device_name = _dev.mic_name(self.mic.device)

        self._intent_q: "queue.Queue[Intent]" = queue.Queue(maxsize=8)
        self._running = False
        self._threads: List[threading.Thread] = []
        self._last_subtitle = ""
        self._subtitle_until = 0.0
        self._chat_log: List[dict] = []  # message history for web UI: [{role, text, ts}]
        self._chat_lock = threading.Lock()
        self._mic_muted_until = 0.0  # avoid hearing our own TTS (simple half-duplex)
        self._flash_until = 0.0

        # debug capture (optional; off by default in the classroom config)
        dbg_cfg = cfg.get("debug", {})
        self.debug_cap = DebugCapture(
            enabled=bool(dbg_cfg.get("enabled", False)),
            base_dir=dbg_cfg.get("dir", "data/debug"),
        )

    # ----------------------------------------------------------------- boot
    def start(self) -> None:
        log.info("Loading models …")
        if self.depth_est is not None:
            try:
                self.depth_est.load()
                log.info("Depth estimator ready (%s)", self.depth_est.device_used)
            except Exception as e:
                log.warning("Depth estimator load failed (falls back to center ROI): %s", e)
                self.depth_est = None
                self.depth_enabled = False
        try:
            self.resnet.load()
        except Exception as e:
            log.error(
                "ResNet18 failed to load (identify/remember degraded): %s\n"
                "  run: python tools/prepare_models.py --resnet18",
                e,
            )
            self._resnet_error = str(e)
        if not getattr(self.resnet, "ready", False):
            if "No module named" in str(getattr(self, "_resnet_error", "")) or self.resnet.device_used == "none":
                log.error(
                    "提示: 识别/录入需要 ResNet18 + onnxruntime。"
                    " 请用 conda activate vision_ai 后再运行。"
                )
        if self.use_voice:
            try:
                self.asr.load()
                self.vad.load()
            except Exception as e:
                log.error("ASR load failed: %s", e)
                self.use_voice = False
        self.tts.start()
        if self._camera_enabled:
            self.camera.start()
        else:
            log.info("Camera left closed (--no-camera); enable it from the web UI")
            self.stats.camera_enabled = False
        if not self._mic_enabled:
            log.info("Microphone left closed (--no-mic); enable it from the web UI")
            self.stats.mic_enabled = False
        if not self.use_voice:
            self.stats.last_reply = "未检测到麦克风或语音模型，可使用网页文字输入发送指令。"

        self._running = True
        self._threads = [
            threading.Thread(target=self._intent_loop, name="intent", daemon=True),
        ]
        if self.depth_enabled and self.depth_est is not None:
            self._threads.append(threading.Thread(target=self._depth_loop, name="depth", daemon=True))
        if self.use_voice and self._mic_enabled:
            self._threads.append(threading.Thread(target=self._voice_loop, name="voice", daemon=True))
        for t in self._threads:
            t.start()

        st = self.memory.stats()
        self.stats.mem_objects = st["objects"]
        self.stats.tts_health = self.tts.health()
        self._refresh_mem_info()
        log.info(
            "Started. memory=%s depth=%s resnet18=%s tts=%s voice=%s",
            st,
            self.depth_est.device_used if self.depth_est else "off",
            self.resnet.device_used,
            self.stats.tts_health,
            self.use_voice,
        )
        self.say("物体识别助手已启动。")

    def stop(self) -> None:
        self._running = False
        for t in self._threads:
            t.join(timeout=2.0)
        self.camera.stop()
        self.tts.stop()
        if self.use_voice:
            self.mic.stop()
        self.memory.close()

    # ------------------------------------------------------------- voice
    def _voice_loop(self) -> None:
        if not self.mic.start():
            log.warning("Mic unavailable — voice disabled, use the web text input")
            self.use_voice = False
            return
        buf: List[np.ndarray] = []
        speaking = False
        silence_frames = 0
        max_end_silence = float(self.cfg.get("asr", {}).get("max_end_silence_ms", 2500))
        max_silence = max(3, int(max_end_silence / self.mic.frame_ms))
        # Early-decision: once the speaker pauses this long, try ASR and fire the
        # command immediately if it resolves to a known intent. Cuts typical
        # latency from ~2.5s to ~0.6s. Falls back to max_silence for incomplete
        # utterances (e.g. "记住这个" waiting for the name).
        early_ms = float(self.cfg.get("asr", {}).get("early_decision_ms", 600))
        early_frames = max(2, int(early_ms / self.mic.frame_ms))
        min_frames = max(3, int(250 / self.mic.frame_ms))
        # Pre-roll: VAD takes a few frames to decide "speech started", which ate
        # the leading audio (e.g. "这是什么" was cut to "这是"). Keep the last
        # 500ms of audio buffered so the utterance starts at the real beginning.
        pre_roll_ms = float(self.cfg.get("asr", {}).get("pre_roll_ms", 500))
        pre_roll = collections.deque(maxlen=max(1, int(pre_roll_ms / self.mic.frame_ms)))
        asr_cache: Optional[str] = None  # cached ASR text for the current buffer

        while self._running:
            frame = self.mic.read(timeout=0.1)
            if frame is None:
                # mic released → don't spin on empty reads
                if not self.mic_enabled:
                    time.sleep(0.3)
                continue
            now = time.time()
            if now < self._mic_muted_until:
                continue

            detected = self.vad.feed(frame)
            if detected and not speaking:
                speaking = True
                # include pre-roll so the utterance's opening words are preserved
                buf = list(pre_roll)
                buf.append(frame)
                pre_roll.clear()
                silence_frames = 0
                asr_cache = None
            elif not speaking:
                pre_roll.append(frame)
            elif speaking:
                buf.append(frame)
                if detected:
                    silence_frames = 0
                    asr_cache = None  # audio changed, cache is stale
                else:
                    silence_frames += 1

                # ---- early decision on a short pause ----
                if (silence_frames >= early_frames and len(buf) >= min_frames):
                    if asr_cache is None:
                        pcm_try = np.concatenate(buf)
                        asr_cache = self.asr.transcribe(pcm_try) or ""
                    intent = parse_intent(asr_cache) if asr_cache else None
                    if (intent is not None and intent.is_known) or silence_frames >= max_silence:
                        pcm = np.concatenate(buf)
                        speaking = False
                        buf = []
                        silence_frames = 0
                        self._handle_pcm(pcm, preset_text=asr_cache)
                        asr_cache = None
                        continue

    def _handle_pcm(self, pcm: np.ndarray, preset_text: Optional[str] = None) -> None:
        t0 = time.perf_counter()
        dur_s = pcm.size / float(self.cfg.get("asr", {}).get("sample_rate", 16000)) if pcm.size else 0.0
        if preset_text:
            text = preset_text  # already transcribed during early decision
        else:
            text = self.asr.transcribe(pcm)
        self.stats.asr_latency_ms = (time.perf_counter() - t0) * 1000.0
        log.info("ASR: %r (audio=%.2fs, %.0fms, cached=%s)",
                 text, dur_s, self.stats.asr_latency_ms, preset_text is not None)
        if not text:
            self._set_subtitle("（没听清，请再说一次）", 2.0)
            return
        log.info("ASR: %s (%.0f ms)", text, self.stats.asr_latency_ms)
        self._set_subtitle(f"你：{text}", 4.0)
        self.push_chat("user", text)
        intent = parse_intent(text)
        self.stats.last_intent = f"{intent.type.value}:{intent.label or '-'}"
        if intent.type == IntentType.UNKNOWN:
            # show what we heard so user knows the ASR result
            self._set_subtitle(f"你：{text}（未识别指令）", 4.0)
        try:
            self._intent_q.put_nowait(intent)
        except queue.Full:
            log.warning("intent queue full, dropping")

    def inject_text(self, text: str) -> None:
        self.push_chat("user", text, source="text")
        intent = parse_intent(text)
        self.stats.last_intent = f"{intent.type.value}:{intent.label or '-'}"
        self._set_subtitle(f"你：{text}", 3.0)
        try:
            self._intent_q.put_nowait(intent)
        except queue.Full:
            pass

    # ------------------------------------------------------------ intent
    def _intent_loop(self) -> None:
        while self._running:
            try:
                intent = self._intent_q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                reply = self._dispatch(intent)
            except Exception as e:
                log.exception("intent handling failed")
                reply = "处理出错了，请再试一次。"
            if reply:
                self.stats.last_reply = reply
                self._set_subtitle(f"助手：{reply}", 4.0)
                self.push_chat("bot", reply)  # web chat bubble (was missing)
                # mute mic briefly so TTS is not re-captured (half-duplex)
                self._mic_muted_until = time.time() + min(8.0, 1.0 + 0.18 * len(reply))
                self.tts.say_async(reply)
            self.stats.mode = "idle"
            self._overlay.learning = False

    def _dispatch(self, intent: Intent) -> str:
        if intent.type == IntentType.IDENTIFY:
            self.stats.mode = "identify"
            return self._do_identify()
        if intent.type == IntentType.REMEMBER:
            self.stats.mode = "remember"
            return self._do_remember(intent.label)
        if intent.type == IntentType.SEARCH:
            self.stats.mode = "search"
            return self._do_search(intent.label)
        if intent.type == IntentType.FORGET:
            if intent.label and self.memory.delete_object(intent.label):
                st = self.memory.stats()
                self.stats.mem_objects = st["objects"]
                self._refresh_mem_info()
                return f"好的，我已经忘记{intent.label}了。"
            return f"没有找到叫{intent.label or '它'}的记忆。"
        if intent.type == IntentType.LIST:
            objs = self.memory.list_objects()
            if not objs:
                return "我还没有录入任何物品。你可以对我说：记住，这个叫……"
            names = "、".join(o.user_name for o in objs[:8])
            more = f"等共{len(objs)}个物品" if len(objs) > 8 else ""
            return f"我录入了：{names}{more}。"
        if intent.type == IntentType.HELP:
            return (
                "你可以问：这是什么？教我记住这个叫……；"
                "按 O 可选择本地图片识别，按 L 查看已录入物品。"
            )
        return f"我没太听懂「{intent.raw_text}」，可以说：这是什么？"

    # ----------------------------------------------------------- embed
    def _embed_crop(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """ResNet18 feature for one crop (intent thread only)."""
        if not getattr(self.resnet, "ready", False):
            return None
        e, ms = self.resnet.timed_embed(crop)
        self.stats.embed_latency_ms = ms
        return e

    @staticmethod
    def _deskew_crop(crop: np.ndarray, mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Straighten an elongated depth foreground without clipping its edges."""
        if mask is None or mask.shape != crop.shape[:2]:
            return None
        binary = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
        width, height = rect[1]
        if min(width, height) < 16 or max(width, height) / min(width, height) < 1.2:
            return None
        points = cv2.boxPoints(rect)
        edges = [points[(i + 1) % 4] - points[i] for i in range(4)]
        edge = max(edges, key=lambda e: float(np.linalg.norm(e)))
        angle = float(np.degrees(np.arctan2(edge[1], edge[0])))
        # A long-axis angle is equivalent modulo 180 degrees.
        angle = (angle + 90.0) % 180.0 - 90.0
        if min(abs(angle), abs(90.0 - abs(angle))) < 5.0:
            return None
        h, w = binary.shape
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        c, s = abs(matrix[0, 0]), abs(matrix[0, 1])
        out_w, out_h = int(np.ceil(w * c + h * s)), int(np.ceil(h * c + w * s))
        matrix[0, 2] += (out_w - w) / 2
        matrix[1, 2] += (out_h - h) / 2
        rotated = cv2.warpAffine(crop, matrix, (out_w, out_h), borderMode=cv2.BORDER_REPLICATE)
        rotated_mask = cv2.warpAffine(binary, matrix, (out_w, out_h), flags=cv2.INTER_NEAREST)
        x, y, rw, rh = cv2.boundingRect(rotated_mask)
        if min(rw, rh) < 16:
            return None
        return np.ascontiguousarray(rotated[y:y + rh, x:x + rw])

    def _match_crop(self, crop: np.ndarray, foreground_mask: Optional[np.ndarray] = None):
        """Match a crop; retry rotated views if the original is uncertain.

        ResNet18 isn't rotation invariant. Rotating the query preserves existing
        enrollment features and the calibrated score/margin checks.
        """
        best_emb, best = None, None
        views = [crop if turns == 0 else np.ascontiguousarray(np.rot90(crop, turns))
                 for turns in range(4)]
        def rotated_views():
            yield from views
            straight = self._deskew_crop(crop, foreground_mask)
            if straight is not None:
                yield from (straight if turns == 0 else np.ascontiguousarray(np.rot90(straight, turns))
                            for turns in range(4))
            # Cover the midpoint between quarter turns even for roughly square
            # objects, where a foreground silhouette cannot determine an axis.
            h, w = crop.shape[:2]
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 45, 1.0)
            side = int(np.ceil(np.hypot(w, h)))
            matrix[0, 2] += (side - w) / 2
            matrix[1, 2] += (side - h) / 2
            diagonal = cv2.warpAffine(crop, matrix, (side, side), borderMode=cv2.BORDER_REPLICATE)
            yield from (diagonal if turns == 0 else np.ascontiguousarray(np.rot90(diagonal, turns))
                        for turns in range(4))

        for view in rotated_views():
            emb = self._embed_crop(view)
            if emb is None:
                continue
            ranked = self.memory.top_k(emb, k=2)
            if not ranked:
                continue
            match = ranked[0]
            if best is None or match.score > best.score:
                best_emb, best = emb, match
            margin = match.score - ranked[1].score if len(ranked) > 1 else 1.0
            if (match.score >= self.memory.threshold
                    and margin >= self.memory.margin_threshold):
                # A more similar but ambiguous orientation must not replace a
                # view that clearly separates the target from other objects.
                return emb, match
        return best_emb, best

    def _is_face_candidate(self, crop: Optional[np.ndarray],
                           foreground_mask: Optional[np.ndarray] = None) -> bool:
        """True if the crop is dominated by a person's face/body.

        Depth treats a person's face as a "near object", so it pollutes the
        candidate pool. OpenCV 5 dropped Haar cascades and the YuNet model is not
        reliably downloadable here, so combine two cheap cues: overall skin ratio
        AND the size of the largest contiguous skin blob (a face/body has one
        large smooth skin region, whereas clutter only has scattered skin-ish
        pixels). This rejects the face crop while keeping real objects.
        """
        if crop is None or crop.size == 0:
            return False
        try:
            if crop.shape[0] < 24 or crop.shape[1] < 24:
                return False
            ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
            cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
            skin = ((cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)).astype(np.uint8)
            area = crop.shape[0] * crop.shape[1]
            if foreground_mask is not None and foreground_mask.shape == skin.shape:
                support = foreground_mask > 0
                area = int(support.sum())
                if area == 0:
                    return False
                skin[~support] = 0
            ratio = float(skin.sum()) / area
            if ratio < 0.12:
                return False
            # largest contiguous skin blob / crop area
            num, _, stats, _ = cv2.connectedComponentsWithStats(skin, connectivity=8)
            if num <= 1:
                return False
            blob_frac = float(stats[1:, cv2.CC_STAT_AREA].max()) / area
            # face/body: sizeable skin ratio AND one dominant skin blob
            return ratio > 0.15 and blob_frac > 0.10
        except Exception:
            return False

    def _depth_candidate_is_face(self, frame: np.ndarray, crop: np.ndarray,
                                 box: tuple, region) -> bool:
        """Check raw foreground, excluding faces/hands OUTSIDE its silhouette.

        The rectangle around a mouse can include the holder's face at a corner.
        Its filled contour keeps that background out of the skin statistics,
        while still checking raw pixels (cleaned crops can conceal real faces).
        """
        x1, y1, x2, y2 = box
        raw = frame[y1:y2, x1:x2]
        if region.mask is not None and region.mask.shape == frame.shape[:2]:
            mask = (region.mask[y1:y2, x1:x2] > 0).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                support = np.zeros_like(mask)
                cv2.fillConvexPoly(support, cv2.convexHull(np.vstack(contours)), 1)
                return self._is_face_candidate(raw, support)
        return self._is_face_candidate(raw) or self._is_face_candidate(crop)

    def _describe_known(self, m: MemoryMatch) -> str:
        pct = int(round(m.score * 100))
        name = m.user_name.strip()
        if name.startswith("我的") or name.startswith("你的") or name.startswith("他的"):
            base = f"这是{name}"
        else:
            base = f"这是你的{name}"
        self.stats.last_similarity = m.score
        return f"{base}，相似度{pct}%。"

    def _describe_unknown(self, m: Optional[MemoryMatch]) -> str:
        thr = int(round(self.memory.threshold * 100))
        if m is None:
            self.stats.last_similarity = 0.0
            if self.memory.stats()["objects"] == 0:
                return "未知物体。特征库还是空的，先说：记住，这个叫……"
            return "未知物体。"
        pct = int(round(m.score * 100))
        self.stats.last_similarity = m.score
        return f"未知物体，最高相似度{pct}%（阈值{thr}%）。"

    # ---------------------------------------------------------- identify
    def _do_identify(self) -> str:
        self._overlay.candidate_snapshot = (0.0, 0, 0, ())
        if not self.recognition_enabled:
            return "识别已暂停，请先点击右上角的「开始识别」。"
        frame, _ = self.camera.latest()
        if frame is None:
            return "摄像头还没有画面。"
        if not getattr(self.resnet, "ready", False):
            return "ResNet18 特征模型未加载，无法识别。请运行 tools/prepare_models.py --resnet18。"

        # start debug capture session (no-op if disabled)
        dc = self.debug_cap
        dc.start("identify", frame)

        best: Optional[MemoryMatch] = None
        best_box: Optional[tuple] = None
        best_emb: Optional[np.ndarray] = None
        cand_boxes: List[tuple] = []
        cand_labels: List[str] = []
        candidates: list[dict] = []
        best_idx = -1

        # 1) Depth-guided candidates (zero-shot, category-free — preferred path).
        # Depth is ambiguous in cluttered scenes, so evaluate the top-k regions and
        # let the feature matcher pick the true target instead of trusting one guess.
        if self.depth_est is not None and self.depth_est.ready:
            depth = self.depth_est.estimate_depth(frame)
            dc.add_stage("depth_estimated")
            if depth is not None:
                regions = self.depth_est.find_top_regions(frame, depth, k=4)
                dc.add_stage("regions_found", count=len(regions),
                             boxes=[list(r.box) for r in regions])
                best_region = None
                # Reference depth = the NEAREST candidate (object held up to the
                # camera is nearer than the person's face/body). Filtering on depth
                # distance alone drops most face/background candidates — no extra
                # face model needed.
                ref_depth = min((r.mean_depth for r in regions), default=0.0)
                for ri, region in enumerate(regions):
                    # reject candidates that sit much farther than the nearest one
                    if region.mean_depth - ref_depth > self.depth_gate:
                        cand_boxes.append(region.box)
                        cand_labels.append(f"{ri}:too_far")
                        candidates.append({"box": region.box, "reason": "too_far"})
                        dc.add_candidate(None, region.box, region.score, region.area_frac,
                                         "depth", match_name="too_far")
                        continue
                    r = self.depth_est.crop_region(frame, region)
                    if r is None:
                        cand_boxes.append(region.box)
                        cand_labels.append(f"{ri}:no_crop")
                        candidates.append({"box": region.box, "reason": "no_crop"})
                        continue
                    crop, box = r
                    cand_boxes.append(box)
                    candidate = {"box": box, "reason": "no_match"}
                    candidates.append(candidate)
                    m = None
                    if not self.cropper.quality_ok(crop):
                        candidate["reason"] = "quality_fail"
                        cand_labels.append(f"{ri}:quality_fail")
                        dc.add_candidate(crop, box, region.score, region.area_frac,
                                         "depth", mask=region.mask, match_name="quality_fail")
                        continue
                    # Face/person rejection must look at the RAW region, not the
                    # mask-cleaned crop — masking replaces skin with median colour
                    # and defeats skin-based detection (measured: skin 0.10 → miss).
                    if self._depth_candidate_is_face(frame, crop, box, region):
                        candidate["reason"] = "face_rejected"
                        cand_labels.append(f"{ri}:face_rejected")
                        dc.add_candidate(crop, box, region.score, region.area_frac,
                                         "depth", mask=region.mask, match_name="face_rejected")
                        continue
                    # Reject oversized boxes — a person/scene spanning most of the
                    # frame is not a discrete held-up object.
                    rb = region.box
                    bbox_frac = ((rb[2]-rb[0]) * (rb[3]-rb[1])) / float(frame.shape[0]*frame.shape[1])
                    if bbox_frac > 0.50:
                        candidate["reason"] = "too_large"
                        cand_labels.append(f"{ri}:too_large")
                        dc.add_candidate(crop, box, region.score, region.area_frac,
                                         "depth", mask=region.mask, match_name="too_large")
                        continue
                    mask = (region.mask[box[1]:box[3], box[0]:box[2]]
                            if region.mask is not None else None)
                    emb, m = self._match_crop(crop, mask)
                    if emb is None:
                        candidate["reason"] = "no_embed"
                        cand_labels.append(f"{ri}:no_embed")
                        dc.add_candidate(crop, box, region.score, region.area_frac,
                                         "depth", mask=region.mask, match_name="no_embed")
                        continue
                    top = self.memory.top_k(emb, k=3) if hasattr(self.memory, "top_k") else []
                    if m is not None:
                        candidate.update(name=m.user_name, score=float(m.score), reason="matched")
                    top_list = [{"name": t.user_name, "score": round(float(t.score), 4)} for t in top]
                    is_better = bool(m and (best is None or m.score > best.score))
                    if is_better:
                        best, best_box = m, box
                        best_region = region
                        best_emb = emb
                        best_idx = len(cand_boxes) - 1
                    cand_labels.append(f"{ri}:{m.user_name if m else '?'}"
                                       f" {m.score:.2f}" if m else f"{ri}:none")
                    dc.add_candidate(crop, box, region.score, region.area_frac, "depth",
                                     mask=region.mask,
                                     match_name=m.user_name if m else "",
                                     match_score=m.score if m else 0.0,
                                     top_matches=top_list,
                                     is_best=is_better)

                if best is not None and best_box is not None:
                    self._update_depth_vis(frame, best_box)
                elif regions:
                    self._update_depth_vis(frame, regions[0].box)

                self._show_candidate_snapshot(frame, candidates)

        # 2) center ROI fallback (last resort) — ONLY when depth was unavailable.
        # If depth ran and every candidate was rejected (face / too_far / quality),
        # the scene has no showable object; falling through to the centre ROI would
        # crop whatever sits mid-frame (often the speaker's own face) and falsely
        # match it against the library.
        depth_ran = False
        if self.depth_est is not None and self.depth_est.ready:
            depth_ran = True
        if best is None and depth_ran:
            text = "我没看到明显的物体，试着把它举到镜头前。"
            dc.finish(text, decision="no_object", asr_text=self.stats.last_intent)
            return text

        if best is None:
            box = self.cropper.center_roi(frame)
            crop = self.cropper.crop_box(frame, box)
            cand_boxes.append(box)
            cand_labels.append("center")
            candidate = {"box": box, "reason": "quality_fail"}
            candidates.append(candidate)
            if not self.cropper.quality_ok(crop):
                self._show_candidate_snapshot(frame, candidates)
                dc.add_frame_annotated(frame, cand_boxes, cand_labels, best_idx)
                dc.finish("我看不清这个物品", decision="quality_fail")
                return "我看不清这个物品，试着把它放到画面中央亮一点。"
            emb, best = self._match_crop(crop)
            if emb is None:
                candidate["reason"] = "no_embed"
                self._show_candidate_snapshot(frame, candidates)
                dc.finish("特征提取失败", decision="no_embed")
                return "特征提取失败，请检查 ResNet18 模型。"
            best_emb = emb
            if best is not None:
                candidate.update(name=best.user_name, score=float(best.score), reason="matched")
            else:
                candidate["reason"] = "no_match"
            self._show_candidate_snapshot(frame, candidates)
            best_box = box
            best_idx = len(cand_boxes) - 1
            dc.add_candidate(crop, box, 0.0, 0.0, "center",
                             match_name=best.user_name if best else "",
                             match_score=best.score if best else 0.0,
                             is_best=True)

        # annotated frame for visual inspection
        dc.add_frame_annotated(frame, cand_boxes, cand_labels, best_idx)

        # Margin-based verdict: ResNet18 features of different objects can score
        # within 0.002–0.07 of each other on similar backgrounds (measured), so a
        # high absolute score alone is not enough. Require top-1 to beat top-2.
        margin = 1.0
        runner_up = ""
        if best is not None and best_emb is not None:
            ranked = self.memory.top_k(best_emb, k=8)
            # Compare against the top DIFFERENT object — comparing two samples of
            # the same object just measures sample noise, not object ambiguity.
            for other in ranked[1:]:
                if other.user_name != best.user_name:
                    margin = float(best.score - other.score)
                    runner_up = other.user_name
                    break
            else:
                margin = 1.0  # only one object type in the bank → no rival
            dc.add_stage("margin_check", top1=best.user_name,
                         top1_score=round(float(best.score), 4),
                         top2=runner_up, margin=round(margin, 4),
                         threshold=round(float(self.memory.threshold), 4))

        margin_need = float(self.memory.margin_threshold)
        if best is None or best.score < self.memory.threshold or margin < margin_need:
            if best is not None and margin < margin_need and best.score >= self.memory.threshold:
                text = (f"不太确定，看起来像{best.user_name}（{int(best.score*100)}%），"
                        f"但{runner_up}也很相似（差距仅{int(margin*100)}%）。"
                        f"换个角度或背景再试试。")
                dc.finish(text, decision="uncertain", asr_text=self.stats.last_intent)
            else:
                text = self._describe_unknown(best)
                dc.finish(text, decision="unknown", asr_text=self.stats.last_intent)
            if best_box is not None and best is not None:
                self._flash_match(best_box, "未知物体", best.score)
            return text

        if best_box is not None:
            self._flash_match(best_box, best.user_name, best.score)
        text = self._describe_known(best)
        dc.finish(text, decision="matched",
                  asr_text=self.stats.last_intent)
        return text

    # ----------------------------------------------------------- remember
    def _do_remember(self, label: str) -> str:
        if not label:
            return "你想给它起什么名字？请说：记住，这个叫……"
        if not self.recognition_enabled:
            return "识别已暂停，请先点击右上角的「开始识别」再录入。"
        if not getattr(self.resnet, "ready", False):
            return (
                "ResNet18 特征模型没有加载，无法录入物品。"
                "请运行 python tools/prepare_models.py --resnet18 后重试。"
            )
        mem_cfg = self.cfg.get("memory", {})
        min_samples = max(1, int(mem_cfg.get("min_samples", 3)))
        max_samples = int(mem_cfg.get("max_samples", 10))
        if max_samples < min_samples:
            return "录入配置错误：max_samples 不能小于 min_samples，请修正配置后重试。"

        self.stats.mode = "remember"
        self._overlay.learning = True
        self._overlay.learning_label = f"录入模式：{label}"
        self._overlay.learning_progress = 0.0
        self._overlay.highlight = []
        self._overlay.match_box = None

        try:
            return self._remember_from_camera(label, mem_cfg, min_samples, max_samples)
        finally:
            self._overlay.learning = False
            self._overlay.match_box = None

    def _remember_from_camera(self, label: str, mem_cfg: dict,
                              min_samples: int, max_samples: int) -> str:
        duration = max(0.1, float(mem_cfg.get("sample_duration_sec", 2.5)))
        interval = max(0.0, float(mem_cfg.get("sample_interval_ms", 250)) / 1000.0)
        # Full-resolution CPU depth inference consumes most of the normal window.
        # Keep collecting fresh frames until the minimum is met, within a bounded
        # total wait. Use a monotonic clock so wall-clock changes cannot end it.
        timeout = max(duration, float(mem_cfg.get("sample_timeout_sec", 15.0)))
        t_start = time.monotonic()
        t_end = t_start + duration
        t_deadline = t_start + timeout
        t_next = t_start
        samples: List[tuple[np.ndarray, tuple[int, int, int, int], str]] = []
        embeds: List[np.ndarray] = []
        generic_guess = ""
        last_frame_id = None
        waiting_for_frame = False
        attempted = quality_rejected = embed_failed = localization_failed = 0

        while len(samples) < max_samples and self._running and self.recognition_enabled:
            now = time.monotonic()
            if now >= t_deadline or (now >= t_end and len(samples) >= min_samples):
                break
            if now >= t_next:
                t_next = now + interval
                frame, frame_id = self.camera.latest()
                if frame is None or frame_id == last_frame_id:
                    waiting_for_frame = True
                    time.sleep(0.02)
                    continue
                waiting_for_frame = False
                last_frame_id = frame_id
                attempted += 1
                self._overlay.learning_progress = min(0.99, len(samples) / min_samples)
                # depth-guided nearest object (zero-shot) preferred, YOLO fallback
                crop, box, source = None, None, ""
                if self.depth_est is not None and self.depth_est.ready:
                    # Each sample must use depth from THAT frame. Reusing the
                    # first map crops stale coordinates if the held object moves.
                    sample_depth = self.depth_est.estimate_depth(frame)
                    if sample_depth is not None:
                        # Use the SAME candidate pipeline as identify: top-k regions
                        # (incl. the skin-free pass) plus the same filters. The old
                        # find_nearest_object() path returned one big "person+object"
                        # blob, so every enrolled sample was full of background.
                        regions = self.depth_est.find_top_regions(frame, sample_depth, k=4)
                        ref_depth = min((r.mean_depth for r in regions), default=0.0)
                        best_crop, best_box, best_score = None, None, -1.0
                        for region in regions:
                            if region.mean_depth - ref_depth > self.depth_gate:
                                continue  # too far behind the nearest candidate
                            r = self.depth_est.crop_region(frame, region)
                            if r is None:
                                continue
                            c, b = r
                            if not self.cropper.quality_ok(c):
                                continue
                            if self._depth_candidate_is_face(frame, c, b, region):
                                continue
                            rb = region.box
                            bbox_frac = ((rb[2]-rb[0]) * (rb[3]-rb[1])) / float(frame.shape[0]*frame.shape[1])
                            if bbox_frac > 0.50:
                                continue
                            if region.score > best_score:
                                best_crop, best_box, best_score = c, b, region.score
                        if best_crop is not None:
                            crop, box, source = best_crop, best_box, "depth"
                            with self._depth_lock:
                                self._depth_heatmap = DepthEstimator.depth_to_colormap(sample_depth)
                                self._depth_box = box
                    if crop is None:
                        # Match identify's policy: don't silently enroll the
                        # speaker/background when depth found no valid object.
                        localization_failed += 1
                        time.sleep(0.02)
                        continue
                if crop is None or crop.size == 0:
                    # Depth unavailable → center ROI fallback (never YOLO).
                    box = self.cropper.center_roi(frame)
                    crop = self.cropper.crop_box(frame, box)
                    source = "center_roi"
                if not self.cropper.quality_ok(crop):
                    quality_rejected += 1
                    time.sleep(0.02)
                    continue
                emb = self._embed_crop(crop)
                if emb is None:
                    embed_failed += 1
                    continue
                # diversity filter — never starve min_samples on a static scene
                force = len(samples) < min_samples
                if embeds and not force:
                    sims = [float(emb @ s) for s in embeds]
                    if max(sims) > 0.995:
                        time.sleep(0.02)
                        continue
                embeds.append(emb)
                samples.append((crop, box, source))
                self._overlay.learning_progress = min(0.99, len(samples) / min_samples)
                self._overlay.match_box = np.array(box, dtype=np.float32)
            time.sleep(0.02)

        elapsed = time.monotonic() - t_start
        log.info("Enrollment '%s': %d/%d samples, %.2fs, %d fresh frames, "
                 "%d quality rejected, %d embedding failures, %d localization failures",
                 label, len(samples), min_samples, elapsed, attempted,
                 quality_rejected, embed_failed, localization_failed)
        # Pausing/stopping cancels even if enough samples have already arrived.
        if not self._running or not self.recognition_enabled:
            return f"录入已取消，没有记住{label}。"
        if len(samples) < min_samples:
            if not getattr(self.resnet, "ready", False):
                return f"特征模型未加载，没有记住{label}。请用 vision_ai 环境重试。"
            if not attempted or waiting_for_frame:
                reason = "摄像头未提供足够的新画面，请检查摄像头连接后重试。"
            elif embed_failed:
                reason = "特征提取失败，请检查模型运行日志后重试。"
            elif quality_rejected:
                reason = "画面质量不足，请把物体拿稳靠近镜头，光线充足一些再试。"
            elif localization_failed:
                reason = "没有定位到完整物体，请把物体举到镜头前再试。"
            else:
                reason = "采样处理超时，请重试或增大 memory.sample_timeout_sec。"
            return (
                f"采样不足（{len(samples)}/{min_samples}），没有记住{label}。"
                f"{reason}"
            )

        self._overlay.learning_progress = 1.0
        # persist images + ResNet18 features
        obj_dir = os.path.join(self.cfg["memory"]["samples_dir"], _safe_name(label))
        os.makedirs(obj_dir, exist_ok=True)
        paths = []
        for i, (crop, box, source) in enumerate(samples):
            p = os.path.join(obj_dir, f"s{i:02d}.png")
            imwrite_unicode(p, crop)
            paths.append(p)
        self.memory.add_object(label, generic_guess or None, embeds, paths)
        st2 = self.memory.stats()
        self.stats.mem_objects = st2["objects"]
        self.stats.mem_info = (
            f"bank {self.stats.mem_objects} objs / {st2['samples']} samples / ResNet18-512d"
        )
        return f"好的，我已经记住{label}了（{len(samples)}张图片，ResNet18特征）。"

    # ------------------------------------------------------------- search
    def _do_search(self, label: str) -> str:
        if not label:
            return "你要找什么？请说：帮我找……"
        if not getattr(self.resnet, "ready", False):
            return "ResNet18 未加载，无法检索。"
        if self.memory.stats()["objects"] == 0:
            return f"特征库还是空的，可先：记住，这个叫……"

        frame, _ = self.camera.latest()
        if frame is None:
            return "摄像头还没有画面。"
        h, w = frame.shape[:2]
        obj = self.memory.find_by_name(label)

        best_box = None
        best_score = -1.0
        best_mem: Optional[MemoryMatch] = None

        box = self.cropper.center_roi(frame)
        crop = self.cropper.crop_box(frame, box)
        if self.cropper.quality_ok(crop):
            emb = self._embed_crop(crop)
            if emb is not None:
                m = self.memory.best_match(emb)
                if m and (
                    _name_related(label, m.user_name)
                    or (obj and m.object_id == obj.object_id)
                ):
                    best_box = box
                    best_score = m.score
                    best_mem = m

        if best_box is None:
            if self.memory.stats()["objects"] == 0:
                return f"画面里没有检测到{label}。特征库是空的，可先：记住，这个叫……"
            return f"画面里没有找到{label}。"
        pos = describe_position(best_box, w, h)
        name = best_mem.user_name if best_mem else label
        self._flash_match(best_box, name, max(best_score, 0.0))
        n = name.strip()
        pct = int(round(max(best_score, 0.0) * 100))
        if n.startswith("我的") or n.startswith("你的"):
            return f"找到{n}了，在{pos}（相似度{pct}%）。"
        return f"找到你的{n}了，在{pos}（相似度{pct}%）。"

    # ------------------------------------------------------- local image
    def recognize_image_file(self, path: str) -> str:
        """Recognize one local image (选题一: 本地图片输入). Thread-safe."""
        if not getattr(self.resnet, "ready", False):
            return "ResNet18 特征模型未加载，无法识别本地图片。"
        img = imread_unicode(path)
        if img is None:
            return f"无法读取图片：{os.path.basename(path)}"
        emb, ms = self.resnet.timed_embed(img)
        self.stats.embed_latency_ms = ms
        if emb is None:
            return "特征提取失败，换一张图片试试。"
        m = self.memory.best_match(emb)
        if m is None or m.score < self.memory.threshold:
            reply = self._describe_unknown(m)
        else:
            reply = self._describe_known(m)
        log.info("recognize_file %s → %s", os.path.basename(path), reply)
        return reply

    def enroll_from_paths(self, label: str, paths: List[str]) -> str:
        """Enroll an object from local image files (batch / CLI)."""
        try:
            return self.enroll_image_files(label, paths)
        except (ValueError, FileExistsError, RuntimeError) as e:
            return str(e)

    def enroll_image_files(self, label: str, paths: List[str]) -> str:
        """Add a new object from image files, preserving existing samples on failure."""
        label = label.strip()
        if not label or len(label) > 80:
            raise ValueError("请输入不超过80个字符的物体名称。")
        if not getattr(self.resnet, "ready", False):
            raise RuntimeError("ResNet18 特征模型未加载，无法录入。")
        min_samples = int(self.cfg.get("memory", {}).get("min_samples", 3))
        max_samples = int(self.cfg.get("memory", {}).get("max_samples", 10))
        if not min_samples <= len(paths) <= max_samples:
            raise ValueError(f"请选择 {min_samples} 到 {max_samples} 张图片。")
        embeds: List[np.ndarray] = []
        images: List[np.ndarray] = []
        saved: List[str] = []
        for src in paths:
            img = imread_unicode(src)
            if img is None or not self.cropper.quality_ok(img):
                continue
            emb = self.resnet.embed(img)
            if emb is None:
                continue
            embeds.append(emb)
            images.append(img)
        if len(embeds) < min_samples:
            raise ValueError(
                f"录入失败：有效图片 {len(embeds)}/{len(paths)}，至少需要 {min_samples} 张。"
            )
        root = self.cfg["memory"]["samples_dir"]
        os.makedirs(root, exist_ok=True)
        obj_dir = tempfile.mkdtemp(prefix="import-", dir=root)
        try:
            for i, img in enumerate(images):
                dest = os.path.join(obj_dir, f"s{i:02d}.png")
                if not imwrite_unicode(dest, img):
                    raise RuntimeError("保存样本图片失败。")
                saved.append(dest)
            self.memory.add_object(label, None, embeds, saved, replace_existing=False)
        except FileExistsError:
            shutil.rmtree(obj_dir, ignore_errors=True)
            raise FileExistsError(f"物体「{label}」已存在，请使用其他名称。")
        except Exception:
            shutil.rmtree(obj_dir, ignore_errors=True)
            raise
        st = self.memory.stats()
        self.stats.mem_objects = st["objects"]
        self._refresh_mem_info()
        return f"好的，我已经记住{label}了（{len(embeds)}张图片，ResNet18特征）。"

    def _show_candidate_snapshot(self, frame: np.ndarray, candidates: List[dict]) -> None:
        """Publish this identification's candidate boxes for four seconds."""
        height, width = frame.shape[:2]
        self._overlay.candidate_snapshot = (
            time.time() + 4.0, width, height,
            tuple((tuple(int(value) for value in c["box"]), c.get("name"),
                   c.get("score"), c.get("reason", "no_match")) for c in candidates),
        )

    # ------------------------------------------------------------- utils
    def _flash_match(self, box, label: str, score: float) -> None:
        self._overlay.highlight = [(np.asarray(box, dtype=np.float32), label, float(score))]
        self._overlay.match_box = np.asarray(box, dtype=np.float32)
        self._overlay.match_label = label
        self._flash_until = time.time() + 4.0

    def say(self, text: str) -> None:
        self.stats.last_reply = text
        self._set_subtitle(f"助手：{text}", 4.0)
        self.push_chat("bot", text)
        self.tts.say_async(text)

    def _set_subtitle(self, text: str, seconds: float) -> None:
        self._last_subtitle = text
        self._subtitle_until = time.time() + seconds

    def push_chat(self, role: str, text: str, source: str = "") -> None:
        """Append a message to the chat history (user/bot/sys) for web UI."""
        with self._chat_lock:
            self._chat_log.append({"role": role, "text": text, "ts": time.time(), "source": source})
            if len(self._chat_log) > 100:
                self._chat_log = self._chat_log[-100:]

    def get_chat_since(self, since_ts: float) -> list:
        """Return messages newer than since_ts."""
        with self._chat_lock:
            return [m for m in self._chat_log if m["ts"] > since_ts]

    def _refresh_mem_info(self) -> None:
        st = self.memory.stats()
        thr = int(round(st.get("threshold", self.memory.threshold) * 100))
        self.stats.mem_info = (
            f"bank {st['objects']} objs / {st['samples']} samples | "
            f"ResNet18 {self.resnet.device_used} | thr {thr}%"
        )

    # ------------------------------------------------------- depth window
    # ------------------------------------------------------- resource control
    @property
    def recognition_enabled(self) -> bool:
        with self._recognition_lock:
            return self._recognition_enabled

    @property
    def camera_enabled(self) -> bool:
        with self._resource_lock:
            return self._camera_enabled

    @property
    def mic_enabled(self) -> bool:
        with self._resource_lock:
            return self._mic_enabled

    def set_camera_enabled(self, enabled: bool) -> bool:
        """Start/stop camera capture. Stopping releases the device entirely."""
        with self._resource_lock:
            if bool(enabled) == self._camera_enabled:
                return self._camera_enabled
            try:
                if enabled:
                    self.camera.start()
                else:
                    self.camera.stop()
            except Exception as e:
                log.warning("camera toggle failed: %s", e)
                return self._camera_enabled
            self._camera_enabled = bool(enabled)
        self.stats.camera_enabled = self._camera_enabled
        log.info("Camera %s", "started" if self._camera_enabled else "released")
        return self._camera_enabled

    def set_mic_enabled(self, enabled: bool) -> bool:
        """Start/stop microphone capture. Stopping releases the device entirely."""
        with self._resource_lock:
            if bool(enabled) == self._mic_enabled:
                return self._mic_enabled
            try:
                if enabled:
                    # A server restarted with --no-mic leaves voice models cold.
                    # Prepare them before starting a worker when the user resumes.
                    if not self.asr.ready:
                        self.asr.load()
                        self.vad.load()
                    # The voice worker is only spawned at start(); bring it up now.
                    alive = any(t.name == "voice" and t.is_alive() for t in self._threads)
                    if not alive:
                        t = threading.Thread(target=self._voice_loop, name="voice", daemon=True)
                        t.start()
                        self._threads.append(t)
                    elif not self.mic.start():
                        log.warning("Mic unavailable — voice stays disabled")
                        self.use_voice = False
                        return self._mic_enabled
                    self.use_voice = True
                else:
                    self.mic.stop()
                    self.use_voice = False
            except Exception as e:
                log.warning("mic toggle failed: %s", e)
                return self._mic_enabled
            self._mic_enabled = bool(enabled)
        self.stats.mic_enabled = self._mic_enabled
        log.info("Microphone %s", "started" if self._mic_enabled else "released")
        return self._mic_enabled

    # ------------------------------------------------------- device switching
    def set_camera_device(self, index: int) -> bool:
        """Hot-switch the camera index, preserving the on/off state.

        ``Camera.reopen`` restores the previous device on failure, so a bad
        selection leaves the running capture pipeline untouched.
        """
        try:
            index = int(index)
        except (TypeError, ValueError):
            return False
        if index < 0:
            return False
        with self._resource_lock:
            if index == self.camera.index:
                return True
            try:
                self.camera.reopen(index)
            except Exception as e:
                log.warning("camera switch to %s failed: %s", index, e)
                return False
            # While the camera is released reopen() does not touch the device;
            # that is intentional — DSHOW probes are flaky (a busy driver can
            # refuse an otherwise valid index), and the picker only offers
            # indexes that came from a successful scan. Enabling the camera
            # later performs the real open() and reports a failure.
            self.cfg.setdefault("app", {})["camera_index"] = index
        log.info("Camera device -> index %s", index)
        return True

    def set_mic_device(self, device) -> bool:
        """Hot-switch the input device, preserving the on/off state.

        The voice worker keeps running: it only reads from the queue, so a
        stop -> change device -> start cycle needs no thread coordination.
        """
        old = self.mic.device
        if str(device) == str(old):
            return True
        with self._resource_lock:
            was_on = self._mic_enabled and self.use_voice
            self.mic.stop()
            self.mic.flush()
            self.mic.device = device
            ok = True
            if was_on:
                try:
                    if not self.asr.ready:
                        self.asr.load()
                        self.vad.load()
                    alive = any(t.name == "voice" and t.is_alive() for t in self._threads)
                    if alive:
                        ok = self.mic.start(device)
                    else:
                        # A worker that died earlier (e.g. --no-mic startup)
                        # opens the stream itself; give it a moment to come up.
                        t = threading.Thread(target=self._voice_loop, name="voice", daemon=True)
                        t.start()
                        self._threads.append(t)
                        for _ in range(12):
                            if self.mic.available:
                                break
                            time.sleep(0.05)
                        ok = self.mic.available
                except Exception as e:
                    log.warning("mic switch to %s failed: %s", device, e)
                    ok = False
            if not ok:
                self.mic.device = old
                if was_on:
                    try:
                        self.mic.start(old)
                    except Exception as e:
                        log.warning("failed to restore mic %s: %s", old, e)
                return False
            self.cfg.setdefault("asr", {})["mic_device"] = device
            from core import devices as _dev
            self.mic.device_name = _dev.mic_name(device)
        log.info("Mic device -> %s", device if device is not None else "system default")
        return True

    @property
    def current_devices(self) -> dict:
        """Labels for /status and the device picker."""
        mic = self.mic.device
        if mic is None:
            mic_label = "系统默认输入设备"
        elif isinstance(mic, int):
            mic_label = self.mic.device_name or f"输入设备 {mic}"
        else:
            mic_label = str(mic)
        # DirectShow friendly name when available, else the plain index label.
        from core import devices as _dev
        name = _dev.camera_name(self.camera.index)
        camera_label = (f"{name}（索引 {self.camera.index}）"
                        if name != f"摄像头 {self.camera.index}"
                        else name)
        return {
            "camera_index": self.camera.index,
            "camera_label": camera_label,
            "mic_device": mic,
            "mic_label": mic_label,
        }

    def set_recognition_enabled(self, enabled: bool) -> bool:
        """Master switch. Pausing frees the heavy resources (depth inference,
        camera capture and MJPEG encoding) so the machine stops spinning; the
        microphone is released too since the assistant is idle."""
        with self._recognition_lock:
            if bool(enabled) == self._recognition_enabled:
                return self._recognition_enabled
            self._recognition_enabled = bool(enabled)
            state = self._recognition_enabled
        self.stats.recognition_enabled = state
        if state:
            self.set_camera_enabled(True)
            self.set_mic_enabled(True)
        else:
            self.set_camera_enabled(False)
            self.set_mic_enabled(False)
            # Drop the stale heatmap so the UI doesn't show a frozen box.
            with self._depth_lock:
                self._depth_box = None
        log.info("Recognition %s", "started" if state else "paused")
        return state

    def _depth_loop(self) -> None:
        """Background thread: depth estimation + nearest-object at ~2 Hz. Never blocks UI."""
        while self._running:
            t0 = time.perf_counter()
            if (self.recognition_enabled and self.depth_est is not None
                    and self.depth_est.ready):
                frame, _ = self.camera.latest()
                if frame is not None:
                    try:
                        # fast=True: low-res session for the heatmap (2Hz vs 0.6Hz)
                        depth = self.depth_est.estimate_depth(frame, fast=True)
                        if depth is not None:
                            heatmap = DepthEstimator.depth_to_colormap(depth)
                            region = self.depth_est.find_nearest_object(frame, depth)
                            box = region.box if region else None
                            with self._depth_lock:
                                self._depth_heatmap = heatmap
                                self._depth_box = box
                    except Exception as e:
                        log.debug("Depth loop error: %s", e)
            elapsed = time.perf_counter() - t0
            # Inference usually exceeds the target interval; guarantee a minimum
            # idle slice so the UI/streaming threads always get CPU time.
            time.sleep(max(self._depth_min_idle, self._depth_interval - elapsed))

    def _update_depth_vis(self, frame: np.ndarray, box: tuple) -> None:
        """Cache depth heatmap for identify/remember (called from intent worker)."""
        if self.depth_est is None or not self.depth_est.ready:
            return
        depth = self.depth_est.estimate_depth(frame)
        if depth is None:
            return
        with self._depth_lock:
            self._depth_heatmap = DepthEstimator.depth_to_colormap(depth)
            self._depth_box = box

def _safe_name(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "·"):
            keep.append(ch)
        else:
            keep.append("_")
    s = "".join(keep).strip("_") or "obj"
    return s[:48]


def _name_related(query: str, stored: str) -> bool:
    q = query.replace("我的", "").strip()
    s = stored.replace("我的", "").strip()
    if not q or not s:
        return True
    if q == s or q in s or s in q:
        return True
    inter = len(set(q) & set(s))
    return inter >= max(1, min(len(q), len(s)) // 2)
