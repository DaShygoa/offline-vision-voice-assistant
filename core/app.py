"""Main application orchestrator (选题一: ResNet18 + 语音交互物体识别).

Threading model:
  ┌ main thread      : OpenCV camera UI loop @ 25–30 FPS (never blocks on TTS/ASR/embed)
  ├ camera thread    : frame capture → latest() buffer
  ├ detect thread    : YOLO every N frames (optional ROI helper), updates shared detections
  ├ voice thread     : mic → VAD → ASR → intent → queue
  ├ intent worker    : identify / remember / search via ResNet18 + cosine feature bank
  └ tts worker       : GPT-SoVITS independent client (async; UI does not wait)

Core flow (assignment):
  enroll   : crop → ResNet18 512-d → save name + features to feature bank
  identify : crop → ResNet18 → cosine vs bank → score >= threshold ? name : 未知物体
  voice    : ASR says the object name / TTS announces the recognition result

ResNet18 only runs on identify / remember / search — never in the camera hot path.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from nlu.intent import Intent, IntentType, parse_intent
from vision.camera import Camera
from vision.detector import Detection, YoloDetector
from vision.feature_extractor import ResNet18Extractor
from vision.memory import MemoryMatch, VisualMemory
from vision.object_cropper import ObjectCropper
from vision.position import describe_position
from vision.visualizer import OverlayState, Visualizer
from ui.text_input import TextInputWindow
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


class AssistantApp:
    def __init__(self, cfg: dict, headless: bool = False, use_voice: bool = True):
        self.cfg = cfg
        self.headless = headless
        self.use_voice = use_voice

        self.stats = AppStats()
        self._overlay = OverlayState()
        self._viz = Visualizer(center_roi_frac=float(cfg.get("memory", {}).get("center_roi_frac", 0.45)))

        # components
        self.camera = Camera(
            index=int(cfg.get("app", {}).get("camera_index", 0)),
            width=int(cfg.get("app", {}).get("width", 1280)),
            height=int(cfg.get("app", {}).get("height", 720)),
        )
        self.yolo_enabled = bool(cfg.get("yolo", {}).get("enabled", True))
        self.detector = YoloDetector(cfg.get("yolo", {})) if self.yolo_enabled else None
        self.resnet = ResNet18Extractor(cfg.get("resnet18", {}))
        self.memory = VisualMemory(cfg.get("memory", {}))
        self.cropper = ObjectCropper(
            center_roi_frac=float(cfg.get("memory", {}).get("center_roi_frac", 0.45)),
            min_crop_px=int(cfg.get("memory", {}).get("min_crop_px", 64)),
        )
        self.tts = TtsClient(cfg.get("tts", {}))
        self.vad = Vad(cfg.get("asr", {}))
        self.asr = ParaformerASR(cfg.get("asr", {}))
        self.mic = MicStream(sample_rate=int(cfg.get("asr", {}).get("sample_rate", 16000)))

        self._det_every = max(1, int(cfg.get("yolo", {}).get("every_n_frames", 4)))
        self._detections: List[Detection] = []
        self._det_lock = threading.Lock()
        self._det_frame_id = -1
        self._det_count = 0
        self._det_fps_t0 = time.time()

        self._intent_q: "queue.Queue[Intent]" = queue.Queue(maxsize=8)
        self._running = False
        self._threads: List[threading.Thread] = []
        self._last_subtitle = ""
        self._subtitle_until = 0.0
        self._mic_muted_until = 0.0  # avoid hearing our own TTS (simple half-duplex)
        self._flash_until = 0.0

        # text command input (no-microphone fallback)
        app_cfg = cfg.get("app", {})
        self._text_input_enabled = bool(app_cfg.get("text_input", True)) and not headless
        self._text_input_always = bool(app_cfg.get("text_input_always", False))
        self.text_input: Optional[TextInputWindow] = None
        self._cv_input_focused = False
        self._cv_input_buf = ""
        self._cv_input_placeholder = "输入指令… (t/Enter 聚焦，回车发送，Esc 取消)"
        self._need_text_input = False

    def _ensure_text_input(self) -> None:
        """Open the Tk IME window once (idempotent). Call from UI/main thread."""
        if not self._text_input_enabled or self.headless:
            return
        if self.text_input is not None and self.text_input.available:
            return
        win = TextInputWindow(on_submit=self.inject_text)
        if win.start():
            self.text_input = win
        else:
            log.info("Falling back to in-frame OpenCV text input (press t)")
        self._need_text_input = False

    # ----------------------------------------------------------------- boot
    def start(self) -> None:
        log.info("Loading models …")
        if self.detector is not None:
            try:
                self.detector.load()
            except Exception as e:
                log.warning("YOLO load failed (ROI falls back to center crop): %s", e)
                self.detector = None
                self.yolo_enabled = False
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
                    "提示: 识别/录入需要 ResNet18 + onnxruntime/openvino。"
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
        self.camera.start()

        # text box: always if no mic, or if user asked for it
        if self._text_input_enabled and (self._text_input_always or not self.use_voice):
            self._ensure_text_input()
            self._overlay.input_visible = True
        elif self._text_input_enabled:
            self._overlay.input_visible = True

        self._running = True
        self._threads = [
            threading.Thread(target=self._intent_loop, name="intent", daemon=True),
        ]
        if self.yolo_enabled and self.detector is not None:
            self._threads.insert(0, threading.Thread(target=self._detect_loop, name="detect", daemon=True))
        if self.use_voice:
            self._threads.append(threading.Thread(target=self._voice_loop, name="voice", daemon=True))
        for t in self._threads:
            t.start()

        st = self.memory.stats()
        self.stats.mem_objects = st["objects"]
        self.stats.tts_health = self.tts.health()
        self._refresh_mem_info()
        log.info(
            "Started. memory=%s yolo=%s resnet18=%s tts=%s voice=%s text_input=%s",
            st,
            self.detector.device_used if self.detector else "off",
            self.resnet.device_used,
            self.stats.tts_health,
            self.use_voice,
            "ime" if self.text_input else ("frame" if self._text_input_enabled else "off"),
        )
        if not self.use_voice:
            self.say("未检测到麦克风，可使用文本输入框发送指令。")
        else:
            self.say("物体识别助手已启动。")

    def stop(self) -> None:
        self._running = False
        for t in self._threads:
            t.join(timeout=2.0)
        if self.text_input is not None:
            self.text_input.stop()
            self.text_input = None
        self.camera.stop()
        self.tts.stop()
        if self.use_voice:
            self.mic.stop()
        self.memory.close()

    # ------------------------------------------------------------ detect
    def _detect_loop(self) -> None:
        while self._running:
            frame, fid = self.camera.latest()
            if frame is None or fid == self._det_frame_id:
                time.sleep(0.005)
                continue
            if fid % self._det_every != 0:
                time.sleep(0.002)
                continue
            self._det_frame_id = fid
            try:
                dets, dt = self.detector.infer(frame)
            except Exception as e:
                log.warning("detect error: %s", e)
                time.sleep(0.05)
                continue
            with self._det_lock:
                self._detections = dets
            self.stats.det_latency_ms = dt
            self._det_count += 1
            now = time.time()
            if now - self._det_fps_t0 >= 1.0:
                self.stats.det_fps = self._det_count / (now - self._det_fps_t0)
                self._det_count = 0
                self._det_fps_t0 = now
            if dets:
                top = dets[0]
                self.stats.det_info = (
                    f"YOLO {self.detector.device_used} | {top.label}:{top.conf:.2f} "
                    f"(+{len(dets)-1}) @{dt:.0f}ms"
                )
            else:
                self.stats.det_info = f"YOLO {self.detector.device_used} | none @{dt:.0f}ms"

    def _current_detections(self) -> List[Detection]:
        with self._det_lock:
            return list(self._detections)

    # ------------------------------------------------------------- voice
    def _voice_loop(self) -> None:
        if not self.mic.start():
            log.warning("Mic unavailable — voice disabled, enable text input")
            self.use_voice = False
            self._need_text_input = True  # opened on UI thread
            return
        buf: List[np.ndarray] = []
        speaking = False
        silence_frames = 0
        max_silence = max(3, int(800 / self.mic.frame_ms))
        min_frames = max(3, int(250 / self.mic.frame_ms))

        while self._running:
            frame = self.mic.read(timeout=0.1)
            if frame is None:
                continue
            now = time.time()
            if now < self._mic_muted_until:
                continue

            detected = self.vad.feed(frame)
            if detected and not speaking:
                speaking = True
                buf = [frame]
                silence_frames = 0
            elif speaking:
                buf.append(frame)
                if detected:
                    silence_frames = 0
                else:
                    silence_frames += 1
                if self.vad.backend == "silero_sherpa" and self.vad.is_speech_finished():
                    pcm = self.vad.pop()
                    if pcm.size == 0:
                        pcm = np.concatenate(buf) if buf else np.zeros(0, np.float32)
                    speaking = False
                    buf = []
                    self._handle_pcm(pcm)
                    continue
                if silence_frames >= max_silence and len(buf) >= min_frames:
                    pcm = np.concatenate(buf)
                    speaking = False
                    buf = []
                    silence_frames = 0
                    self._handle_pcm(pcm)

    def _handle_pcm(self, pcm: np.ndarray) -> None:
        t0 = time.perf_counter()
        text = self.asr.transcribe(pcm)
        self.stats.asr_latency_ms = (time.perf_counter() - t0) * 1000.0
        if not text:
            return
        log.info("ASR: %s (%.0f ms)", text, self.stats.asr_latency_ms)
        self._set_subtitle(f"你：{text}", 3.0)
        intent = parse_intent(text)
        self.stats.last_intent = f"{intent.type.value}:{intent.label or '-'}"
        try:
            self._intent_q.put_nowait(intent)
        except queue.Full:
            log.warning("intent queue full, dropping")

    def inject_text(self, text: str) -> None:
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
        return "我没太听懂，可以说：这是什么？"

    # ----------------------------------------------------------- embed
    def _embed_crop(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """ResNet18 feature for one crop (intent thread only)."""
        if not getattr(self.resnet, "ready", False):
            return None
        e, ms = self.resnet.timed_embed(crop)
        self.stats.embed_latency_ms = ms
        return e

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
        return f"未知物体，最相似的{m.user_name}只有{pct}%（阈值{thr}%）。"

    # ---------------------------------------------------------- identify
    def _do_identify(self) -> str:
        frame, _ = self.camera.latest()
        if frame is None:
            return "摄像头还没有画面。"
        if not getattr(self.resnet, "ready", False):
            return "ResNet18 特征模型未加载，无法识别。请运行 tools/prepare_models.py --resnet18。"

        dets = self._current_detections()
        best: Optional[MemoryMatch] = None
        best_box: Optional[tuple] = None

        # 1) YOLO boxes → embed each → best cosine over the bank
        if dets:
            for crop, box, det in self.cropper.crop_detections(frame, dets):
                if not self.cropper.quality_ok(crop):
                    continue
                emb = self._embed_crop(crop)
                if emb is None:
                    continue
                m = self.memory.best_match(emb)
                if m and (best is None or m.score > best.score):
                    best, best_box = m, box

        # 2) center ROI fallback (also the main path when YOLO is off)
        if best is None:
            box = self.cropper.center_roi(frame)
            crop = self.cropper.crop_box(frame, box)
            if not self.cropper.quality_ok(crop):
                return "我看不清这个物品，试着把它放到画面中央亮一点。"
            emb = self._embed_crop(crop)
            if emb is None:
                return "特征提取失败，请检查 ResNet18 模型。"
            best = self.memory.best_match(emb)
            best_box = box

        if best is None or best.score < self.memory.threshold:
            if best_box is not None and best is not None:
                self._flash_match(best_box, "未知物体", best.score)
            return self._describe_unknown(best)

        if best_box is not None:
            self._flash_match(best_box, best.user_name, best.score)
        return self._describe_known(best)

    # ----------------------------------------------------------- remember
    def _do_remember(self, label: str) -> str:
        if not label:
            return "你想给它起什么名字？请说：记住，这个叫……"
        if not getattr(self.resnet, "ready", False):
            return (
                "ResNet18 特征模型没有加载，无法录入物品。"
                "请运行 python tools/prepare_models.py --resnet18 后重试。"
            )
        mem_cfg = self.cfg.get("memory", {})
        duration = float(mem_cfg.get("sample_duration_sec", 2.5))
        interval = float(mem_cfg.get("sample_interval_ms", 250)) / 1000.0
        max_samples = int(mem_cfg.get("max_samples", 10))
        min_samples = int(mem_cfg.get("min_samples", 3))

        self.stats.mode = "remember"
        self._overlay.learning = True
        self._overlay.learning_label = f"录入模式：{label}"
        self._overlay.learning_progress = 0.0

        t_end = time.time() + duration
        t_next = time.time()
        samples: List[tuple[np.ndarray, tuple[int, int, int, int], str]] = []
        embeds: List[np.ndarray] = []
        generic_guess = ""

        while time.time() < t_end and len(samples) < max_samples and self._running:
            now = time.time()
            if now >= t_next:
                t_next = now + interval
                frame, _ = self.camera.latest()
                if frame is None:
                    time.sleep(0.02)
                    continue
                self._overlay.learning_progress = 1.0 - (t_end - now) / duration
                dets = self._current_detections()
                crop, box, source = self.cropper.crop_for_learning(frame, dets, generic_class=None)
                if not self.cropper.quality_ok(crop):
                    time.sleep(0.02)
                    continue
                emb = self._embed_crop(crop)
                if emb is None:
                    continue
                # diversity filter — never starve min_samples on a static scene
                force = len(samples) < min_samples
                if embeds and not force:
                    sims = [float(emb @ s) for s in embeds]
                    if max(sims) > 0.995:
                        time.sleep(0.02)
                        continue
                if source == "yolo" and not generic_guess:
                    dets_now = self._current_detections()
                    best_det = self.cropper.best_center_detection(dets_now, frame)
                    if best_det is not None:
                        generic_guess = best_det.label
                embeds.append(emb)
                samples.append((crop, box, source))
                self._overlay.match_box = np.array(box, dtype=np.float32)
            time.sleep(0.02)

        self._overlay.learning_progress = 1.0
        if len(samples) < min_samples:
            self._overlay.learning = False
            if not getattr(self.resnet, "ready", False):
                return f"特征模型未加载，没有记住{label}。请用 vision_ai 环境重试。"
            return (
                f"采样不足（{len(samples)}/{min_samples}），没有记住{label}。"
                "请把物体稳定放在中央区域，稍亮一些再试。"
            )

        # persist images + ResNet18 features
        obj_dir = os.path.join(self.cfg["memory"]["samples_dir"], _safe_name(label))
        os.makedirs(obj_dir, exist_ok=True)
        paths = []
        for i, (crop, box, source) in enumerate(samples):
            p = os.path.join(obj_dir, f"s{i:02d}.png")
            cv2.imwrite(p, crop)
            paths.append(p)
        self.memory.add_object(label, generic_guess or None, embeds, paths)
        st2 = self.memory.stats()
        self.stats.mem_objects = st2["objects"]
        self.stats.mem_info = (
            f"bank {self.stats.mem_objects} objs / {st2['samples']} samples / ResNet18-512d"
        )
        self._overlay.learning = False
        self._overlay.match_box = None
        return f"好的，我已经记住{label}了（{len(samples)}张图片，ResNet18特征）。"

    # ------------------------------------------------------------- search
    def _do_search(self, label: str) -> str:
        if not label:
            return "你要找什么？请说：帮我找……"
        if not getattr(self.resnet, "ready", False):
            return "ResNet18 未加载，无法检索。"
        if self.memory.stats()["objects"] == 0 and not self._current_detections():
            return f"特征库还是空的，画面里也暂未检测到{label}。可先：记住，这个叫……"

        frame, _ = self.camera.latest()
        if frame is None:
            return "摄像头还没有画面。"
        h, w = frame.shape[:2]
        obj = self.memory.find_by_name(label)
        dets = self._current_detections()

        best_box = None
        best_score = -1.0
        best_mem: Optional[MemoryMatch] = None

        if dets:
            for crop, box, det in self.cropper.crop_detections(frame, dets):
                if not self.cropper.quality_ok(crop):
                    continue
                emb = self._embed_crop(crop)
                if emb is None:
                    continue
                m = self.memory.best_match(emb)
                if m is None:
                    continue
                if _name_related(label, m.user_name) or (obj and m.object_id == obj.object_id):
                    if m.score > best_score:
                        best_score = m.score
                        best_box = box
                        best_mem = m

        if best_box is None:
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
        img = cv2.imread(path)
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
        if not label:
            return "缺少物体名称。"
        if not getattr(self.resnet, "ready", False):
            return "ResNet18 特征模型未加载，无法录入。"
        min_samples = int(self.cfg.get("memory", {}).get("min_samples", 3))
        obj_dir = os.path.join(self.cfg["memory"]["samples_dir"], _safe_name(label))
        os.makedirs(obj_dir, exist_ok=True)
        embeds: List[np.ndarray] = []
        saved: List[str] = []
        for i, src in enumerate(paths):
            img = cv2.imread(src)
            if img is None:
                log.warning("unreadable image skipped: %s", src)
                continue
            emb = self.resnet.embed(img)
            if emb is None:
                continue
            dest = os.path.join(obj_dir, f"s{len(embeds):02d}.png")
            cv2.imwrite(dest, img)
            embeds.append(emb)
            saved.append(dest)
        if len(embeds) < min_samples:
            return (
                f"录入失败：有效图片 {len(embeds)}/{len(paths)}，至少需要 {min_samples} 张。"
            )
        self.memory.add_object(label, None, embeds, saved)
        st = self.memory.stats()
        self.stats.mem_objects = st["objects"]
        self._refresh_mem_info()
        return f"好的，我已经记住{label}了（{len(embeds)}张图片，ResNet18特征）。"

    # ------------------------------------------------------------- utils
    def _flash_match(self, box, label: str, score: float) -> None:
        self._overlay.highlight = [(np.asarray(box, dtype=np.float32), label, float(score))]
        self._overlay.match_box = np.asarray(box, dtype=np.float32)
        self._overlay.match_label = label
        self._flash_until = time.time() + 4.0

    def say(self, text: str) -> None:
        self.stats.last_reply = text
        self._set_subtitle(f"助手：{text}", 4.0)
        self.tts.say_async(text)

    def _set_subtitle(self, text: str, seconds: float) -> None:
        self._last_subtitle = text
        self._subtitle_until = time.time() + seconds

    def _refresh_mem_info(self) -> None:
        st = self.memory.stats()
        thr = int(round(st.get("threshold", self.memory.threshold) * 100))
        self.stats.mem_info = (
            f"bank {st['objects']} objs / {st['samples']} samples | "
            f"ResNet18 {self.resnet.device_used} | thr {thr}%"
        )

    # ---------------------------------------------------------------- ui
    def run_ui(self) -> None:
        win = "Object Recognition Demo (ResNet18)"
        if not self.headless:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        ui_target = float(self.cfg.get("app", {}).get("ui_fps_target", 30))
        frame_dt = 1.0 / max(1.0, ui_target)
        fps_ema = 0.0
        last_ui = time.time()

        try:
            while self._running:
                t0 = time.perf_counter()
                if self._need_text_input:
                    if not getattr(self, "_text_fix_announced", False):
                        self._ensure_text_input()
                        self._text_fix_announced = True
                        if not self.use_voice:
                            self.stats.last_reply = "未检测到麦克风，请使用文本输入框发送指令"
                frame, _ = self.camera.latest()
                if frame is None:
                    time.sleep(0.01)
                    continue

                if time.time() > self._flash_until:
                    self._overlay.highlight = []
                    self._overlay.match_box = None

                self._overlay.detections = self._current_detections()
                self._overlay.fps = fps_ema
                self._overlay.status_line = (
                    self.stats.last_reply
                    or "就绪 — 说“这是什么？” / 按 O 识别本地图片 / 按 L 查看列表"
                )
                self._overlay.det_info = self.stats.det_info
                if not self.stats.mem_info:
                    self._refresh_mem_info()
                self._overlay.mem_info = self.stats.mem_info
                self._overlay.subtitle = (
                    self._last_subtitle if time.time() < self._subtitle_until else ""
                )

                self._overlay.input_visible = self._text_input_enabled
                if self.text_input is not None and self.text_input.available:
                    live = self.text_input.display_text
                    self._overlay.input_text = live
                    self._overlay.input_focused = self.text_input.is_typing
                    if not live:
                        self._overlay.input_placeholder = (
                            "在文本窗输入中文并回车（无麦克风）"
                            if not self.use_voice
                            else "输入指令… 或用语音"
                        )
                else:
                    self._overlay.input_text = self._cv_input_buf
                    self._overlay.input_focused = self._cv_input_focused
                    self._overlay.input_placeholder = self._cv_input_placeholder

                vis = self._viz.draw(frame, self._overlay)

                if not self.headless:
                    cv2.imshow(win, vis)
                    key = cv2.waitKey(1) & 0xFF
                    if not self._handle_key(key):
                        break
                else:
                    time.sleep(max(0.0, frame_dt - (time.perf_counter() - t0)))

                now = time.time()
                dt = now - last_ui
                last_ui = now
                if dt > 0:
                    inst = 1.0 / dt
                    fps_ema = inst if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst
                self.stats.ui_fps = fps_ema

                elapsed = time.perf_counter() - t0
                sleep_t = frame_dt - elapsed
                if sleep_t > 0 and not self.headless:
                    time.sleep(sleep_t)
        finally:
            if not self.headless:
                try:
                    cv2.destroyWindow(win)
                except Exception:
                    cv2.destroyAllWindows()

    # ---------------------------------------------------------------- keys
    def _handle_key(self, key: int) -> bool:
        """Process one cv2.waitKey code. Returns False to quit."""
        if key < 0 or key == 255:
            return True

        # ---- in-frame text box focused: type / edit / submit ----
        if self._cv_input_focused:
            if key in (13, 10):  # Enter
                text = self._cv_input_buf.strip()
                self._cv_input_buf = ""
                self._cv_input_focused = False
                if text:
                    self.inject_text(text)
                return True
            if key == 27:  # Esc
                self._cv_input_buf = ""
                self._cv_input_focused = False
                return True
            if key in (8, 127):  # Backspace / Delete
                if self._cv_input_buf:
                    self._cv_input_buf = self._cv_input_buf[:-1]
                return True
            if key == 9:  # Tab — leave focus
                self._cv_input_focused = False
                return True
            if 32 <= key < 127:
                if len(self._cv_input_buf) < 120:
                    self._cv_input_buf += chr(key)
                return True
            return True

        # ---- shortcuts when not typing ----
        if key in (ord("q"), 27):
            return False
        if key in (ord("t"), ord("/"), 13, 10):  # focus text box
            self._cv_input_focused = True
            if key in (ord("t"), ord("/")):
                self._cv_input_buf = ""
            return True
        if key == ord("i"):
            self.inject_text("这是什么")
        elif key == ord("r"):
            self.inject_text("记住这个叫测试物品")
        elif key == ord("s"):
            self.inject_text("帮我找测试物品")
        elif key == ord("l"):
            self.inject_text("你记住了什么")
        elif key == ord("o"):
            self._pick_and_recognize_file()
        elif key == ord("h"):
            log.info(
                "keys: q=quit  t//=text box  i=identify  r=remember  "
                "s=search  l=list  o=open image"
            )
        return True

    def _pick_and_recognize_file(self) -> None:
        """Open a file dialog on the UI thread, recognize the chosen image, speak."""
        path = ""
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            path = filedialog.askopenfilename(
                title="选择要识别的图片",
                filetypes=[
                    ("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
                    ("All files", "*.*"),
                ],
            )
            root.destroy()
        except Exception as e:
            log.warning("file dialog failed: %s", e)
            return
        if not path:
            return
        reply = self.recognize_image_file(path)
        self.stats.last_reply = reply
        self._set_subtitle(f"助手：{reply}", 5.0)
        self._mic_muted_until = time.time() + min(8.0, 1.0 + 0.18 * len(reply))
        self.tts.say_async(reply)


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
