"""Main application orchestrator.

Threading model (deployment rules):
  ┌ main thread      : OpenCV camera UI loop @ 25–30 FPS (never blocks on TTS/ASR/DINO)
  ├ camera thread    : frame capture → latest() buffer
  ├ detect thread    : YOLO every N frames (≥5 det FPS), updates shared detections
  ├ voice thread     : mic → VAD → ASR → intent → queue
  ├ intent worker    : identify / remember / search logic (DINO + memory) → reply text → TTS queue
  └ tts worker       : GPT-SoVITS independent client (async; UI does not wait)

DINO only runs on identify / remember / search — never in the camera hot path.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from nlu.intent import Intent, IntentType, parse_intent
from vision.camera import Camera
from vision.detector import Detection, YoloDetector
from vision.feature_extractor import DinoExtractor
from vision.memory import MemoryMatch, VisualMemory
from vision.object_cropper import ObjectCropper
from vision.position import describe_position
from vision.visualizer import OverlayState, Visualizer
from vision.shape_descriptor import ShapeDescriptor
from vision.depth import DepthExtractor
from vision.features import FeatureBundle
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
    dino_latency_ms: float = 0.0
    asr_latency_ms: float = 0.0
    last_intent: str = ""
    last_reply: str = ""
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
        self.detector = YoloDetector(cfg.get("yolo", {}))
        self.dino = DinoExtractor(cfg.get("dino", {}))
        self.shape = ShapeDescriptor(cfg.get("shape", {}))
        self.depth = DepthExtractor(cfg.get("depth", {}))
        mem_cfg = dict(cfg.get("memory", {}))
        # keep fusion flags consistent with extractors
        mem_cfg.setdefault("use_shape", self.shape.enabled)
        mem_cfg.setdefault("use_depth", self.depth.enabled)
        self.memory = VisualMemory(mem_cfg)
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
        self._learning_boxes: List[np.ndarray] = []

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
        self.detector.load()
        try:
            self.dino.load()
        except Exception as e:
            log.error("DINO failed to load (identify/remember degraded): %s", e)
            self._dino_error = str(e)
        if not getattr(self.dino, "ready", False):
            if "No module named" in str(getattr(self, "_dino_error", "")) or self.dino.device_used == "none":
                log.error(
                    "提示: 识别/记忆需要 DINO + onnxruntime/openvino。"
                    " 请用 conda activate vision_ai 后再运行。"
                )
        # shape (pure OpenCV) + depth (MiDaS) side channels
        try:
            self.depth.load()
        except Exception as e:
            log.warning("Depth load failed (depth channel off): %s", e)
        if self.depth.ready:
            log.info("Depth ready: %s dim=%d", self.depth.device_used, self.depth.dim)
        else:
            log.info("Depth channel off (%s)", self.depth._error or "disabled")
        log.info(
            "Shape ready: enabled=%s dim=%d fusion w=[dino %.2f shape %.2f depth %.2f]",
            self.shape.enabled,
            self.shape.dim,
            self.memory.w_dino,
            self.memory.w_shape,
            self.memory.w_depth,
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
            # with mic: still show bar, Tk window optional (off unless text_input_always)
            self._overlay.input_visible = True

        self._running = True
        self._threads = [
            threading.Thread(target=self._detect_loop, name="detect", daemon=True),
            threading.Thread(target=self._intent_loop, name="intent", daemon=True),
        ]
        if self.use_voice:
            self._threads.append(threading.Thread(target=self._voice_loop, name="voice", daemon=True))
        for t in self._threads:
            t.start()

        st = self.memory.stats()
        self.stats.mem_objects = st["objects"]
        self.stats.tts_health = self.tts.health()
        log.info(
            "Started. memory=%s yolo=%s dino=%s shape=%s depth=%s tts=%s voice=%s text_input=%s",
            st,
            self.detector.device_used,
            self.dino.device_used,
            "on" if self.shape.enabled else "off",
            self.depth.device_used if self.depth.ready else "off",
            self.stats.tts_health,
            self.use_voice,
            "ime" if self.text_input else ("frame" if self._text_input_enabled else "off"),
        )
        if not self.use_voice:
            self.say("未检测到麦克风，可使用文本输入框发送指令。")
        else:
            self.say("视觉助手已启动。")

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
                self.stats.det_info = f"YOLO {self.detector.device_used} | {top.label}:{top.conf:.2f} (+{len(dets)-1}) @{dt:.0f}ms"
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
        sr = self.mic.sample_rate
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
                # sherpa VAD path: finished signal
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

    # inject text as if spoken (keyboard test mode)
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
                return f"好的，我已经忘记{intent.label}了。"
            return f"没有找到叫{intent.label or '它'}的记忆。"
        if intent.type == IntentType.LIST:
            objs = self.memory.list_objects()
            if not objs:
                return "我还没有记住任何物品。你可以对我说：记住，这个叫……"
            names = "、".join(o.user_name for o in objs[:8])
            more = f"等共{len(objs)}个物品" if len(objs) > 8 else ""
            return f"我记住了：{names}{more}。"
        if intent.type == IntentType.HELP:
            return "你可以问：这是什么？教我记住这个叫……，或者说：帮我找我的水杯。"
        return "我没太听懂，可以说：这是什么？"

    def _extract_bundle(
        self,
        frame: np.ndarray,
        crop: np.ndarray,
        box: tuple[int, int, int, int],
    ) -> FeatureBundle:
        """DINO + OpenCV shape + monocular depth for one crop (intent thread only)."""
        bundle = FeatureBundle()
        if getattr(self.dino, "ready", False):
            e, ms = self.dino.timed_embed(crop)
            self.stats.dino_latency_ms = ms
            bundle.dino = e
        if self.shape.enabled:
            sv, _ = self.shape.timed_extract(crop, box)
            bundle.shape = sv
        if self.depth.ready:
            dv, _ = self.depth.timed_extract(crop, frame_bgr=frame, box=box)
            bundle.depth = dv
        return bundle

    # ---------------------------------------------------------- identify
    def _do_identify(self) -> str:
        frame, _ = self.camera.latest()
        if frame is None:
            return "摄像头还没有画面。"
        dets = self._current_detections()
        h, w = frame.shape[:2]
        dino_ok = getattr(self.dino, "ready", False)

        # 1) YOLO detections → crop each → multi-channel memory
        if dets:
            if not dino_ok:
                top = dets[0]
                return (
                    f"检测到{self._cnn_to_zh(top.label)}（YOLO），"
                    "但特征模型 DINO 未加载，无法匹配记忆。请用 vision_ai 环境运行。"
                )
            crops = self.cropper.crop_detections(frame, dets)
            embeds: List[Optional[np.ndarray]] = []
            shapes: List[Optional[np.ndarray]] = []
            depths: List[Optional[np.ndarray]] = []
            for crop, box, det in crops:
                if not self.cropper.quality_ok(crop):
                    embeds.append(None)
                    shapes.append(None)
                    depths.append(None)
                    continue
                b = self._extract_bundle(frame, crop, box)
                embeds.append(b.dino)
                shapes.append(b.shape)
                depths.append(b.depth)
            match = self.memory.match_detections(embeds, shapes, depths)
            if match is not None:
                idx, m = match
                box = crops[idx][1]
                self._flash_match(box, m.user_name, m.score)
                return self._reframe_known(m.user_name)
            # no memory → generic YOLO name
            top = dets[0]
            return f"这是一个{self._cnn_to_zh(top.label)}。"

        # 2) no YOLO box → central ROI (unknown object)
        if not dino_ok:
            return "特征模型 DINO 未加载，无法识别记忆中的物品。请用 vision_ai 环境运行。"
        box = self.cropper.center_roi(frame)
        crop = self.cropper.crop_box(frame, box)
        if not self.cropper.quality_ok(crop):
            return "我看不清这个物品，试着把它放到画面中央亮一点。"
        b = self._extract_bundle(frame, crop, box)
        if b.any:
            m = self.memory.best_match(b.dino, shape_embedding=b.shape, depth_embedding=b.depth)
            if m is not None:
                self._flash_match(box, m.user_name, m.score)
                return self._reframe_known(m.user_name)
        return "我还不认识这个物品。你可以教我：记住，这个叫……"

    @staticmethod
    def _reframe_known(name: str) -> str:
        n = name.strip()
        if n.startswith("我的") or n.startswith("你的") or n.startswith("他的"):
            return f"这是{n}。"
        return f"这是你的{n}。"

    # ----------------------------------------------------------- remember
    def _do_remember(self, label: str) -> str:
        if not label:
            return "你想给它起什么名字？请说：记住，这个叫……"
        if not getattr(self.dino, "ready", False):
            return (
                "视觉特征模型没有加载（DINO 未就绪），无法记住物品。"
                "请用 conda activate vision_ai 运行。"
            )
        mem_cfg = self.cfg.get("memory", {})
        duration = float(mem_cfg.get("sample_duration_sec", 2.5))
        interval = float(mem_cfg.get("sample_interval_ms", 250)) / 1000.0
        max_samples = int(mem_cfg.get("max_samples", 10))
        min_samples = int(mem_cfg.get("min_samples", 3))

        self.stats.mode = "remember"
        self._overlay.learning = True
        self._overlay.learning_label = f"学习模式：{label}"
        self._overlay.learning_progress = 0.0

        t_end = time.time() + duration
        t_next = time.time()
        samples: List[tuple[np.ndarray, tuple[int, int, int, int], str]] = []
        seen_embeds: List[np.ndarray] = []
        seen_bundles: List[FeatureBundle] = []
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
                b = self._extract_bundle(frame, crop, box)
                if b.dino is None:
                    continue
                e = b.dino
                # diversity filter — never starve min_samples on a static scene
                force = len(samples) < min_samples
                if seen_embeds and not force:
                    sims = [float(e @ s) for s in seen_embeds]
                    if max(sims) > 0.995:
                        time.sleep(0.02)
                        continue
                # record generic class from best center det
                if source == "yolo" and not generic_guess:
                    best = self.cropper.best_center_detection(dets, frame)
                    if best is not None:
                        generic_guess = best.label
                seen_embeds.append(e)
                seen_bundles.append(b)
                samples.append((crop, box, source))
                # draw last box
                self._overlay.match_box = np.array(box, dtype=np.float32)
            time.sleep(0.02)

        self._overlay.learning_progress = 1.0
        if len(samples) < min_samples:
            self._overlay.learning = False
            if not getattr(self.dino, "ready", False):
                return f"特征模型未加载，没有记住{label}。请用 vision_ai 环境重试。"
            return f"采样不足（{len(samples)}/{min_samples}），没有记住{label}。请把物体稳定放在中央区域，稍亮一些再试。"

        # persist
        obj_dir = os.path.join(self.cfg["memory"]["samples_dir"], _safe_name(label))
        os.makedirs(obj_dir, exist_ok=True)
        emb_list = []
        shape_list = []
        depth_list = []
        paths = []
        for i, (crop, box, source) in enumerate(samples):
            p = os.path.join(obj_dir, f"s{i:02d}.png")
            cv2.imwrite(p, crop)
            b = seen_bundles[i]
            emb_list.append(b.dino)
            shape_list.append(b.shape)
            depth_list.append(b.depth)
            paths.append(p)
        self.memory.add_object(
            label,
            generic_guess or None,
            emb_list,
            paths,
            shape_embeddings=shape_list,
            depth_embeddings=depth_list,
        )
        self.stats.mem_objects = st2["objects"]
        n_ch = 1 + int(st2["shape_shape"][0] > 0) + int(st2["depth_shape"][0] > 0)
        self.stats.mem_info = (
            f"memory {self.stats.mem_objects} objs / {st2['samples']} samples / {n_ch}ch"
        )
        self._overlay.learning = False
        self._overlay.match_box = None
        return f"好的，我已经记住{label}了。"

    # ------------------------------------------------------------- search
    def _do_search(self, label: str) -> str:
        if not label:
            return "你要找什么？请说：帮我找……"
        if self.memory.stats()["objects"] == 0 and not self._current_detections():
            return (
                f"记忆里还没有物品，画面里也暂未检测到{label}。"
                "可先：记住，这个叫……"
            )
        frame, _ = self.camera.latest()
        if frame is None:
            return "摄像头还没有画面。"
        h, w = frame.shape[:2]
        obj = self.memory.find_by_name(label)
        dets = self._current_detections()
        if not obj and self.memory.stats()["objects"] == 0 and not dets:
            return f"画面里没有检测到{label}。记忆是空的，可先记住它。"

        # Scan strategy:
        #  a) if label matches a remembered object → DINO every YOLO box (and center ROI)
        #  b) else treat label as generic class name filter on YOLO
        best_box = None
        best_score = -1.0
        best_mem: Optional[MemoryMatch] = None

        if dets:
            crops = self.cropper.crop_detections(frame, dets)
            for crop, box, det in crops:
                if not self.cropper.quality_ok(crop):
                    continue
                # generic class shortcut
                if not obj and _zh_match(label, det.label):
                    score = 0.99
                    if score > best_score:
                        best_score, best_box = score, box
                    continue
                b = self._extract_bundle(frame, crop, box)
                if not b.any:
                    continue
                m = self.memory.best_match(
                    b.dino, shape_embedding=b.shape, depth_embedding=b.depth
                )
                if m is None:
                    continue
                # name similarity: matched object name should relate to query label
                if _name_related(label, m.user_name) or (obj and m.object_id == obj.object_id):
                    if m.score > best_score:
                        best_score = m.score
                        best_box = box
                        best_mem = m

        # center ROI fallback (unknown object sitting in middle)
        if best_box is None:
            box = self.cropper.center_roi(frame)
            crop = self.cropper.crop_box(frame, box)
            if self.cropper.quality_ok(crop):
                b = self._extract_bundle(frame, crop, box)
                if b.any:
                    m = self.memory.best_match(
                        b.dino, shape_embedding=b.shape, depth_embedding=b.depth
                    )
                    if m and (_name_related(label, m.user_name) or (obj and m.object_id == obj.object_id)):
                        best_box = box
                        best_score = m.score
                        best_mem = m

        if best_box is None:
            if not getattr(self.dino, "ready", False):
                return f"画面里没有检测到{label}，且 DINO 未加载，无法查记忆。请用 vision_ai 环境。"
            if self.memory.stats()["objects"] == 0:
                return f"画面里没有检测到{label}。记忆是空的，可先：记住，这个叫……"
            return f"画面里没有找到{label}。"
        pos = describe_position(best_box, w, h)
        name = best_mem.user_name if best_mem else label
        self._flash_match(best_box, name, max(best_score, 0.0))
        n = name.strip()
        if n.startswith("我的") or n.startswith("你的"):
            return f"找到{n}了，在{pos}。"
        return f"找到你的{n}了，在{pos}。"

    # ------------------------------------------------------------- utils
    def _flash_match(self, box, label: str, score: float) -> None:
        self._overlay.highlight = [(np.asarray(box, dtype=np.float32), label, float(score))]
        self._overlay.match_box = np.asarray(box, dtype=np.float32)
        self._overlay.match_label = label
        # clear after 4s via timestamp
        self._flash_until = time.time() + 4.0

    def say(self, text: str) -> None:
        self.stats.last_reply = text
        self._set_subtitle(f"助手：{text}", 4.0)
        self.tts.say_async(text)

    def _set_subtitle(self, text: str, seconds: float) -> None:
        self._last_subtitle = text
        self._subtitle_until = time.time() + seconds

    @staticmethod
    def _cnn_to_zh(label: str) -> str:
        mapping = {
            "cup": "杯子", "bottle": "瓶子", "cell phone": "手机", "mouse": "鼠标",
            "keyboard": "键盘", "laptop": "笔记本电脑", "book": "书", "chair": "椅子",
            "cat": "猫", "dog": "狗", "apple": "苹果", "person": "人", "tv": "电视",
            "remote": "遥控器", "scissors": "剪刀", "teddy bear": "泰迪熊",
            "banana": "香蕉", "orange": "橙子", "bowl": "碗", "knife": "刀",
            "fork": "叉子", "spoon": "勺子", "sink": "水槽", "refrigerator": "冰箱",
            "microwave": "微波炉", "oven": "烤箱", "toothbrush": "牙刷", "vase": "花瓶",
            "clock": "时钟", "potted plant": "盆栽", "bed": "床", "couch": "沙发",
            "dining table": "餐桌", "toilet": "马桶", "backpack": "背包", "umbrella": "伞",
            "handbag": "手提包", "suitcase": "行李箱", "tie": "领带", "suitcase": "行李箱",
        }
        return mapping.get(label, label)

    # ---------------------------------------------------------------- ui
    def run_ui(self) -> None:
        win = "Offline Vision Voice Assistant"
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
                    # mic failed in voice thread → open IME window on UI thread once
                    if not getattr(self, "_text_fix_announced", False):
                        self._ensure_text_input()
                        self._text_fix_announced = True
                        if not self.use_voice:
                            self.stats.last_reply = "未检测到麦克风，请使用文本输入框发送指令"
                frame, _ = self.camera.latest()
                if frame is None:
                    time.sleep(0.01)
                    continue

                # clear expired highlight
                if hasattr(self, "_flash_until") and time.time() > self._flash_until:
                    self._overlay.highlight = []
                    self._overlay.match_box = None

                self._overlay.detections = self._current_detections()
                self._overlay.fps = fps_ema
                self._overlay.status_line = self.stats.last_reply or "就绪 — 说“这是什么？”或用文本框输入"
                self._overlay.det_info = self.stats.det_info
                self._overlay.mem_info = self.stats.mem_info or f"memory {self.memory.stats()['objects']} objs | DINO {self.dino.device_used} | shape/depth multi-ch"
                self._overlay.subtitle = self._last_subtitle if time.time() < self._subtitle_until else ""

                # text input bar: prefer Tk live/echo, else in-frame buffer
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

                # pace to target
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
            # printable ASCII
            if 32 <= key < 127:
                if len(self._cv_input_buf) < 120:
                    self._cv_input_buf += chr(key)
                return True
            # other keys ignored while typing (so i/r/s don't fire)
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
        elif key == ord("h"):
            log.info("keys: q=quit  t//=text box  i=identify  r=remember  s=search  l=list")
        return True


def _safe_name(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "·"):
            keep.append(ch)
        else:
            keep.append("_")
    s = "".join(keep).strip("_") or "obj"
    return s[:48]


# Chinese query → COCO English labels (substring match both ways)
_ZH_COCO_ALIASES: dict[str, tuple[str, ...]] = {
    "显示器": ("tv", "laptop", "monitor"),
    "显示屏": ("tv", "laptop", "monitor"),
    "屏幕": ("tv", "laptop", "monitor", "cell phone"),
    "电脑": ("laptop", "tv", "keyboard", "mouse"),
    "笔记本电脑": ("laptop",),
    "笔记本": ("laptop",),
    "电视机": ("tv",),
    "电视": ("tv",),
    "手机": ("cell phone",),
    "鼠标": ("mouse",),
    "键盘": ("keyboard",),
    "杯子": ("cup", "bottle"),
    "水杯": ("cup", "bottle"),
    "瓶子": ("bottle",),
    "罐子": ("bottle",),
    "笔": ("pen", "toothbrush"),  # pens often mis-labeled toothbrush
    "签字笔": ("pen", "toothbrush"),
    "圆珠笔": ("pen", "toothbrush"),
    "书": ("book",),
    "书本": ("book",),
    "椅子": ("chair",),
    "凳子": ("chair",),
    "猫": ("cat",),
    "狗": ("dog",),
    "苹果": ("apple",),
    "人": ("person",),
    "手机": ("cell phone",),
    "书包": ("backpack",),
    "背包": ("backpack",),
    "剪刀": ("scissors",),
    "花瓶": ("vase",),
    "时钟": ("clock",),
    "钟": ("clock",),
    "遥控器": ("remote",),
    "冰箱": ("refrigerator",),
    "微波炉": ("microwave",),
    "烤箱": ("oven",),
    "水槽": ("sink",),
    "马桶": ("toilet",),
    "沙发": ("couch",),
    "床": ("bed",),
    "盆栽": ("potted plant",),
    "盆景": ("potted plant",),
    "牙刷": ("toothbrush",),
    "碗": ("bowl",),
    "刀": ("knife",),
    "叉子": ("fork",),
    "勺子": ("spoon",),
    "盘子": ("bowl", "sink"),
    "泰迪熊": ("teddy bear",),
    "玩偶": ("teddy bear",),
    "雨伞": ("umbrella",),
    "伞": ("umbrella",),
    "领带": ("tie",),
    "公文包": ("handbag", "suitcase"),
    "行李箱": ("suitcase",),
    "包": ("handbag", "backpack", "suitcase"),
}


def _zh_match(query: str, label: str) -> bool:
    zh = AssistantApp._cnn_to_zh(label)
    q = query.replace("我的", "").replace("这个", "").strip()
    if q == label or q == zh or q in zh or zh in q or label in query:
        return True
    # synonym: Chinese query → any COCO alias matching this det label
    for cq, aliases in _ZH_COCO_ALIASES.items():
        if cq == q or cq in q or q in cq:
            for a in aliases:
                if a == label or a == zh:
                    return True
    # reverse: English label stem in Chinese aliases of query
    return False


def _name_related(query: str, stored: str) -> bool:
    q = query.replace("我的", "").strip()
    s = stored.replace("我的", "").strip()
    if not q or not s:
        return True
    if q == s or q in s or s in q:
        return True
    # token overlap for Chinese (char bigram-ish)
    inter = len(set(q) & set(s))
    return inter >= max(1, min(len(q), len(s)) // 2)
