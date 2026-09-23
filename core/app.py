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
        self._learning_boxes: List[np.ndarray] = []

    # ----------------------------------------------------------------- boot
    def start(self) -> None:
        log.info("Loading models …")
        self.detector.load()
        try:
            self.dino.load()
        except Exception as e:
            log.error("DINO failed to load (identify/remember degraded): %s", e)
        if self.use_voice:
            try:
                self.asr.load()
                self.vad.load()
            except Exception as e:
                log.error("ASR load failed: %s", e)
                self.use_voice = False
        self.tts.start()
        self.camera.start()

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
        log.info("Started. memory=%s yolo=%s dino=%s tts=%s", st, self.detector.device_used, self.dino.device_used, self.stats.tts_health)
        self.say("视觉助手已启动。")

    def stop(self) -> None:
        self._running = False
        for t in self._threads:
            t.join(timeout=2.0)
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
            log.warning("Mic unavailable — voice disabled, use keyboard mode")
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

    # ---------------------------------------------------------- identify
    def _do_identify(self) -> str:
        frame, _ = self.camera.latest()
        if frame is None:
            return "摄像头还没有画面。"
        dets = self._current_detections()
        h, w = frame.shape[:2]

        # 1) YOLO detections → crop each → DINO → memory
        if dets:
            crops = self.cropper.crop_detections(frame, dets)
            embeds = []
            for crop, box, det in crops:
                if not self.cropper.quality_ok(crop):
                    embeds.append(None)
                    continue
                e, ms = self.dino.timed_embed(crop)
                self.stats.dino_latency_ms = ms
                embeds.append(e)
            match = self.memory.match_detections(embeds)
            if match is not None:
                idx, m = match
                box = crops[idx][1]
                self._flash_match(box, m.user_name, m.score)
                return self._reframe_known(m.user_name)
            # no memory → generic YOLO name
            top = dets[0]
            return f"这是一个{self._cnn_to_zh(top.label)}。"

        # 2) no YOLO box → central ROI (unknown object)
        box = self.cropper.center_roi(frame)
        crop = self.cropper.crop_box(frame, box)
        if not self.cropper.quality_ok(crop):
            return "我看不清这个物品，试着把它放到画面中央亮一点。"
        e, ms = self.dino.timed_embed(crop)
        self.stats.dino_latency_ms = ms
        if e is not None:
            m = self.memory.best_match(e)
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
                e, ms = self.dino.timed_embed(crop)
                self.stats.dino_latency_ms = ms
                if e is None:
                    continue
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
                samples.append((crop, box, source))
                # draw last box
                self._overlay.match_box = np.array(box, dtype=np.float32)
            time.sleep(0.02)

        self._overlay.learning_progress = 1.0
        if len(samples) < min_samples:
            self._overlay.learning = False
            return f"采样不足，没有记住{label}。请把物体稳定放在中央区域再试一次。"

        # persist
        obj_dir = os.path.join(self.cfg["memory"]["samples_dir"], _safe_name(label))
        os.makedirs(obj_dir, exist_ok=True)
        emb_list = []
        paths = []
        for i, (crop, box, source) in enumerate(samples):
            p = os.path.join(obj_dir, f"s{i:02d}.png")
            cv2.imwrite(p, crop)
            emb_list.append(seen_embeds[i])
            paths.append(p)
        self.memory.add_object(label, generic_guess or None, emb_list, paths)
        self.stats.mem_objects = self.memory.stats()["objects"]
        self.stats.mem_info = f"memory {self.stats.mem_objects} objs / {self.memory.stats()['samples']} samples"
        self._overlay.learning = False
        self._overlay.match_box = None
        return f"好的，我已经记住{label}了。"

    # ------------------------------------------------------------- search
    def _do_search(self, label: str) -> str:
        if not label:
            return "你要找什么？请说：帮我找……"
        frame, _ = self.camera.latest()
        if frame is None:
            return "摄像头还没有画面。"
        h, w = frame.shape[:2]
        obj = self.memory.find_by_name(label)
        dets = self._current_detections()

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
                e, ms = self.dino.timed_embed(crop)
                self.stats.dino_latency_ms = ms
                if e is None:
                    continue
                m = self.memory.best_match(e)
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
                e, _ = self.dino.timed_embed(crop)
                if e is not None:
                    m = self.memory.best_match(e)
                    if m and (_name_related(label, m.user_name) or (obj and m.object_id == obj.object_id)):
                        best_box = box
                        best_score = m.score
                        best_mem = m

        if best_box is None:
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
        key_help = "keys: q=quit  i=identify  r=remember  s=search  t=text command"

        try:
            while self._running:
                t0 = time.perf_counter()
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
                self._overlay.status_line = self.stats.last_reply or "就绪 — 说“这是什么？”试试"
                self._overlay.det_info = self.stats.det_info
                self._overlay.mem_info = self.stats.mem_info or f"memory {self.memory.stats()['objects']} objs | DINO {self.dino.device_used}"
                self._overlay.subtitle = self._last_subtitle if time.time() < self._subtitle_until else ""

                vis = self._viz.draw(frame, self._overlay)

                if not self.headless:
                    cv2.imshow(win, vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    elif key == ord("i"):
                        self.inject_text("这是什么")
                    elif key == ord("r"):
                        self.inject_text("记住这个叫测试物品")
                    elif key == ord("s"):
                        self.inject_text("帮我找测试物品")
                    elif key == ord("t"):
                        try:
                            txt = input("command> ").strip()
                            if txt:
                                self.inject_text(txt)
                        except EOFError:
                            pass
                    elif key == ord("h"):
                        log.info(key_help)
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


def _safe_name(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "·"):
            keep.append(ch)
        else:
            keep.append("_")
    s = "".join(keep).strip("_") or "obj"
    return s[:48]


def _zh_match(query: str, label: str) -> bool:
    zh = AssistantApp._cnn_to_zh(label)
    q = query.replace("我的", "").replace("这个", "").strip()
    return q == label or q == zh or q in zh or zh in q or label in query


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
