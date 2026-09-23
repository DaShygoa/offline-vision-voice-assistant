"""OpenCV drawing: bboxes, labels, learning ROI, status bar, FPS."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from .detector import Detection

_BLUE = (255, 180, 60)
_GREEN = (80, 220, 80)
_YELLOW = (60, 220, 240)
_RED = (60, 60, 240)
_GRAY = (160, 160, 160)
_WHITE = (240, 240, 240)
_BG = (30, 30, 30)


def _put_text(img, text, org, color, scale=0.55, thickness=1):
    # Windows Chinese fonts
    for font in ("Microsoft YaHei UI", "Microsoft YaHei", "SimHei", "Arial"):
        try:
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            # prefer native Chinese rendering via PIL-like fallback: cv2.putText cannot draw CJK
        except Exception:
            pass
    # Use PIL if available for CJK, else OpenCV (may show ??? for Chinese)
    try:
        from PIL import Image, ImageDraw, ImageFont

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil)
        font_path = None
        import os

        candidates = [
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\msyhbd.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
        ]
        for c in candidates:
            if os.path.isfile(c):
                font_path = c
                break
        font_size = max(14, int(scale * 24))
        if font_path:
            fnt = ImageFont.truetype(font_path, font_size)
        else:
            fnt = ImageFont.load_default()
        draw.text(org, text, font=fnt, fill=(int(color[2]), int(color[1]), int(color[0])))
        out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        img[:] = out
        return img
    except Exception:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        return img


@dataclass
class OverlayState:
    detections: List[Detection] = field(default_factory=list)
    learning: bool = False
    learning_label: str = ""
    learning_progress: float = 0.0
    match_box: Optional[np.ndarray] = None
    match_label: str = ""
    status_line: str = "就绪 — 按住说话或等待语音指令"
    subtitle: str = ""
    fps: float = 0.0
    det_info: str = ""
    mem_info: str = ""
    highlight: List[tuple[np.ndarray, str, float]] = field(default_factory=list)  # (box, label, conf)
    # text input bar (no-microphone fallback)
    input_visible: bool = True
    input_focused: bool = False
    input_text: str = ""
    input_placeholder: str = "输入指令… (t / Enter 聚焦，回车发送，Esc 取消)"


class Visualizer:
    def __init__(self, show_fps: bool = True, center_roi_frac: float = 0.45):
        self.show_fps = show_fps
        self.center_roi_frac = center_roi_frac

    def draw(self, frame: np.ndarray, state: OverlayState) -> np.ndarray:
        h, w = frame.shape[:2]
        img = frame.copy()

        for d in state.detections:
            x1, y1, x2, y2 = [int(v) for v in d.xyxy]
            cv2.rectangle(img, (x1, y1), (x2, y2), _BLUE, 2)
            caption = f"{d.label} {d.conf:.2f}"
            self._label(img, caption, x1, max(0, y1 - 22), _BLUE)

        for box, label, conf in state.highlight:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(img, (x1, y1), (x2, y2), _GREEN, 3)
            self._label(img, f"{label} ({conf:.2f})", x1, max(0, y1 - 24), _GREEN)

        if state.learning:
            rw = int(w * self.center_roi_frac)
            rh = int(h * self.center_roi_frac)
            x1, y1 = w // 2 - rw // 2, h // 2 - rh // 2
            x2, y2 = x1 + rw, y1 + rh
            cv2.rectangle(img, (x1, y1), (x2, y2), _YELLOW, 2)
            # progress
            bar_w = int(rw * max(0.0, min(1.0, state.learning_progress)))
            cv2.rectangle(img, (x1, y2 - 8), (x1 + bar_w, y2), _YELLOW, -1)
            msg = state.learning_label or "请将物体放入中央区域"
            self._center_text(img, msg, h // 2 - rh // 2 - 36, _YELLOW, scale=0.8)
            tip = "保持物体在框内，可轻微转动"
            self._center_text(img, tip, y2 + 18, _GRAY, scale=0.5)

        # status bar (top strip) + input bar (bottom strip)
        cv2.rectangle(img, (0, 0), (w, 48), _BG, -1)  # info bar at top
        if self.show_fps:
            self._label(img, f"FPS {state.fps:.1f}", 10, 8, _WHITE, scale=0.5)
        self._label(img, state.status_line, 10, 28, _WHITE, scale=0.55)
        if state.det_info:
            self._label(img, state.det_info, int(w * 0.42), 8, _GRAY, scale=0.45)
        if state.mem_info:
            self._label(img, state.mem_info, int(w * 0.42), 28, _GRAY, scale=0.45)

        if state.input_visible:
            self._draw_input_bar(img, state)
        else:
            # no input bar → put status at bottom as before
            cv2.rectangle(img, (0, h - 40), (w, h), _BG, -1)
            self._label(img, state.status_line, 10, h - 30, _WHITE, scale=0.55)

        if state.subtitle:
            # subtitle banner (last TTS / ASR text)
            self._center_banner(img, state.subtitle, 64)

        return img

    def _draw_input_bar(self, img: np.ndarray, state: OverlayState) -> None:
        """In-frame text command box — visible without a microphone."""
        h, w = img.shape[:2]
        bar_y = h - 42
        pad = 10
        bar_h = 30
        border = _YELLOW if state.input_focused else (90, 90, 90)
        thickness = 2 if state.input_focused else 1
        x1, y1 = pad, bar_y
        x2, y2 = w - pad, bar_y + bar_h
        # field bg
        cv2.rectangle(img, (x1, y1), (x2, y2), (45, 45, 45), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), border, thickness)

        try:
            from PIL import Image, ImageDraw, ImageFont
            import os

            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            font_path = None
            for c in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
                if os.path.isfile(c):
                    font_path = c
                    break
            fnt = ImageFont.truetype(font_path, 16) if font_path else ImageFont.load_default()
            # prefix
            prefix = "指令> " if state.input_focused else "指令: "
            draw.text((x1 + 8, y1 + 5), prefix, font=fnt, fill=(120, 220, 255) if state.input_focused else (160, 160, 160))
            prefix_w = draw.textbbox((0, 0), prefix, font=fnt)[2]
            body = state.input_text if state.input_text else state.input_placeholder
            color = (245, 245, 245) if state.input_text else (130, 130, 130)
            # caret
            text = body + ("▌" if state.input_focused and int(time.time() * 2) % 2 == 0 else "")
            draw.text((x1 + 8 + prefix_w, y1 + 5), text, font=fnt, fill=color)
            img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            label = f"cmd: {state.input_text or state.input_placeholder}"
            cv2.putText(img, label, (x1 + 8, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

    def _label(self, img, text, x, y, color, scale=0.55):
        try:
            from PIL import Image, ImageDraw, ImageFont
            import os

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil)
            font_path = None
            for c in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
                if os.path.isfile(c):
                    font_path = c
                    break
            font_size = max(13, int(scale * 22))
            fnt = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
            # background chip for readability
            bbox = draw.textbbox((x, y), text, font=fnt)
            draw.rectangle((bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2), fill=(0, 0, 0))
            draw.text((x, y), text, font=fnt, fill=(int(color[2]), int(color[1]), int(color[0])))
            arr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            img[:] = arr
        except Exception:
            cv2.putText(img, text, (x, y + 14), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    def _center_text(self, img, text, y, color, scale=0.7):
        h, w = img.shape[:2]
        try:
            from PIL import Image, ImageDraw, ImageFont
            import os

            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            font_path = None
            for c in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
                if os.path.isfile(c):
                    font_path = c
                    break
            font_size = max(16, int(scale * 26))
            fnt = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
            bbox = draw.textbbox((0, 0), text, font=fnt)
            tw = bbox[2] - bbox[0]
            x = max(4, (w - tw) // 2)
            draw.rectangle((x - 6, y - 4, x + tw + 6, bbox[3] - bbox[1] + y + 6), fill=(0, 0, 0))
            draw.text((x, y), text, font=fnt, fill=(int(color[2]), int(color[1]), int(color[0])))
            img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            cv2.putText(img, text, (40, y + 20), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

    def _center_banner(self, img, text, y):
        h, w = img.shape[:2]
        try:
            from PIL import Image, ImageDraw, ImageFont
            import os

            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            font_path = None
            for c in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
                if os.path.isfile(c):
                    font_path = c
                    break
            fnt = ImageFont.truetype(font_path, 20) if font_path else ImageFont.load_default()
            bbox = draw.textbbox((0, 0), text, font=fnt)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            x = max(8, (w - tw) // 2)
            draw.rounded_rectangle((x - 10, y - 6, x + tw + 10, y + th + 8), radius=6, fill=(20, 20, 20))
            draw.text((x, y), text, font=fnt, fill=(240, 240, 240))
            img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            cv2.putText(img, text, (20, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _WHITE, 2, cv2.LINE_AA)
