"""Spatial position description in Chinese (画面左侧 / 中间 / 右侧 ...)."""

from __future__ import annotations

import numpy as np


def describe_position(
    box_xyxy: np.ndarray | tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
    with_vertical: bool = True,
) -> str:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5

    # horizontal thirds
    if cx < frame_w / 3:
        h = "画面左侧"
    elif cx < 2 * frame_w / 3:
        h = "画面中间"
    else:
        h = "画面右侧"

    if not with_vertical:
        return h

    if cy < frame_h / 3:
        v = "上方"
    elif cy < 2 * frame_h / 3:
        v = ""
    else:
        v = "下方"

    if not v:
        return h
    if h == "画面中间":
        return f"画面{v}"
    return f"{h}{v}"


def relative_direction(box_xyxy, frame_w: int, frame_h: int) -> str:
    """Shorter form: 左边 / 中间 / 右边."""
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    if cx < frame_w / 3:
        h = "左边"
    elif cx < 2 * frame_w / 3:
        h = "中间"
    else:
        h = "右边"
    if cy < frame_h / 3:
        return f"{h}上方"
    if cy > 2 * frame_h / 3:
        return f"{h}下方"
    return h
