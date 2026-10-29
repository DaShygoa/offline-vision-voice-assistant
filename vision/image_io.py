"""Unicode-safe image I/O.

OpenCV's cv2.imwrite() silently fails on Windows when the path contains
non-ASCII characters (it returns True but writes nothing) — e.g. saving
enrollment samples to "data/samples/手机/s00.png". Work around it by encoding
to a memory buffer and writing the bytes with regular file I/O.
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np


def imwrite_unicode(path: str, image: np.ndarray, params: Optional[list] = None) -> bool:
    """Write an image to a path that may contain non-ASCII characters."""
    if image is None or image.size == 0:
        return False
    ext = os.path.splitext(path)[1] or ".png"
    try:
        ok, buf = cv2.imencode(ext, image, params or [])
        if not ok:
            return False
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        return True
    except Exception:
        return False


def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Read an image from a path that may contain non-ASCII characters."""
    try:
        if not os.path.isfile(path):
            return None
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size == 0:
            return None
        img = cv2.imdecode(buf, flags)
        return img
    except Exception:
        return None
