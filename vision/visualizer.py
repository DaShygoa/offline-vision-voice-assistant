"""Shared overlay state for the classroom release (Web UI only).

The browser layer (web/server.py) draws its own frames with cv2, so all that
remains here is the small mutable state object the intent worker writes to:
match highlights, enrollment progress and subtitle text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class OverlayState:
    learning: bool = False
    learning_label: str = ""
    learning_progress: float = 0.0
    match_box: Optional[np.ndarray] = None
    match_label: str = ""
    highlight: List[tuple] = field(default_factory=list)  # (box, label, conf)
