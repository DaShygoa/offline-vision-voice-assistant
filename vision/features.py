"""Single-channel feature holder for one crop (ResNet18 only, 选题一)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FeatureBundle:
    embed: Optional[np.ndarray] = None  # L2-normalized ResNet18 512-d

    @property
    def any(self) -> bool:
        return self.embed is not None
