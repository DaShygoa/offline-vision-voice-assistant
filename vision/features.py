"""Bundle DINO + shape + depth for one crop — shared by remember/identify/search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FeatureBundle:
    dino: Optional[np.ndarray] = None
    shape: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None

    @property
    def any(self) -> bool:
        return self.dino is not None or self.shape is not None or self.depth is not None

    @property
    def has_dino(self) -> bool:
        return self.dino is not None
