from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

import numpy as np


class ProgressType(Enum):
    CONCORDANCE = 0
    SAMPLING = 1
    OPTIMAL = 2


@dataclass
class SurfaceState:
    """Per-surface state that flows through the CDC pipeline."""

    S_runs: np.ndarray
    Smed: np.ndarray
    Delta: float
    mono: bool
    pickable: bool
    optima_ma: np.ndarray
    rows: List[Dict] = field(default_factory=list)
    rejected: List[Dict] = field(default_factory=list)

