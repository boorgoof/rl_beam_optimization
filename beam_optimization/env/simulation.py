"""
Common simulation contracts for real TraceWin and neural surrogate engines.

Both engines map machine parameters to beam states and a scalar score.  The
source field keeps the physical origin explicit so downstream code can decide
whether a sample is real TraceWin data or a surrogate prediction.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class BeamSimulationResult:
    """Structured output of one beam simulation."""

    params: Dict[str, float]
    beam_states: Optional[np.ndarray]
    score_val: float
    success: bool
    source: str = "tracewin"
    error: Optional[str] = None
    final_beam: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def score(self) -> float:
        return self.score_val


class BeamSimulator(ABC):
    """Common interface for engines that simulate beam transport."""

    def reset_context(self, rng=None) -> None:
        """Prepare per-episode context.

        TraceWin has a fixed input beam in the project files, so it can ignore
        this.  The surrogate uses it to sample beam0 and choose an ensemble
        member for the episode.
        """
        pass

    @abstractmethod
    def simulate(self, params: Dict[str, float]) -> BeamSimulationResult:
        """Run one simulation for a parameter dictionary."""
        raise NotImplementedError
