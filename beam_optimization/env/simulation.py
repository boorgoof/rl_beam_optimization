"""
TraceWin and the surrogate model are the two engines that power the gym environments.
Both do exactly the same thing but TraceWin is more accurate while the surrogate is faster
So both implement the same interface (BeamSimulator), which defines the simulate method: 
takes machine parameters as input and returns a BeamSimulationResult that mainly contains the beam description 
at the output stages plus a score (of the final stage).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class BeamSimulationResult:
    """"One simulation run's output.

    Returns:
    - beam_states wihich describes the beam at each output stage.
    - score_val wihich is the scalar score at the final stage.
    - source wihich tells you whether the result came from TraceWin or the surrogate.
    """

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
    """Interface that both TraceWin and the surrogate implement."""

    def reset_context(self, rng=None) -> None:
        """Reset state before a new episode.

        TraceWin ignores this because its input beam is fixed in the project
        files. The surrogate uses it to sample beam0 and choose an ensemble
        member for the episode.
        """
        pass

    @abstractmethod
    def simulate(self, params: Dict[str, float]) -> BeamSimulationResult:
        """Run one simulation for a given set of machine parameters."""
        raise NotImplementedError
