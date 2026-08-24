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
from typing import Any, Dict, List, Optional

import numpy as np
import torch


PHYSICS_FAILURE_PATTERNS = (
    ("all particles are lost", "all_particles_lost"),
    (
        "synchronous particle never reaches the end of the field map",
        "synchronous_particle_never_reaches_end",
    ),
    (
        "part of the beam distribution never reaches the end of the field map",
        "partial_beam_never_reaches_end",
    ),
)


def canonical_physics_failure_reason(message: Optional[str]) -> Optional[str]:
    """Map a known TraceWin physics-failure message to a stable reason code."""
    normalized = (message or "").casefold()
    for pattern, reason in PHYSICS_FAILURE_PATTERNS:
        if pattern in normalized:
            return reason
    return None


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


@dataclass
class DifferentiableBeamSimulationResult:
    """Autograd-preserving output of a surrogate simulation.

    Unlike :class:`BeamSimulationResult`, every physical value remains a torch
    tensor connected to the input parameters. ``beam_states`` contains the
    predicted output stages; the separately stored ``beam0`` is the input
    stage.
    """

    beam0: torch.Tensor
    params: torch.Tensor
    beam_states: List[torch.Tensor]
    final_beam: torch.Tensor
    score: torch.Tensor
    model_index: int

    def detach(self) -> "DifferentiableBeamSimulationResult":
        """Return an equivalent result disconnected from its autograd graph."""
        beam_states = [stage.detach() for stage in self.beam_states]
        return DifferentiableBeamSimulationResult(
            beam0=self.beam0.detach(),
            params=self.params.detach(),
            beam_states=beam_states,
            final_beam=self.final_beam.detach(),
            score=self.score.detach(),
            model_index=self.model_index,
        )


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
