"""
SurrogateBeamSimulator — BeamSimulator backed by ModularMLP.

This is the surrogate counterpart of TraceWinSimulator: it maps parameters to a
BeamSimulationResult, but everything stays in RAM and no TraceWin files are
written.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import torch

from beam_optimization.config.adige import (
    BEAM_STATE_DIM, BEAM_STATE_VARS,
    params_to_stage_tensors, score,
)
from beam_optimization.env.base_beam_env import ERROR_SCORE
from beam_optimization.env.simulation import BeamSimulationResult, BeamSimulator
from beam_optimization.env.surrogate_env.surrogate.dataset import SurrogateTrainingDataset
from beam_optimization.env.surrogate_env.surrogate.modular_mlp import ModularMLP


class SurrogateBeamSimulator(BeamSimulator):
    """Fast beam simulator using one ModularMLP or an ensemble."""

    def __init__(
        self,
        model: Union[ModularMLP, List[ModularMLP]],
        dataset: SurrogateTrainingDataset,
        beam0_mode: str = "dataset",
        device: Optional[str] = None,
    ):
        self._ensemble = model if isinstance(model, list) else [model]
        self.model = self._ensemble[0]
        self.dataset = dataset
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        for m in self._ensemble:
            m.eval()
            m.to(self.device)

        if beam0_mode not in ("dataset", "gaussian"):
            raise ValueError(f"beam0_mode must be 'dataset' or 'gaussian', got {beam0_mode!r}")
        self.beam0_mode = beam0_mode

        self._initial_beam_states = dataset.get_initial_beam_states()
        self._beam0_mean = self._initial_beam_states.mean(0).numpy().astype(np.float32)
        self._beam0_std = self._initial_beam_states.std(0).numpy().astype(np.float32)
        self._episode_beam0 = np.zeros(BEAM_STATE_DIM, dtype=np.float32)
        self._active_model_index = 0
        self.reset_context()

    def reset_context(self, rng=None) -> None:
        if rng is None:
            rng = np.random.default_rng()

        if len(self._ensemble) > 1:
            midx = int(rng.integers(0, len(self._ensemble)))
            self.model = self._ensemble[midx]
            self._active_model_index = midx
        else:
            self.model = self._ensemble[0]
            self._active_model_index = 0

        if self.beam0_mode == "gaussian":
            self._episode_beam0 = (
                rng.standard_normal(BEAM_STATE_DIM).astype(np.float32)
                * self._beam0_std + self._beam0_mean
            )
        else:
            n = self._initial_beam_states.shape[0]
            idx = int(rng.integers(0, n))
            self._episode_beam0 = self._initial_beam_states[idx].numpy().astype(np.float32)

    def simulate(self, params: Dict[str, float]) -> BeamSimulationResult:
        try:
            beam0_t = torch.tensor(
                self._episode_beam0, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            stage_tensors = params_to_stage_tensors(params, device=self.device)

            with torch.no_grad():
                outputs = self.model(stage_tensors, beam0_t)

            all_stages = [self._episode_beam0]
            for t in outputs:
                all_stages.append(t.squeeze(0).cpu().numpy().astype(np.float32))
            beam_states = np.asarray(all_stages, dtype=np.float32)

            final_beam = {
                v: float(beam_states[-1][i])
                for i, v in enumerate(BEAM_STATE_VARS)
            }
            score_val = score(final_beam)

            return BeamSimulationResult(
                params=params.copy(),
                beam_states=beam_states,
                score_val=score_val,
                success=True,
                source="surrogate",
                final_beam=final_beam,
                metadata={
                    "beam0": self._episode_beam0.copy(),
                    "beam0_mode": self.beam0_mode,
                    "model_index": self._active_model_index,
                },
            )
        except Exception as exc:
            return BeamSimulationResult(
                params=params.copy(),
                beam_states=None,
                score_val=ERROR_SCORE,
                success=False,
                source="surrogate",
                error=str(exc),
                metadata={
                    "beam0": self._episode_beam0.copy(),
                    "beam0_mode": self.beam0_mode,
                    "model_index": self._active_model_index,
                },
            )
