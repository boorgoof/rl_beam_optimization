"""
It is a class that maps ADIGE parameters to a BeamSimulationResult using a surrogate model (ModularMLP or an ensemble of them). It is used in the SurrogateEnv.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from beam_optimization.config.adige import (
    BEAM_STATE_DIM, BEAM_STATE_FEATURES, ERROR_SCORE,
    params_to_stage_tensors, score,
)
from beam_optimization.env.simulation import BeamSimulationResult, BeamSimulator
from beam_optimization.env.dataset import BeamDataset
from beam_optimization.env.surrogate_env.surrogate.model.failure_classifier import FailureClassifier
from beam_optimization.env.surrogate_env.surrogate.model.modular_mlp import ModularMLP


def run_surrogate_forward(
    model: ModularMLP,
    beam0: torch.Tensor,
    params: Dict[str, float],
    device: torch.device,
    classifier: Optional[FailureClassifier] = None,
    classifier_threshold: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, float], float, bool]:
    """Run one surrogate forward pass and score it.

    Args:
        model:  A single ModularMLP, already .eval() and on `device`.
        beam0:  Initial beam state tensor, shape (1, BEAM_STATE_DIM), on `device`.
        params: Flat parameter dict (PARAM_KEYS keys).
        device: Device to build stage tensors on (must match `beam0`'s device).
        classifier: Optional shared FailureClassifier. When given and its
            predicted failure probability exceeds `classifier_threshold`,
            `score_val` is overridden to ERROR_SCORE -- the regressor alone
            can never predict npart_ratio exactly 0, so it cannot reproduce
            the cliff score() checks near the all-particles-lost boundary.
            The regressor still always runs (beam_states is needed
            regardless of the score), so there is no compute saved -- this
            only corrects the score.

    Returns:
        beam_states: (N_STAGES, BEAM_STATE_DIM) float32 ndarray,
                     beam0 followed by each stage output.
        final_beam:  Dict mapping BEAM_STATE_FEATURES -> float, from the last stage.
        score_val:   float, score(final_beam), or ERROR_SCORE if the classifier
                     flagged failure.
        classifier_flagged_failure: bool, False when `classifier` is None.

    Does not catch exceptions: callers keep their own try/except and failure-shape handling.
    """

    # Convert parameters to stage tensors for input
    stage_tensors = params_to_stage_tensors(params, device=device)

    # Run the model
    with torch.no_grad():
        outputs = model(stage_tensors, beam0)

    # build the output for stages
    all_stages = [beam0.squeeze(0).cpu().numpy().astype(np.float32)]
    for t in outputs:
        all_stages.append(t.squeeze(0).cpu().numpy().astype(np.float32))
    beam_states = np.asarray(all_stages, dtype=np.float32)

    # take the finale stage and the relarive score
    final_beam = {
        v: float(beam_states[-1][i])
        for i, v in enumerate(BEAM_STATE_FEATURES)
    }
    score_val = score(final_beam)

    classifier_flagged_failure = False
    if classifier is not None:
        proba = classifier.predict_proba(stage_tensors, beam0)
        classifier_flagged_failure = bool(proba.item() > classifier_threshold)
        if classifier_flagged_failure:
            score_val = ERROR_SCORE

    return beam_states, final_beam, score_val, classifier_flagged_failure


class SurrogateBeamSimulator(BeamSimulator):
    """Fast beam simulator using one ModularMLP or an ensemble.
    """

    def __init__(
        self,
        model: Union[ModularMLP, List[ModularMLP]],
        dataset: BeamDataset,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        classifier: Optional[FailureClassifier] = None,
        classifier_threshold: float = 0.5,
    ):

        # Initialize the SurrogateBeamSimulator with the given parameters.
        self._ensemble = model if isinstance(model, list) else [model]
        self.model = self._ensemble[0]
        self.dataset = dataset
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.classifier = classifier
        self.classifier_threshold = float(classifier_threshold)
        if self.classifier is not None:
            self.classifier.eval()
            self.classifier.to(self.device)

        for m in self._ensemble:
            m.eval()
            m.to(self.device)

        self._initial_beam_states = dataset.get_initial_beam_states()
        self._episode_beam0 = np.zeros(BEAM_STATE_DIM, dtype=np.float32)
        self._active_model_index = 0
        # Internal RNG used when the caller provides none (e.g. SVG's
        # reset_torch). Prefer an explicit `seed`; the fallback derives from
        # the global numpy state (reproducible after set_global_seed(), but
        # dependent on object construction order).
        if seed is None:
            seed = int(np.random.randint(0, 2**31))
        self._rng = np.random.default_rng(seed)
        self.reset_context()

    @property
    def ensemble(self) -> List[ModularMLP]:
        """The underlying ModularMLP ensemble (read-only view)."""
        return self._ensemble

    def sample_model_index(self, rng=None) -> int:
        """Pick a random ensemble member index (epistemic-uncertainty sampling)."""
        if len(self._ensemble) <= 1:
            return 0
        if rng is None:
            rng = self._rng
        return int(rng.integers(0, len(self._ensemble)))

    def set_active_model(self, index: int) -> None:
        """Set the active ensemble member chosen by sample_model_index(), without touching beam0.

        Lets callers (e.g. MBPO) vary the model every step while keeping
        beam0 fixed for the duration of one rollout.
        """
        self.model = self._ensemble[index]
        self._active_model_index = index

    def evaluate_ensemble_disagreement(self, params: Dict[str, float]) -> dict:
        """Run every ensemble member on the same params/episode beam0 and
        report how much they disagree -- a simple epistemic-uncertainty
        diagnostic (variance across independently-trained models is a proxy
        for "is the surrogate confident here, or extrapolating").

        Purely diagnostic: does not touch the active model or any episode
        state, and is not used by `simulate()`/RL training. With a single
        model in the ensemble, every std is exactly 0.
        """
        beam0_t = torch.tensor(
            self._episode_beam0, dtype=torch.float32, device=self.device,
        ).unsqueeze(0)

        final_stage_predictions = []
        scores = []
        for member in self._ensemble:
            _, final_beam, score_val, _ = run_surrogate_forward(member, beam0_t, params, self.device)
            final_stage_predictions.append([final_beam[name] for name in BEAM_STATE_FEATURES])
            scores.append(score_val)

        final_stage_predictions = np.asarray(final_stage_predictions, dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64)

        return {
            "n_models": len(self._ensemble),
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
            "final_stage_std": {
                name: float(final_stage_predictions[:, i].std())
                for i, name in enumerate(BEAM_STATE_FEATURES)
            },
        }

    def sample_beam0(self, rng=None) -> np.ndarray:
        """Sample one real initial beam state from the dataset."""
        # Use the caller-provided RNG when available, so episodes can be reproducible.
        if rng is None:
            rng = self._rng
        n = self._initial_beam_states.shape[0]
        idx = int(rng.integers(0, n))
        return self._initial_beam_states[idx].numpy().astype(np.float32)

    def set_episode_beam0(self, beam0: np.ndarray) -> None:
        self._episode_beam0 = beam0

    def reset_context(self, rng=None) -> None:
        if rng is None:
            rng = self._rng
        # set the initial beam and the model to use from the ensemble
        self.set_active_model(self.sample_model_index(rng))
        self.set_episode_beam0(self.sample_beam0(rng))

    def forward_differentiable(self, model, beam0: torch.Tensor, stage_params_grad: list):
        """Gradient-preserving counterpart of `simulate()`. No `no_grad()`.

        Used by SVGAgent, which needs the forward pass to stay in the
        autograd graph so it can backprop through the surrogate.
        """
        return model(stage_params_grad, beam0)

    def simulate(self, params: Dict[str, float]) -> BeamSimulationResult:
        """Predict one beam trajectory with the active surrogate model.

        This is the standard BeamSimulator API used by BaseBeamEnv.step().
        It uses the episode beam0 and the currently active ensemble member,
        then wraps the predicted beam states in a BeamSimulationResult.
        """
        try:

            # Convert the episode initial beam state from numpy to a batched tensor expected by ModularMLP: (1, BEAM_STATE_DIM).
            beam0_t = torch.tensor( self._episode_beam0, dtype=torch.float32, device=self.device, ).unsqueeze(0)

            # Run the active surrogate model without gradients and compute the final beam dictionary plus the scalar score.
            beam_states, final_beam, score_val, classifier_flagged_failure = run_surrogate_forward(
                self.model, beam0_t, params, self.device,
                classifier=self.classifier, classifier_threshold=self.classifier_threshold,
            )

            metadata = {
                "beam0": self._episode_beam0.copy(),
                "model_index": self._active_model_index,
            }
            if self.classifier is not None:
                metadata["classifier_flagged_failure"] = classifier_flagged_failure

            # Return the BeamSimulationResult
            return BeamSimulationResult(
                params=params.copy(),
                beam_states=beam_states,
                score_val=score_val,
                success=True,
                source="surrogate",
                final_beam=final_beam,
                metadata=metadata,
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
                    "model_index": self._active_model_index,
                },
            )
