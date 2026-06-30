"""
SVGAgent — Stochastic Value Gradients (Heess et al., NeurIPS 2015).
Policy optimization through the differentiable world model (surrogate).

Reference:
    Heess N. et al., "Learning Continuous Control Policies by Stochastic Value
    Gradients", NeurIPS 2015. https://arxiv.org/abs/1510.09142

Key idea (SVG-H, full-episode backprop):
    Instead of learning a separate Q-function (as in SAC/TD3), we exploit the
    fact that the surrogate is differentiable and directly differentiate thema sei sicuro che n
    cumulative reward w.r.t. the policy parameters:

        ∂J/∂θ = ∂/∂θ  Σ_t  score(surrogate(beam0, params_t))

    The computational graph is:
        θ_policy → action_t (rsample) → params_t → surrogate → beam_states_t → score_t

    This collapses the need for a critic: the surrogate *is* the critic.

Difference from GradientOptimizer (also in this repo):
    GradientOptimizer optimizes the parameter vector directly for a fixed beam0.
    SVGAgent optimizes a POLICY π(obs)→Δparams that generalizes across all
    initial beam states sampled from the dataset — it learns *how to tune*,
    not just *what the optimal tune is*.

Practical notes:
    - Uses reparameterization trick (rsample) for low-variance gradients.
    - Entropy bonus (like SAC) encourages exploration; α=0.0 disables it.
    - Gradient clipping is essential: unrolling through many surrogate steps
      can produce large gradients.
    - For stability, score at previous step is detached from the graph
      (only the delta matters for the policy, and we avoid double-counting).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from beam_optimization.algorithms.networks.policy_nets import GaussianPolicyNetwork
from beam_optimization.config.adige import (
    PARAM_KEYS, N_PARAMS, BEAM_STATE_DIM, N_STAGES,
    STAGE_PARAM_SIZES,
    default_params, sensitivity_vec, action_bounds,
    params_to_vec, vec_to_params, score_tensor,
)
from beam_optimization.env.surrogate_env.surrogate_simulator import SurrogateBeamSimulator
from beam_optimization.env.surrogate_env.surrogate.modular_mlp import ModularMLP
from beam_optimization.env.dataset import BeamDataset

OBS_DIM = N_STAGES * BEAM_STATE_DIM  # 108


@dataclass
class SVGResult:
    """Returned by select_action (greedy eval) and optimize_episode (training)."""
    episode_loss: float = 0.0
    final_score: float = 0.0
    score_history: List[float] = field(default_factory=list)
    grad_norm: float = 0.0


class SVGAgent:
    """Policy optimization through world model — SVG-H variant.

    The policy π_θ(obs) → Δparams is trained end-to-end by backpropagating
    the cumulative beam-quality reward through the differentiable surrogate.
    No critic network is needed.

    Args:
        surrogate:     Trained ModularMLP (weights frozen during policy training).
        dataset:       BeamDataset providing initial beam states for resets.
        obs_dim:       Observation dimension (default: 108 = 12 stages × 9 vars).
        act_dim:       Action dimension (default: 16 parameters).
        action_scale:  Multiplier on sensitivity for action bounds.
        hidden_dims:   Policy network hidden layer sizes.
        lr:            Policy Adam learning rate.
        alpha:         Entropy coefficient (0.0 = no entropy regularization).
        H:             Episode horizon (number of steps to unroll).
        sigma_factor:  Gaussian noise scale for initial parameter sampling.
        max_grad_norm: Gradient clipping norm.
        device:        Torch device.
    """

    def __init__(
        self,
        surrogate: Union[ModularMLP, List[ModularMLP]],
        dataset: BeamDataset,
        obs_dim: int = OBS_DIM,
        act_dim: int = N_PARAMS,
        action_scale: float = 1.0,
        hidden_dims: Tuple[int, ...] = (256, 256),
        lr: float = 3e-4,
        alpha: float = 0.01,
        H: int = 20,
        sigma_factor: float = 0.5,
        max_grad_norm: float = 1.0,
        stage_weights: Optional[List[float]] = None,
        obs_mode: str = "full",
        device: Optional[str] = None,
    ):
        self.dataset      = dataset
        self.H            = H
        self.alpha        = alpha
        self.sigma_factor = sigma_factor
        self.max_grad_norm = max_grad_norm
        self.obs_mode     = obs_mode
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Owns the ensemble + beam0 sampling, shared with SurrogateEnv's simulator.
        # Surrogate weights are frozen during SVG — only policy is optimized.
        # We use torch.no_grad() during forward passes instead of permanently
        # disabling requires_grad, so models can still be fine-tuned externally.
        self.simulator = SurrogateBeamSimulator(surrogate, dataset, device=self.device)

        act_low, act_high = action_bounds(action_scale)
        bounds = (act_low.tolist(), act_high.tolist())
        self.policy    = GaussianPolicyNetwork(obs_dim, bounds, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self._sens     = torch.tensor(sensitivity_vec(), dtype=torch.float32, device=self.device)
        self._defaults = torch.tensor(
            params_to_vec(default_params()), dtype=torch.float32, device=self.device
        )

        if stage_weights is not None:
            w = torch.tensor(stage_weights, dtype=torch.float32, device=self.device)
            self._stage_weights = w / w.sum()
        else:
            self._stage_weights = None

        self.best_score  = -float("inf")
        self.best_params = default_params()
        self.train_steps = 0

    @property
    def surrogate(self):
        """Active ensemble member (swapped per episode via self.simulator)."""
        return self.simulator.model

    # ── Public API ─────────────────────────────────────────────────────────────

    def optimize_episode(self, beam0: Optional[torch.Tensor] = None) -> SVGResult:
        """Train policy for one episode via SVG-H.

        Args:
            beam0: (1, 9) initial beam state. If None, samples randomly from dataset.

        Returns:
            SVGResult with loss, final score, per-step history.
        """
        # Thompson sampling: pick one surrogate from ensemble for this episode
        self.simulator.set_active_model(self.simulator.sample_model_index())

        # Freeze surrogate params for this episode (gradient flows through, but
        # params don't accumulate grad). Restored after backward.
        for p in self.surrogate.parameters():
            p.requires_grad_(False)

        if beam0 is None:
            beam0_np = self.simulator.sample_beam0()
            beam0 = torch.tensor(beam0_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Sample initial params with Gaussian noise
        noise  = torch.randn(N_PARAMS, device=self.device) * self._sens * self.sigma_factor
        params = (self._defaults + noise).detach()  # no grad on init params

        # First obs: run surrogate with initial params (no grad needed for obs construction)
        with torch.no_grad():
            outputs_init = self.simulator.forward_differentiable(
                self.surrogate, beam0, self._split_params(params)
            )
        obs = self._build_obs(beam0, outputs_init)  # (1, 108)

        # Score at t=0 (detached — used only as delta baseline)
        score_prev = score_tensor(outputs_init[-1]).detach()  # (1,)

        total_loss   = torch.zeros(1, device=self.device)
        score_history: List[float] = []

        self.policy.train()
        self.optimizer.zero_grad()

        for step in range(self.H):
            # ── Policy forward (reparameterized sample) ──────────────────────
            action, logpa, _, _, _ = self.policy.full_pass(obs)
            # action: (1, 16) — in-graph, gradient flows back to θ_policy

            # ── Parameter update ──────────────────────────────────────────────
            params = params + action.squeeze(0)   # (16,) — stays in graph

            # ── Surrogate rollout ─────────────────────────────────────────────
            # Surrogate weights are frozen, but gradient flows through params → action → θ
            beam_states = self.simulator.forward_differentiable(
                self.surrogate, beam0, self._split_params_grad(params)
            )
            # beam_states: list of 11 tensors (1, 9), all in graph

            # ── Reward ────────────────────────────────────────────────────────
            if self._stage_weights is None:
                score_now = score_tensor(beam_states[-1])  # (1,) — in graph
            else:
                scores = torch.stack([score_tensor(s) for s in beam_states])  # (11, 1)
                score_now = (scores * self._stage_weights.view(-1, 1)).sum(dim=0)
            reward    = score_now - score_prev          # delta score

            # ── Loss accumulation ─────────────────────────────────────────────
            total_loss = total_loss - (reward - self.alpha * logpa)
            score_history.append(float(score_now.detach()))

            # ── Update obs for next step (detach to avoid TBPTT through time) ─
            # We use SVG-1 style: backprop only through current step's params→score
            obs        = self._build_obs_detach(beam0, beam_states)
            score_prev = score_now.detach()
            params     = params.detach()   # prevents gradient explosion across steps

        # ── Policy gradient step ──────────────────────────────────────────────
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.max_grad_norm
        ).item()
        self.optimizer.step()

        for p in self.surrogate.parameters():
            p.requires_grad_(True)

        final_score = score_history[-1] if score_history else 0.0
        if final_score > self.best_score:
            self.best_score  = final_score
            self.best_params = self._params_tensor_to_dict(params)

        self.train_steps += 1
        return SVGResult(
            episode_loss=float(total_loss.detach()),
            final_score=final_score,
            score_history=score_history,
            grad_norm=grad_norm,
        )

    def select_action(self, obs: np.ndarray, training: bool = False) -> np.ndarray:
        """Deterministic (greedy) action for evaluation."""
        return self.policy.select_greedy_action(obs)

    def evaluate(self, n_episodes: int = 20) -> float:
        """Average final score over n_episodes with greedy policy."""
        self.policy.eval()
        scores = []
        with torch.no_grad():
            for _ in range(n_episodes):
                self.simulator.set_active_model(self.simulator.sample_model_index())
                beam0_np = self.simulator.sample_beam0()
                beam0 = torch.tensor(beam0_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                noise = torch.randn(N_PARAMS, device=self.device) * self._sens * self.sigma_factor
                params = self._defaults + noise

                for _ in range(self.H):
                    outputs = self.simulator.forward_differentiable(
                        self.surrogate, beam0, self._split_params(params)
                    )
                    obs     = self._build_obs_detach(beam0, outputs)
                    action  = self.policy.select_greedy_action(obs.cpu().numpy())
                    action_t = torch.tensor(action, dtype=torch.float32, device=self.device)
                    params  = params + action_t

                outputs = self.simulator.forward_differentiable(
                    self.surrogate, beam0, self._split_params(params)
                )
                scores.append(float(score_tensor(outputs[-1])))
        self.policy.train()
        return float(np.mean(scores))

    def save(self, path: str):
        torch.save({
            "policy":    self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_score": self.best_score,
            "train_steps": self.train_steps,
        }, path)

    def load(self, path: str):
        ck = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ck["policy"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.best_score  = ck.get("best_score", -float("inf"))
        self.train_steps = ck.get("train_steps", 0)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _split_params(self, params: torch.Tensor) -> List[torch.Tensor]:
        """Split detached (16,) param vector into stage tensors for surrogate."""
        with torch.no_grad():
            return self._split_params_grad(params.detach())

    def _split_params_grad(self, params: torch.Tensor) -> List[torch.Tensor]:
        """Split in-graph (16,) param vector into stage tensors for surrogate.
        Keeps gradient connection: necessary for backprop to flow.
        """
        tensors = []
        offset  = 0
        for size in STAGE_PARAM_SIZES:
            tensors.append(params[offset:offset + size].unsqueeze(0))  # (1, size)
            offset += size
        return tensors

    def _build_obs(self, beam0: torch.Tensor, outputs: List[torch.Tensor]) -> torch.Tensor:
        """Build observation tensor according to obs_mode."""
        if self.obs_mode == "full":
            stages = [beam0] + outputs
            return torch.cat([s.squeeze(0) for s in stages]).unsqueeze(0)   # (1, 108)
        elif self.obs_mode == "final":
            return outputs[-1]                                                # (1, 9)
        else:  # "final_with_beam0"
            return torch.cat([beam0.squeeze(0), outputs[-1].squeeze(0)]).unsqueeze(0)  # (1, 18)

    def _build_obs_detach(self, beam0: torch.Tensor, outputs: List[torch.Tensor]) -> torch.Tensor:
        """Build observation with detached tensors (used for next-step obs)."""
        if self.obs_mode == "full":
            stages = [beam0.detach()] + [o.detach() for o in outputs]
            return torch.cat([s.squeeze(0) for s in stages]).unsqueeze(0)
        elif self.obs_mode == "final":
            return outputs[-1].detach()
        else:  # "final_with_beam0"
            return torch.cat([beam0.squeeze(0).detach(),
                               outputs[-1].squeeze(0).detach()]).unsqueeze(0)

    def _params_tensor_to_dict(self, params: torch.Tensor) -> Dict[str, float]:
        vec = params.detach().cpu().numpy()
        return {k: float(v) for k, v in zip(PARAM_KEYS, vec)}
