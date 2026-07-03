"""
SVGAgent — Stochastic Value Gradients (Heess et al., NeurIPS 2015).
Policy optimization through the differentiable world model (surrogate).

Reference:
    Heess N. et al., "Learning Continuous Control Policies by Stochastic Value
    Gradients", NeurIPS 2015. https://arxiv.org/abs/1510.09142

Key idea (SVG-H, full-episode backprop):
    Instead of learning a separate Q-function (as in SAC/TD3), we exploit the
    fact that the surrogate is differentiable and directly differentiate the
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
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.optim as optim

from beam_optimization.algorithms.networks.policy_nets import GaussianPolicyNetwork
from beam_optimization.env.surrogate_env import DifferentiableSurrogateEnv
from beam_optimization.env.surrogate_env.surrogate.model.modular_mlp import ModularMLP
from beam_optimization.env.dataset import BeamDataset


@dataclass
class SVGResult:
    """Compact training output returned by optimize_episode()."""

    # Scalar loss used for the last policy update.
    episode_loss: float = 0.0

    # Final beam-quality score reached at the end of the rollout.
    final_score: float = 0.0

    # Per-step score trace, useful for logging and debugging convergence.
    score_history: List[float] = field(default_factory=list)

    # Policy gradient norm after clipping.
    grad_norm: float = 0.0


class SVGAgent:
    """SVG-H variant.

    The policy π_θ(obs) → Δparams is trained end-to-end by backpropagating
    the cumulative beam-quality reward through the differentiable surrogate.
    No critic network is needed.

    Args:
        surrogate:     Trained ModularMLP (weights frozen during policy training).
        dataset:       BeamDataset providing initial beam states for resets.
        obs_dim:       Observation dimension.
        act_dim:       Action dimension.
        action_bounds: Physical action bounds as (low, high).
        param_keys:    Ordered parameter keys matching the policy action vector.
        default_params: Initial best-parameter dictionary.
        hidden_dims:   Policy network hidden layer sizes.
        lr:            Policy Adam learning rate.
        alpha:         Entropy coefficient (0.0 = no entropy regularization).
        n_step:        Episode horizon (number of steps to unroll).
        max_grad_norm: Gradient clipping norm.
        device:        Torch device.
    """

    def __init__(
        self,
        surrogate: Union[ModularMLP, List[ModularMLP]],
        dataset: BeamDataset,
        obs_dim: int,
        act_dim: int,
        action_bounds,
        param_keys: Sequence[str],
        default_params: Mapping[str, float],
        hidden_dims: Tuple[int, ...] = (256, 256),
        lr: float = 3e-4,
        alpha: float = 0.01,
        n_step: int = 20,
        max_grad_norm: float = 1.0,
        stage_weights: Optional[List[float]] = None,
        device: Optional[str] = None,
    ):
        self.dataset = dataset
        self.n_step = n_step
        self.alpha = alpha
        self.max_grad_norm = max_grad_norm

        # Use CUDA automatically when available, unless the caller forces a device.
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # The differentiable environment owns the surrogate simulator and keeps
        # the torch graph alive from reward -> surrogate -> action -> policy.
        # Its inherited reset()/step() stay Gym-compatible for normal evaluation.
        self.env = DifferentiableSurrogateEnv(
            model=surrogate,
            dataset=dataset,
            max_steps=n_step,
            device=str(self.device),
            stage_weights=stage_weights,
        )

        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.param_keys = tuple(param_keys)
        if len(self.param_keys) != self.act_dim:
            raise ValueError(
                f"param_keys length ({len(self.param_keys)}) must match act_dim ({self.act_dim})"
            )

        # action bounds
        action_min, action_max = action_bounds
        bounds = (
            np.asarray(action_min, dtype=np.float32).tolist(),
            np.asarray(action_max, dtype=np.float32).tolist(),
        )
        if len(bounds[0]) != self.act_dim or len(bounds[1]) != self.act_dim:
            raise ValueError("action_bounds must contain low/high vectors with length act_dim")

        # The policy network outputs a Gaussian distribution over actions, which is reparameterized for low-variance gradients. 
        self.policy = GaussianPolicyNetwork(self.obs_dim, bounds, hidden_dims).to(self.device)
        self.policy.action_min = self.policy.action_min.to(self.device)
        self.policy.action_max = self.policy.action_max.to(self.device)

        # optimizer for the policy network parameters
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Track the best score and corresponding parameters for logging and checkpointing.
        self.best_score = -float("inf")
        self.best_params = {str(k): float(v) for k, v in default_params.items()}
        self.train_steps = 0

    def optimize_episode(self, beam0: Optional[torch.Tensor] = None) -> SVGResult:
        """Train policy for one episode via SVG-H.

        Args:
            beam0: (1, 9) initial beam state. If None, samples randomly from dataset.

        Returns:
            SVGResult with loss, final score, per-step history.
        """

        # Reset the differentiable environment to the initial beam state. If beam0 is None, a random initial state is sampled from the dataset.
        # The returned state contains the initial observation and score.
        state = self.env.reset_torch(beam0=beam0)

        total_loss = torch.zeros(1, device=self.device)
        score_history: List[float] = []

        self.policy.train()
        self.optimizer.zero_grad() # clear previous gradients before starting the episode

        # Freeze only the active surrogate weights. The differentiable env still
        # lets gradients flow through the surrogate outputs to action -> policy.
        with self.env.frozen_surrogate_weights():
            for _ in range(self.n_step):

                # policy chooses an action
                action, logpa, _, _, _ = self.policy.full_pass(state.obs)
                # action: (1, act_dim), physical delta applied to machine params.

                # Differentiable env step. The returned reward keeps the path:
                # reward -> score -> surrogate -> params -> action -> policy.
                next_state, reward = self.env.step_torch(state, action)

                # Maximize reward with an entropy bonus:
                #   objective = reward - alpha * log_prob
                # so the minimized loss is the negative objective.
                total_loss = total_loss - (reward - self.alpha * logpa)
                score_history.append(float(next_state.score.detach()))

                # detach the next state to avoid accumulating gradients through time.
                # we only want to backpropagate through the current step's computation graph.
                state = next_state.detach_for_next_step()

            # Backpropagate through all local surrogate transitions accumulated above.
            total_loss.backward()
            # Clip gradients to avoid exploding gradients due to long unrolls through the surrogate.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_grad_norm
            ).item()
            # Update the policy parameters using the optimizer.
            self.optimizer.step()

        # Update best score and parameters if this episode achieved a new high score.
        final_score = score_history[-1] if score_history else 0.0
        if final_score > self.best_score:
            self.best_score  = final_score
            self.best_params = self._params_tensor_to_dict(state.params)

        self.train_steps += 1

        return SVGResult(
            episode_loss=float(total_loss.detach()),
            final_score=final_score,
            score_history=score_history,
            grad_norm=grad_norm,
        )


    def evaluate(self, n_episodes: int = 20) -> float:
        """Average final score over n_episodes with greedy policy."""
        self.policy.eval()
        scores = []

        # Evaluation uses the same differentiable environment API, but disables
        # autograd because no policy update is performed here.
        with torch.no_grad():
            for _ in range(n_episodes):
                state = self.env.reset_torch()
                for _ in range(self.n_step):
                    action  = self.policy.select_greedy_action(state.obs.cpu().numpy())
                    action_t = torch.tensor(action, dtype=torch.float32, device=self.device)
                    state, _ = self.env.step_torch(state, action_t)
                    state = state.detach_for_next_step()

                scores.append(float(state.score))
        self.policy.train()
        return float(np.mean(scores))

    def save(self, path: str):
        """Save only the policy-side state; surrogate weights are external."""
        torch.save({
            "policy":    self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_score": self.best_score,
            "train_steps": self.train_steps,
        }, path)

    def load(self, path: str):
        """Restore a policy checkpoint created by save()."""
        ck = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ck["policy"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.best_score  = ck.get("best_score", -float("inf"))
        self.train_steps = ck.get("train_steps", 0)

    #Internal helper 
    def _params_tensor_to_dict(self, params: torch.Tensor) -> Dict[str, float]:
        """Convert a parameter tensor back to the configured parameter dictionary."""
        vec = params.detach().cpu().numpy()
        return {k: float(v) for k, v in zip(self.param_keys, vec)}
