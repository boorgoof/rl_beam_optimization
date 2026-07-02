"""
MBPO — Model-Based Policy Optimization with a surrogate ensemble.

Reference:
    Janner M. et al., "When to Trust Your Model: Model-Based Policy
    Optimization", NeurIPS 2019. https://arxiv.org/abs/1906.08253
    Reference implementation: https://github.com/Xingyu-Lin/mbpo_pytorch

Algorithm structure:
    1. Store real transitions in the real buffer.
    2. Generate short synthetic rollouts with the surrogate ensemble and store
       them in the synthetic buffer.
    3. Train the inner policy on mixed batches:
       real_ratio * real + (1 - real_ratio) * synthetic.

Project-specific difference from standard MBPO:
    The model is not a learned transition network f(s, a) -> (s', r). It is a
    TraceWin-trained beam surrogate:

        surrogate(beam0, params) -> beam_states_all_stages

    Synthetic rollouts therefore start from freshly sampled beam/parameter
    states and then use SurrogateEnv for normal environment steps.
"""
from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
from beam_optimization.algorithms.utils.replay_buffer import MixedReplayBuffer
from beam_optimization.env.surrogate_env import SurrogateEnv
from beam_optimization.env.dataset import BeamDataset
from beam_optimization.env.surrogate_env.surrogate.model.modular_mlp import ModularMLP


# ── DynaMBPO ──────────────────────────────────────────────────────────────────

class MBPO:
    """MBPO: train a policy on real transitions plus surrogate rollouts.

    Args:
        agent:                SAC or TD3 instance.
        surrogates:           Trained ModularMLP or list of ModularMLPs (ensemble).
                              When a list is given, each synthetic rollout picks a
                              surrogate at random, capturing epistemic uncertainty.
        dataset:              BeamDataset used to sample initial beam states.
        obs_dim:              Observation dimension.
        act_dim:              Action dimension.
        rollout_length:       Steps per synthetic rollout (1=Dyna, >1=MBPO).
        n_synthetic_per_step: Synthetic rollouts generated per real step.
        real_ratio:           Real data fraction in each training batch.
        real_buffer_size:     Max real transitions.
        synth_buffer_size:    Max synthetic transitions.
        device:               Torch device.
    """

    def __init__(
        self,
        agent,
        surrogates: Union[ModularMLP, List[ModularMLP]],
        dataset: BeamDataset,
        obs_dim: int,
        act_dim: int,
        rollout_length: int = 1,
        n_synthetic_per_step: int = 400,
        real_ratio: float = 0.05,
        real_buffer_size: int = int(1e5),
        synth_buffer_size: int = int(1e6),
        device: Optional[str] = None,
    ):
        self.agent            = agent
        self.rollout_length   = int(rollout_length)
        self.n_synthetic_per_step = int(n_synthetic_per_step)

        # MBPO owns a surrogate environment for synthetic rollouts.
        # SurrogateEnv owns the simulator and surrogate ensemble.
        self.synthetic_env = SurrogateEnv(
            model=surrogates,
            dataset=dataset,
            max_steps=max(1, self.rollout_length),
            device=device,
        )

        # Replace inner agent's replay buffer with mixed one
        self.mixed_buffer = MixedReplayBuffer(
            obs_dim, act_dim, real_buffer_size, synth_buffer_size, real_ratio
        )
        agent.replay = self.mixed_buffer

    # ── Public API ─────────────────────────────────────────────────────────────

    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """Process one real transition and trigger synthetic rollout generation.

        Args:
            obs:      Current observation from OBSERVATION_STAGE_MASK.
            action:   Action taken (16-dim delta).
            reward:   Received reward.
            next_obs: Next observation.
            done:     Episode done flag.

        Returns:
            Loss tuple from agent.optimize() or None if buffer not ready.
        """
        self.mixed_buffer.store_real(obs, action, reward, next_obs, float(done))

        if self.mixed_buffer.size >= 256:
            self._generate_synthetic()

        return self.agent.optimize()

    def select_action(self, state, training: bool = True):
        return self.agent.select_action(state, training=training)

    # ── Synthetic rollout generation ───────────────────────────────────────────

    def _generate_synthetic(self):
        """Generate fresh synthetic rollouts using SurrogateEnv."""
        for _ in range(self.n_synthetic_per_step):
            obs_i, _ = self.synthetic_env.reset()
            for _ in range(self.rollout_length):
                action_i  = self.agent.select_action(obs_i, training=True)
                self.synthetic_env.sample_active_model()
                next_obs_i, reward_i, terminated, truncated, _ = self.synthetic_env.step(action_i)
                done = bool(terminated or truncated)
                self.mixed_buffer.store_synth(obs_i, action_i, reward_i, next_obs_i, float(done))
                obs_i = next_obs_i
                if done:
                    break
