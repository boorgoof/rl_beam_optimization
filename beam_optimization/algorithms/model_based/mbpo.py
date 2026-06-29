"""
MBPO — Model-Based Policy Optimization con surrogate ensemble.

Reference:
    Janner M. et al., "When to Trust Your Model: Model-Based Policy
    Optimization", NeurIPS 2019. https://arxiv.org/abs/1906.08253
    Implementazione di riferimento: https://github.com/Xingyu-Lin/mbpo_pytorch

Struttura dell'algoritmo (uguale a mbpo_pytorch):
    ① Raccogli E transizioni reali → real_buffer
    ② Genera K rollout sintetici di lunghezza H con il surrogate ensemble
       → synth_buffer
    ③ Allena SAC su un mix: real_ratio * real + (1-real_ratio) * synth

Differenza rispetto a mbpo_pytorch:
    - Il loro modello è una rete (s,a)→(s',r) addestrata online sui dati reali.
    - Il nostro modello è il surrogato fisico pre-trainato su TraceWin:
      surrogate(beam0, params) → beam_states_all_stages.
      Non è un transition model ma un simulatore globale, quindi i rollout
      partono da (beam0, params) campionati fresh anziché da stati del buffer
      (le due distribuzioni sono equivalenti per il nostro problema).

Generazione rollout sintetici:
    Per ogni rollout:
      1. Campiona beam0 dal dataset
      2. Campiona params ~ N(default, sensitivity × sigma_factor)
      3. Surrogato scelto a caso dall'ensemble → obs iniziale s0
      4. Esegui H step con la policy SAC
    Un surrogato diverso può essere scelto ad ogni step (incertezza epistemica).
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from beam_optimization.algorithms.utils.replay_buffer import MixedReplayBuffer
from beam_optimization.env.surrogate_env.surrogate_simulator import SurrogateBeamSimulator, run_surrogate_forward
from beam_optimization.env.surrogate_env.surrogate.dataset import SurrogateTrainingDataset
from beam_optimization.env.surrogate_env.surrogate.modular_mlp import ModularMLP
from beam_optimization.config.adige import (
    BEAM_STATE_DIM, N_STAGES,
    PARAM_KEYS, N_PARAMS,
    default_params, sensitivity_vec,
    vec_to_params, params_to_vec,
)


# ── DynaMBPO ──────────────────────────────────────────────────────────────────

class MBPO:
    """MBPO: SAC allenato su transizioni reali + rollout sintetici del surrogate ensemble.

    Args:
        agent:                SAC or TD3 instance.
        surrogates:           Trained ModularMLP or list of ModularMLPs (ensemble).
                              When a list is given, each synthetic rollout picks a
                              surrogate at random, capturing epistemic uncertainty.
        dataset:              SurrogateTrainingDataset used to sample initial beam states.
        obs_dim:              Observation dimension.
        act_dim:              Action dimension.
        rollout_length:       Steps per synthetic rollout (1=Dyna, >1=MBPO).
        n_synthetic_per_step: Synthetic rollouts generated per real step.
        sigma_factor:         Gaussian noise scale (× sensitivity) for random params.
        real_ratio:           Real data fraction in each training batch.
        real_buffer_size:     Max real transitions.
        synth_buffer_size:    Max synthetic transitions.
        device:               Torch device.
    """

    def __init__(
        self,
        agent,
        surrogates: Union[ModularMLP, List[ModularMLP]],
        dataset: SurrogateTrainingDataset,
        obs_dim: int,
        act_dim: int,
        rollout_length: int = 1,
        n_synthetic_per_step: int = 400,
        sigma_factor: float = 0.5,
        real_ratio: float = 0.05,
        real_buffer_size: int = int(1e5),
        synth_buffer_size: int = int(1e6),
        obs_mode: str = "full",
        device: Optional[str] = None,
    ):
        self.agent            = agent
        self.rollout_length   = int(rollout_length)
        self.n_synthetic_per_step = int(n_synthetic_per_step)
        self.sigma_factor     = float(sigma_factor)
        self.obs_mode         = obs_mode
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Owns the ensemble + beam0 sampling, shared with SurrogateEnv's simulator
        self.simulator = SurrogateBeamSimulator(surrogates, dataset, device=self.device)

        # Precompute defaults and sensitivities for random param sampling
        self._defaults_vec = params_to_vec(default_params())       # (16,) float32
        self._sens         = sensitivity_vec().astype(np.float32)  # (16,) float32

        # Replace inner agent's replay buffer with mixed one
        self.mixed_buffer = MixedReplayBuffer(
            obs_dim, act_dim, real_buffer_size, synth_buffer_size, real_ratio
        )
        agent.replay = self.mixed_buffer

    @property
    def surrogates(self) -> List[ModularMLP]:
        """Backward-compat alias used by MBPOWithModelUpdate: the underlying
        ModularMLP ensemble, now owned by self.simulator."""
        return self.simulator.ensemble

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
            obs:      Current observation (108-dim).
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
        """Generate fresh synthetic rollouts following the episode design:
           sample beam0 → sample random params → surrogate pre-run → RL rollout.
        """
        rng = np.random.default_rng()

        for _ in range(self.n_synthetic_per_step):
            # 1. Sample beam0 from dataset, fixed for the duration of this rollout
            beam0_np = self.simulator.sample_beam0(rng)
            beam0    = torch.tensor(beam0_np, dtype=torch.float32,
                                    device=self.device).unsqueeze(0)

            # 2. Sample random initial params
            noise  = rng.standard_normal(N_PARAMS).astype(np.float32) * self._sens * self.sigma_factor
            params = vec_to_params(self._defaults_vec + noise)

            # 3. Pick a surrogate from ensemble, get starting obs
            self.simulator.set_active_model(self.simulator.sample_model_index(rng))
            obs_i, score_i = self._surrogate_forward(beam0, params)

            # 4. Roll out for rollout_length steps
            for _ in range(self.rollout_length):
                action_i  = self.agent.select_action(obs_i, training=True)
                new_params = params.copy()
                for key, delta in zip(PARAM_KEYS, action_i):
                    new_params[key] = float(new_params[key]) + float(delta)

                # Optionally pick a different surrogate for the next step
                self.simulator.set_active_model(self.simulator.sample_model_index(rng))
                next_obs_i, next_score_i = self._surrogate_forward(beam0, new_params)

                reward_i = next_score_i - score_i
                self.mixed_buffer.store_synth(obs_i, action_i, reward_i, next_obs_i, 0.0)

                obs_i   = next_obs_i
                score_i = next_score_i
                params  = new_params

    def _surrogate_forward(
        self,
        beam0: torch.Tensor,
        params: dict,
    ) -> Tuple[np.ndarray, float]:
        """One surrogate forward pass (using self.simulator's active model) →
        (obs 108-dim, score)."""
        try:
            beam_states, _, sc = run_surrogate_forward(
                self.simulator.model, beam0, params, self.device
            )

            if self.obs_mode == "full":
                obs = beam_states.reshape(-1).astype(np.float32)
            elif self.obs_mode == "final":
                obs = beam_states[-1].astype(np.float32)
            else:  # "final_with_beam0"
                obs = np.concatenate([beam_states[0], beam_states[-1]]).astype(np.float32)

            return obs, sc
        except Exception:
            if self.obs_mode == "full":
                dim = N_STAGES * BEAM_STATE_DIM
            elif self.obs_mode == "final":
                dim = BEAM_STATE_DIM
            else:
                dim = 2 * BEAM_STATE_DIM
            return np.zeros(dim, dtype=np.float32), -1.0
