"""
SB3SAC — wrapper intorno a stable_baselines3.SAC.

Stable Baselines 3 gestisce internamente buffer e ottimizzazione, quindi
l'API è diversa dagli agenti custom (niente store/optimize step-by-step).
Questo wrapper:
  - espone train(env, n_steps) per la fase di training
  - espone select_action(obs) per l'inference (deterministic)
  - espone save(path) / load(path, env) per i checkpoint

Uso tipico:
    from beam_optimization.algorithms.model_free.sb3_sac import SB3SAC
    env   = SurrogateEnv(model=surrogate, dataset=ds)
    agent = SB3SAC(env, hidden_dims=(256, 256), lr=3e-4)
    best  = agent.train(env, n_steps=200_000)
    agent.save("runs/sb3_sac/model.zip")
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from beam_optimization.algorithms.utils.logger import Logger

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False


def _check_sb3():
    if not _SB3_AVAILABLE:
        raise ImportError(
            "stable-baselines3 non trovato. Installa con:\n"
            "  pip install stable-baselines3"
        )


class _BestScoreCallback(BaseCallback):
    """Traccia il best score dell'env durante il training SB3."""

    def __init__(self, logger: Optional[Logger] = None, log_every: int = 10_000):
        super().__init__()
        self.best_score = -float("inf")
        self.metrics_logger = logger
        self.log_every = max(1, int(log_every))
        self._episode_reward = 0.0
        self._episode_count = 0

    def _on_step(self) -> bool:
        # SB3 espone env info tramite self.locals
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        if len(rewards) > 0:
            self._episode_reward += float(np.asarray(rewards).reshape(-1)[0])
        for info in infos:
            sc = info.get("score")
            if sc is not None and sc > self.best_score:
                self.best_score = sc
            if self.metrics_logger is not None and self.num_timesteps % self.log_every == 0:
                metrics = {
                    "score": float(sc) if sc is not None else 0.0,
                    "best_score": self.best_score,
                    "episode_reward": self._episode_reward,
                    "episode": float(self._episode_count),
                }
                self.metrics_logger.log(metrics, step=self.num_timesteps)
        if len(dones) > 0 and bool(np.asarray(dones).reshape(-1)[0]):
            self._episode_count += 1
            self._episode_reward = 0.0
        return True


class SB3SAC:
    """Wrapper di stable_baselines3.SAC per SurrogateEnv.

    Args:
        env:         Ambiente Gymnasium (SurrogateEnv o TraceWinEnv).
        hidden_dims: Dimensioni hidden layer della policy/value net.
        lr:          Learning rate Adam.
        buffer_size: Dimensione del replay buffer interno SB3.
        batch_size:  Batch size per ogni aggiornamento SAC.
        tau:         Soft update coefficient (target networks).
        gamma:       Discount factor.
        device:      'cpu', 'cuda', o 'auto'.
    """

    def __init__(
        self,
        env,
        hidden_dims: Tuple[int, ...] = (256, 256),
        lr: float = 3e-4,
        buffer_size: int = int(1e6),
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        device: str = "auto",
        tensorboard_log: Optional[str] = None,
    ):
        _check_sb3()
        policy_kwargs = {"net_arch": list(hidden_dims)}
        self._model = SAC(
            "MlpPolicy",
            env,
            learning_rate=lr,
            buffer_size=buffer_size,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            policy_kwargs=policy_kwargs,
            device=device,
            tensorboard_log=tensorboard_log,
            verbose=0,
        )
        self._env = env

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, env=None, n_steps: int = 200_000,
              log_every: int = 10_000,
              logger: Optional[Logger] = None) -> float:
        """Esegue il training SB3 per n_steps step reali.

        Args:
            env:       Se fornito, rimpiazza l'env usato nel costruttore.
            n_steps:   Numero totale di step di interazione con l'env.
            log_every: Ogni quanti step stampare un riepilogo.

        Returns:
            Best score raggiunto durante il training.
        """
        _check_sb3()
        if env is not None and env is not self._env:
            self._model.set_env(env)

        cb = _BestScoreCallback(logger=logger, log_every=log_every)
        self._model.learn(
            total_timesteps=n_steps,
            callback=cb,
            log_interval=max(1, log_every // 1000),
            reset_num_timesteps=True,
        )
        return cb.best_score

    # ── Inference ─────────────────────────────────────────────────────────────

    def select_action(self, obs: np.ndarray,
                      deterministic: bool = True) -> np.ndarray:
        """Seleziona un'azione (deterministica per default, stochastic=False)."""
        _check_sb3()
        action, _ = self._model.predict(obs, deterministic=deterministic)
        return action

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def save(self, path: str):
        """Salva il modello SB3. SB3 aggiunge automaticamente l'estensione .zip."""
        _check_sb3()
        self._model.save(path)

    @classmethod
    def load(cls, path: str, env) -> "SB3SAC":
        """Carica un modello SB3 salvato."""
        _check_sb3()
        instance = cls.__new__(cls)
        instance._model = SAC.load(path, env=env)
        instance._env   = env
        return instance
