"""
MBPOWithModelUpdate — MBPO variant where the surrogate ensemble is also
fine-tuned on real TraceWin transitions as they are collected during RL
training, mirroring the original MBPO paper (Janner 2019) where the dynamics
model is trained on real data.

Differences from MBPO (mbpo.py):
  - Surrogate models are fine-tuned every `model_train_freq` real steps
    by a SurrogateDatasetUpdater.
  - `step()` accepts an optional `sim_result` (from TraceWinEnv info dict)
    so the real TraceWin output can be added to the model-update dataset.
  - One Adam optimizer per surrogate is maintained by the updater.
  - Ogni batch di fine-tuning viene costruito dal SurrogateDatasetUpdater, che
    mescola dati offline (il `dataset` seed) e dati online raccolti in questa
    run con quota target `online_mix_ratio`.
  - Se `dataset_save_path` è fornito, l'unione offline+online viene salvata
    periodicamente in formato flat. Se coincide col dataset base caricato, quel
    file viene aggiornato con i nuovi dati TraceWin.

Intended use:
    env = TraceWinEnv(project_file=...)
    inner = SAC(...)
    agent = MBPOWithModelUpdate(
        agent=inner,
        surrogates=[s1, s2, s3],
        dataset=seed_dataset,
        obs_dim=108, act_dim=16,
        model_train_freq=50,
        model_train_epochs=20,
        dataset_save_path="env/dataset/base/dataset_train.pt",
        surrogate_save_dir="surrogate/models/updated",
    )
    obs, info = env.reset()
    for step in range(N):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.step(obs, action, reward, next_obs, done,
                   sim_result=info.get("sim_result"))
        obs = next_obs
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from beam_optimization.algorithms.model_based.mbpo import MBPO
from beam_optimization.env.surrogate_env.surrogate.modular_mlp import ModularMLP
from beam_optimization.env.simulation import BeamSimulationResult
from beam_optimization.env.dataset import BeamDataset
from beam_optimization.env.surrogate_env.surrogate.updater import SurrogateDatasetUpdater


class MBPOWithModelUpdate(MBPO):
    """MBPO with surrogate model update step on real TraceWin data.

    Args:
        agent:                SAC or TD3 instance.
        surrogates:           Trained ModularMLP or list (ensemble).
        dataset:              Seed BeamDataset for beam0 sampling in synthetic rollouts.
        obs_dim:              Observation dimension (108).
        act_dim:              Action dimension (16).
        rollout_length:       Steps per synthetic rollout (1=Dyna, >1=MBPO).
        n_synthetic_per_step: Synthetic rollouts generated per real step.
        sigma_factor:         Gaussian noise scale (× sensitivity) for random params.
        real_ratio:           Real data fraction in each SAC training batch.
        real_buffer_size:     Max real transitions.
        synth_buffer_size:    Max synthetic transitions.
        model_train_freq:     Fine-tune surrogates every N real env steps.
        model_train_epochs:   Gradient update steps per fine-tuning call per surrogate.
        model_lr:             Adam learning rate for surrogate fine-tuning.
        online_batch_size:    Bootstrap sample size per fine-tuning call.
        min_samples_to_train: Minimum accumulated real samples before first fine-tune.
        online_mix_ratio:     Quota target (0-1) di ogni batch di fine-tuning presa
                              dai dati online dell'updater; il resto viene dal
                              dataset seed offline.
        dataset_save_path:    Se fornito, salva l'unione offline+online (dataset
                              "maestro" aggiornato) dopo ogni fine-tuning del
                              surrogate. Se coincide col path da cui è stato caricato
                              il dataset originale, quel file cresce run dopo run.
        surrogate_save_dir:   Se fornito, salva i pesi fine-tunati dell'ensemble come
                              surrogate_0.pt..surrogate_N.pt in questa cartella dopo
                              ogni fine-tuning del surrogate (e a fine training). Non
                              sovrascrive mai i surrogate_*.pt originali caricati a
                              inizio run: va passata una cartella diversa.
        device:               Torch device string or None for auto-detect.
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
        sigma_factor: float = 0.5,
        real_ratio: float = 0.05,
        real_buffer_size: int = int(1e5),
        synth_buffer_size: int = int(1e6),
        model_train_freq: int = 50,
        model_train_epochs: int = 20,
        model_lr: float = 3e-5,
        online_batch_size: int = 128,
        min_samples_to_train: int = 20,
        online_mix_ratio: float = 0.5,
        dataset_save_path: Optional[str | Path] = None,
        surrogate_save_dir: Optional[str | Path] = None,
        device: Optional[str] = None,
    ):
        super().__init__(
            agent=agent,
            surrogates=surrogates,
            dataset=dataset,
            obs_dim=obs_dim,
            act_dim=act_dim,
            rollout_length=rollout_length,
            n_synthetic_per_step=n_synthetic_per_step,
            sigma_factor=sigma_factor,
            real_ratio=real_ratio,
            real_buffer_size=real_buffer_size,
            synth_buffer_size=synth_buffer_size,
            device=device,
        )
        self.model_train_freq     = int(model_train_freq)
        self.model_train_epochs   = int(model_train_epochs)
        self.online_batch_size    = int(online_batch_size)
        self.min_samples_to_train = int(min_samples_to_train)
        self.online_mix_ratio     = float(online_mix_ratio)
        self.dataset_save_path    = Path(dataset_save_path) if dataset_save_path else None
        self.surrogate_save_dir   = Path(surrogate_save_dir) if surrogate_save_dir else None

        self._updater = SurrogateDatasetUpdater(
            surrogates=self.surrogates,
            offline_dataset=dataset,
            model_dir=self.surrogate_save_dir,
            lr=model_lr,
            batch_size=self.online_batch_size,
            epochs=self.model_train_epochs,
            min_samples=self.min_samples_to_train,
            online_mix_ratio=self.online_mix_ratio,
            device=device,
        )

        self._real_step_count = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        sim_result: Optional[BeamSimulationResult] = None,
    ):
        """Process one real transition, with optional surrogate fine-tuning.

        Args:
            obs:        Current observation.
            action:     Action taken.
            reward:     Received reward.
            next_obs:   Next observation.
            done:       Episode done flag.
            sim_result: Full BeamSimulationResult from TraceWinEnv
                        info["sim_result"]. Only source="tracewin" is added
                        to the online fine-tuning dataset.
                        When provided, added to the online dataset and used
                        to fine-tune the surrogate ensemble.
        """
        if sim_result is not None:
            self._updater.add_tracewin_result(sim_result)

        self._real_step_count += 1

        if self._real_step_count % self.model_train_freq == 0:
            losses = self._updater.update_if_ready()

            if losses is not None and self.dataset_save_path is not None:
                self.save_dataset()

            if losses is not None and self.surrogate_save_dir is not None:
                self.save_surrogates()

        return super().step(obs, action, reward, next_obs, done)

    @property
    def n_online_samples(self) -> int:
        """Numero di risultati TraceWin reali accumulati."""
        return self._updater.n_online_samples

    # ── Persistenza dataset ──────────────────────────────────────────────────────

    def save_dataset(self, path: Optional[str | Path] = None) -> None:
        """Salva l'unione offline+online (dataset 'maestro' aggiornato).

        Se `path` (o `self.dataset_save_path`) coincide col file da cui è stato
        caricato il dataset offline originale, quel file cresce run dopo run.
        """
        save_path = Path(path) if path is not None else self.dataset_save_path
        if save_path is None:
            raise ValueError("Specifica dataset_save_path nel costruttore o path qui")
        self._updater.save_merged_dataset(save_path)

    # ── Persistenza pesi fine-tunati ────────────────────────────────────────────

    def save_surrogates(self, output_dir: Optional[str | Path] = None) -> None:
        """Salva i pesi fine-tunati come surrogate_0.pt..surrogate_N.pt.

        Scrive in `output_dir` (o `self.surrogate_save_dir` se non specificato),
        senza mai toccare i surrogate_*.pt originali caricati a inizio run.
        L'indice del file corrisponde all'indice in `self.surrogates`, che a sua
        volta rispetta l'ordine con cui l'ensemble è stato caricato
        (sorted(glob("surrogate_*.pt"))).
        """
        save_dir = Path(output_dir) if output_dir is not None else self.surrogate_save_dir
        if save_dir is None:
            raise ValueError("Specifica surrogate_save_dir nel costruttore o output_dir qui")
        self._updater.save_surrogates(save_dir)
