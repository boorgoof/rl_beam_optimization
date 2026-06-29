"""
MBPOWithModelUpdate — MBPO variant where the surrogate ensemble is also
fine-tuned on real TraceWin transitions as they are collected during RL
training, mirroring the original MBPO paper (Janner 2019) where the dynamics
model is trained on real data.

Differences from MBPO (mbpo.py):
  - Surrogate models are fine-tuned every `model_train_freq` real steps
    using gradient steps on a bootstrap sample of accumulated real data.
  - `step()` accepts an optional `sim_result` (from TraceWinEnv info dict)
    so the real TraceWin output can be added to the model-update dataset.
  - One Adam optimizer per surrogate is maintained throughout training.
  - Ogni batch di fine-tuning mescola dati offline (il `dataset` seed passato
    al costruttore) e dati online (`_online_dataset`, raccolti in questa
    run), con una quota target `online_mix_ratio` — evita che il surrogate
    scordi la distribuzione originale mentre recepisce il feedback reale.
  - Se `dataset_save_path` è fornito, l'unione offline+online viene salvata
    periodicamente in formato flat (dataset "maestro" aggiornato).

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
        dataset_save_path="dataset/updated/dataset_train.pt",
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
import torch
import torch.nn as nn

from beam_optimization.algorithms.model_based.mbpo import MBPO
from beam_optimization.env.surrogate_env.surrogate.modular_mlp import ModularMLP
from beam_optimization.env.simulation import BeamSimulationResult
from beam_optimization.env.surrogate_env.surrogate.dataset import SurrogateTrainingDataset
from beam_optimization.config.adige import N_OUTPUT_STAGES


class MBPOWithModelUpdate(MBPO):
    """MBPO with surrogate model update step on real TraceWin data.

    Args:
        agent:                SAC or TD3 instance.
        surrogates:           Trained ModularMLP or list (ensemble).
        dataset:              Seed SurrogateTrainingDataset for beam0 sampling in synthetic rollouts.
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
        online_mix_ratio:     Quota target (0-1) di ogni batch di fine-tuning presa da
                              `_online_dataset`; il resto viene da `_offline_dataset`
                              (il `dataset` seed). Se i dati online disponibili sono
                              meno della quota richiesta, si scala automaticamente a
                              quanto c'è e si riempie il resto con dati offline — il
                              batch resta sempre di dimensione `online_batch_size`.
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
        dataset: SurrogateTrainingDataset,
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

        # Dataset offline (seed, statico) + online (cresce con .add() durante la run)
        self._offline_dataset = dataset
        self._online_dataset  = SurrogateTrainingDataset()

        # Persistent optimizer per surrogate
        self._surrogate_optimizers = [
            torch.optim.Adam(s.parameters(), lr=model_lr)
            for s in self.surrogates
        ]

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
        if sim_result is not None and sim_result.source == "tracewin":
            self._online_dataset.add(sim_result)

        self._real_step_count += 1

        if (self._real_step_count % self.model_train_freq == 0
                and len(self._online_dataset) >= self.min_samples_to_train):
            self._finetune_surrogates()

            # Salva l'unione offline+online (dataset "maestro" aggiornato)
            if self.dataset_save_path is not None:
                self.save_dataset()

            # Salva i pesi fine-tunati (mai nei surrogate_*.pt originali)
            if self.surrogate_save_dir is not None:
                self.save_surrogates()

        return super().step(obs, action, reward, next_obs, done)

    # ── Surrogate fine-tuning ──────────────────────────────────────────────────

    def _finetune_surrogates(self):
        """Bootstrap fine-tune each surrogate on a mix of offline + online data."""
        criterion = nn.MSELoss()
        stage_w   = 1.0 / N_OUTPUT_STAGES

        for surrogate, opt in zip(self.surrogates, self._surrogate_optimizers):
            surrogate.train()
            for _ in range(self.model_train_epochs):
                stage_t, beam0_t, targets_t = self._collate_mixed(self.online_batch_size)

                preds = surrogate(stage_t, beam0_t)
                loss  = sum(stage_w * criterion(p, t) for p, t in zip(preds, targets_t))

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(surrogate.parameters(), max_norm=1.0)
                opt.step()

            surrogate.eval()

    def _collate_mixed(self, n_total: int):
        """Bootstrap batch misto: quota online_mix_ratio da _online_dataset, il
        resto da _offline_dataset. Se i dati online disponibili sono meno della
        quota richiesta, si scala automaticamente e si riempie con dati offline
        (il batch resta sempre di dimensione n_total)."""
        N_online  = len(self._online_dataset)
        N_offline = len(self._offline_dataset)
        n_online  = min(int(n_total * self.online_mix_ratio), N_online)
        n_offline = n_total - n_online

        idx_off = np.random.choice(N_offline, size=n_offline, replace=True)
        stage, beam = self._offline_dataset.get_training_batch(idx_off)

        if n_online > 0:
            idx_on = np.random.choice(N_online, size=n_online, replace=True)
            stage_on, beam_on = self._online_dataset.get_training_batch(idx_on)
            stage = [torch.cat([a, b], dim=0) for a, b in zip(stage, stage_on)]
            beam  = [torch.cat([a, b], dim=0) for a, b in zip(beam, beam_on)]

        stage_t  = [t.to(self.device) for t in stage]
        beam_all = [t.to(self.device) for t in beam]
        return stage_t, beam_all[0], beam_all[1:]

    @property
    def n_online_samples(self) -> int:
        """Numero di risultati TraceWin reali accumulati."""
        return len(self._online_dataset)

    # ── Persistenza dataset ──────────────────────────────────────────────────────

    def save_dataset(self, path: Optional[str | Path] = None) -> None:
        """Salva l'unione offline+online (dataset 'maestro' aggiornato).

        Se `path` (o `self.dataset_save_path`) coincide col file da cui è stato
        caricato il dataset offline originale, quel file cresce run dopo run.
        """
        save_path = Path(path) if path is not None else self.dataset_save_path
        if save_path is None:
            raise ValueError("Specifica dataset_save_path nel costruttore o path qui")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        merged = self._offline_dataset.merge(self._online_dataset)
        merged.save_flat(save_path)

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
        save_dir.mkdir(parents=True, exist_ok=True)
        for i, surrogate in enumerate(self.surrogates):
            surrogate.save(
                str(save_dir / f"surrogate_{i}.pt"),
                extra={
                    "normalization_metadata": surrogate._norm_stats,
                    "online_finetune_step": self._real_step_count,
                    "online_samples": len(self._online_dataset),
                },
            )
        print(f"[MBPOWithModelUpdate] salvati {len(self.surrogates)} surrogati fine-tunati in {save_dir}")
