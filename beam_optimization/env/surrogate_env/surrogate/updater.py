"""
SurrogateUpdater — fine-tuning offline dell'ensemble di surrogati.

Raccoglie BeamSimulationResult da TraceWin (tipicamente durante le fasi di valutazione)
e aggiorna ogni surrogate dell'ensemble con un bootstrap draw indipendente,
mantenendo la diversità dell'ensemble.

I dati vengono archiviati nel SurrogateTrainingDataset interno (formato flat nativo).
export_flat() permette di salvare i campioni raccolti come flat .pt per uso futuro.

Flusso tipico:
    updater = SurrogateUpdater(surrogates, model_dir="env/surrogate_env/surrogate/models/models_finetuned")

    # durante il training RL, ogni N episodi:
    result = tracewin_env.evaluate_params(agent.best_params)
    updater.add(sim_result)

    # quando hai abbastanza dati:
    if updater.n_samples >= 20:
        updater.update()   # fine-tuna tutti i surrogati

    # salva i dati raccolti nel formato flat ml_dataset
    updater.export_flat("env/tracewin_env/dataset/updated/finetuned.pt")

    # opzionale: salva i pesi del surrogate
    updater.save()
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from beam_optimization.env.surrogate_env.surrogate.modular_mlp import ModularMLP
from beam_optimization.env.surrogate_env.surrogate.dataset import SurrogateTrainingDataset
from beam_optimization.env.simulation import BeamSimulationResult
from beam_optimization.config.adige import N_STAGES


class SurrogateUpdater:
    """Fine-tuna l'ensemble di surrogati con dati reali da TraceWin.

    Args:
        surrogates:     Lista di ModularMLP (o singolo modello).
        model_dir:      Cartella dove salvare i pesi aggiornati (opzionale).
        lr:             Learning rate Adam per il fine-tuning.
        batch_size:     Dimensione del bootstrap sample per ogni surrogate.
        epochs:         Gradient steps per ogni chiamata a update().
        min_samples:    Numero minimo di campioni prima di aggiornare.
        device:         Device torch ('cpu', 'cuda', o None per auto).
        flat_save_path: Se fornito, salva automaticamente i nuovi campioni
                        su questo path flat .pt dopo ogni update().
    """

    def __init__(
        self,
        surrogates: Union[ModularMLP, List[ModularMLP]],
        model_dir: Optional[str] = None,
        lr: float = 3e-5,
        batch_size: int = 64,
        epochs: int = 20,
        min_samples: int = 20,
        device: Optional[str] = None,
        flat_save_path: Optional[str | Path] = None,
    ):
        self.surrogates  = surrogates if isinstance(surrogates, list) else [surrogates]
        self.model_dir   = Path(model_dir) if model_dir else None
        self.batch_size  = int(batch_size)
        self.epochs      = int(epochs)
        self.min_samples = int(min_samples)
        self.device      = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.flat_save_path = Path(flat_save_path) if flat_save_path else None

        for s in self.surrogates:
            s.to(self.device)

        self._dataset = SurrogateTrainingDataset()

        # Un ottimizzatore persistente per surrogate (evita cold-start Adam)
        self._optimizers = [
            torch.optim.Adam(s.parameters(), lr=lr)
            for s in self.surrogates
        ]

        self._update_count = 0

    # ── Aggiunta dati ──────────────────────────────────────────────────────────

    def add(self, result: BeamSimulationResult) -> bool:
        """Aggiungi un risultato TraceWin al dataset interno.

        Returns:
            True se il risultato è stato aggiunto (simulazione riuscita).
        """
        if result.source != "tracewin":
            return False
        return self._dataset.add(result)

    def add_many(self, results: List[BeamSimulationResult]) -> int:
        """Aggiungi una lista di risultati. Restituisce quanti sono stati aggiunti."""
        return sum(1 for r in results if self.add(r))

    @property
    def n_samples(self) -> int:
        return len(self._dataset)

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self) -> Optional[dict]:
        """Fine-tuna ogni surrogate con un bootstrap draw indipendente.

        Returns:
            Dict con le loss finali per ogni surrogate, o None se skip.
        """
        N = len(self._dataset)
        if N < self.min_samples:
            print(f"[SurrogateUpdater] skip: {N} < {self.min_samples} campioni minimi")
            return None

        criterion = nn.MSELoss()
        stage_w   = 1.0 / N_STAGES
        final_losses = {}

        for i, (surrogate, opt) in enumerate(zip(self.surrogates, self._optimizers)):
            idx = np.random.choice(N, size=min(N, self.batch_size), replace=True)

            surrogate.train()
            for p in surrogate.parameters():
                p.requires_grad_(True)
            last_loss = 0.0
            for _ in range(self.epochs):
                np.random.shuffle(idx)
                stage_t, beam0_t, targets_t = self._collate(idx)

                preds = surrogate(stage_t, beam0_t)
                loss  = sum(stage_w * criterion(p, t) for p, t in zip(preds, targets_t))

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(surrogate.parameters(), max_norm=1.0)
                opt.step()
                last_loss = float(loss.detach())

            surrogate.eval()
            final_losses[f"surrogate_{i}"] = last_loss

        self._update_count += 1
        print(f"[SurrogateUpdater] update #{self._update_count}  "
              f"N={N}  losses={{{', '.join(f'{k}: {v:.5f}' for k, v in final_losses.items())}}}")

        # Auto-save flat .pt se configurato
        if self.flat_save_path is not None:
            self.export_flat(self.flat_save_path)

        return final_losses

    def update_if_ready(self) -> Optional[dict]:
        """Chiama update() solo se ci sono abbastanza campioni."""
        if self.n_samples >= self.min_samples:
            return self.update()
        return None

    # ── Export flat ───────────────────────────────────────────────────────────

    def export_flat(self, path: str | Path) -> int:
        """Salva tutti i campioni accumulati come flat .pt (formato ml_dataset).

        Se il file esiste già, i nuovi campioni vengono aggiunti in coda.

        Returns:
            Numero di campioni nel dataset dopo il salvataggio.
        """
        if path is None:
            raise ValueError("Specifica un path per export_flat()")
        self._dataset.save_flat(path)
        return len(self._dataset)

    # ── Salvataggio pesi ──────────────────────────────────────────────────────

    def save(self, model_dir: Optional[str] = None):
        """Salva i pesi aggiornati come surrogate_0.pt … surrogate_N.pt."""
        save_dir = Path(model_dir) if model_dir else self.model_dir
        if save_dir is None:
            raise ValueError("Specifica model_dir nel costruttore o in save()")
        save_dir.mkdir(parents=True, exist_ok=True)
        for i, surrogate in enumerate(self.surrogates):
            surrogate.save(
                str(save_dir / f"surrogate_{i}.pt"),
                extra={
                    "normalization_metadata": surrogate._norm_stats,
                    "update_count": self._update_count,
                },
            )
        print(f"[SurrogateUpdater] salvati {len(self.surrogates)} surrogati in {save_dir}")

    # ── Interno ───────────────────────────────────────────────────────────────

    def _collate(self, indices: np.ndarray):
        """Collate dei dati per il training: usa get_training_batch() del dataset."""
        stage_params, beam_states = self._dataset.get_training_batch(indices)
        # Sposta sul device corretto
        stage_t  = [t.to(self.device) for t in stage_params]   # list[11] × (batch, sz)
        beam_all = [t.to(self.device) for t in beam_states]    # list[12] × (batch, 9)
        return stage_t, beam_all[0], beam_all[1:]               # stage_params, beam0, targets
