"""
Train — allena tutti gli algoritmi in sequenza sul surrogate.

Algoritmi allenati:
  model-free      SAC, TD3, PPO, DDPG, A2C, REINFORCE, TRPO
  model-based     SVGAgent, DynaMBPO (con SAC interno)

Uso rapido (test, pochi step):
    python -m beam_optimization train --quick

Uso completo:
    python -m beam_optimization train \\
        --dataset   env/dataset/base/dataset_base.pt \\
        --single-surrogate env/surrogate_env/surrogate/trained_models/base/surrogate_0.pt \\
        --base-ensemble env/surrogate_env/surrogate/trained_models/base \\
        --output    runs/all \\
        --rl-steps  300000 \\
        --svg-episodes 2000

I checkpoint vengono salvati in:
    runs/all/sac/sac_agent.pt
    runs/all/td3/td3_agent.pt
    runs/all/ppo/ppo_agent.pt
    runs/all/ddpg/ddpg_agent.pt
    runs/all/a2c/a2c_agent.pt
    runs/all/reinforce/reinforce_agent.pt
    runs/all/trpo/trpo_agent.pt
    runs/all/svg_finale/svg_agent.pt
    runs/all/svg_uniform/svg_agent.pt
    runs/all/dyna/dyna_agent.pt
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from beam_optimization.algorithms.utils.logger import Logger
from beam_optimization.config.paths import (
    DEFAULT_BASE_SURROGATE_DIR,
    DEFAULT_DATASET,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SINGLE_SURROGATE_MODEL,
    DEFAULT_TRACEWIN_INI,
    DEFAULT_UPDATED_SURROGATE_DIR,
)
from beam_optimization.env.surrogate_env.surrogate.model.modular_mlp import ModularMLP
from beam_optimization.env.dataset import BeamDataset
from beam_optimization.env.surrogate_env import SurrogateEnv
from beam_optimization.config.adige import (
    N_PARAMS,
    PARAM_KEYS,
    action_bounds,
    default_params,
    observation_dim,
)

ACT_DIM = N_PARAMS  # 16
# OBS_DIM is computed dynamically from the env observation mask.

STAGE_WEIGHT_CONFIGS = {
    "finale":  None,
    "uniform": [1.0] * 11,
}


# ── Surrogate loading ─────────────────────────────────────────────────────────

def _surrogate_files(folder: Path) -> List[Path]:
    return sorted(folder.glob("surrogate_*.pt"))


def _missing_surrogate_message(path: Path) -> str:
    return (
        f"Surrogate checkpoint not found: {path}\n"
        "Create base surrogates with:\n"
        "  python -m beam_optimization setup --target-samples N"
    )


def load_single_surrogate(path: str | Path):
    """Load the single base surrogate used by model-free algorithms."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(_missing_surrogate_message(path))
    model = ModularMLP.load(str(path))
    model.eval()
    print(f"Caricato surrogate singolo per model-free: {path}")
    return model


def load_surrogate_ensemble(folder: str | Path, *, label: str):
    """Load all surrogate_*.pt files from a folder in deterministic order."""
    folder = Path(folder)
    model_files = _surrogate_files(folder)
    if not model_files:
        raise FileNotFoundError(
            f"Nessun surrogate_*.pt trovato in {folder}\n"
            "Create base surrogates with:\n"
            "  python -m beam_optimization setup --target-samples N"
        )
    ensemble = [ModularMLP.load(str(path)) for path in model_files]
    for model in ensemble:
        model.eval()
    print(f"Caricati {len(ensemble)} surrogate per {label}: {folder}")
    return ensemble


def initialize_updated_ensemble_from_base(base_dir: str | Path, updated_dir: str | Path) -> None:
    """Copy base surrogate_*.pt files into updated when updated is empty."""
    base_dir = Path(base_dir)
    updated_dir = Path(updated_dir)

    if _surrogate_files(updated_dir):
        return

    base_files = _surrogate_files(base_dir)
    if not base_files:
        raise FileNotFoundError(
            f"Cannot initialize {updated_dir}: no surrogate_*.pt files in {base_dir}"
        )

    updated_dir.mkdir(parents=True, exist_ok=True)
    for source in base_files:
        shutil.copy2(source, updated_dir / source.name)
    print(
        f"Inizializzato ensemble updated copiando {len(base_files)} surrogate "
        f"da {base_dir} a {updated_dir}"
    )


def apply_legacy_surrogate_arg(args) -> None:
    """Map the legacy --surrogate argument onto the explicit new arguments."""
    if args.surrogate is None:
        return

    legacy_path = Path(args.surrogate)
    print(
        "WARNING: --surrogate è legacy. Usa --single-surrogate per model-free "
        "oppure --base-ensemble per SVG/MBPO."
    )
    if legacy_path.suffix == ".pt":
        args.single_surrogate = str(legacy_path)
    else:
        args.base_ensemble = str(legacy_path)


def _make_logger(out_dir: Path, algorithm: str, enable_tensorboard: bool):
    if not enable_tensorboard:
        return None
    return Logger(out_dir, algorithm=algorithm)


def _loss_metrics(algo: str, losses) -> dict[str, float]:
    if losses is None:
        return {}
    values = list(losses) if isinstance(losses, (tuple, list)) else [losses]
    values = [None if value is None else float(value) for value in values]

    if algo == "sac":
        names = ["critic_loss", "actor_loss", "entropy_loss"]
    elif algo in {"td3", "ddpg"}:
        names = ["critic_loss", "actor_loss", "entropy_loss"]
    else:
        names = ["value_loss", "policy_loss"]

    return {
        name: value
        for name, value in zip(names, values)
        if value is not None
    }


# ── Training loops ────────────────────────────────────────────────────────────

def train_rl(algo: str, surrogate, dataset, n_steps, max_ep_steps,
             hidden, out_dir: Path,
             enable_tensorboard: bool = True) -> float:
    """Allena un algoritmo model-free nell'ambiente surrogate."""
    # Create env first so obs_dim is known before building the agent
    env = SurrogateEnv(model=surrogate, dataset=dataset, max_steps=max_ep_steps)
    obs_dim = env.observation_space.shape[0]

    act_bds = action_bounds()
    bounds  = (act_bds[0].tolist(), act_bds[1].tolist())

    if algo == "sac":
        from beam_optimization.algorithms.model_free.sac import SAC
        agent = SAC(obs_dim, ACT_DIM, bounds, hidden_dims=tuple(hidden))
    elif algo == "td3":
        from beam_optimization.algorithms.model_free.td3 import TD3
        agent = TD3(obs_dim, ACT_DIM, bounds, hidden_dims=tuple(hidden))
    elif algo == "ppo":
        from beam_optimization.algorithms.model_free.ppo import PPO
        agent = PPO(obs_dim, ACT_DIM, bounds, hidden_dims=tuple(hidden))
    elif algo == "ddpg":
        from beam_optimization.algorithms.model_free.ddpg import DDPG
        agent = DDPG(obs_dim, ACT_DIM, bounds, hidden_dims=tuple(hidden))
    elif algo == "a2c":
        from beam_optimization.algorithms.model_free.a2c import A2C
        agent = A2C(obs_dim, ACT_DIM, bounds, hidden_dims=tuple(hidden))
    elif algo == "reinforce":
        from beam_optimization.algorithms.model_free.reinforce import REINFORCE
        agent = REINFORCE(obs_dim, ACT_DIM, bounds, hidden_dims=tuple(hidden))
    elif algo == "trpo":
        from beam_optimization.algorithms.model_free.trpo import TRPO
        agent = TRPO(obs_dim, ACT_DIM, bounds, hidden_dims=tuple(hidden))
    else:
        raise ValueError(algo)

    obs, _     = env.reset()
    best_score = -np.inf
    ep_reward  = 0.0
    ep_count   = 0
    log_every  = max(1, n_steps // 20)
    logger = _make_logger(out_dir, algo, enable_tensorboard)

    on_policy = algo in ("ppo", "a2c", "trpo", "reinforce")

    try:
        for step in range(1, n_steps + 1):
            if on_policy:
                action, logpa, value = agent.select_action(obs, training=True)
            else:
                action = agent.select_action(obs, training=True)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if on_policy:
                agent.store(obs, action, reward, value, logpa, float(done))
            else:
                agent.store(obs, action, reward, next_obs, float(done))

            optimize_result = None
            if on_policy:
                if done:
                    last_val = 0.0
                    if not terminated:
                        _, _, last_val = agent.select_action(obs, training=True)
                    optimize_result = agent.optimize(last_value=float(last_val))
            else:
                optimize_result = agent.optimize()

            if logger is not None:
                metrics = {
                    "reward": float(reward),
                    "action_norm": float(np.linalg.norm(action)),
                }
                metrics.update(_loss_metrics(algo, optimize_result))
                if len(metrics) > 2 or step % log_every == 0:
                    logger.log(metrics, step=step)

            obs        = next_obs
            ep_reward += reward

            if done:
                ep_count += 1
                sc = info.get("score", 0.0)
                best_score = max(best_score, sc)
                if logger is not None:
                    logger.log(
                        {
                            "episode_reward": float(ep_reward),
                            "score": float(sc),
                            "best_score": float(best_score),
                            "episode": float(ep_count),
                        },
                        step=step,
                    )
                if step % log_every == 0:
                    print(f"  step={step:>7d}  ep={ep_count:>4d}  "
                          f"reward={ep_reward:.2f}  score={sc:.3f}  best={best_score:.3f}")
                obs, _ = env.reset()
                ep_reward = 0.0

        out_dir.mkdir(parents=True, exist_ok=True)
        agent.save(str(out_dir / f"{algo}_agent.pt"))
        print(f"  Salvato → {out_dir / f'{algo}_agent.pt'}")
        return best_score
    finally:
        if logger is not None:
            logger.close()


def train_sb3_sac(surrogate, dataset, n_steps, max_ep_steps,
                  hidden, out_dir: Path,
                  enable_tensorboard: bool = True) -> float:
    """Allena SAC di Stable Baselines 3 sull'ambiente surrogate."""
    from beam_optimization.algorithms.model_free.sb3_sac import SB3SAC

    env = SurrogateEnv(model=surrogate, dataset=dataset, max_steps=max_ep_steps)

    logger = _make_logger(out_dir, "sb3_sac", enable_tensorboard)
    agent = SB3SAC(
        env,
        hidden_dims=tuple(hidden),
        tensorboard_log=str(out_dir) if enable_tensorboard else None,
    )
    try:
        best_score = agent.train(
            env,
            n_steps=n_steps,
            log_every=max(1, n_steps // 20),
            logger=logger,
        )
    finally:
        if logger is not None:
            logger.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    agent.save(str(out_dir / "sb3_sac_agent"))
    print(f"  Salvato → {out_dir / 'sb3_sac_agent.zip'}")
    return best_score


def train_svg(surrogate, dataset, n_episodes, horizon, hidden,
              stage_weights, out_dir: Path,
              enable_tensorboard: bool = True) -> float:
    from beam_optimization.algorithms.model_based.svg import SVGAgent

    agent = SVGAgent(
        surrogate=surrogate,
        dataset=dataset,
        obs_dim=observation_dim(),
        act_dim=ACT_DIM,
        action_bounds=tuple(v.tolist() for v in action_bounds()),
        param_keys=PARAM_KEYS,
        default_params=default_params(),
        hidden_dims=tuple(hidden),
        n_step=horizon,
        stage_weights=stage_weights,
    )

    best_score = -np.inf
    log_every  = max(1, n_episodes // 20)
    logger = _make_logger(out_dir, "svg", enable_tensorboard)

    try:
        for ep in range(1, n_episodes + 1):
            result     = agent.optimize_episode()
            best_score = max(best_score, result.final_score)
            if logger is not None:
                logger.log(
                    {
                        "episode_loss": float(result.episode_loss),
                        "final_score": float(result.final_score),
                        "best_score": float(best_score),
                        "grad_norm": float(result.grad_norm),
                        "episode": float(ep),
                    },
                    step=ep,
                )
            if ep % log_every == 0:
                print(f"  ep={ep:>5d}  loss={result.episode_loss:.4f}  "
                      f"score={result.final_score:.3f}  best={best_score:.3f}")

        out_dir.mkdir(parents=True, exist_ok=True)
        agent.save(str(out_dir / "svg_agent.pt"))
        print(f"  Salvato → {out_dir / 'svg_agent.pt'}")
        return best_score
    finally:
        if logger is not None:
            logger.close()


def train_dyna(surrogate, dataset, n_steps, max_ep_steps,
               rollout_length, hidden, out_dir: Path,
               tracewin_project: Optional[str] = None,
               online_finetune: bool = False,
               online_mix_ratio: float = 0.5,
               update_dataset_path: Optional[str] = None,
               surrogate_path: Optional[Path] = None,
               dataset_path: Optional[Path] = None,
               update_surrogates_path: Optional[str] = None,
               enable_tensorboard: bool = True) -> float:
    """
    Allena MBPO con surrogato ensemble per i rollout sintetici.

    Se tracewin_project è fornito, l'ambiente REALE usa TraceWin (fisica vera,
    ~30 s/step). I rollout SINTETICI usano sempre il surrogato (veloce).
    Senza tracewin_project, sia l'env reale che i rollout usano il surrogato.

    Se online_finetune=True (richiede tracewin_project), usa MBPOWithModelUpdate
    che affina il surrogato ad ogni model_train_freq step reali mescolando dati
    offline (dataset originale) e online (TraceWin raccolto in questa run, quota
    online_mix_ratio). In questo caso:
      - i pesi fine-tunati vengono salvati in models/updated. Se parti da
        models/base, base resta conservato; se parti da models/updated, la run
        aggiorna la working copy in-place. Override con update_surrogates_path;
      - il dataset aggiornato (offline+online unito) viene salvato di default
        nello stesso file passato con --dataset. Per default questo e
        env/dataset/base/dataset_base.pt, quindi il dataset base cresce run
        dopo run. Puoi usare update_dataset_path per salvarlo altrove.
    """
    from beam_optimization.algorithms.model_free.sac import SAC

    use_model_update = online_finetune and bool(tracewin_project)

    if use_model_update:
        from beam_optimization.algorithms.model_based.mbpo_model_update import MBPOWithModelUpdate
        MBPOClass = MBPOWithModelUpdate
    else:
        from beam_optimization.algorithms.model_based.mbpo import MBPO
        MBPOClass = MBPO

    if tracewin_project:
        from beam_optimization.env.tracewin_env import TraceWinEnv
        env = TraceWinEnv(
            project_file=tracewin_project,
            max_steps=max_ep_steps,
        )
        label = "MBPOWithModelUpdate" if use_model_update else "MBPO"
        print(f"  Env reale: TraceWin  ({tracewin_project})  [{label}]")
    else:
        env = SurrogateEnv(model=surrogate, dataset=dataset, max_steps=max_ep_steps)
        print("  Env reale: surrogato (SurrogateEnv)  [MBPO]")

    obs_dim = env.observation_space.shape[0]
    act_bds = action_bounds()
    bounds  = (act_bds[0].tolist(), act_bds[1].tolist())
    inner   = SAC(obs_dim, ACT_DIM, bounds, hidden_dims=tuple(hidden))

    mbpo_kwargs = dict(
        agent=inner,
        surrogates=surrogate,
        dataset=dataset,
        obs_dim=obs_dim,
        act_dim=ACT_DIM,
        rollout_length=rollout_length,
    )
    if use_model_update:
        mbpo_kwargs["online_mix_ratio"] = online_mix_ratio

        if update_dataset_path is not None:
            dataset_save_path = Path(update_dataset_path)
        elif dataset_path is not None:
            # By default MBPOWithModelUpdate writes the merged offline+online
            # dataset back to the base dataset used for beam0/offline samples.
            dataset_save_path = dataset_path
        else:
            dataset_save_path = out_dir / "updated_dataset.pt"
        mbpo_kwargs["dataset_save_path"] = dataset_save_path

        if update_surrogates_path is not None:
            surrogate_save_dir = Path(update_surrogates_path)
        elif surrogate_path is not None:
            base_dir = surrogate_path if surrogate_path.is_dir() else surrogate_path.parent
            if base_dir.name == "base":
                surrogate_save_dir = base_dir.parent / "updated"
            elif base_dir.name == "updated":
                surrogate_save_dir = base_dir
            else:
                surrogate_save_dir = base_dir / "updated"
        else:
            surrogate_save_dir = out_dir / "updated"
        mbpo_kwargs["surrogate_save_dir"] = surrogate_save_dir
    mbpo = MBPOClass(**mbpo_kwargs)

    obs, _     = env.reset()
    best_score = -np.inf
    ep_reward  = 0.0
    ep_count   = 0
    log_every  = max(1, n_steps // 20)
    logger = _make_logger(out_dir, "mbpo", enable_tensorboard)

    try:
        for step in range(1, n_steps + 1):
            action   = mbpo.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done     = terminated or truncated

            if use_model_update:
                optimize_result = mbpo.step(obs, action, reward, next_obs, done,
                                            sim_result=info.get("sim_result"))
            else:
                optimize_result = mbpo.step(obs, action, reward, next_obs, done)

            if logger is not None:
                metrics = {
                    "real_reward": float(reward),
                    "reward": float(reward),
                    "action_norm": float(np.linalg.norm(action)),
                }
                metrics.update(_loss_metrics("sac", optimize_result))
                if use_model_update:
                    metrics["online_samples"] = float(mbpo.n_online_samples)
                    update_losses = getattr(mbpo, "last_update_losses", None)
                    update_freq = max(1, int(getattr(mbpo, "model_train_freq", 1)))
                    if update_losses is not None and step % update_freq == 0:
                        update_values = [float(v) for v in update_losses.values()]
                        if update_values:
                            metrics["model_update_loss"] = float(np.mean(update_values))
                        for name, value in update_losses.items():
                            metrics[f"model_update_loss/{name}"] = float(value)
                if len(metrics) > 3 or step % log_every == 0:
                    logger.log(metrics, step=step)

            obs       = next_obs
            ep_reward += reward

            if done:
                ep_count  += 1
                sc         = info.get("score", 0.0)
                best_score = max(best_score, sc)
                if logger is not None:
                    metrics = {
                        "episode_reward": float(ep_reward),
                        "score": float(sc),
                        "best_score": float(best_score),
                        "episode": float(ep_count),
                    }
                    if use_model_update:
                        metrics["online_samples"] = float(mbpo.n_online_samples)
                    logger.log(metrics, step=step)
                if step % log_every == 0:
                    extra = (f"  online_samples={mbpo.n_online_samples}"
                             if use_model_update else "")
                    print(f"  step={step:>7d}  ep={ep_count:>4d}  "
                          f"reward={ep_reward:.2f}  score={sc:.3f}  best={best_score:.3f}{extra}")
                obs, _ = env.reset()
                ep_reward = 0.0

        out_dir.mkdir(parents=True, exist_ok=True)
        inner.save(str(out_dir / "dyna_agent.pt"))
        print(f"  Salvato → {out_dir / 'dyna_agent.pt'}")

        if use_model_update:
            # flush finale, anche fuori dal periodo model_train_freq
            mbpo.save_surrogates()
            mbpo.save_dataset()

        return best_score
    finally:
        if logger is not None:
            logger.close()


# ── Tabella finale ────────────────────────────────────────────────────────────

def print_summary(scores: Dict[str, float]):
    print(f"\n{'='*50}")
    print("RIEPILOGO TRAINING")
    print(f"{'Algoritmo':<35}  {'Best Score':>10}")
    print("-" * 50)
    for name, sc in sorted(scores.items()):
        print(f"{name:<35}  {sc:>10.3f}")
    print(f"{'='*50}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Allena tutti gli algoritmi")
    parser.add_argument("--surrogate",      default=None,
                        help="Legacy alias: file .pt -> --single-surrogate; "
                             "cartella -> --base-ensemble.")
    parser.add_argument("--single-surrogate", default=str(DEFAULT_SINGLE_SURROGATE_MODEL),
                        metavar="PATH",
                        help="Surrogate singolo usato da SAC/PPO/TD3/DDPG/A2C/"
                             "REINFORCE/TRPO/SB3-SAC. Default: "
                             f"{DEFAULT_SINGLE_SURROGATE_MODEL}")
    parser.add_argument("--base-ensemble", default=str(DEFAULT_BASE_SURROGATE_DIR),
                        metavar="PATH",
                        help="Cartella con surrogate_*.pt usata da SVG e MBPO. "
                             f"Default: {DEFAULT_BASE_SURROGATE_DIR}")
    parser.add_argument("--updated-ensemble", default=str(DEFAULT_UPDATED_SURROGATE_DIR),
                        metavar="PATH",
                        help="Cartella working usata solo da MBPOWithModelUpdate. "
                             "Se vuota viene inizializzata copiando --base-ensemble. "
                             f"Default: {DEFAULT_UPDATED_SURROGATE_DIR}")
    parser.add_argument("--dataset",        default=str(DEFAULT_DATASET))
    parser.add_argument("--output",         default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--rl-steps",       type=int, default=200_000)
    parser.add_argument("--svg-episodes",   type=int, default=1000)
    parser.add_argument("--svg-horizon",    type=int, default=20)
    parser.add_argument("--rollout-length", type=int, default=1,
                        help="Rollout length DynaMBPO (1=Dyna, >1=MBPO)")
    parser.add_argument("--max-ep-steps",   type=int, default=20)
    parser.add_argument("--hidden",         type=int, nargs="+", default=[256, 256])
    parser.add_argument("--quick",          action="store_true",
                        help="Budget ridotto per test rapido")
    parser.add_argument("--no-tensorboard", action="store_true",
                        help="Disabilita logging TensorBoard e metrics.csv.")
    parser.add_argument("--skip",           nargs="*", default=[],
                        help="Algoritmi da saltare (es: --skip dyna ppo)")
    parser.add_argument("--tracewin",       default=None, metavar="INI",
                        nargs="?", const=str(DEFAULT_TRACEWIN_INI),
                        help="Usa TraceWin come env reale per MBPO. "
                             "Senza argomento usa il path default del progetto.")
    parser.add_argument("--online-finetune", action="store_true",
                        help="Fine-tuna il surrogate ensemble sui dati reali "
                             "durante il training (MBPOWithModelUpdate). "
                             "Richiede --tracewin.")
    parser.add_argument("--online-mix-ratio", type=float, default=0.5, metavar="FLOAT",
                        help="Quota target (0-1) di ogni batch di fine-tuning presa dai "
                             "dati online raccolti in questa run; il resto viene dal "
                             "dataset offline originale (evita di scordare la "
                             "distribuzione originale). Default: 0.5.")
    parser.add_argument("--update-dataset",  default=None, metavar="PATH",
                        help="Path .pt dove salvare il dataset aggiornato (offline+online "
                             "uniti) raccolto da MBPOWithModelUpdate. Default: stesso "
                             "path di --dataset, quindi dataset_base.pt cresce run dopo "
                             "run. Richiede --online-finetune.")
    parser.add_argument("--update-surrogates", default=None, metavar="PATH",
                        help="Cartella dove salvare i surrogate_*.pt fine-tunati da "
                             "MBPOWithModelUpdate. Legacy alias di --updated-ensemble.")
    args = parser.parse_args()

    if args.quick:
        args.rl_steps     = 200
        args.svg_episodes = 5

    apply_legacy_surrogate_arg(args)
    if args.update_surrogates is not None:
        print("WARNING: --update-surrogates è legacy. Usa --updated-ensemble.")
        args.updated_ensemble = args.update_surrogates
    if args.online_finetune and not args.tracewin:
        parser.error("--online-finetune richiede --tracewin")

    out_root = Path(args.output)
    skip     = set(args.skip)
    enable_tensorboard = not args.no_tensorboard

    single_surrogate_path = Path(args.single_surrogate)
    base_ensemble_path = Path(args.base_ensemble)
    updated_ensemble_path = Path(args.updated_ensemble)

    model_free_algos = ["sac", "td3", "ppo", "ddpg", "a2c", "reinforce", "trpo"]
    run_model_free = any(algo not in skip for algo in model_free_algos)
    run_sb3_sac = "sb3_sac" not in skip
    run_dyna = "dyna" not in skip
    run_svg = any(
        f"svg_{name}" not in skip and "svg" not in skip
        for name in STAGE_WEIGHT_CONFIGS
    )
    use_model_update = args.online_finetune and bool(args.tracewin)

    single_surrogate = None
    if run_model_free or run_sb3_sac:
        single_surrogate = load_single_surrogate(single_surrogate_path)

    base_ensemble = None
    if run_svg or (run_dyna and not use_model_update):
        base_ensemble = load_surrogate_ensemble(base_ensemble_path, label="base ensemble")

    updated_ensemble = None
    if run_dyna and use_model_update:
        initialize_updated_ensemble_from_base(base_ensemble_path, updated_ensemble_path)
        updated_ensemble = load_surrogate_ensemble(
            updated_ensemble_path,
            label="updated ensemble",
        )

    print(f"Carico dataset:   {args.dataset}")
    dataset = BeamDataset.load(args.dataset)

    scores: Dict[str, float] = {}

    # ── Model-free RL ─────────────────────────────────────────────────────────
    for algo in ["sac", "td3", "ppo", "ddpg", "a2c", "reinforce", "trpo"]:
        if algo in skip:
            print(f"\n[SKIP] {algo.upper()}")
            continue
        print(f"\n{'='*50}\nTraining {algo.upper()}  ({args.rl_steps} steps)\n{'='*50}")
        scores[algo] = train_rl(
            algo, single_surrogate, dataset, args.rl_steps,
            args.max_ep_steps, args.hidden,
            out_root / algo,
            enable_tensorboard=enable_tensorboard,
        )

    # ── SB3-SAC ───────────────────────────────────────────────────────────────
    if "sb3_sac" not in skip:
        print(f"\n{'='*50}\nTraining SB3-SAC  ({args.rl_steps} steps)\n{'='*50}")
        scores["sb3_sac"] = train_sb3_sac(
            single_surrogate, dataset, args.rl_steps,
            args.max_ep_steps, args.hidden,
            out_root / "sb3_sac",
            enable_tensorboard=enable_tensorboard,
        )

    # ── MBPO ──────────────────────────────────────────────────────────────────
    if "dyna" not in skip:
        tw_label  = "TraceWin" if args.tracewin else "surrogato"
        use_mu    = args.online_finetune and bool(args.tracewin)
        alg_label = "MBPOWithModelUpdate" if use_mu else "MBPO"
        print(f"\n{'='*50}\nTraining {alg_label} [{tw_label}]  ({args.rl_steps} steps, "
              f"rollout={args.rollout_length})\n{'='*50}")
        dyna_surrogate = updated_ensemble if use_mu else base_ensemble
        dyna_surrogate_path = updated_ensemble_path if use_mu else base_ensemble_path
        scores["mbpo"] = train_dyna(
            dyna_surrogate, dataset, args.rl_steps,
            args.max_ep_steps, args.rollout_length, args.hidden,
            out_root / "dyna",
            tracewin_project=args.tracewin,
            online_finetune=args.online_finetune,
            online_mix_ratio=args.online_mix_ratio,
            update_dataset_path=args.update_dataset,
            surrogate_path=dyna_surrogate_path,
            dataset_path=Path(args.dataset),
            update_surrogates_path=str(updated_ensemble_path) if use_mu else None,
            enable_tensorboard=enable_tensorboard,
        )

    # ── SVGAgent ─────────────────────────────────────────────────────────────
    for name, weights in STAGE_WEIGHT_CONFIGS.items():
        label = f"svg_{name}"
        if label in skip or "svg" in skip:
            print(f"\n[SKIP] {label}")
            continue
        print(f"\n{'='*50}\nTraining SVGAgent [{name}]  ({args.svg_episodes} episodi)\n{'='*50}")
        scores[label] = train_svg(
            base_ensemble, dataset, args.svg_episodes, args.svg_horizon,
            args.hidden, weights,
            out_root / label,
            enable_tensorboard=enable_tensorboard,
        )

    print_summary(scores)

    summary_path = out_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\nRiepilogo salvato → {summary_path}")


if __name__ == "__main__":
    main()
