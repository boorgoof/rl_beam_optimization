"""
Train — allena tutti gli algoritmi in sequenza sul surrogate.

Algoritmi allenati:
  model-free      SAC, TD3, PPO, DDPG, A2C, REINFORCE, TRPO
  model-based     SVGAgent, DynaMBPO (con SAC interno)

Uso rapido (test, pochi step):
    python -m beam_optimization train --quick

Uso completo:
    python -m beam_optimization train \\
        --surrogate env/surrogate_env/surrogate/models/updated \\
        --dataset   env/dataset/base/dataset_train.pt \\
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
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from beam_optimization.config.paths import (
    DEFAULT_DATASET,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SURROGATE_DIR,
    DEFAULT_TRACEWIN_INI,
)
from beam_optimization.env.surrogate_env.surrogate.modular_mlp import ModularMLP
from beam_optimization.env.dataset import SurrogateTrainingDataset
from beam_optimization.env.surrogate_env import SurrogateEnv
from beam_optimization.config.adige import (
    N_PARAMS, N_STAGES, BEAM_STATE_DIM,
    action_bounds,
)

ACT_DIM = N_PARAMS  # 16
# OBS_DIM is computed dynamically from the env (depends on obs_mode)

STAGE_WEIGHT_CONFIGS = {
    "finale":  None,
    "uniform": [1.0] * 11,
}


# ── Training loops ────────────────────────────────────────────────────────────

def train_rl(algo: str, surrogate, dataset, n_steps, action_scale, max_ep_steps,
             hidden, out_dir: Path, obs_mode: str = "full") -> float:
    """Allena un algoritmo model-free nell'ambiente surrogate."""
    # Create env first so obs_dim is known before building the agent
    env = SurrogateEnv(model=surrogate, dataset=dataset,
                  action_scale=action_scale, max_steps=max_ep_steps,
                  obs_mode=obs_mode)
    obs_dim = env.observation_space.shape[0]

    act_bds = action_bounds(action_scale)
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

    on_policy = algo in ("ppo", "a2c", "trpo", "reinforce")

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

        if on_policy:
            if done:
                last_val = 0.0
                if not terminated:
                    _, _, last_val = agent.select_action(obs, training=True)
                agent.optimize(last_value=float(last_val))
        else:
            agent.optimize()

        obs        = next_obs
        ep_reward += reward

        if done:
            ep_count += 1
            sc = info.get("score", 0.0)
            best_score = max(best_score, sc)
            if step % log_every == 0:
                print(f"  step={step:>7d}  ep={ep_count:>4d}  "
                      f"reward={ep_reward:.2f}  score={sc:.3f}  best={best_score:.3f}")
            obs, _ = env.reset()
            ep_reward = 0.0

    out_dir.mkdir(parents=True, exist_ok=True)
    agent.save(str(out_dir / f"{algo}_agent.pt"))
    print(f"  Salvato → {out_dir / f'{algo}_agent.pt'}")
    return best_score


def train_sb3_sac(surrogate, dataset, n_steps, action_scale, max_ep_steps,
                  hidden, out_dir: Path, obs_mode: str = "full") -> float:
    """Allena SAC di Stable Baselines 3 sull'ambiente surrogate."""
    from beam_optimization.algorithms.model_free.sb3_sac import SB3SAC

    env = SurrogateEnv(model=surrogate, dataset=dataset,
                  action_scale=action_scale, max_steps=max_ep_steps,
                  obs_mode=obs_mode)

    agent = SB3SAC(env, hidden_dims=tuple(hidden))
    best_score = agent.train(env, n_steps=n_steps, log_every=max(1, n_steps // 20))

    out_dir.mkdir(parents=True, exist_ok=True)
    agent.save(str(out_dir / "sb3_sac_agent"))
    print(f"  Salvato → {out_dir / 'sb3_sac_agent.zip'}")
    return best_score


def train_svg(surrogate, dataset, n_episodes, horizon, action_scale, hidden,
              stage_weights, out_dir: Path, obs_mode: str = "full") -> float:
    from beam_optimization.algorithms.model_based.svg import SVGAgent, OBS_DIM

    if obs_mode == "full":
        obs_dim = OBS_DIM
    elif obs_mode == "final":
        obs_dim = BEAM_STATE_DIM
    else:  # "final_with_beam0"
        obs_dim = 2 * BEAM_STATE_DIM

    agent = SVGAgent(
        surrogate=surrogate,
        dataset=dataset,
        obs_dim=obs_dim,
        action_scale=action_scale,
        hidden_dims=tuple(hidden),
        H=horizon,
        stage_weights=stage_weights,
        obs_mode=obs_mode,
    )

    best_score = -np.inf
    log_every  = max(1, n_episodes // 20)

    for ep in range(1, n_episodes + 1):
        result     = agent.optimize_episode()
        best_score = max(best_score, result.final_score)
        if ep % log_every == 0:
            print(f"  ep={ep:>5d}  loss={result.episode_loss:.4f}  "
                  f"score={result.final_score:.3f}  best={best_score:.3f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    agent.save(str(out_dir / "svg_agent.pt"))
    print(f"  Salvato → {out_dir / 'svg_agent.pt'}")
    return best_score


def train_dyna(surrogate, dataset, n_steps, action_scale, max_ep_steps,
               rollout_length, hidden, out_dir: Path,
               tracewin_project: Optional[str] = None,
               online_finetune: bool = False,
               online_mix_ratio: float = 0.5,
               update_dataset_path: Optional[str] = None,
               obs_mode: str = "full",
               surrogate_path: Optional[Path] = None,
               dataset_path: Optional[Path] = None,
               update_surrogates_path: Optional[str] = None) -> float:
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
        nello stesso file caricato per beam0, quindi il dataset base cresce run
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
            action_scale=action_scale,
            max_steps=max_ep_steps,
            obs_mode=obs_mode,
        )
        label = "MBPOWithModelUpdate" if use_model_update else "MBPO"
        print(f"  Env reale: TraceWin  ({tracewin_project})  [{label}]")
    else:
        env = SurrogateEnv(model=surrogate, dataset=dataset,
                      action_scale=action_scale, max_steps=max_ep_steps,
                      obs_mode=obs_mode)
        print("  Env reale: surrogato (SurrogateEnv)  [MBPO]")

    obs_dim = env.observation_space.shape[0]
    act_bds = action_bounds(action_scale)
    bounds  = (act_bds[0].tolist(), act_bds[1].tolist())
    inner   = SAC(obs_dim, ACT_DIM, bounds, hidden_dims=tuple(hidden))

    mbpo_kwargs = dict(
        agent=inner,
        surrogates=surrogate,
        dataset=dataset,
        obs_dim=obs_dim,
        act_dim=ACT_DIM,
        rollout_length=rollout_length,
        obs_mode=obs_mode,
    )
    if use_model_update:
        mbpo_kwargs["online_mix_ratio"] = online_mix_ratio

        if update_dataset_path is not None:
            dataset_save_path = Path(update_dataset_path)
        elif dataset_path is not None:
            # By default MBPOWithModelUpdate writes the merged offline+online
            # dataset back to the dataset used for beam0 sampling.
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

    for step in range(1, n_steps + 1):
        action   = mbpo.select_action(obs, training=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done     = terminated or truncated

        if use_model_update:
            mbpo.step(obs, action, reward, next_obs, done,
                      sim_result=info.get("sim_result"))
        else:
            mbpo.step(obs, action, reward, next_obs, done)

        obs       = next_obs
        ep_reward += reward

        if done:
            ep_count  += 1
            sc         = info.get("score", 0.0)
            best_score = max(best_score, sc)
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
                        help="Singolo .pt, oppure cartella con surrogate_*.pt. "
                             "Default: models/updated se contiene checkpoint, "
                             f"altrimenti models/base. Risolto ora a: {DEFAULT_SURROGATE_DIR}")
    parser.add_argument("--dataset",        default=str(DEFAULT_DATASET))
    parser.add_argument("--output",         default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--rl-steps",       type=int, default=200_000)
    parser.add_argument("--svg-episodes",   type=int, default=1000)
    parser.add_argument("--svg-horizon",    type=int, default=20)
    parser.add_argument("--rollout-length", type=int, default=1,
                        help="Rollout length DynaMBPO (1=Dyna, >1=MBPO)")
    parser.add_argument("--action-scale",   type=float, default=1.0)
    parser.add_argument("--max-ep-steps",   type=int, default=20)
    parser.add_argument("--hidden",         type=int, nargs="+", default=[256, 256])
    parser.add_argument("--quick",          action="store_true",
                        help="Budget ridotto per test rapido")
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
                             "path di --dataset, quindi il dataset base cresce run dopo "
                             "run. Richiede --online-finetune.")
    parser.add_argument("--update-surrogates", default=None, metavar="PATH",
                        help="Cartella dove salvare i surrogate_*.pt fine-tunati da "
                             "MBPOWithModelUpdate. Default: models/updated; "
                             "models/base resta conservato.")
    parser.add_argument("--obs-mode",        default="full",
                        choices=["full", "final", "final_with_beam0"],
                        help="Stato RL: 'full'=108 dim (tutti gli stadi), "
                             "'final'=9 dim (solo fascio finale), "
                             "'final_with_beam0'=18 dim (beam0 + fascio finale).")
    args = parser.parse_args()

    if args.quick:
        args.rl_steps     = 200
        args.svg_episodes = 5

    out_root = Path(args.output)
    skip     = set(args.skip)

    # Load surrogate(s): folder → ensemble, single file → list of one
    surrogate_path = Path(args.surrogate) if args.surrogate else DEFAULT_SURROGATE_DIR
    if surrogate_path.is_dir():
        model_files = sorted(surrogate_path.glob("surrogate_*.pt"))
        if not model_files:
            raise FileNotFoundError(f"Nessun surrogate_*.pt trovato in {surrogate_path}")
        surrogate = [ModularMLP.load(str(p)) for p in model_files]
        for m in surrogate:
            m.eval()
        print(f"Caricati {len(surrogate)} surrogati dall'ensemble: {surrogate_path}")
    else:
        surrogate = ModularMLP.load(str(surrogate_path))
        surrogate.eval()
        print(f"Caricato surrogate singolo: {surrogate_path}")

    print(f"Carico dataset:   {args.dataset}")
    dataset = SurrogateTrainingDataset.load(args.dataset)

    scores: Dict[str, float] = {}

    # ── Model-free RL ─────────────────────────────────────────────────────────
    for algo in ["sac", "td3", "ppo", "ddpg", "a2c", "reinforce", "trpo"]:
        if algo in skip:
            print(f"\n[SKIP] {algo.upper()}")
            continue
        print(f"\n{'='*50}\nTraining {algo.upper()}  ({args.rl_steps} steps)\n{'='*50}")
        scores[algo] = train_rl(
            algo, surrogate, dataset, args.rl_steps,
            args.action_scale, args.max_ep_steps, args.hidden,
            out_root / algo,
            obs_mode=args.obs_mode,
        )

    # ── SB3-SAC ───────────────────────────────────────────────────────────────
    if "sb3_sac" not in skip:
        print(f"\n{'='*50}\nTraining SB3-SAC  ({args.rl_steps} steps)\n{'='*50}")
        scores["sb3_sac"] = train_sb3_sac(
            surrogate, dataset, args.rl_steps,
            args.action_scale, args.max_ep_steps, args.hidden,
            out_root / "sb3_sac",
            obs_mode=args.obs_mode,
        )

    # ── MBPO ──────────────────────────────────────────────────────────────────
    if "dyna" not in skip:
        tw_label  = "TraceWin" if args.tracewin else "surrogato"
        use_mu    = args.online_finetune and bool(args.tracewin)
        alg_label = "MBPOWithModelUpdate" if use_mu else "MBPO"
        print(f"\n{'='*50}\nTraining {alg_label} [{tw_label}]  ({args.rl_steps} steps, "
              f"rollout={args.rollout_length})\n{'='*50}")
        scores["mbpo"] = train_dyna(
            surrogate, dataset, args.rl_steps, args.action_scale,
            args.max_ep_steps, args.rollout_length, args.hidden,
            out_root / "dyna",
            tracewin_project=args.tracewin,
            online_finetune=args.online_finetune,
            online_mix_ratio=args.online_mix_ratio,
            update_dataset_path=args.update_dataset,
            obs_mode=args.obs_mode,
            surrogate_path=surrogate_path,
            dataset_path=Path(args.dataset),
            update_surrogates_path=args.update_surrogates,
        )

    # ── SVGAgent ─────────────────────────────────────────────────────────────
    for name, weights in STAGE_WEIGHT_CONFIGS.items():
        label = f"svg_{name}"
        if label in skip or "svg" in skip:
            print(f"\n[SKIP] {label}")
            continue
        print(f"\n{'='*50}\nTraining SVGAgent [{name}]  ({args.svg_episodes} episodi)\n{'='*50}")
        scores[label] = train_svg(
            surrogate, dataset, args.svg_episodes, args.svg_horizon,
            args.action_scale, args.hidden, weights,
            out_root / label,
            obs_mode=args.obs_mode,
        )

    print_summary(scores)

    summary_path = out_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\nRiepilogo salvato → {summary_path}")


if __name__ == "__main__":
    main()
