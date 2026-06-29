"""
Benchmark — confronta i metodi di ottimizzazione sul surrogate.

Tutti i metodi ricevono lo stesso surrogate, lo stesso beam0 iniziale
e lo stesso budget di valutazioni.

Metodi:
  pso               Particle Swarm Optimization
  bayesian_opt      Bayesian Optimization (GP)
  svg_finale        SVGAgent — reward solo stage finale
  svg_uniform       SVGAgent — reward uniforme su tutti gli stage
  sac / td3 / ppo   Model-free RL (da checkpoint --sac/--td3/--ppo)

Uso:
    python -m beam_optimization benchmark \\
        --surrogate env/surrogate_env/surrogate/models/updated/surrogate_0.pt \\
        --dataset   env/dataset/base/dataset_train.pt \\
        --output    results/benchmark.json \\
        --n-runs    3 \\
        --eval-budget 3000 \\
        --svg-episodes 500

Con agenti model-free già allenati:
    python -m beam_optimization benchmark \\
        --sac runs/all/sac/sac_agent.pt \\
        --td3 runs/all/td3/td3_agent.pt \\
        --ppo runs/all/ppo/ppo_agent.pt

Test rapido:
    python -m beam_optimization benchmark --quick
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from beam_optimization.config.paths import (
    DEFAULT_BENCHMARK_OUTPUT,
    DEFAULT_DATASET,
    DEFAULT_SURROGATE_MODEL,
)
from beam_optimization.env.surrogate_env.surrogate.modular_mlp import ModularMLP
from beam_optimization.env.dataset import SurrogateTrainingDataset
from beam_optimization.env.surrogate_env import SurrogateEnv
from beam_optimization.config.adige import (
    N_PARAMS, N_STAGES, BEAM_STATE_DIM,
    action_bounds, params_to_stage_tensors, BEAM_STATE_FEATURES, score,
)

OBS_DIM = N_STAGES * BEAM_STATE_DIM  # 108
ACT_DIM = N_PARAMS                               # 16

STAGE_WEIGHT_CONFIGS = {
    "finale":  None,
    "uniform": [1.0] * 11,
}


# ── Benchmark functions ───────────────────────────────────────────────────────

def run_pso(surrogate, dataset, budget, seed) -> Dict:
    from beam_optimization.algorithms.baselines.pso import PSOOptimizer
    beam0 = _pick_beam(dataset, seed)
    surrogate.eval()

    def objective(params):
        with torch.no_grad():
            outs = surrogate(params_to_stage_tensors(params), beam0)
            return score({v: float(outs[-1][0, i]) for i, v in enumerate(BEAM_STATE_FEATURES)})

    n_particles  = 30
    n_iterations = max(1, budget // n_particles - 1)
    result = PSOOptimizer(n_particles=n_particles, n_iterations=n_iterations, seed=seed).optimize(objective)
    return {"best_score": result.best_score, "history": result.score_history}


def run_bo(surrogate, dataset, budget, seed) -> Dict:
    from beam_optimization.algorithms.baselines.bayesian_opt import BayesianOptimizer
    beam0 = _pick_beam(dataset, seed)
    surrogate.eval()

    def objective(params):
        with torch.no_grad():
            outs = surrogate(params_to_stage_tensors(params), beam0)
            return score({v: float(outs[-1][0, i]) for i, v in enumerate(BEAM_STATE_FEATURES)})

    result = BayesianOptimizer(n_calls=min(budget, 200), seed=seed).optimize(objective)
    return {"best_score": result.best_score, "history": result.score_history}


def run_svg(surrogate, dataset, n_episodes, horizon, seed, stage_weights) -> Dict:
    from beam_optimization.algorithms.model_based.svg import SVGAgent
    import random
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    agent = SVGAgent(surrogate=surrogate, dataset=dataset, H=horizon,
                     stage_weights=stage_weights)
    history = []
    for ep in range(n_episodes):
        result = agent.optimize_episode()
        history.append(result.final_score)
        if (ep + 1) % max(1, n_episodes // 5) == 0:
            print(f"    ep {ep+1}/{n_episodes}  score={result.final_score:.3f}")

    return {"best_score": float(max(history)), "history": history}


def eval_model_free(algo: str, ckpt_path: str, surrogate, dataset, n_eval=20) -> Dict:
    act_bds = action_bounds(1.0)
    bounds  = (act_bds[0].tolist(), act_bds[1].tolist())

    if algo == "sac":
        from beam_optimization.algorithms.model_free.sac import SAC
        agent = SAC(OBS_DIM, ACT_DIM, bounds)
    elif algo == "td3":
        from beam_optimization.algorithms.model_free.td3 import TD3
        agent = TD3(OBS_DIM, ACT_DIM, bounds)
    elif algo == "ppo":
        from beam_optimization.algorithms.model_free.ppo import PPO
        agent = PPO(OBS_DIM, ACT_DIM, bounds)
    else:
        raise ValueError(algo)

    agent.load(ckpt_path)
    env = SurrogateEnv(model=surrogate, dataset=dataset, action_scale=1.0, max_steps=20)

    scores = []
    for _ in range(n_eval):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs, training=False)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        scores.append(info["score"])

    return {"best_score": float(max(scores)), "mean_score": float(np.mean(scores)),
            "history": scores}


# ── Utility ───────────────────────────────────────────────────────────────────

def _pick_beam(dataset, seed) -> torch.Tensor:
    idx = int(np.random.default_rng(seed).integers(0, len(dataset.get_initial_beam_states())))
    return dataset.get_initial_beam_states()[idx:idx+1].float()


def print_table(results: Dict):
    print(f"\n{'='*65}")
    print("BENCHMARK SUMMARY")
    print(f"{'Method':<35}  {'Mean':>8}  {'Std':>7}  {'Best':>8}")
    print("-" * 65)
    for method, runs in sorted(results.items()):
        bests = [r["best_score"] for r in runs]
        print(f"{method:<35}  {np.mean(bests):>8.3f}  {np.std(bests):>7.3f}  {np.max(bests):>8.3f}")
    print(f"{'='*65}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surrogate",    default=str(DEFAULT_SURROGATE_MODEL))
    parser.add_argument("--dataset",      default=str(DEFAULT_DATASET))
    parser.add_argument("--output",       default=str(DEFAULT_BENCHMARK_OUTPUT))
    parser.add_argument("--n-runs",       type=int, default=3)
    parser.add_argument("--eval-budget",  type=int, default=3000)
    parser.add_argument("--svg-episodes", type=int, default=500)
    parser.add_argument("--svg-horizon",  type=int, default=20)
    parser.add_argument("--quick",        action="store_true")
    parser.add_argument("--sac",          default=None, metavar="CKPT")
    parser.add_argument("--td3",          default=None, metavar="CKPT")
    parser.add_argument("--ppo",          default=None, metavar="CKPT")
    args = parser.parse_args()

    if args.quick:
        args.eval_budget  = 30
        args.svg_episodes = 1
        args.n_runs       = 1

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Surrogate: {args.surrogate}")
    surrogate = ModularMLP.load(args.surrogate)
    surrogate.eval()

    print(f"Dataset:   {args.dataset}")
    dataset = SurrogateTrainingDataset.load(args.dataset)

    results: Dict = {}

    for run in range(args.n_runs):
        seed = 42 + run
        print(f"\n{'='*65}\nRun {run+1}/{args.n_runs}  (seed={seed})\n{'='*65}")

        print("PSO...")
        r = run_pso(surrogate, dataset, args.eval_budget, seed)
        results.setdefault("pso", []).append(r)
        print(f"  best={r['best_score']:.3f}")

        print("Bayesian Optimization...")
        r = run_bo(surrogate, dataset, args.eval_budget, seed)
        results.setdefault("bayesian_opt", []).append(r)
        print(f"  best={r['best_score']:.3f}")

        for name, weights in STAGE_WEIGHT_CONFIGS.items():
            label = f"svg_{name}"
            print(f"SVGAgent [{name}]...")
            r = run_svg(surrogate, dataset, args.svg_episodes, args.svg_horizon,
                        seed, weights)
            results.setdefault(label, []).append(r)
            print(f"  best={r['best_score']:.3f}")

        for algo, ckpt in [("sac", args.sac), ("td3", args.td3), ("ppo", args.ppo)]:
            if ckpt and Path(ckpt).exists():
                print(f"{algo.upper()} (checkpoint)...")
                r = eval_model_free(algo, ckpt, surrogate, dataset)
                results.setdefault(algo, []).append(r)
                print(f"  best={r['best_score']:.3f}")
            elif ckpt:
                print(f"  WARN: {ckpt} non trovato, salto {algo.upper()}")

    print_table(results)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRisultati salvati → {args.output}")


if __name__ == "__main__":
    main()
