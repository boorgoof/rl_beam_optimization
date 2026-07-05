"""
Quick health check for the beam_optimization project.

Verifies:
  1. Imports from all project paths
  2. Dataset and surrogate ensemble loading
  3. SurrogateEnv: reset + 5 random-action steps
  4. SurrogateEnv: Thompson sampling over the ensemble
  5. SAC / MBPO / SVGAgent instantiation + SurrogateDatasetUpdater bootstrap
  6. TraceWin import + optional local binary/workspace check

Usage:
    python -m beam_optimization check
"""
from __future__ import annotations

import argparse
import sys
import traceback

from beam_optimization.config.paths import (
    DEFAULT_BASE_DATASET,
    DEFAULT_BASE_SURROGATE_DIR,
    PROJECT_ROOT,
)

PASS = "✓"
FAIL = "✗"


class Checker:
    def __init__(self):
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, fn):
        try:
            fn()
            self.results.append((name, True, ""))
            print(f"  {PASS}  {name}")
        except Exception as e:
            self.results.append((name, False, traceback.format_exc()))
            print(f"  {FAIL}  {name}  →  {e}")

    def summary(self) -> int:
        n_pass = sum(1 for _, ok, _ in self.results if ok)
        n_fail = sum(1 for _, ok, _ in self.results if not ok)
        print(f"\n{'='*50}")
        print(f"  {PASS} {n_pass} / {n_pass + n_fail} checks passed")
        if n_fail:
            print(f"  {FAIL} {n_fail} failed:")
            for name, ok, tb in self.results:
                if not ok:
                    print(f"\n  [{name}]")
                    print(tb)
        print(f"{'='*50}\n")
        return 0 if n_fail == 0 else 1


def main() -> int:
    argparse.ArgumentParser(
        description="Run quick project checks without launching the real TraceWin."
    ).parse_args()

    c = Checker()
    state: dict = {}

    # ── 1. Imports ────────────────────────────────────────────────────────────
    print("\n[1/6] Imports")

    c.check("SurrogateEnv, ModularMLP, BeamDataset from env.surrogate_env",
            lambda: __import__("beam_optimization.env.surrogate_env",
                               fromlist=["SurrogateEnv", "ModularMLP", "BeamDataset"]))
    c.check("TraceWinEnv, TraceWinSimulator, BeamSimulationResult from env.tracewin_env",
            lambda: __import__("beam_optimization.env.tracewin_env",
                               fromlist=["TraceWinEnv", "TraceWinSimulator", "BeamSimulationResult"]))
    c.check("BeamSimulationResult, BeamSimulator (common)",
            lambda: __import__("beam_optimization.env.simulation",
                               fromlist=["BeamSimulationResult", "BeamSimulator"]))
    c.check("algorithm registry (make_agent)",
            lambda: __import__("beam_optimization.algorithms", fromlist=["make_agent"]))
    c.check("MBPO", lambda: __import__("beam_optimization.algorithms.model_based.mbpo",
                                       fromlist=["MBPO"]))
    c.check("SVGAgent", lambda: __import__("beam_optimization.algorithms.model_based.svg",
                                           fromlist=["SVGAgent"]))
    c.check("config adige", lambda: __import__("beam_optimization.config.adige",
                                               fromlist=["PARAM_KEYS", "BEAM_STATE_FEATURES"]))

    # ── 2. Dataset and surrogates ─────────────────────────────────────────────
    print("\n[2/6] Dataset and surrogates")

    def _load_dataset():
        from beam_optimization.env.dataset import BeamDataset
        state["ds"] = BeamDataset.load(str(DEFAULT_BASE_DATASET))
        assert len(state["ds"]) > 0, "Empty dataset"

    c.check("Load dataset_base.pt", _load_dataset)

    def _load_ensemble():
        from beam_optimization.env.surrogate_env import ModularMLP
        files = sorted(DEFAULT_BASE_SURROGATE_DIR.glob("surrogate_*.pt"))
        assert len(files) > 0, f"No surrogate_*.pt in {DEFAULT_BASE_SURROGATE_DIR}"
        state["surrogates"] = [ModularMLP.load(str(f)) for f in files]
        print(f"       loaded {len(files)} surrogates from {DEFAULT_BASE_SURROGATE_DIR}")

    c.check("Load surrogate ensemble", _load_ensemble)

    # ── 3. SurrogateEnv reset + step ──────────────────────────────────────────
    print("\n[3/6] SurrogateEnv — reset + 5 steps")

    def _beam_env_single():
        from beam_optimization.env.surrogate_env import SurrogateEnv
        from beam_optimization.config.adige import N_STAGES, BEAM_STATE_DIM, observation_dim
        obs_dim = observation_dim()
        env = SurrogateEnv(model=state["surrogates"][0], dataset=state["ds"], max_steps=5)
        obs, info = env.reset()
        assert obs.shape == (obs_dim,), f"obs shape wrong: {obs.shape}"
        assert "score" in info
        assert info["sim_result"].source == "surrogate"
        assert info["sim_result"].beam_states.shape == (N_STAGES, BEAM_STATE_DIM)
        for _ in range(5):
            action = env.action_space.sample()
            obs, rew, term, trunc, info = env.step(action)
            assert obs.shape == (obs_dim,)
            assert info["sim_result"].source == "surrogate"
        print(f"       best_score={env.best_score:.4f}")

    c.check("SurrogateEnv with one surrogate (observation mask)", _beam_env_single)

    # ── 4. Thompson sampling over the ensemble ────────────────────────────────
    print("\n[4/6] SurrogateEnv — ensemble Thompson sampling")

    def _thompson():
        from beam_optimization.env.surrogate_env import SurrogateEnv
        env = SurrogateEnv(model=state["surrogates"], dataset=state["ds"], max_steps=3)
        assert len(env.simulator.ensemble) == len(state["surrogates"])
        seen = set()
        for _ in range(30):
            env.reset()
            seen.add(id(env.simulator.model))
        if len(state["surrogates"]) > 1:
            assert len(seen) > 1, "Thompson sampling does not diversify the surrogate"
        print(f"       {len(seen)} distinct surrogates over 30 resets")

    c.check("Ensemble Thompson sampling active", _thompson)

    # ── 5. RL agents ──────────────────────────────────────────────────────────
    print("\n[5/6] RL agents")

    def _bounds():
        from beam_optimization.config.adige import action_bounds
        low, high = action_bounds()
        return (low.tolist(), high.tolist())

    def _sac():
        from beam_optimization.algorithms import make_agent
        from beam_optimization.config.adige import N_PARAMS, observation_dim
        agent = make_agent("sac", observation_dim(), N_PARAMS, _bounds())
        assert agent is not None

    c.check("SAC via registry", _sac)

    def _mbpo():
        from beam_optimization.algorithms import make_agent
        from beam_optimization.algorithms.model_based.mbpo import MBPO
        from beam_optimization.config.adige import N_PARAMS, observation_dim
        obs_dim = observation_dim()
        inner = make_agent("sac", obs_dim, N_PARAMS, _bounds())
        agent = MBPO(agent=inner, surrogates=state["surrogates"], dataset=state["ds"],
                     obs_dim=obs_dim, act_dim=N_PARAMS)
        assert agent is not None

    c.check("MBPO with ensemble", _mbpo)

    def _svg():
        from beam_optimization.algorithms.model_based.svg import SVGAgent
        from beam_optimization.config.adige import (
            N_PARAMS, PARAM_KEYS, default_params, observation_dim,
        )
        agent = SVGAgent(
            surrogate=state["surrogates"],
            dataset=state["ds"],
            obs_dim=observation_dim(),
            act_dim=N_PARAMS,
            action_bounds=_bounds(),
            param_keys=PARAM_KEYS,
            default_params=default_params(),
            n_step=3,
        )
        assert len(agent.env.simulator.ensemble) == len(state["surrogates"])
        result = agent.optimize_episode()
        assert result.final_score != 0.0
        print(f"       SVG episode score={result.final_score:.4f}")

    c.check("SVGAgent with ensemble (1 episode)", _svg)

    print("\n[5b] SurrogateDatasetUpdater — bootstrap ensemble fine-tuning")

    def _surrogate_updater():
        import numpy as np
        from beam_optimization.env.surrogate_env.surrogate.model.updater import (
            SurrogateDatasetUpdater,
        )
        from beam_optimization.env.simulation import BeamSimulationResult
        from beam_optimization.config.adige import BEAM_STATE_FEATURES, default_params, score

        updater = SurrogateDatasetUpdater(
            state["surrogates"], min_samples=5, batch_size=8, epochs=3
        )
        assert updater.n_online_samples == 0

        for i in range(8):
            _, beam_states = state["ds"].get_training_batch([i])
            fake_bs = np.stack(
                [stage.squeeze(0).numpy() for stage in beam_states],
                axis=0,
            ).astype(np.float32)
            fake_final = {v: float(fake_bs[-1][vi])
                          for vi, v in enumerate(BEAM_STATE_FEATURES)}
            res = BeamSimulationResult(
                params=default_params(),
                beam_states=fake_bs,
                final_beam=fake_final,
                score_val=score(fake_final),
                success=True,
                source="tracewin",
            )
            updater.add_tracewin_result(res)

        assert updater.n_online_samples == 8
        losses = updater.update()
        assert losses is not None
        assert len(losses) == len(state["surrogates"])
        print(f"       update OK, losses: {losses}")

    c.check("SurrogateDatasetUpdater: add + bootstrap update", _surrogate_updater)

    # ── 6. TraceWin import ────────────────────────────────────────────────────
    print("\n[6/6] TraceWin (import + local setup only, no execution)")

    c.check("Import TraceWinSimulator",
            lambda: __import__("beam_optimization.env.tracewin_env.tracewin.tracewin_simulator",
                               fromlist=["TraceWinSimulator"]))
    c.check("Import TraceWinEnv",
            lambda: __import__("beam_optimization.env.tracewin_env",
                               fromlist=["TraceWinEnv"]))

    def _tw_local_setup():
        wrapper_dir = PROJECT_ROOT / "env/tracewin_env/tracewin/pyTraceWin_wrapper"
        tw_dir = wrapper_dir / "TraceWin_program"
        tw_bin = tw_dir / "TraceWin"
        launcher = wrapper_dir / "run_tracewin_with_permissions.sh"
        workspace = PROJECT_ROOT / "env/tracewin_env/tracewin/TraceWin_workspace"
        missing = [p for p in (tw_bin, launcher, workspace) if not p.exists()]
        if missing:
            print("       local TraceWin setup incomplete; skipping binary/workspace check:")
            for path in missing:
                print(f"         - missing: {path}")
            print("       see README.md to create these unversioned local files.")
            return

        license_logs = [tw_dir / "tracewin_key.log", tw_dir / "toutatis_key.log"]
        missing_logs = [p for p in license_logs if not p.exists()]
        if missing_logs:
            print("       binary/workspace OK; license/log files not found:")
            for path in missing_logs:
                print(f"         - optional/missing: {path}")
        else:
            print(f"       local TraceWin setup OK, {len(license_logs)} license/log files found")

    c.check("Optional local TraceWin setup", _tw_local_setup)

    return c.summary()


if __name__ == "__main__":
    sys.exit(main())
