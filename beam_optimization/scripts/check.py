"""
Test rapido del progetto beam_optimization.

Verifica:
  1. Import da tutti i path del progetto
  2. Caricamento dataset e surrogati (ensemble)
  3. SurrogateEnv: reset + 5 step con azione casuale
  4. SurrogateEnv: Thompson sampling (ensemble)
  5. Istanziazione SAC, MBPO, SVGAgent
  6. Import TraceWinSimulator e TraceWinEnv

Uso:
    python -m beam_optimization check
"""
from __future__ import annotations

import argparse
import sys
import traceback

from beam_optimization.config.paths import DEFAULT_DATASET, DEFAULT_SURROGATE_DIR, PROJECT_ROOT

argparse.ArgumentParser(
    description="Esegue i controlli rapidi del progetto senza lanciare TraceWin reale."
).parse_args()

PASS = "✓"
FAIL = "✗"

results: list[tuple[str, bool, str]] = []


def check(name: str, fn):
    try:
        fn()
        results.append((name, True, ""))
        print(f"  {PASS}  {name}")
    except Exception as e:
        results.append((name, False, traceback.format_exc()))
        print(f"  {FAIL}  {name}  →  {e}")


# ── 1. Import ─────────────────────────────────────────────────────────────────

print("\n[1/6] Import")

check("SurrogateEnv, ModularMLP, BeamDataset da env.surrogate_env",
      lambda: __import__("beam_optimization.env.surrogate_env",
                          fromlist=["SurrogateEnv", "ModularMLP", "SurrogateTrainingDataset", "BeamDataset"]))

check("TraceWinEnv, TraceWinSimulator, BeamSimulationResult da env.tracewin_env",
      lambda: __import__("beam_optimization.env.tracewin_env",
                          fromlist=["TraceWinEnv", "TraceWinSimulator", "BeamSimulationResult", "SimResult"]))

check("import breve: from beam_optimization.env import ...", lambda: (
    __import__("beam_optimization.env",
               fromlist=["SurrogateEnv", "ModularMLP", "BeamDataset",
                         "TraceWinEnv", "TraceWinSimulator", "BeamSimulationResult"])
))

check("BeamSimulationResult, BeamSimulator comuni", lambda: (
    __import__("beam_optimization.env.simulation",
               fromlist=["BeamSimulationResult", "BeamSimulator"])
))

check("MBPO", lambda: __import__("beam_optimization.algorithms.model_based.mbpo",
                                  fromlist=["MBPO"]))

check("SVGAgent", lambda: __import__("beam_optimization.algorithms.model_based.svg",
                                      fromlist=["SVGAgent"]))

check("SAC", lambda: __import__("beam_optimization.algorithms.model_free.sac",
                                 fromlist=["SAC"]))

check("config adige", lambda: __import__("beam_optimization.config.adige",
                                          fromlist=["PARAM_KEYS", "BEAM_STATE_FEATURES"]))

# ── 2. Dataset e surrogati ────────────────────────────────────────────────────

print("\n[2/6] Dataset e surrogati")

HERE = PROJECT_ROOT

def _load_dataset():
    from beam_optimization.env.surrogate_env import BeamDataset
    global ds
    ds = BeamDataset.load(str(DEFAULT_DATASET))
    assert len(ds) > 0, "Dataset vuoto"

check("Carica dataset_train.pt", _load_dataset)

def _load_ensemble():
    from beam_optimization.env.surrogate_env import ModularMLP
    global surrogates
    model_dir = DEFAULT_SURROGATE_DIR
    files = sorted(model_dir.glob("surrogate_*.pt"))
    assert len(files) > 0, f"Nessun .pt in {model_dir}"
    surrogates = [ModularMLP.load(str(f)) for f in files]
    assert len(surrogates) == 4
    print(f"       caricati {len(surrogates)} surrogati da {model_dir}")

check("Carica ensemble 4 surrogati", _load_ensemble)

# ── 3. SurrogateEnv reset + step ───────────────────────────────────────────────────

print("\n[3/6] SurrogateEnv — reset + 5 step")

def _beam_env_single():
    from beam_optimization.env.surrogate_env import SurrogateEnv
    env = SurrogateEnv(model=surrogates[0], dataset=ds, max_steps=5, obs_mode="full")
    obs, info = env.reset()
    assert obs.shape == (108,), f"obs shape wrong: {obs.shape}"
    assert "score" in info
    assert info["sim_result"].source == "surrogate"
    assert info["sim_result"].beam_states.shape == (12, 9)
    for _ in range(5):
        action = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(action)
        assert obs.shape == (108,)
        assert info["sim_result"].source == "surrogate"
        assert info["sim_result"].beam_states.shape == (12, 9)
    print(f"       best_score={env.best_score:.4f}")

check("SurrogateEnv singolo surrogate (obs_mode=full)", _beam_env_single)

def _beam_env_final():
    from beam_optimization.env.surrogate_env import SurrogateEnv
    env = SurrogateEnv(model=surrogates[0], dataset=ds, max_steps=3, obs_mode="final")
    obs, _ = env.reset()
    assert obs.shape == (9,), f"obs shape wrong: {obs.shape}"

check("SurrogateEnv obs_mode=final (9 dim)", _beam_env_final)

def _beam_env_beam0():
    from beam_optimization.env.surrogate_env import SurrogateEnv
    env = SurrogateEnv(model=surrogates[0], dataset=ds, max_steps=3, obs_mode="final_with_beam0")
    obs, _ = env.reset()
    assert obs.shape == (18,), f"obs shape wrong: {obs.shape}"

check("SurrogateEnv obs_mode=final_with_beam0 (18 dim)", _beam_env_beam0)

# ── 4. Thompson sampling ensemble ────────────────────────────────────────────

print("\n[4/6] SurrogateEnv — ensemble Thompson sampling")

def _thompson():
    from beam_optimization.env.surrogate_env import SurrogateEnv
    env = SurrogateEnv(model=surrogates, dataset=ds, max_steps=3)
    assert len(env.simulator.ensemble) == 4
    seen = set()
    for _ in range(30):
        env.reset()
        seen.add(id(env.simulator.model))
    assert len(seen) > 1, "Thompson sampling non diversifica il surrogate"
    print(f"       {len(seen)} surrogati diversi su 30 reset")

check("Ensemble 4 surrogati, Thompson sampling attivo", _thompson)

# ── 5. Agenti RL ─────────────────────────────────────────────────────────────

print("\n[5/6] Agenti RL")

def _sac():
    from beam_optimization.algorithms.model_free.sac import SAC
    from beam_optimization.config.adige import action_bounds
    bounds = action_bounds(1.0)
    agent = SAC(108, 16, (bounds[0].tolist(), bounds[1].tolist()))
    assert agent is not None

check("SAC istanziabile", _sac)

def _mbpo():
    from beam_optimization.algorithms.model_based.mbpo import MBPO
    from beam_optimization.algorithms.model_free.sac import SAC
    from beam_optimization.config.adige import action_bounds
    bounds = action_bounds(1.0)
    inner = SAC(108, 16, (bounds[0].tolist(), bounds[1].tolist()))
    agent = MBPO(agent=inner, surrogates=surrogates, dataset=ds,
                 obs_dim=108, act_dim=16)
    assert agent is not None

check("MBPO con ensemble", _mbpo)

def _svg():
    from beam_optimization.algorithms.model_based.svg import SVGAgent
    agent = SVGAgent(surrogate=surrogates, dataset=ds, obs_dim=108, H=3)
    assert len(agent._ensemble) == 4
    result = agent.optimize_episode()
    assert result.final_score != 0.0
    print(f"       SVG episode score={result.final_score:.4f}")

check("SVGAgent con ensemble (1 episodio)", _svg)

# ── 5b. SurrogateDatasetUpdater ──────────────────────────────────────────────

print("\n[5b] SurrogateDatasetUpdater — bootstrap fine-tuning ensemble")

def _surrogate_updater():
    from beam_optimization.env.surrogate_env.surrogate.updater import SurrogateDatasetUpdater
    from beam_optimization.env.tracewin_env.tracewin.tracewin_simulator import SimResult
    from beam_optimization.config.adige import default_params, score
    import numpy as np

    updater = SurrogateDatasetUpdater(surrogates, min_samples=5, batch_size=8, epochs=3)
    assert updater.n_samples == 0

    for i in range(8):
        _, beam_states = ds.get_training_batch([i])
        fake_bs = np.stack(
            [stage.squeeze(0).numpy() for stage in beam_states],
            axis=0,
        ).astype(np.float32)
        fake_final = {v: float(fake_bs[-1][vi])
                      for vi, v in enumerate(__import__(
                          "beam_optimization.config.adige",
                          fromlist=["BEAM_STATE_FEATURES"]).BEAM_STATE_FEATURES)}
        res = SimResult(
            params=default_params(),
            beam_states=fake_bs,
            final_beam=fake_final,
            score_val=score(fake_final),
            success=True,
            source="tracewin",
        )
        updater.add(res)

    assert updater.n_samples == 8
    losses = updater.update()
    assert losses is not None
    assert len(losses) == 4
    print(f"       update OK, losses: {losses}")

check("SurrogateDatasetUpdater: add + bootstrap update su 4 surrogati", _surrogate_updater)

# ── 6. TraceWin import ────────────────────────────────────────────────────────

print("\n[6/6] TraceWin (solo import + istanziazione, senza eseguire)")

check("Import TraceWinSimulator",
      lambda: __import__("beam_optimization.env.tracewin_env.tracewin.tracewin_simulator",
                          fromlist=["TraceWinSimulator", "SimResult"]))

check("Import TraceWinEnv",
      lambda: __import__("beam_optimization.env.tracewin_env",
                          fromlist=["TraceWinEnv"]))

def _tw_local_setup():
    wrapper_dir = HERE / "env/tracewin_env/tracewin/pyTraceWin_wrapper"
    tw_dir = HERE / "env/tracewin_env/tracewin/pyTraceWin_wrapper/TraceWin_program"
    tw_bin = tw_dir / "TraceWin"
    launcher = wrapper_dir / "run_tracewin_with_permissions.sh"
    workspace = HERE / "env/tracewin_env/tracewin/TraceWin_workspace"
    required = [tw_bin, launcher, workspace]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("       setup TraceWin locale non completo; salto controllo binario/workspace:")
        for path in missing:
            print(f"         - mancante: {path}")
        print("       consulta README.md per creare questi file locali non versionati.")
        return

    license_logs = [tw_dir / "tracewin_key.log", tw_dir / "toutatis_key.log"]
    missing_logs = [p for p in license_logs if not p.exists()]
    if missing_logs:
        print("       binario/workspace OK; file licenza/log non trovati:")
        for path in missing_logs:
            print(f"         - opzionale/mancante: {path}")
    else:
        print(f"       setup TraceWin locale OK, {len(license_logs)} file licenza/log trovati")

check("Setup TraceWin locale opzionale", _tw_local_setup)

# ── Riepilogo ─────────────────────────────────────────────────────────────────

n_pass = sum(1 for _, ok, _ in results if ok)
n_fail = sum(1 for _, ok, _ in results if not ok)

print(f"\n{'='*50}")
print(f"  {PASS} {n_pass} / {n_pass + n_fail} test passati")
if n_fail:
    print(f"  {FAIL} {n_fail} falliti:")
    for name, ok, tb in results:
        if not ok:
            print(f"\n  [{name}]")
            print(tb)
print(f"{'='*50}\n")

sys.exit(0 if n_fail == 0 else 1)
