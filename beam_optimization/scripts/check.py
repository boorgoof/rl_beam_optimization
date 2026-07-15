"""Procedural onboarding check for the beam_optimization project.

The command verifies the same prerequisites described in README.md:
Python dependencies, local TraceWin files, a real TraceWinEnv reset/step,
dataset and surrogate artifacts, SurrogateEnv, algorithms, and the online
surrogate updater.

Usage:
    python -m beam_optimization check
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from beam_optimization.config.paths import (
    DEFAULT_BASE_SURROGATE_DIR,
    DEFAULT_DATASET_ROOT,
    DEFAULT_TRACEWIN_INI,
    PROJECT_ROOT,
    configure_matplotlib_cache,
    default_dataset_path,
)


SETUP_COMMAND = (
    "python -m beam_optimization build_dataset --target-samples 100 && "
    "python -m beam_optimization train_surrogate --train-dataset <dataset_train.pt> --n-surrogates 4"
)
INSTALL_COMMAND = (
    "beam_optimization/.venv/bin/pip install -r beam_optimization/requirements.txt"
)


@dataclass
class CheckResult:
    name: str
    status: str
    details: str = ""
    problem: str = ""
    action: str = ""
    path_command: str = ""
    traceback_text: str = ""


@dataclass
class CheckMessage:
    status: str = "PASS"
    details: str = ""
    problem: str = ""
    action: str = ""
    path_command: str = ""


class CheckFailure(RuntimeError):
    def __init__(self, problem: str, action: str = "", path_command: str = ""):
        super().__init__(problem)
        self.problem = problem
        self.action = action
        self.path_command = path_command


class Checker:
    def __init__(self):
        self.results: list[CheckResult] = []

    def check(
        self,
        name: str,
        fn: Callable[[], Optional[str | CheckMessage]],
        *,
        default_action: str = "",
        default_path_command: str = "",
        skip_reason: str | None = None,
        skip_action: str = "",
    ) -> bool:
        if skip_reason is not None:
            result = CheckResult(
                name=name,
                status="SKIP",
                problem=skip_reason,
                action=skip_action,
            )
            self.results.append(result)
            self._print_result(result)
            return False

        try:
            value = fn()
            if isinstance(value, CheckMessage):
                result = CheckResult(
                    name=name,
                    status=value.status,
                    details=value.details,
                    problem=value.problem,
                    action=value.action,
                    path_command=value.path_command,
                )
            else:
                result = CheckResult(
                    name=name,
                    status="PASS",
                    details=value or "",
                )
        except CheckFailure as exc:
            result = CheckResult(
                name=name,
                status="FAIL",
                problem=exc.problem,
                action=exc.action or default_action,
                path_command=exc.path_command or default_path_command,
                traceback_text=traceback.format_exc(),
            )
        except Exception as exc:
            result = CheckResult(
                name=name,
                status="FAIL",
                problem=str(exc),
                action=default_action,
                path_command=default_path_command,
                traceback_text=traceback.format_exc(),
            )

        self.results.append(result)
        self._print_result(result)
        return result.status in {"PASS", "WARN"}

    def _print_result(self, result: CheckResult) -> None:
        print(f"  [{result.status}] {result.name}")
        if result.details:
            print(f"         {result.details}")
        if result.problem:
            print(f"         Problem: {result.problem}")
        if result.action:
            print(f"         Action:  {result.action}")
        if result.path_command:
            print(f"         Path/Command: {result.path_command}")

    def summary(self) -> int:
        counts = {status: 0 for status in ("PASS", "WARN", "FAIL", "SKIP")}
        for result in self.results:
            counts[result.status] = counts.get(result.status, 0) + 1

        print("\n" + "=" * 72)
        print("CHECK SUMMARY")
        print(
            f"  PASS {counts['PASS']}   WARN {counts['WARN']}   "
            f"FAIL {counts['FAIL']}   SKIP {counts['SKIP']}"
        )

        actions = [
            result for result in self.results
            if result.status == "FAIL" and (result.action or result.path_command)
        ]
        if actions:
            print("\nACTIONS")
            for result in actions:
                print(f"  - {result.name}")
                if result.action:
                    print(f"    Action: {result.action}")
                if result.path_command:
                    print(f"    Path/Command: {result.path_command}")

        failures = [result for result in self.results if result.status == "FAIL"]
        if failures:
            print("\nTRACEBACKS")
            for result in failures:
                if result.traceback_text:
                    print(f"\n[{result.name}]")
                    print(result.traceback_text.rstrip())

        print("=" * 72 + "\n")
        return 0 if counts["FAIL"] == 0 else 1


def _require_path(
    path: Path,
    description: str,
    *,
    action: str,
    must_be_file: bool | None = None,
    readable: bool = False,
    executable: bool = False,
) -> None:
    if not path.exists():
        raise CheckFailure(
            f"{description} does not exist: {path}",
            action=action,
            path_command=str(path),
        )
    if must_be_file is True and not path.is_file():
        raise CheckFailure(
            f"{description} is not a file: {path}",
            action=action,
            path_command=str(path),
        )
    if must_be_file is False and not path.is_dir():
        raise CheckFailure(
            f"{description} is not a directory: {path}",
            action=action,
            path_command=str(path),
        )
    if readable and not os.access(path, os.R_OK):
        raise CheckFailure(
            f"{description} is not readable: {path}",
            action=action,
            path_command=f"chmod +r {path}",
        )
    if executable and not os.access(path, os.X_OK):
        raise CheckFailure(
            f"{description} is not executable: {path}",
            action=action,
            path_command=f"chmod +x {path}",
        )


def _assert_obs_shape(obs: np.ndarray, expected_dim: int, context: str) -> None:
    if tuple(obs.shape) != (expected_dim,):
        raise CheckFailure(
            f"{context} observation shape is {tuple(obs.shape)}, expected {(expected_dim,)}.",
            action="Check OBSERVATION_STAGE_MASK and environment construction.",
        )


def _assert_beam_states_shape(beam_states, context: str) -> None:
    from beam_optimization.config.adige import BEAM_STATE_DIM, N_STAGES

    if beam_states is None:
        raise CheckFailure(
            f"{context} did not return beam_states.",
            action="Check the simulator result conversion.",
        )
    if tuple(beam_states.shape) != (N_STAGES, BEAM_STATE_DIM):
        raise CheckFailure(
            f"{context} beam_states shape is {tuple(beam_states.shape)}, "
            f"expected {(N_STAGES, BEAM_STATE_DIM)}.",
            action="Check simulator output parsing and config/adige.py stage settings.",
        )


def _assert_successful_sim_result(result, context: str) -> None:
    if result is None:
        raise CheckFailure(
            f"{context} did not return a simulation result.",
            action="Check environment/simulator result conversion.",
        )
    if not getattr(result, "success", False):
        metadata = getattr(result, "metadata", {}) or {}
        calc_dir = metadata.get("calc_dir", "")
        error = getattr(result, "error", "") or "unknown TraceWin failure"
        raise CheckFailure(
            f"{context} TraceWin simulation failed: {error}",
            action=(
                "TraceWin files exist, but the real simulation did not complete. "
                "Check SSH to comunian@localhost, TraceWin license, launcher permissions, "
                ".ini/.dat/field maps/.dst and TraceWin output files."
            ),
            path_command=str(calc_dir),
        )
    if getattr(result, "source", None) != "tracewin":
        raise CheckFailure(
            f"{context} returned source={getattr(result, 'source', None)!r}, expected 'tracewin'.",
            action="Check TraceWinEnv/TraceWinSimulator result conversion.",
        )


def _extract_action(value) -> np.ndarray:
    if isinstance(value, tuple):
        value = value[0]
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _setup_action() -> str:
    return "Generate the base dataset and base surrogate checkpoints."


def _setup_command() -> str:
    return SETUP_COMMAND


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run procedural onboarding checks: Python dependencies, local "
            "TraceWin files, real TraceWinEnv reset/step, dataset/surrogates, "
            "SurrogateEnv, algorithms, and updater."
        )
    )
    parser.add_argument(
        "--tracewin-calc-dir",
        default="/tmp/tracewin_check",
        metavar="PATH",
        help="TraceWin calculation directory for the real reset+step check.",
    )
    parser.add_argument(
        "--tracewin-timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for each TraceWin call.",
    )
    parser.add_argument(
        "--surrogate-steps",
        type=int,
        default=5,
        help="Random SurrogateEnv steps to run in the surrogate environment check.",
    )
    parser.add_argument(
        "--skip-tracewin",
        action="store_true",
        help=(
            "Skip the TraceWin file and real reset/step checks (sections 3-4). "
            "Use on surrogate-only machines; full onboarding requires TraceWin."
        ),
    )
    args = parser.parse_args()

    checker = Checker()
    state: dict = {}

    print("\n[1/10] Python environment")

    def _python_environment():
        configure_matplotlib_cache()
        packages = ["numpy", "torch", "gymnasium", "matplotlib", "stable_baselines3"]
        missing = []
        for package in packages:
            try:
                importlib.import_module(package)
            except Exception as exc:
                missing.append(f"{package} ({exc})")
        try:
            importlib.import_module("beam_optimization")
        except Exception as exc:
            missing.append(f"beam_optimization ({exc})")
        if missing:
            raise CheckFailure(
                "Missing or broken imports: " + "; ".join(missing),
                action="Install the project requirements in the active environment.",
                path_command=INSTALL_COMMAND,
            )
        return "numpy, torch, gymnasium, matplotlib, stable_baselines3 and package import OK"

    python_ok = checker.check("Python packages and package import", _python_environment)

    print("\n[2/10] Project paths")

    def _project_paths():
        paths = [
            (PROJECT_ROOT, "package root", False),
            (PROJECT_ROOT / "requirements.txt", "requirements.txt", True),
            (PROJECT_ROOT / "config/paths.py", "config/paths.py", True),
            (DEFAULT_DATASET_ROOT, "dataset root", False),
            (DEFAULT_BASE_SURROGATE_DIR.parent, "surrogate trained_models root", False),
            (DEFAULT_TRACEWIN_INI.parent, "TraceWin workspace", False),
        ]
        missing = [f"{label}: {path}" for path, label, is_file in paths
                   if not path.exists() or (is_file and not path.is_file())
                   or (not is_file and not path.is_dir())]
        if missing:
            raise CheckFailure(
                "Missing project paths: " + "; ".join(missing),
                action=(
                    "Create/copy local TraceWin files if a TraceWin path is missing; "
                    "run setup if dataset/surrogate artifact folders are missing."
                ),
                path_command=_setup_command(),
            )
        return "core package, requirements, dataset, surrogate and TraceWin roots exist"

    paths_ok = checker.check("README/config paths", _project_paths)

    print("\n[3/10] TraceWin local setup from README")

    tracewin_action = (
        "Copy the TraceWin workspace/program files described in README.md, "
        "fix launcher permissions, then rerun check."
    )

    def _tracewin_local_setup():
        workspace = DEFAULT_TRACEWIN_INI.parent
        wrapper_dir = PROJECT_ROOT / "env/tracewin_env/tracewin/pyTraceWin_wrapper"
        tracewin_program_dir = wrapper_dir / "TraceWin_program"
        tracewin_binary = tracewin_program_dir / "TraceWin"
        launcher = wrapper_dir / "run_tracewin_with_permissions.sh"
        calc_dir = Path(args.tracewin_calc_dir)

        _require_path(workspace, "TraceWin workspace", action=tracewin_action,
                      must_be_file=False, readable=True)
        _require_path(DEFAULT_TRACEWIN_INI, "TraceWin project .ini", action=tracewin_action,
                      must_be_file=True, readable=True)
        _require_path(workspace / "condensed.dat", "TraceWin lattice .dat",
                      action=tracewin_action, must_be_file=True, readable=True)
        _require_path(workspace / "16O5.dst", "TraceWin input distribution .dst",
                      action=tracewin_action, must_be_file=True, readable=True)
        _require_path(launcher, "TraceWin permission launcher", action=tracewin_action,
                      must_be_file=True, readable=True, executable=True)
        _require_path(tracewin_binary, "TraceWin binary", action=tracewin_action,
                      must_be_file=True, readable=True, executable=True)

        try:
            calc_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise CheckFailure(
                f"Cannot create TraceWin calc dir {calc_dir}: {exc}",
                action="Use --tracewin-calc-dir with a writable directory or fix permissions.",
                path_command=f"mkdir -p {calc_dir}",
            ) from exc
        if not os.access(calc_dir, os.W_OK):
            raise CheckFailure(
                f"TraceWin calc dir is not writable: {calc_dir}",
                action="Use --tracewin-calc-dir with a writable directory or fix permissions.",
                path_command=f"mkdir -p {calc_dir}",
            )
        return f"TraceWin files OK; calc dir writable: {calc_dir}"

    if args.skip_tracewin:
        tracewin_skip_reason = "--skip-tracewin requested."
        tracewin_skip_action = "Rerun without --skip-tracewin for full onboarding."
    elif not paths_ok:
        tracewin_skip_reason = "Project path check failed."
        tracewin_skip_action = "Fix missing project paths first."
    else:
        tracewin_skip_reason = None
        tracewin_skip_action = ""

    tracewin_files_ok = checker.check(
        "TraceWin workspace, binary, launcher and permissions",
        _tracewin_local_setup,
        skip_reason=tracewin_skip_reason,
        skip_action=tracewin_skip_action,
    )

    print("\n[4/10] TraceWinEnv real reset + step")

    def _tracewin_env_real():
        from beam_optimization.config.adige import N_PARAMS, observation_dim
        from beam_optimization.env.tracewin_env import TraceWinEnv

        obs_dim = observation_dim()
        env = TraceWinEnv(
            project_file=str(DEFAULT_TRACEWIN_INI),
            calc_dir=str(Path(args.tracewin_calc_dir)),
            max_steps=1,
            timeout=args.tracewin_timeout,
        )
        obs, info = env.reset(options={"randomize_params": False})
        _assert_obs_shape(obs, obs_dim, "TraceWinEnv reset")
        if "score" not in info or not np.isfinite(float(info["score"])):
            raise CheckFailure(
                "TraceWinEnv reset did not return a finite score.",
                action="Check TraceWin output parsing and generated partran files.",
            )
        result = info.get("sim_result")
        _assert_successful_sim_result(result, "TraceWinEnv reset")
        _assert_beam_states_shape(result.beam_states, "TraceWinEnv reset")

        zero_action = np.zeros(N_PARAMS, dtype=np.float32)
        obs2, reward, terminated, truncated, info2 = env.step(zero_action)
        _assert_obs_shape(obs2, obs_dim, "TraceWinEnv step")
        if not np.isfinite(float(reward)):
            raise CheckFailure(
                "TraceWinEnv step returned a non-finite reward.",
                action="Check score calculation and TraceWin simulation output.",
            )
        result2 = info2.get("sim_result")
        _assert_successful_sim_result(result2, "TraceWinEnv step")
        _assert_beam_states_shape(result2.beam_states, "TraceWinEnv step")
        return (
            f"reset_score={float(info['score']):.6g}, "
            f"step_score={float(info2['score']):.6g}, reward={float(reward):.6g}"
        )

    tracewin_env_ok = checker.check(
        "TraceWinEnv nominal reset and zero-action step",
        _tracewin_env_real,
        default_action=(
            "TraceWin files exist, but the real simulation failed. Check the "
            ".ini/.dat/field maps/.dst, license, launcher permissions and TraceWin output."
        ),
        default_path_command=str(Path(args.tracewin_calc_dir)),
        skip_reason=(
            "--skip-tracewin requested." if args.skip_tracewin
            else None if tracewin_files_ok else "TraceWin local setup did not pass."
        ),
        skip_action=(
            "Rerun without --skip-tracewin for full onboarding."
            if args.skip_tracewin else "Fix TraceWin local setup first."
        ),
    )
    state["tracewin_env_ok"] = tracewin_env_ok

    print("\n[5/10] Base dataset")

    def _dataset():
        from beam_optimization.env.dataset import BeamDataset

        dataset_path = default_dataset_path()
        if not dataset_path.exists():
            raise CheckFailure(
                f"Base dataset not found: {dataset_path}",
                action=_setup_action(),
                path_command=_setup_command(),
            )
        dataset = BeamDataset.load(dataset_path)
        if len(dataset) <= 0:
            raise CheckFailure(
                f"Base dataset is empty: {dataset_path}",
                action=_setup_action(),
                path_command=_setup_command(),
            )
        if tuple(dataset.X.shape[1:]) != (25,) or tuple(dataset.Y.shape[1:]) != (99,):
            raise CheckFailure(
                f"Dataset shapes are X={tuple(dataset.X.shape)}, Y={tuple(dataset.Y.shape)}.",
                action="Regenerate dataset/surrogates with the current config.",
                path_command=_setup_command(),
            )
        if dataset.scores.shape[0] != len(dataset):
            raise CheckFailure(
                f"Dataset scores length {dataset.scores.shape[0]} != samples {len(dataset)}.",
                action="Regenerate dataset/surrogates with the current config.",
                path_command=_setup_command(),
            )
        state["dataset"] = dataset
        return f"loaded {len(dataset):,} samples from {dataset_path}"

    dataset_ok = checker.check(
        "Load and validate dataset_base.pt",
        _dataset,
        default_action=_setup_action(),
        default_path_command=_setup_command(),
    )

    print("\n[6/10] Base surrogates")

    def _surrogates():
        from beam_optimization.env.surrogate_env import ModularMLP

        files = sorted(DEFAULT_BASE_SURROGATE_DIR.glob("surrogate_*.pt"))
        if not files:
            raise CheckFailure(
                f"No surrogate_*.pt files found in {DEFAULT_BASE_SURROGATE_DIR}",
                action=_setup_action(),
                path_command=_setup_command(),
            )
        surrogates = []
        for path in files:
            model = ModularMLP.load(str(path))
            model.eval()
            surrogates.append(model)
        state["surrogates"] = surrogates
        return f"loaded {len(surrogates)} surrogate(s) from {DEFAULT_BASE_SURROGATE_DIR}"

    surrogates_ok = checker.check(
        "Load base surrogate checkpoints",
        _surrogates,
        default_action=_setup_action(),
        default_path_command=_setup_command(),
    )

    print("\n[7/10] SurrogateEnv")

    def _surrogate_env():
        from beam_optimization.config.adige import observation_dim
        from beam_optimization.env.surrogate_env import SurrogateEnv

        obs_dim = observation_dim()
        env = SurrogateEnv(
            model=state["surrogates"][0],
            dataset=state["dataset"],
            max_steps=max(1, int(args.surrogate_steps)),
        )
        obs, info = env.reset()
        _assert_obs_shape(obs, obs_dim, "SurrogateEnv reset")
        result = info.get("sim_result")
        if result is None or result.source != "surrogate":
            raise CheckFailure(
                "SurrogateEnv reset did not return a surrogate simulation result.",
                action="Check surrogate simulator result conversion.",
            )
        _assert_beam_states_shape(result.beam_states, "SurrogateEnv reset")
        if "score" not in info or not np.isfinite(float(info["score"])):
            raise CheckFailure(
                "SurrogateEnv reset did not return a finite score.",
                action="Regenerate dataset/surrogates with the current config.",
                path_command=_setup_command(),
            )

        for _ in range(max(1, int(args.surrogate_steps))):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            _assert_obs_shape(obs, obs_dim, "SurrogateEnv step")
            if not np.isfinite(float(reward)):
                raise CheckFailure(
                    "SurrogateEnv step returned a non-finite reward.",
                    action="Regenerate dataset/surrogates with the current config.",
                    path_command=_setup_command(),
                )
            result = info.get("sim_result")
            if result is None or result.source != "surrogate":
                raise CheckFailure(
                    "SurrogateEnv step did not return a surrogate simulation result.",
                    action="Check surrogate simulator result conversion.",
                )
            _assert_beam_states_shape(result.beam_states, "SurrogateEnv step")

        state["surrogate_env"] = env
        return f"reset + {max(1, int(args.surrogate_steps))} random step(s) OK"

    surrogate_env_ok = checker.check(
        "SurrogateEnv reset and random steps",
        _surrogate_env,
        default_action=(
            "Dataset/surrogates exist but are incompatible with the current config; "
            "regenerate them."
        ),
        default_path_command=_setup_command(),
        skip_reason=None if (dataset_ok and surrogates_ok) else (
            "Dataset and surrogate checks must pass first."
        ),
        skip_action=_setup_action(),
    )

    print("\n[8/10] Surrogate ensemble")

    def _surrogate_ensemble():
        from beam_optimization.env.surrogate_env import SurrogateEnv

        surrogates = state["surrogates"]
        if len(surrogates) < 2:
            return CheckMessage(
                status="WARN",
                details="Only one surrogate is available; Thompson sampling diversity is skipped.",
                action="Train more base surrogates if you want ensemble uncertainty.",
                path_command=_setup_command(),
            )
        env = SurrogateEnv(model=surrogates, dataset=state["dataset"], max_steps=3)
        if len(env.simulator.ensemble) != len(surrogates):
            raise CheckFailure(
                "SurrogateEnv ensemble size does not match loaded surrogates.",
                action="Check SurrogateEnv/SurrogateBeamSimulator ensemble construction.",
            )
        seen = set()
        for _ in range(30):
            env.reset()
            seen.add(id(env.simulator.model))
        if len(seen) <= 1:
            raise CheckFailure(
                "Thompson sampling did not switch active surrogate over 30 resets.",
                action="Check SurrogateBeamSimulator.sample_model_index/reset_context.",
            )
        return f"{len(seen)} distinct active surrogate objects over 30 resets"

    checker.check(
        "Surrogate ensemble Thompson sampling",
        _surrogate_ensemble,
        skip_reason=None if (dataset_ok and surrogates_ok) else (
            "Dataset and surrogate checks must pass first."
        ),
        skip_action=_setup_action(),
    )

    print("\n[9/10] Algorithms")

    def _model_free_algorithms():
        from beam_optimization.algorithms import MODEL_FREE_ALGORITHMS, make_agent
        from beam_optimization.config.adige import N_PARAMS, action_bounds, observation_dim

        obs_dim = observation_dim()
        bounds = action_bounds()
        action_bounds_tuple = (bounds[0].tolist(), bounds[1].tolist())
        dummy_obs = np.zeros(obs_dim, dtype=np.float32)
        checked = []
        for name in MODEL_FREE_ALGORITHMS:
            agent = make_agent(name, obs_dim, N_PARAMS, action_bounds_tuple)
            action = _extract_action(agent.select_action(dummy_obs, training=False))
            if action.shape != (N_PARAMS,):
                raise CheckFailure(
                    f"{name} action shape is {action.shape}, expected {(N_PARAMS,)}.",
                    action="Fix the algorithm select_action interface.",
                )
            checked.append(name)
        return "checked custom model-free agents: " + ", ".join(checked)

    model_free_ok = checker.check(
        "Custom model-free agents",
        _model_free_algorithms,
        default_action="Fix algorithm imports/constructors/select_action interfaces.",
    )

    def _sb3_sac():
        from beam_optimization.algorithms.model_free.sb3_sac import SB3SAC
        from beam_optimization.env.surrogate_env import SurrogateEnv

        env = SurrogateEnv(
            model=state["surrogates"][0],
            dataset=state["dataset"],
            max_steps=1,
        )
        agent = SB3SAC(env, hidden_dims=(32, 32), buffer_size=1024, batch_size=32)
        if agent is None:
            raise CheckFailure(
                "SB3SAC constructor returned None.",
                action="Fix SB3SAC wrapper construction.",
            )
        return "SB3SAC wrapper construction OK"

    checker.check(
        "Stable-Baselines3 SAC wrapper",
        _sb3_sac,
        default_action=(
            "Install stable-baselines3 from requirements or fix the SB3SAC wrapper."
        ),
        default_path_command=INSTALL_COMMAND,
        skip_reason=None if (dataset_ok and surrogates_ok) else (
            "Dataset and surrogate checks must pass first."
        ),
        skip_action=_setup_action(),
    )

    def _mbpo():
        from beam_optimization.algorithms import make_agent
        from beam_optimization.algorithms.model_based.mbpo import MBPO
        from beam_optimization.config.adige import N_PARAMS, action_bounds, observation_dim

        obs_dim = observation_dim()
        bounds = action_bounds()
        action_bounds_tuple = (bounds[0].tolist(), bounds[1].tolist())
        inner = make_agent("sac", obs_dim, N_PARAMS, action_bounds_tuple)
        agent = MBPO(
            agent=inner,
            surrogates=state["surrogates"],
            dataset=state["dataset"],
            obs_dim=obs_dim,
            act_dim=N_PARAMS,
            rollout_length=1,
            n_synthetic_per_step=1,
            n_grad_updates=1,
        )
        action = _extract_action(agent.select_action(np.zeros(obs_dim, dtype=np.float32), training=False))
        if action.shape != (N_PARAMS,):
            raise CheckFailure(
                f"MBPO inner policy action shape is {action.shape}, expected {(N_PARAMS,)}.",
                action="Fix MBPO/inner SAC select_action interface.",
            )
        return "MBPO construction and inner action selection OK"

    checker.check(
        "MBPO construction",
        _mbpo,
        default_action="Fix MBPO construction or inner agent compatibility.",
        skip_reason=None if (dataset_ok and surrogates_ok and model_free_ok) else (
            "Dataset, surrogate and model-free checks must pass first."
        ),
        skip_action="Fix earlier dependency checks first.",
    )

    def _svg():
        from beam_optimization.algorithms.model_based.svg import SVGAgent
        from beam_optimization.config.adige import (
            N_PARAMS,
            PARAM_KEYS,
            action_bounds,
            default_params,
            observation_dim,
        )

        bounds = action_bounds()
        action_bounds_tuple = (bounds[0].tolist(), bounds[1].tolist())
        agent = SVGAgent(
            surrogate=state["surrogates"],
            dataset=state["dataset"],
            obs_dim=observation_dim(),
            act_dim=N_PARAMS,
            action_bounds=action_bounds_tuple,
            param_keys=PARAM_KEYS,
            default_params=default_params(),
            hidden_dims=(32, 32),
            n_step=1,
        )
        result = agent.optimize_episode()
        if not np.isfinite(float(result.final_score)):
            raise CheckFailure(
                "SVG mini optimize_episode returned a non-finite final score.",
                action="Fix DifferentiableSurrogateEnv/SVG backward pass.",
            )
        return f"SVG n_step=1 optimize_episode OK, final_score={result.final_score:.6g}"

    checker.check(
        "SVG differentiable mini episode",
        _svg,
        default_action="Fix SVG/DifferentiableSurrogateEnv differentiable rollout.",
        skip_reason=None if (dataset_ok and surrogates_ok) else (
            "Dataset and surrogate checks must pass first."
        ),
        skip_action=_setup_action(),
    )

    print("\n[10/10] Online surrogate updater")

    def _surrogate_updater():
        from beam_optimization.config.adige import BEAM_STATE_FEATURES, default_params, score
        from beam_optimization.env.simulation import BeamSimulationResult
        from beam_optimization.env.surrogate_env.surrogate.model.updater import (
            SurrogateDatasetUpdater,
        )

        updater = SurrogateDatasetUpdater(
            state["surrogates"],
            offline_dataset=state["dataset"],
            min_samples=5,
            batch_size=8,
            epochs=1,
        )
        for i in range(5):
            _, beam_states = state["dataset"].get_training_batch([i])
            fake_bs = np.stack(
                [stage.squeeze(0).numpy() for stage in beam_states],
                axis=0,
            ).astype(np.float32)
            fake_final = {
                name: float(fake_bs[-1][idx])
                for idx, name in enumerate(BEAM_STATE_FEATURES)
            }
            result = BeamSimulationResult(
                params=default_params(),
                beam_states=fake_bs,
                final_beam=fake_final,
                score_val=score(fake_final),
                success=True,
                source="tracewin",
            )
            if not updater.add_tracewin_result(result):
                raise CheckFailure(
                    "SurrogateDatasetUpdater rejected a valid fake TraceWin result.",
                    action="Fix tracewin_result_to_flat_sample/updater ingestion.",
                )
        losses = updater.update()
        if losses is None or len(losses) != len(state["surrogates"]):
            raise CheckFailure(
                "SurrogateDatasetUpdater.update did not return one loss per surrogate.",
                action="Fix online surrogate fine-tuning before using MBPO model update.",
            )
        return f"accepted {updater.n_online_samples} fake TraceWin samples; losses={losses}"

    checker.check(
        "SurrogateDatasetUpdater ingestion and tiny update",
        _surrogate_updater,
        default_action=(
            "Fix online surrogate updater before using MBPO with model update/online fine-tuning."
        ),
        skip_reason=None if (dataset_ok and surrogates_ok) else (
            "Dataset and surrogate checks must pass first."
        ),
        skip_action=_setup_action(),
    )

    # Keep otherwise-unused variables visible for debugger/readability.
    _ = python_ok, surrogate_env_ok
    return checker.summary()


if __name__ == "__main__":
    sys.exit(main())
