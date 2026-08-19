"""Iterative sim-to-real SAC with alternating surrogate and TraceWin phases."""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym

from beam_optimization.algorithms.model_free.stable_baselines import (
    StableBaselinesAgent,
)
from beam_optimization.algorithms.utils.atomic_save import atomic_save
from beam_optimization.config.adige import TEST_RESET_SCALE, TRAIN_RESET_SCALE
from beam_optimization.config.paths import configure_matplotlib_cache
from beam_optimization.env.dataset import BeamDataset
from beam_optimization.env.surrogate_env import SurrogateEnv
from beam_optimization.env.surrogate_env.surrogate.model.modular_mlp import ModularMLP
from beam_optimization.env.surrogate_env.surrogate.model.updater import (
    SurrogateDatasetUpdater,
)
from beam_optimization.scripts.common import evaluate_policy


STATE_VERSION = 1
EVAL_SEED = 10_000
POLICY_RELATIVE_PATH = Path("sac/sac_agent.zip")
LATEST_POLICY_RELATIVE_PATH = Path("sac/latest_agent.zip")
PRETRAINED_POLICY_RELATIVE_PATH = Path("sac/pretrained_agent.zip")
BEST_SURROGATE_POLICY_RELATIVE_PATH = Path("sac/best_surrogate_agent.zip")
BEST_TRACEWIN_POLICY_RELATIVE_PATH = Path("sac/best_tracewin_agent.zip")
REPLAY_RELATIVE_PATH = Path("sac/replay_buffer.pkl")
ONLINE_RELATIVE_PATH = Path("cumulative_online_dataset.pt")
WORKING_SURROGATE_RELATIVE_PATH = Path("working_surrogate/surrogate_0.pt")
LEGACY_OPTIONAL_CONFIG_KEYS = {
    "surrogate_learning_rate",
    "real_learning_rate",
    "real_update_interval",
    "surrogate_refresh",
    "freeze_entropy_on_tracewin",
    "fixed_entropy_coefficient",
    "best_tracewin_window",
}


@dataclass
class IterativeSim2RealSACConfig:
    """Configuration for one resumable iterative sim-to-real SAC run."""

    surrogate: str
    dataset: str
    tracewin: str
    output: str
    initial_policy: Optional[str] = None
    cycles: int = 1
    initial_surrogate_steps: int = 200_000
    subsequent_surrogate_steps: int = 20_000
    real_steps_per_cycle: int = 2_000
    real_learning_starts: int = 1_000
    surrogate_learning_rate: float = 3e-4
    real_learning_rate: float = 1e-5
    real_update_interval: int = 20
    surrogate_refresh: bool = False
    freeze_entropy_on_tracewin: bool = True
    fixed_entropy_coefficient: Optional[float] = None
    best_tracewin_window: int = 5
    max_ep_steps: int = 20
    online_mix_ratio: float = 0.25
    surrogate_update_steps: int = 200
    surrogate_update_batch_size: int = 128
    surrogate_update_lr: float = 3e-5
    checkpoint_every_real_steps: int = 30
    checkpoint_every_surrogate_steps: int = 10_000
    eval_every: int = 1_000
    eval_episodes: int = 5
    hidden: tuple[int, ...] = (256, 256)
    seed: int = 42
    distance_penalty_weight: float = 0.02
    action_penalty_weight: float = 0.30
    action_smoothness_penalty_weight: float = 0.50
    score_regression_penalty_weight: float = 5.0
    tracewin_timeout: float = 180.0
    tracewin_threads: Optional[int] = None
    kill_stale: bool = False
    resume: bool = False
    no_tensorboard: bool = False
    enable_learning_curve: bool = True


class _LearningCurveRecorder:
    """Small self-contained recorder used by the model-based algorithm."""

    fieldnames = [
        "step", "episode", "eval_mean_reward", "eval_mean_score",
        "eval_std_score", "eval_best_score", "eval_episodes",
    ]

    def __init__(self, out_dir: Path, algorithm: str):
        self.out_dir = Path(out_dir)
        self.algorithm = algorithm
        self.rows: list[dict] = []

    def add(self, *, step: int, episode: int, metrics: dict) -> None:
        self.rows.append({
            "step": int(step),
            "episode": int(episode),
            "eval_mean_reward": float(metrics["mean_reward"]),
            "eval_mean_score": float(metrics["mean_score"]),
            "eval_std_score": float(metrics["std_score"]),
            "eval_best_score": float(metrics["best_score"]),
            "eval_episodes": int(metrics["episodes"]),
        })
        self.save_csv()
        self.save_plot()

    def save_csv(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / "learning_curve.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        return path

    def save_plot(self) -> Optional[Path]:
        if not self.rows:
            return None
        configure_matplotlib_cache()
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        steps = [row["step"] for row in self.rows]
        scores = [row["eval_mean_score"] for row in self.rows]
        stds = [row["eval_std_score"] for row in self.rows]
        fig, axis = plt.subplots(figsize=(7.2, 4.2))
        axis.plot(steps, scores, marker="o", linestyle="-", linewidth=1.8,
                  label=self.algorithm)
        if len(steps) > 1:
            axis.fill_between(
                steps,
                np.asarray(scores) - np.asarray(stds),
                np.asarray(scores) + np.asarray(stds),
                alpha=0.18,
            )
        axis.set_title(f"{self.algorithm} learning curve")
        axis.set_xlabel("Environment steps")
        axis.set_ylabel("Mean evaluation score")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        path = self.out_dir / "learning_curve.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return path


class TraceWinExperienceCollector(gym.Wrapper):
    """Collect already-paid reset and step simulations for surrogate updates."""

    def __init__(self, env, updater: SurrogateDatasetUpdater):
        super().__init__(env)
        self.updater = updater
        self.accepted_reset_samples = 0
        self.accepted_step_samples = 0
        self.episode_reward = 0.0

    def _collect(self, info: dict, *, reset: bool) -> None:
        result = info.get("sim_result")
        if result is None:
            return
        accepted = self.updater.add_tracewin_result(result)
        if accepted and reset:
            self.accepted_reset_samples += 1
        elif accepted:
            self.accepted_step_samples += 1

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_reward = 0.0
        self._collect(info, reset=True)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._collect(info, reset=False)
        self.episode_reward += float(reward)
        if (terminated or truncated) and not info.get("technical_failure", False):
            info["tracewin_episode_final_score"] = float(
                info.get("score", float("nan"))
            )
            info["tracewin_episode_reward"] = float(self.episode_reward)
        return obs, reward, terminated, truncated, info


def _atomic_json(path: Path, payload: dict) -> None:
    def write(tmp_path: str) -> None:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    atomic_save(path, write)


def _atomic_dataset(dataset: BeamDataset, path: Path) -> None:
    atomic_save(path, lambda tmp_path: dataset.save_flat(tmp_path))


def _atomic_copy(source: Path, target: Path) -> None:
    atomic_save(target, lambda tmp_path: shutil.copyfile(source, tmp_path))


def _validate_config(config: IterativeSim2RealSACConfig) -> None:
    positive = {
        "cycles": config.cycles,
        "initial_surrogate_steps": config.initial_surrogate_steps,
        "subsequent_surrogate_steps": config.subsequent_surrogate_steps,
        "real_steps_per_cycle": config.real_steps_per_cycle,
        "max_ep_steps": config.max_ep_steps,
        "surrogate_update_steps": config.surrogate_update_steps,
        "surrogate_update_batch_size": config.surrogate_update_batch_size,
        "checkpoint_every_real_steps": config.checkpoint_every_real_steps,
        "checkpoint_every_surrogate_steps": config.checkpoint_every_surrogate_steps,
        "eval_every": config.eval_every,
        "eval_episodes": config.eval_episodes,
        "best_tracewin_window": config.best_tracewin_window,
    }
    errors = [f"{name} must be positive" for name, value in positive.items() if value <= 0]
    if config.real_learning_starts < 0:
        errors.append("real_learning_starts must be non-negative")
    if not math.isfinite(config.surrogate_learning_rate) or config.surrogate_learning_rate <= 0.0:
        errors.append("surrogate_learning_rate must be positive")
    if not math.isfinite(config.real_learning_rate) or config.real_learning_rate <= 0.0:
        errors.append("real_learning_rate must be positive")
    if config.real_update_interval <= 0:
        errors.append("real_update_interval must be positive")
    if (
        config.fixed_entropy_coefficient is not None
        and (
            not math.isfinite(config.fixed_entropy_coefficient)
            or config.fixed_entropy_coefficient <= 0.0
        )
    ):
        errors.append("fixed_entropy_coefficient must be positive when provided")
    if not 0.0 <= config.online_mix_ratio <= 1.0:
        errors.append("online_mix_ratio must be between 0 and 1")
    if config.surrogate_update_lr <= 0.0:
        errors.append("surrogate_update_lr must be positive")
    if config.tracewin_timeout <= 0.0:
        errors.append("tracewin_timeout must be positive")
    if config.tracewin_threads is not None and config.tracewin_threads <= 0:
        errors.append("tracewin_threads must be positive when provided")
    if errors:
        raise ValueError("; ".join(errors))


def _new_state(args: IterativeSim2RealSACConfig) -> dict:
    return {
        "version": STATE_VERSION,
        "phase": "real" if args.initial_policy else "initial_surrogate",
        "cycle": 1,
        "phase_steps_completed": 0,
        "global_sac_steps": 0,
        "real_learning_start_timestep": None,
        "online_samples": 0,
        "policy_path": str(POLICY_RELATIVE_PATH),
        "latest_policy_path": str(LATEST_POLICY_RELATIVE_PATH),
        "pretrained_policy_path": None,
        "best_surrogate_policy_path": None,
        "best_tracewin_policy_path": None,
        "best_tracewin_score": None,
        "recent_tracewin_final_scores": [],
        "replay_buffer_path": str(REPLAY_RELATIVE_PATH),
        "online_dataset_path": str(ONLINE_RELATIVE_PATH),
        "working_surrogate_path": str(WORKING_SURROGATE_RELATIVE_PATH),
        "config": _config_payload(args),
    }


def _config_payload(config: IterativeSim2RealSACConfig) -> dict:
    payload = {
        key: value
        for key, value in vars(config).items()
        if key != "resume"
    }
    payload["hidden"] = list(config.hidden)
    return payload


def _load_state(output: Path) -> dict:
    state_path = output / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Cannot resume: missing {state_path}")
    with open(state_path, encoding="utf-8") as handle:
        state = json.load(handle)
    if state.get("version") != STATE_VERSION:
        raise ValueError(
            f"Unsupported workflow state version: {state.get('version')}"
        )
    return state


def _save_checkpoint(
    *,
    output: Path,
    agent: StableBaselinesAgent,
    updater: SurrogateDatasetUpdater,
    state: dict,
) -> None:
    policy_path = output / POLICY_RELATIVE_PATH
    replay_path = output / REPLAY_RELATIVE_PATH
    online_path = output / ONLINE_RELATIVE_PATH

    agent.save(str(policy_path))
    _atomic_copy(policy_path, output / LATEST_POLICY_RELATIVE_PATH)
    agent.save_replay_buffer(replay_path)
    _atomic_dataset(updater.online_dataset, online_path)
    state["global_sac_steps"] = agent.num_timesteps
    state["online_samples"] = updater.n_online_samples
    _atomic_json(output / "state.json", state)


def _save_pretrained_policy(output: Path, agent: StableBaselinesAgent, state: dict) -> None:
    """Save the immutable policy used at the start of real fine-tuning."""
    path = output / PRETRAINED_POLICY_RELATIVE_PATH
    if not path.is_file():
        agent.save(str(path))
    state["pretrained_policy_path"] = str(PRETRAINED_POLICY_RELATIVE_PATH)


def _consider_best_tracewin_policy(
    *, output: Path, agent: StableBaselinesAgent, state: dict, infos: list[dict]
) -> None:
    """Select a protected checkpoint by rolling real-episode final score."""
    scores = [
        float(info["tracewin_episode_final_score"])
        for info in infos
        if "tracewin_episode_final_score" in info
        and math.isfinite(float(info["tracewin_episode_final_score"]))
    ]
    if not scores:
        return
    window = int(state["config"].get("best_tracewin_window", 5))
    recent = list(state.get("recent_tracewin_final_scores", []))
    recent.extend(scores)
    recent = recent[-window:]
    state["recent_tracewin_final_scores"] = recent
    metric = float(sum(recent) / len(recent))
    previous_best = state.get("best_tracewin_score")
    if previous_best is not None and metric <= float(previous_best):
        return
    agent.save(str(output / BEST_TRACEWIN_POLICY_RELATIVE_PATH))
    state["best_tracewin_score"] = metric
    state["best_tracewin_final_score"] = float(scores[-1])
    state["best_tracewin_step"] = int(agent.num_timesteps)
    state["best_tracewin_policy_path"] = str(BEST_TRACEWIN_POLICY_RELATIVE_PATH)


def _phase_checkpoint_hook(
    *,
    output: Path,
    agent: StableBaselinesAgent,
    updater: SurrogateDatasetUpdater,
    state: dict,
    starting_steps: int,
    interval: int,
    budget: int,
    progress_label: Optional[str] = None,
    gradient_update_interval: Optional[int] = None,
) -> Callable[[int, int, list[dict]], None]:
    def hook(local_steps: int, _global_steps: int, infos: list[dict]) -> None:
        completed = starting_steps + int(local_steps)
        state["phase_steps_completed"] = completed
        if gradient_update_interval is not None:
            agent.set_off_policy_gradient_steps(
                1 if completed % gradient_update_interval == 0 else 0
            )
        if progress_label is not None:
            info = infos[-1] if infos else {}
            score = info.get("score")
            score_text = "n/a" if score is None else f"{float(score):.4g}"
            print(
                f"{progress_label} real_step={completed}/{budget} "
                f"score={score_text} online_samples={updater.n_online_samples}",
                flush=True,
            )
            _consider_best_tracewin_policy(
                output=output, agent=agent, state=state, infos=infos
            )
        if local_steps % interval == 0:
            _save_checkpoint(
                output=output,
                agent=agent,
                updater=updater,
                state=state,
            )

    return hook


def _train_phase(
    *,
    output: Path,
    agent: StableBaselinesAgent,
    env,
    updater: SurrogateDatasetUpdater,
    state: dict,
    budget: int,
    checkpoint_interval: int,
    reset_num_timesteps: bool,
    eval_fn: Optional[Callable] = None,
    eval_logger: Optional[Callable] = None,
    eval_every: int = 1000,
    eval_episodes: int = 5,
    progress_label: Optional[str] = None,
    learning_rate: Optional[float] = None,
    gradient_update_interval: Optional[int] = None,
) -> None:
    completed = int(state["phase_steps_completed"])
    remaining = int(budget) - completed
    if remaining <= 0:
        return
    if learning_rate is not None:
        agent.configure_off_policy_updates(
            learning_rate=learning_rate,
            gradient_steps=0 if gradient_update_interval is not None else 1,
        )
    hook = _phase_checkpoint_hook(
        output=output,
        agent=agent,
        updater=updater,
        state=state,
        starting_steps=completed,
        interval=checkpoint_interval,
        budget=budget,
        progress_label=progress_label,
        gradient_update_interval=gradient_update_interval,
    )
    agent.train(
        env,
        n_steps=remaining,
        log_every=max(1, remaining // 20),
        eval_every=eval_every,
        eval_episodes=eval_episodes,
        eval_fn=eval_fn,
        eval_logger=eval_logger,
        reset_num_timesteps=reset_num_timesteps,
        step_hook=hook,
    )
    state["phase_steps_completed"] = int(budget)
    _save_checkpoint(output=output, agent=agent, updater=updater, state=state)


def _commit_surrogate_update(
    *,
    output: Path,
    cycle: int,
    updater: SurrogateDatasetUpdater,
    offline_dataset: BeamDataset,
) -> Path:
    cycle_dir = output / f"cycle_{cycle:02d}"
    if cycle_dir.is_dir():
        checkpoint = cycle_dir / "updated_surrogate/surrogate_0.pt"
        if not checkpoint.is_file():
            raise RuntimeError(f"Incomplete existing cycle output: {cycle_dir}")
        return checkpoint

    pending = Path(tempfile.mkdtemp(prefix=f".cycle_{cycle:02d}.", dir=output))
    try:
        updater.update()
        updater.save_surrogates(pending / "updated_surrogate")
        updater.online_dataset.save_flat(pending / "tracewin_online_dataset.pt")
        offline_dataset.merge(updater.online_dataset).save_flat(
            pending / "merged_dataset.pt"
        )
        os.replace(pending, cycle_dir)
    except BaseException:
        shutil.rmtree(pending, ignore_errors=True)
        raise
    return cycle_dir / "updated_surrogate/surrogate_0.pt"


def _run_workflow(
    args: IterativeSim2RealSACConfig,
    *,
    surrogate_env_factory: Optional[Callable] = None,
    tracewin_env_factory: Optional[Callable] = None,
    agent_factory: Optional[Callable] = None,
    updater_factory: Optional[Callable] = None,
) -> dict:
    """Run or resume the iterative workflow.

    Factories are injectable so the complete state machine can be tested
    without launching TraceWin.
    """
    _validate_config(args)
    output = Path(args.output)
    base_surrogate_path = Path(args.surrogate)
    dataset_path = Path(args.dataset)

    if args.resume:
        state = _load_state(output)
        saved_config = state.get("config", {})
        current_config = _config_payload(args)
        # States created before TraceWin-only mode alternated through a
        # surrogate refresh and left SAC entropy automatic. Preserve those
        # semantics when resuming an old in-progress run.
        legacy_runtime = {}
        if "surrogate_refresh" not in saved_config:
            legacy_runtime["surrogate_refresh"] = True
            current_config["surrogate_refresh"] = True
        if "freeze_entropy_on_tracewin" not in saved_config:
            legacy_runtime["freeze_entropy_on_tracewin"] = False
            current_config["freeze_entropy_on_tracewin"] = False
        if legacy_runtime:
            args = replace(args, **legacy_runtime)
        mismatches = {
            key: (saved_config.get(key), value)
            for key, value in current_config.items()
            if saved_config.get(key) != value
            and not (key in LEGACY_OPTIONAL_CONFIG_KEYS and key not in saved_config)
        }
        if mismatches:
            details = ", ".join(
                f"{key}: saved={old!r}, requested={new!r}"
                for key, (old, new) in sorted(mismatches.items())
            )
            raise ValueError(f"Resume configuration does not match state.json: {details}")
        # Checkpoints created before phase-specific SAC tuning existed do not
        # contain these keys. Adopt the explicitly requested/default values
        # without weakening mismatch checks for any pre-existing option.
        missing_legacy_keys = LEGACY_OPTIONAL_CONFIG_KEYS - saved_config.keys()
        if missing_legacy_keys:
            state["config"] = {
                **saved_config,
                **{key: current_config[key] for key in missing_legacy_keys},
            }
            _atomic_json(output / "state.json", state)
    else:
        if output.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing output: {output}. Use --resume."
            )
        if not base_surrogate_path.is_file():
            raise FileNotFoundError(f"Surrogate not found: {base_surrogate_path}")
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        if not Path(args.tracewin).is_file():
            raise FileNotFoundError(f"TraceWin project not found: {args.tracewin}")
        if args.initial_policy:
            initial_policy_path = Path(args.initial_policy)
            if initial_policy_path.is_dir():
                initial_policy_path = initial_policy_path / "sac_agent.zip"
            if not initial_policy_path.is_file():
                raise FileNotFoundError(
                    f"Initial SAC policy not found: {initial_policy_path}"
                )
        output.mkdir(parents=True)
        _atomic_copy(
            base_surrogate_path,
            output / WORKING_SURROGATE_RELATIVE_PATH,
        )
        state = _new_state(args)

    working_surrogate_path = output / state["working_surrogate_path"]
    if not working_surrogate_path.is_file():
        raise FileNotFoundError(
            f"Working surrogate checkpoint not found: {working_surrogate_path}"
        )

    offline_dataset = BeamDataset.load(dataset_path)
    online_path = output / ONLINE_RELATIVE_PATH
    online_dataset = (
        BeamDataset.load(online_path)
        if args.resume and online_path.is_file()
        else BeamDataset()
    )
    surrogate = ModularMLP.load(str(working_surrogate_path))
    surrogate.eval()

    updater_kwargs = dict(
        surrogates=surrogate,
        offline_dataset=offline_dataset,
        lr=args.surrogate_update_lr,
        batch_size=args.surrogate_update_batch_size,
        epochs=args.surrogate_update_steps,
        min_samples=1,
        online_mix_ratio=args.online_mix_ratio,
        initial_online_dataset=online_dataset,
    )
    updater = (
        updater_factory(**updater_kwargs)
        if updater_factory is not None
        else SurrogateDatasetUpdater(**updater_kwargs)
    )

    def make_surrogate_env(reset_scale: float = TRAIN_RESET_SCALE):
        kwargs = dict(
            model=surrogate,
            dataset=offline_dataset,
            max_steps=args.max_ep_steps,
            reset_scale=reset_scale,
            distance_penalty_weight=args.distance_penalty_weight,
            action_penalty_weight=args.action_penalty_weight,
            action_smoothness_penalty_weight=args.action_smoothness_penalty_weight,
            score_regression_penalty_weight=args.score_regression_penalty_weight,
        )
        return (
            surrogate_env_factory(**kwargs)
            if surrogate_env_factory is not None
            else SurrogateEnv(**kwargs)
        )

    initial_env = make_surrogate_env()
    recorder = _LearningCurveRecorder(output / "sac", "iterative_sim2real_sac")
    curve_csv = output / "sac/learning_curve.csv"
    if args.resume and curve_csv.is_file():
        with open(curve_csv, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                recorder.rows.append(
                    {
                        "step": int(row["step"]),
                        "episode": int(row["episode"]),
                        "eval_mean_reward": float(row["eval_mean_reward"]),
                        "eval_mean_score": float(row["eval_mean_score"]),
                        "eval_std_score": float(row["eval_std_score"]),
                        "eval_best_score": float(row["eval_best_score"]),
                        "eval_episodes": int(row["eval_episodes"]),
                    }
                )

    def make_eval_env():
        return make_surrogate_env(TEST_RESET_SCALE)

    def surrogate_eval(current_agent, n_episodes: int):
        return evaluate_policy(
            current_agent,
            make_eval_env,
            n_episodes,
            seed=EVAL_SEED,
        )

    def record_surrogate_eval(step: int, metrics: dict) -> None:
        recorder.add(step=step, episode=0, metrics=metrics)
        mean_score = float(metrics["mean_score"])
        previous = state.get("best_surrogate_validation_score")
        if previous is not None and mean_score <= float(previous):
            return
        agent.save(str(output / BEST_SURROGATE_POLICY_RELATIVE_PATH))
        state["best_surrogate_validation_score"] = mean_score
        state["best_surrogate_step"] = int(step)
        state["best_surrogate_policy_path"] = str(
            BEST_SURROGATE_POLICY_RELATIVE_PATH
        )
    policy_path = output / POLICY_RELATIVE_PATH
    replay_path = output / REPLAY_RELATIVE_PATH
    if args.resume:
        agent = StableBaselinesAgent.load("sac", str(policy_path), env=initial_env)
        agent.load_replay_buffer(replay_path)
        learning_start = state.get("real_learning_start_timestep")
        if learning_start is not None:
            agent.set_learning_starts_timestep(learning_start)
    elif args.initial_policy:
        initial_policy_path = Path(args.initial_policy)
        if initial_policy_path.is_dir():
            initial_policy_path = initial_policy_path / "sac_agent.zip"
        agent = StableBaselinesAgent.load(
            "sac", str(initial_policy_path), env=initial_env
        )
        agent.reset_replay_buffer()
        _save_pretrained_policy(output, agent, state)
        if args.freeze_entropy_on_tracewin:
            state["frozen_entropy_coefficient"] = agent.freeze_entropy_coefficient(
                args.fixed_entropy_coefficient
            )
        state["real_learning_start_timestep"] = agent.delay_learning(
            args.real_learning_starts
        )
        print(
            "Loaded initial SAC policy from "
            f"{initial_policy_path} at timestep {agent.num_timesteps}."
        )
        print(
            "Replay cleared; starting cycle 1 directly on TraceWin after "
            f"{args.real_learning_starts} collection-only steps."
        )
        _save_checkpoint(output=output, agent=agent, updater=updater, state=state)
    else:
        factory = agent_factory or StableBaselinesAgent
        replay_capacity = max(
            args.initial_surrogate_steps,
            args.subsequent_surrogate_steps + args.real_steps_per_cycle,
            args.real_learning_starts + 1,
        )
        agent = factory(
            "sac",
            initial_env,
            hidden_dims=tuple(args.hidden),
            seed=args.seed,
            tensorboard_log=None if args.no_tensorboard else str(output / "tensorboard"),
            model_kwargs={
                "buffer_size": replay_capacity,
                "learning_rate": args.surrogate_learning_rate,
                "train_freq": 1,
                "gradient_steps": 1,
            },
        )
        _save_checkpoint(output=output, agent=agent, updater=updater, state=state)

    try:
        while state["phase"] != "complete":
            phase = state["phase"]
            cycle = int(state["cycle"])
            print(
                f"\n[iterative_sim2real_sac] phase={phase} "
                f"cycle={cycle}/{args.cycles} "
                f"completed={state['phase_steps_completed']} "
                f"global_sac_steps={agent.num_timesteps} "
                f"online_samples={updater.n_online_samples}"
            )

            if phase == "initial_surrogate":
                print(
                    f"Training SAC on the base SurrogateEnv for "
                    f"{args.initial_surrogate_steps} total phase steps "
                    f"(learning_rate={args.surrogate_learning_rate:g})."
                )
                env = make_surrogate_env()
                _train_phase(
                    output=output,
                    agent=agent,
                    env=env,
                    updater=updater,
                    state=state,
                    budget=args.initial_surrogate_steps,
                    checkpoint_interval=args.checkpoint_every_surrogate_steps,
                    reset_num_timesteps=(agent.num_timesteps == 0),
                    eval_fn=surrogate_eval if args.enable_learning_curve else None,
                    eval_logger=(
                        record_surrogate_eval if args.enable_learning_curve else None
                    ),
                    eval_every=args.eval_every,
                    eval_episodes=args.eval_episodes,
                    learning_rate=args.surrogate_learning_rate,
                )
                env.close()
                best_surrogate_path = output / BEST_SURROGATE_POLICY_RELATIVE_PATH
                if best_surrogate_path.is_file():
                    completed_surrogate_steps = agent.num_timesteps
                    selected_agent = StableBaselinesAgent.load(
                        "sac", str(best_surrogate_path), env=initial_env
                    )
                    # Keep the workflow's monotonic global counter while
                    # restoring the weights/optimizers/entropy from the best
                    # validation point.
                    selected_agent._model.num_timesteps = completed_surrogate_steps
                    agent = selected_agent
                    state["selected_surrogate_step"] = state.get(
                        "best_surrogate_step"
                    )
                agent.reset_replay_buffer()
                _save_pretrained_policy(output, agent, state)
                if args.freeze_entropy_on_tracewin:
                    state["frozen_entropy_coefficient"] = agent.freeze_entropy_coefficient(
                        args.fixed_entropy_coefficient
                    )
                print(
                    "Synthetic replay cleared before TraceWin; "
                    f"real learning starts after {args.real_learning_starts} new steps."
                )
                state.update(
                    phase="real",
                    cycle=1,
                    phase_steps_completed=0,
                    real_learning_start_timestep=agent.delay_learning(
                        args.real_learning_starts
                    ),
                )
                _save_checkpoint(
                    output=output, agent=agent, updater=updater, state=state
                )
                continue

            if phase == "surrogate_refresh":
                print(
                    f"Continuing the same SAC on the updated SurrogateEnv for "
                    f"{args.subsequent_surrogate_steps} total phase steps; "
                    f"learning_rate={args.surrogate_learning_rate:g}; "
                    "the real replay is retained during this phase."
                )
                env = make_surrogate_env()
                _train_phase(
                    output=output,
                    agent=agent,
                    env=env,
                    updater=updater,
                    state=state,
                    budget=args.subsequent_surrogate_steps,
                    checkpoint_interval=args.checkpoint_every_surrogate_steps,
                    reset_num_timesteps=False,
                    eval_fn=surrogate_eval if args.enable_learning_curve else None,
                    eval_logger=(
                        record_surrogate_eval if args.enable_learning_curve else None
                    ),
                    eval_every=args.eval_every,
                    eval_episodes=args.eval_episodes,
                    learning_rate=args.surrogate_learning_rate,
                )
                env.close()
                agent.reset_replay_buffer()
                print(
                    "Mixed refresh replay cleared before the next TraceWin block."
                )
                state.update(
                    phase="real",
                    phase_steps_completed=0,
                    real_learning_start_timestep=agent.delay_learning(
                        args.real_learning_starts
                    ),
                )
                _save_checkpoint(
                    output=output, agent=agent, updater=updater, state=state
                )
                continue

            if phase == "real":
                print(
                    f"Continuing the same SAC on TraceWinEnv for "
                    f"{args.real_steps_per_cycle} total phase steps. "
                    f"learning_rate={args.real_learning_rate:g}; one gradient "
                    f"update every {args.real_update_interval} real steps after "
                    f"the {args.real_learning_starts}-step collection window. "
                    "No TraceWin evaluation episodes will be run."
                )
                tracewin_kwargs = dict(
                    project_file=args.tracewin,
                    max_steps=args.max_ep_steps,
                    reset_scale=TRAIN_RESET_SCALE,
                    distance_penalty_weight=args.distance_penalty_weight,
                    action_penalty_weight=args.action_penalty_weight,
                    action_smoothness_penalty_weight=args.action_smoothness_penalty_weight,
                    score_regression_penalty_weight=args.score_regression_penalty_weight,
                    distance_dataset=offline_dataset,
                    timeout=args.tracewin_timeout,
                    num_threads=args.tracewin_threads,
                    kill_stale=args.kill_stale,
                )
                if tracewin_env_factory is not None:
                    real_env = tracewin_env_factory(**tracewin_kwargs)
                else:
                    from beam_optimization.env.tracewin_env import TraceWinEnv

                    real_env = TraceWinEnv(**tracewin_kwargs)
                env = TraceWinExperienceCollector(real_env, updater)
                _train_phase(
                    output=output,
                    agent=agent,
                    env=env,
                    updater=updater,
                    state=state,
                    budget=args.real_steps_per_cycle,
                    checkpoint_interval=args.checkpoint_every_real_steps,
                    reset_num_timesteps=False,
                    progress_label=f"cycle={cycle}/{args.cycles}",
                    learning_rate=args.real_learning_rate,
                    gradient_update_interval=args.real_update_interval,
                )
                env.close()
                if not args.surrogate_refresh:
                    if cycle >= args.cycles:
                        state.update(
                            phase="complete",
                            phase_steps_completed=0,
                            real_learning_start_timestep=None,
                        )
                    else:
                        # TraceWin-only cycles are contiguous checkpoint blocks:
                        # retain all real replay and do not repeat warm-up.
                        state.update(
                            phase="real",
                            cycle=cycle + 1,
                            phase_steps_completed=0,
                        )
                else:
                    state.update(
                        phase="surrogate_update",
                        phase_steps_completed=0,
                    )
                _save_checkpoint(
                    output=output, agent=agent, updater=updater, state=state
                )
                continue

            if phase == "surrogate_update":
                print(
                    f"Fine-tuning the working surrogate with "
                    f"{updater.n_online_samples} cumulative TraceWin samples."
                )
                updated_checkpoint = _commit_surrogate_update(
                    output=output,
                    cycle=cycle,
                    updater=updater,
                    offline_dataset=offline_dataset,
                )
                _atomic_copy(
                    updated_checkpoint,
                    output / WORKING_SURROGATE_RELATIVE_PATH,
                )
                # Usually the updater mutated this exact object. Loading the
                # committed checkpoint as well covers resume after the cycle
                # directory was atomically published but before state.json
                # advanced to the next phase.
                committed = ModularMLP.load(str(updated_checkpoint))
                surrogate.load_state_dict(committed.state_dict())
                surrogate.eval()
                if cycle >= args.cycles:
                    state.update(
                        phase="complete",
                        phase_steps_completed=0,
                        real_learning_start_timestep=None,
                    )
                else:
                    state.update(
                        phase="surrogate_refresh",
                        cycle=cycle + 1,
                        phase_steps_completed=0,
                        real_learning_start_timestep=None,
                    )
                _save_checkpoint(
                    output=output, agent=agent, updater=updater, state=state
                )
                continue

            raise ValueError(f"Unknown workflow phase in state.json: {phase}")
    finally:
        close = getattr(initial_env, "close", None)
        if callable(close):
            close()

    best_surrogate_score = (
        max(row["eval_best_score"] for row in recorder.rows)
        if recorder.rows
        else None
    )
    best_tracewin_score = state.get("best_tracewin_score")
    summary = {
        "status": "complete",
        "cycles": args.cycles,
        "initial_policy": args.initial_policy,
        "global_sac_steps": agent.num_timesteps,
        "online_samples": updater.n_online_samples,
        "policy": str(output / POLICY_RELATIVE_PATH),
        "latest_policy": str(output / LATEST_POLICY_RELATIVE_PATH),
        "pretrained_policy": (
            str(output / PRETRAINED_POLICY_RELATIVE_PATH)
            if (output / PRETRAINED_POLICY_RELATIVE_PATH).is_file()
            else None
        ),
        "best_surrogate_policy": (
            str(output / BEST_SURROGATE_POLICY_RELATIVE_PATH)
            if (output / BEST_SURROGATE_POLICY_RELATIVE_PATH).is_file()
            else None
        ),
        "best_surrogate_validation_score": state.get(
            "best_surrogate_validation_score"
        ),
        "selected_surrogate_step": state.get("selected_surrogate_step"),
        "best_tracewin_policy": (
            str(output / BEST_TRACEWIN_POLICY_RELATIVE_PATH)
            if (output / BEST_TRACEWIN_POLICY_RELATIVE_PATH).is_file()
            else None
        ),
        "best_tracewin_score": best_tracewin_score,
        "best_tracewin_step": state.get("best_tracewin_step"),
        "frozen_entropy_coefficient": state.get("frozen_entropy_coefficient"),
        "replay_buffer": str(output / REPLAY_RELATIVE_PATH),
        "working_surrogate": str(output / WORKING_SURROGATE_RELATIVE_PATH),
        # Public training summaries should describe the real phase when one
        # has completed, not a historical surrogate-only maximum.
        "best_score": (
            best_tracewin_score
            if best_tracewin_score is not None
            else best_surrogate_score
        ),
        "best_surrogate_score": best_surrogate_score,
        "learning_curve": list(recorder.rows),
    }
    _atomic_json(output / "summary.json", summary)
    return summary


class IterativeSim2RealSAC:
    """Resumable model-based workflow built around one persistent SB3 SAC."""

    def __init__(
        self,
        config: IterativeSim2RealSACConfig,
        *,
        surrogate_env_factory: Optional[Callable] = None,
        tracewin_env_factory: Optional[Callable] = None,
        agent_factory: Optional[Callable] = None,
        updater_factory: Optional[Callable] = None,
    ):
        _validate_config(config)
        self.config = config
        self._factories = {
            "surrogate_env_factory": surrogate_env_factory,
            "tracewin_env_factory": tracewin_env_factory,
            "agent_factory": agent_factory,
            "updater_factory": updater_factory,
        }
        self.summary: Optional[dict] = None

    def train(self) -> dict:
        """Start a new workflow and refuse to overwrite an existing output."""
        self.summary = _run_workflow(replace(self.config, resume=False), **self._factories)
        return self.summary

    def resume(self) -> dict:
        """Resume the phase recorded in ``state.json``."""
        self.summary = _run_workflow(replace(self.config, resume=True), **self._factories)
        return self.summary


def run_workflow(
    config: IterativeSim2RealSACConfig,
    **factories,
) -> dict:
    """Compatibility function for Python callers; prefer the public class."""
    algorithm = IterativeSim2RealSAC(config, **factories)
    return algorithm.resume() if config.resume else algorithm.train()
