"""
It is a base class for two beam optimization environments: TraceWinEnv and SurrogateEnv.

Both environments perform the same Gym loop (reset, step, render) but they differ 
in the backend simulator that produces the beam states and score for a given set of machine parameters.

"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from beam_optimization.config.adige import (
    ERROR_SCORE, MAX_STEPS, MAX_TERMINAL_RESET_ATTEMPTS,
    below_rl_min_npart_ratio,
    REWARD_SCORE_SCALE, TERMINAL_FAILURE_REWARD,
    TRAIN_RESET_SCALE, SCORE_REFERENCES,
    PARAM_KEYS, PARAMETERS, BEAM_STATE_DIM,
    BEAM_STATE_FEATURES, default_params, action_bounds, reset_std_vec,
    observation_dim, observation_stage_labels, observation_stage_indices,
    select_observation_stages, clip_params_to_hw, params_to_vec,
)

from beam_optimization.env.dataset.dataset import param_knn_distance
from beam_optimization.env.simulation import (
    BeamSimulationResult,
    BeamSimulator,
    canonical_physics_failure_reason,
)


@dataclass(eq=False)
class EpisodeState:
    """All per-episode mutable state for BaseBeamEnv: current step, best-of-episode,
    and the render() history. Reconstructed wholesale by __init__/reset()."""

    # Current step
    step_count: int = 0
    current_params: Dict[str, float] = field(default_factory=default_params)
    current_obs: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    current_score: float = ERROR_SCORE
    current_result: BeamSimulationResult | None = None
    previous_obs: np.ndarray | None = None
    last_action: np.ndarray | None = None
    last_reward: float = 0.0

    # Best-of-episode
    best_score: float = ERROR_SCORE
    best_params: Dict[str, float] = field(default_factory=default_params)
    best_step: int = 0

    # Per-episode history for render(): index 0 is the state right after
    # reset(), index k is the state after the k-th step().
    params_history: list = field(default_factory=list)
    obs_history: list = field(default_factory=list)
    score_history: list = field(default_factory=list)
    reward_history: list = field(default_factory=list)


class BaseBeamEnv(gym.Env, ABC):
    """Common reset/step/evaluate_params scaffolding for the two beam envs.

    Args:
        max_steps:    Episode length.
        reset_scale:  Gaussian reset width in sensitivity units. Training is
                      the default; evaluation workflows pass TEST_RESET_SCALE.
        Observation stages are selected by OBSERVATION_STAGE_MASK.
    """

    metadata = {"render_modes": ["human"]}

    # Construction: simulator, Gym spaces(observations and actions), and episode state (step count, current params, current obs, current score, previous obs, last action, last reward, best score, best params).
    # -------------------------------------------------------------------------
    def __init__(
        self,
        max_steps: int = MAX_STEPS,
        reset_scale: float = TRAIN_RESET_SCALE,
        distance_penalty_weight: float = 0.0,
        action_penalty_weight: float = 0.0,
        score_regression_penalty_weight: float = 0.0,
        action_smoothness_penalty_weight: float = 0.0,
    ):
        super().__init__()

        # Linear reward penalty on how far the current parameters are from the
        # training dataset (see param_knn_distance()), off by default so this
        # never changes existing behavior or forces the default dataset to
        # exist unless explicitly enabled.
        self.distance_penalty_weight = float(distance_penalty_weight)
        self.action_penalty_weight = float(action_penalty_weight)
        self.action_smoothness_penalty_weight = float(
            action_smoothness_penalty_weight
        )
        self.score_regression_penalty_weight = float(score_regression_penalty_weight)
        for name, value in (
            ("distance_penalty_weight", self.distance_penalty_weight),
            ("action_penalty_weight", self.action_penalty_weight),
            (
                "action_smoothness_penalty_weight",
                self.action_smoothness_penalty_weight,
            ),
            ("score_regression_penalty_weight", self.score_regression_penalty_weight),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number, got {value}")

        # every subclass must have a BeamSimulator
        self.simulator = self._build_simulator()
        if not isinstance(self.simulator, BeamSimulator):
            raise TypeError(
                f"{type(self).__name__}._build_simulator() must return a "
                f"BeamSimulator, got {type(self.simulator).__name__}"
            )

        # Gym spaces 
        
        # 1) Observation space
        obs_dim = observation_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # 2) Action space
        self.max_steps = int(max_steps)
        # Action space is set from per-parameter scales in ParameterSpec.
        act_low, act_high = action_bounds()
        self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)

        # Per-parameter reset stddevs from ParameterSpec. Callers explicitly
        # choose the training or test/evaluation reset distribution.
        self.reset_scale = float(reset_scale)
        self._reset_std = reset_std_vec(self.reset_scale).astype(np.float32)

        # Episode state (current step, best-of-episode, render() history).
        # KNN distances are derived from state.params_history lazily in
        # render(), so training steps never pay the k-d tree query (nor
        # force the default dataset file to exist).
        self.state = EpisodeState(current_obs=np.zeros(obs_dim, dtype=np.float32))


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        options = dict(options or {})
        explicit_params = options.get("initial_params")
        if explicit_params is not None and options.get("randomize_params", False):
            raise ValueError(
                "reset options 'initial_params' and randomize_params=True are mutually exclusive"
            )

        randomize_params = bool(options.get("randomize_params", True))
        if explicit_params is not None:
            if not isinstance(explicit_params, dict):
                raise ValueError("reset option 'initial_params' must be a parameter dictionary")
            expected_keys = set(PARAM_KEYS)
            supplied_keys = set(explicit_params)
            if supplied_keys != expected_keys:
                missing = sorted(expected_keys - supplied_keys)
                extra = sorted(supplied_keys - expected_keys)
                raise ValueError(
                    "reset option 'initial_params' must contain every configured parameter "
                    f"exactly once; missing={missing}, extra={extra}"
                )
            fixed_params = {key: float(explicit_params[key]) for key in PARAM_KEYS}
            if not np.isfinite(
                np.asarray(list(fixed_params.values()), dtype=np.float64)
            ).all():
                raise ValueError("reset option 'initial_params' contains NaN or infinite values")
            randomize_params = False
            reset_source = "explicit_params"
            active_reset_scale = self.reset_scale
            active_reset_std = self._reset_std
        else:
            active_reset_scale = self.reset_scale
            active_reset_std = self._reset_std
            reset_source = "gaussian" if randomize_params else "defaults"

        # Keep one beam/model context while rejecting only terminal parameter
        # samples. This avoids silently changing the physical input beam during
        # one reset() call.
        self.simulator.reset_context(self.np_random)

        max_attempts = MAX_TERMINAL_RESET_ATTEMPTS if randomize_params else 1
        rejected_terminal_resets = 0
        for reset_attempt in range(1, max_attempts + 1):
            if explicit_params is not None:
                params = dict(fixed_params)
            else:
                params = default_params()
                if randomize_params:
                    for key, std in zip(PARAM_KEYS, active_reset_std):
                        params[key] += float(self.np_random.normal(0.0, std))
            params = clip_params_to_hw(params)

            result = self.simulator.simulate(params)
            if self._is_technical_failure(result):
                raise RuntimeError(
                    "Simulator failed while creating the initial episode state: "
                    f"{result.error or 'no usable beam states were produced'}"
                )

            failure_reason = self._terminal_failure_reason(result)
            if failure_reason is None:
                break
            rejected_terminal_resets += 1
            if not randomize_params:
                raise RuntimeError(
                    "Initial parameters produce a terminal physical state "
                    f"({failure_reason}); reset cannot start an episode."
                )
        else:
            raise RuntimeError(
                "Could not sample a non-terminal initial state after "
                f"{MAX_TERMINAL_RESET_ATTEMPTS} Gaussian reset attempts."
            )

        obs, score, extra = self._result_to_obs_score_info(result)
        # Best-of-episode state must be reset here, not only in __init__:
        # benchmark environments are reused for multiple independent episodes.
        self.state = EpisodeState(
            current_params=params,
            current_obs=obs,
            current_score=score,
            current_result=result,
            params_history=[dict(params)],
            obs_history=[obs.copy()],
            score_history=[float(score)],
            reward_history=[0.0],
            best_score=float(score),
            best_params=dict(params),
            best_step=0,
        )

        info = {
            "score": score,
            "step": 0,
            "reset_randomized": randomize_params,
            "reset_source": reset_source,
            "reset_scale": active_reset_scale,
            "reset_attempts": reset_attempt,
            "rejected_terminal_resets": rejected_terminal_resets,
            **extra,
        }
        return obs.copy(), info

    @property
    def current_params(self) -> Dict[str, float]:
        """Return a defensive copy of the parameters at the current episode step."""
        return dict(self.state.current_params)

    def step(self, action: np.ndarray):

        previous_params = dict(self.state.current_params)

        # action to perform. It is a delta to apply to the current parameters.
        # The action is clipped to the action space bounds.
        action = np.clip(action, self.action_space.low, self.action_space.high)
        prev_obs = self.state.current_obs.copy()

        # modify parameter with deltas to the current parameters.
        for key, delta in zip(PARAM_KEYS, action):
            self.state.current_params[key] = float(self.state.current_params[key]) + float(delta)
        self.state.current_params = clip_params_to_hw(self.state.current_params)

        # perform concretely the action (the simulation) and get the new observation and final score.
        prev_score = self.state.current_score
        result = self.simulator.simulate(self.state.current_params)

        # Infrastructure failures are not physical transitions. Restore the
        # last usable state, add no reward, and truncate this rollout so the
        # replay buffer never learns from an SSH/Qt/timeout artifact.
        if self._is_technical_failure(result):
            self.state.current_params = previous_params
            self.state.current_result = result
            self.state.previous_obs = prev_obs
            self.state.last_action = action.copy()
            self.state.last_reward = 0.0
            self.state.step_count += 1

            self.state.params_history.append(dict(self.state.current_params))
            self.state.obs_history.append(prev_obs.copy())
            self.state.score_history.append(float(prev_score))
            self.state.reward_history.append(0.0)

            info = {
                "score": prev_score,
                "prev_score": prev_score,
                "reward": 0.0,
                "step": self.state.step_count,
                "best_score": self.state.best_score,
                "score_reward": 0.0,
                "distance_penalty": 0.0,
                "action_penalty": 0.0,
                "action_smoothness_penalty": 0.0,
                "score_regression_penalty": 0.0,
                "technical_failure": True,
                "sim_result": result,
            }
            return prev_obs, 0.0, False, True, info

        obs, score, extra = self._result_to_obs_score_info(result)

        # A physical beam loss is terminal. Valid states use the normalized
        # absolute physical score plus optional training-only regularizers.
        # The reported physical score itself is never modified.
        if extra["terminal_failure"]:
            distance_penalty = 0.0
            action_penalty = 0.0
            action_smoothness_penalty = 0.0
            score_regression_penalty = 0.0
            reward = TERMINAL_FAILURE_REWARD
        else:
            distance_penalty = self._distance_penalty(self.state.current_params)
            (
                action_penalty,
                action_smoothness_penalty,
                score_regression_penalty,
            ) = self._control_penalties(action, prev_score, score)
            reward = (
                score / REWARD_SCORE_SCALE
                - distance_penalty
                - action_penalty
                - action_smoothness_penalty
                - score_regression_penalty
            )

        # update episode state
        self.state.current_obs    = obs
        self.state.current_score  = score
        self.state.current_result = result
        self.state.previous_obs   = prev_obs
        self.state.last_action    = action.copy()
        self.state.last_reward    = float(reward)

        self.state.params_history.append(dict(self.state.current_params))
        self.state.obs_history.append(obs.copy())
        self.state.score_history.append(float(score))
        self.state.reward_history.append(float(reward))

        # update best score and best parameters if the current score is better than the best score so far.
        if score > self.state.best_score:
            self.state.best_score  = score
            self.state.best_params = self.state.current_params.copy()
            self.state.best_step   = self.state.step_count + 1

        # update step count
        self.state.step_count += 1
        terminated = bool(extra["terminal_failure"])
        truncated = False if terminated else self.state.step_count >= self.max_steps

        info = {"score": score, "prev_score": prev_score, "reward": reward,
                "step": self.state.step_count,
                "best_score": self.state.best_score,
                "score_reward": (
                    0.0 if extra["terminal_failure"] else score / REWARD_SCORE_SCALE
                ),
                "distance_penalty": (
                    0.0 if extra["terminal_failure"] else distance_penalty
                ),
                "action_penalty": (
                    0.0 if extra["terminal_failure"] else action_penalty
                ),
                "action_smoothness_penalty": (
                    0.0
                    if extra["terminal_failure"]
                    else action_smoothness_penalty
                ),
                "score_regression_penalty": (
                    0.0 if extra["terminal_failure"] else score_regression_penalty
                ),
                "physics_failure": self._is_physics_failure(result),
                "technical_failure": False,
                **extra}

        return obs.copy(), reward, terminated, truncated, info

    def _control_penalties(
        self,
        action: np.ndarray,
        prev_score: float,
        score: float,
    ) -> tuple[float, float, float]:
        """Return action-effort, smoothness, and score-regression costs.

        The quadratic action cost teaches a policy to emit zero once further
        corrections are not worth their effort. The regression cost makes an
        overshoot followed by an opposite correction worse than holding the
        better state. Both are dimensionless and affect only the RL reward.
        """
        action_penalty = 0.0
        if self.action_penalty_weight > 0.0:
            scale = np.maximum(
                np.abs(self.action_space.low),
                np.abs(self.action_space.high),
            )
            normalized_action = np.divide(
                np.asarray(action, dtype=np.float64),
                scale,
                out=np.zeros_like(scale, dtype=np.float64),
                where=scale > 0.0,
            )
            action_penalty = self.action_penalty_weight * float(
                np.mean(np.square(normalized_action))
            )

        smoothness_penalty = 0.0
        if (
            self.action_smoothness_penalty_weight > 0.0
            and self.state.last_action is not None
        ):
            scale = np.maximum(
                np.abs(self.action_space.low),
                np.abs(self.action_space.high),
            )
            normalized_delta = np.divide(
                np.asarray(action, dtype=np.float64)
                - np.asarray(self.state.last_action, dtype=np.float64),
                scale,
                out=np.zeros_like(scale, dtype=np.float64),
                where=scale > 0.0,
            )
            smoothness_penalty = (
                self.action_smoothness_penalty_weight
                * float(np.mean(np.square(normalized_delta)))
            )

        regression_penalty = 0.0
        if self.score_regression_penalty_weight > 0.0:
            normalized_drop = max(0.0, float(prev_score) - float(score)) / REWARD_SCORE_SCALE
            regression_penalty = (
                self.score_regression_penalty_weight * normalized_drop
            )

        return action_penalty, smoothness_penalty, regression_penalty

    def _distance_penalty(self, params: Dict[str, float]) -> float:
        """Linear penalty on how far `params` sit from the reference dataset
        (sensitivity-normalized k-NN distance, same metric render() already
        plots). 0.0 when disabled (self.distance_penalty_weight <= 0), which
        also skips the k-d tree query/dataset load entirely.

        Uses self.simulator.dataset when the simulator has one (e.g.
        SurrogateBeamSimulator, trained on a specific dataset) so the distance
        is measured against the dataset actually relevant to that surrogate,
        rather than whatever the process-wide default happens to be.
        TraceWinSimulator has no such attribute, so this falls back to
        param_knn_distance()'s own default-dataset cache in that case.
        """
        if self.distance_penalty_weight <= 0.0:
            return 0.0
        dataset = getattr(
            self,
            "_distance_dataset",
            getattr(self.simulator, "dataset", None),
        )
        distance = param_knn_distance(params_to_vec(params)[None, :], dataset=dataset)[0]
        return self.distance_penalty_weight * float(distance)

    def render(self, save_path: str | Path | None = None, fps: int = 2) -> dict:
        """Render how machine parameters, beam features, and score/reward
        evolved over the whole episode so far (since the last reset()).

        Draws one line panel per machine parameter, one line panel per
        (feature, observed stage) combination, and one panel each for score
        and reward, with every step recorded since the last reset(). A dashed
        line marks the value right after reset() ("start"). The x-axis always
        spans the full episode (0..max_steps) even if only some steps have
        happened so far, so repeated snapshots of the same episode stay on a
        comparable scale. Parameter panels are always blue (a parameter has
        no inherent "good" direction). Beam-feature and score/reward panels
        are colored per segment: each step-to-step move is green if that
        stage's feature improved (or score/reward went up), red if it
        worsened (see _feature_improved) — so a single line can show several
        colors along its path, not one color for the whole episode.

        Args:
            save_path: if None (default), show the figures with all steps
                already drawn, and return them. If given, instead save
                step-by-step animations (one per figure) built frame by
                frame, and return their paths too. A ".gif" path (or no
                extension) is saved with Pillow; a ".mp4" path requires the
                ffmpeg binary to be installed.
            fps: animation frame rate, only used when save_path is given.

        Returns:
            {"params": Figure, "state": Figure, "score": Figure, "knn": Figure,
            "delta": Figure} normally, plus {"params_video": Path,
            "state_video": Path, "score_video": Path, "knn_video": Path,
            "delta_video": Path} when save_path is given.
            The "knn" figure is a dedicated pair of panels showing the
            episode's parameter-space KNN distance to the base dataset next
            to the score trend, so the two can be compared side by side.
            The "delta" figure is one bar per beam feature (final observed
            stage only) plus score, showing final - start over the episode
            so far, green if that change is an improvement (see
            _feature_improved), red if it is a regression, gray if unchanged.
        """
        import matplotlib.pyplot as plt

        state = self.state
        n_frames = len(state.obs_history)
        steps = np.arange(n_frames)
        animate = save_path is not None
        n_init = 1 if animate else n_frames

        # ── Parameters figure: one panel per machine parameter ──────────────
        n_params = len(PARAM_KEYS)
        ncols = 4
        nrows = -(-n_params // ncols)  # ceil division
        params_fig, params_axes = plt.subplots(
            nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows), squeeze=False
        )
        params_fig.suptitle("Parameter value trends over one full episode", fontweight="bold")

        params_updaters: list = []
        for ax, spec in zip(params_axes.ravel(), PARAMETERS):
            values = [float(p[spec.key]) for p in state.params_history]
            ax.axhline(values[0], color="0.4", lw=1, linestyle="--", label="start")
            line, = ax.plot(steps[:n_init], values[:n_init], color="tab:blue", marker="o", markersize=3)
            params_updaters.append(self._line_updater(line, steps, values))
            ax.set_xlim(0, self.max_steps)
            ax.set_ylim(*self._series_ylim(values, pad_factor=0.3))
            ax.set_title(spec.name, fontsize=9)
            ax.set_xlabel("step", fontsize=8)
            ax.set_ylabel("value", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25)
        for ax in params_axes.ravel()[n_params:]:
            ax.set_visible(False)
        params_fig.tight_layout(rect=(0, 0, 1, 0.96))

        # ── Beam-feature figure: one panel per (feature, observed stage) ────
        stage_titles = [f"stage {idx}" for idx in observation_stage_indices()]
        n_stages = len(stage_titles)
        stage_frames = [self._obs_to_stage_frame(obs) for obs in state.obs_history]

        state_fig, state_axes = plt.subplots(
            len(BEAM_STATE_FEATURES), n_stages,
            figsize=(4.2 * n_stages, 2.6 * len(BEAM_STATE_FEATURES)),
            squeeze=False,
        )
        state_fig.suptitle("Beam feature trends over one full episode", fontweight="bold")

        state_updaters: list = []
        for row, feature in enumerate(BEAM_STATE_FEATURES):
            reference = self._STATE_FEATURE_REFERENCE.get(feature)
            for col, stage in enumerate(stage_titles):
                values = [float(df.loc[col, feature]) for df in stage_frames]

                ax = state_axes[row, col]
                ax.axhline(values[0], color="0.4", lw=1, linestyle="--", label="start")
                if reference is not None:
                    ax.axhline(reference, color="tab:blue", lw=1, linestyle=":", label="target")
                lc, points, segments = self._plot_colored_trend(ax, steps, values, n_init, feature=feature)
                state_updaters.append(self._colored_trend_updater(lc, points, segments, steps, values))
                ax.set_xlim(0, self.max_steps)
                ax.set_ylim(*self._series_ylim(values, reference=reference, pad_factor=0.3))
                if row == 0:
                    ax.set_title(stage, fontsize=9)
                if col == 0:
                    ax.set_ylabel(feature, fontsize=9)
                ax.set_xlabel("step", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(alpha=0.25)
        state_fig.tight_layout(rect=(0, 0, 1, 0.97))

        # ── Score/reward figure: two panels, whole episode ──────────────────
        score_fig, score_axes = plt.subplots(1, 2, figsize=(8.4, 3.2), squeeze=False)
        score_fig.suptitle("Score and reward trends over one full episode", fontweight="bold")

        score_updaters: list = []
        for ax, key, values in zip(
            score_axes.ravel(),
            ("score", "reward"),
            (state.score_history, state.reward_history),
        ):
            ax.axhline(values[0], color="0.4", lw=1, linestyle="--", label="start")
            lc, points, segments = self._plot_colored_trend(ax, steps, values, n_init, feature=None)
            score_updaters.append(self._colored_trend_updater(lc, points, segments, steps, values))
            ax.set_xlim(0, self.max_steps)
            ax.set_ylim(*self._series_ylim(values))
            ax.set_title(key, fontsize=10)
            ax.set_xlabel("step", fontsize=8)
            ax.set_ylabel("value", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=6, loc="upper left")
        score_fig.tight_layout(rect=(0, 0, 1, 0.86))

        # ── KNN-distance figure: dedicated pair (param KNN distance, score) ─
        # Computed lazily here (one vectorized k-d tree query over the whole
        # episode) instead of once per training step.
        knn_history = [
            float(v)
            for v in param_knn_distance(
                np.stack([params_to_vec(p) for p in state.params_history]),
                dataset=getattr(
                    self,
                    "_distance_dataset",
                    getattr(self.simulator, "dataset", None),
                ),
            )
        ]
        knn_fig, knn_axes = plt.subplots(1, 2, figsize=(8.4, 3.2), squeeze=False)
        knn_fig.suptitle("Parameter KNN distance vs. score over one full episode", fontweight="bold")

        knn_updaters: list = []
        for ax, key, values, feature in (
            (knn_axes[0, 0], "knn_distance", knn_history, "knn_distance"),
            (knn_axes[0, 1], "score", state.score_history, None),
        ):
            ax.axhline(values[0], color="0.4", lw=1, linestyle="--", label="start")
            lc, points, segments = self._plot_colored_trend(ax, steps, values, n_init, feature=feature)
            knn_updaters.append(self._colored_trend_updater(lc, points, segments, steps, values))
            ax.set_xlim(0, self.max_steps)
            ax.set_ylim(*self._series_ylim(values))
            ax.set_title(key, fontsize=10)
            ax.set_xlabel("step", fontsize=8)
            ax.set_ylabel("value", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=6, loc="upper left")
        knn_fig.tight_layout(rect=(0, 0, 1, 0.86))

        # ── Delta figure: final-stage feature + score change, start vs. now ─
        # One small panel per beam feature (final observed stage only, not
        # every stage — the trend panels above already cover intermediate
        # stages) plus one for score. Each panel has its own y-scale (values
        # span very different ranges: score deltas are tens, npart_ratio
        # deltas are hundredths). For x0/y0/x'0/y'0 (features whose target is
        # 0, see _OFFSET_ANGLE_FEATURES), the bar/label show the change in
        # *distance to zero* (|final| - |start|) instead of the raw signed
        # value change -- otherwise a value that moved from -0.4 to -0.09
        # (a real improvement, closer to 0) would show as "+0.31" and look
        # like a worsening. Colored the same way as the trend lines
        # (_feature_improved), gray if unchanged; for these offset features
        # the sign of the displayed delta and the color always agree
        # (negative = closer to 0 = green).
        final_stage_idx = n_stages - 1
        delta_labels = list(BEAM_STATE_FEATURES) + ["score"]

        def _delta_history(label: str) -> list[float]:
            if label == "score":
                start = state.score_history[0]
                return [float(v) - start for v in state.score_history]
            start = float(stage_frames[0].loc[final_stage_idx, label])
            return [float(df.loc[final_stage_idx, label]) - start for df in stage_frames]

        delta_start_values = {
            label: (
                state.score_history[0] if label == "score"
                else float(stage_frames[0].loc[final_stage_idx, label])
            )
            for label in delta_labels
        }
        delta_histories = {label: _delta_history(label) for label in delta_labels}

        def _display_history(label: str) -> list[float]:
            """The delta actually plotted/labeled -- distance-to-zero change
            for offset features, raw value change otherwise."""
            if label not in self._OFFSET_ANGLE_FEATURES:
                return delta_histories[label]
            start = delta_start_values[label]
            return [
                abs(start + delta) - abs(start) for delta in delta_histories[label]
            ]

        display_histories = {label: _display_history(label) for label in delta_labels}

        ncols = 5
        nrows = -(-len(delta_labels) // ncols)  # ceil division
        delta_fig, delta_axes = plt.subplots(
            nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows), squeeze=False
        )
        delta_fig.suptitle(
            f"Final stage ({stage_titles[final_stage_idx]}) + score: "
            "change from start to now",
            fontweight="bold",
        )

        delta_updaters: list = []
        for ax, label in zip(delta_axes.ravel(), delta_labels):
            history = delta_histories[label]
            display_history = display_histories[label]
            is_offset = label in self._OFFSET_ANGLE_FEATURES
            bar = ax.bar([0], [0.0], width=0.5, color="tab:gray",
                         edgecolor="0.3", linewidth=0.8, zorder=3)[0]
            # xytext is an offset in points, not data units, and the label
            # spans two lines (delta + %) -- a data-fraction pad (like the
            # other trend panels use) would need to scale with font size, not
            # the series range, so the gap to the bar tip is fixed in points
            # instead, and ylim below gets extra headroom so it clears the title.
            text = ax.annotate(
                "", xy=(0, 0), xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=8, zorder=4,
            )
            delta_updaters.append(
                self._delta_bar_updater(
                    bar, text, history, display_history, delta_start_values[label], label,
                )
            )
            ax.axhline(0, color="0.3", lw=1, zorder=2)
            ax.set_xlim(-1, 1)
            ax.set_xticks([])
            lo, hi = self._series_ylim(display_history)
            headroom = 0.4 * max(hi - lo, 1e-6)
            ax.set_ylim(lo - headroom, hi + headroom)
            ax.set_title(label, fontsize=10)
            ax.set_ylabel("Δ|value| vs. start" if is_offset else "Δ vs. start", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25, axis="y", zorder=0)
        for ax in delta_axes.ravel()[len(delta_labels):]:
            ax.set_visible(False)

        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, color=color, label=text)
            for color, text in (
                ("tab:green", "improved"), ("tab:red", "worsened"), ("tab:gray", "unchanged"),
            )
        ]
        delta_fig.legend(handles=legend_handles, fontsize=8, loc="upper right", ncol=3)

        for updater in delta_updaters:
            updater(n_init - 1)
        delta_fig.tight_layout(rect=(0, 0, 1, 0.90))

        if not animate:
            plt.show()
            return {
                "params": params_fig, "state": state_fig, "score": score_fig,
                "knn": knn_fig, "delta": delta_fig,
            }

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = save_path.suffix or ".gif"
        params_path = save_path.with_name(f"{save_path.stem}_params{suffix}")
        state_path = save_path.with_name(f"{save_path.stem}_state{suffix}")
        score_path = save_path.with_name(f"{save_path.stem}_score{suffix}")
        knn_path = save_path.with_name(f"{save_path.stem}_knn{suffix}")
        delta_path = save_path.with_name(f"{save_path.stem}_delta{suffix}")

        self._save_trend_animation(params_fig, params_updaters, n_frames, params_path, fps)
        self._save_trend_animation(state_fig, state_updaters, n_frames, state_path, fps)
        self._save_trend_animation(score_fig, score_updaters, n_frames, score_path, fps)
        self._save_trend_animation(knn_fig, knn_updaters, n_frames, knn_path, fps)
        self._save_trend_animation(delta_fig, delta_updaters, n_frames, delta_path, fps)
        plt.close(params_fig)
        plt.close(state_fig)
        plt.close(score_fig)
        plt.close(knn_fig)
        plt.close(delta_fig)

        return {
            "params": params_fig,
            "state": state_fig,
            "score": score_fig,
            "knn": knn_fig,
            "delta": delta_fig,
            "params_video": params_path,
            "state_video": state_path,
            "score_video": score_path,
            "knn_video": knn_path,
            "delta_video": delta_path,
        }

    @classmethod
    def _segment_colors(cls, values: list[float], feature: str | None = None) -> list[str]:
        """Per-segment green/red/gray color for each consecutive pair in values.

        Gray if the value did not change between the two steps. Otherwise,
        if `feature` is given, uses _feature_improved's per-feature trend
        convention; otherwise (score/reward) higher is always better.
        """
        colors = []
        for before, after in zip(values[:-1], values[1:]):
            if np.isclose(before, after):
                colors.append("tab:gray")
                continue
            improved = (
                cls._feature_improved(feature, before, after)
                if feature is not None
                else after >= before
            )
            colors.append("tab:green" if improved else "tab:red")
        return colors

    @classmethod
    def _plot_colored_trend(cls, ax, steps: np.ndarray, values: list[float],
                             n_init: int, feature: str | None = None):
        """Draw a per-segment colored trend line (green/red per step-to-step
        move) plus neutral markers on ax.

        Returns (LineCollection, markers Line2D, full list of segments) so
        the caller can later grow the line frame by frame for animation.
        """
        from matplotlib.collections import LineCollection

        segments = [
            [(steps[i], values[i]), (steps[i + 1], values[i + 1])]
            for i in range(len(values) - 1)
        ]
        colors = cls._segment_colors(values, feature=feature)

        n_segments_shown = max(0, n_init - 1)
        lc = LineCollection(segments[:n_segments_shown], colors=colors, linewidths=2)
        ax.add_collection(lc)
        points, = ax.plot(
            steps[:n_init], values[:n_init],
            linestyle="None", marker="o", markersize=3, color="0.25",
        )
        return lc, points, segments

    @staticmethod
    def _line_updater(line, steps: np.ndarray, values: list[float]):
        """Animation updater for a single-color Line2D (used by parameters)."""
        def update(frame_idx: int) -> None:
            line.set_data(steps[: frame_idx + 1], values[: frame_idx + 1])
        return update

    @staticmethod
    def _colored_trend_updater(lc, points, segments: list, steps: np.ndarray, values: list[float]):
        """Animation updater for a per-segment colored trend (state/score)."""
        def update(frame_idx: int) -> None:
            lc.set_segments(segments[:frame_idx])
            points.set_data(steps[: frame_idx + 1], values[: frame_idx + 1])
        return update

    @classmethod
    def _delta_bar_updater(
        cls, bar, text, history: list[float], display_history: list[float],
        start_value: float, label: str,
    ):
        """Animation updater for one panel of the delta bar chart.

        `history[frame_idx]` is `value_at_frame_idx - value_at_reset` (see
        render()'s _delta_history()) and always drives the green/red/gray
        color, the same way trend segments are colored (_feature_improved) --
        this needs the raw signed value to tell direction from a reference.
        `display_history[frame_idx]` is what's actually drawn as the bar
        height and the text label: for most features it's identical to
        `history`, but for x0/y0/x'0/y'0 (see render()'s _display_history())
        it's the *distance-to-zero* change instead, so the number on screen
        never disagrees with the color (e.g. a bar that moved closer to 0
        shows a negative number and is green, not a positive "raw" delta
        that would look like a regression).
        """
        feature = None if label == "score" else label

        def update(frame_idx: int) -> None:
            delta = history[frame_idx]
            display_value = display_history[frame_idx]
            bar.set_height(display_value)
            if np.isclose(delta, 0.0):
                color = "tab:gray"
            else:
                current_value = start_value + delta
                improved = (
                    cls._feature_improved(feature, start_value, current_value)
                    if feature is not None
                    else current_value >= start_value
                )
                color = "tab:green" if improved else "tab:red"
            bar.set_color(color)

            text.set_text(f"{display_value:+.3g}")
            # `xy` anchors to the bar tip in data coordinates; the label
            # itself sits a fixed number of points above/below that anchor
            # (set at creation via textcoords="offset points"), so the gap
            # never shrinks to nothing regardless of the panel's data range.
            text.xy = (0, display_value)
            text.set_position((0, 6 if display_value >= 0 else -6))
            text.set_va("bottom" if display_value >= 0 else "top")
        return update

    @staticmethod
    def _save_trend_animation(fig, updaters: list, n_frames: int, path: Path, fps: int) -> None:
        """Animate one render() trend figure step by step and save it."""
        from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

        def update(frame_idx):
            for updater in updaters:
                updater(frame_idx)
            return []

        anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
        writer = FFMpegWriter(fps=fps) if path.suffix.lower() == ".mp4" else PillowWriter(fps=fps)
        try:
            anim.save(str(path), writer=writer)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not save {path}: the ffmpeg binary was not found on PATH. "
                "Install ffmpeg, or save to a .gif path instead (uses Pillow)."
            ) from exc


    def evaluate_params(self, params: Dict[str, float]) -> float:
        """Evaluate a fixed parameter set without stepping the episode.
        Returns the final score of a specific simulation."""

        result = self.simulator.simulate(params)
        _, score, _ = self._result_to_obs_score_info(result)
        return score

    
    @abstractmethod
    def _build_simulator(self) -> BeamSimulator:
        """Return the simulator for the environment to perform actions.
        BaseBeamEnv define the common Gymnasium loop; while subclasses decide which BeamSimulator implementation powers that loop.
        """
        raise NotImplementedError


    # Observation helpers for reset() and step()
    # -------------------------------------------------------------------------
    def _result_to_obs_score_info(self, result: BeamSimulationResult) -> tuple[np.ndarray, float, dict]:
        """Convert a BeamSimulationResult into (obs, score, info_extras)."""
        
        # A missing trajectory is a technical failure. Physical beam losses
        # retain all available stages and therefore remain observable.
        if result.beam_states is None:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            failure_reason = self._terminal_failure_reason(result)
            return obs, ERROR_SCORE, {
                "sim_result": result,
                "low_transmission": failure_reason is not None,
                "terminal_failure": failure_reason is not None,
                "failure_reason": failure_reason,
            }

        # Select the beam stages configured in adige.py for the Gym observation.
        obs = select_observation_stages(result.beam_states)
        failure_reason = self._terminal_failure_reason(result)
        return obs, result.score_val, {
            "sim_result": result,
            "low_transmission": failure_reason is not None,
            "terminal_failure": failure_reason is not None,
            "failure_reason": failure_reason,
        }

    @staticmethod
    def _is_physics_failure(result: BeamSimulationResult) -> bool:
        return bool(
            (result.metadata or {}).get("physics_failure")
            or canonical_physics_failure_reason(result.error)
        )

    @classmethod
    def _terminal_failure_reason(
        cls,
        result: BeamSimulationResult,
    ) -> str | None:
        metadata = result.metadata or {}
        if cls._is_physics_failure(result):
            return str(
                metadata.get("physics_failure_reason")
                or canonical_physics_failure_reason(result.error)
                or "low_transmission"
            )
        npart_index = BEAM_STATE_FEATURES.index("npart_ratio")
        npart_ratio: object | None = None
        if result.final_beam is not None:
            npart_ratio = result.final_beam["npart_ratio"]
        elif result.beam_states is not None:
            final_beam = np.asarray(result.beam_states[-1])
            if final_beam.size > npart_index:
                npart_ratio = final_beam[npart_index]
        if npart_ratio is not None and below_rl_min_npart_ratio(npart_ratio):
            return "low_transmission"
        if result.score_val == ERROR_SCORE:
            return "low_transmission"
        return None

    @classmethod
    def _is_technical_failure(cls, result: BeamSimulationResult) -> bool:
        if cls._is_physics_failure(result):
            return False
        return (
            result.beam_states is None
            or not result.success
        )

    
    # Render helpers
    # -------------------------------------------------------------------------
    def _obs_to_stage_frame(self, obs: np.ndarray):
        """Convert a Gym observation array into a DataFrame with columns 
        ["stage", *BEAM_STATE_FEATURES] = ["stage",  "npart_ratio", "x0", "y0", "SizeX", "SizeY", "ex", "ey", "x'0", "y'0"].
        It is used for rendering the before/after values of each beam feature for each stage.
        """
        import pandas as pd

        obs = np.asarray(obs, dtype=np.float32)
        # Only the beam prefix is rendered here. The normalized parameter
        # suffix has its own physical-value figure above.
        labels = observation_stage_labels()
        beam_obs_dim = len(labels) * BEAM_STATE_DIM
        arr = obs[:beam_obs_dim].reshape(len(labels), BEAM_STATE_DIM)
        df = pd.DataFrame(arr, columns=BEAM_STATE_FEATURES)
        df.insert(0, "stage", labels)
        return df

    _OFFSET_ANGLE_FEATURES = frozenset({"x0", "y0", "x'0", "y'0"})

    @classmethod
    def _feature_improved(cls, feature: str, before: float, after: float) -> bool:
        """Return True if the feature value's trend from before to after is good.

        - npart_ratio: maximized (green when it goes up).
        - x0/y0/x'0/y'0 (reference 0): trend toward zero — green when the
          distance to zero shrinks (|after| < |before|), regardless of sign.
        - anything else (ex/ey, SizeX/SizeY, knn_distance, ...): minimized
          (green when it goes down, red when it goes up).
        """
        if feature == "npart_ratio":
            return round(float(after), 3) >= round(float(before), 3)
        if feature in cls._OFFSET_ANGLE_FEATURES:
            return abs(float(after)) < abs(float(before))
        return after < before

    # Target value shown as a reference line on each state panel, so a trend
    # can be read as "is it heading toward its goal" instead of just "is it
    # moving". Reuses SCORE_REFERENCES (adige.py); npart_ratio has no entry
    # there (score() rewards its raw value, not a distance from a reference)
    # but its goal is unambiguously full transmission, so it is added here
    # for the render only.
    _STATE_FEATURE_REFERENCE: Dict[str, float] = {**SCORE_REFERENCES, "npart_ratio": 1.0}

    @staticmethod
    def _series_ylim(
        values, reference: float | None = None, pad_factor: float = 0.12
    ) -> tuple[float, float]:
        """Y-axis limits for a full-episode line trend.

        Does not force zero into the range: a parameter or feature that
        never crosses zero (e.g. always negative) should not have its axis
        padded down to 0. `reference`, when given, is always included in the
        range (padded like any other point) so the target value stays
        visible even if the episode never gets close to it. `pad_factor`
        controls how much headroom is added above/below the data range (as a
        fraction of that range) -- the params/state panels use a larger
        value than the default so late-episode oscillations read as small
        relative to the full climb, instead of filling the whole axis.
        """
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if reference is not None and np.isfinite(reference):
            values = np.append(values, float(reference))
        if values.size == 0:
            return -1.0, 1.0
        lo, hi = float(np.min(values)), float(np.max(values))
        if np.isclose(lo, hi):
            pad = max(abs(hi) * 0.1, 1e-6)
            return lo - pad, hi + pad
        pad = pad_factor * (hi - lo)
        return lo - pad, hi + pad
