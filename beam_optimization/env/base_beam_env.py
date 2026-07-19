"""
It is a base class for two beam optimization environments: TraceWinEnv and SurrogateEnv.

Both environments perform the same Gym loop (reset, step, render) but they differ 
in the backend simulator that produces the beam states and score for a given set of machine parameters.

"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from beam_optimization.config.adige import (
    ERROR_SCORE, FAILURE_PENALTY, MAX_STEPS, PARAM_KEYS, PARAMETERS, BEAM_STATE_DIM,
    BEAM_STATE_FEATURES, default_params, action_bounds, reset_std_vec,
    observation_dim, observation_stage_labels, observation_stage_indices,
    select_observation_stages, clip_params_to_hw, params_to_vec,
)

from beam_optimization.env.dataset.dataset import param_knn_distance
from beam_optimization.env.simulation import BeamSimulationResult, BeamSimulator


class BaseBeamEnv(gym.Env, ABC):
    """Common reset/step/evaluate_params scaffolding for the two beam envs.

    Args:
        max_steps:    Episode length.
        Observation stages are selected by OBSERVATION_STAGE_MASK in adige.py.
    """

    metadata = {"render_modes": ["human"]}

    # Construction: simulator, Gym spaces(observations and actions), and episode state (step count, current params, current obs, current score, previous obs, last action, last reward, best score, best params).
    # -------------------------------------------------------------------------
    def __init__(
        self,
        max_steps: int = MAX_STEPS,
    ):
        super().__init__()

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

        # Per-parameter reset stddevs from ParameterSpec.
        self._reset_std = reset_std_vec().astype(np.float32)

        # Episode state
        self._step_count     = 0
        self._current_params = default_params()
        self._current_obs    = np.zeros(obs_dim, dtype=np.float32)
        self._current_score  = ERROR_SCORE
        self._current_result: BeamSimulationResult | None = None
        self._previous_obs   = None
        self._last_action    = None
        self._last_reward    = 0.0
        self.best_score      = ERROR_SCORE
        self.best_params     = default_params()

        # Per-episode history for render(): index 0 is the state right
        # after reset(), index k is the state after the k-th step().
        # KNN distances are derived from _params_history lazily in render(),
        # so training steps never pay the k-d tree query (nor force the
        # default dataset file to exist).
        self._params_history: list[dict] = []
        self._obs_history: list[np.ndarray] = []
        self._score_history: list[float] = []
        self._reward_history: list[float] = []


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0

        # Sample random initial params and perturb them with per-parameter reset stddevs.
        # Tests against the real TraceWin backend can disable this through
        # options={"randomize_params": False} to start from the nominal machine.
        randomize_params = True
        if options is not None and "randomize_params" in options:
            randomize_params = bool(options["randomize_params"])

        params = default_params()
        if randomize_params:
            for key, std in zip(PARAM_KEYS, self._reset_std):
                params[key] += float(self.np_random.normal(0.0, std))
        params = clip_params_to_hw(params)
        self._current_params = params

        # Let the simulator prepare its context to start the episode. 
        # For the surrogate, it samples beam0 and chooses the active ensemble member.
        self.simulator.reset_context(self.np_random)

        # Run the simulator with the initial parameters.
        result = self.simulator.simulate(params)

        obs, score, extra = self._result_to_obs_score_info(result)
        self._current_obs    = obs
        self._current_score  = score
        self._current_result = result
        self._previous_obs   = None
        self._last_action    = None
        self._last_reward   = 0.0

        self._params_history = [dict(self._current_params)]
        self._obs_history = [obs.copy()]
        self._score_history = [float(score)]
        self._reward_history = [0.0]

        info = {"score": score, "step": 0, "reset_randomized": randomize_params, **extra}
        return obs.copy(), info

    def step(self, action: np.ndarray):

        # action to perform. It is a delta to apply to the current parameters. 
        # The action is clipped to the action space bounds.
        action = np.clip(action, self.action_space.low, self.action_space.high)
        prev_obs = self._current_obs.copy()

        # modify parameter with deltas to the current parameters. 
        for key, delta in zip(PARAM_KEYS, action):
            self._current_params[key] = float(self._current_params[key]) + float(delta)
        self._current_params = clip_params_to_hw(self._current_params)

        # perform concretely the action (the simulation) and get the new observation and final score.
        prev_score = self._current_score
        result = self.simulator.simulate(self._current_params)
        obs, score, extra = self._result_to_obs_score_info(result)

        # compute reward. A failed simulation must not enter the delta-score
        # reward as ERROR_SCORE (-999): that would produce a ~-1000 spike now
        # and a ~+1000 spike on the next successful step, both of which would
        # dominate any training batch. Instead the episode ends here
        # (terminated=True: the machine state is invalid, there is no next
        # state to bootstrap from) with a bounded penalty.
        failed = not result.success
        reward = -FAILURE_PENALTY if failed else score - prev_score

        # update episode state
        self._current_obs    = obs
        self._current_score  = score
        self._current_result = result
        self._previous_obs   = prev_obs
        self._last_action    = action.copy()
        self._last_reward    = float(reward)

        self._params_history.append(dict(self._current_params))
        self._obs_history.append(obs.copy())
        self._score_history.append(float(score))
        self._reward_history.append(float(reward))

        # update best score and best parameters if the current score is better than the best score so far.
        if score > self.best_score:
            self.best_score  = score
            self.best_params = self._current_params.copy()

        # update step count
        self._step_count += 1
        terminated = failed
        truncated = not terminated and self._step_count >= self.max_steps

        info = {"score": score, "prev_score": prev_score, "step": self._step_count,
                "best_score": self.best_score, **extra}

        return obs.copy(), reward, terminated, truncated, info


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
            {"params": Figure, "state": Figure, "score": Figure, "knn": Figure}
            normally, plus {"params_video": Path, "state_video": Path,
            "score_video": Path, "knn_video": Path} when save_path is given.
            The "knn" figure is a dedicated pair of panels showing the
            episode's parameter-space KNN distance to the base dataset next
            to the score trend, so the two can be compared side by side.
        """
        import matplotlib.pyplot as plt

        n_frames = len(self._obs_history)
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
            values = [float(p[spec.key]) for p in self._params_history]
            ax.axhline(values[0], color="0.4", lw=1, linestyle="--", label="start")
            line, = ax.plot(steps[:n_init], values[:n_init], color="tab:blue", marker="o", markersize=3)
            params_updaters.append(self._line_updater(line, steps, values))
            ax.set_xlim(0, self.max_steps)
            ax.set_ylim(*self._series_ylim(values))
            ax.set_title(spec.name, fontsize=9)
            ax.set_xlabel("step", fontsize=8)
            ax.set_ylabel("value", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=6, loc="upper left")
        for ax in params_axes.ravel()[n_params:]:
            ax.set_visible(False)
        params_fig.tight_layout(rect=(0, 0, 1, 0.96))

        # ── Beam-feature figure: one panel per (feature, observed stage) ────
        stage_titles = [f"stage {idx}" for idx in observation_stage_indices()]
        n_stages = len(stage_titles)
        stage_frames = [self._obs_to_stage_frame(obs) for obs in self._obs_history]

        state_fig, state_axes = plt.subplots(
            len(BEAM_STATE_FEATURES), n_stages,
            figsize=(4.2 * n_stages, 2.6 * len(BEAM_STATE_FEATURES)),
            squeeze=False,
        )
        state_fig.suptitle("Beam feature trends over one full episode", fontweight="bold")

        state_updaters: list = []
        for row, feature in enumerate(BEAM_STATE_FEATURES):
            for col, stage in enumerate(stage_titles):
                values = [float(df.loc[col, feature]) for df in stage_frames]

                ax = state_axes[row, col]
                ax.axhline(values[0], color="0.4", lw=1, linestyle="--", label="start")
                lc, points, segments = self._plot_colored_trend(ax, steps, values, n_init, feature=feature)
                state_updaters.append(self._colored_trend_updater(lc, points, segments, steps, values))
                ax.set_xlim(0, self.max_steps)
                ax.set_ylim(*self._series_ylim(values))
                if row == 0:
                    ax.set_title(stage, fontsize=9)
                if col == 0:
                    ax.set_ylabel(feature, fontsize=9)
                ax.set_xlabel("step", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(alpha=0.25)
                ax.legend(fontsize=6, loc="upper left")
        state_fig.tight_layout(rect=(0, 0, 1, 0.97))

        # ── Score/reward figure: two panels, whole episode ──────────────────
        score_fig, score_axes = plt.subplots(1, 2, figsize=(8.4, 3.2), squeeze=False)
        score_fig.suptitle("Score and reward trends over one full episode", fontweight="bold")

        score_updaters: list = []
        for ax, key, values in zip(
            score_axes.ravel(),
            ("score", "reward"),
            (self._score_history, self._reward_history),
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
                np.stack([params_to_vec(p) for p in self._params_history])
            )
        ]
        knn_fig, knn_axes = plt.subplots(1, 2, figsize=(8.4, 3.2), squeeze=False)
        knn_fig.suptitle("Parameter KNN distance vs. score over one full episode", fontweight="bold")

        knn_updaters: list = []
        for ax, key, values, feature in (
            (knn_axes[0, 0], "knn_distance", knn_history, "knn_distance"),
            (knn_axes[0, 1], "score", self._score_history, None),
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

        if not animate:
            plt.show()
            return {"params": params_fig, "state": state_fig, "score": score_fig, "knn": knn_fig}

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = save_path.suffix or ".gif"
        params_path = save_path.with_name(f"{save_path.stem}_params{suffix}")
        state_path = save_path.with_name(f"{save_path.stem}_state{suffix}")
        score_path = save_path.with_name(f"{save_path.stem}_score{suffix}")
        knn_path = save_path.with_name(f"{save_path.stem}_knn{suffix}")

        self._save_trend_animation(params_fig, params_updaters, n_frames, params_path, fps)
        self._save_trend_animation(state_fig, state_updaters, n_frames, state_path, fps)
        self._save_trend_animation(score_fig, score_updaters, n_frames, score_path, fps)
        self._save_trend_animation(knn_fig, knn_updaters, n_frames, knn_path, fps)
        plt.close(params_fig)
        plt.close(state_fig)
        plt.close(score_fig)
        plt.close(knn_fig)

        return {
            "params": params_fig,
            "state": state_fig,
            "score": score_fig,
            "knn": knn_fig,
            "params_video": params_path,
            "state_video": state_path,
            "score_video": score_path,
            "knn_video": knn_path,
        }

    @classmethod
    def _segment_colors(cls, values: list[float], feature: str | None = None) -> list[str]:
        """Per-segment green/red/gray color for each consecutive pair in values.

        Gray if the value did not change between the two steps. Otherwise,
        if `feature` is given, reuses _feature_improved's "minimize
        everything except npart_ratio" convention; otherwise (score/reward)
        higher is always better.
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
        
        # If the simulation failed, return a zero observation and the error score.
        if not result.success or result.beam_states is None:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, ERROR_SCORE, {"sim_result": result}

        # Select the beam stages configured in adige.py for the Gym observation.
        obs = select_observation_stages(result.beam_states)
        return obs, result.score_val, {"sim_result": result}

    
    # Render helpers
    # -------------------------------------------------------------------------
    def _obs_to_stage_frame(self, obs: np.ndarray):
        """Convert a Gym observation array into a DataFrame with columns 
        ["stage", *BEAM_STATE_FEATURES] = ["stage",  "npart_ratio", "x0", "y0", "SizeX", "SizeY", "ex", "ey", "x'0", "y'0"].
        It is used for rendering the before/after values of each beam feature for each stage.
        """
        import pandas as pd

        obs = np.asarray(obs, dtype=np.float32)
        # Convert the observation into a DataFrame with columns  ["stage", *BEAM_STATE_FEATURES] = ["stage",  "npart_ratio", "x0", "y0", "SizeX", "SizeY", "ex", "ey", "x'0", "y'0"].
        labels = observation_stage_labels()
        arr = obs.reshape(len(labels), BEAM_STATE_DIM)
        df = pd.DataFrame(arr, columns=BEAM_STATE_FEATURES)
        df.insert(0, "stage", labels)
        return df

    @staticmethod
    def _feature_improved(feature: str, before: float, after: float) -> bool:
        """Return True if the feature value improved from before to after, False otherwise.
        Every feature is minimized (green when it goes down, red when it goes up),
        except npart_ratio which is maximized (green when it goes up).
        """
        if feature == "npart_ratio":
            return round(float(after), 3) >= round(float(before), 3)
        return after < before

    @staticmethod
    def _series_ylim(values) -> tuple[float, float]:
        """Y-axis limits for a full-episode line trend.

        Does not force zero into the range: a parameter or feature that
        never crosses zero (e.g. always negative) should not have its axis
        padded down to 0.
        """
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return -1.0, 1.0
        lo, hi = float(np.min(values)), float(np.max(values))
        if np.isclose(lo, hi):
            pad = max(abs(hi) * 0.1, 1e-6)
            return lo - pad, hi + pad
        pad = 0.12 * (hi - lo)
        return lo - pad, hi + pad
