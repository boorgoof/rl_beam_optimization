"""
It is a base class for two beam optimization environments: TraceWinEnv and SurrogateEnv.

Both environments perform the same Gym loop (reset, step, render) but they differ 
in the backend simulator that produces the beam states and score for a given set of machine parameters.

"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from beam_optimization.config.adige import (
    PARAM_KEYS, BEAM_STATE_DIM,
    BEAM_STATE_FEATURES, default_params, action_bounds, reset_std_vec,
    observation_dim, observation_stage_labels, select_observation_stages,
    clip_params_to_hw,
)

from beam_optimization.env.simulation import BeamSimulationResult, BeamSimulator

ERROR_SCORE = -99.0


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
        max_steps: int = 20,
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
        self._previous_obs   = None
        self._last_action    = None
        self._last_reward    = 0.0
        self.best_score      = ERROR_SCORE
        self.best_params     = default_params()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0

        # Sample random initial params and perturb them with per-parameter reset stddevs.
        params = default_params()
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
        self._current_obs   = obs
        self._current_score = score
        self._previous_obs  = None
        self._last_action   = None
        self._last_reward   = 0.0

        info = {"score": score, "step": 0, **extra}
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

        # compute reward 
        reward = score - prev_score

        # update episode state
        self._current_obs   = obs
        self._current_score = score
        self._previous_obs  = prev_obs
        self._last_action   = action.copy()
        self._last_reward   = float(reward)

        # update best score and best parameters if the current score is better than the best score so far.
        if score > self.best_score:
            self.best_score  = score
            self.best_params = self._current_params.copy()

        # update step count
        self._step_count += 1
        truncated = self._step_count >= self.max_steps

        info = {"score": score, "prev_score": prev_score, "step": self._step_count,
                "best_score": self.best_score, **extra}
        
        # return the observation next state, reward, terminated with success flag (False), truncated flag, and info dictionary.
        return obs.copy(), reward, False, truncated, info


    def render(self, mode: str = "human"):
        """Render the current observation as graphical beam-feature bars ().

        It visualizes a graphical representation of the current observation,
        showing the before and after values of each beam feature (npart_ratio, x0, y0, SizeX, SizeY, ex, ey, x'0, y'0) for each stage, 
        along with the delta compared to the previous observation. 
        
        The number of stages shown is determined by OBSERVATION_STAGE_MASK.  
        
        If called after ``step()``, the previous observation is shown as dashed reference segments; 
        if called after ``reset()``, the reference is the current observation itself.
        """
        if mode != "human":
            raise NotImplementedError(f"Only render(mode='human') is supported, got {mode!r}")

        import matplotlib.pyplot as plt

        # Convert current and previous observations to DataFrames
        # (rows = stages, columns = beam features)
        current = self._obs_to_stage_frame(self._current_obs)
    
        previous_obs = self._previous_obs if self._previous_obs is not None else self._current_obs # If no previous obs exists (e.g. just after reset), use current as reference
        previous = self._obs_to_stage_frame(previous_obs)

        # Build a flat list of dicts, one per (stage, feature) combination,
        # storing before/after values, the delta, and the trend direction
        rows = []
        for row_idx, stage in enumerate(current["stage"]):
            for feature in BEAM_STATE_FEATURES:
                before = float(previous.loc[row_idx, feature])
                after = float(current.loc[row_idx, feature])
                rows.append(
                    {
                        "stage": stage,
                        "feature": feature,
                        "before": before,
                        "after": after,
                        "delta": after - before,
                        "trend": self._feature_trend(feature, before, after),
                    }
                )

        # Create a 3×3 grid of subplots, one panel per beam feature
        fig, axes = plt.subplots(3, 3, figsize=(16, 10.2))
        # Title bar with environment name, obs mode, step count, score, and last reward
        fig.suptitle(
            f"{type(self).__name__} render | observation_stages={observation_stage_labels()} | "
            f"step={self._step_count} | score={self._current_score:.6g} | "
            f"last reward={self._last_reward:.6g}",
            fontsize=13,
        )

        # Fill each subplot with the bar chart for its feature
        for ax, feature in zip(axes.ravel(), BEAM_STATE_FEATURES):
            feature_rows = [row for row in rows if row["feature"] == feature]
            self._plot_render_feature_panel(ax, feature, feature_rows)

        plt.tight_layout()  # Prevent overlapping labels between subplots
        plt.show()
        return fig 


    
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
        We want npart_ratio to increase, SizeX/SizeY/ex/ey to decrease, and x0/y0/x'0/y'0 to move closer to zero.
        """
        if feature == "npart_ratio":
            return round(float(after), 3) >= round(float(before), 3)
        if feature in {"SizeX", "SizeY", "ex", "ey"}:
            return after < before
        if feature in {"x0", "y0", "x'0", "y'0"}:
            return abs(after) < abs(before)
        return after >= before

    @classmethod
    def _feature_trend(cls, feature: str, before: float, after: float) -> str:
        """Return 'improved', 'worse', or 'same' based on the before and after values of a feature. """
        
        if feature == "npart_ratio":
            return "improved" if cls._feature_improved(feature, before, after) else "worse"
        if np.isclose(before, after):
            return "same"
        return "improved" if cls._feature_improved(feature, before, after) else "worse"

    @staticmethod
    def _feature_ylim(before_values, after_values) -> tuple[float, float]:
        """
        Compute the y-axis limits (min, max) for a feature panel based on the before and after values of the feature.
        It is used in _plot_render_feature_panel() to set the y-axis limits for each feature panel in the render() visualization.
        """
        values = np.asarray(list(before_values) + list(after_values), dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return -1.0, 1.0
        lo = min(0.0, float(np.nanmin(values)))
        hi = max(0.0, float(np.nanmax(values)))
        if np.isclose(lo, hi):
            pad = max(abs(hi) * 0.1, 1e-6)
            return lo - pad, hi + pad
        pad = 0.28 * (hi - lo)
        return lo - pad, hi + pad

    @classmethod
    def _plot_render_feature_panel(cls, ax, feature: str, feature_rows: list[dict]) -> None:
        """it is used in render() to plot a single feature panel, showing before/after values and delta for each stage."""
        
        stages = [str(row["stage"]) for row in feature_rows]
        before = np.asarray([row["before"] for row in feature_rows], dtype=float)
        after = np.asarray([row["after"] for row in feature_rows], dtype=float)
        delta = np.asarray([row["delta"] for row in feature_rows], dtype=float)
        colors = [
            "tab:green" if row["trend"] == "improved"
            else "tab:red" if row["trend"] == "worse"
            else "tab:gray"
            for row in feature_rows
        ]

        x = np.arange(len(stages))
        width = 0.68 if len(stages) <= 3 else 0.78
        ax.bar(x, after, width=width, color=colors, alpha=0.82)
        ax.axhline(0, color="0.2", lw=0.8)
        ax.hlines(
            before,
            x - width / 2,
            x + width / 2,
            color="0.15",
            lw=1.6,
            linestyle=(0, (5, 3)),
            alpha=0.85,
        )
        ax.set_ylim(*cls._feature_ylim(before, after))
        ax.set_xticks(x)
        ax.set_xticklabels(stages, fontsize=7, rotation=90 if len(stages) > 4 else 0)
        ax.set_title(feature, fontsize=10)
        ax.grid(axis="y", alpha=0.24)
        ax.tick_params(axis="y", labelsize=8)

        y0, y1 = ax.get_ylim()
        yrange = y1 - y0
        top_pad = 0.035 * yrange
        delta_y = y0 + 0.08 * yrange
        for xi, before_i, after_i, delta_i, color_i in zip(x, before, after, delta, colors):
            va = "bottom" if after_i >= 0 else "top"
            ax.text(
                xi,
                after_i + (top_pad if after_i >= 0 else -top_pad),
                f"{after_i:.3g}",
                ha="center",
                va=va,
                fontsize=7 if len(stages) > 4 else 8,
            )
            if len(stages) <= 3:
                ax.text(
                    xi + width * 0.44,
                    before_i,
                    f"before {before_i:.3g}",
                    ha="left",
                    va="center",
                    fontsize=7,
                    color="0.2",
                )
            ax.text(
                xi,
                delta_y,
                f"D {delta_i:+.2g}",
                ha="center",
                va="bottom",
                fontsize=6 if len(stages) > 4 else 8,
                color=color_i,
                fontweight="bold",
                rotation=90 if len(stages) > 4 else 0,
            )
