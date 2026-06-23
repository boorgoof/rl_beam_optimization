"""
BaseBeamEnv — Gymnasium scaffolding shared by TraceWinEnv and SurrogateEnv.

Both environments are "the same loop, a different simulator": same observation/
action spaces, same parameter-delta episode design, same reward (score delta),
same truncation logic.  Subclasses only install `self.simulator`, which must
implement BeamSimulator.

Subclass contract:
    - Set `self.simulator` before calling `super().__init__(...)`.
    - Optionally override `_on_episode_start()` for extra per-episode setup.
"""
from __future__ import annotations

from abc import ABC
from typing import Dict

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from beam_optimization.config.adige import (
    PARAM_KEYS, BEAM_STATE_DIM, N_BEAM_STATE_STAGES,
    BEAM_STATE_VARS, default_params, action_bounds, sensitivity_vec,
)
from beam_optimization.env.simulation import BeamSimulationResult

ERROR_SCORE = -99.0


class BaseBeamEnv(gym.Env, ABC):
    """Common reset/step/evaluate_params scaffolding for the two beam envs.

    Args:
        action_scale: Multiplier on sensitivity for action bounds.
        max_steps:    Episode length.
        sigma_factor: Gaussian noise scale (× sensitivity) for initial parameters.
        obs_mode:     'full' (108), 'final' (9), or 'final_with_beam0' (18).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        action_scale: float = 1.0,
        max_steps: int = 20,
        sigma_factor: float = 0.5,
        obs_mode: str = "full",
    ):
        super().__init__()

        self.max_steps    = int(max_steps)
        self.sigma_factor = float(sigma_factor)
        self.obs_mode     = obs_mode

        if obs_mode == "full":
            obs_dim = N_BEAM_STATE_STAGES * BEAM_STATE_DIM  # 108
        elif obs_mode == "final":
            obs_dim = BEAM_STATE_DIM                         # 9
        elif obs_mode == "final_with_beam0":
            obs_dim = 2 * BEAM_STATE_DIM                     # 18
        else:
            raise ValueError(f"obs_mode must be 'full', 'final', or 'final_with_beam0', got {obs_mode!r}")
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        act_low, act_high = action_bounds(action_scale)
        self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)

        self._sens = sensitivity_vec().astype(np.float32)

        # Episode state
        self._step_count     = 0
        self._current_params = default_params()
        self._current_obs    = np.zeros(obs_dim, dtype=np.float32)
        self._current_score  = ERROR_SCORE
        self._previous_obs   = None
        self._last_action    = None
        self._last_reward    = 0.0
        self.best_score       = ERROR_SCORE
        self.best_params      = default_params()

    # ── Gym interface ──────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0

        # Sample random initial params
        params = default_params()
        for key, sens in zip(PARAM_KEYS, self._sens):
            params[key] += float(self.np_random.normal(0.0, self.sigma_factor * sens))
        self._current_params = params

        self._on_episode_start()
        self.simulator.reset_context(self.np_random)

        result = self.simulator.simulate(params)
        obs, sc, extra = self._result_to_obs_score_info(result)
        self._current_obs   = obs
        self._current_score = sc
        self._previous_obs  = None
        self._last_action   = None
        self._last_reward   = 0.0

        info = {"score": sc, "step": 0, **extra}
        return obs.copy(), info

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        prev_obs = self._current_obs.copy()

        # Apply delta
        for key, delta in zip(PARAM_KEYS, action):
            self._current_params[key] = float(self._current_params[key]) + float(delta)

        prev_score = self._current_score
        result = self.simulator.simulate(self._current_params)
        obs, sc, extra = self._result_to_obs_score_info(result)

        reward = sc - prev_score
        self._current_obs   = obs
        self._current_score = sc
        self._previous_obs  = prev_obs
        self._last_action   = action.copy()
        self._last_reward   = float(reward)

        if sc > self.best_score:
            self.best_score  = sc
            self.best_params = self._current_params.copy()

        self._step_count += 1
        truncated = self._step_count >= self.max_steps

        info = {"score": sc, "prev_score": prev_score, "step": self._step_count,
                "best_score": self.best_score, **extra}
        return obs.copy(), reward, False, truncated, info

    def render(self, mode: str = "human"):
        """Render the current observation as beam-feature bars.

        The render intentionally does not draw a synthetic particle cloud.  It
        only visualizes the features actually exposed by the selected
        ``obs_mode``.  If called after ``step()``, the previous observation is
        shown as dashed reference segments; after ``reset()``, the reference is
        the current observation itself.
        """
        if mode != "human":
            raise NotImplementedError(f"Only render(mode='human') is supported, got {mode!r}")

        import matplotlib.pyplot as plt

        current = self._obs_to_stage_frame(self._current_obs)
        previous_obs = self._previous_obs if self._previous_obs is not None else self._current_obs
        previous = self._obs_to_stage_frame(previous_obs)

        rows = []
        for row_idx, stage in enumerate(current["stage"]):
            for feature in BEAM_STATE_VARS:
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

        fig, axes = plt.subplots(3, 3, figsize=(16, 10.2))
        fig.suptitle(
            f"{type(self).__name__} render | obs_mode={self.obs_mode!r} | "
            f"step={self._step_count} | score={self._current_score:.6g} | "
            f"last reward={self._last_reward:.6g}",
            fontsize=13,
        )

        for ax, feature in zip(axes.ravel(), BEAM_STATE_VARS):
            feature_rows = [row for row in rows if row["feature"] == feature]
            self._plot_render_feature_panel(ax, feature, feature_rows)

        plt.tight_layout()
        plt.show()
        return fig

    # ── Convenience ───────────────────────────────────────────────────────────

    def evaluate_params(self, params: Dict[str, float]) -> float:
        """Evaluate a fixed parameter set without stepping the episode."""
        result = self.simulator.simulate(params)
        _, sc, _ = self._result_to_obs_score_info(result)
        return sc

    # ── Hooks for subclasses ──────────────────────────────────────────────────

    def _on_episode_start(self) -> None:
        """Optional per-episode setup beyond parameter perturbation.

        No-op by default.  Most per-episode simulator setup should live in
        `self.simulator.reset_context()`.
        """
        pass

    def _result_to_obs_score_info(
        self,
        result: BeamSimulationResult,
    ) -> tuple[np.ndarray, float, dict]:
        """Convert a BeamSimulationResult into Gym obs, score and info extras."""
        if not result.success or result.beam_states is None:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, ERROR_SCORE, {"sim_result": result}

        obs = self._slice_obs(result.beam_states, self.obs_mode)
        return obs, result.score_val, {"sim_result": result}

    @staticmethod
    def _slice_obs(stages, obs_mode: str) -> np.ndarray:
        """Slice a (12, 9) beam-states array/list according to obs_mode.

        Shared by both simulators. The only difference is how `stages` is
        produced (TraceWin output vs surrogate forward pass), not how it is
        sliced into an observation.
        """
        if obs_mode == "full":
            return np.asarray(stages, dtype=np.float32).flatten()
        elif obs_mode == "final":
            return np.asarray(stages[-1], dtype=np.float32)
        else:  # "final_with_beam0"
            return np.concatenate([stages[0], stages[-1]]).astype(np.float32)

    def _obs_to_stage_frame(self, obs: np.ndarray):
        import pandas as pd

        obs = np.asarray(obs, dtype=np.float32)
        if self.obs_mode == "full":
            df = pd.DataFrame(obs.reshape(-1, BEAM_STATE_DIM), columns=BEAM_STATE_VARS)
            df.insert(0, "stage", np.arange(len(df)))
            return df
        if self.obs_mode == "final":
            df = pd.DataFrame([obs], columns=BEAM_STATE_VARS)
            df.insert(0, "stage", ["final"])
            return df

        arr = obs.reshape(2, BEAM_STATE_DIM)
        df = pd.DataFrame(arr, columns=BEAM_STATE_VARS)
        df.insert(0, "stage", ["beam0", "final"])
        return df

    @staticmethod
    def _feature_improved(feature: str, before: float, after: float) -> bool:
        if feature == "npart_ratio":
            return round(float(after), 3) >= round(float(before), 3)
        if feature in {"SizeX", "SizeY", "ex", "ey"}:
            return after < before
        if feature in {"x0", "y0", "x'0", "y'0"}:
            return abs(after) < abs(before)
        return after >= before

    @classmethod
    def _feature_trend(cls, feature: str, before: float, after: float) -> str:
        if feature == "npart_ratio":
            return "improved" if cls._feature_improved(feature, before, after) else "worse"
        if np.isclose(before, after):
            return "same"
        return "improved" if cls._feature_improved(feature, before, after) else "worse"

    @staticmethod
    def _feature_ylim(before_values, after_values):
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
