"""
TraceWinEnv provides REAL transitions (actual physics, ~30 s/step)

Shares its reset/step scaffolding with SurrogateEnv via BaseBeamEnv (env/base_beam_env.py).

State / Observation:
    Beam states selected by OBSERVATION_STAGE_MASK in adige.py and flattened
    into a 1-D vector. Machine parameters are not appended.
    Stage 0 is fixed by the .ini project file, not sampled.

Action:
    Delta on all configured parameters, bounded by per-parameter action_step_vec().

Reward:
    TERMINAL_FAILURE_REWARD for particle loss; otherwise
    score(t+1) / REWARD_SCORE_SCALE minus configured training regularizers

Episode design (consistent with the rest of the project):
    RESET:
        1. Sample params randomly: param_i ~ N(default_i, reset_std_i)
        2. Run TraceWin(params) → beam_states at all 12 stages
        3. obs = selected/flattened beam states
    STEP:
        params_{t+1} = params_t + action
        TraceWin(params_{t+1}) → obs_{t+1}
        reward = bounded failure reward or normalized score minus regularizers
    
        Truncated after max_steps steps. Never terminated early.

Note: the input beam (stage 0) is fixed by the .ini project file.
"""
from __future__ import annotations

from pathlib import Path

from beam_optimization.config.adige import (
    MAX_STEPS,
    TRAIN_RESET_SCALE,
)
from beam_optimization.config.paths import new_tracewin_env_calc_dir
from beam_optimization.env.base_beam_env import BaseBeamEnv
from beam_optimization.env.dataset import BeamDataset
from beam_optimization.env.tracewin_env.tracewin.tracewin_simulator import TraceWinSimulator


class TraceWinEnv(BaseBeamEnv):
    """Real-physics Gymnasium environment using TraceWin.

    Args:
        project_file:  Path to the TraceWin .ini project file.
        calc_dir:      Working directory for TraceWin output files. When omitted,
                       a unique temporary directory is assigned to this instance.
        max_steps:     Episode length (number of TraceWin calls per episode).
        observation:    Selected by OBSERVATION_STAGE_MASK in adige.py.
        timeout:       Seconds before aborting a single TraceWin call.
        retries:       Retry attempts on TraceWin failure.
        reset_scale:   Gaussian reset width in sensitivity units.
        distance_dataset: Dataset used as the KNN reference when the distance
                          reward penalty is enabled.
        capture_reset_distribution: If True, load the full particle
                          distribution from ``part_dtl1.dst`` into memory
                          right after every reset() (before any step()
                          overwrites the calc dir), so
                          render_initial_beam_distribution() can show it
                          later even after the episode has moved on. Off by
                          default: every reset() already writes this file,
                          but parsing it is extra work that training/
                          benchmarking loops (which never render it) should
                          not pay for.
        kill_stale: If True, perform the legacy global cleanup of TraceWin and
                    Xvfb processes. Set False when independent workspaces run
                    concurrently so one run cannot terminate another.
        num_threads: TraceWin threads per simulation. None uses all CPUs.
    """

    def __init__(
        self,
        project_file: str,
        calc_dir: str | None = None,
        max_steps: int = MAX_STEPS,
        timeout: float = 45.0,
        retries: int = 2,
        reset_scale: float = TRAIN_RESET_SCALE,
        distance_penalty_weight: float = 0.0,
        action_penalty_weight: float = 0.0,
        score_regression_penalty_weight: float = 0.0,
        distance_dataset: BeamDataset | None = None,
        capture_reset_distribution: bool = False,
        action_smoothness_penalty_weight: float = 0.0,
        kill_stale: bool = True,
        num_threads: int | None = None,
    ):

        if calc_dir is None:
            workspace_dir = Path(project_file).expanduser().resolve().parent
            calc_dir = str(new_tracewin_env_calc_dir(workspace_dir))

        # Store the simulator kwargs for later use in _build_simulator() for the TraceWin simulator
        self._simulator_kwargs = {
            "project_file": project_file,
            "calc_dir": calc_dir,
            "timeout": timeout,
            "retries": retries,
            "kill_stale": kill_stale,
            "num_threads": num_threads,
        }
        # TraceWinSimulator has no dataset of its own. When a distance penalty
        # is enabled, measure it against the dataset selected for this run.
        self._distance_dataset = distance_dataset
        self._capture_reset_distribution = bool(capture_reset_distribution)
        self._initial_beam_snapshot: dict | None = None

        # Call the base class constructor
        super().__init__(
            max_steps=max_steps,
            reset_scale=reset_scale,
            distance_penalty_weight=distance_penalty_weight,
            action_penalty_weight=action_penalty_weight,
            action_smoothness_penalty_weight=action_smoothness_penalty_weight,
            score_regression_penalty_weight=score_regression_penalty_weight,
        )

    def _build_simulator(self) -> TraceWinSimulator:
        return TraceWinSimulator(**self._simulator_kwargs)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # Snapshot part_dtl1.dst into memory now, right after this reset()'s
        # own simulate() call wrote it -- the next simulate() call (the first
        # step()) deletes and recreates calc_dir, so this is the only window
        # in which the initial-state file exists on disk.
        self._initial_beam_snapshot = (
            self._capture_current_beam_distribution() if self._capture_reset_distribution else None
        )
        return obs, info

    def _capture_current_beam_distribution(self) -> dict | None:
        """Load the calc dir's current completed distribution into memory.

        Returns {"distribution", "final_beam", "score_val", "dst_name"}, or
        None if no valid completed distribution is available right now (mirrors
        the same checks render_final_beam_distribution() makes). Loads every
        particle (no subsampling) so a later render can pick its own
        max_particles independently of whatever this snapshot was taken for.
        """
        from beam_optimization.env.tracewin_env.tracewin.visualization import (
            find_final_tracewin_dst_path,
            tracewin_distribution_from_dst,
        )

        dst_path = find_final_tracewin_dst_path(self.simulator.calc_dir)
        result = self.state.current_result
        if dst_path is None or result is None or not result.success or result.final_beam is None:
            return None

        return {
            "distribution": tracewin_distribution_from_dst(dst_path, max_particles=None),
            "final_beam": dict(result.final_beam),
            "score_val": result.score_val,
            "dst_name": dst_path.name,
        }

    def render(
        self,
        save_path: str | None = None,
        fps: int = 2,
        render_beam_distribution: bool = False,
        max_particles: int | None = None,
        bins: int = 200,
        axis_range_mm: float | None = None,
        xy_range_mm: float = 20.0,
        angle_range_mrad: float = 40.0,
    ):
        """
        The inherited render shows the same parameter/beam-feature episode
        trends used by SurrogateEnv (see BaseBeamEnv.render()).

        TraceWin can additionally render the real final particle
        distribution written by TraceWin in ``calc/part_dtl1.dst``: ``x-y``, ``x-x'`` and ``y-y'``.
        """

        # Call the base class render for the parameter/beam-feature trends.
        result = super().render(save_path=save_path, fps=fps)

        # If requested, render the final particle distribution in a second figure.
        if render_beam_distribution:
            result["beam_distribution"] = self.render_final_beam_distribution(
                max_particles=max_particles,
                bins=bins,
                axis_range_mm=axis_range_mm,
                xy_range_mm=xy_range_mm,
                angle_range_mrad=angle_range_mrad,
            )

        return result

    def render_final_beam_distribution(
        self,
        max_particles: int | None = None,
        bins: int = 200,
        axis_range_mm: float | None = None,
        xy_range_mm: float = 20.0,
        angle_range_mrad: float = 40.0,
    ):
        """Render the completed TraceWin distribution from ``part_dtl1.dst``.

        Uses the same default zoom (position +/-20 mm, angle +/-40 mrad),
        figure size, beam-state table and score as ``visualize_distributions.ipynb`` and
        ``visualize_distributions_python_run.ipynb`` — three phase-space
        panels plus the beam-state/score table underneath — via
        ``plot_tracewin_distribution_with_state()``, so a GUI run, a direct
        ``TraceWinSimulator`` run, and a ``TraceWinEnv`` step all render
        identically for the same beam state.
        """

        from beam_optimization.env.tracewin_env.tracewin.visualization import (
            find_final_tracewin_dst_path,
            plot_tracewin_distribution_with_state,
            tracewin_distribution_from_dst,
        )

        # Find the final .dst file in the TraceWin calc_dir
        dst_path = find_final_tracewin_dst_path(self.simulator.calc_dir)
        if dst_path is None:
            print(
                "TraceWin final beam distribution render skipped: no final .dst file found in "
                f"{self.simulator.calc_dir}."
            )
            return None

        result = self.state.current_result
        if result is None or not result.success or result.final_beam is None:
            print(
                "TraceWin final beam distribution render skipped for this step: "
                "no valid final particle distribution is available. "
                "A physical beam-loss result is terminal; no recovery step follows."
            )
            return None

        # Load the particle distribution from the .dst file
        distribution = tracewin_distribution_from_dst(
            dst_path,
            max_particles=max_particles,
        )

        # Backward compatibility: the old single range option controls both
        # position and angle only when explicitly supplied by the caller.
        if axis_range_mm is not None:
            xy_range_mm = float(axis_range_mm)
            angle_range_mrad = float(axis_range_mm)

        # Plot the distribution plus the beam-state/score table underneath.
        return plot_tracewin_distribution_with_state(
            distribution,
            result.final_beam,
            result.score_val,
            state_source=f"environment step {self.state.step_count}",
            title=(
                f"{type(self).__name__} final beam distribution | "
                f"{dst_path.name} | {len(distribution['x']):,} plotted particles"
            ),
            figure_name=f"{type(self).__name__} TraceWin final beam distribution",
            bins=bins,
            xy_range_mm=xy_range_mm,
            angle_range_mrad=angle_range_mrad,
            figsize=(22, 8.5),
            show=True,
        )

    def render_initial_beam_distribution(
        self,
        max_particles: int | None = None,
        bins: int = 200,
        axis_range_mm: float | None = None,
        xy_range_mm: float = 20.0,
        angle_range_mrad: float = 40.0,
    ):
        """Render the distribution captured right after reset(), before any
        step() has run this episode.

        Requires the env to have been built with
        capture_reset_distribution=True (off by default); otherwise nothing
        was captured and this prints an explanation and returns None. Same
        panels/layout as render_final_beam_distribution(), so the two can be
        compared side by side for "where the episode started" vs. "where it
        ended up".
        """
        if not self._capture_reset_distribution:
            print(
                "TraceWin initial beam distribution render skipped: this env was built with "
                "capture_reset_distribution=False, so no reset() snapshot was ever captured. "
                "Pass capture_reset_distribution=True to TraceWinEnv(...) to enable this."
            )
            return None

        snapshot = self._initial_beam_snapshot
        if snapshot is None:
            print(
                "TraceWin initial beam distribution render skipped: no snapshot captured yet "
                "(reset() has not produced a valid completed distribution)."
            )
            return None

        from beam_optimization.env.tracewin_env.tracewin.visualization import (
            plot_tracewin_distribution_with_state,
            subsample_distribution,
        )

        distribution = subsample_distribution(snapshot["distribution"], max_particles)

        # Backward compatibility: the old single range option controls both
        # position and angle only when explicitly supplied by the caller.
        if axis_range_mm is not None:
            xy_range_mm = float(axis_range_mm)
            angle_range_mrad = float(axis_range_mm)

        return plot_tracewin_distribution_with_state(
            distribution,
            snapshot["final_beam"],
            snapshot["score_val"],
            state_source="reset (initial state, before any step)",
            title=(
                f"{type(self).__name__} initial beam distribution | "
                f"{snapshot['dst_name']} | {len(distribution['x']):,} plotted particles"
            ),
            figure_name=f"{type(self).__name__} TraceWin initial beam distribution",
            bins=bins,
            xy_range_mm=xy_range_mm,
            angle_range_mrad=angle_range_mrad,
            figsize=(22, 8.5),
            show=True,
        )
