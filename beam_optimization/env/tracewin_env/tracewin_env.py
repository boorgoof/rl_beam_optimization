"""
TraceWinEnv provides REAL transitions (actual physics, ~30 s/step)

Shares its reset/step scaffolding with SurrogateEnv via BaseBeamEnv (env/base_beam_env.py).

State / Observation:
    Beam states selected by OBSERVATION_STAGE_MASK in adige.py and flattened
    into a 1-D vector.
    Stage 0 is fixed by the .ini project file, not sampled.

Action:
    Delta on all configured parameters, bounded by per-parameter action_step_vec().

Reward:
    score(t+1) - score(t) 

Episode design (consistent with the rest of the project):
    RESET:
        1. Sample params randomly: param_i ~ N(default_i, reset_std_i)
        2. Run TraceWin(params) → beam_states at all 12 stages
        3. obs = selected/flattened beam_states ← initial RL state
    STEP:
        params_{t+1} = params_t + action
        TraceWin(params_{t+1}) → obs_{t+1}
        reward = score(t+1) - score(t)
    
        Truncated after max_steps steps. Never terminated early.

Note: the input beam (stage 0) is fixed by the .ini project file.
"""
from __future__ import annotations

from pathlib import Path

from beam_optimization.config.adige import MAX_STEPS, TRAIN_RESET_SCALE
from beam_optimization.config.paths import new_tracewin_env_calc_dir
from beam_optimization.env.base_beam_env import BaseBeamEnv
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
    """

    def __init__(
        self,
        project_file: str,
        calc_dir: str | None = None,
        max_steps: int = MAX_STEPS,
        timeout: float = 45.0,
        retries: int = 2,
        reset_scale: float = TRAIN_RESET_SCALE,
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
        }

        # Call the base class constructor
        super().__init__(
            max_steps=max_steps,
            reset_scale=reset_scale,
        )

    def _build_simulator(self) -> TraceWinSimulator:
        return TraceWinSimulator(**self._simulator_kwargs)

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
        """Render the final TraceWin particle distribution from the latest calc/*.dst.

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

        result = self._current_result
        if result is None or not result.success or result.final_beam is None:
            print(
                "TraceWin final beam distribution render skipped: no successful "
                "simulation result yet (call reset()/step() first)."
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
            state_source=f"environment step {self._step_count}",
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
