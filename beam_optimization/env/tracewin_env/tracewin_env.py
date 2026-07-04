"""
TraceWinEnv provides REAL transitions (actual physics, ~30 s/step)

Shares its reset/step scaffolding with SurrogateEnv via BaseBeamEnv (env/base_beam_env.py).

State / Observation:
    Beam states selected by OBSERVATION_STAGE_MASK in adige.py and flattened
    into a 1-D vector.
    Stage 0 is fixed by the .ini project file, not sampled.

Action:
    Delta on all 16 parameters, bounded by per-parameter action_step_vec().

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

from beam_optimization.config.adige import MAX_STEPS
from beam_optimization.config.paths import DEFAULT_TRACEWIN_ENV_CALC_DIR
from beam_optimization.env.base_beam_env import BaseBeamEnv
from beam_optimization.env.tracewin_env.tracewin.tracewin_simulator import TraceWinSimulator


class TraceWinEnv(BaseBeamEnv):
    """Real-physics Gymnasium environment using TraceWin.

    Args:
        project_file:  Path to the TraceWin .ini project file.
        calc_dir:      Working directory for TraceWin output files.
        max_steps:     Episode length (number of TraceWin calls per episode).
        observation:    Selected by OBSERVATION_STAGE_MASK in adige.py.
        timeout:       Seconds before aborting a single TraceWin call.
        retries:       Retry attempts on TraceWin failure.
    """

    def __init__(
        self,
        project_file: str,
        calc_dir: str = str(DEFAULT_TRACEWIN_ENV_CALC_DIR),
        max_steps: int = MAX_STEPS,
        timeout: float = 120.0,
        retries: int = 2,
    ):

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
        )

    def _build_simulator(self) -> TraceWinSimulator:
        return TraceWinSimulator(**self._simulator_kwargs)

    def render(
        self,
        save_path: str | None = None,
        fps: int = 2,
        render_beam_distribution: bool = False,
        max_particles: int = 40000,
        bins: int = 150,
        axis_range_mm: float = 50.0,
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
            )

        return result

    def render_final_beam_distribution(
        self,
        max_particles: int = 40000,
        bins: int = 150,
        axis_range_mm: float = 50.0,
    ):
        """Render the final TraceWin particle distribution from the latest calc/*.dst."""
        
        from beam_optimization.env.tracewin_env.tracewin.visualization import (
            find_final_tracewin_dst_path,
            plot_tracewin_distribution,
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

        # Load the particle distribution from the .dst file
        distribution = tracewin_distribution_from_dst(
            dst_path,
            max_particles=max_particles,
        )
        
        # Plot the distribution using the provided parameters
        return plot_tracewin_distribution(
            distribution,
            title=(
                f"{type(self).__name__} final beam distribution | "
                f"{dst_path.name} | {len(distribution['x']):,} plotted particles"
            ),
            figure_name=f"{type(self).__name__} TraceWin final beam distribution",
            bins=bins,
            axis_range_mm=axis_range_mm,
            show=True,
        )
