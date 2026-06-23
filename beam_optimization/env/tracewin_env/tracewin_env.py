"""
TraceWinEnv — Gymnasium environment backed by the real TraceWin physics simulator.

Same interface as SurrogateEnv (108-dim observation, 16-dim action) so it can be
used as a drop-in replacement for SurrogateEnv when real physics data is available.
Both share their reset/step scaffolding via BaseBeamEnv (env/base_env.py); this
class only configures TraceWinSimulator as the backend.

Intended use with DynaMBPO:
    - TraceWinEnv provides REAL transitions (actual physics, ~30 s/step)
    - Surrogate ensemble provides SYNTHETIC transitions (cheap, ~1 ms/step)
    The combination gives physically grounded data without requiring thousands
    of TraceWin calls.

Episode design (consistent with the rest of the project):
    RESET:
        1. Sample params randomly: param_i ~ N(default_i, sensitivity_i * sigma_factor)
        2. Run TraceWin(params) → beam_states at all 12 stages
        3. obs = flatten(beam_states) = (108,)   ← initial RL state
    STEP:
        params_{t+1} = params_t + action
        TraceWin(params_{t+1}) → obs_{t+1}
        reward = score(t+1) - score(t)

Note: the input beam (stage 0) is fixed by the .ini project file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from beam_optimization.env.base_beam_env import BaseBeamEnv
from beam_optimization.env.tracewin_env.tracewin.simulator import TraceWinSimulator
from beam_optimization.env.tracewin_env.tracewin.pyTraceWin_wrapper.files import Dst


class TraceWinEnv(BaseBeamEnv):
    """Real-physics Gymnasium environment using TraceWin.

    Args:
        project_file:  Path to the TraceWin .ini project file.
        calc_dir:      Working directory for TraceWin output files.
        action_scale:  Multiplier on sensitivity for action bounds.
        max_steps:     Episode length (number of TraceWin calls per episode).
        sigma_factor:  Gaussian noise scale (× sensitivity) for initial params.
        obs_mode:      'full' (108), 'final' (9), or 'final_with_beam0' (18).
        timeout:       Seconds before aborting a single TraceWin call.
        retries:       Retry attempts on TraceWin failure.
        use_local_project_cache:
                      If True, run TraceWin on a local copy of the project
                      workspace while keeping calc_dir as the output folder.
        local_project_cache_root:
                      Root folder for the local project cache.
    """

    def __init__(
        self,
        project_file: str,
        calc_dir: str = "/tmp/tracewin_calc",
        action_scale: float = 1.0,
        max_steps: int = 20,
        sigma_factor: float = 0.5,
        obs_mode: str = "full",
        timeout: float = 120.0,
        retries: int = 2,
        use_local_project_cache: bool = True,
        local_project_cache_root: str | None = None,
    ):
        self.simulator = TraceWinSimulator(
            project_file=project_file,
            calc_dir=calc_dir,
            timeout=timeout,
            retries=retries,
            use_local_project_cache=use_local_project_cache,
            local_project_cache_root=local_project_cache_root,
        )
        super().__init__(
            action_scale=action_scale,
            max_steps=max_steps,
            sigma_factor=sigma_factor,
            obs_mode=obs_mode,
        )

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def n_simulations(self) -> int:
        return self.simulator.n_simulations

    def render(
        self,
        mode: str = "human",
        include_phase_space: bool = True,
        max_particles: int = 40000,
        bins: int = 150,
    ):
        """Render TraceWin observation features and, optionally, true particles.

        The inherited render shows the same observation-feature bars used by
        SurrogateEnv.  TraceWin can additionally render the real final particle
        distribution written by TraceWin in ``calc/part_dtl1.dst``:
        ``x-y``, ``x-x'`` and ``y-y'``.

        The particle plots are diagnostics only; they are not part of the RL
        observation space.
        """
        fig = super().render(mode=mode)
        if include_phase_space:
            self.render_phase_space(max_particles=max_particles, bins=bins)
        return fig

    def render_phase_space(self, max_particles: int = 40000, bins: int = 150):
        """Render true final TraceWin phase-space images from the latest calc/*.dst."""
        dst_path = self._final_dst_path()
        if dst_path is None:
            print(
                "TraceWin phase-space render skipped: no final .dst file found in "
                f"{self.simulator.calc_dir}."
            )
            return None

        import matplotlib.pyplot as plt

        cloud = self._dst_cloud(dst_path, max_particles=max_particles)
        panels = [
            ("x", "y", "x [mm]", "y [mm]", "final x-y"),
            ("x", "xp", "x [mm]", "x' [mrad]", "final x-x'"),
            ("y", "yp", "y [mm]", "y' [mrad]", "final y-y'"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
        fig.suptitle(
            f"{type(self).__name__} true final particle phase spaces | "
            f"{dst_path.name} | {cloud['n_total']:,} particles",
            fontsize=13,
        )

        for ax, (a_key, b_key, xlabel, ylabel, title) in zip(axes, panels):
            a = cloud[a_key]
            b = cloud[b_key]
            arange = self._limits_with_padding(a)
            brange = self._limits_with_padding(b)
            h, aedge, bedge = np.histogram2d(a, b, bins=bins, range=[arange, brange])
            h_log = np.log1p(h)
            h_log[h_log == 0] = np.nan

            im = ax.pcolormesh(aedge, bedge, h_log.T, cmap="Blues", shading="auto")
            ax.axhline(0, color="black", lw=0.8, linestyle="--")
            ax.axvline(0, color="black", lw=0.8, linestyle="--")
            ax.plot(np.mean(a), np.mean(b), marker="+", color="red", markersize=13, mew=2)
            ax.set_xlim(*arange)
            ax.set_ylim(*brange)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=10)
            ax.tick_params(labelsize=8)
            cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
            cb.set_label("log(counts+1)", fontsize=7)
            cb.ax.tick_params(labelsize=6)

        plt.tight_layout()
        plt.show()
        return fig

    def _final_dst_path(self) -> Path | None:
        """Return the best available final distribution file in calc_dir.

        TraceWin projects are not fully consistent about the output name:
        some write ``part_dtl1.dst`` while PLOT_DST diagnostics often write
        numbered files such as ``1.dst``, ``2.dst``, ``3.dst``.  Prefer
        ``part_dtl1.dst`` when present; otherwise use the highest numbered
        ``*.dst`` file as the final diagnostic.
        """
        calc_dir = Path(self.simulator.calc_dir)
        preferred = calc_dir / "part_dtl1.dst"
        if preferred.exists():
            return preferred

        numbered = []
        for path in calc_dir.glob("*.dst"):
            try:
                numbered.append((int(path.stem), path))
            except ValueError:
                continue
        if numbered:
            return max(numbered, key=lambda item: item[0])[1]

        fallback = sorted(calc_dir.glob("*.dst"))
        return fallback[-1] if fallback else None

    @staticmethod
    def _dst_cloud(dst_path: Path, max_particles: int):
        dst = Dst(str(dst_path))
        n = int(dst.Np)
        idx = np.arange(n)
        if n > max_particles:
            rng = np.random.default_rng(123)
            idx = rng.choice(idx, size=max_particles, replace=False)

        return {
            "n_total": n,
            "x": np.asarray(dst["x"])[idx] * 1e3,
            "xp": np.asarray(dst["xp"])[idx] * 1e3,
            "y": np.asarray(dst["y"])[idx] * 1e3,
            "yp": np.asarray(dst["yp"])[idx] * 1e3,
        }

    @staticmethod
    def _limits_with_padding(values, low: float = 0.5, high: float = 99.5):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return -1.0, 1.0

        lo, hi = np.nanpercentile(values, [low, high])
        if np.isclose(lo, hi):
            pad = max(abs(float(hi)) * 0.1, 1e-6)
            return float(lo - pad), float(hi + pad)

        pad = 0.08 * (hi - lo)
        return float(lo - pad), float(hi + pad)
