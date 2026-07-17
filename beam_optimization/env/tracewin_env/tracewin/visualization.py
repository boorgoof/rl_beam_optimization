"""Plotting utilities for TraceWin particle distributions."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np

from beam_optimization.config.adige import BEAM_STATE_FEATURES
from beam_optimization.env.tracewin_env.tracewin.pyTraceWin_wrapper.files import Dst


# Table headers/formatting for the beam-state row shown under a distribution
# plot. Shared by the GUI reference viewer (visualize_distributions.ipynb),
# the Python TraceWinSimulator runs (visualize_distributions_python_run.ipynb)
# and TraceWinEnv.render_final_beam_distribution(), so all four draw the
# exact same figure from the same code.
BEAM_STATE_FEATURE_LABELS: Dict[str, str] = {
    "npart_ratio": "npart_ratio\nparticle fraction",
    "x0": "x0\nX centroid [mm]", "y0": "y0\nY centroid [mm]",
    "SizeX": "SizeX\nX RMS size [mm]", "SizeY": "SizeY\nY RMS size [mm]",
    "ex": "ex\nX emittance [mm.mrad]", "ey": "ey\nY emittance [mm.mrad]",
    "x'0": "x'0\nX angular centroid [mrad]",
    "y'0": "y'0\nY angular centroid [mrad]",
}


def format_beam_state_value(feature: str, value: float) -> str:
    """Format one beam-state feature value for the table cell text."""
    if feature == "npart_ratio":
        return f"{value:.4f} ({value * 100:.2f}%)"
    return f"{value:.6g}"


def find_final_tracewin_dst_path(calc_dir: str | Path) -> Path | None:
    """Return the final distribution file in a TraceWin calc dir."""
    
    # try to find the default output file first.
    calc_dir = Path(calc_dir)
    preferred = calc_dir / "part_dtl1.dst"
    if preferred.exists():
        return preferred

    # otherwise, look for the latest numbered .dst file.
    numbered = []
    for path in calc_dir.glob("*.dst"):
        try:
            numbered.append((int(path.stem), path))
        except ValueError:
            continue
    if numbered:
        return max(numbered, key=lambda item: item[0])[1]

    # if no numbered files, just return the last .dst file in sorted order.
    fallback = sorted(calc_dir.glob("*.dst"))
    return fallback[-1] if fallback else None


def tracewin_distribution_from_dst(
    dst_path: str | Path,
    *,
    max_particles: Optional[int] = None,
    seed: int = 123,
) -> Dict[str, np.ndarray]:
    """Read x/y/xp/yp particle arrays from a TraceWin .dst file.

    Values are returned in TraceWin native units: x/y in metres and xp/yp in
    radians. Plotting helpers convert them to mm/mrad for display.
    """

    # get the .dist file path and check that it exists.
    dst_path = Path(dst_path)
    if not dst_path.exists():
        raise FileNotFoundError(f"TraceWin distribution file not found: {dst_path}")

    # read the .dst file and extract the number of particle to create an array of indices.
    dst = Dst(str(dst_path))
    n_particles = int(dst.Np)
    indices = np.arange(n_particles)

    # if a maximum number of particles is specified, randomly select that many indices.
    if max_particles is not None and n_particles > max_particles:
        rng = np.random.default_rng(seed)
        indices = rng.choice(indices, size=max_particles, replace=False)

    # return the selected particle features of the array as a dictionary
    return {
        "x": np.asarray(dst["x"], dtype=float)[indices],
        "y": np.asarray(dst["y"], dtype=float)[indices],
        "xp": np.asarray(dst["xp"], dtype=float)[indices],
        "yp": np.asarray(dst["yp"], dtype=float)[indices],
    }

# daniele
def plot_tracewin_distribution(
    distr: Dict[str, np.ndarray],
    *,
    title: Optional[str] = None,
    figure: Optional[plt.Figure] = None,
    figure_name: str = "TraceWin Output Distribution",
    bins: int = 200,
    axis_range_mm: float = 50.0,
    xy_range_mm: Optional[float] = None,
    angle_range_mrad: Optional[float] = None,
    aperture_radius_mm: Optional[float] = None,
    figsize: tuple[float, float] = (18, 5),
    save_path: Optional[str] = None,
    dpi: int = 140,
    show: bool = False,
) -> plt.Figure:
    """Plot TraceWin output distributions as x-y, x-x', and y-y' panels.

    ``axis_range_mm`` is kept for backward compatibility and is used for both
    position and angle axes unless ``xy_range_mm`` or ``angle_range_mrad`` are
    provided explicitly.
    """
    
    # create a new figure if none is provided 
    if figure is None or not plt.fignum_exists(figure.number):
        figure = plt.figure(figure_name, figsize=figsize)

    # clear the figure and create a 1x3 grid of subplots
    figure.clf()
    axes = figure.subplots(1, 3)

    if xy_range_mm is None:
        xy_range_mm = axis_range_mm
    if angle_range_mrad is None:
        angle_range_mrad = axis_range_mm

    # define the specifications for each subplot, including the keys for x and y values, labels, and titles 
    plot_specs = [
        ("x", "y", "x [mm]", "y [mm]", "x-y plane", xy_range_mm, xy_range_mm),
        ("x", "xp", "x [mm]", "x' [mrad]", "x-xp emittance", xy_range_mm, angle_range_mrad),
        ("y", "yp", "y [mm]", "yp [mrad]", "y-yp emittance", xy_range_mm, angle_range_mrad),
    ]

    for ax, (x_key, y_key, xlabel, ylabel, panel_title, x_limit, y_limit) in zip(axes, plot_specs):
        # Convert m/rad -> mm/mrad for display
        x_vals = np.asarray(distr[x_key], dtype=float) * 1000.0
        y_vals = np.asarray(distr[y_key], dtype=float) * 1000.0
        # Filter out any non-finite values to avoid issues with histogram calculation
        mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        # Filter the x and y values based on the mask
        x_vals = x_vals[mask]
        y_vals = y_vals[mask]
        hist_range = [[-x_limit, x_limit], [-y_limit, y_limit]]

        # transform  particles into a 2D histogram for plotting
        hist, xedges, yedges = np.histogram2d(
            x_vals,
            y_vals,
            bins=bins,
            range=hist_range,
            density=False,
        )

        # normalize the histogram to represent a probability density function
        total = hist.sum()
        if total > 0:
            dx = xedges[1] - xedges[0]
            dy = yedges[1] - yedges[0]
            hist = hist / (total * dx * dy)

        # plot the histogram as an image on the current axis
        image = ax.imshow(
            hist.T,
            interpolation="none",
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            aspect="auto",
        )
        # set the title and labels for the current axis, and add a colorbar to indicate the density values
        ax.set_title(panel_title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if x_key == "x" and y_key == "y" and aperture_radius_mm is not None:
            aperture = plt.Circle(
                (0.0, 0.0),
                aperture_radius_mm,
                fill=False,
                color="white",
                linewidth=1.8,
                linestyle="--",
                label=f"tube radius {aperture_radius_mm:g} mm",
            )
            ax.add_patch(aperture)
            ax.legend(fontsize=8, loc="upper right")

        figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    if title is not None and title.strip():
        figure.suptitle(title)
    figure.tight_layout()

    if save_path is not None:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, dpi=dpi)

    if show:
        plt.show(block=False)

    return figure


def plot_tracewin_distribution_with_state(
    distr: Dict[str, np.ndarray],
    beam_state: Dict[str, float],
    score_val: float,
    *,
    state_source: str,
    title: Optional[str] = None,
    beam_state_features: Iterable[str] = BEAM_STATE_FEATURES,
    figure: Optional[plt.Figure] = None,
    figure_name: str = "TraceWin Output Distribution",
    bins: int = 200,
    xy_range_mm: float = 20.0,
    angle_range_mrad: float = 40.0,
    aperture_radius_mm: Optional[float] = None,
    figsize: tuple[float, float] = (22, 8.5),
    save_path: Optional[str] = None,
    dpi: int = 160,
    show: bool = False,
) -> plt.Figure:
    """Plot a TraceWin distribution plus the beam-state/score table underneath.

    This is the shared rendering path for the GUI reference viewer
    (visualize_distributions.ipynb), the explicit- and default-parameter
    ``TraceWinSimulator`` runs (visualize_distributions_python_run.ipynb),
    and ``TraceWinEnv.render_final_beam_distribution()`` — all four draw the
    same figure from this one function instead of duplicated notebook code.
    """
    figure = plot_tracewin_distribution(
        distr,
        title=title,
        figure=figure,
        figure_name=figure_name,
        bins=bins,
        xy_range_mm=xy_range_mm,
        angle_range_mrad=angle_range_mrad,
        aperture_radius_mm=aperture_radius_mm,
        figsize=figsize,
        save_path=None,
        show=False,
    )
    figure.subplots_adjust(bottom=0.28, top=0.88)
    figure.text(
        0.5, 0.205,
        f"Beam state | Score = {score_val:.6g} | {state_source}",
        ha="center", va="center", fontsize=12.5, fontweight="bold",
    )

    state_axis = figure.add_axes([0.035, 0.025, 0.93, 0.145])
    state_axis.axis("off")
    beam_state_features = list(beam_state_features)
    headers = [BEAM_STATE_FEATURE_LABELS[feature] for feature in beam_state_features]
    values = [format_beam_state_value(feature, beam_state[feature]) for feature in beam_state_features]
    table = state_axis.table(
        cellText=[headers, values], cellLoc="center", loc="center", bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#7892b0")
        cell.set_linewidth(1.0)
        cell.set_facecolor("#dce7f3" if row == 0 else "#f8fafc")
        if row == 0:
            cell.set_text_props(fontweight="bold")

    if save_path is not None:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show(block=False)

    return figure
