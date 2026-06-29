"""Plotting utilities for TraceWin particle distributions."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from beam_optimization.env.tracewin_env.tracewin.pyTraceWin_wrapper.files import Dst


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
    save_path: Optional[str] = None,
    dpi: int = 140,
    show: bool = False,
) -> plt.Figure:
    """Plot TraceWin output distributions as x-y, x-x', and y-y' panels."""
    
    # create a new figure if none is provided 
    if figure is None or not plt.fignum_exists(figure.number):
        figure = plt.figure(figure_name, figsize=(18, 5))

    # clear the figure and create a 1x3 grid of subplots
    figure.clf()
    axes = figure.subplots(1, 3)

    # define the specifications for each subplot, including the keys for x and y values, labels, and titles 
    plot_specs = [
        ("x", "y", "x [mm]", "y [mm]", "x-y plane"),
        ("x", "xp", "x [mm]", "x' [mrad]", "x-xp emittance"),
        ("y", "yp", "y [mm]", "yp [mrad]", "y-yp emittance"),
    ]

    hist_range = [[-axis_range_mm, axis_range_mm], [-axis_range_mm, axis_range_mm]]

    
    for ax, (x_key, y_key, xlabel, ylabel, panel_title) in zip(axes, plot_specs):
        # Convert m/rad -> mm/mrad for display
        x_vals = np.asarray(distr[x_key], dtype=float) * 1000.0
        y_vals = np.asarray(distr[y_key], dtype=float) * 1000.0
        # Filter out any non-finite values to avoid issues with histogram calculation
        mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        # Filter the x and y values based on the mask
        x_vals = x_vals[mask]
        y_vals = y_vals[mask]

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
