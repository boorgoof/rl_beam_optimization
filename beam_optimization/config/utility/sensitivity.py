"""
Sensitivity analysis for ADIGE beam line parameters.

Computes the sensitivity of the score to each tunable parameter via central
finite differences, using TraceWin as the physics simulator. The values are
expressed in the same convention used by adige.py:

    sensitivity_p = 2δ / |score(p+δ) - score(p-δ)|   [Δparam / Δscore]

This is the inverse of the standard derivative (Δscore/Δparam). It represents
how much you need to move parameter p to change the score by 1 point — the
natural unit for action step sizing in the RL environment.

Run as a script:
    python -m beam_optimization.config.utility.sensitivity

The script prints a human-readable stability table and a copy-paste block ready
to replace the sensitivity= values in adige.py. It does NOT modify any files.
"""
from __future__ import annotations

import copy
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

from beam_optimization.config.adige import PARAMETERS, default_params


def compute_sensitivity(
    simulator,
    *,
    delta_scales: List[float] | None = None,
    repeats: int = 1,
    min_score_diff: float = 1e-9,
    aggregation: str = "median",
    checkpoint_path: str | Path | None = None,
    resume: bool = True,
    verbose: bool = True,
) -> Dict[str, List[Optional[float]]]:
    """Compute per-parameter sensitivity via central finite differences.

    For each parameter p and each delta_scale s, perturbs p by δ = s * sensitivity_current
    (keeping all other parameters at their nominal defaults) and computes:

        sensitivity_p = 2δ / |score(p+δ) - score(p-δ)|   [Δparam / Δscore]

    Repeating at three δ scales lets you verify gradient stability:
      - ratio sens(δ) / sens(δ/2) near 1.0 → gradient is linear, values are reliable
      - large ratio → δ too large (nonlinear regime) or too small (numerical noise)

    Args:
        simulator: Any object with a .simulate(params: dict) method returning a
                   result with .score_val (float) and .success (bool).
        delta_scales: Fractions of the current sensitivity to use as δ.
                      Defaults to [1.0, 0.5, 0.25].
        repeats: Number of repeated TraceWin runs for each +δ and -δ point.
                 With PARTRAN noise, use 3-5 and aggregate the scores.
        min_score_diff: Minimum accepted |score(+δ) - score(-δ)| after
                        aggregation. Smaller differences are treated as noise.
        aggregation: "median" or "mean" score aggregation across repeats.
        checkpoint_path: Optional JSON file updated after each completed
                         parameter/scale.
        resume: If True and checkpoint_path exists, reuse completed values.
        verbose: Print progress to stdout during the (potentially long) run.

    Returns:
        Dict mapping param name → list of sensitivity values, one per delta_scale.
        A None entry means that simulation failed or the score difference was ~0.
    """
    if delta_scales is None:
        delta_scales = [1.0, 0.5, 0.25]
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if aggregation not in {"median", "mean"}:
        raise ValueError("aggregation must be 'median' or 'mean'")

    defaults = default_params()
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    results: Dict[str, List[Optional[float]]] = (
        _load_checkpoint(checkpoint, delta_scales) if checkpoint is not None and resume else {}
    )
    n_total = len(PARAMETERS) * len(delta_scales) * 2 * repeats
    run_count = 0

    for p in PARAMETERS:
        sens_list: List[Optional[float]] = list(results.get(p.name, []))

        for scale_idx, scale in enumerate(delta_scales):
            if scale_idx < len(sens_list):
                if verbose:
                    print(
                        f"  [resume] {p.name:<14}  scale={scale}  "
                        f"sens={_format_optional_sensitivity(sens_list[scale_idx])}",
                        flush=True,
                    )
                continue

            delta = scale * p.sensitivity

            params_plus  = copy.copy(defaults)
            params_minus = copy.copy(defaults)
            params_plus[p.key]  = p.default + delta
            params_minus[p.key] = p.default - delta

            plus_scores: List[float] = []
            minus_scores: List[float] = []

            for repeat_idx in range(repeats):
                run_count += 1
                if verbose:
                    print(
                        f"  [{run_count:3d}/{n_total}] {p.name:<14}  +δ  "
                        f"(scale={scale}, repeat={repeat_idx + 1}/{repeats})",
                        flush=True,
                    )
                res_plus = simulator.simulate(params_plus)
                if res_plus.success:
                    plus_scores.append(float(res_plus.score_val))

                run_count += 1
                if verbose:
                    print(
                        f"  [{run_count:3d}/{n_total}] {p.name:<14}  -δ  "
                        f"(scale={scale}, repeat={repeat_idx + 1}/{repeats})",
                        flush=True,
                    )
                res_minus = simulator.simulate(params_minus)
                if res_minus.success:
                    minus_scores.append(float(res_minus.score_val))

            if not plus_scores or not minus_scores:
                if verbose:
                    print(f"    WARNING: simulation failed for {p.name} at scale={scale}")
                sens_list.append(None)
                results[p.name] = sens_list
                if checkpoint is not None:
                    _save_checkpoint(checkpoint, delta_scales, results)
                continue

            score_plus = _aggregate_scores(plus_scores, aggregation)
            score_minus = _aggregate_scores(minus_scores, aggregation)
            score_diff = abs(score_plus - score_minus)
            if score_diff < min_score_diff:
                if verbose:
                    print(
                        f"    WARNING: |Δscore| too small for {p.name} at scale={scale} "
                        f"(Δscore = {score_diff:.4f}, min = {min_score_diff:.4f})"
                )
                sens_list.append(None)
                results[p.name] = sens_list
                if checkpoint is not None:
                    _save_checkpoint(checkpoint, delta_scales, results)
                continue

            sensitivity_val = (2.0 * abs(delta)) / score_diff
            if verbose:
                deriv = score_diff / (2.0 * abs(delta))
                plus_std = _score_std(plus_scores)
                minus_std = _score_std(minus_scores)
                print(
                    f"    sens = {sensitivity_val:.6e}  "
                    f"(Δscore/Δparam = {deriv:.4f},  Δscore = {score_diff:.4f}, "
                    f"score+ = {score_plus:.4f}±{plus_std:.4f}, "
                    f"score- = {score_minus:.4f}±{minus_std:.4f})"
                )
            sens_list.append(sensitivity_val)
            results[p.name] = sens_list
            if checkpoint is not None:
                _save_checkpoint(checkpoint, delta_scales, results)

        results[p.name] = sens_list

    return results


def _aggregate_scores(scores: List[float], aggregation: str) -> float:
    if aggregation == "mean":
        return statistics.fmean(scores)
    return statistics.median(scores)


def _score_std(scores: List[float]) -> float:
    if len(scores) < 2:
        return 0.0
    return statistics.pstdev(scores)


def _format_optional_sensitivity(value: Optional[float]) -> str:
    return "FAILED" if value is None else f"{value:.6e}"


def _load_checkpoint(
    checkpoint_path: Path,
    delta_scales: List[float],
) -> Dict[str, List[Optional[float]]]:
    if not checkpoint_path.exists():
        return {}
    with open(checkpoint_path) as f:
        payload = json.load(f)

    saved_scales = [float(value) for value in payload.get("delta_scales", [])]
    if saved_scales != [float(value) for value in delta_scales]:
        raise ValueError(
            f"Checkpoint {checkpoint_path} was created with delta_scales={saved_scales}, "
            f"but this run uses delta_scales={delta_scales}. Use --no-resume or a new --checkpoint."
        )

    raw_results = payload.get("results", {})
    results: Dict[str, List[Optional[float]]] = {}
    for name, values in raw_results.items():
        results[name] = [None if value is None else float(value) for value in values]
    return results


def _save_checkpoint(
    checkpoint_path: Path,
    delta_scales: List[float],
    results: Dict[str, List[Optional[float]]],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "delta_scales": [float(value) for value in delta_scales],
        "results": results,
    }
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(checkpoint_path)


def _select_stable_sensitivity(
    vals: List[Optional[float]],
    *,
    stability_low: float,
    stability_high: float,
) -> Optional[float]:
    """Use the median of values that belong to at least one stable adjacent pair."""
    stable_indices: set[int] = set()
    for i in range(len(vals) - 1):
        a = vals[i]
        b = vals[i + 1]
        if a is None or b is None or b == 0:
            continue
        ratio = a / b
        if stability_low <= ratio <= stability_high:
            stable_indices.add(i)
            stable_indices.add(i + 1)

    if not stable_indices:
        return None

    stable_vals = [vals[i] for i in sorted(stable_indices) if vals[i] is not None]
    return statistics.median(stable_vals) if stable_vals else None


def print_sensitivity_report(
    results: Dict[str, List[Optional[float]]],
    delta_scales: List[float] | None = None,
    stability_low: float = 0.7,
    stability_high: float = 1.3,
) -> None:
    """Print a stability table and a copy-paste block ready for adige.py.

    Table columns:
        name | default | current_sensitivity | sens(δ×1.0) | sens(δ×0.5) | sens(δ×0.25) | stability

    Stability ratio = sens(δ×1.0) / sens(δ×0.5). Values near 1.0 mean the
    gradient is stable and the result is reliable. A '!' flag marks ratios
    outside [0.7, 1.3].

    The copy-paste block uses the median of values that belong to a stable
    adjacent-scale plateau. If no plateau exists, it keeps the old value.
    """
    if delta_scales is None:
        delta_scales = [1.0, 0.5, 0.25]

    scale_headers = [f"sens(δ×{s})" for s in delta_scales]
    col_w = 14

    # ── stability table ───────────────────────────────────────────────────────
    header = (
        f"{'Parameter':<14} {'default':>12} {'current_sens':>14}"
        + "".join(f" {h:>{col_w}}" for h in scale_headers)
        + f" {'stability':>10}"
    )
    sep = "=" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    for p in PARAMETERS:
        vals = results.get(p.name, [None] * len(delta_scales))
        sens_strs = [f"{v:.6e}" if v is not None else "    FAILED" for v in vals]

        stab_str = "         —"
        ratios = [
            vals[i] / vals[i + 1]
            for i in range(len(vals) - 1)
            if vals[i] is not None and vals[i + 1] not in (None, 0)
        ]
        if ratios:
            stable = any(stability_low <= r <= stability_high for r in ratios)
            ratio_preview = min(ratios, key=lambda r: abs(r - 1.0))
            flag = "  " if stable else " !"
            stab_str = f"{ratio_preview:>9.3f}{flag}"

        row = (
            f"{p.name:<14} {p.default:>12.6g} {p.sensitivity:>14.6e}"
            + "".join(f" {s:>{col_w}}" for s in sens_strs)
            + f" {stab_str}"
        )
        print(row)

    print(sep)
    print(
        "\nstability = best adjacent ratio sens(δ×a) / sens(δ×b).  "
        f"Near 1.0 → stable.  '!' → no adjacent ratio inside [{stability_low}, {stability_high}].\n"
    )

    # ── copy-paste block ──────────────────────────────────────────────────────
    bar = "─" * 80
    print(bar)
    print("Copy-paste block for adige.py  (uses stable adjacent-scale plateau only):")
    print(bar)
    for p in PARAMETERS:
        vals = results.get(p.name, [])
        new_sens = _select_stable_sensitivity(
            vals,
            stability_low=stability_low,
            stability_high=stability_high,
        )

        hw_min_s = "None" if p.hw_min is None else repr(p.hw_min)
        hw_max_s = "None" if p.hw_max is None else repr(p.hw_max)

        if new_sens is None:
            print(
                f"    # {p.name}: NO STABLE PLATEAU — keep old value {p.sensitivity:.15e}"
            )
            new_sens = p.sensitivity
        print(
            f'    ParameterSpec("{p.name}", "{p.key}", marker={p.marker}, '
            f'default={p.default!r}, sensitivity={new_sens:.15e}, '
            f'hw_min={hw_min_s}, hw_max={hw_max_s}, '
            f'action_scale_rl={p.action_scale_rl:.15e}, '
            f'reset_scale={p.reset_scale:.15e}),'
        )
    print(bar)


if __name__ == "__main__":
    import argparse

    from beam_optimization.config.paths import (
        DEFAULT_SENSITIVITY_CALC_DIR,
        DEFAULT_SENSITIVITY_CHECKPOINT,
        DEFAULT_TRACEWIN_INI,
    )
    from beam_optimization.env.tracewin_env.tracewin.tracewin_simulator import (
        TraceWinSimulator,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Compute ADIGE parameter sensitivity from TraceWin finite differences. "
            "Prints a stability table and a copy-paste block — does NOT modify adige.py."
        )
    )
    parser.add_argument(
        "--ini",
        default=str(DEFAULT_TRACEWIN_INI),
        help="Path to TraceWin .ini project file (default: %(default)s)",
    )
    parser.add_argument(
        "--calc-dir",
        default=str(DEFAULT_SENSITIVITY_CALC_DIR),
        help="Temporary calculation directory for TraceWin outputs",
    )
    parser.add_argument(
        "--delta-scales",
        nargs="+",
        type=float,
        default=[4.0, 2.0, 1.0, 0.5],
        metavar="S",
        help="Fractions of current sensitivity used as δ (default: 4.0 2.0 1.0 0.5)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Repeated TraceWin runs for each +δ and -δ point (default: %(default)s)",
    )
    parser.add_argument(
        "--min-score-diff",
        type=float,
        default=0.5,
        help=(
            "Minimum accepted aggregated |score(+δ)-score(-δ)|. "
            "Smaller differences are treated as noise (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--aggregation",
        choices=["median", "mean"],
        default="median",
        help="How to aggregate repeated scores (default: %(default)s)",
    )
    parser.add_argument(
        "--stability-low",
        type=float,
        default=0.7,
        help="Lower accepted adjacent stability ratio (default: %(default)s)",
    )
    parser.add_argument(
        "--stability-high",
        type=float,
        default=1.3,
        help="Upper accepted adjacent stability ratio (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="TraceWin timeout per simulation in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_SENSITIVITY_CHECKPOINT),
        help=(
            "JSON checkpoint updated after each completed parameter/scale "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore an existing checkpoint and recompute from scratch.",
    )
    args = parser.parse_args()

    print(f"TraceWin project : {args.ini}")
    print(f"Calc dir         : {args.calc_dir}")
    print(f"Checkpoint       : {args.checkpoint}")
    print(f"Resume           : {not args.no_resume}")
    print(f"Delta scales     : {args.delta_scales}")
    print(f"Repeats          : {args.repeats}")
    print(f"Aggregation      : {args.aggregation}")
    print(f"Min score diff   : {args.min_score_diff}")
    print(f"Stability window : [{args.stability_low}, {args.stability_high}]")
    print(f"Parameters       : {len(PARAMETERS)}")
    print(f"Total TW runs    : {len(PARAMETERS) * len(args.delta_scales) * 2 * args.repeats}")
    print()

    simulator = TraceWinSimulator(
        project_file=args.ini,
        calc_dir=args.calc_dir,
        timeout=args.timeout,
    )

    sensitivity_results = compute_sensitivity(
        simulator,
        delta_scales=args.delta_scales,
        repeats=args.repeats,
        min_score_diff=args.min_score_diff,
        aggregation=args.aggregation,
        checkpoint_path=args.checkpoint,
        resume=not args.no_resume,
        verbose=True,
    )

    print_sensitivity_report(
        sensitivity_results,
        delta_scales=args.delta_scales,
        stability_low=args.stability_low,
        stability_high=args.stability_high,
    )
