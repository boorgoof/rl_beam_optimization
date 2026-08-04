"""Summarize a directory of qualitative policy-test JSON files."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def summarize(test_root: Path) -> list[dict]:
    rows: list[dict] = []
    for result_path in sorted(test_root.glob("*/test.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        episodes = payload.get("episodes", [payload["episode"]])
        final_scores = [float(episode["final_score"]) for episode in episodes]
        total_rewards = [float(episode["total_reward"]) for episode in episodes]
        best_scores = []
        losses = []
        action_norms = []
        n_steps = []
        npart_ratios = []
        for episode in episodes:
            steps = episode.get("steps", [])
            scores = [
                float(step["score"])
                for step in steps
                if math.isfinite(float(step.get("score", math.nan)))
            ]
            episode_final = float(episode["final_score"])
            episode_best = max(scores, default=episode_final)
            best_scores.append(episode_best)
            losses.append(episode_best - episode_final)
            action_norms.extend(
                float(step["action_norm"])
                for step in steps
                if math.isfinite(float(step.get("action_norm", math.nan)))
            )
            n_steps.append(int(episode.get("n_steps", len(steps))))
            npart_ratios.append(float(episode.get("final_features", {}).get("npart_ratio", math.nan)))

        finite_npart = [value for value in npart_ratios if math.isfinite(value)]
        rows.append(
            {
                "configuration": result_path.parent.name,
                "n_episodes": len(episodes),
                "final_score": statistics.fmean(final_scores),
                "final_score_std": statistics.pstdev(final_scores),
                "best_score": statistics.fmean(best_scores),
                "best_to_final_loss": statistics.fmean(losses),
                "total_reward": statistics.fmean(total_rewards),
                "n_steps": statistics.fmean(n_steps),
                "rms_action_norm": (
                    math.sqrt(sum(value * value for value in action_norms) / len(action_norms))
                    if action_norms
                    else 0.0
                ),
                "final_npart_ratio": statistics.fmean(finite_npart) if finite_npart else math.nan,
                "policy": str(payload["policy"]),
            }
        )
    return sorted(rows, key=lambda row: row["final_score"], reverse=True)


def save_csv(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["configuration"] for row in rows]
    panels = (
        ("final_score", "Final score"),
        ("best_to_final_loss", "Best → final loss"),
        ("total_reward", "Total reward"),
        ("final_npart_ratio", "Final npart ratio"),
    )
    fig_width = max(14.0, 0.5 * len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, 9))
    for axis, (key, title) in zip(axes.flat, panels):
        values = [row[key] for row in rows]
        bars = axis.bar(range(len(rows)), values)
        axis.set_title(title)
        axis.set_xticks(range(len(rows)), labels, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            if math.isfinite(float(value)):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.3g}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    fig.suptitle("SAC penalty/horizon qualitative test grid")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-root", required=True, type=Path)
    args = parser.parse_args()

    rows = summarize(args.test_root)
    if not rows:
        raise FileNotFoundError(f"No */test.json files found in {args.test_root}")

    csv_path = args.test_root / "grid_summary.csv"
    plot_path = args.test_root / "grid_summary.png"
    save_csv(rows, csv_path)
    save_plot(rows, plot_path)

    print("\nSAC GRID TEST SUMMARY")
    for rank, row in enumerate(rows, 1):
        print(
            f"{rank:2d}. {row['configuration']:<34} "
            f"final={row['final_score']:.4f}±{row['final_score_std']:.4f} "
            f"best-final={row['best_to_final_loss']:.4f} "
            f"npart={row['final_npart_ratio']:.4f}"
        )
    print(f"CSV:  {csv_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
