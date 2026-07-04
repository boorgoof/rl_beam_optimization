"""Test one trained policy on one qualitative episode.

This command is intentionally separate from benchmark:
benchmark compares methods numerically over many episodes, while test runs one
trained policy for one episode and can save qualitative render images.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

from beam_optimization.algorithms import MODEL_FREE_ALGORITHMS, load_agent
from beam_optimization.config.adige import MAX_STEPS, N_PARAMS, action_bounds
from beam_optimization.config.paths import (
    DEFAULT_BASE_DATASET,
    DEFAULT_BASE_SURROGATE_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRACEWIN_INI,
    configure_matplotlib_cache,
    default_eval_calc_dir,
)
from beam_optimization.env.surrogate_env import SurrogateEnv
from beam_optimization.env.dataset import BeamDataset
from beam_optimization.env.surrogate_env.surrogate.model.modular_mlp import ModularMLP
from beam_optimization.scripts.common import run_episode as run_common_episode


ACT_DIM = N_PARAMS


def load_surrogate(path: str | Path):
    """Load one ModularMLP or an ensemble from a directory."""
    path = Path(path)
    if path.is_dir():
        model_files = sorted(path.glob("surrogate_*.pt"))
        if not model_files:
            raise FileNotFoundError(f"No surrogate_*.pt files found in {path}")
        models = [ModularMLP.load(str(p)) for p in model_files]
        for model in models:
            model.eval()
        return models

    model = ModularMLP.load(str(path))
    model.eval()
    return model


def make_env(args):
    """Create the selected test environment."""
    if args.env == "surrogate":
        surrogate = load_surrogate(args.surrogate)
        dataset = BeamDataset.load(args.dataset)
        return SurrogateEnv(
            model=surrogate,
            dataset=dataset,
            max_steps=args.max_ep_steps,
        )

    from beam_optimization.env.tracewin_env import TraceWinEnv

    project_file = Path(args.tracewin_project)
    calc_dir = Path(args.calc_dir) if args.calc_dir else default_eval_calc_dir(project_file)
    return TraceWinEnv(
        project_file=str(project_file),
        calc_dir=str(calc_dir),
        max_steps=args.max_ep_steps,
        timeout=args.tracewin_timeout,
    )


def make_agent(algo: str, policy_path: str, obs_dim: int, hidden: list[int], env=None):
    """Instantiate and load a trained policy."""
    if algo == "sb3_sac":
        from beam_optimization.algorithms.model_free.sb3_sac import SB3SAC
        if env is None:
            raise ValueError("SB3 SAC loading requires the test env.")
        return SB3SAC.load(policy_path, env=env)

    bounds = action_bounds()
    return load_agent(algo, policy_path, obs_dim, ACT_DIM,
                      (bounds[0].tolist(), bounds[1].tolist()), hidden_dims=hidden)


def algorithm_render_dir(args) -> Path:
    """Return the per-algorithm render directory under the requested base dir."""
    base_dir = Path(args.render_dir)
    if base_dir.name == args.algo:
        return base_dir
    return base_dir / args.algo


def save_render(env, args, episode_idx: int, step_idx: int) -> None:
    configure_matplotlib_cache()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    render_dir = algorithm_render_dir(args)
    render_dir.mkdir(parents=True, exist_ok=True)

    prefix = render_dir / f"{args.env}_{args.algo}_ep{episode_idx:03d}_step{step_idx:03d}"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive.*")
        result = env.render()
    result["params"].savefig(prefix.with_name(prefix.name + "_params.png"), dpi=args.dpi)
    result["state"].savefig(prefix.with_name(prefix.name + "_state.png"), dpi=args.dpi)
    result["score"].savefig(prefix.with_name(prefix.name + "_score.png"), dpi=args.dpi)
    plt.close(result["params"])
    plt.close(result["state"])
    plt.close(result["score"])

    if args.env == "tracewin" and args.tracewin_phase_space:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive.*")
            phase_fig = env.render_final_beam_distribution(
                max_particles=args.max_particles,
                bins=args.bins,
            )
        if phase_fig is not None:
            phase_fig.savefig(prefix.with_name(prefix.name + "_phase_space.png"), dpi=args.dpi)
            plt.close(phase_fig)


def save_episode_video(env, args, episode_idx: int) -> None:
    """Save the parameter/beam-feature trend GIFs for a just-finished episode."""
    configure_matplotlib_cache()

    import matplotlib
    matplotlib.use("Agg")

    render_dir = algorithm_render_dir(args)
    render_dir.mkdir(parents=True, exist_ok=True)
    prefix = render_dir / f"{args.env}_{args.algo}_ep{episode_idx:03d}_episode"

    result = env.render(save_path=str(prefix), fps=args.episode_video_fps)
    print(f"  saved episode videos: {result['params_video']}, {result['state_video']}, {result['score_video']}")


def save_phase_space_frame(env, args, episode_idx: int, step_idx: int) -> Path | None:
    """Save one TraceWin phase-space distribution frame (x-y, x-x', y-y') for the
    episode video. Returns None if no .dst file is available yet (e.g. right
    after a failed simulation)."""
    configure_matplotlib_cache()

    import matplotlib
    matplotlib.use("Agg")

    render_dir = algorithm_render_dir(args)
    render_dir.mkdir(parents=True, exist_ok=True)
    frame_path = render_dir / f"{args.env}_{args.algo}_ep{episode_idx:03d}_phaseframe{step_idx:03d}.png"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive.*")
        fig = env.render_final_beam_distribution(
            max_particles=args.max_particles,
            bins=args.bins,
        )
    if fig is None:
        return None

    import matplotlib.pyplot as plt

    fig.savefig(frame_path, dpi=args.dpi)
    plt.close(fig)
    return frame_path


def save_phase_space_video(frame_paths: list[Path], args, episode_idx: int) -> None:
    """Assemble collected phase-space frames into one GIF, then delete the frames."""
    from PIL import Image

    render_dir = algorithm_render_dir(args)
    gif_path = render_dir / f"{args.env}_{args.algo}_ep{episode_idx:03d}_episode_phase_space.gif"

    images = [Image.open(p) for p in frame_paths]
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / args.episode_video_fps),
        loop=0,
    )
    for image in images:
        image.close()
    for frame_path in frame_paths:
        frame_path.unlink(missing_ok=True)

    print(f"  saved phase-space episode video: {gif_path}")


def run_episode(env, agent, args, episode_idx: int) -> dict:
    """Run one qualitative episode via the shared runner, with rendering hooks."""
    want_phase_space_video = (
        args.env == "tracewin" and args.episode_video and args.tracewin_phase_space
    )
    phase_space_frames: list[Path] = []

    def on_step(step_idx: int, env_, info: dict, done: bool) -> None:
        if step_idx == 0:
            print(f"  reset_randomized={info.get('reset_randomized', True)}")
        else:
            print(
                f"  ep={episode_idx} step={step_idx:02d} "
                f"reward={info.get('score', 0.0) - info.get('prev_score', 0.0):.4g} "
                f"score={info.get('score', float('nan')):.4g}"
            )
        if args.render and (step_idx == 0 or step_idx % args.render_every == 0 or done):
            save_render(env_, args, episode_idx, step_idx)
        if want_phase_space_video:
            frame_path = save_phase_space_frame(env_, args, episode_idx, step_idx)
            if frame_path is not None:
                phase_space_frames.append(frame_path)

    reset_options = {"randomize_params": False} if args.deterministic_reset else None
    result = run_common_episode(
        env, agent,
        seed=args.seed + episode_idx,
        reset_options=reset_options,
        step_callback=on_step,
    )

    if args.episode_video:
        save_episode_video(env, args, episode_idx)
        if len(phase_space_frames) >= 2:
            save_phase_space_video(phase_space_frames, args, episode_idx)

    return {"episode": episode_idx, **result}


def main():
    parser = argparse.ArgumentParser(
        description="Run one trained policy for one qualitative test episode."
    )
    parser.add_argument("--algo", required=True,
                        choices=[*MODEL_FREE_ALGORITHMS, "sb3_sac"])
    parser.add_argument("--policy", required=True, help="Path to the trained policy checkpoint.")
    parser.add_argument("--env", default="surrogate", choices=["surrogate", "tracewin"])
    parser.add_argument("--max-ep-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic-reset", action="store_true",
                        help="Start the episode from nominal default parameters instead of a randomized reset.")

    parser.add_argument("--surrogate", default=str(DEFAULT_BASE_SURROGATE_DIR))
    parser.add_argument("--dataset", default=str(DEFAULT_BASE_DATASET))
    parser.add_argument("--tracewin-project", default=str(DEFAULT_TRACEWIN_INI))
    parser.add_argument("--calc-dir", default=None)
    parser.add_argument("--tracewin-timeout", type=float, default=120.0)

    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "test.json"))
    parser.add_argument("--render", action="store_true", help="Save render PNG files during the test episode.")
    parser.add_argument("--render-dir", default=str(DEFAULT_OUTPUT_DIR / "renders"))
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument("--episode-video", action="store_true",
                        help="Save parameter/beam-feature trend GIFs for the test episode into --render-dir.")
    parser.add_argument("--episode-video-fps", type=int, default=2,
                        help="Frame rate for --episode-video GIFs (default: %(default)s)")
    parser.add_argument("--tracewin-phase-space", action=argparse.BooleanOptionalAction, default=True,
                        help="For TraceWin render, also save true final phase-space images from .dst files.")
    parser.add_argument("--max-particles", type=int, default=40000)
    parser.add_argument("--bins", type=int, default=150)
    args = parser.parse_args()

    if not Path(args.policy).exists():
        raise FileNotFoundError(args.policy)

    env = make_env(args)
    obs_dim = env.observation_space.shape[0]
    agent = make_agent(args.algo, args.policy, obs_dim, args.hidden, env=env)

    print(f"Environment: {args.env}")
    print(f"Policy:      {args.policy}")
    print(f"Observation: configured mask ({obs_dim} dims)")
    if args.render or args.episode_video:
        print(f"Render dir:  {algorithm_render_dir(args)}")

    print("\nTest episode")
    result = run_episode(env, agent, args, 0)

    summary = {
        "algo": args.algo,
        "policy": args.policy,
        "env": args.env,
        "observation_dim": obs_dim,
        "episode": result,
        "final_score": float(result["final_score"]),
        "total_reward": float(result["total_reward"]),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nTEST SUMMARY")
    print(f"  final_score:  {summary['final_score']:.6g}")
    print(f"  total_reward: {summary['total_reward']:.6g}")
    print(f"  saved: {output}")


if __name__ == "__main__":
    main()
