"""Registry of the custom RL algorithms.

The seven model-free classes share the same constructor signature
(obs_dim, act_dim, action_bounds, hidden_dims=..., **hyperparams) and the
select_action/store/optimize/save/load convention, so scripts can build any
of them through make_agent(). SB3-SAC, SVG and MBPO need an env/surrogate/
dataset at construction time and are built explicitly by the scripts.

Adding an algorithm = one line in _REGISTRY.
"""
from __future__ import annotations

from importlib import import_module

_REGISTRY: dict[str, tuple[str, str]] = {
    "sac":       ("beam_optimization.algorithms.model_free.sac",       "SAC"),
    "td3":       ("beam_optimization.algorithms.model_free.td3",       "TD3"),
    "ppo":       ("beam_optimization.algorithms.model_free.ppo",       "PPO"),
    "ddpg":      ("beam_optimization.algorithms.model_free.ddpg",      "DDPG"),
    "a2c":       ("beam_optimization.algorithms.model_free.a2c",       "A2C"),
    "reinforce": ("beam_optimization.algorithms.model_free.reinforce", "REINFORCE"),
    "trpo":      ("beam_optimization.algorithms.model_free.trpo",      "TRPO"),
}

MODEL_FREE_ALGORITHMS: tuple[str, ...] = tuple(_REGISTRY)

# On-policy agents use store(state, action, reward, value, logpa, done) +
# optimize(last_value); off-policy ones use store(s, a, r, ns, done) + optimize().
ON_POLICY_ALGORITHMS: frozenset[str] = frozenset({"ppo", "a2c", "reinforce", "trpo"})


def make_agent(name: str, obs_dim: int, act_dim: int, action_bounds,
               hidden_dims=(256, 256), **kwargs):
    """Build one of the custom model-free agents by name (lazy import)."""
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown algorithm '{name}'. Available: {', '.join(MODEL_FREE_ALGORITHMS)}"
        )
    module_name, class_name = _REGISTRY[key]
    cls = getattr(import_module(module_name), class_name)
    return cls(obs_dim, act_dim, action_bounds, hidden_dims=tuple(hidden_dims), **kwargs)


def load_agent(name: str, checkpoint: str, obs_dim: int, act_dim: int,
               action_bounds, hidden_dims=(256, 256)):
    """Build a custom model-free agent and load its checkpoint."""
    agent = make_agent(name, obs_dim, act_dim, action_bounds, hidden_dims=hidden_dims)
    agent.load(checkpoint)
    return agent


def is_on_policy(name: str) -> bool:
    return name.lower() in ON_POLICY_ALGORITHMS
