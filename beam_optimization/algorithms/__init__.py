"""Public algorithm names and model-free factories.

Stable Baselines3 implementations use their plain algorithm names. The
project implementations use an explicit ``_custom`` suffix so checkpoints,
plots, and CLI flags are never ambiguous.
"""
from __future__ import annotations

from importlib import import_module

_CUSTOM_REGISTRY: dict[str, tuple[str, str]] = {
    "sac_custom":       ("beam_optimization.algorithms.model_free.sac",       "SAC"),
    "td3_custom":       ("beam_optimization.algorithms.model_free.td3",       "TD3"),
    "ppo_custom":       ("beam_optimization.algorithms.model_free.ppo",       "PPO"),
    "ddpg_custom":      ("beam_optimization.algorithms.model_free.ddpg",      "DDPG"),
    "a2c_custom":       ("beam_optimization.algorithms.model_free.a2c",       "A2C"),
    "reinforce_custom": ("beam_optimization.algorithms.model_free.reinforce", "REINFORCE"),
    "trpo_custom":      ("beam_optimization.algorithms.model_free.trpo",      "TRPO"),
}

STABLE_BASELINES_ALGORITHMS: tuple[str, ...] = ("sac", "ppo", "td3", "ddpg", "a2c")
CUSTOM_MODEL_FREE_ALGORITHMS: tuple[str, ...] = tuple(_CUSTOM_REGISTRY)
MODEL_FREE_ALGORITHMS: tuple[str, ...] = (
    *STABLE_BASELINES_ALGORITHMS,
    *CUSTOM_MODEL_FREE_ALGORITHMS,
)
MODEL_BASED_ALGORITHMS: tuple[str, ...] = (
    "mbpo",
    "svg_final",
    "svg_uniform",
    "iterative_sim2real_sac",
)
ALGORITHMS: tuple[str, ...] = (*MODEL_FREE_ALGORITHMS, *MODEL_BASED_ALGORITHMS)
LEGACY_ALGORITHM_ALIASES = {"sb3_sac": "sac"}

# On-policy agents use store(state, action, reward, value, logpa, done) +
# optimize(last_value); off-policy ones use store(s, a, r, ns, done) + optimize().
CUSTOM_ON_POLICY_ALGORITHMS: frozenset[str] = frozenset(
    {"ppo_custom", "a2c_custom", "reinforce_custom", "trpo_custom"}
)


def canonical_algorithm_name(name: str) -> str:
    """Resolve supported legacy public names to their canonical identifier."""
    key = name.lower()
    return LEGACY_ALGORITHM_ALIASES.get(key, key)


def make_custom_agent(name: str, obs_dim: int, act_dim: int, action_bounds,
                      hidden_dims=(256, 256), **kwargs):
    """Build one of the custom model-free agents by suffixed public name."""
    key = canonical_algorithm_name(name)
    if key not in _CUSTOM_REGISTRY:
        raise ValueError(
            f"Unknown custom algorithm '{name}'. Available: "
            f"{', '.join(CUSTOM_MODEL_FREE_ALGORITHMS)}"
        )
    module_name, class_name = _CUSTOM_REGISTRY[key]
    cls = getattr(import_module(module_name), class_name)
    return cls(obs_dim, act_dim, action_bounds, hidden_dims=tuple(hidden_dims), **kwargs)


def load_custom_agent(name: str, checkpoint: str, obs_dim: int, act_dim: int,
                      action_bounds, hidden_dims=(256, 256)):
    """Build a custom model-free agent and load its checkpoint."""
    agent = make_custom_agent(
        name, obs_dim, act_dim, action_bounds, hidden_dims=hidden_dims
    )
    agent.load(checkpoint)
    return agent


def is_custom_on_policy(name: str) -> bool:
    return canonical_algorithm_name(name) in CUSTOM_ON_POLICY_ALGORITHMS


# Compatibility aliases for Python callers. Public algorithm identifiers must
# still use the explicit ``_custom`` suffix.
make_agent = make_custom_agent
load_agent = load_custom_agent
is_on_policy = is_custom_on_policy
