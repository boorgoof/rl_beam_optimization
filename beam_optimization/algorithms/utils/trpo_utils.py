"""TRPO utility functions: conjugate gradient, Fisher-vector product,
KL divergence, flat-param helpers, backtracking line search.
Adapted from reinforcement_learning_2/rl/utils/trpo_utils.py.

Reference:
    Schulman J. et al., "Trust Region Policy Optimization", ICML 2015.
    https://arxiv.org/abs/1502.05477
"""
import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Callable


def conjugate_gradient(Avp_fn: Callable, b: torch.Tensor,
                       n_steps: int = 10, tol: float = 1e-10) -> torch.Tensor:
    """Solve Ax = b via CG without forming A. A is the Fisher matrix."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    r_dot_r = torch.dot(r, r)
    for _ in range(n_steps):
        Ap      = Avp_fn(p)
        alpha   = r_dot_r / (torch.dot(p, Ap) + 1e-8)
        x       = x + alpha * p
        r       = r - alpha * Ap
        new_rdr = torch.dot(r, r)
        if new_rdr < tol:
            break
        beta    = new_rdr / (r_dot_r + 1e-8)
        p       = r + beta * p
        r_dot_r = new_rdr
    return x


def fisher_vector_product(policy: nn.Module, states: torch.Tensor,
                          vector: torch.Tensor, damping: float = 0.1) -> torch.Tensor:
    """Compute F*v via double-backprop on the KL divergence (Gaussian policy)."""
    with torch.no_grad():
        mean_old, log_std_old = policy.forward(states)
        mean_old    = mean_old.detach()
        log_std_old = log_std_old.detach()

    mean_new, log_std_new = policy.forward(states)
    dist_old = Normal(mean_old, log_std_old.exp())
    dist_new = Normal(mean_new, log_std_new.exp())
    kl = torch.distributions.kl_divergence(dist_old, dist_new).sum(-1).mean()

    params    = [p for p in policy.parameters() if p.requires_grad]
    grads     = torch.autograd.grad(kl, params, create_graph=True, allow_unused=True)
    flat_grad = torch.cat([
        g.contiguous().view(-1) if g is not None else torch.zeros_like(p).view(-1)
        for g, p in zip(grads, params)])

    grad_v  = (flat_grad * vector).sum()
    grads2  = torch.autograd.grad(grad_v, params, allow_unused=True)
    flat_hv = torch.cat([
        g.contiguous().view(-1) if g is not None else torch.zeros_like(p).view(-1)
        for g, p in zip(grads2, params)])
    return flat_hv + damping * vector


def compute_kl_divergence(policy: nn.Module, states: torch.Tensor,
                           old_dist_params) -> torch.Tensor:
    mean_old, log_std_old = old_dist_params
    mean_new, log_std_new = policy.forward(states)
    dist_old = Normal(mean_old, log_std_old.exp())
    dist_new = Normal(mean_new, log_std_new.exp())
    return torch.distributions.kl_divergence(dist_old, dist_new).sum(-1).mean()


def get_flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in model.parameters() if p.requires_grad])


def set_flat_params(model: nn.Module, flat_params: torch.Tensor) -> None:
    idx = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        p.data.copy_(flat_params[idx: idx + n].view_as(p))
        idx += n


def get_flat_grad(loss: torch.Tensor, model: nn.Module,
                  retain_graph: bool = False) -> torch.Tensor:
    params = [p for p in model.parameters() if p.requires_grad]
    grads  = torch.autograd.grad(loss, params,
                                  retain_graph=retain_graph, allow_unused=True)
    return torch.cat([
        g.contiguous().view(-1) if g is not None else torch.zeros_like(p).view(-1)
        for g, p in zip(grads, params)])


def _surrogate(policy, states, actions, advantages, old_log_probs):
    log_probs = policy.log_prob(states, actions)
    ratios = (log_probs - old_log_probs).exp()
    return (ratios * advantages.unsqueeze(1)).mean()


def line_search(policy, states, actions, advantages, old_log_probs,
                old_dist_params, full_step, max_kl,
                max_backtracks=10, accept_ratio=0.1) -> bool:
    """Backtracking line search to enforce KL ≤ max_kl and surrogate improvement."""
    old_params  = get_flat_params(policy)
    surr_before = _surrogate(policy, states, actions, advantages, old_log_probs).item()
    step_size   = 1.0
    for _ in range(max_backtracks):
        set_flat_params(policy, old_params + step_size * full_step)
        with torch.no_grad():
            kl   = compute_kl_divergence(policy, states, old_dist_params).item()
            surr = _surrogate(policy, states, actions, advantages, old_log_probs).item()
        if kl <= max_kl and surr - surr_before > accept_ratio * step_size:
            return True
        step_size *= 0.5
    set_flat_params(policy, old_params)
    return False
