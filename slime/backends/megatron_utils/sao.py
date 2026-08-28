"""Pure SAO advantage and importance-sampling helpers."""

from __future__ import annotations

import torch
import torch.distributed as dist


def compute_explained_variance(
    values: torch.Tensor,
    returns: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    process_group=None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute explained variance over valid action tokens."""
    if values.shape != returns.shape or returns.shape != loss_mask.shape:
        raise ValueError("values, returns, and loss_mask must have the same shape")

    valid_values = values.reshape(-1).to(dtype=torch.float64)[loss_mask.reshape(-1).bool()]
    valid_returns = returns.reshape(-1).to(dtype=torch.float64)[loss_mask.reshape(-1).bool()]
    residuals = valid_returns - valid_values
    stats = torch.stack(
        (
            torch.tensor(valid_returns.numel(), dtype=torch.float64, device=values.device),
            valid_returns.sum(),
            valid_returns.square().sum(),
            residuals.sum(),
            residuals.square().sum(),
        )
    )
    if process_group is not None:
        dist.all_reduce(stats, group=process_group)

    count = stats[0]
    safe_count = count.clamp_min(1.0)
    target_mean = stats[1] / safe_count
    target_variance = stats[2] / safe_count - target_mean.square()
    residual_mean = stats[3] / safe_count
    residual_variance = stats[4] / safe_count - residual_mean.square()
    explained_variance = 1.0 - residual_variance / target_variance.clamp_min(eps)
    return torch.where((count >= 2) & (target_variance > eps), explained_variance, torch.nan)


def compute_skip_observation_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    gamma: float,
    lambda_: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute token-level GAE while skipping non-action tokens."""
    if rewards.shape != values.shape or values.shape != loss_mask.shape:
        raise ValueError("rewards, values, and loss_mask must have the same shape")

    action_indices = loss_mask.to(dtype=torch.bool).nonzero(as_tuple=True)[0]
    advantages = torch.zeros_like(values)
    returns = torch.zeros_like(values)
    if action_indices.numel() == 0:
        return advantages, returns

    action_values = values.index_select(0, action_indices)
    action_rewards = rewards.index_select(0, action_indices)
    next_values = torch.cat((action_values[1:], torch.zeros_like(action_values[:1])), dim=0)
    deltas = action_rewards + gamma * next_values - action_values

    discount = gamma * lambda_
    if discount == 0.0:
        action_advantages = deltas
    else:
        # Reuse the chunked scan used by generic GAE. A full-sequence
        # power/cumsum formulation underflows for long responses.
        from slime.utils.ppo_utils import chunked_discounted_returns

        action_advantages = chunked_discounted_returns(deltas.unsqueeze(0), discount)[0]

    advantages.index_copy_(0, action_indices, action_advantages)
    returns.index_copy_(0, action_indices, action_values + action_advantages)

    return advantages, returns


def compute_sao_dis_weights(
    train_log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    *,
    clip_low: float,
    clip_high: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return SAO's ratio, rejection weight, and valid-token mask."""
    if train_log_probs.shape != rollout_log_probs.shape:
        raise ValueError("train_log_probs and rollout_log_probs must have the same shape")

    ratios = torch.exp(train_log_probs - rollout_log_probs)
    lower = 1.0 - clip_low
    upper = 1.0 + clip_high
    valid = (ratios >= lower) & (ratios <= upper)
    weights = torch.where(valid, ratios, torch.zeros_like(ratios))
    return ratios, weights, valid
