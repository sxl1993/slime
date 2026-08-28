from __future__ import annotations

import math

import torch

from slime.backends.megatron_utils.sao import compute_explained_variance
from slime.utils.sao import run_critic_updates


def test_critic_updates_refresh_values_after_all_updates():
    events = []

    refreshed_values = run_critic_updates(
        lambda: events.append("critic"),
        lambda: events.append("refresh") or "updated-values",
        update_count=2,
    )

    assert events == ["critic", "critic", "refresh"]
    assert refreshed_values == "updated-values"


def test_explained_variance_uses_only_action_tokens():
    values = torch.tensor([1.0, 100.0, 5.0])
    returns = torch.tensor([1.0, 999.0, 5.0])
    loss_mask = torch.tensor([1, 0, 1])

    explained_variance = compute_explained_variance(values, returns, loss_mask)

    torch.testing.assert_close(explained_variance, torch.tensor(1.0, dtype=explained_variance.dtype))


def test_explained_variance_can_be_negative():
    values = torch.tensor([2.0, 1.0, 0.0])
    returns = torch.tensor([0.0, 1.0, 2.0])
    loss_mask = torch.ones(3, dtype=torch.int32)

    explained_variance = compute_explained_variance(values, returns, loss_mask)

    torch.testing.assert_close(explained_variance, torch.tensor(-3.0, dtype=explained_variance.dtype))


def test_explained_variance_is_nan_for_constant_returns():
    values = torch.tensor([1.0, 2.0])
    returns = torch.ones(2)
    loss_mask = torch.ones(2, dtype=torch.int32)

    explained_variance = compute_explained_variance(values, returns, loss_mask)

    assert math.isnan(explained_variance.item())
