from __future__ import annotations

# Install the lightweight Megatron stub before importing the CP-aware helper.
import _cp_dist_helpers  # noqa: F401
import pytest
import torch

from slime.backends.megatron_utils.sao import compute_sao_dis_weights, compute_skip_observation_gae
from slime.utils.ppo_utils import get_advantages_and_returns_batch


def _reference_skip_observation_gae(rewards, values, loss_mask, *, gamma, lambda_):
    """Independent scalar reference for the action-token GAE recurrence."""
    device = rewards.device
    dtype = values.dtype
    reward_values = rewards.detach().cpu().tolist()
    value_values = values.detach().cpu().tolist()
    mask_values = loss_mask.detach().cpu().tolist()

    advantages = [0.0] * len(value_values)
    returns = [0.0] * len(value_values)
    action_indices = [index for index, is_action in enumerate(mask_values) if is_action]
    next_advantage = 0.0

    for action_offset in range(len(action_indices) - 1, -1, -1):
        index = action_indices[action_offset]
        next_value = (
            value_values[action_indices[action_offset + 1]] if action_offset + 1 < len(action_indices) else 0.0
        )
        delta = reward_values[index] + gamma * next_value - value_values[index]
        next_advantage = delta + gamma * lambda_ * next_advantage
        advantages[index] = next_advantage
        returns[index] = value_values[index] + next_advantage

    return torch.tensor(advantages, device=device, dtype=dtype), torch.tensor(returns, device=device, dtype=dtype)


def test_skip_observation_gae_bridges_only_action_tokens():
    rewards = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
    values = torch.tensor([0.5, 9.0, 0.4, 8.0, 0.2])
    loss_mask = torch.tensor([1, 0, 1, 0, 1])

    advantages, returns = compute_skip_observation_gae(rewards, values, loss_mask, gamma=1.0, lambda_=1.0)

    torch.testing.assert_close(advantages, torch.tensor([0.5, 0.0, 0.6, 0.0, 0.8]))
    torch.testing.assert_close(returns, torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0]))


def test_skip_observation_gae_matches_nontrivial_known_values():
    rewards = torch.tensor([0.1, 99.0, 0.2, 88.0, 77.0, 1.0])
    values = torch.tensor([0.5, 9.0, 0.4, 8.0, 7.0, 0.3])
    loss_mask = torch.tensor([1, 0, 1, 0, 0, 1])

    advantages, returns = compute_skip_observation_gae(rewards, values, loss_mask, gamma=0.9, lambda_=0.8)

    torch.testing.assert_close(advantages, torch.tensor([0.37328, 0.0, 0.574, 0.0, 0.0, 0.7]))
    torch.testing.assert_close(returns, torch.tensor([0.87328, 0.0, 0.974, 0.0, 0.0, 1.0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the numerical regression")
@pytest.mark.parametrize("length", [1, 7, 129, 1024, 4096])
@pytest.mark.parametrize(("gamma", "lambda_"), [(0.0, 0.95), (0.99, 0.95), (1.0, 1.0)])
def test_skip_observation_gae_matches_scalar_reference_on_cuda(length, gamma, lambda_):
    device = torch.device("cuda")
    torch.manual_seed(length)
    rewards = torch.randn(length, device=device)
    values = torch.randn(length, device=device)
    loss_mask = torch.tensor([(index * 7 + 3) % 5 != 0 for index in range(length)], device=device)

    expected_advantages, expected_returns = _reference_skip_observation_gae(
        rewards, values, loss_mask, gamma=gamma, lambda_=lambda_
    )
    actual_advantages, actual_returns = compute_skip_observation_gae(
        rewards, values, loss_mask, gamma=gamma, lambda_=lambda_
    )

    torch.testing.assert_close(actual_advantages, expected_advantages, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual_returns, expected_returns, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required to detect device scalar reads")
def test_skip_observation_gae_does_not_read_cuda_indices_as_scalars():
    device = torch.device("cuda")
    rewards = torch.zeros(8, device=device)
    rewards[-1] = 1.0
    values = torch.linspace(0.1, 0.8, 8, device=device)
    loss_mask = torch.tensor([1, 0, 1, 0, 1, 0, 1, 1], device=device)
    torch.cuda.synchronize()

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
        compute_skip_observation_gae(rewards, values, loss_mask, gamma=0.99, lambda_=0.95)
        torch.cuda.synchronize()

    item_count = sum(event.count for event in profiler.key_averages() if event.key == "aten::item")
    assert item_count == 0


def test_sao_dis_masks_ratios_outside_strict_bounds():
    train_log_probs = torch.log(torch.tensor([0.1, 0.5, 1.0, 4.0, 4.1]))
    rollout_log_probs = torch.zeros(5)

    ratios, weights, mask = compute_sao_dis_weights(
        train_log_probs,
        rollout_log_probs,
        clip_low=0.8,
        clip_high=3.0,
    )

    torch.testing.assert_close(ratios, torch.tensor([0.1, 0.5, 1.0, 4.0, 4.1]))
    torch.testing.assert_close(weights, torch.tensor([0.0, 0.5, 1.0, 4.0, 0.0]))
    assert mask.tolist() == [False, True, True, True, False]


def test_sao_cp_keeps_loss_mask_in_response_space(monkeypatch):
    """SAO must not CP-gather the already-complete response loss mask."""
    from megatron.core import mpu

    import slime.backends.megatron_utils.cp_utils as cp_utils

    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: 0)

    gathered = []

    def fake_all_gather(tensor, total_length, response_length):
        gathered.append(tensor)
        if tensor.dtype == torch.int32:
            raise AssertionError("SAO loss masks must not be passed to all_gather_with_cp")
        return torch.arange(response_length, dtype=tensor.dtype)

    monkeypatch.setattr(cp_utils, "all_gather_with_cp", fake_all_gather)

    advantages, returns = get_advantages_and_returns_batch(
        total_lengths=[12],
        response_lengths=[8],
        values_list=[torch.zeros(2)],
        rewards_list=[torch.zeros(2)],
        gamma=1.0,
        lambd=1.0,
        loss_masks=[torch.ones(8, dtype=torch.int32)],
        skip_observation=True,
        terminal_rewards=[1.0],
    )

    assert len(gathered) == 2  # values and rewards only
    assert advantages[0].numel() == returns[0].numel()


def test_sao_length_adaptive_gae_uses_inverse_alpha_length():
    advantages, returns = get_advantages_and_returns_batch(
        total_lengths=[10],
        response_lengths=[10],
        values_list=[torch.zeros(10)],
        rewards_list=[torch.zeros(10)],
        gamma=1.0,
        lambd=1.0,
        loss_masks=[torch.ones(10, dtype=torch.int32)],
        skip_observation=True,
        length_adaptive_alpha=1.5,
        terminal_rewards=[1.0],
    )

    torch.testing.assert_close(advantages[0][0], torch.tensor(0.53744124), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(returns[0][0], torch.tensor(0.53744124), atol=1e-5, rtol=1e-5)
