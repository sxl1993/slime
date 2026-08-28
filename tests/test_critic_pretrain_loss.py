from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.coding_agent_rl.critic_pretrain.loss import (  # noqa: E402
    ValueMetricAccumulator,
    trajectory_equal_masked_mse,
)


def test_trajectory_equal_masked_mse_weights_trajectories_equally():
    loss = trajectory_equal_masked_mse(
        values=[torch.tensor([0.0, 0.0]), torch.ones(8)],
        targets=[torch.ones(2), torch.zeros(8)],
        masks=[torch.ones(2), torch.ones(8)],
    )
    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_trajectory_equal_masked_mse_ignores_masked_positions():
    values = [torch.tensor([0.0, 0.0, 0.0])]
    masks = [torch.tensor([1.0, 0.0, 1.0])]
    first = trajectory_equal_masked_mse(values, [torch.tensor([1.0, 100.0, 1.0])], masks)
    second = trajectory_equal_masked_mse(values, [torch.tensor([1.0, -100.0, 1.0])], masks)
    torch.testing.assert_close(first, second)


def test_value_metric_accumulator_reports_calibration_metrics():
    metrics = ValueMetricAccumulator()
    metrics.update(
        values=[torch.tensor([0.8, 0.8]), torch.tensor([0.2, 0.2])],
        targets=[torch.ones(2), torch.zeros(2)],
        masks=[torch.ones(2), torch.ones(2)],
    )
    result = metrics.compute()
    assert result["trajectory_equal_mse"] == pytest.approx(0.04)
    assert result["resolved_mean"] == pytest.approx(0.8)
    assert result["unresolved_mean"] == pytest.approx(0.2)
    assert result["auroc"] == pytest.approx(1.0)
    assert result["explained_variance"] == pytest.approx(0.84)


def test_value_metric_accumulator_returns_none_auc_for_one_class():
    metrics = ValueMetricAccumulator()
    metrics.update(
        values=[torch.tensor([0.4, 0.6])],
        targets=[torch.ones(2)],
        masks=[torch.ones(2)],
    )
    assert metrics.compute()["auroc"] is None
