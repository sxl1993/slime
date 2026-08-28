from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch


def trajectory_equal_masked_mse(
    values: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    masks: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Average each trajectory's masked MSE before averaging trajectories."""
    if not values:
        return torch.tensor(0.0)
    per_trajectory = [
        ((value - target).square() * mask).sum() / mask.sum().clamp_min(1)
        for value, target, mask in zip(values, targets, masks, strict=True)
    ]
    return torch.stack(per_trajectory).mean()


def critic_pretrain_loss(
    args: Any,
    batch: dict[str, Any],
    logits: torch.Tensor,
    sum_of_sample_mean,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Regress response-aligned values to fixed terminal returns."""
    from slime.backends.megatron_utils.loss import get_values

    values = get_values(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
    )["values"]
    targets = batch["returns"]
    masks = batch["loss_masks"]
    if not (len(values) == len(targets) == len(masks)):
        raise ValueError("critic values, returns, and masks must have the same number of trajectories")

    errors = []
    for value, target, mask in zip(values, targets, masks, strict=True):
        if value.shape != target.shape or target.shape != mask.shape:
            raise ValueError(
                f"critic target shape mismatch: values={tuple(value.shape)}, "
                f"targets={tuple(target.shape)}, masks={tuple(mask.shape)}"
            )
        errors.append((value - target).square())
    flat_errors = torch.cat(errors, dim=0)
    loss = sum_of_sample_mean(flat_errors)
    value_mean = sum_of_sample_mean(torch.cat(values, dim=0)) / len(values)
    target_mean = sum_of_sample_mean(torch.cat(targets, dim=0)) / len(targets)
    return loss, {
        "value_loss": loss.detach(),
        "value_mean": value_mean.detach(),
        "target_mean": target_mean.detach(),
    }


@dataclass
class ValueMetricAccumulator:
    squared_error_sum: float = 0.0
    trajectory_count: int = 0
    scores: list[float] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    target_means: list[float] = field(default_factory=list)

    def update(self, values, targets, masks) -> None:
        for value, target, mask in zip(values, targets, masks, strict=True):
            value = torch.as_tensor(value).detach().float().cpu()
            target = torch.as_tensor(target).detach().float().cpu()
            mask = torch.as_tensor(mask).detach().float().cpu()
            denominator = float(mask.sum().clamp_min(1).item())
            prediction_mean = float(((value * mask).sum() / denominator).item())
            target_mean = float(((target * mask).sum() / denominator).item())
            squared_error = float((((value - target).square() * mask).sum() / denominator).item())
            self.squared_error_sum += squared_error
            self.trajectory_count += 1
            self.scores.append(prediction_mean)
            self.target_means.append(target_mean)
            self.labels.append(int(target_mean >= 0.5))

    def merge(self, other: ValueMetricAccumulator) -> None:
        self.squared_error_sum += other.squared_error_sum
        self.trajectory_count += other.trajectory_count
        self.scores.extend(other.scores)
        self.labels.extend(other.labels)
        self.target_means.extend(other.target_means)

    @staticmethod
    def _average_ranks(scores: Sequence[float]) -> list[float]:
        ordered = sorted(enumerate(scores), key=lambda item: item[1])
        ranks = [0.0] * len(scores)
        cursor = 0
        while cursor < len(ordered):
            end = cursor + 1
            while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
                end += 1
            average_rank = (cursor + 1 + end) / 2
            for index, _ in ordered[cursor:end]:
                ranks[index] = average_rank
            cursor = end
        return ranks

    def _auroc(self) -> float | None:
        positives = sum(self.labels)
        negatives = len(self.labels) - positives
        if not positives or not negatives:
            return None
        ranks = self._average_ranks(self.scores)
        positive_rank_sum = sum(rank for rank, label in zip(ranks, self.labels, strict=True) if label)
        return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)

    def compute(self) -> dict[str, float | None]:
        resolved = [score for score, label in zip(self.scores, self.labels, strict=True) if label]
        unresolved = [score for score, label in zip(self.scores, self.labels, strict=True) if not label]
        if self.trajectory_count == 0:
            return {
                "trajectory_equal_mse": float("nan"),
                "resolved_mean": None,
                "unresolved_mean": None,
                "explained_variance": None,
                "auroc": None,
            }
        target_mean = sum(self.target_means) / len(self.target_means)
        target_variance = sum((value - target_mean) ** 2 for value in self.target_means) / len(self.target_means)
        explained_variance = (
            1.0 - self.squared_error_sum / self.trajectory_count / target_variance if target_variance > 0 else None
        )
        return {
            "trajectory_equal_mse": self.squared_error_sum / self.trajectory_count,
            "resolved_mean": sum(resolved) / len(resolved) if resolved else None,
            "unresolved_mean": sum(unresolved) / len(unresolved) if unresolved else None,
            "explained_variance": explained_variance,
            "auroc": self._auroc(),
        }
