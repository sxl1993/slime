from __future__ import annotations

import argparse
import json
import math
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .data import build_critic_data_refs, iter_critic_records
from .loss import ValueMetricAccumulator

logger = logging.getLogger(__name__)


def iter_record_batches(records: Iterable[Any], batch_size: int) -> Iterator[list[Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    batch = []
    for record in records:
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def total_optimizer_steps(record_count: int, global_batch_size: int) -> int:
    if record_count < 1:
        return 0
    return math.ceil(record_count / global_batch_size)


def should_save_checkpoint(value: float, best_value: float) -> bool:
    return value < best_value


def build_selection_payload(
    *, best_iteration: int, best_value: float, global_batch_size: int, train_limit: int | None
):
    return {
        "schema_version": 1,
        "selection_metric": "dev/trajectory_equal_mse",
        "best_iteration": best_iteration,
        "best_value": best_value,
        "global_batch_size": global_batch_size,
        "train_limit": train_limit,
    }


def write_selection_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def add_critic_pretrain_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--critic-pretrain-data", type=Path, required=True)
    parser.add_argument("--critic-pretrain-train-limit", type=int, default=None)
    parser.add_argument("--critic-pretrain-eval-interval", type=int, default=100)
    parser.add_argument("--critic-pretrain-eval-batch-size", type=int, default=128)
    parser.add_argument("--critic-pretrain-mode", choices=("train", "eval"), default="train")
    parser.add_argument("--critic-pretrain-eval-split", choices=("dev", "test"), default="dev")
    parser.add_argument("--critic-pretrain-selection-json", type=Path, required=True)
    return parser


def _manifest_count(manifest: dict[str, Any], split: str) -> int:
    split_manifest = manifest["splits"][split]
    return min(split_manifest["selected_resolved"], split_manifest["selected_unresolved"])


def evaluate_split(group, artifact_dir: Path, split: str, parallel_config, batch_size: int) -> dict[str, float | None]:
    """Evaluate a balanced artifact split without touching optimizer state."""
    import ray

    aggregate = ValueMetricAccumulator()
    for records in iter_record_batches(iter_critic_records(artifact_dir, split), batch_size=batch_size):
        refs = build_critic_data_refs(group.args, parallel_config, records)
        states = ray.get(group.async_evaluate(refs))
        for state in states:
            if state:
                aggregate.merge(ValueMetricAccumulator(**state))
    return aggregate.compute()


def _constant_prior_baseline(artifact_dir: Path, split: str) -> dict[str, float | None]:
    metrics = ValueMetricAccumulator()
    for record in iter_critic_records(artifact_dir, split):
        import torch

        metrics.update(
            [torch.full((record.response_length,), 0.5)],
            [torch.full((record.response_length,), record.reward)],
            [torch.tensor(record.loss_mask)],
        )
    return metrics.compute()


def train(args) -> None:
    import ray

    from slime.observability.logging_utils import configure_logger
    from slime.ray.placement_group import create_placement_groups

    from .actor import CriticPretrainGroup, CriticPretrainRayActor

    configure_logger()
    args.debug_train_only = True
    args.debug_rollout_only = False
    args.use_critic = True
    args.loss_type = "custom_loss"
    args.custom_loss_function_path = "examples.coding_agent_rl.critic_pretrain.loss.critic_pretrain_loss"
    args.calculate_per_token_loss = False
    args.n_samples_per_prompt = 1
    args.rollout_batch_size = args.global_batch_size

    artifact_dir = Path(args.critic_pretrain_data)
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    selected_train_count = _manifest_count(manifest, "train")
    train_limit = args.critic_pretrain_train_limit or selected_train_count
    train_limit = min(train_limit, selected_train_count)
    if train_limit < 1:
        raise ValueError("critic artifact has no selected training trajectories")

    pgs = create_placement_groups(args)
    group = CriticPretrainGroup(
        args=args,
        num_nodes=args.actor_num_nodes,
        num_gpus_per_node=args.actor_num_gpus_per_node,
        pg=pgs["actor"],
        role="critic",
        actor_cls=CriticPretrainRayActor,
    )
    group.create(rollout_manager=None)
    parallel_config = group.get_train_parallel_config()

    if args.critic_pretrain_mode == "eval":
        metrics = evaluate_split(
            group,
            artifact_dir,
            args.critic_pretrain_eval_split,
            parallel_config,
            args.critic_pretrain_eval_batch_size,
        )
        output = args.critic_pretrain_selection_json.parent / f"eval-{args.critic_pretrain_eval_split}.json"
        output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        logger.info("critic evaluation: %s", metrics)
        return

    baseline = _constant_prior_baseline(artifact_dir, "dev")
    logger.info("constant prior baseline: %s", baseline)
    best_value = float("inf")
    step = 0
    for records in iter_record_batches(
        iter_critic_records(artifact_dir, "train", limit=train_limit),
        batch_size=args.global_batch_size,
    ):
        step += 1
        refs = build_critic_data_refs(args, parallel_config, records)
        ray.get(group.async_train(step, refs))
        if step % args.critic_pretrain_eval_interval == 0 or step == total_optimizer_steps(
            train_limit, args.global_batch_size
        ):
            metrics = evaluate_split(
                group,
                artifact_dir,
                "dev",
                parallel_config,
                args.critic_pretrain_eval_batch_size,
            )
            current = metrics["trajectory_equal_mse"]
            if current is not None and should_save_checkpoint(current, best_value):
                best_value = current
                group.save_model(step, force_sync=True)
                write_selection_json(
                    args.critic_pretrain_selection_json,
                    build_selection_payload(
                        best_iteration=step,
                        best_value=current,
                        global_batch_size=args.global_batch_size,
                        train_limit=train_limit,
                    ),
                )
            logger.info("critic dev step=%d metrics=%s", step, metrics)


if __name__ == "__main__":
    from slime.utils.arguments import parse_args

    train(parse_args(add_custom_arguments=add_critic_pretrain_arguments))
