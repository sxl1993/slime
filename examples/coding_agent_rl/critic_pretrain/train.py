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


def total_optimizer_steps(record_count: int, global_batch_size: int) -> int:
    if global_batch_size < 1:
        raise ValueError("global_batch_size must be positive")
    if record_count < 1:
        return 0
    return record_count // global_batch_size


def initial_training_step(start_rollout_ids: list[int]) -> int:
    if not start_rollout_ids:
        return 0
    if any(rollout_id != start_rollout_ids[0] for rollout_id in start_rollout_ids):
        raise ValueError(f"workers loaded different checkpoint iterations: {start_rollout_ids}")
    return start_rollout_ids[0] - 1


def selected_record_count(manifest: dict[str, Any], split: str) -> int:
    split_manifest = manifest["splits"][split]
    counts = (split_manifest["selected_resolved"], split_manifest["selected_unresolved"])
    if counts[0] != counts[1]:
        raise ValueError(f"critic artifact split={split} is not outcome-balanced: {counts}")
    return sum(counts)


def configure_critic_pretrain_schedule(args, train_limit: int) -> None:
    """Use the selected corpus size to size Megatron's iteration scheduler."""
    args.num_rollout = total_optimizer_steps(train_limit, args.global_batch_size)


def validate_artifact_context(manifest: dict[str, Any], sequence_length: int) -> None:
    artifact_length = manifest.get("max_seq_length")
    if artifact_length != sequence_length:
        raise ValueError(
            f"critic artifact max_seq_length={artifact_length} does not match model seq_length={sequence_length}"
        )


def validate_gradient_states(states: list[dict[str, Any]]) -> None:
    if not states or any("grad_norm" not in state for state in states):
        raise ValueError("critic pretraining did not report a gradient norm for every worker")
    if any(not math.isfinite(float(state["grad_norm"])) for state in states):
        raise ValueError("critic pretraining produced a non-finite gradient norm")


def validate_canary_metrics(metrics: dict[str, float | None]) -> None:
    loss = metrics.get("trajectory_equal_mse")
    resolved = metrics.get("resolved_mean")
    unresolved = metrics.get("unresolved_mean")
    if loss is None or not math.isfinite(float(loss)):
        raise ValueError("canary produced a non-finite value loss")
    if resolved is None or unresolved is None:
        raise ValueError("canary requires both resolved and unresolved value means")
    if not math.isfinite(float(resolved)) or not math.isfinite(float(unresolved)):
        raise ValueError("canary produced non-finite value means")
    if resolved <= unresolved:
        raise ValueError(f"canary failed value separation: resolved={resolved}, unresolved={unresolved}")


def should_save_checkpoint(value: float, best_value: float) -> bool:
    return value < best_value


def should_save_training_checkpoint(step: int, final_step: int, interval: int) -> bool:
    if interval < 1:
        raise ValueError("checkpoint save interval must be positive")
    return step % interval == 0 or step == final_step


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
    parser.add_argument("--critic-pretrain-eval-limit", type=int, default=512)
    parser.add_argument("--critic-pretrain-mode", choices=("train", "eval"), default="train")
    parser.add_argument("--critic-pretrain-canary", action="store_true")
    parser.add_argument("--critic-pretrain-eval-split", choices=("dev", "test"), default="dev")
    parser.add_argument("--critic-pretrain-selection-json", type=Path, required=True)
    return parser


def _manifest_count(manifest: dict[str, Any], split: str) -> int:
    return selected_record_count(manifest, split)


def evaluate_split(
    group,
    artifact_dir: Path,
    split: str,
    parallel_config,
    batch_size: int,
    limit: int | None,
) -> dict[str, float | None]:
    """Evaluate a balanced artifact split without touching optimizer state."""
    import ray

    aggregate = ValueMetricAccumulator()
    for records in iter_record_batches(iter_critic_records(artifact_dir, split, limit=limit), batch_size=batch_size):
        refs = build_critic_data_refs(group.args, parallel_config, records)
        states = ray.get(group.async_evaluate(refs))
        for state in states:
            if state:
                aggregate.merge(ValueMetricAccumulator(**state))
    return aggregate.compute()


def _constant_prior_baseline(artifact_dir: Path, split: str, *, limit: int | None) -> dict[str, float | None]:
    metrics = ValueMetricAccumulator()
    for record in iter_critic_records(artifact_dir, split, limit=limit):
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
    args.check_for_nan_in_loss_and_grad = False
    args.n_samples_per_prompt = 1
    args.rollout_batch_size = args.global_batch_size

    artifact_dir = Path(args.critic_pretrain_data)
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    validate_artifact_context(manifest, args.seq_length)
    selected_train_count = _manifest_count(manifest, "train")
    requested_limit = args.critic_pretrain_train_limit
    if getattr(args, "critic_pretrain_canary", False):
        canary_count = int(manifest["canary_count"])
        if selected_train_count < canary_count:
            raise ValueError(
                f"critic artifact has only {selected_train_count} selected training trajectories; "
                f"canary needs {canary_count}"
            )
        if requested_limit is not None and requested_limit != canary_count:
            raise ValueError(f"critic canary requires train limit {canary_count}, got {requested_limit}")
        train_limit = canary_count
    else:
        train_limit = selected_train_count if requested_limit is None else min(requested_limit, selected_train_count)
    if train_limit < 1:
        raise ValueError("critic artifact has no selected training trajectories")

    if args.critic_pretrain_mode == "train":
        configure_critic_pretrain_schedule(args, train_limit)

    pgs = create_placement_groups(args)
    group = CriticPretrainGroup(
        args=args,
        num_nodes=args.actor_num_nodes,
        num_gpus_per_node=args.actor_num_gpus_per_node,
        pg=pgs["actor"],
        role="critic",
        actor_cls=CriticPretrainRayActor,
    )
    start_rollout_ids = group.create(rollout_manager=None)
    parallel_config = group.get_train_parallel_config()

    if args.critic_pretrain_mode == "eval":
        metrics = evaluate_split(
            group,
            artifact_dir,
            args.critic_pretrain_eval_split,
            parallel_config,
            args.critic_pretrain_eval_batch_size,
            args.critic_pretrain_eval_limit,
        )
        output = args.critic_pretrain_selection_json.parent / f"eval-{args.critic_pretrain_eval_split}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        logger.info("critic evaluation: %s", metrics)
        return

    baseline = _constant_prior_baseline(artifact_dir, "dev", limit=args.critic_pretrain_eval_limit)
    logger.info("constant prior baseline: %s", baseline)
    best_value = float("inf")
    step = initial_training_step(start_rollout_ids)
    final_metrics = None
    final_step = step + total_optimizer_steps(train_limit, args.global_batch_size)
    for records in iter_record_batches(
        iter_critic_records(artifact_dir, "train", limit=train_limit),
        batch_size=args.global_batch_size,
    ):
        step += 1
        refs = build_critic_data_refs(args, parallel_config, records)
        validate_gradient_states(ray.get(group.async_train(step, refs)))
        if step % args.critic_pretrain_eval_interval == 0 or step == final_step:
            metrics = evaluate_split(
                group,
                artifact_dir,
                "dev",
                parallel_config,
                args.critic_pretrain_eval_batch_size,
                args.critic_pretrain_eval_limit,
            )
            current = metrics["trajectory_equal_mse"]
            if current is not None and should_save_checkpoint(current, best_value):
                best_value = current
                write_selection_json(
                    args.critic_pretrain_selection_json,
                    build_selection_payload(
                        best_iteration=step,
                        best_value=current,
                        global_batch_size=args.global_batch_size,
                        train_limit=train_limit,
                    ),
                )
            if should_save_training_checkpoint(step, final_step, args.save_interval):
                group.save_model(step, force_sync=True)
            logger.info("critic dev step=%d metrics=%s", step, metrics)
            final_metrics = metrics

    if getattr(args, "critic_pretrain_canary", False):
        if final_metrics is None:
            raise ValueError("critic canary did not produce final development metrics")
        validate_canary_metrics(final_metrics)
        selection_path = Path(args.critic_pretrain_selection_json)
        checkpoint_root = Path(args.save)
        if not (checkpoint_root / "latest_checkpointed_iteration.txt").is_file():
            raise ValueError(f"critic canary did not produce a native checkpoint under {checkpoint_root}")
        selection = json.loads(selection_path.read_text())
        best_iteration = int(selection["best_iteration"])
        loaded_iterations = group.reload_checkpoint(checkpoint_root, best_iteration)
        if any(int(iteration) != best_iteration for iteration in loaded_iterations):
            raise ValueError(
                f"critic canary checkpoint reload returned {loaded_iterations}, expected {best_iteration}"
            )
        reloaded_metrics = evaluate_split(
            group,
            artifact_dir,
            "dev",
            parallel_config,
            args.critic_pretrain_eval_batch_size,
            args.critic_pretrain_eval_limit,
        )
        if not math.isclose(
            float(reloaded_metrics["trajectory_equal_mse"]),
            float(selection["best_value"]),
            rel_tol=1e-4,
            abs_tol=1e-6,
        ):
            raise ValueError(f"critic canary checkpoint changed evaluation loss after reload: {reloaded_metrics}")
        validate_canary_metrics(reloaded_metrics)


if __name__ == "__main__":
    from slime.utils.arguments import parse_args

    train(parse_args(add_custom_arguments=add_critic_pretrain_arguments))
