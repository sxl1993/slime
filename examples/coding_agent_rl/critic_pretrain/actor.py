from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def should_initialize_value_head(load_path: str | Path | None) -> bool:
    if load_path is None:
        return False
    path = Path(load_path)
    return (path / "config.json").is_file() and not (path / "latest_checkpointed_iteration.txt").is_file()


def _model_chunks(model):
    if isinstance(model, (list, tuple)):
        return list(model)
    return [model]


@torch.no_grad()
def initialize_prior_centered_value_head(model, optimizer) -> None:
    """Set every scalar output head to weight=0 and bias=0.5."""
    try:
        from megatron.core.utils import unwrap_model

        chunks = list(unwrap_model(_model_chunks(model)))
    except (ImportError, ModuleNotFoundError):
        chunks = _model_chunks(model)

    heads = []
    for chunk in chunks:
        output_layer = getattr(chunk, "output_layer", None)
        if output_layer is None and hasattr(chunk, "module"):
            output_layer = getattr(chunk.module, "output_layer", None)
        if output_layer is not None:
            heads.append(output_layer)
    if not heads:
        raise ValueError("critic model has no output_layer")
    for output_layer in heads:
        if tuple(output_layer.weight.shape[:1]) != (1,):
            raise ValueError(f"offline critic requires scalar output_layer, got {tuple(output_layer.weight.shape)}")
        output_layer.weight.zero_()
        if output_layer.bias is None:
            raise ValueError("offline critic requires a bias so the prior can be centered")
        output_layer.bias.fill_(0.5)
    if optimizer is not None and hasattr(optimizer, "reload_model_params"):
        optimizer.reload_model_params()
    logger.info(
        "Initialized offline critic value head: chunks=%d weight_checksum=%s bias=%s",
        len(heads),
        sum(float(head.weight.float().sum().item()) for head in heads),
        [float(head.bias.float().mean().item()) for head in heads],
    )


def _assert_full_parameter_training(model) -> tuple[int, int]:
    parameters = [parameter for chunk in _model_chunks(model) for parameter in chunk.parameters()]
    frozen = sum(parameter.numel() for parameter in parameters if not parameter.requires_grad)
    if frozen:
        raise ValueError(f"offline critic requires all parameters to be trainable; frozen parameters={frozen}")
    return sum(parameter.numel() for parameter in parameters), len(parameters)


def restore_critic_values_for_evaluation(values, rollout_data, *, allgather_cp: bool = False):
    """Restore CP-local values to full response sequences for metric reporting."""
    if allgather_cp or mpu.get_context_parallel_world_size() == 1:
        return values

    from slime.backends.megatron_utils.cp_utils import all_gather_with_cp

    return [
        all_gather_with_cp(value, int(total_length), int(response_length))
        for value, total_length, response_length in zip(
            values,
            rollout_data["total_lengths"],
            rollout_data["response_lengths"],
            strict=True,
        )
    ]


try:
    import ray
    from megatron.core import mpu

    from slime.backends.megatron_utils.actor import MegatronTrainRayActor
    from slime.backends.megatron_utils.data import get_data_iterator
    from slime.backends.megatron_utils.loss import get_values
    from slime.backends.megatron_utils.model import forward_only, train as megatron_train
    from slime.ray.actor_group import RayTrainGroup
except (ImportError, ModuleNotFoundError):  # pragma: no cover - CPU tests do not install the training stack
    ray = None
    mpu = None
    MegatronTrainRayActor = object
    RayTrainGroup = object


class CriticPretrainRayActor(MegatronTrainRayActor):
    """Megatron actor used by the offline critic-only trainer."""

    def init(self, args, role="critic", with_ref=False, with_opd_teacher=False):
        if role != "critic":
            raise ValueError(f"offline critic trainer requires role=critic, got {role}")
        initialize_head = should_initialize_value_head(args.load)
        start_id = super().init(args, role, with_ref, with_opd_teacher)
        self._critic_train_parallel_config = dict(self.train_parallel_config)
        del self.train_parallel_config
        total_parameters, trainable_tensors = _assert_full_parameter_training(self.model)
        logger.info(
            "Offline critic parameters: total=%d trainable_tensors=%d",
            total_parameters,
            trainable_tensors,
        )
        if initialize_head:
            initialize_prior_centered_value_head(self.model, self.optimizer)
        return start_id

    def get_train_parallel_config(self) -> dict[str, int]:
        return dict(self._critic_train_parallel_config)

    def _get_rollout_data(self, rollout_data_ref):
        rollout_data = super()._get_rollout_data(rollout_data_ref)
        device = self._device()
        rollout_data["returns"] = [
            tensor.to(device=device, dtype=torch.float32, non_blocking=True) for tensor in rollout_data["returns"]
        ]
        return rollout_data

    @staticmethod
    def _device():
        from slime.utils import accelerator

        return accelerator.current_device()

    def train_critic(self, rollout_id, rollout_data):
        data_iterator = get_data_iterator(rollout_data)
        self.args.loss_type = "custom_loss"
        self.args.custom_loss_function_path = "examples.coding_agent_rl.critic_pretrain.loss.critic_pretrain_loss"
        self.args.calculate_per_token_loss = False
        grad_norm = megatron_train(
            rollout_id,
            self.model,
            self.optimizer,
            self.opt_param_scheduler,
            data_iterator,
            rollout_data["num_microbatches"],
            rollout_data["global_batch_sizes"],
        )
        return {"grad_norm": float(grad_norm)}

    def reload_critic_checkpoint(self, path, step):
        from slime.backends.megatron_utils.checkpoint import load_checkpoint

        old_args = (
            self.args.load,
            self.args.ckpt_step,
            self.args.no_load_optim,
            self.args.no_load_rng,
            self.args.finetune,
        )
        self.args.load = str(path)
        self.args.ckpt_step = step
        self.args.no_load_optim = True
        self.args.no_load_rng = True
        self.args.finetune = True
        try:
            iteration, _ = load_checkpoint(self.model, None, None, checkpointing_context={})
            return iteration
        finally:
            (
                self.args.load,
                self.args.ckpt_step,
                self.args.no_load_optim,
                self.args.no_load_rng,
                self.args.finetune,
            ) = old_args

    def evaluate_critic(self, rollout_data_ref) -> dict[str, Any]:
        from .loss import ValueMetricAccumulator

        rollout_data = self._get_rollout_data(rollout_data_ref)
        if not mpu.is_pipeline_last_stage():
            return {}
        data_iterator = get_data_iterator(rollout_data)
        values = forward_only(
            get_values,
            self.args,
            self.model,
            data_iterator,
            rollout_data["num_microbatches"],
        )["values"]
        values = restore_critic_values_for_evaluation(
            values,
            rollout_data,
            allgather_cp=getattr(self.args, "allgather_cp", False),
        )
        if mpu.get_tensor_model_parallel_rank() != 0 or mpu.get_context_parallel_rank() != 0:
            return {}
        metrics = ValueMetricAccumulator()
        metrics.update(values, rollout_data["returns"], rollout_data["loss_masks"])
        return metrics.__dict__


if ray is not None:

    class CriticPretrainGroup(RayTrainGroup):
        def get_train_parallel_config(self) -> dict[str, int]:
            return ray.get(self._actor_handlers[0].get_train_parallel_config.remote())

        def async_evaluate(self, rollout_data_ref):
            return [actor.evaluate_critic.remote(rollout_data_ref) for actor in self._actor_handlers]

        def reload_checkpoint(self, path, step):
            return ray.get([actor.reload_critic_checkpoint.remote(path, step) for actor in self._actor_handlers])

else:

    class CriticPretrainGroup:  # pragma: no cover - import-only fallback for CPU test collection
        def __init__(self, *args, **kwargs):
            raise RuntimeError("critic-only Ray training requires ray and Megatron")
