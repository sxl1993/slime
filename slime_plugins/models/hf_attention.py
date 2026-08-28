import json
import os
from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
from megatron.core import mpu, tensor_parallel
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule


def _load_hf_config(checkpoint_path):
    """Load HF config with fallback for unsupported model types."""
    try:
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
    except (ValueError, KeyError):
        config_path = os.path.join(checkpoint_path, "config.json")
        with open(config_path) as f:
            config_dict = json.load(f)

        _DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

        def _fix_dtype(d):
            if "torch_dtype" in d:
                d["torch_dtype"] = _DTYPE_MAP.get(d["torch_dtype"], d["torch_dtype"])
            if "dtype" in d:
                d["dtype"] = _DTYPE_MAP.get(d["dtype"], d["dtype"])

        _fix_dtype(config_dict)
        ns = type("HFConfig", (), config_dict)()
        if "text_config" in config_dict:
            _fix_dtype(config_dict["text_config"])
            ns.text_config = type("TextConfig", (), config_dict["text_config"])()
        return ns


class _AllGatherForDuplicatedComputation(torch.autograd.Function):
    """All-gather whose backward just returns the local gradient slice (no reduce).

    Use this instead of ``dist.nn.all_gather`` when the computation after the
    gather is *duplicated* across ranks (same weights, same full input →
    identical gradients).  The default ``all_gather`` backward performs a
    reduce-scatter, which incorrectly sums ``world_size`` identical copies of
    the gradient.
    """

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        ctx.rank = dist.get_rank(group=group)
        out = [torch.empty_like(x) for _ in range(dist.get_world_size(group=group))]
        dist.all_gather(out, x.contiguous(), group=group)
        return tuple(out)

    @staticmethod
    def backward(ctx, *grads):
        return grads[ctx.rank], None


def _get_cp_slice_bounds(packed_seq_params, cp_size):
    cached = getattr(packed_seq_params, "_slime_cp_slice_bounds", None)
    if cached is not None and cached[0] == cp_size:
        return cached[1], cached[2]

    cu_seqlens = tuple(packed_seq_params.cu_seqlens_q.detach().cpu().tolist())
    local_cu_seqlens = tuple(value // cp_size for value in cu_seqlens)
    gather_slices = []
    output_slices_by_rank = [[] for _ in range(cp_size)]

    for i in range(len(cu_seqlens) - 1):
        chunk_size = (cu_seqlens[i + 1] - cu_seqlens[i]) // (2 * cp_size)
        local_start = local_cu_seqlens[i]
        local_end = local_cu_seqlens[i + 1]
        gather_slices.extend((rank, local_start, local_start + chunk_size) for rank in range(cp_size))
        gather_slices.extend(reversed([(rank, local_start + chunk_size, local_end) for rank in range(cp_size)]))

        sequence_start = cu_seqlens[i]
        for rank in range(cp_size):
            output_slices_by_rank[rank].extend(
                [
                    (
                        sequence_start + rank * chunk_size,
                        sequence_start + (rank + 1) * chunk_size,
                    ),
                    (
                        sequence_start + (2 * cp_size - 1 - rank) * chunk_size,
                        sequence_start + (2 * cp_size - rank) * chunk_size,
                    ),
                ]
            )

    gather_slices = tuple(gather_slices)
    output_slices_by_rank = tuple(tuple(items) for items in output_slices_by_rank)
    packed_seq_params._slime_cp_slice_bounds = (
        cp_size,
        gather_slices,
        output_slices_by_rank,
    )
    return gather_slices, output_slices_by_rank


class HuggingfaceAttention(MegatronModule, ABC):
    """Attention layer abstract class.

    This layer only contains common modules required for the "self attn" and
    "cross attn" specializations.
    """

    def __init__(
        self,
        args,
        config,
        layer_number: int,
        cp_comm_type: str = "p2p",
        pg_collection=None,
    ):
        super().__init__(config=config)
        self.args = args
        self.config = config
        # Note that megatron layer_number starts at 1
        self.layer_number = layer_number
        self.hf_layer_idx = layer_number - 1
        self.hf_config = _load_hf_config(args.hf_checkpoint)
        # hardcode to fa2 at the moment.
        self.hf_config._attn_implementation = "flash_attention_2"

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        rotary_pos_emb: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        rotary_pos_cos: torch.Tensor | None = None,
        rotary_pos_sin: torch.Tensor | None = None,
        rotary_pos_cos_sin: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: int | None = None,
        *,
        inference_params: BaseInferenceContext | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert packed_seq_params is not None

        if self.args.sequence_parallel:
            # tensor_parallel_output_grad=False: the linear attention after this
            # gather is NOT TP-sharded (duplicated on all ranks), so the backward
            # should split (not reduce-scatter) to avoid inflating gradients by TP.
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states,
                tensor_parallel_output_grad=False,
                group=mpu.get_tensor_model_parallel_group(),
            )

        if mpu.get_context_parallel_world_size() > 1:
            cp_size = mpu.get_context_parallel_world_size()
            gather_slices, output_slices_by_rank = _get_cp_slice_bounds(
                packed_seq_params,
                cp_size,
            )
            # Use custom all-gather whose backward returns local gradient
            # instead of reduce-scatter, since the computation is duplicated.
            hidden_states_list = _AllGatherForDuplicatedComputation.apply(
                hidden_states,
                mpu.get_context_parallel_group(),
            )
            hidden_states = torch.cat(
                [hidden_states_list[source_rank][start:end] for source_rank, start, end in gather_slices],
                dim=0,
            )

        hidden_states = hidden_states.permute(1, 0, 2)  # [bsz, seq_len, hidden_dim]

        output = self.hf_forward(hidden_states, packed_seq_params)
        bias = None

        output = output.permute(1, 0, 2)  # [seq_len, bsz, hidden_dim]

        if mpu.get_context_parallel_world_size() > 1:
            cp_rank = mpu.get_context_parallel_rank()
            output = torch.cat(
                [output[start:end] for start, end in output_slices_by_rank[cp_rank]],
                dim=0,
            )

        if self.args.sequence_parallel:
            output = tensor_parallel.scatter_to_sequence_parallel_region(
                output, group=mpu.get_tensor_model_parallel_group()
            )

        return output, bias

    @abstractmethod
    def hf_forward(self, hidden_states, packed_seq_params):
        """Huggingface forward function"""
