import base64
from argparse import Namespace

import numpy as np
import pytest
import torch

from slime.observability.rollout_metrics import _compute_top_p_kept_vocab_metrics, compute_metrics_from_samples
from slime.utils.misc import decode_int32_meta_array
from slime.utils.types import Sample

NUM_GPUS = 0


def _make_args():
    return Namespace(sglang_speculative_algorithm=False, num_layers=2, moe_router_topk=2)


@pytest.mark.unit
def test_top_p_kept_vocab_metric_uses_loss_mask():
    samples = [
        Sample(
            response_length=4,
            loss_mask=torch.tensor([1, 0, 1, 0], dtype=torch.int32),
            rollout_top_p_token_offsets=torch.tensor([0, 3, 8, 10, 20], dtype=torch.int32),
        ),
        Sample(
            response_length=2,
            loss_mask=None,
            rollout_top_p_token_offsets=torch.tensor([0, 4, 9], dtype=torch.int32),
        ),
    ]

    metrics = _compute_top_p_kept_vocab_metrics(samples)

    assert metrics["top_p_kept_vocab_per_token"] == pytest.approx(3.5)


@pytest.mark.unit
def test_top_p_kept_vocab_metric_skips_removed_samples():
    samples = [
        Sample(
            response_length=3,
            loss_mask=[1, 1, 1],
            remove_sample=True,
            rollout_top_p_token_offsets=torch.tensor([0, 2, 4, 6], dtype=torch.int32),
        )
    ]

    assert _compute_top_p_kept_vocab_metrics(samples) == {}


@pytest.mark.unit
def test_trajectory_expansion_metrics_deduplicate_segments():
    args = Namespace(
        advantage_estimator="ppo",
        log_reward_category=None,
        sglang_speculative_algorithm=None,
    )
    shared_stats = {
        "trajectory_routing_leaf_count": 2,
        "trajectory_num_segments": 2,
        "trajectory_flat_tokens": 15,
        "trajectory_unique_tokens": 11,
        "trajectory_expansion_factor": 15 / 11,
    }
    samples = [
        Sample(rollout_id=7, tokens=[1, 2, 3], response_length=3, metadata=shared_stats),
        Sample(rollout_id=7, tokens=[1, 2, 4], response_length=3, metadata=shared_stats),
        Sample(
            rollout_id=8,
            tokens=[5, 6],
            response_length=2,
            metadata={
                "trajectory_routing_leaf_count": 1,
                "trajectory_num_segments": 1,
                "trajectory_flat_tokens": 4,
                "trajectory_unique_tokens": 4,
                "trajectory_expansion_factor": 1.0,
            },
        ),
    ]

    metrics = compute_metrics_from_samples(args, samples)

    assert metrics["trajectory/count"] == 2
    assert metrics["trajectory/routing_leaves_total"] == 3
    assert metrics["trajectory/num_segments_total"] == 3
    assert metrics["trajectory/num_segments_mean"] == pytest.approx(1.5)
    assert metrics["trajectory/num_segments_max"] == 2
    assert metrics["trajectory/flat_tokens_total"] == 19
    assert metrics["trajectory/flat_tokens_mean"] == pytest.approx(9.5)
    assert metrics["trajectory/flat_tokens_max"] == 15
    assert metrics["trajectory/unique_tokens_total"] == 15
    assert metrics["trajectory/unique_tokens_mean"] == pytest.approx(7.5)
    assert metrics["trajectory/unique_tokens_max"] == 11
    assert metrics["trajectory/expansion_factor_mean"] == pytest.approx((15 / 11 + 1) / 2)
    assert metrics["trajectory/expansion_factor_max"] == pytest.approx(15 / 11)
    assert metrics["trajectory/expansion_factor_weighted"] == pytest.approx(19 / 15)


@pytest.mark.unit
def test_trajectory_expansion_metrics_skip_samples_without_stats():
    args = Namespace(
        advantage_estimator="ppo",
        log_reward_category=None,
        sglang_speculative_algorithm=None,
    )

    metrics = compute_metrics_from_samples(args, [Sample(response_length=1, tokens=[1])])

    assert not any(key.startswith("trajectory/") for key in metrics)


def _b64_int32(values: list[int]) -> str:
    return base64.b64encode(np.array(values, dtype=np.int32).tobytes()).decode("ascii")


@pytest.mark.unit
def test_decode_int32_meta_array_decodes_base64_to_tensor():
    decoded = decode_int32_meta_array({"routed_experts": _b64_int32([1, 2, 3])}, "routed_experts")

    assert torch.is_tensor(decoded)
    assert decoded.dtype == torch.int32
    torch.testing.assert_close(decoded, torch.tensor([1, 2, 3], dtype=torch.int32))


@pytest.mark.unit
def test_append_response_tokens_merges_top_p_tensors():
    sample = Sample(
        tokens=[0, 1],
        response_length=1,
        loss_mask=[1],
        rollout_log_probs=[-0.3],
        rollout_top_p_token_ids=torch.tensor([1], dtype=torch.int32),
        rollout_top_p_token_offsets=torch.tensor([0, 1], dtype=torch.int32),
    )

    sample.append_response_tokens(
        _make_args(),
        tokens=[10, 20],
        log_probs=[-0.1, -0.2],
        trainable=True,
        meta_info={
            "top_p_token_ids": _b64_int32([10, 11, 20]),
            "top_p_token_offsets": _b64_int32([0, 2, 3]),
            "finish_reason": {"type": "stop"},
        },
    )

    assert sample.tokens == [0, 1, 10, 20]
    assert sample.response_length == 3
    assert sample.loss_mask == [1, 1, 1]
    assert sample.rollout_log_probs == [-0.3, -0.1, -0.2]
    torch.testing.assert_close(sample.rollout_top_p_token_ids, torch.tensor([1, 10, 11, 20], dtype=torch.int32))
    torch.testing.assert_close(sample.rollout_top_p_token_offsets, torch.tensor([0, 1, 3, 4], dtype=torch.int32))


@pytest.mark.unit
def test_append_response_tokens_can_skip_terminal_status_for_streaming_chunks():
    sample = Sample(
        tokens=[0, 1],
        response_length=1,
        loss_mask=[1],
        rollout_log_probs=[-0.3],
        rollout_top_p_token_ids=torch.tensor([1], dtype=torch.int32),
        rollout_top_p_token_offsets=torch.tensor([0, 1], dtype=torch.int32),
    )

    sample.append_response_tokens(
        _make_args(),
        tokens=[10, 20],
        log_probs=[-0.1, -0.2],
        trainable=True,
        meta_info={
            "top_p_token_ids": _b64_int32([10, 11, 20]),
            "top_p_token_offsets": _b64_int32([0, 2, 3]),
            "finish_reason": {"type": "stop"},
        },
        update_terminal_info=False,
    )

    assert sample.status is Sample.Status.PENDING
    assert sample.loss_mask == [1, 1, 1]
    assert sample.rollout_log_probs == [-0.3, -0.1, -0.2]
    torch.testing.assert_close(sample.rollout_top_p_token_ids, torch.tensor([1, 10, 11, 20], dtype=torch.int32))
    torch.testing.assert_close(sample.rollout_top_p_token_offsets, torch.tensor([0, 1, 3, 4], dtype=torch.int32))


@pytest.mark.unit
def test_append_response_tokens_decodes_routed_experts():
    sample = Sample(tokens=[101, 102, 103])

    sample.append_response_tokens(
        _make_args(),
        tokens=[],
        trainable=True,
        meta_info={
            "routed_experts": _b64_int32([0, 1, 2, 3, 4, 5, 6, 7]),
            "finish_reason": {"type": "stop"},
        },
    )

    assert sample.rollout_routed_experts.shape == (2, 2, 2)
    torch.testing.assert_close(
        sample.rollout_routed_experts,
        torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=torch.int32),
    )


@pytest.mark.unit
def test_append_response_tokens_ignores_split_pd_routed_experts():
    sample = Sample(tokens=[101, 102, 103, 104])

    sample.append_response_tokens(
        _make_args(),
        tokens=[],
        trainable=True,
        meta_info={
            "pd_prefill_routed_experts": _b64_int32([0, 1, 2, 3, 4, 5, 6, 7]),
            "pd_decode_routed_experts": _b64_int32([8, 9, 10, 11]),
            "finish_reason": {"type": "stop"},
        },
    )

    assert sample.rollout_routed_experts is None


@pytest.mark.unit
def test_append_response_tokens_rejects_mismatched_routed_experts_shape():
    sample = Sample(tokens=[101, 102, 103])

    with pytest.raises(ValueError, match="routed_experts element count"):
        sample.append_response_tokens(
            _make_args(),
            tokens=[],
            trainable=True,
            meta_info={
                "routed_experts": _b64_int32([0, 1, 2, 3]),
                "finish_reason": {"type": "stop"},
            },
        )


@pytest.mark.unit
def test_append_response_tokens_pads_top_p_for_non_trainable_tokens():
    sample = Sample(
        tokens=[0, 1],
        response_length=1,
        loss_mask=[1],
        rollout_log_probs=[-0.1],
        rollout_top_p_token_ids=torch.tensor([10, 11], dtype=torch.int32),
        rollout_top_p_token_offsets=torch.tensor([0, 2], dtype=torch.int32),
    )

    sample.append_response_tokens(tokens=[200, 201, 202], trainable=False)

    assert sample.tokens == [0, 1, 200, 201, 202]
    assert sample.response_length == 4
    assert sample.loss_mask == [1, 0, 0, 0]
    assert sample.rollout_log_probs == [-0.1, 0.0, 0.0, 0.0]
    torch.testing.assert_close(sample.rollout_top_p_token_ids, torch.tensor([10, 11], dtype=torch.int32))
    torch.testing.assert_close(sample.rollout_top_p_token_offsets, torch.tensor([0, 2, 2, 2, 2], dtype=torch.int32))


@pytest.mark.unit
def test_append_response_tokens_requires_trainable_log_probs():
    sample = Sample()

    with pytest.raises(ValueError, match="trainable response tokens require rollout log probabilities"):
        sample.append_response_tokens(tokens=[10], trainable=True)


@pytest.mark.unit
def test_append_response_tokens_rejects_non_trainable_log_probs():
    sample = Sample()

    with pytest.raises(ValueError, match="non-trainable response tokens should not pass rollout log probabilities"):
        sample.append_response_tokens(tokens=[10], log_probs=[-0.1], trainable=False)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
