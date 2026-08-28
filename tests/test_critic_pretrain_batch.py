from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.coding_agent_rl.critic_pretrain.data import CriticRecord  # noqa: E402
from examples.coding_agent_rl.critic_pretrain.data import (  # noqa: E402
    build_critic_data_refs,
    build_critic_train_data,
)


def make_record(index: int) -> CriticRecord:
    return CriticRecord(
        record_id=f"record-{index}",
        instance_id=f"instance-{index}",
        repo="repo",
        source="test",
        outcome="resolved" if index % 2 == 0 else "unresolved",
        reward=float(index % 2 == 0),
        tokens=[1, 2, 3, 4],
        response_length=2,
        loss_mask=[1, 1],
        returns=[float(index % 2 == 0)] * 2,
    )


def test_build_critic_train_data_expands_fixed_target():
    records = [make_record(index) for index in range(4)]
    train_data = build_critic_train_data(records)
    assert train_data["response_lengths"] == [record.response_length for record in records]
    assert train_data["rollout_mask_sums"].tolist() == [sum(record.loss_mask) for record in records]
    assert torch.equal(train_data["returns"][0], torch.tensor(records[0].returns))


def test_build_critic_data_refs_partitions_each_sample_once():
    records = [make_record(index) for index in range(4)]

    class Args:
        use_dynamic_batch_size = False
        micro_batch_size = 1
        max_tokens_per_gpu = None
        balance_data = False
        balance_by_flops = False

    refs = build_critic_data_refs(
        Args(),
        {"dp_size": 2, "cp_size": 1, "vpp_size": 1, "microbatch_group_size_per_vp_stage": 1},
        records,
        ray_put=lambda value: value,
    )
    assert len(refs) == 2
    partitions = [ref.inner["partition"] for ref in refs]
    assert sorted(index for partition in partitions for index in partition) == list(range(4))
    assert all(len(ref.inner["micro_batch_indices"]) == 2 for ref in refs)
