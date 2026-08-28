from __future__ import annotations

import asyncio
import queue

import slime.rollout.fully_async_rollout as fully_async_rollout


def test_fully_async_uses_configured_sao_batch_size(monkeypatch):
    worker = fully_async_rollout.AsyncRolloutWorker.__new__(fully_async_rollout.AsyncRolloutWorker)
    worker.output_queue = queue.Queue()
    for gid in range(10):
        worker.output_queue.put((gid, []))

    monkeypatch.setattr(fully_async_rollout, "_get_global_worker", lambda args, data_buffer: worker)
    args = type(
        "Args",
        (),
        {"rollout_global_dataset": True, "rollout_batch_size": 8, "sao_batch_size": 2, "advantage_estimator": "sao"},
    )()

    result = asyncio.run(fully_async_rollout._generate_rollout_async(args, rollout_id=0, data_buffer=None))

    assert len(result) == 2
    assert worker.queue_size() == 8
