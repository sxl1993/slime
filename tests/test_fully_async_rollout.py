"""CPU tests for shared group retry and fully-async queue contracts."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
import types
from collections import deque
from types import SimpleNamespace

if "sglang_router" not in sys.modules:
    _router_stub = types.ModuleType("sglang_router")
    _router_stub.__version__ = "0.2.3"
    sys.modules["sglang_router"] = _router_stub
if "transformers" not in sys.modules:
    _tf_stub = types.ModuleType("transformers")
    for _name in ("AutoProcessor", "AutoTokenizer", "PreTrainedTokenizerBase", "ProcessorMixin"):
        setattr(_tf_stub, _name, type(_name, (), {}))
    sys.modules["transformers"] = _tf_stub

import pytest

import slime.rollout.fully_async_rollout as fa
import slime.rollout.sglang_rollout as sglang_rollout
from slime.observability.rollout_metrics import compute_perf_metrics_from_samples
from slime.utils.types import Sample

NUM_GPUS = 0


class _FakeGenerateState:
    def __init__(self, args):
        self.sampling_params = {}


class _FakeDataBuffer:
    def __init__(self, groups):
        self._groups = deque(groups)
        self.requeued = []

    def get_samples(self, n):
        assert n == 1
        if not self._groups:
            return []
        return [self._groups.popleft()]

    def add_samples(self, groups):
        self.requeued.extend(groups)


def _make_group(index: int) -> list[Sample]:
    sample = Sample(index=index, prompt=f"p{index}")
    sample.status = Sample.Status.COMPLETED
    return [sample]


def _make_worker(monkeypatch, data_buffer=None, concurrency=4) -> fa.AsyncRolloutWorker:
    monkeypatch.setattr(fa, "GenerateState", _FakeGenerateState)
    args = SimpleNamespace(rollout_global_dataset=True, rollout_batch_size=4)
    return fa.AsyncRolloutWorker(args, data_buffer or _FakeDataBuffer([]), concurrency=concurrency)


@pytest.mark.unit
def test_rollout_takes_target_groups_and_leaves_surplus_queued(monkeypatch):
    worker = _make_worker(monkeypatch)
    for gid in range(10):
        worker.output_queue.put((gid, _make_group(gid)))
    monkeypatch.setattr(fa, "_get_global_worker", lambda args, data_buffer: worker)
    out = asyncio.run(
        fa._generate_rollout_async(SimpleNamespace(rollout_global_dataset=True, rollout_batch_size=4), 0, None)
    )
    assert [group[0].index for group in out] == [0, 1, 2, 3]
    assert all(group[0].metadata["rollout_batch_collect_time"] >= 0 for group in out)
    assert worker.queue_size() == 6
    assert [gid for gid, _ in worker.get_completed_groups()] == [4, 5, 6, 7, 8, 9]


@pytest.mark.unit
def test_existing_rollout_time_metric_keeps_collection_time_semantics():
    sample = Sample(response="done", response_length=4)
    args = SimpleNamespace(rollout_num_gpus=2)

    metrics = compute_perf_metrics_from_samples(args, [sample], rollout_time=12.5)

    assert metrics["rollout_time"] == 12.5


@pytest.mark.unit
def test_rollout_sorts_nested_fanout_groups_by_sample_index(monkeypatch):
    worker = _make_worker(monkeypatch)
    for gid, index in enumerate((30, 10, 20, 0)):
        sample = Sample(index=index, rollout_id=index, prompt=f"p{index}")
        sample.status = Sample.Status.COMPLETED
        worker.output_queue.put((gid, [[sample]]))
    monkeypatch.setattr(fa, "_get_global_worker", lambda args, data_buffer: worker)
    out = asyncio.run(
        fa._generate_rollout_async(SimpleNamespace(rollout_global_dataset=True, rollout_batch_size=4), 0, None)
    )
    assert [group[0][0].index for group in out] == [0, 10, 20, 30]


@pytest.mark.unit
def test_get_completed_groups_limit(monkeypatch):
    worker = _make_worker(monkeypatch)
    for gid in range(5):
        worker.output_queue.put((gid, _make_group(gid)))
    assert [gid for gid, _ in worker.get_completed_groups(limit=2)] == [0, 1]
    assert [gid for gid, _ in worker.get_completed_groups()] == [2, 3, 4]


@pytest.mark.unit
def test_shared_group_runner_retries_from_pristine_group_with_fresh_sessions(monkeypatch):
    attempts = []

    class _State:
        aborted = False

    async def _generate(_args, sample, _sampling_params, evaluation=False):
        attempts.append((sample.session_id, sample.status, sample.response, sample.reward, dict(sample.metadata)))
        if len(attempts) == 1:
            sample.status = Sample.Status.FAILED
            sample.response = "partial"
            sample.reward = 0.0
            sample.remove_sample = True
            sample.metadata["invalid_reason"] = "agent_exit:server_error"
        else:
            sample.status = Sample.Status.COMPLETED
            sample.response = "fixed"
            sample.reward = 1.0
        return sample

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: _State())
    monkeypatch.setattr(sglang_rollout, "generate_and_rm", _generate)
    monkeypatch.setattr(sglang_rollout, "_GROUP_RETRY_DELAY_SECONDS", 0.0)
    args = SimpleNamespace(sglang_enable_deterministic_inference=False, group_rm=False)
    result = asyncio.run(
        sglang_rollout.generate_and_rm_group(
            args, [Sample(index=7, prompt="solve", metadata={"instance_id": "demo"})], sampling_params={}
        )
    )

    assert fa.generate_and_rm_group is sglang_rollout.generate_and_rm_group
    assert len(attempts) == 2
    assert attempts[0][0] != attempts[1][0]
    assert [(status, response, reward, metadata) for _, status, response, reward, metadata in attempts] == [
        (Sample.Status.PENDING, "", None, {"instance_id": "demo"}),
        (Sample.Status.PENDING, "", None, {"instance_id": "demo"}),
    ]
    assert result[0].status == Sample.Status.COMPLETED


@pytest.mark.unit
def test_shared_group_runner_bounds_interruptions(monkeypatch):
    attempts = 0

    class _State:
        aborted = False

    async def _generate(_args, sample, _sampling_params, evaluation=False):
        nonlocal attempts
        attempts += 1
        sample.status = Sample.Status.ABORTED
        return [sample]

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: _State())
    monkeypatch.setattr(sglang_rollout, "generate_and_rm", _generate)
    args = SimpleNamespace(sglang_enable_deterministic_inference=False, group_rm=False)
    result = asyncio.run(
        sglang_rollout.generate_and_rm_group(args, [Sample(index=1, prompt="p1")], sampling_params={})
    )
    assert attempts == 3
    assert sglang_rollout.group_outcome(result) == "terminal"


@pytest.mark.unit
def test_shared_group_runner_allows_only_one_failed_retry(monkeypatch):
    attempts = 0

    class _State:
        aborted = False

    async def _generate(_args, sample, _sampling_params, evaluation=False):
        nonlocal attempts
        attempts += 1
        sample.status = Sample.Status.FAILED
        return sample

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: _State())
    monkeypatch.setattr(sglang_rollout, "generate_and_rm", _generate)
    monkeypatch.setattr(sglang_rollout, "_GROUP_RETRY_DELAY_SECONDS", 0.0)
    args = SimpleNamespace(sglang_enable_deterministic_inference=False, group_rm=False)
    result = asyncio.run(
        sglang_rollout.generate_and_rm_group(args, [Sample(index=1, prompt="p1")], sampling_params={})
    )
    assert attempts == 2
    assert sglang_rollout.group_outcome(result) == "terminal"


@pytest.mark.unit
def test_shared_group_outcome_uses_terminal_precedence_for_nested_fanout():
    interrupted = Sample()
    interrupted.status = Sample.Status.ABORTED
    terminal = Sample()
    terminal.status = Sample.Status.TERMINAL_FAILED

    assert sglang_rollout.group_outcome([[interrupted], [terminal]]) == "terminal"


@pytest.mark.unit
def test_shared_group_runner_marks_unknown_exception_terminal(monkeypatch):
    class _State:
        aborted = False

    async def _generate(_args, sample, _sampling_params, evaluation=False):
        raise ValueError("unexpected producer failure")

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: _State())
    monkeypatch.setattr(sglang_rollout, "generate_and_rm", _generate)
    args = SimpleNamespace(sglang_enable_deterministic_inference=False, group_rm=False)

    result = asyncio.run(
        sglang_rollout.generate_and_rm_group(args, [Sample(index=1, prompt="p1")], sampling_params={})
    )

    assert sglang_rollout.group_outcome(result) == "terminal"
    assert result[0].metadata["invalid_reason"] == "group_exception:ValueError"


@pytest.mark.unit
def test_shared_group_runner_honors_and_caps_retry_after(monkeypatch):
    attempts = 0
    sleeps = []

    class _State:
        aborted = False

    async def _generate(_args, sample, _sampling_params, evaluation=False):
        nonlocal attempts
        attempts += 1
        sample.status = Sample.Status.FAILED if attempts == 1 else Sample.Status.COMPLETED
        sample.retry_after_seconds = 120.0
        return sample

    async def _sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: _State())
    monkeypatch.setattr(sglang_rollout, "generate_and_rm", _generate)
    monkeypatch.setattr(sglang_rollout.asyncio, "sleep", _sleep)
    args = SimpleNamespace(sglang_enable_deterministic_inference=False, group_rm=False)
    result = asyncio.run(
        sglang_rollout.generate_and_rm_group(args, [Sample(index=1, prompt="p1")], sampling_params={})
    )
    assert result[0].status == Sample.Status.COMPLETED
    assert sleeps == [60.0]


@pytest.mark.unit
def test_terminal_group_is_replaced_by_fully_async_worker(monkeypatch):
    data_buffer = _FakeDataBuffer([[Sample(index=1, prompt="bad")], [Sample(index=2, prompt="good")]])

    async def _generate(_args, group, sampling_params, evaluation):
        group[0].status = Sample.Status.TERMINAL_FAILED if group[0].prompt == "bad" else Sample.Status.COMPLETED
        return group

    monkeypatch.setattr(fa, "generate_and_rm_group", _generate)
    worker = _make_worker(monkeypatch, data_buffer=data_buffer, concurrency=1)
    worker.poll_interval = 0.01
    worker.start()
    try:
        deadline = time.time() + 2
        completed = []
        while time.time() < deadline and not completed:
            completed = worker.get_completed_groups()
            time.sleep(0.01)
    finally:
        worker.stop()
    assert completed[0][1][0].prompt == "good"


@pytest.mark.unit
def test_terminal_group_is_replaced_by_synchronous_collector(monkeypatch):
    groups = deque([[Sample(index=1, prompt="bad")], [Sample(index=2, prompt="good")]])

    class _State:
        def __init__(self):
            self.remaining_batch_size = 0
            self.pendings = set()

        def submit_generate_tasks(self, samples):
            for group in samples:

                async def _result(group=group):
                    group[0].status = (
                        Sample.Status.TERMINAL_FAILED if group[0].prompt == "bad" else Sample.Status.COMPLETED
                    )
                    group[0].reward = 1.0
                    return group

                self.pendings.add(asyncio.create_task(_result()))
            self.remaining_batch_size += len(samples)

        def reset(self):
            self.remaining_batch_size = 0
            self.pendings = set()

    state = _State()
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)

    async def _abort(_args, _rollout_id):
        return []

    monkeypatch.setattr(sglang_rollout, "abort", _abort)
    args = SimpleNamespace(
        rollout_global_dataset=True,
        dynamic_sampling_filter_path=None,
        rollout_batch_size=1,
        n_samples_per_prompt=1,
        over_sampling_batch_size=1,
        rollout_sample_filter_path=None,
        rollout_all_samples_process_path=None,
    )

    def _data_source(_count):
        return [groups.popleft()]

    output, _ = asyncio.run(sglang_rollout.generate_rollout_async(args, 0, _data_source))
    assert output.samples[0][0].prompt == "good"


@pytest.mark.unit
def test_fully_async_collector_fails_after_shared_no_progress_window(monkeypatch):
    class _EmptyWorker:
        def get_completed_groups(self, limit=None):
            return []

        def queue_size(self):
            return 0

    monkeypatch.setattr(fa, "_get_global_worker", lambda args, data_buffer: _EmptyWorker())
    monkeypatch.setattr(sglang_rollout, "_GROUP_NO_PROGRESS_SECONDS", 0.02, raising=False)
    args = SimpleNamespace(rollout_global_dataset=True, rollout_batch_size=1)

    with pytest.raises(RuntimeError, match="no completed group"):
        asyncio.run(asyncio.wait_for(fa._generate_rollout_async(args, 9, None), timeout=0.5))


@pytest.mark.unit
def test_synchronous_collector_fails_after_shared_no_progress_window(monkeypatch):
    class _State:
        def __init__(self):
            self.remaining_batch_size = 0
            self.pendings = set()

        def submit_generate_tasks(self, samples):
            async def _terminal():
                sample = Sample(index=1, prompt="bad")
                sample.status = Sample.Status.TERMINAL_FAILED
                return [sample]

            self.pendings.add(asyncio.create_task(_terminal()))
            self.remaining_batch_size += 1

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: _State())
    monkeypatch.setattr(sglang_rollout, "_GROUP_NO_PROGRESS_SECONDS", 0.0, raising=False)
    args = SimpleNamespace(
        rollout_global_dataset=True,
        dynamic_sampling_filter_path=None,
        rollout_batch_size=1,
        n_samples_per_prompt=1,
        over_sampling_batch_size=1,
    )

    with pytest.raises(RuntimeError, match="no completed group"):
        asyncio.run(sglang_rollout.generate_rollout_async(args, 7, lambda _count: [[Sample(prompt="bad")]]))


@pytest.mark.unit
def test_done_callback_requeues_nested_aborted_group(monkeypatch):
    data_buffer = _FakeDataBuffer([])
    worker = _make_worker(monkeypatch, data_buffer=data_buffer)
    sample = Sample(index=7, prompt="p7")
    sample.status = Sample.Status.ABORTED

    class _DoneTask:
        def result(self):
            return [[sample]]

    worker._make_done_cb(0)(_DoneTask())
    assert data_buffer.requeued == [[sample]]
    assert worker.queue_size() == 0


@pytest.mark.unit
def test_done_callback_never_blocks_event_loop_thread(monkeypatch):
    worker = _make_worker(monkeypatch)

    class _DoneTask:
        def __init__(self, gid):
            self._result = _make_group(gid)

        def result(self):
            return self._result

    def _push_all():
        for gid in range(1001):
            worker._make_done_cb(gid)(_DoneTask(gid))

    pusher = threading.Thread(target=_push_all, daemon=True)
    pusher.start()
    pusher.join(timeout=30)
    assert not pusher.is_alive(), "done-callback blocked on a full output queue"
    assert worker.queue_size() == 1001


@pytest.mark.unit
def test_loop_backpressure_stops_topping_up_when_queue_is_full(monkeypatch):
    concurrency = 3
    data_buffer = _FakeDataBuffer([_make_group(i) for i in range(60)])

    async def _instant_generate(args, group, sampling_params, evaluation):
        return group

    monkeypatch.setattr(fa, "generate_and_rm_group", _instant_generate)
    worker = _make_worker(monkeypatch, data_buffer=data_buffer, concurrency=concurrency)
    worker.poll_interval = 0.01
    worker.start()
    try:
        deadline = time.time() + 3.0
        max_seen = 0
        while time.time() < deadline:
            max_seen = max(max_seen, worker.queue_size())
            if max_seen > 2 * concurrency:
                break
            time.sleep(0.02)
    finally:
        worker.stop()
    assert 0 < max_seen <= 2 * concurrency


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
