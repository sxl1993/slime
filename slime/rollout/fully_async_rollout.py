"""Fully-async rollout for slime.

Decouples ``max_concurrent_tasks`` from ``rollout_batch_size``: a background
asyncio worker keeps a fixed pool of in-flight trajectories across rollout
boundaries, so the next training step doesn't have to wait for the slowest
in-flight sample to finish.

Use with ``--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async``.
Plug in per-sample logic via ``--custom-generate-function-path`` and
per-sample reward via ``--custom-rm-path`` — the worker calls the stock
per-sample or group entrypoint according to the selected estimator.

Concurrency is sourced from ``args.sglang_server_concurrency`` and scaled by
the number of sglang engines to match the per-sample semaphore cap in
:mod:`slime.rollout.sglang_rollout`.

GRPO uses the shared
:func:`slime.rollout.sglang_rollout.generate_and_rm_group` seam. SAO uses
:func:`slime.rollout.sglang_rollout.generate_and_rm` once per prompt and does
not apply group retry or group admission. This worker owns continuous
background concurrency and queue backpressure for both paths.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import queue
import threading
import time

from slime.rollout.sglang_rollout import (
    GenerateState,
    ensure_group_progress,
    generate_and_rm,
    generate_and_rm_group,
    group_outcome,
)
from slime.utils.async_utils import run
from slime.utils.http_utils import get_rollout_num_engines
from slime.utils.types import Sample

__all__ = [
    "AsyncRolloutWorker",
    "generate_rollout_fully_async",
]

logger = logging.getLogger("slime.rollout.fully_async")


# Global worker, shared across rollout calls so the queue stays warm.
_global_worker: AsyncRolloutWorker | None = None
_worker_lock = threading.Lock()


def _get_global_worker(args, data_buffer) -> AsyncRolloutWorker:
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            logger.info("starting fully-async rollout worker")
            _global_worker = AsyncRolloutWorker(
                args, data_buffer, concurrency=args.sglang_server_concurrency * get_rollout_num_engines(args)
            )
            _global_worker.start()
        return _global_worker


def _stop_global_worker() -> None:
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


atexit.register(_stop_global_worker)


class AsyncRolloutWorker:
    """Background worker for GRPO groups or individual SAO rollout units."""

    def __init__(self, args, data_buffer, concurrency: int = 10):
        self.args = args
        self.data_buffer = data_buffer
        self.concurrency = concurrency
        self.running = True
        # Unbounded on purpose: put() runs inside the event-loop thread (task
        # done-callback), so a bounded queue that fills up would block the loop
        # and freeze every in-flight generation. Backpressure lives in _loop()
        # instead, which stops topping up while a full pool of completed groups
        # is already waiting to be consumed.
        self.output_queue: queue.Queue[tuple[int, list[Sample] | list[list[Sample]]]] = queue.Queue()
        self.poll_interval = 1.0
        self.worker_thread: threading.Thread | None = None
        self.state = GenerateState(args)

    # -- public --------------------------------------------------------------

    def start(self) -> None:
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._thread_main, name="fully-async-rollout", daemon=True)
            self.worker_thread.start()

    def stop(self) -> None:
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

    def get_completed_groups(self, limit: int | None = None) -> list[tuple[int, list[Sample] | list[list[Sample]]]]:
        """Pop up to ``limit`` completed groups (all of them when ``None``).

        Callers that only need a fixed number of groups must pass ``limit`` —
        anything popped beyond it would otherwise have to be thrown away, and
        these groups are fully generated and reward-scored, with their prompts
        already consumed from ``data_buffer``.
        """
        completed: list[tuple[int, list[Sample] | list[list[Sample]]]] = []
        while limit is None or len(completed) < limit:
            try:
                completed.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return completed

    def get_completed_rollouts(self, limit: int | None = None) -> list[tuple[int, list[Sample]]]:
        """Pop completed SAO rollout units from the persistent output queue."""
        completed: list[tuple[int, list[Sample]]] = []
        while limit is None or len(completed) < limit:
            try:
                rollout_id, samples = self.output_queue.get_nowait()
            except queue.Empty:
                break
            completed.append((rollout_id, samples if isinstance(samples, list) else [samples]))
        return completed

    def queue_size(self) -> int:
        return self.output_queue.qsize()

    # -- internals -----------------------------------------------------------

    def _thread_main(self) -> None:
        asyncio.run(self._loop())

    async def _loop(self) -> None:
        active_tasks: set[asyncio.Task] = set()
        max_concurrent = self.concurrency
        gid_counter = 0

        while self.running:
            try:
                # Reap done tasks
                if active_tasks:
                    done = {t for t in active_tasks if t.done()}
                    for t in done:
                        try:
                            t.result()  # results already handled in callback
                        except Exception as e:  # noqa: BLE001
                            logger.warning("fully-async task crashed: %r", e)
                    active_tasks -= done

                # Top up. The qsize gate is the queue's backpressure: once a
                # full pool of completed results is waiting, stop pulling new
                # prompts until the training side drains some.
                while (
                    len(active_tasks) < max_concurrent and self.output_queue.qsize() < max_concurrent and self.running
                ):
                    prompts = self.data_buffer.get_samples(1)
                    if not prompts:
                        break
                    for prompt_samples in prompts:
                        gid = gid_counter
                        gid_counter += 1
                        if getattr(self.args, "advantage_estimator", None) == "sao":
                            if len(prompt_samples) != 1:
                                logger.warning(
                                    "fully-async SAO expected one sample per prompt, got %d", len(prompt_samples)
                                )
                                continue
                            task = asyncio.create_task(
                                _generate_sao_rollout(
                                    self.args,
                                    prompt_samples[0],
                                    sampling_params=self.state.sampling_params.copy(),
                                )
                            )
                        else:
                            task = asyncio.create_task(
                                generate_and_rm_group(
                                    self.args,
                                    prompt_samples,
                                    sampling_params=self.state.sampling_params.copy(),
                                    evaluation=False,
                                )
                            )
                        task.add_done_callback(self._make_done_cb(gid))
                        active_tasks.add(task)

                await asyncio.sleep(self.poll_interval)
            except Exception as e:  # noqa: BLE001
                logger.exception("fully-async loop iteration error: %s", e)
                await asyncio.sleep(self.poll_interval)

        if active_tasks:
            logger.info(
                "fully-async: waiting for %d in-flight tasks to drain",
                len(active_tasks),
            )
            try:
                await asyncio.wait(active_tasks, timeout=30)
            except Exception:  # noqa: BLE001
                pass

    def _make_done_cb(self, gid: int):
        def _cb(done_task: asyncio.Task) -> None:
            try:
                result = done_task.result()
            except Exception:  # noqa: BLE001
                logger.exception("fully-async: process task raised")
                return
            if getattr(self.args, "advantage_estimator", None) == "sao":
                if isinstance(result, Sample):
                    samples = [result]
                elif isinstance(result, list) and all(isinstance(sample, Sample) for sample in result):
                    samples = result
                else:
                    logger.warning("fully-async SAO: rollout returned %r; dropping", type(result).__name__)
                    return
                if any(
                    sample.remove_sample or sample.status not in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED)
                    for sample in samples
                ):
                    logger.warning("fully-async SAO: non-trainable rollout dropped")
                    return
                self.output_queue.put((gid, samples))
                return
            if not isinstance(result, list):
                logger.warning(
                    "fully-async: generate_and_rm_group returned %r, expected list[Sample] or list[list[Sample]]; dropping",
                    type(result).__name__,
                )
                return
            groups = result if result and isinstance(result[0], list) else [result]
            outcome = group_outcome(result)
            if outcome == "terminal" or outcome == "retryable":
                return
            if outcome == "interrupted":
                try:
                    self.data_buffer.add_samples(groups)
                except Exception:  # noqa: BLE001
                    logger.exception("fully-async: failed to requeue interrupted group")
                return
            self.output_queue.put((gid, result))

        return _cb


async def _generate_sao_rollout(args, sample: Sample, sampling_params: dict) -> Sample | list[Sample]:
    """Generate exactly one SAO rollout unit without group scheduling."""
    return await generate_and_rm(args, sample, sampling_params=sampling_params, evaluation=False)


async def _generate_rollout_async(args, rollout_id: int, data_buffer) -> list[list[Sample]]:
    assert args.rollout_global_dataset
    worker = _get_global_worker(args, data_buffer)

    target = args.sao_batch_size if getattr(args, "advantage_estimator", None) == "sao" else args.rollout_batch_size
    logger.info(
        "fully-async rollout %d: target=%d queue_warm=%d",
        rollout_id,
        target,
        worker.queue_size(),
    )

    is_sao = getattr(args, "advantage_estimator", None) == "sao"
    collected: dict[int, list[Sample] | list[list[Sample]]] = {}
    started = time.time()
    last_progress = time.monotonic()
    last_log = started
    LOG_EVERY = 30.0

    while len(collected) < target:
        if is_sao:
            if time.monotonic() - last_progress >= 5400.0:
                raise RuntimeError(f"rollout {rollout_id} produced no completed rollout")
        else:
            ensure_group_progress(last_progress, rollout_id)
        # Pull only what this rollout still needs; the surplus stays queued for
        # the next rollout (that is the "queue stays warm" contract).
        drained = 0
        completed = (
            worker.get_completed_rollouts(limit=target - len(collected))
            if is_sao
            else worker.get_completed_groups(limit=target - len(collected))
        )
        for gid, result in completed:
            collected[gid] = result
            drained += 1

        if drained:
            last_progress = time.monotonic()

        if not drained:
            await asyncio.sleep(0.05)

        now = time.time()
        if now - last_log > LOG_EVERY:
            logger.info(
                "fully-async rollout %d: collected %d/%d, queue=%d, elapsed=%.1fs",
                rollout_id,
                len(collected),
                target,
                worker.queue_size(),
                now - started,
            )
            last_log = now

    # Order by sample.index for determinism (slime convention).
    def _key(result: list[Sample] | list[list[Sample]]) -> int:
        if result and isinstance(result[0], list):
            result = result[0]
        for s in result:
            idx = getattr(s, "index", None)
            if idx is not None:
                return int(idx)
        return 0

    out = sorted(collected.values(), key=_key)
    batch_collect_time = time.time() - started
    for result in out:
        groups = result if result and isinstance(result[0], list) else [result]
        for group in groups:
            for sample in group:
                sample.metadata["rollout_batch_collect_time"] = batch_collect_time
    logger.info(
        "fully-async rollout %d: done in %.1fs, queue_left=%d",
        rollout_id,
        batch_collect_time,
        worker.queue_size(),
    )
    return out


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation: bool = False):
    """Slime ``--rollout-function-path`` entrypoint."""

    if evaluation:
        raise ValueError("fully-async rollout doesn't support evaluation mode")
    return run(_generate_rollout_async(args, rollout_id, data_buffer))
