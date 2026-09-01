from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace

# The fully-async seam under test does not load Ray actors. Stub the optional
# runtime dependency so this CPU-only test remains a narrow local test.
if "ray" not in sys.modules:
    ray = types.ModuleType("ray")
    ray.get = lambda value: value
    sys.modules["ray"] = ray
if "pylatexenc" not in sys.modules:
    pylatexenc = types.ModuleType("pylatexenc")
    latex2text = types.ModuleType("pylatexenc.latex2text")
    latex2text.LatexNodes2Text = type("LatexNodes2Text", (), {})
    pylatexenc.latex2text = latex2text
    sys.modules.update({"pylatexenc": pylatexenc, "pylatexenc.latex2text": latex2text})

from slime.rollout import fully_async_rollout as rollout
from slime.utils.types import Sample


def test_sao_rollout_calls_single_sample_entrypoint(monkeypatch):
    sample = Sample(index=3)
    calls: list[Sample] = []

    async def fake_generate_and_rm(args, received, sampling_params, evaluation=False):
        calls.append(received)
        assert evaluation is False
        return received

    monkeypatch.setattr(rollout, "generate_and_rm", fake_generate_and_rm)

    result = asyncio.run(rollout._generate_sao_rollout(SimpleNamespace(), sample, {"temperature": 1.0}))

    assert result is sample
    assert calls == [sample]


def test_sao_worker_drops_terminal_failed_rollout_without_group_retry(monkeypatch):
    monkeypatch.setattr(rollout, "GenerateState", lambda _args: SimpleNamespace(sampling_params={}))
    worker = rollout.AsyncRolloutWorker(SimpleNamespace(advantage_estimator="sao"), data_buffer=None)
    sample = Sample(index=3, status=Sample.Status.TERMINAL_FAILED, remove_sample=True)
    called = {"group": False}

    async def fake_group(*_args, **_kwargs):
        called["group"] = True
        raise AssertionError("SAO must not call the group entrypoint")

    monkeypatch.setattr(rollout, "generate_and_rm_group", fake_group)
    worker._make_done_cb(0)(SimpleNamespace(result=lambda: sample))

    assert called["group"] is False
    assert worker.get_completed_rollouts() == []
