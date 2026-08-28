from types import SimpleNamespace

import slime.observability.profile_utils as profile_utils


def _args(**overrides):
    values = {
        "use_pytorch_profiler": True,
        "profile_ranks": [],
        "record_memory_history": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_train_profiler_skips_unselected_rank(monkeypatch):
    monkeypatch.setattr(profile_utils.torch.distributed, "get_rank", lambda: 3)

    def fail_if_created(*args, **kwargs):
        raise AssertionError("profiler must not be created on an unselected rank")

    monkeypatch.setattr(profile_utils, "_create_torch_profiler", fail_if_created)

    profiler = profile_utils.TrainProfiler(_args(profile_ranks=[0, 8]), name="train_critic")

    assert profiler._torch_profiler_overall is None


def test_train_profiler_uses_role_in_trace_name(monkeypatch):
    sentinel = object()
    captured = {}
    monkeypatch.setattr(profile_utils.torch.distributed, "get_rank", lambda: 8)

    def create_profiler(args, name):
        captured["name"] = name
        return sentinel

    monkeypatch.setattr(profile_utils, "_create_torch_profiler", create_profiler)

    profiler = profile_utils.TrainProfiler(_args(profile_ranks=[0, 8]), name="train_critic")

    assert profiler._torch_profiler_overall is sentinel
    assert captured["name"] == "train_critic"
