from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def test_train_skips_weight_updates_during_critic_warmup(monkeypatch):
    events: list[str] = []

    class RemoteMethod:
        def __init__(self, name):
            self.name = name

        def remote(self, *args):
            events.append(f"{self.name}:{args[0] if args else ''}")
            return args[0] if args else None

    class RolloutManager:
        generate = RemoteMethod("generate")

        dispose = RemoteMethod("dispose")

    class Model:
        def __init__(self, role):
            self.role = role
            self.update_weight_calls = 0

        def update_weights(self):
            self.update_weight_calls += 1
            events.append(f"update:{self.role}")

        def async_train(self, rollout_id, rollout_data_ref, external_data=None):
            events.append(f"train:{self.role}:{rollout_id}")
            return [rollout_id]

        def save_model(self, rollout_id, force_sync=False):
            return None

    actor_model = Model("actor")
    critic_model = Model("critic")
    rollout_manager = RolloutManager()

    fake_ray = types.ModuleType("ray")
    fake_ray.get = lambda value: value
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    logging_utils = types.ModuleType("slime.observability.logging_utils")
    logging_utils.configure_logger = lambda: None
    logging_utils.finish_tracking = lambda args: None
    logging_utils.init_tracking = lambda args: None
    monkeypatch.setitem(sys.modules, "slime.observability.logging_utils", logging_utils)

    placement_group = types.ModuleType("slime.ray.placement_group")
    placement_group.create_placement_groups = lambda args: {"rollout": object()}
    placement_group.create_rollout_manager = lambda args, pg: (rollout_manager, args.num_rollout)
    placement_group.create_training_models = lambda args, pgs, manager: (actor_model, critic_model)
    monkeypatch.setitem(sys.modules, "slime.ray.placement_group", placement_group)

    arguments = types.ModuleType("slime.utils.arguments")
    arguments.parse_args = lambda: None
    monkeypatch.setitem(sys.modules, "slime.utils.arguments", arguments)

    misc = types.ModuleType("slime.utils.misc")
    misc.should_run_periodic_action = lambda *args: False
    monkeypatch.setitem(sys.modules, "slime.utils.misc", misc)

    module_path = Path(__file__).resolve().parents[1] / "train_async.py"
    spec = importlib.util.spec_from_file_location("train_async_under_test", module_path)
    assert spec is not None and spec.loader is not None
    train_async = importlib.util.module_from_spec(spec)
    sys.modules["train_async_under_test"] = train_async
    spec.loader.exec_module(train_async)

    args = SimpleNamespace(
        colocate=False,
        check_weight_update_equal=False,
        release_train=False,
        use_critic=True,
        start_rollout_id=0,
        num_rollout=3,
        num_critic_only_steps=2,
        save_interval=100,
        update_weights_interval=1,
        eval_interval=0,
        rollout_global_dataset=False,
    )
    train_async.train(args)

    assert actor_model.update_weight_calls == 2
    assert events.count("train:actor:0") == 0
    assert events.count("train:actor:1") == 0
    assert events.count("train:actor:2") == 1
