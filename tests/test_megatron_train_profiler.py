import subprocess
import sys
import textwrap


def test_critic_train_advances_profiler_once_per_rollout():
    code = textwrap.dedent("""
        from types import SimpleNamespace

        from slime.backends.megatron_utils.actor import MegatronTrainRayActor

        actor = object.__new__(MegatronTrainRayActor)
        actor.args = SimpleNamespace(debug_rollout_only=False, offload_train=False)
        actor.role = "critic"
        actor._get_rollout_data = lambda rollout_data_ref: rollout_data_ref
        actor.train_critic = lambda rollout_id, rollout_data: {"values": [rollout_id]}

        profiled_rollout_ids = []
        actor.prof = SimpleNamespace(step=lambda rollout_id: profiled_rollout_ids.append(rollout_id))

        result = actor.train(7, {"num_microbatches": [1]})

        assert result["values"] == [7]
        assert result["critic_total_time"] >= 0
        assert profiled_rollout_ids == [7]
        """)

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
