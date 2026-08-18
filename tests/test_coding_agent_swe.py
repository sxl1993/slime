import asyncio
from types import SimpleNamespace

from examples.coding_agent_rl import swe


def test_swebench_empty_hints_are_evaluable(monkeypatch):
    monkeypatch.setattr(swe, "_build_test_spec", lambda _inst: SimpleNamespace(eval_script="pytest"))
    instance = {
        "instance_id": "astropy__astropy-12907",
        "repo": "astropy/astropy",
        "version": "4.3",
        "base_commit": "abc",
        "problem_statement": "problem",
        "hints_text": "",
        "test_patch": "patch",
        "FAIL_TO_PASS": '["test_fail"]',
        "PASS_TO_PASS": "[]",
        "environment_setup_commit": "abc",
        "image": "image",
        "eval_script": "pytest",
        "log_parser": "pytest",
        "eval_type": "test_patch",
    }

    assert swe.evaluability_check({"protocol": swe.PROTOCOL_SWEBENCH, "grading": {"sweb_instance": instance}}) is None


def test_swebench_empty_diff_skips_empty_patch_write(monkeypatch):
    writes = []

    class FakeSandbox:
        privileged_user = "root"

        async def write_file(self, path, content, *, user):
            assert content
            writes.append((path, content, user))

        async def exec(self, _cmd, **_kwargs):
            return 0, "", ""

    class FakeSandboxContext:
        def __init__(self):
            self.sandbox = FakeSandbox()

        async def __aenter__(self):
            return self.sandbox

        async def __aexit__(self, _exc_type, _exc, _tb):
            return False

    async def fake_exec_and_wait(*_args, **_kwargs):
        return 0, "test log"

    instance_id = "astropy__astropy-12907"
    md = {
        "instance_id": instance_id,
        "image": "image",
        "workdir": "/testbed",
        "grading": {"sweb_instance": {"instance_id": instance_id}},
    }
    monkeypatch.setattr(swe, "_build_test_spec", lambda _inst: SimpleNamespace(eval_script="echo tests"))
    monkeypatch.setattr(swe, "create_sandbox", lambda *_args, **_kwargs: FakeSandboxContext())
    monkeypatch.setattr(swe, "exec_and_wait", fake_exec_and_wait)
    monkeypatch.setattr(
        swe,
        "_eval_report_from_log",
        lambda *_args: {instance_id: {"resolved": False, "patch_successfully_applied": True}},
    )

    result = asyncio.run(swe._grade_swebench(md, "", 1))

    assert result.reward == 0.0
    assert [path for path, _content, _user in writes] == ["/tmp/eval.sh"]
