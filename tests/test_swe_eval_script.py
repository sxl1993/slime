"""Regression tests for the LocalSandbox swebench eval-script patcher.

Guards against a silent reward-collapse regression: ``_patch_eval_sh_for_local``
must NOT add ``set -e`` to swebench's eval script.

swebench writes ``: '>>>>> End Test Output'`` *after* the pytest invocation so
its ``get_logs_eval`` parser can delimit the test output.  FAIL_TO_PASS tests
fail on any patch that doesn't fully fix the bug, so pytest commonly exits
non-zero.  With ``set -e`` the script aborts the instant pytest returns —
*before* the End marker runs — so ``get_logs_eval`` returns ``found=False``,
``patch_successfully_applied`` stays False, ``tests_status`` is never populated,
and the reward gating collapses every sample (even a correct patch) to
reward=0, zeroing all gradients.  swebench deliberately omits ``-e`` for this
reason; we must respect it.

These tests run the *real* patched script under ``bash`` with a failing
"pytest" stand-in (``false``) and assert the End marker is still emitted —
i.e. they exercise the actual failure mechanism on the actual transformed
output.  No swebench / sandbox / GPU required; deterministic and fast.
"""

import importlib.util
import subprocess
import sys
from unittest.mock import MagicMock

import pytest


def _import_swe():
    """Import swe.py as a module (it lives under examples/)."""
    spec = importlib.util.spec_from_file_location(
        "swe",
        "examples/coding_agent_rl/swe.py",
    )
    mod = importlib.util.module_from_spec(spec)
    # Stub heavy imports that swe.py does at module level.
    sys.modules.setdefault("swebench", MagicMock())
    sys.modules.setdefault("swebench.harness.constants", MagicMock())
    sys.modules.setdefault("swebench.harness.grading", MagicMock())
    sys.modules.setdefault("swebench.harness.test_spec.test_spec", MagicMock())
    spec.loader.exec_module(mod)
    return mod


# The two swebench log delimiters that get_logs_eval splits on.  Keeping them
# in sync with swebench.harness.constants (START_TEST_OUTPUT / END_TEST_OUTPUT)
# lets a drift there break this test loudly rather than silently.
START_MARKER = ">>>>> Start Test Output"
END_MARKER = ">>>>> End Test Output"


def _synthetic_eval_script() -> str:
    """A minimal swebench-shaped eval script.

    Mirrors the structure swebench's ``make_test_spec`` emits: ``set -uxo
    pipefail``, a Start marker, the pytest invocation, an End marker, and a
    trailing cleanup line.  The pytest stand-in is ``false`` (exit 1) to model
    a FAIL_TO_PASS test failing on an incomplete patch — swebench's normal
    steady state.
    """
    return (
        "#!/bin/bash\n"
        "set -uxo pipefail\n"
        f": '{START_MARKER}'\n"
        "false\n"  # pytest exits 1 because a FAIL_TO_PASS test failed
        f": '{END_MARKER}'\n"
        "echo cleanup-ran\n"
    )


class TestNoSetEAdded:
    """The patcher must not introduce ``set -e``."""

    def test_preserves_swebench_set_flags_without_e(self):
        swe = _import_swe()
        out = swe._patch_eval_sh_for_local(_synthetic_eval_script(), env_dir=None)
        # The first ``set`` line must still be swebench's ``set -uxo pipefail``
        # (no injected ``-e``).
        first_set = next(line for line in out.splitlines() if line.startswith("set "))
        assert first_set == "set -uxo pipefail"
        # And nowhere in the script must `-e` have been bolted on via the set line.
        assert "set -euxo pipefail" not in out

    def test_failing_pytest_still_emits_end_marker(self):
        """The actual regression: under a failing (exit-1) pytest, the patched
        script must still reach the End marker so swebench can parse the result."""
        swe = _import_swe()
        patched = swe._patch_eval_sh_for_local(_synthetic_eval_script(), env_dir=None)
        proc = subprocess.run(
            ["bash", "-c", patched],
            capture_output=True,
            text=True,
        )
        combined = proc.stdout + proc.stderr
        # pytest failed, so the script's overall exit status is non-zero — that
        # is expected and fine.  What matters is that the End marker was emitted
        # before the abort.
        assert END_MARKER in combined, (
            f"End marker missing after a failing pytest — `set -e` likely "
            f"re-introduced, aborting before swebench can delimit test output.\n"
            f"--- patched script ---\n{patched}\n--- output ---\n{combined}"
        )
        assert "cleanup-ran" in combined  # script ran past the End marker

    def test_end_marker_text_preserved(self):
        """The End marker line text must survive patching verbatim."""
        swe = _import_swe()
        out = swe._patch_eval_sh_for_local(_synthetic_eval_script(), env_dir=None)
        assert f": '{END_MARKER}'" in out
        assert f": '{START_MARKER}'" in out


def _info(*, resolved=False, applied=True, f2p=(0, 0), p2p=(0, 0)):
    """Build a swebench eval-report ``info`` dict with the given test buckets."""
    f2p_pass, f2p_total = f2p
    p2p_pass, p2p_total = p2p
    f2p_fail = f2p_total - f2p_pass
    p2p_fail = p2p_total - p2p_pass
    return {
        "resolved": resolved,
        "patch_successfully_applied": applied,
        "tests_status": {
            "FAIL_TO_PASS": {"success": ["t"] * f2p_pass, "failure": ["t"] * f2p_fail},
            "PASS_TO_PASS": {"success": ["t"] * p2p_pass, "failure": ["t"] * p2p_fail},
        },
    }


class TestShapedReward:
    """``_swebench_shaped_reward`` is the single source of truth shared by the
    grader (dispatched reward) and the log line (greped value), so they never
    silently diverge — the bug this guards against was a hardcoded
    ``reward=0`` log that hid actual 0.30 rewards."""

    def test_resolved_is_one(self):
        swe = _import_swe()
        assert swe._swebench_shaped_reward(_info(resolved=True)) == 1.0

    def test_resolved_log_still_emits_f2p_p2p(self, caplog):
        # Guard against re-introducing an early ``resolved`` return in
        # _log_swebench_result that would drop the F2P/P2P buckets from the
        # log line — resolved samples should still report them (F2P/P2P all
        # green) so the grading log is uniformly parseable.
        swe = _import_swe()
        import logging as _logging

        with caplog.at_level(_logging.INFO, logger=swe.logger.name):
            swe._log_swebench_result(
                "astropy__x-1",
                0,
                _info(resolved=True, f2p=(2, 2), p2p=(141, 141)),
                "no markers here",
            )
        line = caplog.records[-1].getMessage()
        assert "reward=1.000" in line
        assert "F2P=(2/2)" in line
        assert "P2P=(141/141)" in line

    def test_preserve_p2p_noop_patch_earns_beta_only(self):
        # F2P=(0/2) P2P=(68/68): patch didn't fix the bug but broke nothing —
        # the exact astropy-13398 case that printed reward=0 but was really 0.30.
        swe = _import_swe()
        r = swe._swebench_shaped_reward(_info(f2p=(0, 2), p2p=(68, 68)))
        assert r == pytest.approx(0.3)  # 0.7*0 + 0.3*1.0

    def test_broken_patch_collects_no_tests_is_zero(self):
        # F2P=(0/4) P2P=(0/68): patch had a syntax error, collection failed, so
        # every test failed — genuinely zero reward (not a logging artifact).
        swe = _import_swe()
        assert swe._swebench_shaped_reward(_info(f2p=(0, 4), p2p=(0, 68))) == 0.0

    def test_no_tests_parsed_is_zero(self):
        # patch_applied=True but F2P=(0/0) P2P=(0/0): the pre-fix symptom (End
        # marker missing) — no signal, reward 0.
        swe = _import_swe()
        assert swe._swebench_shaped_reward(_info(applied=True, f2p=(0, 0), p2p=(0, 0))) == 0.0

    def test_patch_not_applied_is_zero(self):
        swe = _import_swe()
        assert swe._swebench_shaped_reward(_info(applied=False, f2p=(0, 2), p2p=(68, 68))) == 0.0

    def test_partial_f2p_and_full_p2p(self):
        # F2P=(1/2) P2P=(68/68): half the bug fixed, nothing broken.
        swe = _import_swe()
        r = swe._swebench_shaped_reward(_info(f2p=(1, 2), p2p=(68, 68)))
        assert r == pytest.approx(0.7 * 0.5 + 0.3 * 1.0)  # 0.65
