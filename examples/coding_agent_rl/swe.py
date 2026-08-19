"""SWE task layer: dataset parsing, workspace prep, diff capture, fresh-sandbox eval.

One module, two grading protocols selected per-call (never an import-time side
effect):

  - "scaleswe" (default): scaleswe data shape (image_url + pre_commands +
    swepro/eval_cmd/f2p_script); custom "exit 0 == solved" grading.
  - "swebench": SWE-bench Verified v5 (remote_env_info.{image,base_commit,
    test_patch,FAIL_TO_PASS,PASS_TO_PASS,version,eval_script,log_parser,
    eval_type}); graded with swebench's
    official make_test_spec + get_eval_report so each repo uses its own
    test_cmd and log parser.

The only thing that varies by protocol is the dataset schema and how a
diff is scored. Everything sandbox-side (prepare_workspace / git_diff /
apply_diff / pre_commands) is shared and lives here once.
``get_metadata(sample, protocol)`` produces the ``md`` dict; the
protocol-specific grading payload is carried under ``md["grading"]``
and is opaque to generate.py (which only reads instance_id / image / workdir).

Harness-agnostic on purpose -- nothing here is Claude-specific. ``SWE_PROMPT`` is
the task instruction (semantics, not CLI syntax). The only place a task meets a
harness is the prompt, which the orchestrator passes into ``harness.run()``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

from slime.agent import sandbox as agent_sandbox
from slime.agent.adapters.common import flatten_content
from slime.agent.sandbox import Sandbox, create_sandbox, exec_and_wait
from slime.utils.types import Sample

from swebench.harness.grading import get_eval_report  # type: ignore
from swebench.harness.utils import make_test_spec  # type: ignore

logger = logging.getLogger(__name__)

PROTOCOL_SCALESWE = "scaleswe"
PROTOCOL_SWEBENCH = "swebench"
_SWEBENCH_REQUIRED_FIELDS = (
    "instance_id",
    "repo",
    "version",
    "base_commit",
    "problem_statement",
    "test_patch",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "environment_setup_commit",
    "image",
    "eval_script",
    "log_parser",
    "eval_type",
)

# Paths inside the sandbox (avoid clashes with image-shipped paths).
_PATCH = "/workspace/__cagent_patch__.diff"
_PRE = "/workspace/__cagent_pre__.sh"
_F2P = "/workspace/__cagent_f2p__.py"
_SWEPRO_DIR = "/workspace/swepro_eval"

SWE_PROMPT = os.environ.get(
    "SWE_CC_PROMPT",
    "Read PROBLEM_STATEMENT.md in the current directory and resolve the issue. "
    "Edit source files only (do NOT touch tests). After editing, run the relevant "
    "tests to verify your fix passes. Do NOT modify PROBLEM_STATEMENT.md and do "
    "NOT commit. When finished, print a one-line summary and exit.",
)


class EvalResult(NamedTuple):
    """Grading outcome; ``applied_cleanly`` only tracks model patch application."""

    reward: float
    applied_cleanly: bool


def get_metadata(sample: Sample, protocol: str = PROTOCOL_SCALESWE) -> dict[str, Any]:
    if protocol == PROTOCOL_SWEBENCH:
        return _metadata_swebench(sample)
    return _metadata_scaleswe(sample)


def _metadata_scaleswe(sample: Sample) -> dict[str, Any]:
    """scaleswe shape: flat ``metadata.*`` (+ a few ``remote_env_info`` fallbacks).

    ``f2p_script`` (a self-contained pytest file ending in
    ``sys.exit(pytest.main(...))``) is carried verbatim; the grader materializes
    and runs it via ``write_file`` so no shell-quoting workaround is needed here.
    """
    m = sample.metadata or {}
    rem = m.get("remote_env_info") or {}
    label = sample.label if (isinstance(sample.label, str) and len(sample.label) < 256) else None
    swepro = m.get("swepro")
    eval_cmd = m.get("eval_cmd")
    f2p_script = rem.get("f2p_script")
    looks_swebench = bool(rem.get("test_patch")) and not (swepro or eval_cmd or f2p_script)
    return {
        "protocol": PROTOCOL_SCALESWE,
        "instance_id": m.get("instance_id") or rem.get("instance_id") or label or "unknown",
        "image": m.get("image") or rem.get("image_url"),
        "workdir": m.get("workdir") or rem.get("workdir"),
        "problem_statement": m.get("problem_statement") or _coerce_prompt(sample.prompt),
        "looks_swebench": looks_swebench,
        "grading": {
            "swepro": swepro,
            "eval_cmd": eval_cmd,
            "f2p_script": f2p_script,
            "pre_commands": m.get("pre_commands") or rem.get("pre_commands"),
        },
    }


def _metadata_swebench(sample: Sample) -> dict[str, Any]:
    """SWE-bench Verified v5 shape: carry all fields through unchanged.

    The v5 dataset is the contract with the official ``make_test_spec`` API.
    Validation is intentionally deferred to ``evaluability_check`` so rollout
    code can report the exact missing fields and instance ID.
    """
    m = sample.metadata or {}
    rem = m.get("remote_env_info") or {}
    instance = dict(rem)
    instance["instance_id"] = rem.get("instance_id") or sample.label or "unknown"
    instance["problem_statement"] = rem.get("problem_statement") or _coerce_prompt(sample.prompt)
    instance.setdefault("hints_text", "")
    instance.setdefault("workdir", "/testbed")
    return {
        "protocol": PROTOCOL_SWEBENCH,
        "instance_id": instance["instance_id"],
        "image": instance.get("image"),
        "workdir": rem.get("workdir") or "/testbed",
        "problem_statement": instance["problem_statement"],
        "grading": {"sweb_instance": instance},
    }


def _coerce_prompt(prompt) -> str:
    """Extract the user-message text from a prompt (str or chat-message list)."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        for m in prompt:
            if isinstance(m, dict) and m.get("role") == "user":
                return flatten_content(m.get("content"))
    return ""


def evaluability_check(md: dict) -> str | None:
    if md.get("protocol") == PROTOCOL_SWEBENCH:
        return _evaluability_check_swebench(md)
    return "protocol_row_mismatch:looks_swebench" if md.get("looks_swebench") else None


def _evaluability_check_swebench(md: dict) -> str | None:
    inst = md.get("grading", {}).get("sweb_instance") or {}
    missing_fields = [
        field
        for field in _SWEBENCH_REQUIRED_FIELDS
        if field not in inst or inst[field] is None or (isinstance(inst[field], str) and not inst[field].strip())
    ]
    if missing_fields:
        return f"missing_swebench_fields:{','.join(missing_fields)}"
    if not inst.get("repo"):
        return "missing_repo"
    if not inst.get("base_commit"):
        return "missing_base_commit"
    if not (inst.get("test_patch") or "").strip():
        return "missing_test_patch"
    try:
        _ = _build_test_spec(inst).eval_script  # surfaces per-repo construction errors here, not later
    except Exception as e:  # KeyError on unknown repo/version, etc.
        return f"make_test_spec_failed:{type(e).__name__}"
    return None


# ---------------------------------------------------------------------------
# Workspace prep (agent sandbox, before harness.run)
# ---------------------------------------------------------------------------
async def prepare_workspace(sb: Sandbox, workdir: str, md: dict) -> None:
    """Prep the agent sandbox, then drop PROBLEM_STATEMENT.md.

    E2B provisions its ``agent`` user idempotently; ARCA uses the image-provided
    ``admin`` user directly.
    """
    await agent_sandbox.prepare_work_user(sb, workdir)
    if md.get("protocol") == PROTOCOL_SCALESWE:
        grading = md.get("grading") or {}
        swepro = grading.get("swepro")
        if swepro:
            await apply_before_repo_set_cmd(sb, workdir, swepro)
        pre_commands = grading.get("pre_commands")
        if pre_commands:
            await apply_pre_commands(sb, workdir, pre_commands)
    await sb.write_file(
        f"{workdir}/PROBLEM_STATEMENT.md",
        md.get("problem_statement") or "",
        user=sb.work_user,
    )


async def apply_before_repo_set_cmd(sb: Sandbox, workdir: str, swepro: dict) -> None:
    """Run swepro['before_repo_set_cmd'] in the sandbox if present (no-op if not)."""
    before = swepro.get("before_repo_set_cmd")
    if not before:
        return
    payload = f"set -e\ncd {workdir}\n{before}\n"
    setup_cmd = "mkdir -p /workspace/swepro_setup"
    if sb.privileged_user != sb.work_user:
        setup_cmd += f" && chown {sb.work_user}:{sb.work_user} /workspace/swepro_setup"
    await sb.exec(setup_cmd, user=sb.privileged_user, check=True)
    await sb.write_file("/workspace/swepro_setup/before.sh", payload, user=sb.work_user)
    await sb.exec("bash /workspace/swepro_setup/before.sh", user=sb.work_user, check=False, timeout=600)


async def apply_pre_commands(sb: Sandbox, workdir: str, pre: list[str] | str) -> None:
    # Public: also called for the work sandbox to keep its baseline aligned with
    # eval (sweb-style pre_commands typically `git checkout <base_sha> -f`, so
    # skipping in the work sandbox makes the model's diff context mismatch the
    # eval base -> 100% apply failure).
    if isinstance(pre, str):
        body = pre.replace("\\n", "\n")
    else:
        body = "\n".join(c for c in (pre or []) if c)
    await sb.write_file(_PRE, "set -e\n" + body, user=sb.work_user)
    await sb.exec(
        f"chmod 755 {_PRE} && cd {workdir} && bash {_PRE}",
        user=sb.work_user,
        check=False,
        timeout=600,
    )


# ---------------------------------------------------------------------------
# Diff capture (agent sandbox, after harness.run)
# ---------------------------------------------------------------------------
async def git_diff(sb: Sandbox, workdir: str) -> tuple[str, int, str]:
    cmd = f"cd {workdir} && git add -N . && git diff -- . ':(exclude)PROBLEM_STATEMENT.md' ':(exclude).harness/'"
    exit_code, out, err = await sb.exec(cmd, user=sb.work_user, timeout=120)
    return out, exit_code, err


# ---------------------------------------------------------------------------
# Eval dispatch (fresh sandbox, apply diff, run dataset tests)
# ---------------------------------------------------------------------------
async def run_evaluation(md: dict, *, diff_text: str, timeout_sec: int) -> EvalResult:
    """Uniform entry point: dispatch to the protocol's grader.

    No-test-cheating guarantee (both grading protocols): the eval sandbox is built from
    the same image but starts CLEAN, so only the model-produced diff affects
    reward."""
    if md.get("protocol") == PROTOCOL_SWEBENCH:
        return await _grade_swebench(md, diff_text, timeout_sec)
    return await _grade_scaleswe(md, diff_text, timeout_sec)


# ---------------------------------------------------------------------------
# scaleswe grader
# ---------------------------------------------------------------------------
async def _grade_scaleswe(md: dict, diff_text: str, timeout_sec: int) -> EvalResult:
    """Three mutually-exclusive grading paths, in priority order: swepro test
    harness, a shell ``eval_cmd``, or a self-contained ``f2p_script`` pytest
    file. All resolve to "exit 0 == solved", reward is 1.0 iff solved."""
    image = md["image"]
    workdir = md["workdir"]
    grading = md.get("grading") or {}
    swepro = grading.get("swepro")
    eval_cmd = grading.get("eval_cmd")
    f2p_script = grading.get("f2p_script")
    pre_commands = grading.get("pre_commands")

    if not (swepro or eval_cmd or f2p_script):
        logger.warning("[swe.scaleswe] no swepro/eval_cmd/f2p_script; reward=0")
        return EvalResult(0.0, True)

    async with create_sandbox(
        image,
        metadata={"instance_id": str(md["instance_id"]), "role": "eval", "attempt": "1"},
    ) as ev:
        await agent_sandbox.prepare_work_user(ev, workdir)
        if swepro:
            await _setup_swepro_assets(ev, swepro)
            await apply_before_repo_set_cmd(ev, workdir, swepro)
        if pre_commands:
            await apply_pre_commands(ev, workdir, pre_commands)

        applied = await _apply_diff(ev, workdir, diff_text)
        if not applied:
            return EvalResult(0.0, False)

        if swepro:
            r = await _run_swepro(ev, workdir, swepro, timeout_sec)
        elif eval_cmd:
            r = await _run_eval_cmd(ev, workdir, eval_cmd, timeout_sec)
        else:
            r = await _run_f2p_script(ev, workdir, f2p_script, timeout_sec)
        return EvalResult(r, True)


async def _setup_swepro_assets(ev: Sandbox, swepro: dict) -> None:
    await ev.exec(
        f"mkdir -p {_SWEPRO_DIR} && chmod 777 {_SWEPRO_DIR}",
        user=ev.privileged_user,
        check=True,
    )
    for k, dst in [("run_script_path", "run_script.sh"), ("parser_script_path", "parser.py")]:
        host_p = swepro.get(k)
        if host_p:
            await ev.write_file(f"{_SWEPRO_DIR}/{dst}", Path(host_p), user=ev.privileged_user)
    ownership = ""
    if ev.privileged_user != ev.work_user:
        ownership = f" && chown -R {ev.work_user}:{ev.work_user} {_SWEPRO_DIR}"
    await ev.exec(
        f"chmod 755 {_SWEPRO_DIR}/*{ownership}",
        user=ev.privileged_user,
        check=True,
    )


async def _apply_diff(ev: Sandbox, workdir: str, diff_text: str) -> bool:
    if not diff_text.strip():
        return True
    await ev.write_file(_PATCH, diff_text, user=ev.work_user)
    # First-success-wins ladder collapsed into one exec (one sandbox round-trip).
    ladder = " || ".join(
        f"({cmd})"
        for cmd in (
            f"git apply --3way --whitespace=nowarn {_PATCH}",
            f"git apply --whitespace=nowarn {_PATCH}",
            f"patch -p1 --no-backup-if-mismatch < {_PATCH}",
        )
    )
    ec, _, _ = await ev.exec(f"cd {workdir} && ({ladder})", user=ev.work_user, check=False, timeout=120)
    return ec == 0


async def _run_swepro(ev: Sandbox, workdir: str, swepro: dict, timeout: int) -> float:
    test_arg = ",".join(swepro.get("selected_test_files") or [])
    stdout_f = f"{_SWEPRO_DIR}/stdout.log"
    stderr_f = f"{_SWEPRO_DIR}/stderr.log"
    result_f = f"{_SWEPRO_DIR}/result.json"
    await ev.exec(
        f"cd {workdir} && bash {_SWEPRO_DIR}/run_script.sh {json.dumps(test_arg)} > {stdout_f} 2> {stderr_f} || true",
        user=ev.work_user,
        check=False,
        timeout=timeout,
    )
    await ev.exec(
        f"python3 {_SWEPRO_DIR}/parser.py {stdout_f} {stderr_f} {result_f}",
        user=ev.work_user,
        check=False,
        timeout=120,
    )
    raw = await ev.read_file(result_f, user=ev.work_user)
    parsed = json.loads(raw) if raw else {"tests": []}
    passed = {t["name"] for t in parsed.get("tests", []) if t.get("status") == "PASSED"}
    required = set(swepro.get("fail_to_pass") or []) | set(swepro.get("pass_to_pass") or [])
    solved = bool(required) and required.issubset(passed)
    return 1.0 if solved else 0.0


async def _run_eval_cmd(ev: Sandbox, workdir: str, cmd: str, timeout: int) -> float:
    ec, _, _ = await ev.exec(f"cd {workdir} && {cmd}", user=ev.work_user, check=False, timeout=timeout)
    return 1.0 if ec == 0 else 0.0


async def _run_f2p_script(ev: Sandbox, workdir: str, script: str, timeout: int) -> float:
    # sweb f2p_script is a self-contained pytest file ending in
    # `sys.exit(pytest.main([...]))`; write it verbatim (no shell quoting) and
    # let python's exit code carry the pass/fail signal.
    await ev.write_file(_F2P, script, user=ev.work_user)
    ec, _, _ = await ev.exec(f"cd {workdir} && python {_F2P}", user=ev.work_user, check=False, timeout=timeout)
    return 1.0 if ec == 0 else 0.0


# Mirror of swebench.harness.run_evaluation.GIT_APPLY_CMDS: try each in order,
# first success wins. The `patch --fuzz` tier rescues diffs `git apply` rejects.
_GIT_APPLY_CMDS = (
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
)


async def _apply_model_patch(ev: Sandbox, workdir: str) -> bool:
    """Apply /tmp/patch.diff via the GIT_APPLY_CMDS ladder; True if applied
    (or empty). Empty patch is a no-op success -- eval then scores it 0 on its
    own (no source change -> tests still fail)."""
    ladder = " || ".join(f"{cmd} /tmp/patch.diff" for cmd in _GIT_APPLY_CMDS)
    cmd = (
        f"cd {workdir} && git config --global --add safe.directory {workdir} "
        f"&& if [ -s /tmp/patch.diff ]; then {ladder}; fi"
    )
    ec, _, _ = await ev.exec(cmd, user=ev.privileged_user, check=False, timeout=120)
    return ec == 0


def _build_test_spec(inst: dict):
    """make_test_spec(inst). Shared by evaluability_check and the grader; may
    raise (KeyError on unknown repo/version)."""
    return make_test_spec(inst)  # type: ignore[misc]


def _eval_report_from_log(ts, instance_id: str, diff_text: str, log: str) -> dict:
    """Run swebench's get_eval_report against the captured test log. It reads
    from a file path, so write the log to a tempfile, parse, and clean up."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False)
    try:
        tmp.write(log)
        tmp.flush()
        tmp.close()
        prediction = {
            "instance_id": instance_id,
            "model_patch": diff_text or "",
            "model_name_or_path": "swe",
        }
        return get_eval_report(  # type: ignore[misc]
            test_spec=ts,
            prediction=prediction,
            test_log_path=tmp.name,
            include_tests_status=True,
        )
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _ratio(d: dict) -> tuple[int, int]:
    """(passed, total) from a {success: [...], failure: [...]} bucket."""
    passed, failed = d.get("success", []), d.get("failure", [])
    return len(passed), len(passed) + len(failed)


def _log_swebench_result(
    instance_id: str,
    exit_code,
    *,
    model_patch_apply_ok: bool,
    info: dict,
    log: str,
) -> None:
    """Log patch application and eval-log parsing as independent states."""
    # SWE-bench sets this legacy field only after get_logs_eval finds a valid
    # test-output section; the actual model patch was applied above.
    eval_log_parse_ok = bool(info.get("patch_successfully_applied"))
    ts_status = info.get("tests_status") or {}
    f2p_pass, f2p_total = _ratio(ts_status.get("FAIL_TO_PASS", {}))
    p2p_pass, p2p_total = _ratio(ts_status.get("PASS_TO_PASS", {}))
    nothing_parsed = not (f2p_total or p2p_total)
    tail = log[-800:] if nothing_parsed else ""
    logger.info(
        "[swe.swebench] %s: reward=%d exit_code=%s model_patch_apply_ok=%s "
        "eval_log_parse_ok=%s F2P=(%d/%d) P2P=(%d/%d)%s",
        instance_id,
        1 if info.get("resolved") else 0,
        exit_code,
        model_patch_apply_ok,
        eval_log_parse_ok,
        f2p_pass,
        f2p_total,
        p2p_pass,
        p2p_total,
        f" tail={tail!r}" if tail else "",
    )


async def _grade_swebench(md: dict, diff_text: str, timeout_sec: int) -> EvalResult:
    """reward=1.0 iff sweb's get_eval_report declares the instance ``resolved``."""
    instance_id = md["instance_id"]
    inst = md["grading"]["sweb_instance"]

    try:
        ts = _build_test_spec(inst)
        eval_sh = ts.eval_script  # may raise on unknown repo/version
    except Exception as e:
        logger.warning("[swe.swebench] %s: make_test_spec/eval_script failed: %s; reward=0", instance_id, e)
        return EvalResult(0.0, True)

    image = md["image"]
    if not image:
        logger.warning("[swe.swebench] %s: missing image; reward=0", instance_id)
        return EvalResult(0.0, True)

    async with create_sandbox(
        image,
        metadata={"instance_id": str(instance_id), "role": "eval", "attempt": "1"},
    ) as ev:
        writes = [ev.write_file("/tmp/eval.sh", eval_sh, user=ev.privileged_user)]
        if diff_text:
            writes.append(ev.write_file("/tmp/patch.diff", diff_text, user=ev.privileged_user))
        await asyncio.gather(*writes)
        # Apply the model patch first (eval_script assumes it is already applied);
        # if no apply strategy works, the instance is unsolvable -- skip the eval.
        if not await _apply_model_patch(ev, md["workdir"]):
            logger.warning(
                "[swe.swebench] %s: reward=0 model_patch_apply_ok=False "
                "eval_log_parse_ok=None; model patch failed to apply",
                instance_id,
            )
            return EvalResult(0.0, False)
        exit_code, log = await exec_and_wait(
            ev,
            cmd="bash /tmp/eval.sh",
            user=ev.privileged_user,
            time_budget_sec=timeout_sec,
            tag="eval",
            want_output=True,
        )

    try:
        report = _eval_report_from_log(ts, instance_id, diff_text, log)
    except Exception as e:
        logger.warning(
            "[swe.swebench] %s: reward=0 model_patch_apply_ok=True "
            "eval_log_parse_ok=False; get_eval_report failed: %s (tail=%r)",
            instance_id,
            e,
            log[-600:],
        )
        return EvalResult(0.0, True)

    info = report.get(instance_id, {})
    _log_swebench_result(
        instance_id,
        exit_code,
        model_patch_apply_ok=True,
        info=info,
        log=log,
    )
    return EvalResult(1.0 if info.get("resolved") else 0.0, True)
