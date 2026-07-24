"""SWE task layer: dataset parsing, workspace prep, diff capture, fresh-sandbox eval.

One module, two grading protocols selected per-call (never an import-time side
effect):

  - "scaleswe" (default): scaleswe data shape (image_url + pre_commands +
    swepro/eval_cmd/f2p_script); custom "exit 0 == solved" grading.
  - "swebench": SWE-bench Verified (remote_env_info.{image,base_commit,
    test_patch,FAIL_TO_PASS,PASS_TO_PASS,version}); graded with swebench's
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

try:
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS_PY  # type: ignore
    from swebench.harness.grading import get_eval_report  # type: ignore
    from swebench.harness.test_spec.test_spec import make_test_spec  # type: ignore

    _SWEBENCH_IMPORT_ERROR: Exception | None = None
except Exception as _exc:  # pragma: no cover - import-time diagnostic
    MAP_REPO_VERSION_TO_SPECS_PY = None  # type: ignore
    get_eval_report = None  # type: ignore
    make_test_spec = None  # type: ignore
    _SWEBENCH_IMPORT_ERROR = _exc

logger = logging.getLogger(__name__)


def _patch_pip_install_e(script: str) -> str:
    """Patch ``pip install -e .[...]`` lines in an eval script for LocalSandbox.

    In Docker containers ``pip install -e .[test]`` works out of the box
    because the full build toolchain is present. On the host (LocalSandbox)
    pip's default build-isolation fetches the *latest* setuptools into a
    temporary virtual env, which may lack APIs that the repo's ``setup.py``
    depends on (e.g. ``setuptools.dep_util`` removed after v57).  Using
    ``--no-build-isolation`` forces pip to reuse the packages already
    present in the conda env (where ``Cython``, ``setuptools<58``, and
    ``extension-helpers`` are pre-installed by ``prepare_swebench_envs.py``).
    """
    import re

    return re.sub(
        r"^([^\n]*pip[^\n]*install[^\n]*-e[^\n]*)$",
        r"\1 --no-build-isolation",
        script,
        flags=re.MULTILINE,
    )


def _patch_eval_sh_for_local(script: str, env_dir: str | None = None) -> str:
    """Patch swebench's eval script for LocalSandbox execution.

    Four problems need fixing:

    1. ``conda activate testbed`` references a ``testbed`` env that does
       not exist on the host — only versioned envs like
       ``sweb_astropy_astropy_5.1`` do.  We replace ``conda activate
       testbed`` with ``conda activate <real-env-name>`` when *env_dir*
       is provided, and also inject an explicit ``export PATH=.../bin:$PATH``
       so that the conda env's python/pip are found first (``conda
       activate`` in non-interactive bash only sets ``CONDA_PREFIX``, not
       ``PATH``).

    2. ``conda activate`` in non-interactive bash does NOT modify PATH —
       it only sets ``CONDA_PREFIX`` and ``CONDA_DEFAULT_ENV``.  So
       ``python`` and ``pip`` still resolve to the host's system Python
       (3.12) instead of the conda env's Python (e.g. 3.9).  We inject
       an explicit ``export PATH=.../bin:$PATH`` right after the
       ``conda activate`` line.

    3. ``pip install -e .[test]`` with build isolation pulls the *latest*
       setuptools (which breaks repos expecting old APIs like
       ``setuptools.dep_util``).  We add ``--no-build-isolation`` so pip
       reuses the conda env's pre-installed build deps (Cython,
       setuptools<58, etc.).

    4. ``pip install -e .[test] --no-build-isolation`` depends on build
       deps like ``Cython`` and ``extension-helpers`` that swebench's spec
       does not list (Docker images already have them).  The conda env
       may be missing them if ``_create_conda_env_sync`` or
       ``prepare_swebench_envs.py`` failed to install them silently.
       We inject ``pip install Cython extension-helpers`` before each
       ``pip install -e .`` line so the deps are present even when the
       env was only partially provisioned.  Combined with the ``set -e``
       added in step 2, a pip failure here will exit the script rather
       than silently continuing with a broken environment.
    """
    import re

    # (1) Replace "conda activate testbed" with the real env name + PATH
    # override so that the conda env's python/pip are found first.
    if env_dir:
        env_name = Path(env_dir).name  # e.g. "sweb_astropy_astropy_5.1"
        script = re.sub(
            r"conda activate testbed\n",
            f"conda activate {env_name}\nexport PATH={env_dir}/bin:$PATH\n",
            script,
        )
    else:
        script = re.sub(
            r"(conda activate testbed\n)",
            r"\1export PATH=" + str(_CONDA_ROOT / "envs" / "testbed" / "bin") + r":$PATH\n",
            script,
        )

    # (2) Replace "set -uxo pipefail" with "set -euxo pipefail" so that
    # pip install failures cause the eval script to exit immediately instead
    # of silently continuing.  swebench omits -e by design (Docker cleanup
    # needs to run), but LocalSandbox discards the whole workspace on exit
    # so early termination is safe and preferable.
    script = script.replace("set -uxo pipefail", "set -euxo pipefail", 1)

    # (3) Inject build-dep installation before `pip install -e .` lines
    # so that --no-build-isolation can find Cython / extension-helpers.
    # Must run BEFORE _patch_pip_install_e so the regex matches cleanly.
    # Add --index-url / --trusted-host from SWEBENCH_PIP_INDEX_URL so that
    # pip inside the sandbox uses the same mirror as _create_conda_env_sync.
    idx = _pip_index_args()
    script = re.sub(
        r"^([^\n]*pip[^\n]*install[^\n]*-e[^\n]*)$",
        rf"pip install {idx}Cython extension-helpers\n\1",
        script,
        flags=re.MULTILINE,
    )

    # (4) Add --no-build-isolation to pip install -e lines.
    script = _patch_pip_install_e(script)

    return script


_CONDA_ROOT = Path(os.environ.get("SWEB_CONDA_ROOT", "/opt/miniconda3"))
_CONDA_BIN = _CONDA_ROOT / "bin" / "conda"
_MINICONDA_URL = "https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh"
_CONDA_MIRROR_MAIN = "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/"
_CONDA_MIRROR_FORGE = "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/"
# Per-(repo, version) lock to serialize conda env creation.  Without this,
# two concurrent _ensure_conda_env calls for the same key can both pass the
# os.path.isdir check and race on ``conda create``, potentially corrupting
# the environment directory.
_env_locks: dict[tuple[str, str], asyncio.Lock] = {}

# Module-level lock to serialize miniconda downloads so only one
# _ensure_conda_env call downloads the installer at a time.
_miniconda_lock = asyncio.Lock()

# Track which envs have been successfully created so we can skip the lock
# entirely on subsequent calls.
_env_created: set[tuple[str, str]] = set()

# Track whether miniconda has been successfully provisioned (or was already
# present) so we can skip the download on subsequent calls.
_miniconda_ready = False


async def _ensure_miniconda() -> Path | None:
    """Ensure miniconda3 is installed and return the conda binary path.

    If ``_CONDA_BIN`` already exists, return it immediately.  Otherwise
    download the miniconda3 installer from the Tsinghua mirror and run a
    silent install (``-b -p <conda_root>``).  Returns the conda binary
    path on success, ``None`` on failure.  Concurrent calls are serialized
    via ``_miniconda_lock`` so only one download runs at a time.
    """
    global _miniconda_ready
    if _miniconda_ready or _CONDA_BIN.is_file():
        _miniconda_ready = True
        return _CONDA_BIN

    async with _miniconda_lock:
        # Re-check after acquiring the lock — another coroutine may have
        # completed the installation while we waited.
        if _miniconda_ready or _CONDA_BIN.is_file():
            _miniconda_ready = True
            return _CONDA_BIN

        logger.info(
            "[swe.swebench] conda not found at %s — downloading miniconda3 from %s ...",
            _CONDA_BIN,
            _MINICONDA_URL,
        )

        import subprocess

        installer = f"/tmp/miniconda3-{os.getpid()}.sh"
        try:
            # Download installer.
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    f"curl -fsSL -o {installer} {_MINICONDA_URL}",
                    shell=True,
                    timeout=300,
                    check=True,
                    capture_output=True,
                    text=True,
                ),
            )
            # Silent install.
            await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    f"bash {installer} -b -p {_CONDA_ROOT}",
                    shell=True,
                    timeout=600,
                    check=True,
                    capture_output=True,
                    text=True,
                ),
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            stderr = getattr(exc, "stderr", "") or ""
            stdout = getattr(exc, "stdout", "") or ""
            logger.error(
                "[swe.swebench] miniconda3 installation FAILED: %s%s%s",
                type(exc).__name__,
                f"\n  stdout: {stdout[:500]}" if stdout else "",
                f"\n  stderr: {stderr[:500]}" if stderr else "",
            )
            return None
        finally:
            # Clean up installer regardless of success/failure.
            if os.path.isfile(installer):
                os.remove(installer)

        if _CONDA_BIN.is_file():
            logger.info("[swe.swebench] miniconda3 installed successfully at %s", _CONDA_ROOT)
            _miniconda_ready = True
            return _CONDA_BIN

        logger.error(
            "[swe.swebench] miniconda3 installation completed but %s not found",
            _CONDA_BIN,
        )
        return None


def _get_env_lock(repo: str, version: str) -> asyncio.Lock:
    key = (repo, version)
    if key not in _env_locks:
        _env_locks[key] = asyncio.Lock()
    return _env_locks[key]


async def _ensure_conda_env(repo: str, version: str) -> str | None:
    """Lazily create a conda env for ``(repo, version)`` if missing.

    Returns the env directory path on success (or if it already existed),
    ``None`` on failure.  Concurrent calls for the same ``(repo, version)``
    are serialized via a per-key asyncio.Lock so only one ``conda create``
    runs at a time.  If miniconda3 is not installed, it is auto-provisioned
    first (see ``_ensure_miniconda``).
    """
    slug = repo.replace("/", "_") if repo else ""
    name = f"sweb_{slug}_{version}"
    env_dir = str(_CONDA_ROOT / "envs" / name)
    key = (repo, version)

    # Fast path: env already exists (or was created in a prior call).
    if os.path.isdir(env_dir) or key in _env_created:
        return env_dir

    if MAP_REPO_VERSION_TO_SPECS_PY is None:
        logger.warning("[swe.swebench] swebench not importable — cannot auto-create conda env %s", name)
        return None

    spec = MAP_REPO_VERSION_TO_SPECS_PY.get(repo, {}).get(version, {})
    if not spec:
        logger.warning("[swe.swebench] no spec for %s@%s — cannot auto-create conda env", repo, version)
        return None

    # Ensure miniconda3 is available before creating the env.
    conda_bin = await _ensure_miniconda()
    if conda_bin is None:
        logger.error(
            "[swe.swebench] cannot create conda env %s — miniconda3 not available "
            "and auto-provision failed (see logs above). Skipping this instance.",
            name,
        )
        return None

    # Serialize creation per (repo, version).
    async with _get_env_lock(repo, version):
        # Re-check after acquiring the lock — another coroutine may have
        # created the env while we waited.
        if os.path.isdir(env_dir) or key in _env_created:
            return env_dir

        py_version = spec.get("python", "3.9")
        pip_packages = spec.get("pip_packages", [])
        packages_str = spec.get("packages", "")
        logger.info(
            "[swe.swebench] auto-creating conda env %s (python=%s, %d pip pkgs) ...",
            name,
            py_version,
            len(pip_packages),
        )

        loop = asyncio.get_running_loop()
        created = await loop.run_in_executor(
            None,
            _create_conda_env_sync,
            name,
            py_version,
            packages_str,
            pip_packages,
        )

        if created and os.path.isdir(env_dir):
            logger.info("[swe.swebench] conda env %s created successfully", name)
            _env_created.add(key)
            return env_dir

        logger.error("[swe.swebench] conda env %s auto-creation failed — skipping this instance", name)
        return None


def _pip_index_args() -> str:
    """Return ``--index-url ... --trusted-host ...`` if SWEBENCH_PIP_INDEX_URL is set."""
    pip_index = os.environ.get("SWEBENCH_PIP_INDEX_URL", "")
    pip_trusted = os.environ.get("SWEBENCH_PIP_TRUSTED_HOST", "")
    if pip_index:
        args = f"--index-url {pip_index}"
        if pip_trusted:
            args += f" --trusted-host {pip_trusted}"
        return args + " "
    return ""


def _create_conda_env_sync(name: str, py_version: str, packages_str: str, pip_packages: list[str]) -> bool:
    """Blocking conda env creation (runs in executor)."""
    import subprocess

    env_dir = _CONDA_ROOT / "envs" / name
    if env_dir.exists():
        return True

    conda = str(_CONDA_BIN)
    if not _CONDA_BIN.is_file():
        logger.warning("[swe.swebench] %s not found — cannot create conda env", conda)
        return False

    # Create conda env with the target Python version.
    try:
        subprocess.run(
            f"{conda} create -n {name} python={py_version} -y " f"--override-channels -c {_CONDA_MIRROR_MAIN}",
            shell=True,
            timeout=900,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        # Retry with conda-forge mirror.
        logger.warning("[swe.swebench] env %s: main channel failed, trying conda-forge mirror ...", name)
        try:
            subprocess.run(
                f"{conda} create -n {name} python={py_version} -y " f"--override-channels -c {_CONDA_MIRROR_FORGE}",
                shell=True,
                timeout=900,
                check=True,
                capture_output=True,
                text=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            logger.error("[swe.swebench] env %s: FAILED to create (python=%s)", name, py_version)
            return False

    pip_prefix = f"{_CONDA_ROOT}/envs/{name}/bin/pip"
    idx = _pip_index_args()

    # Install conda packages if specified.
    if packages_str:
        try:
            subprocess.run(
                f"{pip_prefix} install {idx}{packages_str}",
                shell=True,
                timeout=600,
                check=True,
                capture_output=True,
                text=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            logger.warning("[swe.swebench] env %s: conda packages install had issues", name)

    # Install pip packages.
    if pip_packages:
        req_file = f"/tmp/{name}_requirements.txt"
        with open(req_file, "w") as f:
            f.write("\n".join(pip_packages) + "\n")
        try:
            subprocess.run(
                f"{pip_prefix} install {idx}-r {req_file}",
                shell=True,
                timeout=1200,
                check=True,
                capture_output=True,
                text=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            logger.warning("[swe.swebench] env %s: some pip packages failed to install", name)

    # Install build dependencies so that ``pip install -e .[test]
    # --no-build-isolation`` in the eval script can succeed.  Cython and
    # extension-helpers are required by astropy (and many other SWE repos) to
    # compile C extensions, but swebench's spec does not list them because
    # Docker images already have them.  On the host (LocalSandbox) they must
    # be pre-installed when --no-build-isolation is used.
    build_ok = False
    for attempt in range(2):
        try:
            subprocess.run(
                f"{pip_prefix} install {idx}Cython extension-helpers",
                shell=True,
                timeout=300,
                check=True,
                capture_output=True,
                text=True,
            )
            build_ok = True
            break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            if attempt == 0:
                logger.warning(
                    "[swe.swebench] env %s: build deps install failed (attempt 1/2), retrying ...",
                    name,
                )
            else:
                stderr = getattr(exc, "stderr", "") or ""
                logger.error(
                    "[swe.swebench] env %s: build deps install FAILED after 2 attempts: %s%s",
                    name,
                    type(exc).__name__,
                    f"\n  stderr: {stderr[:500]}" if stderr else "",
                )

    # Verify build deps are actually importable (not just "pip said ok").
    if build_ok:
        py_prefix = f"{_CONDA_ROOT}/envs/{name}/bin/python"
        try:
            subprocess.run(
                f'{py_prefix} -c "import Cython; import extension_helpers"',
                shell=True,
                timeout=30,
                check=True,
                capture_output=True,
                text=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            logger.error(
                "[swe.swebench] env %s: Cython/extension-helpers installed but not importable — "
                "pip install -e . --no-build-isolation in eval.sh will likely fail",
                name,
            )

    return env_dir.exists()


PROTOCOL_SCALESWE = "scaleswe"
PROTOCOL_SWEBENCH = "swebench"

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
    """Grading outcome. Tuple-compatible: ``reward, applied = run_evaluation(...)``."""

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
        "image": m.get("image") or rem.get("image_url") or rem.get("image"),
        "workdir": m.get("workdir") or rem.get("workdir"),
        "problem_statement": m.get("problem_statement") or _coerce_prompt(sample.prompt),
        "repo": rem.get("repo") or "",
        "base_commit": rem.get("base_commit") or "",
        "version": rem.get("version") or "",
        "looks_swebench": looks_swebench,
        "grading": {
            "swepro": swepro,
            "eval_cmd": eval_cmd,
            "f2p_script": f2p_script,
            "pre_commands": m.get("pre_commands") or rem.get("pre_commands"),
        },
    }


def _metadata_swebench(sample: Sample) -> dict[str, Any]:
    """SWE-bench Verified shape: carry the full instance dict through so
    make_test_spec gets every field it needs (version, hints_text, ...)."""
    m = sample.metadata or {}
    rem = m.get("remote_env_info") or {}
    instance = {
        "instance_id": rem.get("instance_id") or "unknown",
        "repo": rem.get("repo") or "",
        "version": rem.get("version"),
        "base_commit": rem.get("base_commit") or "",
        "problem_statement": rem.get("problem_statement") or _coerce_prompt(sample.prompt),
        "hints_text": rem.get("hints_text") or "",
        "test_patch": rem.get("test_patch") or "",
        "FAIL_TO_PASS": rem.get("FAIL_TO_PASS"),
        "PASS_TO_PASS": rem.get("PASS_TO_PASS"),
        "environment_setup_commit": rem.get("environment_setup_commit"),
    }
    return {
        "protocol": PROTOCOL_SWEBENCH,
        "instance_id": instance["instance_id"],
        "image": rem.get("image"),
        "workdir": rem.get("workdir") or "/testbed",
        "problem_statement": instance["problem_statement"],
        "repo": instance["repo"],
        "base_commit": instance["base_commit"],
        "version": instance.get("version") or "",
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
    if _SWEBENCH_IMPORT_ERROR is not None:
        return f"swebench_import_failed:{type(_SWEBENCH_IMPORT_ERROR).__name__}"
    inst = md.get("grading", {}).get("sweb_instance") or {}
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

    Assumes the agent user already owns ``workdir`` (the harness's ``run()`` calls
    ``ensure_agent_user``; the orchestrator runs this before ``run()`` and the
    agent user is created lazily there). To stay independent of call order we
    create the agent user here too -- it is idempotent.
    """
    await agent_sandbox.ensure_agent_user(sb, workdir)
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
        user="agent",
    )


async def apply_before_repo_set_cmd(sb: Sandbox, workdir: str, swepro: dict) -> None:
    """Run swepro['before_repo_set_cmd'] in the sandbox if present (no-op if not)."""
    before = swepro.get("before_repo_set_cmd")
    if not before:
        return
    payload = f"set -e\ncd {workdir}\n{before}\n"
    await sb.exec(
        "mkdir -p /workspace/swepro_setup && chown agent:agent /workspace/swepro_setup", user="root", check=True
    )
    await sb.write_file("/workspace/swepro_setup/before.sh", payload, user="agent")
    await sb.exec("bash /workspace/swepro_setup/before.sh", user="agent", check=False, timeout=600)


async def apply_pre_commands(sb: Sandbox, workdir: str, pre: list[str] | str) -> None:
    # Public: also called for the work sandbox to keep its baseline aligned with
    # eval (sweb-style pre_commands typically `git checkout <base_sha> -f`, so
    # skipping in the work sandbox makes the model's diff context mismatch the
    # eval base -> 100% apply failure).
    if isinstance(pre, str):
        body = pre.replace("\\n", "\n")
    else:
        body = "\n".join(c for c in (pre or []) if c)
    await sb.write_file(_PRE, "set -e\n" + body, user="agent")
    await sb.exec(f"chmod 755 {_PRE} && cd {workdir} && bash {_PRE}", user="agent", check=False, timeout=600)


# ---------------------------------------------------------------------------
# Diff capture (agent sandbox, after harness.run)
# ---------------------------------------------------------------------------
async def git_diff(sb: Sandbox, workdir: str) -> str:
    cmd = f"cd {workdir} && git add -N . && git diff -- . ':(exclude)PROBLEM_STATEMENT.md' ':(exclude).harness/'"
    _, out, _ = await sb.exec(cmd, user="agent", timeout=120)
    return out


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

    async with create_sandbox(image, instance_id=md["instance_id"]) as ev:
        await agent_sandbox.ensure_agent_user(ev, workdir)
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
    await ev.exec(f"mkdir -p {_SWEPRO_DIR} && chmod 777 {_SWEPRO_DIR}", user="root", check=True)
    for k, dst in [("run_script_path", "run_script.sh"), ("parser_script_path", "parser.py")]:
        host_p = swepro.get(k)
        if host_p:
            await ev.write_file(f"{_SWEPRO_DIR}/{dst}", Path(host_p), user="root")
    await ev.exec(f"chmod 755 {_SWEPRO_DIR}/* && chown -R agent:agent {_SWEPRO_DIR}", user="root", check=True)


async def _apply_diff(ev: Sandbox, workdir: str, diff_text: str) -> bool:
    if not diff_text.strip():
        return True
    await ev.write_file(_PATCH, diff_text, user="agent")
    # First-success-wins ladder collapsed into one exec (one sandbox round-trip).
    ladder = " || ".join(
        f"({cmd})"
        for cmd in (
            f"git apply --3way --whitespace=nowarn {_PATCH}",
            f"git apply --whitespace=nowarn {_PATCH}",
            f"patch -p1 --no-backup-if-mismatch < {_PATCH}",
        )
    )
    ec, _, _ = await ev.exec(f"cd {workdir} && ({ladder})", user="agent", check=False, timeout=120)
    return ec == 0


async def _run_swepro(ev: Sandbox, workdir: str, swepro: dict, timeout: int) -> float:
    test_arg = ",".join(swepro.get("selected_test_files") or [])
    stdout_f = f"{_SWEPRO_DIR}/stdout.log"
    stderr_f = f"{_SWEPRO_DIR}/stderr.log"
    result_f = f"{_SWEPRO_DIR}/result.json"
    await ev.exec(
        f"cd {workdir} && bash {_SWEPRO_DIR}/run_script.sh {json.dumps(test_arg)} > {stdout_f} 2> {stderr_f} || true",
        user="agent",
        check=False,
        timeout=timeout,
    )
    await ev.exec(
        f"python3 {_SWEPRO_DIR}/parser.py {stdout_f} {stderr_f} {result_f}",
        user="agent",
        check=False,
        timeout=120,
    )
    raw = await ev.read_file(result_f, user="agent")
    parsed = json.loads(raw) if raw else {"tests": []}
    passed = {t["name"] for t in parsed.get("tests", []) if t.get("status") == "PASSED"}
    required = set(swepro.get("fail_to_pass") or []) | set(swepro.get("pass_to_pass") or [])
    solved = bool(required) and required.issubset(passed)
    return 1.0 if solved else 0.0


async def _run_eval_cmd(ev: Sandbox, workdir: str, cmd: str, timeout: int) -> float:
    ec, _, _ = await ev.exec(f"cd {workdir} && {cmd}", user="agent", check=False, timeout=timeout)
    return 1.0 if ec == 0 else 0.0


async def _run_f2p_script(ev: Sandbox, workdir: str, script: str, timeout: int) -> float:
    # sweb f2p_script is a self-contained pytest file ending in
    # `sys.exit(pytest.main([...]))`; write it verbatim (no shell quoting) and
    # let python's exit code carry the pass/fail signal.
    await ev.write_file(_F2P, script, user="agent")
    ec, _, _ = await ev.exec(f"cd {workdir} && python {_F2P}", user="agent", check=False, timeout=timeout)
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
    ec, _, _ = await ev.exec(cmd, user="root", check=False, timeout=120)
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


def _log_swebench_result(instance_id: str, exit_code, info: dict, log: str) -> None:
    """Emit the per-instance grading outcome with test-bucket ratios; always
    surface the eval log tail so failures can be diagnosed."""
    if info.get("resolved"):
        logger.info("[swe.swebench] %s: reward=1 exit_code=%s", instance_id, exit_code)
        return
    ts_status = info.get("tests_status") or {}
    f2p_pass, f2p_total = _ratio(ts_status.get("FAIL_TO_PASS", {}))
    p2p_pass, p2p_total = _ratio(ts_status.get("PASS_TO_PASS", {}))
    # Always show the eval log tail — P2P=0 with parsed tests can still
    # hide ImportPathMismatchError / conda failures.
    start_marker = ">>>>> Start Test Output"
    end_marker = ">>>>> End Test Output"
    if start_marker in log and end_marker in log:
        test_log = log[log.find(start_marker) + len(start_marker) : log.find(end_marker)]
    else:
        test_log = log
    tail = test_log[-1200:]
    logger.info(
        "[swe.swebench] %s: reward=0 exit_code=%s patch_applied=%s F2P=(%d/%d) P2P=(%d/%d) tail=%s",
        instance_id,
        exit_code,
        bool(info.get("patch_successfully_applied")),
        f2p_pass,
        f2p_total,
        p2p_pass,
        p2p_total,
        repr(tail),
    )


async def _grade_swebench(md: dict, diff_text: str, timeout_sec: int) -> EvalResult:
    """reward=1.0 iff sweb's get_eval_report declares the instance ``resolved``."""
    from slime.agent.local_sandbox import LocalSandbox

    instance_id = md["instance_id"]
    inst = md["grading"]["sweb_instance"]

    if _SWEBENCH_IMPORT_ERROR is not None:
        logger.error(
            "[swe.swebench] %s: swebench import failed: %r; reward=0",
            instance_id,
            _SWEBENCH_IMPORT_ERROR,
        )
        return EvalResult(0.0, True)

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

    # Pass repo / base_commit / version so LocalSandbox creates a git
    # worktree at /testbed (otherwise testbed is an empty directory and
    # eval.sh's `git checkout <base_commit>` exits with code 128).
    repo = md.get("repo") or inst.get("repo")
    base_commit = md.get("base_commit") or inst.get("base_commit")
    version = md.get("version") or inst.get("version")
    async with create_sandbox(
        image, instance_id=instance_id, repo=repo, base_commit=base_commit, version=version
    ) as ev:
        _is_local = isinstance(ev, LocalSandbox)

        # Resolve the correct conda env for LocalSandbox.
        # We cannot use ``add_bind_mount`` to map the env into the sandbox
        # because mount --bind fails silently on overlay/containerd
        # filesystems.  Instead, we patch the eval script to reference the
        # real conda env name directly.
        env_dir: str | None = None
        if _is_local:
            slug = repo.replace("/", "_") if repo else ""
            env_dir = str(_CONDA_ROOT / "envs" / f"sweb_{slug}_{version}")
            if not os.path.isdir(env_dir):
                env_dir = await _ensure_conda_env(repo, version)
            if env_dir and os.path.isdir(env_dir):
                logger.info("[swe.swebench] %s: using conda env %s", instance_id, env_dir)
            else:
                logger.warning(
                    "[swe.swebench] %s: conda env for %s@%s not found and "
                    "auto-creation failed — eval will use system Python "
                    "(tests likely fail due to version mismatch)",
                    instance_id,
                    repo,
                    version,
                )
                env_dir = None

        # For LocalSandbox: patch the eval script so that the conda env's
        # Python is used instead of the host's system Python.
        if _is_local and _CONDA_BIN.is_file():
            eval_sh = _patch_eval_sh_for_local(eval_sh, env_dir=env_dir)

        await asyncio.gather(
            ev.write_file("/tmp/patch.diff", diff_text or "", user="root"),
            ev.write_file("/tmp/eval.sh", eval_sh, user="root"),
        )
        if not await _apply_model_patch(ev, md["workdir"]):
            logger.warning("[swe.swebench] %s: model patch failed to apply; reward=0", instance_id)
            return EvalResult(0.0, False)
        exit_code, log = await exec_and_wait(
            ev,
            cmd="bash /tmp/eval.sh",
            user="root",
            time_budget_sec=timeout_sec,
            tag="eval",
            want_output=True,
            env={"HOME": "/root"},
        )

    try:
        report = _eval_report_from_log(ts, instance_id, diff_text, log)
    except Exception as e:
        logger.warning(
            "[swe.swebench] %s: get_eval_report failed: %s; reward=0 (tail=%r)",
            instance_id,
            e,
            log[-600:],
        )
        return EvalResult(0.0, True)

    info = report.get(instance_id, {})
    _log_swebench_result(instance_id, exit_code, info, log)

    if info.get("resolved"):
        return EvalResult(1.0, True)

    # Continuous reward shaping so that partial progress (passing some
    # FAIL_TO_PASS tests, preserving PASS_TO_PASS tests) produces non-zero
    # gradients even when the instance is not fully resolved.  Without this,
    # the 0/1 binary reward collapses to zero for every sample in a batch
    # (Qwen3-4B currently resolves 0/16), yielding advantage=0 and
    # grad_norm=0 — the model cannot learn at all.
    ts_status = info.get("tests_status") or {}
    f2p_pass, f2p_total = _ratio(ts_status.get("FAIL_TO_PASS", {}))
    p2p_pass, p2p_total = _ratio(ts_status.get("PASS_TO_PASS", {}))
    patch_applied = bool(info.get("patch_successfully_applied"))

    if not patch_applied or (f2p_total == 0 and p2p_total == 0):
        # Eval never ran or no tests parsed — no signal.
        return EvalResult(0.0, patch_applied)

    # α weights F2P (fixing the target bug), β weights P2P (not breaking
    # existing tests).  P2P alone shouldn't give high reward — a no-op
    # patch preserves all P2P but fixes nothing.
    alpha: float = float(os.environ.get("SWE_REWARD_F2P_WEIGHT", "0.7"))
    beta: float = float(os.environ.get("SWE_REWARD_P2P_WEIGHT", "0.3"))
    f2p_ratio = f2p_pass / f2p_total if f2p_total else 0.0
    p2p_ratio = p2p_pass / p2p_total if p2p_total else 1.0
    reward = alpha * f2p_ratio + beta * p2p_ratio
    return EvalResult(reward, patch_applied)
