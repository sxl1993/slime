# Coding-Agent RL

This directory provides an example of running end-to-end **SWE (Software-Engineering) coding-agent RL** with slime: a real coding agent (claude-code CLI) drives `Read/Edit/Grep/Bash/Agent` tools inside a fresh sandbox per sample, the model produces a `git diff`, and the diff is graded against the dataset's test harness in a second clean sandbox (no test-cheating).

Two example files, the shared harness package, and one shared adapter implement the loop:

- `generate.py` — per-sample `generate()` registered via `--custom-generate-function-path`. Boots the sandbox, prepares the SWE workspace, runs the coding harness (claude-code), captures the diff, scores it, and emits one or more `Sample`s back to slime.
- `slime.agent.adapters.AnthropicAdapter` — the shared Anthropic Messages adapter. claude-code talks to it as if it were Anthropic; the adapter tokenizes the current message history each turn, records prompt/output token snapshots, preserves model-generated tokens (`loss_mask=1`) only while later prompts stitch onto them, and masks template/observation tokens (`0`). Each turn is routed into a per-session message tree inside `slime.agent.trajectory.TrajectoryManager`; any divergence in the prompt prefix forks a new branch, so sub-agent dispatches and auto-compaction are handled as separate root-to-leaf chains. `get_trajectory` linearizes each leaf chain into one `Sample`.
- `slime.agent.harness` — harness-agnostic coding-agent lifecycle (install CLI, write config, spawn detached, poll done-marker). `BaseHarness` defines the contract; `CLAUDE_CODE` / `CODEX` are the shipped implementations. Adding a harness is one new file. The shared sandbox contract lives in `slime.agent.sandbox.Sandbox`.
- `swe.py` — harness-agnostic SWE task layer built on `slime.agent.sandbox`: `prepare_workspace` (pre_commands + PROBLEM_STATEMENT.md), `git_diff` (patch capture), and `evaluate` (fresh-sandbox grading). `SWE_PROMPT` is the task instruction handed to whichever harness runs.

`generate.py` owns one `AnthropicAdapter` instance. For each sample it calls
`adapter.open_session(...)` before starting claude-code, serves `adapter.app` as
the Anthropic-compatible endpoint, and drains trainable `TokenSegment`s with
`await adapter.finish_session(...)` when the trajectory ends.

## Environment Setup

The slime training stack itself follows the standard setup. Select one sandbox
backend with `SLIME_AGENT_SANDBOX_BACKEND=e2b|arca` (default: `e2b`):

1. **E2B:** configure `E2B_API_KEY`. The official SDK validates this value
   locally, so compatible internal gateways that ignore auth still need a
   syntactically valid `e2b_` + 40 hex-character placeholder.
2. **ARCA:** install the optional `arca-sandbox==1.1.0` SDK on every Ray node,
   then configure `SLIME_ARCA_APP_NAME`, `SLIME_ARCA_BASE_URL`,
   `SLIME_ARCA_API_KEY`, and `SLIME_AGENT_ARCA_TEMPLATE_ID`. Slime writes these
   values to a mode-`0600` temporary YAML only while constructing one
   process-shared `SandboxFactory`, then deletes the file.
3. **E2B host-side tarballs** uploaded into each sandbox at boot:
   - Node 22 (`node-v22.x-linux-x64.tar.xz`) — exported as `SLIME_AGENT_NODE_TARBALL`.
   - Claude Code CLI npm tarball (`anthropic-ai-claude-code-local-linux-x64.tgz`) — exported as `SLIME_AGENT_CC_TARBALL`.
4. **An E2B image routing key** (`SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY`, legacy `SWE_SANDBOX_IMAGE_METADATA_KEY` still accepted) — the metadata key the E2B gateway uses to route a boot to a specific image. ARCA passes the dataset image directly as the SDK `image` override.
5. **Network reachability:** use `ADAPTER_PUBLIC_URL=https://<gateway-domain>`
   for a TLS gateway, or retain the E2B-compatible
   `http://${ADAPTER_PUBLIC_HOST}:${ADAPTER_PORT}` fallback. Never use
   `127.0.0.1` as a sandbox callback address.

## Dataset Format

Standard slime JSONL with three keys:

```jsonc
{
  "prompt": "<falls back here if metadata.problem_statement is missing>",
  "label": "<instance_id or grader label>",
  "metadata": {
    "image": "your-registry/swe-image:<tag>",  // sandbox image reference
    "workdir": "/workspace/<repo>",            // repo path inside the sandbox
    "problem_statement": "<issue body>",
    // exactly one of the following two graders:
    "swepro": { /* SWE-bench Pro test harness — preferred */ },
    "eval_cmd": "pytest -x tests/..."          // last-resort: exit 0 = solved
    // sweb-style rows: metadata.remote_env_info.f2p_script (Python file
    // ending in `sys.exit(pytest.main(...))`) is auto-wrapped into eval_cmd.
  }
}
```

Wire it up with `--input-key prompt --label-key label --metadata-key metadata`.

## Running the Script

Override the paths at the top of the launcher, then run from a long-lived shell on the Ray head node (do **not** wrap in `nohup` — Ray child processes get cleaned up with it):

```bash
cd slime/

export HF_CHECKPOINT=/path/to/Qwen3.6-35B-A3B
export REF_MODEL_PATH=/path/to/Qwen3.6-35B-A3B_torch_dist
export PROMPT_DATA=/path/to/swe_train.jsonl
export SLIME_AGENT_NODE_TARBALL=/path/to/node-v22.20.0-linux-x64.tar.xz
export SLIME_AGENT_CC_TARBALL=/path/to/anthropic-ai-claude-code-local-linux-x64.tgz

# E2B provider (default):
export SLIME_AGENT_SANDBOX_BACKEND=e2b
export E2B_API_KEY=e2b_xxx                       # real key for e2b.dev; a syntactically
                                                 # valid placeholder if your gateway ignores auth
export SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY=image   # metadata key your gateway routes images by

bash examples/coding_agent_rl/run_qwen36_35b_a3b_swe_8nodes.sh
```

ARCA uses the same launcher and dataset `metadata.image` field:

```bash
uv pip install 'arca-sandbox==1.1.0' \
  --index-url https://artifacts.antgroup-inc.cn/simple/

export SLIME_AGENT_SANDBOX_BACKEND=arca
export SLIME_ARCA_APP_NAME=a3training
export SLIME_ARCA_BASE_URL=http://arca-sandbox.global.alipay.com:8080
export SLIME_ARCA_API_KEY='<secret>'
export SLIME_AGENT_ARCA_TEMPLATE_ID=ARCA-TEMPLATE-xxxxxxxxxxxxxxxx
export ADAPTER_PUBLIC_URL=https://<approved-adapter-gateway-domain>

bash examples/coding_agent_rl/run_qwen36_35b_a3b_swe_8nodes.sh
```

The ARCA instance image must already contain the `admin` user, `/testbed`,
`/home/admin/bin/arca_envd`, and Claude Code. Slime does not create an `agent`
user or install Node/Claude Code in ARCA sandboxes.

The launcher fans Ray out to every worker listed in `$HOSTFILE` (default
`/root/mpi_rack_hostfile`, one worker IP per line, reachable over passwordless
SSH as `root`) — create that file (or point `HOSTFILE` at your own) before
launching. It then dumps every rollout to `runs/${EXP_TAG}_${STAMP}/rollout_dumps/`
and tees stdout into `runs/${EXP_TAG}_${STAMP}/run.log`.

## New Arguments

`generate.py` is wired in through slime's standard custom-generate hook:

```bash
ROLLOUT_ARGS=(
   --custom-generate-function-path examples.coding_agent_rl.generate.generate
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-context-len 96000
   --rollout-max-response-len 32768
   --rollout-stop-token-ids 248046 248044
   --save-debug-rollout-data "${RUN_ROOT}/rollout_dumps/rollout_{rollout_id}.pt"
)
```

The SGLang server must expose Qwen3.6's tool-call and reasoning parsers so claude-code's tool invocations are parsed correctly:

```bash
SGLANG_ARGS=(
   --sglang-tool-call-parser qwen3_coder
   --sglang-reasoning-parser qwen3
   ...
)
```

## SWE-specific Environment Knobs

All set in the launcher; tune per cluster.

Env vars split by layer. `SLIME_AGENT_*` are the reusable agent library's
contract (read inside `slime/agent/`); `SWE_*` are this SWE example's task knobs;
`ADAPTER_*` are host-side deployment/reply-path addresses read only by
`generate.py`. Keep new vars on the prefix that matches the layer that reads them.

| Variable | Default | Meaning |
| --- | --- | --- |
| `SLIME_AGENT_SANDBOX_BACKEND` | `e2b` | Explicit provider: `e2b` or `arca`. |
| `ADAPTER_PUBLIC_URL` | unset | Full sandbox-visible adapter URL, including scheme. Takes precedence over `ADAPTER_PUBLIC_HOST`; use the approved HTTPS gateway for ARCA. |
| `ADAPTER_PUBLIC_HOST` | `${MASTER_ADDR}` | Public IP the sandbox uses to reach the Anthropic adapter. **Must be routable from inside the sandbox.** |
| `ADAPTER_BIND_HOST` / `ADAPTER_PORT` | `0.0.0.0` / `18001` | Bind address of the Anthropic adapter on the host. |
| `E2B_API_KEY` | — | E2B (or compatible) API key. |
| `SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY` | — | **Required.** Which metadata key the E2B gateway routes images by (e.g. `image`); each sample's `metadata.image` is passed under it. (Legacy `SWE_SANDBOX_IMAGE_METADATA_KEY` still accepted.) |
| `SLIME_AGENT_NODE_TARBALL` | — | Host path to Node 22 tarball uploaded into each sandbox. |
| `SLIME_AGENT_CC_TARBALL` | — | Host path to the Claude Code CLI npm tarball. |
| `SLIME_ARCA_APP_NAME` | — | ARCA service-authorized application name. Required by ARCA. |
| `SLIME_ARCA_BASE_URL` | — | ARCA SDK gateway URL. Required by ARCA. |
| `SLIME_ARCA_API_KEY` | — | ARCA API key. Required by ARCA; propagated to Ray but not printed by the launcher. |
| `SLIME_AGENT_ARCA_TEMPLATE_ID` | — | Base template controlling ARCA permission/network/probe policy. Required by ARCA. |
| `SLIME_AGENT_ARCA_TTL_MINUTES` | `40` | ARCA lease TTL in minutes. |
| `SLIME_AGENT_ARCA_CPU` / `SLIME_AGENT_ARCA_MEMORY` / `SLIME_AGENT_ARCA_DISK` | `2` / `4` / `25` | ARCA `ResourceSpecification` fields. |
| `SLIME_AGENT_ARCA_CREATE_TIMEOUT_SEC` | `150` | Timeout for the single ARCA create request. |
| `SLIME_AGENT_ARCA_READY_TIMEOUT_SEC` | `120` | Bounded terminal readiness window after a sandbox ID is known. |
| `SLIME_AGENT_ARCA_READY_POLL_INTERVAL_SEC` | `2` | Delay between terminal readiness probes. |
| `SLIME_AGENT_CC_EXTRA_ARGS` | (see launcher) | Extra flags appended to the `claude` CLI invocation — registers the read-only `investigator` sub-agent, disables `WebFetch`/`WebSearch`, disables slash commands. |
| `SLIME_AGENT_CC_EXTRA_ENVS` | unset | JSON object of extra env vars exported into the `claude` process — escape hatch for env-only knobs (`MAX_THINKING_TOKENS`, `BASH_MAX_TIMEOUT_MS`, ...). Merged last, so it can also override the built-in defaults. |
| `SWE_AGENT_TIME_BUDGET_SEC` | `1800` | Wallclock budget for the in-sandbox agent CLI itself (think/edit/run). |
| `SWE_EVAL_TIMEOUT_SEC` | `600` | Wallclock cap on the evaluator sandbox. |
| `SWE_ROLLOUT_GUARD_SEC` | `agent+eval+180` | Outer safety net wrapping the whole rollout (boot + workspace + agent + diff + eval). Auto-derived if unset. |
| `SWE_BOOT_CONCURRENCY` | `16` | Cap on simultaneous sandbox boots (eases h2/SSL long-tail). |
| `SWE_CC_PROMPT` | unset | Optional override for the user-turn prompt. Setting this to require sub-agent dispatch is the most reliable way to maximize fan-out. |

The multi-node launcher serializes Ray's runtime environment into a mode-`0600`
temporary file and removes it on shell exit, so `SLIME_ARCA_API_KEY` is not
embedded in the `ray job submit` command line or emitted by shell xtrace.

`--rollout-max-response-len` is the per-turn generation cap passed to each
SGLang `/generate` call as `max_new_tokens`. `--rollout-max-context-len` is the
multi-turn prompt+response budget enforced only during generation: each turn
clamps `max_new_tokens` to the remaining context. Trajectory merge/export keeps
the emitted segments and does not drop them for length.
The Anthropic adapter reuses `--sglang-tool-call-parser` and
`--sglang-reasoning-parser` for output parsing, so those flags must match the
served model.

## String-in, Token-out Trajectories

The coding-agent environment is string/message based: claude-code sends
Anthropic Messages requests, receives streamed text/thinking/tool-use blocks,
and later sends back rendered tool observations. Training, however, must stay
token based. A trajectory is only a valid RL target when the optimized tokens
are the same tokens the rollout model actually sampled.

The Anthropic adapter therefore follows a **string in, token out** contract:

- Each incoming message history is rendered with the served model's chat
  template and sent to SGLang as `input_ids`.
- SGLang is called with `return_logprob=True`; the adapter records the exact
  `prompt_ids`, sampled `output_ids`, and per-token rollout logprobs for that
  turn.
- At training export time, samples are assembled from those saved token ids.
  The decoded `response` field is only a readable sidecar; it is not
  re-tokenized to recover the training sequence.

Multi-turn agents still force the adapter to tokenize later message
histories, because tool observations and claude-code's own compacted messages
arrive as strings. `slime.agent.trajectory.TrajectoryManager` routes
those later prompts against the saved token stream:

- New prompt suffixes that are tool/user/environment context are appended with
  `loss_mask=0`.
- Fresh model outputs from SGLang are appended with `loss_mask=1`.
- If a later prompt no longer token-matches an earlier sampled output, the
  unmatched suffix is dropped. If the drift cuts through the middle of a
  previous model output, the retained prefix of that whole output turn is also
  assigned `loss_mask=0`.

That last case is the important correctness guard. A re-tokenization mismatch
can make a string-level conversation look continuous while token-level
provenance is broken. slime keeps the context needed to continue the agent, but
does not backprop through tokens whose sampled origin can no longer be proven.
The unit tests in `tests/test_agent/test_trajectory_manager_branching.py` cover matched
prefixes, skipped turns, split-output drift, changed token counts, and
prompt-base restarts.

## Fan-out Semantics

- `generate()` returns `list[Sample]` — one Sample per root-to-leaf chain in the per-session message tree.
- Per-trajectory reward is split as `reward / K` across chains; `rollout_id` is shared so the per-rollout-mean loss reducer still counts the trajectory once.
- Sub-agent dispatch and auto-compaction increase `K` (each prompt-prefix divergence forks a new branch), so the effective batch after flatten can be much larger than `rollout_batch_size * n_samples_per_prompt`.

## Porting to a New Sandbox Backend

`slime.agent.sandbox.Sandbox` exposes the shared sandbox contract;
`create_sandbox()` selects `E2BSandbox` or `ArcaSandbox`:

```python
await sb.exec(cmd, user=..., check=..., timeout=...)
await sb.write_file(sandbox_path, content_or_host_path, user=...)
await sb.read_file(sandbox_path, user=...)
async with create_sandbox(image, metadata=metadata) as sb: ...
```

ARCA uses only the SDK's async create, terminal, filesystem, and destroy APIs.
If create returns without a sandbox ID, `AmbiguousCreate` stops the outer boot
retry to avoid allocating a duplicate lease. Agent and evaluator sandboxes both
use the same selected backend.
