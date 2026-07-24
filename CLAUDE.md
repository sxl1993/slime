# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 语言风格

中文提问用中文回答。中英混排规则：

- **句子骨架用中文**：连接词、动词、介词、冠词一律中文，不要写"这是 the baseline"这种夹英文虚词的句子。
- **仅以下三种情况用英文**：
  1. 代码标识符、文件路径、命令行参数（如 `reverse_kl`、`--opd-kl-coef`、`placement_group.py`）
  2. RL / ML 社区固定术语且无公认中译（如 `rollout`、`advantage`、`baseline`、`on-policy`、`forward`、`reward`、`loss`、`mode-seeking`）
  3. 引用英文原文或专有名词（如 `Megatron`、`SGLang`、`GRPO`、`P1`）
- **不要中英重复**：不写"baseline 基线"、"forward 前向传播"这种同义并列。
- **整句英文只在以下场景使用**：引用代码注释、报错信息、命令输出、英文专有术语缩写。
- 不会翻译的英文术语宁可直接用英文，不要硬造中文译名。

## 项目概述

slime 是一个 LLM 后训练（post-training）框架，专注于 RL scaling。它把 **Megatron 训练**与 **SGLang rollout** 通过同一条 "training / rollout / Data Buffer" 路径串起来，避免训练器、rollout 服务、agent 框架各自为政。是 GLM-4.5 ~ GLM-5.2 等开源模型背后的 RL 训练基础设施。

设计取向（理解这些取向后再动手改代码）：
- **原生 pass-through**：Megatron 参数直接透传（如 `--tensor-model-parallel-size`）；SGLang 参数加 `--sglang-` 前缀透传（如 `--sglang-mem-fraction-static`）。slime 不在两个引擎之上再包一层抽象。
- **轻量、有主见**：只支持 SGLang 一个 rollout 后端，以深度利用其路由、缓存、PD 分离、weight-sync 等能力，而不是抹平多个推理引擎的最大公约数。
- **正确性优先**：RL bug 通常静默失效，因此 dataflow 保持显式，并提供独立的 "rollout-only" 与 "train-only" 调试路径。
- **协作范围**：仅接受 bug fix 与可被 CI/常规训练验证的通用 RL 优化；不接受大规模重构、抽象/标准提案、对 Megatron 的大改动、与 RL 框架独立的算法复现流水线。详见 `CONTRIBUTING.md`。

## 常用命令

### 安装与依赖
```bash
# 安装 slime 本身（依赖需先准备好 Megatron-LM 与 SGLang）
pip install -e .
# 或使用 conda 一键环境（含 SGLang/Megatron 的固定 commit）
bash build_conda.sh
```
关键环境变量：`PYTHONPATH` 必须包含 Megatron-LM 路径（脚本里通常设为 `/root/Megatron-LM/`），`CUDA_DEVICE_MAX_CONNECTIONS=1`、`NCCL_NVLS_ENABLE` 视 NVLink 是否存在而定（`scripts/run-*.sh` 里有示例）。

### 代码风格
```bash
pre-commit install
pre-commit run --all-files --show-diff-on-failure --color=always
```
- `black`（line-length 119）、`isort`（black profile，`known_first_party = slime, slime_plugins`）、`ruff`（E/F/B/UP）、`autoflake` 均通过 pre-commit 跑。
- 单测用 `pytest`，`pyproject.toml` 配置了 `--pyargs --durations=0 --strict-markers`，`testpaths=["./tests"]`。可用 marker：`unit / integration / system / acceptance / docs / skipduringci / pleasefixme`。

### 运行单个测试
仓库里的端到端测试不是普通 pytest，而是脚本式（见 `tests/test_*.py`）。每个测试文件实现 `prepare()` + `execute()`，通过 `slime/utils/external_utils/command_utils.py` 中的 `U.exec_command / U.hf_download_dataset / U.convert_checkpoint / U.execute_train` 调起真实训练。
```bash
# 直接运行某个 GPU 端到端测试
python tests/test_qwen2.5_0.5B_short.py
# 标记筛选
pytest -m unit tests/
```
GPU 测试需要先拿到 GPU 锁，CI 用 `tests/ci/gpu_lock_exec.py` 通过 `fcntl` 对 `/dev/nvidia*` 加锁避免冲突。

### 启动训练
入口在 `train.py`（同步）与 `train_async.py`（异步，下一个 rollout 提前发起）。典型启动方式是 `scripts/run-<model>.sh`：
1. `source scripts/models/<type>.sh` 注入 `MODEL_ARGS`、转换好的 torch_dist ckpt 路径等；
2. `ray start --head ...` 起 ray 集群；
3. `ray job submit ... -- python3 train.py <args>` 提交训练作业。

`scripts/run-*.sh` 可作为参考模板；新模型先看 `scripts/models/` 是否已有对应 `<type>.sh`。

### 权重转换
HF ↔ Megatron torch_dist 转换在 `tools/convert_hf_to_torch_dist.py`、`tools/convert_torch_dist_to_hf*.py`，以及 fp8/int4 量化转换工具（`tools/convert_hf_to_fp8.py`、`convert_hf_to_int4*.py`）。`U.convert_checkpoint()` 是测试里调用的封装。

## 架构总览

入口 `train.py:train(args)` 的主循环就是理解整个框架的脊梁：

```
parse_args → create_placement_groups → create_rollout_manager → create_training_models
            → update_weights (actor → rollout)
            → for rollout_id in range(num_rollout):
                 rollout_manager.generate(rollout_id)      # SGLang 生成 + reward
                 actor_model.async_train(...)              # Megatron 训练
                 save / offload / update_weights / eval
```

### 三大模块（与目录对应）
- **training (Megatron)** — `slime/ray/train_actor.py`、`slime/backends/megatron_utils/`、`slime_plugins/megatron_bridge/`、`slime_plugins/mbridge/`、`slime_plugins/models/`。slime 把 Megatron 参数读进来直接用，新模型通过 `slime_plugins/mbridge/<model>.py`（mbridge fork 的模型实现）或 `slime_plugins/megatron_bridge/` 接入。**不要在 slime 内维护 Megatron fork**。
- **rollout (SGLang + router)** — `slime/rollout/sglang_rollout.py`（默认 `--rollout-function-path`）、`slime/rollout/sglang_streaming_rollout.py`、`slime/rollout/fully_async_rollout.py`、`slime/rollout/sft_rollout.py`、`slime/rollout/on_policy_distillation.py`，启动 SGLang server 的胶水在 `slime/backends/sglang_utils/`。rollout 与 router 是同一进程组内由 Ray actor 管理。
- **data buffer** — `slime_plugins/rollout_buffer/buffer.py` + `slime_plugins/rollout_buffer/generator/`，作为 prompt 初始化 / 自定义数据 / rollout 生成方法的交汇点。Agentic workflow 也是塞进同一个 rollout 接口而不是另起框架。

### Ray 编排
`slime/ray/placement_group.py` 负责 placement group 分配、`create_rollout_manager` / `create_training_models`。`slime/ray/actor_group.py`、`train_actor.py`、`rollout.py`、`rollout_validation.py` 是各类 Ray actor。colocate 模式（`--colocate`）下训练与 rollout 共享 GPU，必须配 `--offload-train` / `--offload-rollout` 在两者之间切换。

### 参数系统（重要）
参数在 `slime/utils/arguments.py`，由三部分组合：
1. **Megatron 参数**：直接读，无前缀；
2. **SGLang 参数**：`--sglang-` 前缀，由 `slime/backends/sglang_utils/arguments.py` 解析；SGLang 还可通过 YAML（`--sglang-config`）做 topology 控制（prefill/decode/EPD 异构组、多模型、按组覆盖）；
3. **slime 自身参数**：`--actor-num-nodes`、`--rollout-num-gpus`、`--colocate`、`--offload*`、`--num-rollout`、`--rollout-batch-size`、`--update-weights-interval` 等。

### 自定义扩展点（编写 / 审查代码时优先考虑这些接口而不是改主干）
- `--rollout-function-path`：替换整个 rollout 函数。签名 `def generate_rollout(args, rollout_id, data_source, evaluation=False) -> RolloutFnTrainOutput | RolloutFnEvalOutput`，输出 sample 至少要填 `tokens / response_length / reward / status`。多 agent、多轮、工具调用走这里。
- `--custom-generate-function-path`：只替换 `generate(args, sample, sampling_params)` 这一层，保留默认 rollout 的其余脚手架。多轮 / function calling 常用此口。
- `--custom-rm-path`：自定义 reward 函数 `def custom_rm(args, sample) -> float`，覆盖 `--rm-type` 默认逻辑。
- `--custom-reward-post-process-path`：奖励后处理（默认是 GRPO 的归一化）。
- `--custom-convert-samples-to-train-data-path`：替换 `_convert_samples_to_train_data`，把 samples 转成 Megatron 训练输入。
- `--custom-rollout-log-function-path` / `--custom-eval-rollout-log-function-path`：自定义 rollout 日志，返回 `True` 跳过默认日志。
- `slime/rollout/filter_hub/`：dynamic filter / sample-group 选择 / buffer 训练前过滤。
- `slime/agent/`：agent harness、sandbox、trajectory、parsing 工具，配合 `examples/` 下的 coding_agent_rl / multi_agent / search-r1 / fully_async 使用。

新增这些 hook 时，对应的 skill（`add-rollout-function`、`add-reward-function`、`add-dynamic-filter`、`add-eval-dataset-config`、`add-tests-and-ci`）描述了签名与 CI 接入要求；评审时参考 `slime-code-review-preferences`。

### Agent 系统架构（`slime/agent/` + `examples/`）

slime 的 agent 能力不是独立框架，而是塞进 `--custom-generate-function-path` / `--rollout-function-path` 的同一套 rollout 接口。理解 agent 数据流的关键是分清三层：

**1. 适配器层（`slime/agent/adapters/`）— 反向代理模式**

`BaseAdapter` 把自己伪装成 Anthropic / OpenAI API，在本地 `aiohttp` daemon 线程上跑。agent CLI（Claude Code、Codex）拨回这个服务器，每轮对话被转发到 SGLang `/generate`，同时 token ids 和 logprobs 被精确记录到 `TrajectoryManager`。客户端看到的是标准 wire format，训练侧拿到的是对齐的 token 序列——不需要改 agent CLI 代码。

- `AnthropicAdapter`：服务 `/v1/messages`，处理 thinking blocks / tool_use / SSE 流。特殊处理：非首位的 system messages 被折叠为 `<system-reminder>` 塞进 user blocks（chat template 拒绝中间 system）。
- `OpenAIAdapter`：服务 `/v1/chat/completions`，处理 tool_calls（JSON-string arguments 被强转为 dict 以匹配 TrajectoryManager 的 dict equality）。

**2. 轨迹层（`slime/agent/trajectory.py`）— 多分支 token 树**

`TrajectoryManager` 为每个 `sid` 维护一棵消息树（`MessageNode`）。当后续轮次的 prompt prefix 与已有节点不匹配时（子 agent 调度、上下文压缩、re-tokenization drift），树自动 fork 新分支。`get_trajectory()` 把每条 root→leaf 路径线性化为独立 `Sample`，各自带正确的 `loss_mask`。

token drift 处理分三档：`CLEAN`（精确前缀扩展，直接拼接）、`REALIGN`（response 内部短漂移，覆盖并标 `loss_mask=0`）、`FORK`（漂移过大，关闭当前 builder 开新分支）。

**3. Harness 层（`slime/agent/harness/`）— 可插拔的编码 agent 安装/启动**

`BaseHarness` 定义 `install_cli → write_config → launch_and_wait` 生命周期，不含任务逻辑。已有实现：
- `ClaudeCodeHarness`：npm 安装 Claude Code CLI，preset `bypassPermissions`，设 `ANTHROPIC_BASE_URL` 指向适配器。
- `CodexHarness`：npm 安装 Codex CLI，写 `config.toml` 内联 `base_url`（Codex 只对默认 provider 读 env），设 `OPENAI_BASE_URL` 指向适配器。

新增 harness 只需一个文件继承 `BaseHarness`。

**4. 沙箱层（`slime/agent/sandbox.py`）— E2B 隔离**

`Sandbox` 协议定义 `exec / write_file / read_file` 三个异步方法。`E2BSandbox` 是唯一实现，包装 `e2b.AsyncSandbox`，带瞬态 RPC 重试和 idle-GC keepalive。测评在独立干净沙箱中运行。

**核心约定**

| 约定 | 说明 |
|------|------|
| **String-In / Token-Out** | agent 环境传字符串/JSON，训练必须基于 token。适配器对每轮消息做 chat template tokenize → `input_ids` 发 SGLang，记录精确 token ids 和 logprobs。`response` 只是可读副产物，不re-tokenize |
| **loss_mask 纪律** | 模型生成 token `loss_mask=1`（可训练），observation/tool 输出 token `loss_mask=0`（不可训练），由 `Sample.append_response_tokens(trainable=...)` 统一保证 |
| **rollout_id 归并** | 一次 agent 执行 fan-out 为多个 Sample 时（TrajectoryManager 多分支、multi-agent 多角色），所有兄弟 Sample 必须共享同一 `rollout_id`，否则 per-rollout loss reducer 会把一次执行算 N 次 |
| **per-turn 上下文预算** | 适配器和自定义 generate 每轮 clamp `max_new_tokens` 到剩余预算，不超 `--rollout-max-context-len` |
| **custom generate 可返回 list[Sample]** | fan-out 路径在 `generate_and_rm_group` 和 `_get_rollout_data` 中被 flatten |
| **环境变量前缀** | `SLIME_AGENT_*` 给可复用 agent 库，`SWE_*` 给 SWE 示例任务，`ADAPTER_*` 给部署侧地址 |

**Session 生命周期**：`open_session(sid)` → 多轮 `_run_turn` → `finish_session(sid, base_sample, reward)` 下发训练数据，或 `drop_session(sid)` 丢弃。sid 从 `Authorization: Bearer` / `X-Api-Key` / `metadata.session_id` / `user` 字段派生。

**examples 中的 Agent 模式**

| 示例 | 入口 hook | 模式 | 说明 |
|------|-----------|------|------|
| `coding_agent_rl` | `--custom-generate-function-path` | 适配器 + harness + sandbox | 最完整：E2B sandbox 启动 agent CLI，CLI 拨回适配器，TrajectoryManager 捕获 token，独立 sandbox 跑 eval |
| `multi_agent` | `--custom-generate-function-path` | 直接 HTTP 驱动 SGLang | Solver/Rewriter/Selector 三角色，不走适配器/harness |
| `search-r1` | `--custom-generate-function-path` | 最小多轮 + tool calling | `append_response_tokens(trainable=...)` 手动拼 token/loss_mask/logprobs |
| `fully_async` | `--rollout-function-path` | 后台线程异步消费 buffer | `AsyncRolloutWorker` 解耦并发与 `rollout_batch_size`，ABORTED 样本回推 buffer |
| `geo3k_vlm_multi_turn` | `--custom-generate-function-path` + `--rollout-interaction-env-path` | VLM 多轮交互 | `BaseInteractionEnv` 定义 `reset/step/format_observation`，支持多模态 |
| `retool` | `--custom-generate-function-path` | Code interpreter tool calling | Jinja2 模板 + tool_registry 执行 Python |
| `strands_sglang` | `--custom-generate-function-path` | StrandsSDK + 自定义 SGLangModel | `RolloutTracker` 直接捕获 token ids/logprobs/loss_mask |
| `tau-bench` | `--custom-generate-function-path` | tau-bench 环境交互 | 多轮循环内化在 `trainable_agents` 模块 |

### Weight Sync / PD 分离
- `--update-weights-interval` 控制 actor→rollout 权重同步频率；
- `examples/delta_weight_sync/` + `slime/utils/disk_delta.py`：增量权重同步，用于 train/infer 分离、跨机型 / 跨厂商 rollout；
- `docs/en/advanced/pd-disaggregation.md`：prefill/decode 分离 + 多轮 agent 路由策略；
- `docs/en/advanced/external-rollout-engines.md`：rollout 引擎跑在训练 job 之外（独立环境 / 不同 GPU 厂商）。

### 数值正确性 / 调试 / 可复现性
相关文档：`docs/en/developer_guide/debug.md`、`docs/en/advanced/reproducibility.md`、`docs/en/advanced/fault-tolerance.md`、`docs/en/developer_guide/trace.md`、`docs/en/developer_guide/profiling.md`。`slime/utils/trace_utils.py`、`tools/trace_timeline_viewer.py`、`tools/profile_rollout.py`、`tools/analyze_profile.py` 是配套工具。改训练 / rollout 数据流时优先考虑是否需要新的 trace / 数值校验。

## 代码约定（来自 skill 与 reviewer 偏好）

- **避免无用 wrapper**：slime 评审不喜欢"为了对称再加一层薄封装"。如果一个函数只是转发到另一个，直接调用即可。
- **控制流直接可读**：参数校验、分支选择写在主路径里能看懂的位置，不要藏到 helper 里。新选项必须在 `slime/utils/arguments.py` 里加 `help=` 文案，并写清楚默认值与互斥条件。
- **pass-through 优先**：新 SGLang/Megatron 能力尽量走 pass-through，不要在 slime 里复制参数定义。SGLang 新参数应该已经能用 `--sglang-` 前缀，无需改 slime 代码。
- **正确性优先于"看起来跑通"**：改 dataflow 时配套加 CPU 单测或 GPU e2e 测试（`tests/test_*.py` 模板），并在 PR 里说明可复现路径。无法被 CI 或常规训练验证的改动不会被接受。
- **不要 fork Megatron**：模型实现走 `slime_plugins/mbridge/` 或 `slime_plugins/models/`，不要篡改 Megatron 内部。

## CI

- `.github/workflows/pr-test.yml`（由 `pr-test.yml.j2` + `generate_github_workflows.py` 生成，**改 CI 改 j2 模板**）。
- CPU / 单测在 GitHub-hosted runner 上跑；GPU e2e 在 self-hosted runner 上跑，受 PR label 控制（如 `run-ci-sglang-config`）。push 到 main 只触发便宜的 CPU job。
- 改测试矩阵时同步更新 j2 模板与 `generate_github_workflows.py`，再 `python .github/workflows/generate_github_workflows.py` 重新生成 yml。
- self-hosted runner 配置见 `tests/ci/README.md`。

## Agent skills

### Issue tracker

Issues 在 GitHub Issues 中管理（`THUDM/slime`），使用 `gh` CLI 操作。详见 `docs/agents/issue-tracker.md`。

### Triage labels

五角色标准标签：`needs-triage`、`needs-info`、`ready-for-agent`、`ready-for-human`、`wontfix`。详见 `docs/agents/triage-labels.md`。

### Domain docs

Single-context 布局：根目录 `CONTEXT.md` + `docs/adr/`。详见 `docs/agents/domain.md`。