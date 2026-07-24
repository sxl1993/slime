# CONTEXT.md — slime 领域术语表

## SWE-bench 评估

| 术语 | 定义 |
|------|------|
| **sweb conda env** | 为某个 SWE instance 创建的 conda 环境，命名 `sweb_{repo_slug}_{version}`（如 `sweb_astropy_astropy_5.1`），包含该 instance 所需的 Python 版本和依赖 |
| **conda root** | miniconda3 的安装根目录，默认 `/opt/miniconda3`，可通过 `SWEB_CONDA_ROOT` 环境变量覆盖 |
| **conda auto-provision** | 当 `conda` 不存在时，在 `_ensure_conda_env` 内自动下载并静默安装 miniconda3 的行为 |
| **conda mirror** | conda 安装包和 package channel 的镜像源。当前环境官方源不可达，必须走清华源 |
| **instance-level fail** | 单个 SWE instance 的 conda env 创建失败不中断 batch，打 error 日志后跳过该 instance |

## Sandbox

| 术语 | 定义 |
|------|------|
| **LocalSandbox** | 基于 git worktree 的本地沙箱，进程共享宿主机 |
| **E2BSandbox** | 基于 E2B 的远程沙箱，进程在云端隔离环境运行 |